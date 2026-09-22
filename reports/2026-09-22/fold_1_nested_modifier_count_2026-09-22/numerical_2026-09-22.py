"""Prepare and run six independent native numerical jobs, with no oracle access."""

import concurrent.futures
import importlib.metadata
import json
import logging
import multiprocessing
import os
from pathlib import Path
import time
import traceback

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/stage2-mpl")

from common import HERE, ROOT, BASE, INPUTS, DATE, OLD_DATE, SEED, MODEL, ENDPOINT, now, read, sha, write, verify, manifest, numerical_arguments


def prepare():
    from oci.inference.stage2_elastic_net_selection import statistical_selection_config_from_mapping

    path = HERE / f"input_manifest_{DATE}.json"
    if path.exists():
        return manifest()
    original = read(BASE / f"input_manifest_{OLD_DATE}.json")
    frozen = read(INPUTS.parent / f"inputs_frozen_{OLD_DATE}.json")
    verify(frozen["files"])
    files = dict(frozen["files"])
    for name in ("selected_definitions.json", "role_report.json", "statistical_evidence.json", f"input_manifest_{OLD_DATE}.json"):
        source = BASE / name
        files[str(source)] = sha(source)
    features = read(BASE / "selected_definitions.json")["features"]
    assert sum("confounder" in f["roles"] for f in features) == 189
    policy = statistical_selection_config_from_mapping(original["policy"])
    assert policy.multi_model.modifier_count.enabled
    sources = [Path(__file__), HERE / "common.py", HERE / f"PROTOCOL_{DATE}.md"]
    sources += [ROOT / rel for rel in (
        "oci/inference/stage2_modifier_count.py", "oci/inference/stage2_modifier_ranking.py",
        "oci/inference/stage2_multi_model_selection.py", "oci/inference/stage2_multi_model_config.py",
        "oci/inference/stage2_multi_model_adjudication.py", "oci/inference/stage2_role_adjudication.py",
        "oci/inference/stage2_elastic_net_selection.py", "oci/inference/stage2_statistical_selection.py",
        "oci/inference/plain_handoff_stage2.py", "oci/inference/plain_handoff_stage2_analysis.py",
        "oci/models/causal_forest_head.py", "oci/models/elastic_net_nuisance.py",
    )]
    value = dict(created_at=now(), outer_fold=1, seed=SEED, training_rows=800, heldout_rows=200,
                 candidate_features=352, policy=policy.public_dict(), role_policy=original["role_policy"],
                 evaluation_plan=original["evaluation_plan"], input_files=files,
                 sources={str(p): sha(p) for p in sources},
                 adjudication={"endpoint": ENDPOINT, "model": MODEL, "reasoning_effort": "high", "max_tokens": 100000},
                 scheduling={"numerical_workers": 6, "ranking_workers": 6, "production_workers": 3},
                 confounder_policy="Reuse the frozen 189-confounder broad review; verify reproduced full-training aggregate evidence before ranking.",
                 prior_fold_1_oracle_results_seen=True, exploratory_post_hoc_comparison=True,
                 oracle_values_used_by_selection_or_fitting=False, extraction_refreshed_again=False,
                 selection_rule="minimum_r_loss", one_standard_error_result="diagnostic only; do not fit a competing selected model",
                 versions={p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scipy", "scikit-learn", "econml", "statsmodels")})
    write(path, value)
    return value


def worker(position):
    from oci.inference.stage2_multi_model_selection import select_stage2_features_multi_model

    root = HERE / "numerical_jobs" / f"job_{position}"
    root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=root / "fit.log", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", force=True)
    start = time.monotonic()
    write(root / "status.json", {"phase": "running", "position": position, "pid": os.getpid(), "at": now()})
    try:
        args = numerical_arguments(position)
        _, report, definitions, latent = select_stage2_features_multi_model(**args)
        assert len(definitions) == 352 and not latent
        path = root / "statistical_evidence.json"
        write(path, report)
        result = {"position": position, "phase": "complete", "seconds": time.monotonic() - start,
                  "at": now(), "training_rows": len(args["extracted_fit"]), "folds": len(args["inner_splits"]),
                  "input_fingerprint": report["input_fingerprint"], "report_path": str(path), "report_sha256": sha(path)}
        write(root / "status.json", result)
        return result
    except BaseException as exc:
        write(root / "status.json", {"phase": "failed", "position": position, "at": now(),
                                      "error": str(exc), "traceback": traceback.format_exc()})
        raise


def main():
    plan = prepare()
    ready = HERE / "numerical_complete.json"
    if ready.exists():
        verify(read(ready)["files"])
        print(json.dumps({"phase": "numerical_reused"}), flush=True)
        return
    write(HERE / "status.json", {"phase": "numerical_evidence", "at": now(), "pid": os.getpid(), "jobs": 6})
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=plan["scheduling"]["numerical_workers"],
                                               mp_context=multiprocessing.get_context("spawn")) as pool:
        for future in concurrent.futures.as_completed([pool.submit(worker, i) for i in range(6)]):
            result = future.result()
            results.append(result)
            print(json.dumps(result), flush=True)
            write(HERE / "status.json", {"phase": "numerical_evidence", "at": now(),
                                         "completed_jobs": sorted(r["position"] for r in results), "total_jobs": 6})
    manifest()
    write(ready, {"at": now(), "jobs": sorted(results, key=lambda r: r["position"]),
                  "files": {r["report_path"]: r["report_sha256"] for r in results}})
    write(HERE / "status.json", {"phase": "numerical_complete", "at": now()})


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        write(HERE / "status.json", {"phase": "numerical_failed", "at": now(), "error": str(exc), "traceback": traceback.format_exc()})
        raise
