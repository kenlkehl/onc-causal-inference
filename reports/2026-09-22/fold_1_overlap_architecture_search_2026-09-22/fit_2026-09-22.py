"""Refit the frozen, CV-selected architecture using native estimation; no oracle."""

import concurrent.futures
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

from common import HERE, INPUTS, DATE, now, read, write, sha, verify, manifest, policy_from_manifest


def inputs():
    import pandas as pd

    train, test = pd.read_pickle(INPUTS / "training.pkl"), pd.read_pickle(INPUTS / "heldout.pkl")
    tl, vl = pd.read_parquet(INPUTS / "training_labels.parquet"), pd.read_parquet(INPUTS / "heldout_labels.parquet")
    split = read(INPUTS / "split.json")
    assert train._oci_row_id.tolist() == tl._oci_row_id.tolist() == split["fit_row_ids"]
    assert test._oci_row_id.tolist() == vl._oci_row_id.tolist() == split["heldout_row_ids"]
    return train, test, tl, vl, split


def production(seed):
    import pandas as pd
    from threadpoolctl import threadpool_limits
    from oci.inference.plain_handoff_stage2_analysis import estimate_outer_fold

    dest = HERE / "production" / f"seed_{seed}"
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / "fit_frozen.json"
    if marker.exists():
        result = read(marker)
        verify(result["files"])
        return result
    logging.basicConfig(filename=dest / "fit.log", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", force=True)
    train, test, tl, vl, split = inputs()
    definitions = read(HERE / "selected_definitions.json")["features"]
    plan = read(HERE / f"input_manifest_{DATE}.json")["evaluation_plan"]
    data = pd.concat([tl, vl]).sort_values("_oci_row_id")[["_oci_row_id", "treatment", "outcome"]].reset_index(drop=True)
    assert data._oci_row_id.tolist() == list(range(1000))
    data["analysis_unit"] = data._oci_row_id
    architecture = read(HERE / "count_result.json")["chosen_estimator"]
    start = time.monotonic()
    write(dest / "started.json", {"at": now(), "pid": os.getpid(), "seed": seed})
    with threadpool_limits(limits=1):
        diag = estimate_outer_fold(dataset=data, extracted_fit=train, extracted_heldout=test, definitions=definitions,
                split=split, unit_id_column="analysis_unit", treatment_column="treatment", outcome_column="outcome",
                outcome_type="binary", inner_folds=5, seed=seed, propensity_clip=plan["propensity_clip"],
                min_propensity=plan["min_propensity"], max_propensity=plan["max_propensity"],
                estimation_trees=plan["trees"], output_dir=dest, estimator=architecture, statistical_policy=policy_from_manifest())
    assert diag["estimator"] == architecture
    assert diag["nuisance_model_family"] == "elastic_net"
    assert diag["confounders"] == 189
    assert diag["effect_modifiers"] == sum("effect_modifier" in f["roles"] for f in definitions)
    result = {"seed": seed, "estimator": architecture, "seconds": time.monotonic() - start, "fit_n": diag["effect_fit_rows"],
              "test_n": diag["effect_estimation_rows"], "files": {str(p): sha(p) for p in dest.iterdir() if p.is_file() and p.suffix != ".log"}}
    write(marker, result)
    return result


def main():
    config = manifest()
    selected_freeze = HERE / f"selection_frozen_{DATE}.json"
    verify(read(selected_freeze)["files"])
    freeze_path = HERE / f"predictions_frozen_{DATE}.json"
    if freeze_path.exists():
        verify(read(freeze_path)["files"])
        return
    assert not (HERE / "oracle_access_started.json").exists()
    plan = config["evaluation_plan"]
    architecture = read(HERE / "count_result.json")["chosen_estimator"]
    identity = {"source_sha256": sha(__file__), "selection_freeze_sha256": sha(selected_freeze),
                "plan": plan, "chosen_estimator": architecture, "parallel_production_workers": 3}
    path = HERE / "fit_input.json"
    if path.exists():
        assert read(path) == identity, "Fitting procedure changed"
    else:
        write(path, identity)
    all_files = [path, Path(__file__)]
    write(HERE / "status.json", {"phase": "production_fits", "at": now(), "chosen_estimator": architecture,
                                "seeds": plan["estimation_seeds"]})
    with concurrent.futures.ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context("spawn")) as pool:
        for future in concurrent.futures.as_completed([pool.submit(production, seed) for seed in plan["estimation_seeds"]]):
            result = future.result()
            all_files += [Path(p) for p in result["files"]]
            all_files.append(HERE / "production" / f"seed_{result['seed']}" / "fit_frozen.json")
            print(json.dumps({"phase": "production_fit_complete", "seed": result["seed"],
                              "estimator": architecture, "seconds": result["seconds"]}), flush=True)
    manifest()
    verify(read(selected_freeze)["files"])
    write(freeze_path, {"at": now(), "selection_freeze_sha256": sha(selected_freeze),
          "files": {str(p): sha(p) for p in all_files}, "production_fits": 3,
          "chosen_estimator": architecture, "oracle_values_read_by_fit": False,
          "prior_oracle_exposure_acknowledged": True})
    write(HERE / "status.json", {"phase": "predictions_frozen", "at": now(), "chosen_estimator": architecture})


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        write(HERE / "status.json", {"phase": "fit_failed", "at": now(), "error": str(exc), "traceback": traceback.format_exc()})
        raise
