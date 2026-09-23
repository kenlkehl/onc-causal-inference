"""Prefetch independent fold rankings, then run the native count selector."""

import concurrent.futures
import json
import logging
import os
from pathlib import Path
import threading
import time
import traceback
from unittest.mock import patch

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/stage2-mpl")

from common import HERE, ROOT, BASE, SOURCE, DATE, SEED, MODEL, ENDPOINT, read, write, sha, now, verify, manifest, load_training, policy_from_manifest


def wait_for_report(position):
    root = HERE / "numerical_jobs" / f"job_{position}"
    while True:
        if (root / "status.json").exists():
            status = read(root / "status.json")
            if status["phase"] == "failed":
                raise RuntimeError(f"Numerical job {position} failed: {status['error']}")
            if status["phase"] == "complete":
                verify({status["report_path"]: status["report_sha256"]})
                return read(status["report_path"])
        time.sleep(5)


def runtime(plan):
    from oci.inference.plain_handoff_stage2 import PlainHandoffStage2, plain_stage2_config_from_mapping

    config = read(SOURCE / "inputs/refresh_config.json")
    config.update(endpoint=ENDPOINT, model=MODEL, api_key="EMPTY", extraction_llm=None, vllm=None,
                  min_propensity=0.1, max_propensity=0.9, workers=16, max_tokens=100000, interpretation_reasoning_effort="high", max_prompt_chars=200000,
                  role_adjudication=plan["role_policy"],
                  statistical_selection={k: v for k, v in plan["policy"].items() if k not in {"min_propensity", "max_propensity"}})
    config["selection_consolidation"]["enabled"] = False
    service = PlainHandoffStage2(config=plain_stage2_config_from_mapping(config, default_workers=16), clinical_question="")
    service._check_and_record_model_identity(HERE / "llm_runtime")
    identity = {"created_from": str(SOURCE / "inputs/refresh_config.json"), "model": service.model_identity,
                "runtime_config": service.config.public_dict(), "source_sha256": sha(__file__),
                "input_manifest_sha256": sha(HERE / f"input_manifest_{DATE}.json")}
    # JSON serializes tuple-valued configuration fields as lists. Canonicalize
    # before equality checks; the previous sequential runtime is retained.
    identity = json.loads(json.dumps(identity))
    path = HERE / "llm_runtime/input.json"
    if path.exists():
        assert read(path) == identity, "Ranking runtime changed"
    else:
        write(path, identity)
    return service


def compare_evidence(left, right):
    """Require reproduction of full-population confounder association evidence."""
    import math

    differences = []

    def compare(a, b, path):
        if isinstance(a, dict):
            assert isinstance(b, dict) and set(a) == set(b), path
            for key in a:
                compare(a[key], b[key], path + "/" + key)
        elif isinstance(a, list):
            assert isinstance(b, list) and len(a) == len(b), path
            for i, (x, y) in enumerate(zip(a, b)):
                compare(x, y, path + f"/{i}")
        elif isinstance(a, float):
            assert isinstance(b, (int, float)) and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-10), path
            if a != b:
                differences.append(abs(a - b))
        else:
            assert a == b, path

    def associations(report):
        return {key: [row for row in rows if row["family"] in {"penalized_main", "predictive_forest"}
                      or (row["family"] == "univariable" and row["role"] in {"treatment", "outcome"})]
                for key, rows in report["multi_model_evidence"].items()}
    compare(associations(left), associations(right), "confounder_associations")
    return {"equivalent": True, "nonexact_float_values": len(differences), "max_absolute_difference": max(differences, default=0.0)}


def main():
    from oci.inference import stage2_modifier_count as count
    from oci.inference import stage2_modifier_ranking as ranking
    from oci.inference import stage2_multi_model_selection as numerical
    from oci.inference import plain_handoff_stage2_analysis as analysis
    from oci.inference.stage2_role_adjudication import role_adjudication_config_from_mapping, _fingerprint
    from oci.inference.plain_handoff_stage2 import _request_json
    from oci.inference import stage2_request_audit

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    plan = manifest()
    freeze = HERE / f"selection_frozen_{DATE}.json"
    if freeze.exists():
        verify(read(freeze)["files"])
        return
    assert not (HERE / "oracle_access_started.json").exists()
    data, train, definitions, split = load_training()
    policy = policy_from_manifest()
    role_policy = role_adjudication_config_from_mapping(plan["role_policy"])
    selected = read(BASE / "selected_definitions.json")["features"]
    roles = read(BASE / "role_report.json")
    frame, labels = numerical._validated_inputs(data, train, definitions, split["inner_splits"], "treatment", "outcome")
    base_fp = _fingerprint(numerical.numerical_identity(frame=frame, labels=labels, definitions=definitions,
                          inner_splits=split["inner_splits"], outcome_type="binary", seed=SEED, policy=policy))
    # Derive the native checkpoint location; native input.json is checked below.
    identity = {"schema_version": count.SCHEMA_VERSION, "numerical_input_fingerprint": base_fp,
                "count_policy": policy.multi_model.modifier_count.public_dict(), "role_policy": role_policy.public_dict(),
                "role_decisions": _fingerprint(roles), "selected_definitions": selected, "model": MODEL,
                "estimation_trees": plan["evaluation_plan"]["trees"],
                "sources": {Path(p).name: sha(p) for p in (count.__file__, ranking.__file__, analysis.__file__, ROOT / "oci/inference/stage2_effect_estimators.py")}}
    fp = _fingerprint(identity)
    root = HERE / "modifier_count" / fp[:20]
    service = runtime(plan)
    state = threading.local()

    def request(messages, validate, *, request_kind="interpretation"):
        name = getattr(state, "name", "full_training_ranking")
        number = getattr(state, "number", 0) + 1
        state.number = number
        payload = json.loads(messages[-1]["content"])
        target = HERE / "ranking_jobs" / name
        write(target / "status.json", {"phase": "request", "name": name, "request": number,
                                       "task": payload["task"], "candidates": len(payload["candidates"]), "at": now()})
        with stage2_request_audit.context(_audit_path=str(target / "request_events.jsonl"),
                                         phase=payload["task"], ranking=name, experiment_request=number):
            result = _request_json(messages=messages, config=service.config, completion=service.completion,
                                   validate=validate, request_kind=request_kind)
        write(target / "status.json", {"phase": "response_complete", "name": name, "request": number, "at": now()})
        return result

    def rank_fold(position):
        name = f"fold_{int(split['inner_splits'][position - 1]['inner_fold']):03d}"
        state.name, state.number = name, 0
        write(HERE / "ranking_jobs" / name / "status.json", {"phase": "waiting_for_training_evidence", "at": now()})
        try:
            report = wait_for_report(position)
            result = ranking.rank_modifier_candidates(definitions=definitions, statistical_report=report,
                      request_json=request, output_dir=root / name / "ranking", role_policy=role_policy,
                      maximum=policy.multi_model.modifier_count.max_ranked_modifiers,
                      maximum_chars=policy.multi_model.max_prompt_chars, model_identity=MODEL)
            write(HERE / "ranking_jobs" / name / "status.json", {"phase": "complete", "at": now(),
                  "ranked": len(result["ranking"]), "reviewed": len(result["reviewed_candidate_ids"]),
                  "requests_this_process": state.number, "evidence_fingerprint": report["input_fingerprint"]})
            print(json.dumps({"phase": "fold_ranking_complete", "fold": name, "requests": state.number}), flush=True)
        except BaseException as exc:
            write(HERE / "ranking_jobs" / name / "status.json", {"phase": "failed", "at": now(),
                  "error": str(exc), "traceback": traceback.format_exc()})
            raise

    write(HERE / "selection_status.json", {"phase": "nested_rankings", "at": now(), "pid": os.getpid()})
    with concurrent.futures.ThreadPoolExecutor(max_workers=plan["scheduling"]["ranking_workers"]) as pool:
        futures = [pool.submit(rank_fold, i) for i in range(1, 6)]
        full_report = wait_for_report(0)
        assert full_report["input_fingerprint"] == base_fp
        check = compare_evidence(read(BASE / "statistical_evidence.json"), full_report)
        write(HERE / "reproduced_confounder_evidence_check.json", check)
        for future in concurrent.futures.as_completed(futures):
            future.result()

    def missing(*args, **kwargs):
        raise RuntimeError("Prefetched native evidence/ranking checkpoint was not reusable")

    original_checkpoint = numerical._checkpoint

    def cached_numerical(arguments):
        def cached(directory, name, fingerprint, compute):
            return original_checkpoint(directory, name, fingerprint, missing)
        with patch.object(numerical, "_checkpoint", cached):
            return numerical.select_stage2_features_multi_model(**arguments)

    original_rank = ranking.rank_modifier_candidates

    def checked_rank(**kwargs):
        if Path(kwargs["output_dir"]).name == "ranking":
            kwargs["request_json"] = missing
        else:
            assert Path(kwargs["output_dir"]).name == "full_training_ranking"
        return original_rank(**kwargs)

    write(HERE / "selection_status.json", {"phase": "native_architecture_count_validation", "at": now()})
    with patch.object(ranking, "rank_modifier_candidates", checked_rank):
        kept, updated, report = count.select_modifier_count(dataset=data, extracted_fit=train, definitions=definitions,
            inner_splits=split["inner_splits"], treatment_column="treatment", outcome_column="outcome", outcome_type="binary",
            seed=SEED, policy=policy, selected=selected, role_report=roles, statistical_report=full_report,
            request_json=request, role_policy=role_policy, output_dir=HERE / "modifier_count", numerical_checkpoint_dir=HERE / "numerical",
            model_identity=MODEL, estimation_trees=plan["evaluation_plan"]["trees"], run_numerical=cached_numerical)
    assert read(root / "input.json") == {**identity, "input_fingerprint": fp}
    assert {f["feature_id"] for f in kept if "confounder" in f["roles"]} == {f["feature_id"] for f in selected if "confounder" in f["roles"]}
    write(HERE / "selected_definitions.json", {"features": kept})
    write(HERE / "role_report.json", updated)
    write(HERE / "count_result.json", report)
    write(HERE / "one_standard_error_diagnostic.json", count.choose_effect_model(
        {(e, int(k)): v for e, sizes in report["choice"]["fold_losses"].items() for k, v in sizes.items()}, rule="one_standard_error"))
    counts = {role: sum(role in f["roles"] for f in kept) for role in ("confounder", "effect_modifier")}
    counts["both"] = sum(len(f["roles"]) == 2 for f in kept)
    manifest()
    files = [Path(__file__), HERE / "selected_definitions.json", HERE / "role_report.json", HERE / "count_result.json",
             HERE / "one_standard_error_diagnostic.json", HERE / "reproduced_confounder_evidence_check.json",
             HERE / "llm_runtime/input.json", HERE / f"select_parallel_{DATE}.py"]
    files += sorted(root.rglob("*.json"))
    write(freeze, {"at": now(), "input_manifest_sha256": sha(HERE / f"input_manifest_{DATE}.json"),
          "files": {str(p): sha(p) for p in files}, "selected_count": len(kept), "roles": counts, "chosen_estimator": report["chosen_estimator"],
          "modifier_order": [r["feature_id"] for r in report["full_training_ranking"]["ranking"][:report["choice"]["chosen_additional_count"]]],
          "native_fold_ranking_and_evidence_replay_without_cache_misses": True,
          "oracle_or_heldout_labels_read": False})
    write(HERE / "selection_status.json", {"phase": "complete", "at": now(), "roles": counts, "choice": report["choice"]})
    write(HERE / "status.json", {"phase": "selection_frozen", "at": now(), "roles": counts})
    print(json.dumps({"phase": "selection_frozen", "roles": counts, "choice": report["choice"]}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        write(HERE / "selection_status.json", {"phase": "failed", "at": now(), "error": str(exc), "traceback": traceback.format_exc()})
        raise
