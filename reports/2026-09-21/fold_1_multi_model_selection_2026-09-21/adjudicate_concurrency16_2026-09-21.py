"""Live theme review and role assignment; aggregate training evidence only."""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import importlib.util
import threading

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("numerical_run", HERE / "numerical_2026-09-21.py")
n = importlib.util.module_from_spec(spec)
spec.loader.exec_module(n)


def runtime():
    from oci.inference.plain_handoff_stage2 import PlainHandoffStage2, plain_stage2_config_from_mapping

    plan = n.read(HERE / f"input_manifest_{n.DATE}.json")
    cfg = n.read(n.SOURCE / "inputs/refresh_config.json")
    cfg.update(endpoint=plan["adjudication"]["endpoint"], model=plan["adjudication"]["model"],
               api_key="EMPTY", extraction_llm=None, vllm=None, workers=16,
               statistical_selection=plan["policy"], role_adjudication=plan["role_policy"],
               max_tokens=plan["adjudication"]["max_tokens"],
               interpretation_reasoning_effort=plan["adjudication"]["reasoning_effort"],
               # Reserve room for the model's response and validator-guided repairs.
               max_prompt_chars=200000)
    cfg["selection_consolidation"]["enabled"] = False
    cfg["statistical_selection"] = {k: v for k, v in plan["policy"].items()
                                     if k not in {"min_propensity", "max_propensity"}}
    instance = PlainHandoffStage2(config=plain_stage2_config_from_mapping(cfg, default_workers=1),
                                  clinical_question="")
    instance._check_and_record_model_identity(HERE / "llm_runtime")
    return instance


def main(preflight=False):
    from oci.inference import stage2_request_audit
    from oci.inference.plain_handoff_stage2 import _request_json
    from oci.inference.stage2_role_adjudication import (
        adjudicate_stage2_roles, role_adjudication_config_from_mapping,
    )
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    manifest = n.read(HERE / f"input_manifest_{n.DATE}.json")
    n.verify(manifest["sources"])
    n.verify(manifest["input_files"])
    service = runtime()
    primary = service.model_identity["primary"]
    if preflight:
        print(json.dumps({"phase": "adjudicator_ready", "identity": primary}), flush=True)
        return
    numerical = n.read(HERE / f"numerical_frozen_{n.DATE}.json")
    assert numerical["input_manifest_sha256"] == n.sha(HERE / f"input_manifest_{n.DATE}.json")
    n.verify(numerical["files"])
    runtime_manifest = {
        "source_sha256": n.sha(Path(__file__)), "numerical_freeze_sha256": n.sha(HERE / f"numerical_frozen_{n.DATE}.json"),
        "parallel_review_sha256": n.sha(HERE / "parallel_review.py"), "parallel_workers": 16,
        "predecessor_runtime_manifest_sha256": n.sha(HERE / "llm_runtime/input.json"),
        "operational_change": "Increase independent request concurrency from 4 to 16; preserve scientific policy and native checkpoints.",
        "model_identity": primary, "role_policy": manifest["role_policy"],
        "max_tokens": service.config.max_tokens, "max_prompt_chars_with_repairs": service.config.max_prompt_chars,
        "interpretation_reasoning_effort": service.config.interpretation_reasoning_effort,
        "max_response_repairs": service.config.max_response_repairs,
        "request_attempt_timeout": service.config.request_attempt_timeout,
        "request_timeout": service.config.request_timeout,
    }
    path = HERE / "llm_runtime/input_concurrency16.json"
    if path.exists():
        assert n.read(path) == runtime_manifest, "Adjudication runtime changed"
    else:
        n.write(path, runtime_manifest)
    stat = n.read(HERE / "statistical_evidence.json")
    stat["adjudication_model_identity"] = primary
    count = 0
    count_lock = threading.Lock()

    def request(messages, validate, *, request_kind="interpretation"):
        nonlocal count
        payload = json.loads(messages[-1]["content"])
        phase = payload["task"]
        with count_lock:
            count += 1
            request_number = count
            n.write(HERE / "status.json", {"phase": phase, "request": request_number, "updated_at": n.now(),
                                          "candidates": len(payload.get("candidates", [])),
                                          "themes": len(payload.get("themes", []))})
        with stage2_request_audit.context(_audit_path=str(HERE / "llm_runtime/request_events.jsonl"),
                                         phase=phase, experiment_request=request_number, runtime_revision="concurrency16"):
            return _request_json(messages=messages, config=service.config,
                                 completion=service.completion, validate=validate, request_kind=request_kind)

    definitions = n.read(n.INPUTS / "inputs/definitions.json")
    from parallel_review import prefetch
    role_policy = role_adjudication_config_from_mapping(manifest["role_policy"])
    parallel_audit = prefetch(definitions=definitions, statistical_report=stat, request_json=request,
                             output_dir=HERE / "role_adjudication", policy=role_policy, workers=16)

    def forbid_cache_miss(*args, **kwargs):
        raise RuntimeError("Parallel review differed from the native adjudicator: unexpected cache miss")

    selected, report, evidence = adjudicate_stage2_roles(
        definitions=definitions, statistical_report=stat, request_json=forbid_cache_miss,
        output_dir=HERE / "role_adjudication",
        policy=role_policy,
    )
    n.write(HERE / "llm_runtime/parallel_review_validation.json", {
        **parallel_audit, "native_replay_validated_all_responses_without_new_requests": True,
    })
    assert len(report["decisions"]) == len(definitions)
    n.write(HERE / "selected_definitions.json", {"features": selected})
    n.write(HERE / "role_report.json", report)
    files = [HERE / "selected_definitions.json", HERE / "role_report.json", path,
             Path(__file__), HERE / "parallel_review.py", HERE / "llm_runtime/parallel_review_validation.json",
             HERE / "role_adjudication/evidence.json", HERE / "role_adjudication/themes.json"]
    files += [HERE / "adjudicate_2026-09-21.py", HERE / "llm_runtime/input.json", HERE / "concurrency_transition_2026-09-21.json"]
    files += sorted((HERE / "role_adjudication").glob("**/response.json"))
    n.verify(manifest["sources"])
    n.verify(numerical["files"])
    counts = {role: sum(role in d["roles"] for d in selected) for role in ["confounder", "effect_modifier"]}
    n.write(HERE / f"selection_frozen_{n.DATE}.json", {
        "frozen_at": n.now(), "files": {str(p.resolve()): n.sha(p) for p in files},
        "input_manifest_sha256": n.sha(HERE / f"input_manifest_{n.DATE}.json"),
        "candidate_count": len(definitions), "selected_count": len(selected), "roles": counts,
        "oracle_or_outer_test_information_read": False,
    })
    n.write(HERE / "status.json", {"phase": "selection_complete", "updated_at": n.now(),
                                  "selected_count": len(selected), "roles": counts})
    print(json.dumps({"phase": "selection_complete", "selected_count": len(selected), "roles": counts}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    main(args.preflight)
