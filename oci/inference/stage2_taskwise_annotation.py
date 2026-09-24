"""Optional clinical explanations of fixed numerical selection decisions."""
from pathlib import Path

from . import stage2_clinical_prompts as clinical_prompts
from .stage2_prompt_io import request_review
from .stage2_taskwise_policy import INDEPENDENT_TASKS, route_from_statistical_report

SCHEMA_VERSION = "stage2_taskwise_advisory_roles_v2_clinical_explanation"
SYSTEM_PROMPT = clinical_prompts.SYSTEM_PROMPTS["20_advisory_roles"]


def _annotation_validator(value):
    if set(value) != {"interpretation", "limitations"} or any(not isinstance(value[k], str) or not value[k].strip() for k in value):
        raise ValueError("return interpretation and limitations as text")
    return dict(value)


def annotate_taskwise_selection(*, definitions, statistical_report, request_json, output_dir, policy):
    from .stage2_role_adjudication import _fingerprint, _write_json, build_stage2_role_evidence
    policy.validate()
    selected, decisions, routing = route_from_statistical_report(definitions, statistical_report)
    evidence = build_stage2_role_evidence(definitions=definitions, statistical_report=statistical_report, policy=policy)
    directory = Path(output_dir) / "advisory"
    _write_json(directory / "evidence.json", evidence)
    identity = {"version": SCHEMA_VERSION, "prompt": clinical_prompts.PROMPT_VERSION, "evidence": evidence,
                "decisions": decisions, "model": statistical_report.get("adjudication_model_identity")}
    by_id = {r["feature_id"]: r for r in decisions}
    annotations, failures = [], []
    for index, card in enumerate(evidence["candidates"]):
        feature_id = card["feature_id"]
        messages = clinical_prompts.messages("20_advisory_roles", clinical_prompts.evidence_input(evidence, [card])
            + "\n\nRecorded decision\n" + clinical_prompts.readable(by_id[feature_id]))
        try:
            response = request_review(directory / "batches" / f"batch_{index + 1:03d}", messages,
                _annotation_validator, request_json=request_json, identity=identity)
        except Exception as exc:
            if isinstance(exc, OSError) and not isinstance(exc, TimeoutError):
                raise
            failure = {"feature_id": feature_id, "error_type": type(exc).__name__}
            failures.append(failure)
            _write_json(directory / "batches" / f"batch_{index + 1:03d}" / "failure.json", failure)
            continue
        annotations.append({"feature_id": feature_id, **response})
    report = {"schema_version": SCHEMA_VERSION, "status": "complete_with_annotation_failures" if failures else "complete",
        "mode": "annotation_only", "selection_authority": INDEPENDENT_TASKS, "llm_may_change_selection": False,
        "failure_policy": "keep_numerical_selection_record_annotation_failure", "input_fingerprint": _fingerprint(identity),
        "prompt_data_contract": evidence["evidence_boundary"], "summary": "Clinical explanations of fixed selection decisions",
        "decisions": decisions, "annotations": annotations, "annotation_failures": failures, "taskwise_routing": routing,
        "retained_feature_ids": [str(f.get("feature_id") or f["name"]) for f in selected]}
    _write_json(directory / "report.json", report)
    return selected, report, evidence
