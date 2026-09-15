"""Optional, nonbinding LLM interpretation of task-wise numerical selection."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .stage2_taskwise_policy import INDEPENDENT_TASKS, route_from_statistical_report

SCHEMA_VERSION = "stage2_taskwise_advisory_roles_v1"
SYSTEM_PROMPT = """Interpret clinical measurements inside one outer-training fold.
Your role assessments are advisory annotations only: they cannot add, remove,
or reroute any modeling input. All supplied measurements are pretreatment by
an upstream contract, not a conclusion you need to infer from feature names.
Distinguish causal common causes from prognosis-only or treatment-only signals;
statistical predictiveness alone cannot establish confounding or an instrument.
Discuss effect modification on the outcome risk-difference scale for binary
outcomes. Address uncertainty, method disagreement, and correlated evidence.
Never assume a synthetic data-generating process or unseen oracle information.
Use only supplied definitions and aggregate statistics. Preserve investigator
roles in annotations. Return the requested JSON with one entry per candidate.
""".strip()


def annotate_taskwise_selection(
    *,
    definitions: Sequence[Mapping[str, Any]],
    statistical_report: Mapping[str, Any],
    request_json: Callable[..., Any],
    output_dir: Path,
    policy: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Freeze numerical routing first; isolate annotation failures and caches."""
    # Lazy import avoids an initialization cycle with the entry-point dispatch.
    from .stage2_role_adjudication import (
        _canonical_json, _fingerprint, _role_request_payload,
        _role_response_validator, _write_json, build_stage2_role_evidence,
    )

    policy.validate()
    definitions = copy.deepcopy(list(definitions))
    selected, decisions, routing = route_from_statistical_report(definitions, statistical_report)
    evidence = build_stage2_role_evidence(
        definitions=definitions, statistical_report=statistical_report, policy=policy,
    )
    annotation_dir = Path(output_dir) / "advisory"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    _write_json(annotation_dir / "evidence.json", evidence)
    fingerprint = _fingerprint({
        "schema_version": SCHEMA_VERSION,
        "system_prompt": SYSTEM_PROMPT,
        "evidence": evidence,
        "numerical_decisions": decisions,
        "policy": policy.public_dict(),
    })
    size = int(policy.max_candidates_per_request)
    total = (len(definitions) + size - 1) // size
    annotations: list[dict[str, Any]] = []
    summaries: list[str] = []
    failures: list[dict[str, Any]] = []
    for offset in range(0, len(definitions), size):
        batch_index = offset // size + 1
        batch_defs = definitions[offset:offset + size]
        batch_evidence = copy.deepcopy(evidence)
        batch_evidence["candidates"] = batch_evidence["candidates"][offset:offset + size]
        payload = _role_request_payload(
            evidence=batch_evidence, batch_index=batch_index, batch_count=total,
        )
        payload.update({"task": "annotate_stage2_roles_only", "prompt_version": SCHEMA_VERSION})
        payload["decision_policy"].update({
            "annotation_only": True,
            "may_change_numerical_selection": False,
        })
        batch_dir = annotation_dir / "batches" / f"batch_{batch_index:03d}"
        cache_path = batch_dir / "response.json"
        batch_fingerprint = _fingerprint({"run": fingerprint, "payload": payload})
        _write_json(batch_dir / "prompt.json", {"system": SYSTEM_PROMPT, "payload": payload})
        validator = _role_response_validator(definitions=batch_defs)
        response = None
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if cached.get("fingerprint") == batch_fingerprint:
                    response = validator(cached["response"])
            except (OSError, ValueError, TypeError, KeyError):
                response = None
        if response is None:
            try:
                raw = request_json(
                    [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": _canonical_json(payload)}],
                    validator,
                    request_kind="interpretation",
                )
                response = validator(raw)
            except Exception as exc:
                # Transport/validation failure is not a numerical selection gate.
                # Do not serialize arbitrary exception messages or patient data.
                failure = {"batch_index": batch_index, "error_type": type(exc).__name__}
                failures.append(failure)
                _write_json(batch_dir / "failure.json", failure)
                continue
            # Filesystem failures remain visible; do not pretend audit was saved.
            _write_json(cache_path, {"fingerprint": batch_fingerprint, "response": response})
        annotations.extend(response["decisions"])
        if response.get("summary"):
            summaries.append(response["summary"])
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete_with_annotation_failures" if failures else "complete",
        "mode": "annotation_only",
        "selection_authority": INDEPENDENT_TASKS,
        "llm_may_change_selection": False,
        "failure_policy": "keep_numerical_selection_record_annotation_failure",
        "input_fingerprint": fingerprint,
        "prompt_data_contract": evidence["evidence_boundary"],
        "summary": " ".join(summaries)[:6000],
        # The existing orchestrator uses 'decisions' as the final selection.
        # Never put model-authored annotations in this field.
        "decisions": decisions,
        "annotations": annotations,
        "annotation_failures": failures,
        "taskwise_routing": routing,
        "retained_feature_ids": [str(f.get("feature_id") or f["name"]) for f in selected],
    }
    _write_json(annotation_dir / "report.json", report)
    return selected, report, evidence
