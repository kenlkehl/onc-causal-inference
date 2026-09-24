"""Fold-honest all-evidence packaging and LLM causal-role adjudication.

This module receives only frozen candidate definitions and aggregate statistical
evidence computed inside one outer-training fold.  It deliberately has no
dataset argument: row-level values, outer-heldout outcomes, oracle columns, and
data-generation metadata cannot enter the role prompt through this interface.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence


SCHEMA_VERSION = "stage2_all_evidence_llm_role_adjudication_v1"
EVIDENCE_SCHEMA_VERSION = "stage2_fold_honest_role_evidence_v1"
from .stage2_clinical_prompts import PROMPT_VERSION, SYSTEM_PROMPTS
from . import stage2_clinical_prompts as clinical_prompts
TEMPORAL_SCOPE = "pre_index_treatment"
ALLOWED_ROLES = ("confounder", "effect_modifier")


class RequestJSON(Protocol):
    def __call__(
        self,
        messages: Sequence[Mapping[str, str]],
        validate: Callable[[Mapping[str, Any]], dict[str, Any]],
        *,
        request_kind: str = "interpretation",
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class Stage2RoleAdjudicationConfig:
    """Prompt-bounding policy for the final causal-role adjudicator."""

    enabled: bool = True
    max_description_chars: int = 1_200
    max_measurement_definition_chars: int = 1_600
    max_categories_per_feature: int = 40
    max_candidates_per_request: int = 20

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("stage2.role_adjudication.enabled must be boolean")
        for name in (
            "max_description_chars",
            "max_measurement_definition_chars",
            "max_categories_per_feature",
            "max_candidates_per_request",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(
                    f"stage2.role_adjudication.{name} must be a positive integer"
                )

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)


def role_adjudication_config_from_mapping(
    value: Mapping[str, Any] | None,
) -> Stage2RoleAdjudicationConfig:
    if value is not None and not isinstance(value, Mapping):
        raise ValueError("stage2.role_adjudication must be an object")
    raw = dict(value or {})
    known = set(Stage2RoleAdjudicationConfig.__dataclass_fields__)
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(
            "stage2.role_adjudication contains unsupported fields: " f"{unknown}"
        )
    config = Stage2RoleAdjudicationConfig(**raw)
    config.validate()
    return config


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _feature_id(feature: Mapping[str, Any]) -> str:
    return str(feature.get("feature_id") or feature["name"])


def _bounded_text(value: Any, limit: int) -> str:
    rendered = str(value or "").strip()
    return rendered[: int(limit)]


def _prompt_safe_definition(
    feature: Mapping[str, Any],
    *,
    policy: Stage2RoleAdjudicationConfig,
) -> dict[str, Any]:
    """Project one definition through an explicit prompt allowlist."""

    raw_categories = feature.get("categories_or_unit") or []
    if isinstance(raw_categories, (str, bytes)):
        raw_categories = [raw_categories]
    categories = [
        _bounded_text(value, 240)
        for value in list(raw_categories)[: int(policy.max_categories_per_feature)]
    ]
    locked = feature.get("configured_explicit_feature") is True
    configured_roles = (
        [
            role
            for role in ALLOWED_ROLES
            if role in set(map(str, feature.get("roles") or []))
        ]
        if locked
        else []
    )
    return {
        "feature_id": _feature_id(feature),
        "name": _bounded_text(feature.get("name"), 300),
        "description": _bounded_text(
            feature.get("description"), policy.max_description_chars
        ),
        "value_type": _bounded_text(feature.get("value_type"), 80),
        "categories_or_unit": categories,
        "categories_or_unit_truncated": len(list(raw_categories)) > len(categories),
        "measurement_definition": _bounded_text(
            feature.get("measurement_definition"),
            policy.max_measurement_definition_chars,
        ),
        "missing_value_rule": _bounded_text(
            feature.get("missing_value_rule"), 800
        ),
        "supporting_architectures": [
            _bounded_text(value, 160)
            for value in list(feature.get("supporting_architectures") or [])[:20]
        ],
        "evidence_axes": [
            _bounded_text(value, 160)
            for value in list(feature.get("evidence_axes") or [])[:20]
        ],
        "derived_equivalent_measurement": bool(
            feature.get("derived_structured_latent")
        ),
        "source_feature_ids": [
            _bounded_text(value, 300)
            for value in list(feature.get("source_feature_ids") or [])[:40]
        ],
        "investigator_locked": locked,
        "configured_roles": configured_roles,
    }


def _row_for_feature(
    rows: Sequence[Mapping[str, Any]],
    feature_id: str,
) -> Mapping[str, Any] | None:
    return next(
        (
            row
            for row in rows
            if str(row.get("feature_id") or "") == str(feature_id)
        ),
        None,
    )


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        rendered = float(value)
    except (TypeError, ValueError):
        return None
    return rendered if math.isfinite(rendered) else None


def _candidate_statistical_evidence(
    feature_id: str,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    nuisance = dict(report.get("nuisance_screen") or {})
    univariable = dict(report.get("confounder_univariable_screen") or {})
    candidate_r = dict(report.get("effect_modifier_screen") or {})
    joint_modifier = dict(
        report.get("multivariable_modifier_elastic_net_screen") or {}
    )
    nuisance_folds: list[dict[str, Any]] = []
    for fold in nuisance.get("folds") or []:
        treatment = dict(fold.get("treatment") or {})
        outcome = dict(fold.get("outcome") or {})
        nuisance_folds.append(
            {
                "inner_fold": fold.get("inner_fold"),
                "treatment_selected": feature_id
                in set(map(str, treatment.get("selected_feature_ids") or [])),
                "treatment_group_l2_norm": _finite_or_none(
                    (treatment.get("feature_group_l2_norms") or {}).get(feature_id)
                ),
                "outcome_selected": feature_id
                in set(map(str, outcome.get("selected_feature_ids") or [])),
                "outcome_group_l2_norm": _finite_or_none(
                    (outcome.get("feature_group_l2_norms") or {}).get(feature_id)
                ),
            }
        )

    univariable_folds: list[dict[str, Any]] = []
    for fold in univariable.get("folds") or []:
        row = _row_for_feature(fold.get("tests") or [], feature_id)
        if row is None:
            continue
        univariable_folds.append(
            {
                "inner_fold": fold.get("inner_fold"),
                "treatment_p_value": _finite_or_none(
                    row.get("treatment_p_value")
                ),
                "treatment_q_value": _finite_or_none(
                    row.get("treatment_q_value")
                ),
                "outcome_p_value": _finite_or_none(row.get("outcome_p_value")),
                "outcome_q_value": _finite_or_none(row.get("outcome_q_value")),
                "outcome_adjusted_for_treatment_p_value": _finite_or_none(
                    row.get("outcome_adjusted_for_treatment_p_value")
                ),
                "outcome_adjusted_for_treatment_q_value": _finite_or_none(
                    row.get("outcome_adjusted_for_treatment_q_value")
                ),
                "nominal_joint_support": bool(
                    row.get("nominal_joint_support")
                ),
                "multiplicity_adjusted_joint_support": bool(
                    row.get("multiplicity_adjusted_joint_support")
                ),
                "treatment_test_status": str(
                    (row.get("treatment_test") or {}).get("status") or ""
                ),
                "outcome_test_status": str(
                    (row.get("outcome_test") or {}).get("status") or ""
                ),
                "outcome_adjusted_for_treatment_test_status": str(
                    (row.get("outcome_adjusted_for_treatment_test") or {}).get(
                        "status"
                    )
                    or ""
                ),
            }
        )

    candidate_r_folds: list[dict[str, Any]] = []
    for fold in candidate_r.get("folds") or []:
        row = _row_for_feature(fold.get("tests") or [], feature_id)
        if row is None:
            continue
        candidate_r_folds.append(
            {
                "inner_fold": fold.get("inner_fold"),
                "status": str(row.get("status") or ""),
                "reason": _bounded_text(row.get("reason"), 500),
                "rank": row.get("rank"),
                "selected_top_n": bool(row.get("selected_top_n")),
                "heldout_r_loss_improvement": _finite_or_none(
                    row.get("heldout_r_loss_improvement")
                ),
                "heldout_relative_r_loss_improvement": _finite_or_none(
                    row.get("heldout_relative_r_loss_improvement")
                ),
                "interaction_degrees_of_freedom": row.get(
                    "interaction_degrees_of_freedom"
                ),
            }
        )

    joint_modifier_folds: list[dict[str, Any]] = []
    for fold in joint_modifier.get("folds") or []:
        selected = set(map(str, fold.get("selected_feature_ids") or []))
        joint_modifier_folds.append(
            {
                "inner_fold": fold.get("inner_fold"),
                "selected": feature_id in selected,
                "feature_group_l2_norm": _finite_or_none(
                    (fold.get("feature_group_l2_norms") or {}).get(feature_id)
                ),
                "joint_model_heldout_r_loss_improvement": _finite_or_none(
                    fold.get("heldout_r_loss_improvement")
                ),
                "joint_model_status": str(fold.get("status") or ""),
            }
        )

    provisional = _row_for_feature(report.get("decisions") or [], feature_id) or {}
    return {
        "multivariable_nuisance_elastic_net": {
            "treatment_votes": int(
                (nuisance.get("treatment_votes") or {}).get(feature_id, 0)
            ),
            "outcome_votes": int(
                (nuisance.get("outcome_votes") or {}).get(feature_id, 0)
            ),
            "folds": nuisance_folds,
        },
        "univariable_confounder_screen": {
            "nominal_joint_support_votes": int(
                (univariable.get("nominal_joint_support_votes") or {}).get(
                    feature_id, 0
                )
            ),
            "multiplicity_adjusted_joint_support_votes": int(
                (
                    univariable.get(
                        "multiplicity_adjusted_joint_support_votes"
                    )
                    or {}
                ).get(feature_id, 0)
            ),
            "folds": univariable_folds,
        },
        "candidate_augmented_r_learner": {
            "top_n_votes": int(
                (candidate_r.get("votes") or {}).get(feature_id, 0)
            ),
            "folds": candidate_r_folds,
        },
        "multivariable_modifier_elastic_net": {
            "selection_votes": int(
                (joint_modifier.get("votes") or {}).get(feature_id, 0)
            ),
            "folds": joint_modifier_folds,
        },
        "provisional_statistical_roles": [
            role
            for role in ALLOWED_ROLES
            if role in set(map(str, provisional.get("roles") or []))
        ],
    }


def build_stage2_role_evidence(
    *,
    definitions: Sequence[Mapping[str, Any]],
    statistical_report: Mapping[str, Any],
    policy: Stage2RoleAdjudicationConfig,
) -> dict[str, Any]:
    """Build the only payload permitted to enter final role adjudication."""

    policy.validate()
    definitions = [copy.deepcopy(dict(feature)) for feature in definitions]
    feature_ids = [_feature_id(feature) for feature in definitions]
    if len(feature_ids) != len(set(feature_ids)):
        raise ValueError("Stage 2 role evidence requires unique feature IDs")
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "study_context": clinical_prompts.study_context(statistical_report),
        "temporal_scope": TEMPORAL_SCOPE,
        "evidence_boundary": {
            "candidate_measurements_are_pre_index_treatment": True,
            "statistical_evidence_uses_outer_training_rows_only": True,
            "inner_heldout_rows_are_used_only_for_fold_honest_evaluation": True,
            "outer_heldout_rows_are_excluded": True,
            "row_level_values_are_excluded": True,
            "patient_identifiers_are_excluded": True,
            "oracle_columns_are_excluded": True,
            "data_generation_metadata_is_excluded": True,
            "dataset_paths_and_dataset_names_are_excluded": True,
            "definition_fields_use_an_explicit_allowlist": True,
        },
        "methodology": {
            "confounder": [
                "multivariable grouped elastic-net treatment and marginal-outcome support",
                (
                    "candidate-wise treatment and outcome association tests, "
                    "including outcome adjusted for treatment"
                ),
            ],
            "effect_modifier": [
                "candidate-augmented univariable R-learner held-out R-loss comparisons",
                "joint multivariable grouped elastic-net R-loss interaction selection",
            ],
            "all_statistics_are_evidence_not_automatic_role_labels": True,
        },
        "candidates": [
            {
                "feature_id": _feature_id(feature),
                "definition": _prompt_safe_definition(feature, policy=policy),
                "statistical_evidence": _candidate_statistical_evidence(
                    _feature_id(feature), statistical_report
                ),
            }
            for feature in definitions
        ],
    }


ROLE_ADJUDICATION_SYSTEM_PROMPT = SYSTEM_PROMPTS["14_default_roles"]


def _role_response_validator(
    *,
    definitions: Sequence[Mapping[str, Any]],
) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    ordered_ids = [_feature_id(feature) for feature in definitions]
    by_id = {_feature_id(feature): feature for feature in definitions}

    def validate(value: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise ValueError("role adjudication response must be one JSON object")
        if set(value) == {"confounder", "effect_modifier"} and len(definitions) == 1:
            value = {"summary": "Clinical role review", "decisions": [clinical_prompts.role_decision(value, definitions[0])]}
        raw_decisions = value.get("decisions")
        if not isinstance(raw_decisions, list):
            raise ValueError("role adjudication requires a decisions list")
        decisions: dict[str, dict[str, Any]] = {}
        for raw in raw_decisions:
            if not isinstance(raw, Mapping):
                raise ValueError("each role decision must be an object")
            feature_id = str(raw.get("feature_id") or "")
            if feature_id not in by_id or feature_id in decisions:
                raise ValueError(f"unknown or duplicate role candidate {feature_id!r}")
            raw_roles = raw.get("roles")
            if not isinstance(raw_roles, list):
                raise ValueError("each role decision requires a roles list")
            unknown_roles = sorted(set(map(str, raw_roles)) - set(ALLOWED_ROLES))
            if unknown_roles:
                raise ValueError(f"unsupported causal roles: {unknown_roles}")
            roles = [role for role in ALLOWED_ROLES if role in set(map(str, raw_roles))]
            feature = by_id[feature_id]
            if feature.get("configured_explicit_feature") is True:
                configured = [
                    role
                    for role in ALLOWED_ROLES
                    if role in set(map(str, feature.get("roles") or []))
                ]
                if roles != configured:
                    raise ValueError(
                        "investigator-locked features must preserve exactly their "
                        "configured roles"
                    )
            rationale = _bounded_text(raw.get("rationale"), 4_000)
            consistency = _bounded_text(
                raw.get("inner_fold_consistency"), 3_000
            )
            reconciliation = _bounded_text(
                raw.get("cross_method_reconciliation"), 3_000
            )
            if not rationale or not consistency or not reconciliation:
                raise ValueError(
                    "each role decision requires rationale, inner_fold_consistency, "
                    "and cross_method_reconciliation"
                )
            decisions[feature_id] = {
                "feature_id": feature_id,
                "roles": roles,
                "evidence_for": [
                    _bounded_text(item, 800)
                    for item in list(raw.get("evidence_for") or [])[:30]
                ],
                "evidence_against": [
                    _bounded_text(item, 800)
                    for item in list(raw.get("evidence_against") or [])[:30]
                ],
                "inner_fold_consistency": consistency,
                "cross_method_reconciliation": reconciliation,
                "rationale": rationale,
            }
            for key in ("role_assessments", "evidence_attachment", "evidence_ids", "stability"):
                if key in raw:
                    decisions[feature_id][key] = copy.deepcopy(raw[key])
        if set(decisions) != set(ordered_ids):
            raise ValueError(
                "role adjudication must cover every candidate exactly once; "
                f"missing={sorted(set(ordered_ids) - set(decisions))}, "
                f"extra={sorted(set(decisions) - set(ordered_ids))}"
            )
        return {
            "summary": _bounded_text(value.get("summary"), 6_000),
            "decisions": [decisions[feature_id] for feature_id in ordered_ids],
        }

    return validate


def _selected_from_adjudication(
    *,
    definitions: Sequence[Mapping[str, Any]],
    adjudication: Mapping[str, Any],
) -> list[dict[str, Any]]:
    decisions = {
        str(row["feature_id"]): row for row in adjudication.get("decisions") or []
    }
    selected: list[dict[str, Any]] = []
    for raw_feature in definitions:
        feature = copy.deepcopy(dict(raw_feature))
        feature_id = _feature_id(feature)
        roles = list(map(str, decisions[feature_id]["roles"]))
        if not roles:
            continue
        locked = feature.get("configured_explicit_feature") is True
        feature["roles"] = roles
        feature["nuisance_model_roles"] = (
            ["treatment", "outcome"] if "confounder" in roles else []
        )
        feature["selection_source"] = (
            "investigator_locked"
            if locked
            else "llm_all_evidence_role_adjudication"
        )
        selected.append(feature)
    return selected


def _role_request_payload(
    *,
    evidence: Mapping[str, Any],
    batch_index: int,
    batch_count: int,
) -> dict[str, Any]:
    return {
        "task": "adjudicate_stage2_roles_from_all_evidence",
        "prompt_version": PROMPT_VERSION,
        "candidate_batch": {
            "batch_index": int(batch_index),
            "batch_count": int(batch_count),
            "candidate_count": len(evidence.get("candidates") or []),
        },
        "role_evidence": copy.deepcopy(dict(evidence)),
        "decision_policy": {
            "statistical_methods_are_evidence_not_gates": True,
            "assess_disagreement_explicitly": True,
            "assess_inner_fold_consistency_explicitly": True,
            "preserve_investigator_locked_roles_exactly": True,
            "allow_no_role": True,
        },
        "required_response": {
            "summary": "string",
            "decisions": [
                {
                    "feature_id": "every supplied feature ID exactly once",
                    "roles": ["zero or more of confounder, effect_modifier"],
                    "evidence_for": ["specific supplied statistical facts"],
                    "evidence_against": ["specific supplied statistical facts"],
                    "inner_fold_consistency": "string",
                    "cross_method_reconciliation": "string",
                    "rationale": "causal-role conclusion grounded in supplied evidence",
                }
            ],
        },
    }


def adjudicate_stage2_roles(
    *,
    definitions: Sequence[Mapping[str, Any]],
    statistical_report: Mapping[str, Any],
    request_json: RequestJSON,
    output_dir: Path,
    policy: Stage2RoleAdjudicationConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Package evidence, checkpoint bounded LLM decisions, and apply them."""

    policy.validate()
    if not policy.enabled:
        raise ValueError("adjudicate_stage2_roles requires role adjudication to be enabled")
    if (statistical_report.get("policy") or {}).get("selection_mode") == "multi_model":
        from .stage2_multi_model_adjudication import adjudicate_multi_model_roles

        return adjudicate_multi_model_roles(
            definitions=definitions, statistical_report=statistical_report,
            request_json=request_json, output_dir=output_dir, policy=policy,
        )
    from .stage2_taskwise_policy import independent_tasks_enabled

    if independent_tasks_enabled(statistical_report):
        from .stage2_taskwise_annotation import annotate_taskwise_selection

        return annotate_taskwise_selection(
            definitions=definitions, statistical_report=statistical_report,
            request_json=request_json, output_dir=output_dir, policy=policy,
        )
    from .stage2_prompt_io import request_review

    definitions = [copy.deepcopy(dict(feature)) for feature in definitions]
    evidence = build_stage2_role_evidence(definitions=definitions, statistical_report=statistical_report, policy=policy)
    identity = {"evidence": _fingerprint(evidence), "policy": policy.public_dict(),
                "model": statistical_report.get("adjudication_model_identity"), "prompt": PROMPT_VERSION}
    _write_json(output_dir / "evidence.json", evidence)
    decisions = []
    for index, (feature, card) in enumerate(zip(definitions, evidence["candidates"])):
        messages = clinical_prompts.messages("14_default_roles", clinical_prompts.evidence_input(evidence, [card]))
        response = request_review(output_dir / "batches" / f"batch_{index + 1:03d}", messages,
            _role_response_validator(definitions=[feature]), request_json=request_json, identity=identity)
        decisions.extend(response["decisions"])
    adjudication = _role_response_validator(definitions=definitions)({"summary": "Clinical role reviews", "decisions": decisions})
    selected = _selected_from_adjudication(definitions=definitions, adjudication=adjudication)
    report = {"schema_version": SCHEMA_VERSION, "prompt_version": PROMPT_VERSION, "status": "complete",
              "evidence_fingerprint": identity["evidence"], "adjudication_fingerprint": _fingerprint(identity),
              "batch_count": len(definitions), "max_candidates_per_request": 1,
              "failure_policy": "fail_outer_fold_without_statistical_fallback",
              "prompt_data_contract": evidence["evidence_boundary"], **adjudication,
              "retained_feature_ids": [_feature_id(f) for f in selected]}
    _write_json(output_dir / "response.json", report)
    _write_json(output_dir / "complete.json", {"status": "complete", "identity": identity})
    return selected, report, evidence


__all__ = [
    "ALLOWED_ROLES",
    "EVIDENCE_SCHEMA_VERSION",
    "PROMPT_VERSION",
    "ROLE_ADJUDICATION_SYSTEM_PROMPT",
    "SCHEMA_VERSION",
    "Stage2RoleAdjudicationConfig",
    "adjudicate_stage2_roles",
    "build_stage2_role_evidence",
    "role_adjudication_config_from_mapping",
]
