from __future__ import annotations

import inspect
import json

import pytest

from oci.inference.stage2_role_adjudication import (
    EVIDENCE_SCHEMA_VERSION,
    ROLE_ADJUDICATION_SYSTEM_PROMPT,
    Stage2RoleAdjudicationConfig,
    adjudicate_stage2_roles,
    build_stage2_role_evidence,
)


def _definition(feature_id: str, *, locked_roles=None):
    result = {
        "feature_id": feature_id,
        "name": f"measurement_{feature_id}",
        "description": "A pretreatment patient measurement.",
        "value_type": "continuous",
        "categories_or_unit": ["unitless"],
        "measurement_definition": "Extract the pretreatment value.",
        "missing_value_rule": "Return null when absent.",
        "supporting_architectures": ["bow_nuisance", "bow_r_loss"],
        "evidence_axes": ["treatment", "outcome", "residual_effect"],
    }
    if locked_roles is not None:
        result["configured_explicit_feature"] = True
        result["roles"] = list(locked_roles)
    return result


def _statistical_report(feature_ids):
    return {
        "nuisance_screen": {
            "treatment_votes": {feature_id: 2 for feature_id in feature_ids},
            "outcome_votes": {feature_id: 1 for feature_id in feature_ids},
            "folds": [
                {
                    "inner_fold": 1,
                    "treatment": {
                        "selected_feature_ids": list(feature_ids),
                        "feature_group_l2_norms": {
                            feature_id: 0.5 for feature_id in feature_ids
                        },
                    },
                    "outcome": {
                        "selected_feature_ids": list(feature_ids),
                        "feature_group_l2_norms": {
                            feature_id: 0.4 for feature_id in feature_ids
                        },
                    },
                }
            ],
        },
        "confounder_univariable_screen": {
            "nominal_joint_support_votes": {
                feature_id: 1 for feature_id in feature_ids
            },
            "multiplicity_adjusted_joint_support_votes": {
                feature_id: 1 for feature_id in feature_ids
            },
            "folds": [
                {
                    "inner_fold": 1,
                    "tests": [
                        {
                            "feature_id": feature_id,
                            "treatment_p_value": 0.01,
                            "treatment_q_value": 0.02,
                            "outcome_p_value": 0.02,
                            "outcome_q_value": 0.03,
                            "outcome_adjusted_for_treatment_p_value": 0.03,
                            "outcome_adjusted_for_treatment_q_value": 0.04,
                            "nominal_joint_support": True,
                            "multiplicity_adjusted_joint_support": True,
                            "treatment_test": {"status": "ok"},
                            "outcome_test": {"status": "ok"},
                        }
                        for feature_id in feature_ids
                    ],
                }
            ],
        },
        "effect_modifier_screen": {
            "votes": {feature_id: 1 for feature_id in feature_ids},
            "folds": [
                {
                    "inner_fold": 1,
                    "tests": [
                        {
                            "feature_id": feature_id,
                            "status": "ok",
                            "rank": index,
                            "selected_top_n": True,
                            "heldout_r_loss_improvement": 0.05,
                            "heldout_relative_r_loss_improvement": 0.03,
                            "interaction_degrees_of_freedom": 1,
                        }
                        for index, feature_id in enumerate(feature_ids, start=1)
                    ],
                }
            ],
        },
        "multivariable_modifier_elastic_net_screen": {
            "votes": {feature_id: 1 for feature_id in feature_ids},
            "folds": [
                {
                    "inner_fold": 1,
                    "status": "ok",
                    "selected_feature_ids": list(feature_ids),
                    "feature_group_l2_norms": {
                        feature_id: 0.25 for feature_id in feature_ids
                    },
                    "heldout_r_loss_improvement": 0.04,
                }
            ],
        },
        "decisions": [
            {
                "feature_id": feature_id,
                "roles": ["confounder", "effect_modifier"],
            }
            for feature_id in feature_ids
        ],
    }


def test_role_evidence_uses_allowlisted_aggregate_inputs_only():
    definition = {
        **_definition("candidate_a"),
        "oracle_role": "DO_NOT_LEAK_ORACLE_MARKER",
        "ground_truth": "DO_NOT_LEAK_TRUTH_MARKER",
        "data_generation_notes": "DO_NOT_LEAK_DGP_MARKER",
        "source_dataset_path": "DO_NOT_LEAK_DATASET_PATH_MARKER",
    }

    evidence = build_stage2_role_evidence(
        definitions=[definition],
        statistical_report=_statistical_report(["candidate_a"]),
        policy=Stage2RoleAdjudicationConfig(),
    )

    rendered = json.dumps(evidence)
    assert evidence["schema_version"] == EVIDENCE_SCHEMA_VERSION
    assert evidence["evidence_boundary"]["outer_heldout_rows_are_excluded"] is True
    assert evidence["evidence_boundary"]["oracle_columns_are_excluded"] is True
    assert "DO_NOT_LEAK" not in rendered
    assert "oracle_role" not in rendered
    assert "ground_truth" not in rendered
    assert "source_dataset_path" not in rendered
    assert "dataset" not in inspect.signature(build_stage2_role_evidence).parameters


def test_role_prompt_has_no_fixture_specific_truth_hints():
    normalized = ROLE_ADJUDICATION_SYSTEM_PROMPT.casefold()
    for forbidden in (
        "nsclc",
        "one_confounder",
        "five_confounders",
        "performance_status is the confounder",
        "biomarker is the modifier",
    ):
        assert forbidden not in normalized


def test_adjudication_applies_roles_preserves_lock_and_reuses_checkpoint(tmp_path):
    definitions = [
        _definition("candidate_a"),
        _definition("locked_b", locked_roles=["confounder"]),
    ]
    calls = []

    def request_json(messages, validate, *, request_kind="interpretation"):
        assert request_kind == "interpretation"
        payload = json.loads(messages[1]["content"])
        assert payload["task"] == "adjudicate_stage2_roles_from_all_evidence"
        assert "dataset" not in payload
        calls.append(payload)
        return validate(
            {
                "summary": "Reconciled all statistical views.",
                "decisions": [
                    {
                        "feature_id": "candidate_a",
                        "roles": ["effect_modifier"],
                        "evidence_for": ["Positive held-out R-loss evidence."],
                        "evidence_against": ["Confounder evidence was inconsistent."],
                        "inner_fold_consistency": "Modifier support recurred.",
                        "cross_method_reconciliation": "Both modifier views agreed.",
                        "rationale": "Retain only as an effect modifier.",
                    },
                    {
                        "feature_id": "locked_b",
                        "roles": ["confounder"],
                        "evidence_for": ["Investigator-locked role."],
                        "evidence_against": [],
                        "inner_fold_consistency": "The lock is invariant.",
                        "cross_method_reconciliation": "Empirical evidence is advisory.",
                        "rationale": "Preserve the configured role exactly.",
                    },
                ],
            }
        )

    arguments = {
        "definitions": definitions,
        "statistical_report": _statistical_report(
            ["candidate_a", "locked_b"]
        ),
        "request_json": request_json,
        "output_dir": tmp_path / "role_adjudication",
        "policy": Stage2RoleAdjudicationConfig(),
    }
    selected, report, _evidence = adjudicate_stage2_roles(**arguments)
    selected_by_id = {row["feature_id"]: row for row in selected}
    assert selected_by_id["candidate_a"]["roles"] == ["effect_modifier"]
    assert selected_by_id["candidate_a"]["nuisance_model_roles"] == []
    assert selected_by_id["candidate_a"]["selection_source"] == (
        "llm_all_evidence_role_adjudication"
    )
    assert selected_by_id["locked_b"]["roles"] == ["confounder"]
    assert selected_by_id["locked_b"]["nuisance_model_roles"] == [
        "treatment",
        "outcome",
    ]
    assert report["failure_policy"] == (
        "fail_outer_fold_without_statistical_fallback"
    )
    assert len(calls) == 1

    cached, _cached_report, _cached_evidence = adjudicate_stage2_roles(**arguments)
    assert cached == selected
    assert len(calls) == 1


def test_adjudication_rejects_changed_investigator_locked_role(tmp_path):
    definitions = [_definition("locked", locked_roles=["confounder"])]

    def request_json(_messages, validate, *, request_kind="interpretation"):
        assert request_kind == "interpretation"
        return validate(
            {
                "summary": "Invalid fixture response.",
                "decisions": [
                    {
                        "feature_id": "locked",
                        "roles": ["effect_modifier"],
                        "evidence_for": [],
                        "evidence_against": [],
                        "inner_fold_consistency": "Not relevant.",
                        "cross_method_reconciliation": "Not relevant.",
                        "rationale": "Attempt to change the lock.",
                    }
                ],
            }
        )

    with pytest.raises(ValueError, match="investigator-locked"):
        adjudicate_stage2_roles(
            definitions=definitions,
            statistical_report=_statistical_report(["locked"]),
            request_json=request_json,
            output_dir=tmp_path / "role_adjudication",
            policy=Stage2RoleAdjudicationConfig(),
        )


def test_adjudication_batches_large_candidate_sets_and_aggregates_in_order(tmp_path):
    feature_ids = [f"candidate_{index}" for index in range(5)]
    definitions = [_definition(feature_id) for feature_id in feature_ids]
    observed_batches = []

    def request_json(messages, validate, *, request_kind="interpretation"):
        assert request_kind == "interpretation"
        payload = json.loads(messages[1]["content"])
        supplied_ids = [
            row["feature_id"]
            for row in payload["role_evidence"]["candidates"]
        ]
        observed_batches.append(
            (
                payload["candidate_batch"]["batch_index"],
                payload["candidate_batch"]["batch_count"],
                supplied_ids,
            )
        )
        return validate(
            {
                "summary": "Batch evidence reconciled.",
                "decisions": [
                    {
                        "feature_id": feature_id,
                        "roles": ["confounder"],
                        "evidence_for": ["Joint treatment/outcome support."],
                        "evidence_against": [],
                        "inner_fold_consistency": "Support recurred.",
                        "cross_method_reconciliation": "Screens agreed.",
                        "rationale": "Retain as a confounder.",
                    }
                    for feature_id in supplied_ids
                ],
            }
        )

    selected, report, _evidence = adjudicate_stage2_roles(
        definitions=definitions,
        statistical_report=_statistical_report(feature_ids),
        request_json=request_json,
        output_dir=tmp_path / "role_adjudication",
        policy=Stage2RoleAdjudicationConfig(max_candidates_per_request=2),
    )

    assert observed_batches == [
        (1, 3, feature_ids[0:2]),
        (2, 3, feature_ids[2:4]),
        (3, 3, feature_ids[4:5]),
    ]
    assert [row["feature_id"] for row in selected] == feature_ids
    assert report["batch_count"] == 3
    assert report["max_candidates_per_request"] == 2
