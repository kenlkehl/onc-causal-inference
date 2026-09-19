"""Task-selection contracts, annotation isolation, and numerical smoke tests."""
from __future__ import annotations

import copy
import json
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from oci.inference import stage2_elastic_net_selection as selection
from oci.inference.stage2_role_adjudication import (
    Stage2RoleAdjudicationConfig, adjudicate_stage2_roles,
)
from oci.inference.stage2_taskwise_policy import (
    finalize_taskwise_report, route_taskwise_features,
)


def definitions():
    return [{"feature_id": name, "name": name, "value_type": "continuous",
             "description": "Pretreatment measurement.", "roles": [],
             "categories_or_unit": [], "measurement_definition": "Extract the measurement.",
             "missing_value_rule": "Return null if missing."}
            for name in ("t_only", "y_only", "effect_only", "shared", "noise")]


def evidence():
    defs = definitions()
    keys = [f["feature_id"] for f in defs]
    votes = lambda positive: {key: int(key in positive) for key in keys}
    return {
        "policy": {"selection_mode": "independent_tasks"}, "inner_folds": 2,
        "nuisance_screen": {
            "treatment_votes": votes({"t_only", "shared"}),
            "outcome_votes": votes({"y_only", "shared"}),
            "folds": [],
        },
        "cross_fitted_nuisance_models": {},
        "effect_modifier_screen": {
            "votes": {key: 2 for key in keys},
            "stable_effect_modifier_feature_ids": keys, "folds": [],
        },
        "multivariable_modifier_elastic_net_screen": {
            "votes": votes({"effect_only", "shared"}), "folds": [],
        },
        "confounder_univariable_screen": {"folds": []},
    }


def test_separate_tasks_and_no_top_n_promotion():
    raw = evidence()
    before = copy.deepcopy(raw)
    selected, report = finalize_taskwise_report(definitions(), raw)
    by_id = {f["feature_id"]: f for f in selected}
    assert set(by_id) == {"t_only", "y_only", "effect_only", "shared"}
    assert by_id["t_only"]["nuisance_model_roles"] == ["treatment"]
    assert by_id["y_only"]["nuisance_model_roles"] == ["outcome"]
    assert by_id["effect_only"]["nuisance_model_roles"] == []
    assert by_id["effect_only"]["roles"] == ["effect_modifier"]
    assert by_id["shared"]["forest_role"] == "X"
    routing = report["taskwise_routing"]
    assert routing["forest_x_feature_ids"] == ["effect_only", "shared"]
    assert routing["forest_w_feature_ids"] == ["t_only", "y_only"]
    assert not set(routing["forest_x_feature_ids"]) & set(routing["forest_w_feature_ids"])
    assert report["cross_fitted_nuisance_models"]["treatment_feature_ids"] == sorted(f["feature_id"] for f in definitions())
    assert raw == before


@pytest.mark.parametrize("roles", [["confounder"], ["effect_modifier"], ["confounder", "effect_modifier"]])
def test_investigator_locks_are_exact(roles):
    defs = definitions()
    defs[-1].update(configured_explicit_feature=True, roles=roles)
    selected, _ = finalize_taskwise_report(defs, evidence())
    locked = next(f for f in selected if f["feature_id"] == "noise")
    assert locked["roles"] == roles
    assert locked["forest_role"] == ("X" if "effect_modifier" in roles else "W")
    assert locked["nuisance_model_roles"] == (["treatment", "outcome"] if "confounder" in roles else [])


def test_explicit_confounder_cannot_be_promoted_by_effect_votes():
    defs = definitions()
    defs[3].update(configured_explicit_feature=True, roles=["confounder"])
    selected, _ = finalize_taskwise_report(defs, evidence())
    locked = next(f for f in selected if f["feature_id"] == "shared")
    assert locked["forest_role"] == "W"
    assert locked["roles"] == ["confounder"]


@pytest.mark.parametrize("bad", [-1, 3, 0.5, True])
def test_invalid_votes_fail(bad):
    raw = evidence()
    raw["nuisance_screen"]["treatment_votes"]["noise"] = bad
    with pytest.raises(ValueError):
        finalize_taskwise_report(definitions(), raw)


def test_empty_effect_selection_is_not_padded_with_top_n():
    raw = evidence()
    raw["multivariable_modifier_elastic_net_screen"]["votes"] = dict.fromkeys(raw["nuisance_screen"]["treatment_votes"], 0)
    _, report = finalize_taskwise_report(definitions(), raw)
    assert report["taskwise_routing"]["forest_x_feature_ids"] == []


def test_default_config_does_not_change_legacy_fingerprint():
    config = selection.Stage2ElasticNetSelectionConfig()
    expected = asdict(config)
    for field in (
        "min_propensity",
        "max_propensity",
        "selection_mode",
        "nuisance_selection_frequency",
        "modifier_selection_frequency",
        "nuisance_forest_trees",
        "nuisance_forest_min_samples_leaf",
        "modifier_min_fold_r_loss_improvement",
        "modifier_min_mean_r_loss_improvement",
        "modifier_min_positive_fold_fraction",
    ):
        expected.pop(field)
    assert config.public_dict() == expected
    configured = selection.statistical_selection_config_from_mapping({"selection_mode": "independent_tasks"})
    assert configured.public_dict()["selection_mode"] == "independent_tasks"
    with pytest.raises(ValueError):
        selection.statistical_selection_config_from_mapping({"selection_mode": "typo"})


def response_for(payload, roles):
    return {"summary": "Advisory only.", "decisions": [
        {"feature_id": row["feature_id"], "roles": roles,
         "rationale": "Uncertain causal interpretation.", "inner_fold_consistency": "Mixed.",
         "cross_method_reconciliation": "Different tasks answer different questions."}
        for row in payload["role_evidence"]["candidates"]]}


@pytest.mark.parametrize("roles", [[], ["confounder", "effect_modifier"]])
def test_llm_cannot_veto_or_promote(tmp_path, roles):
    raw = evidence()
    numerical, numeric_report = finalize_taskwise_report(definitions(), raw)
    def request(messages, validate, **kwargs):
        payload = json.loads(messages[-1]["content"])
        assert payload["decision_policy"]["annotation_only"]
        return validate(response_for(payload, roles))
    selected, report, _ = adjudicate_stage2_roles(
        definitions=definitions(), statistical_report=numeric_report, request_json=request,
        output_dir=tmp_path, policy=Stage2RoleAdjudicationConfig(max_candidates_per_request=2),
    )
    assert selected == numerical
    assert report["decisions"] == numeric_report["decisions"]
    assert all(row["roles"] == roles for row in report["annotations"])
    assert report["mode"] == "annotation_only"
    assert report["selection_authority"] == "independent_tasks"


@pytest.mark.parametrize("failure", [TimeoutError, ValueError])
def test_annotation_failures_preserve_selection(tmp_path, failure):
    selected, numeric_report = finalize_taskwise_report(definitions(), evidence())
    def request(*args, **kwargs):
        raise failure("not to be serialized")
    actual, report, _ = adjudicate_stage2_roles(
        definitions=definitions(), statistical_report=numeric_report, request_json=request,
        output_dir=tmp_path, policy=Stage2RoleAdjudicationConfig(),
    )
    assert actual == selected
    assert report["annotation_failures"][0]["error_type"] == failure.__name__
    assert "not to be serialized" not in json.dumps(report)


def test_annotations_cached_but_not_binding(tmp_path):
    _, raw = finalize_taskwise_report(definitions(), evidence())
    calls = []
    def request(messages, validator, **kwargs):
        calls.append(1)
        return validator(response_for(json.loads(messages[-1]["content"]), []))
    kwargs = dict(definitions=definitions(), statistical_report=raw, request_json=request,
                  output_dir=tmp_path, policy=Stage2RoleAdjudicationConfig(max_candidates_per_request=2))
    first = adjudicate_stage2_roles(**kwargs)
    second = adjudicate_stage2_roles(**kwargs)
    assert len(calls) == 3
    assert first == second
    assert not (tmp_path / "complete.json").exists()  # no binding-role cache overwritten


def test_advisory_payload_excludes_unallowlisted_data(tmp_path):
    defs = definitions()
    defs[0]["patient_rows"] = [{"oracle": "DO_NOT_SEND"}]
    raw = evidence(); raw["dataset"] = "DO_NOT_SEND"
    def request(messages, validator, **kwargs):
        assert "DO_NOT_SEND" not in json.dumps(messages)
        return validator(response_for(json.loads(messages[-1]["content"]), []))
    adjudicate_stage2_roles(definitions=defs, statistical_report=raw, request_json=request,
                           output_dir=tmp_path, policy=Stage2RoleAdjudicationConfig())


def test_legacy_roles_remain_binding(tmp_path):
    raw = evidence(); raw["policy"] = {}
    def request(messages, validator, **kwargs):
        return validator(response_for(json.loads(messages[-1]["content"]), []))
    selected, report, _ = adjudicate_stage2_roles(
        definitions=definitions(), statistical_report=raw, request_json=request,
        output_dir=tmp_path, policy=Stage2RoleAdjudicationConfig(),
    )
    assert selected == []
    assert report["schema_version"] == "stage2_all_evidence_llm_role_adjudication_v1"
    assert (tmp_path / "complete.json").exists()


def numerical_inputs():
    rng = np.random.default_rng(318)
    n = 120
    x = rng.normal(size=(n, 3))
    treatment = rng.binomial(1, 1 / (1 + np.exp(-x[:, 0])))
    risk = 1 / (1 + np.exp(-(0.3 * x[:, 0] + x[:, 1] + treatment * (0.2 + x[:, 2]))))
    dataset = pd.DataFrame({"a": treatment, "y": rng.binomial(1, risk), "oracle_do_not_use": np.nan})
    frame = pd.DataFrame(x, columns=["v0", "v1", "v2"])
    frame["_oci_row_id"] = np.arange(n)
    defs = [{"feature_id": f"v{k}", "name": f"v{k}", "value_type": "continuous", "roles": []} for k in range(3)]
    # Last 20 rows are outer-heldout and must not enter this selector.
    ids = np.arange(100)
    splits = [{"inner_fold": k + 1, "fit_row_ids": ids[ids % 2 != k].tolist(),
               "heldout_row_ids": ids[ids % 2 == k].tolist()} for k in range(2)]
    config = selection.Stage2ElasticNetSelectionConfig(
        selection_mode="independent_tasks", internal_cv_folds=2,
        regularization_grid_size=3, max_iter=80, optimization_tolerance=1e-4,
    )
    return dict(dataset=dataset, extracted_fit=frame.iloc[:100], definitions=defs,
                inner_splits=splits, treatment_column="a", outcome_column="y",
                outcome_type="binary", seed=21, policy=config)


def test_numerical_selector_smoke_and_all_candidate_nuisances():
    kwargs = numerical_inputs()
    selected, report, dependencies, latents = selection.select_stage2_features_elastic_net(**kwargs)
    assert report["selection_authority"] == "independent_tasks"
    assert latents == [] and dependencies == selected
    assert report["cross_fitted_nuisance_models"]["cross_fold_screen_union_is_used"] is False
    for fold in report["cross_fitted_nuisance_models"]["folds"]:
        assert fold["treatment_features"] == 3
        assert fold["outcome_features"] == 3
    assert np.isfinite(report["cross_fitted_nuisance_models"]["overall_treatment_log_loss"])
    assert np.isfinite(report["cross_fitted_nuisance_models"]["overall_outcome_loss"])
    assert len(report["multivariable_modifier_elastic_net_screen"]["folds"]) == 2


def test_outer_heldout_outcomes_do_not_affect_selection():
    kwargs = numerical_inputs()
    first = selection.select_stage2_features_elastic_net(**kwargs)
    changed = kwargs["dataset"].copy()
    changed.loc[100:, ["a", "y"]] = 99  # invalid values must remain unread
    changed["oracle_do_not_use"] = "unseen truth"
    kwargs["dataset"] = changed
    second = selection.select_stage2_features_elastic_net(**kwargs)
    assert first == second
