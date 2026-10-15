"""Orthogonal-test modifier selection, the estimator ensemble, and their wiring.

Covers the default ``orthogonal_tests`` rule end to end through
``run_fold_analysis``: training-only testing nuisances that equal the final
estimation's training out-of-fold nuisances, ``modifier_tests.json``, compact
``modifier_evidence`` in ``final_definitions.json``, the ensemble's columns,
and reporting-only outer-fold confirmation that changes no other artifact.
"""

import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oci.inference import plain_handoff_stage2_analysis as analysis
from oci.inference import stage2_dina
from oci.inference import stage2_modifier_count as count
from oci.inference import stage2_modifier_tests as modifier_tests
from oci.inference import stage2_multi_model_selection as numerical
from oci.inference.plain_handoff_stage2 import (
    PlainHandoffStage2Config,
    Stage2ExtractionLLMConfig,
)
from oci.inference.stage2_diagnostics import write_heterogeneity_diagnostics
from oci.inference.stage2_elastic_net_selection import (
    statistical_selection_config_from_mapping,
)
from oci.inference.stage2_multi_model_config import ModifierSelectionConfig
from oci.inference.stage2_role_adjudication import Stage2RoleAdjudicationConfig
from tests.stage2_prompt_spy import install, prompt_inputs
from tests.test_stage2_final_modeling import candidates as final_candidates
from tests.test_stage2_final_modeling import inputs as final_inputs
from tests.test_stage2_multi_model import sample_inputs, small_policy


def _tree_hashes(root: Path, *, exclude: set[str]) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in exclude
    }


def _fold_inputs(n=300, seed=4):
    rng = np.random.default_rng(seed)
    ecog = np.tile([0, 1, 2], n // 3)
    smoker = rng.choice(["yes", "no"], size=n)
    treatment = rng.binomial(1, 0.5, size=n)
    # ECOG modifies the effect strongly; smoking is noise.
    base = np.array([0.3, 0.4, 0.5])[ecog]
    effect = np.array([0.5, 0.0, -0.4])[ecog]
    outcome = rng.binomial(1, np.clip(base + treatment * effect, 0.02, 0.98))
    dataset = pd.DataFrame({
        "patient_id": [f"p{index:03d}" for index in range(n)],
        "clinical_text": [
            f"Pretreatment ECOG {e}. Smoker: {s}." for e, s in zip(ecog, smoker)
        ],
        "treatment_indicator": treatment,
        "outcome_indicator": outcome,
        "true_ite_prob": effect,
    })
    definitions = [
        {
            "feature_id": "outer_001_feature_001",
            "name": "performance_status",
            "description": "Baseline ECOG performance status.",
            "value_type": "ordinal",
            "categories_or_unit": ["ECOG 0", "ECOG 1", "ECOG 2"],
            "roles": [],
            "measurement_definition": "Extract the pretreatment ECOG score.",
            "missing_value_rule": "Use null when undocumented.",
        },
        {
            "feature_id": "outer_001_feature_002",
            "name": "smoking_status",
            "description": "Current smoking status.",
            "value_type": "binary",
            "categories_or_unit": ["yes", "no"],
            "roles": [],
            "measurement_definition": "Extract whether the patient smokes.",
            "missing_value_rule": "Use null when undocumented.",
        },
    ]
    fit = list(range(200))
    split = {
        "outer_fold": 1,
        "fit_row_ids": fit,
        "heldout_row_ids": list(range(200, n)),
        "inner_splits": [
            {"inner_fold": 1, "fit_row_ids": fit[:100], "heldout_row_ids": fit[100:]},
            {"inner_fold": 2, "fit_row_ids": fit[100:], "heldout_row_ids": fit[:100]},
        ],
    }
    return dataset, definitions, split


def _request_json(messages, validate, *, request_kind="interpretation", **_kwargs):
    body = prompt_inputs(messages)
    if body["job"] == "extract_stage2_patient_variables":
        rows = []
        for patient in body["patients"]:
            values = {}
            for feature in body["features"]:
                if feature["name"] == "performance_status":
                    values[feature["name"]] = "ECOG " + re.search(
                        r"ECOG\s*([0-2])", patient["text"]).group(1)
                else:
                    values[feature["name"]] = re.search(
                        r"Smoker:\s*(yes|no)", patient["text"]).group(1)
            rows.append({"row_id": patient["row_id"], "values": values})
        return validate({"rows": rows})
    assert body["job"] == "review_stage2_small_model_extraction_ontology", body["job"]
    return validate({"action": "keep", "reason": "Consistent aggregate extraction."})


def test_run_fold_analysis_selects_modifiers_by_orthogonal_tests_and_ensembles(
    tmp_path, monkeypatch
):
    install(monkeypatch)
    dataset, definitions, split = _fold_inputs()
    nested = []
    monkeypatch.setattr(count, "_score_prefix", lambda **k: nested.append(k))
    stage2_dir = tmp_path / "stage2"
    output = stage2_dir / "outer_001"
    result = analysis.run_fold_analysis(
        dataset=dataset,
        definitions=definitions,
        split=split,
        clinical_question="Estimate treatment effect.",
        unit_id_column="patient_id",
        text_column="clinical_text",
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        outcome_type="binary",
        inner_folds=2,
        seed=11,
        output_dir=output,
        request_json=_request_json,
        config=PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
            model="test-model",
            extraction_llm=Stage2ExtractionLLMConfig(
                endpoint="http://small-stage2.test/v1", model="small-test-model", workers=1,
            ),
            max_prompt_chars=12_000,
            extraction_max_prompt_chars=20_000,
            max_review_rounds=1,
            estimation_trees=12,
            min_informative_patients=2,
            min_propensity=0.05,
            max_propensity=0.95,
            statistical_selection=small_policy(),
        ),
    )
    # No nested count search under the default rule.
    assert nested == []
    selection = json.loads((output / "selection" / "elastic_net_selection.json").read_text())
    assert selection["final_role_assignment"] == (
        "predictive_nuisances_and_orthogonal_test_modifiers"
    )
    report = selection["modifier_count_selection"]
    assert report["rule"] == "orthogonal_tests"
    assert report["chosen_estimator"] == "ensemble"
    assert report["final_estimators"] == ["causal_forest", "linear_interactions", "dina"]
    assert report["boundaries"]["nested_count_search_run"] is False
    assert not list((output / "selection" / "modifier_count").glob("*/fold_*"))

    # modifier_tests.json: schema, per-candidate fields, and the selection.
    tests = json.loads((output / "selection" / "modifier_tests.json").read_text())
    assert tests["schema_version"] == modifier_tests.SCHEMA_VERSION
    assert (tests["rule"], tests["q_threshold"]) == ("orthogonal_tests", 0.1)
    assert tests["weights"] == modifier_tests.BINARY_WEIGHTS
    assert tests["weights_source"] == "outcome_type_default"
    assert tests["propensity_clip"] == 0.02
    assert {"combined_p_value", "labels", "extreme_propensity_signal"} <= set(
        tests["interpretation"])
    by_name = {record["name"]: record for record in tests["candidates"]}
    assert set(by_name) == {"performance_status", "smoking_status"}
    for record in by_name.values():
        assert {
            "feature_id", "name", "components", "combined_p_value", "combined_q_value",
            "selected", "labels", "extreme_propensity_signal", "n_rows", "n_eligible",
            "nuisance_source", "combination_weights", "q_threshold", "excluded_by_cap",
        } <= set(record)
        assert (record["n_rows"], record["n_eligible"]) == (200, 200)
        assert record["nuisance_source"] == count.TESTING_NUISANCE_SOURCE
        for name in modifier_tests.COMPONENTS:
            component = record["components"][name]
            assert {"p_value", "q_value", "passes"} <= set(component)
            if component["p_value"] is not None:
                assert {"statistic", "degrees_of_freedom"} <= set(component)
    ecog, smoking = by_name["performance_status"], by_name["smoking_status"]
    assert ecog["selected"] and ecog["selection_rank"] == 1
    assert ecog["components"]["r_projection"]["degrees_of_freedom"] == 2
    assert "absolute" in ecog["labels"]
    assert not smoking["selected"] and smoking["labels"] == []
    assert tests["selected_feature_ids"] == [ecog["feature_id"]]

    # final_definitions.json carries compact evidence for each modifier.
    final = json.loads((output / "final_definitions.json").read_text())
    assert final["final_estimator"] == "ensemble"
    assert final["final_model_family"] == "unweighted_estimator_ensemble"
    modifiers = [f for f in final["features"] if "effect_modifier" in f["roles"]]
    assert [f["name"] for f in modifiers] == ["performance_status"]
    evidence = modifiers[0]["modifier_evidence"]
    assert evidence["rule"] == "orthogonal_tests"
    assert evidence["combined_q_value"] == ecog["combined_q_value"]
    assert set(evidence["component_p_values"]) == set(modifier_tests.COMPONENTS)

    # Testing nuisances equal the final estimation's training out-of-fold values.
    testing = pd.read_csv(report["testing_nuisances"]["predictions_path"])
    fitted = pd.read_csv(output / "estimation" / "nuisance_fit_predictions.csv")
    assert testing["_oci_row_id"].tolist() == fitted["_oci_row_id"].tolist() == split[
        "fit_row_ids"]
    for column in ("propensity", "mu0", "mu1", "effect_eligible", "inner_fold"):
        np.testing.assert_array_equal(testing[column].to_numpy(), fitted[column].to_numpy())

    # The ensemble: per-estimator columns, their unweighted mean, empty intervals.
    diagnostics = result["estimation"]
    assert diagnostics["estimator"] == "ensemble"
    assert diagnostics["estimator_policy"] == "ensemble"
    assert diagnostics["ensemble"]["fitted_estimators"] == [
        "causal_forest", "linear_interactions", "dina"]
    assert set(diagnostics["ensemble"]["component_audits"]) == {
        "causal_forest", "linear_interactions", "dina"}
    predictions = pd.read_csv(output / "estimation" / "predictions.csv")
    eligible = predictions["effect_eligible"].to_numpy(bool)
    components = predictions[[
        "estimated_cate_causal_forest", "estimated_cate_linear_interactions",
        "estimated_cate_dina",
    ]]
    np.testing.assert_allclose(
        predictions.loc[eligible, "estimated_cate"], components[eligible].mean(axis=1),
        rtol=0, atol=1e-12,
    )
    for column in ("estimated_cate_lower_95", "estimated_cate_upper_95",
                   "estimated_cate_standard_error"):
        assert predictions[column].isna().all()
    assert predictions.loc[eligible, "estimated_cate_causal_forest_lower_95"].notna().all()

    # Confirmation is post hoc: it reads frozen artifacts and changes none.
    frame = predictions.copy()
    frame.insert(1, "outer_fold", 1)
    frame.to_csv(stage2_dir / "cross_fitted_predictions.csv", index=False)
    snapshot = frame.copy()
    before = _tree_hashes(stage2_dir, exclude={"heterogeneity_diagnostics.json"})
    _, summary = write_heterogeneity_diagnostics(
        output_dir=stage2_dir, predictions=frame, propensity_clip=0.02, seed=11,
        n_bootstrap=10,
        modifier_confirmation_policy={
            "outcome_type": "binary", "q": 0.1, "weights": None, "min_count": 2,
        },
    )
    assert _tree_hashes(stage2_dir, exclude={"heterogeneity_diagnostics.json"}) == before
    pd.testing.assert_frame_equal(frame, snapshot)
    confirmation = json.loads(
        (stage2_dir / "heterogeneity_diagnostics.json").read_text())["modifier_confirmation"]
    assert confirmation["status"] == "ok"
    fold = confirmation["per_fold"]["1"]
    assert fold["selected_modifiers"] == 1 and fold["heldout_rows"] == 100
    assert [m["name"] for m in fold["modifiers"]] == ["performance_status"]
    assert summary["modifier_confirmation"]["total_selected"] == 1


def _selection_arguments(tmp_path, *, binary=False, policy=None):
    arguments = sample_inputs()
    if binary:
        arguments["dataset"]["outcome"] = (arguments["dataset"]["outcome"] > 0).astype(int)
        arguments["outcome_type"] = "binary"
    if policy is not None:
        arguments["policy"] = policy
    selected, report, _, _ = numerical.select_stage2_features_multi_model(
        **arguments, checkpoint_dir=tmp_path / "numerical")
    extras = {
        "selected": selected,
        "role_report": {"decisions": report["decisions"]},
        "statistical_report": report,
        "request_json": lambda *a, **k: pytest.fail("orthogonal tests never call an LLM"),
        "role_policy": Stage2RoleAdjudicationConfig(),
        "output_dir": tmp_path / "count",
        "numerical_checkpoint_dir": tmp_path / "numerical",
        "model_identity": "controlled-response",
        "estimation_trees": 12,
        "run_numerical": lambda a: pytest.fail("orthogonal tests skip the nested evidence"),
    }
    return arguments, extras


def test_orthogonal_selection_locks_roles_and_resumes_without_refitting(tmp_path, monkeypatch):
    policy = small_policy()
    arguments = sample_inputs()
    arguments["definitions"][-1].update(configured_explicit_feature=True,
                                        roles=["effect_modifier"])
    selected, report, _, _ = numerical.select_stage2_features_multi_model(
        **{**arguments, "policy": policy}, checkpoint_dir=tmp_path / "numerical")
    _, extras = _selection_arguments(tmp_path / "unused")
    extras.update(selected=selected, role_report={"decisions": report["decisions"]},
                  statistical_report=report, numerical_checkpoint_dir=tmp_path / "numerical",
                  output_dir=tmp_path / "count")
    kept, roles, audit = count.select_modifier_count(**arguments, **extras)
    tests = json.loads((tmp_path / "count" / "modifier_tests.json").read_text())
    records = {r["feature_id"]: r for r in tests["candidates"]}
    # The locked modifier is always included and stays outside the BH family.
    locked = records["other"]
    assert locked["investigator_locked"] and not locked["bh_family"]
    assert locked["selected"] and locked["combined_q_value"] is None
    assert locked["selection_reason"] == "investigator_locked_modifier"
    assert audit["locked_modifier_ids"] == ["other"]
    assert tests["summary"]["tested_family"] == 3
    # The simulated modifier "m" is discovered; confounder roles are preserved.
    assert "m" in audit["discovered_feature_ids"]
    assert {f["feature_id"] for f in kept if "confounder" in f["roles"]} == {
        f["feature_id"] for f in selected if "confounder" in f["roles"]}
    roles_by_id = {f["feature_id"]: f["roles"] for f in kept}
    assert "effect_modifier" in roles_by_id["other"] and "effect_modifier" in roles_by_id["m"]
    assert next(f for f in kept if f["feature_id"] == "other")["modifier_evidence"][
        "investigator_locked"] is True
    decision = next(d for d in roles["decisions"] if d["feature_id"] == "m")
    assert "Benjamini-Hochberg selection" in decision["rationale"]
    assert decision["modifier_count_selection"]["selection_rule"] == "orthogonal_tests"
    # Resume reads the checkpoint: no nuisance refit and no retesting.
    monkeypatch.setattr(analysis, "_cross_fitted_nuisance",
                        lambda **k: pytest.fail("testing nuisances refitted"))
    monkeypatch.setattr(modifier_tests, "candidate_tests",
                        lambda *a, **k: pytest.fail("modifier tests rerun"))
    assert count.select_modifier_count(**arguments, **extras) == (kept, roles, audit)
    # The selection fingerprint includes the rule's settings.
    stricter = replace(policy, multi_model=replace(
        policy.multi_model, modifier_selection=ModifierSelectionConfig(q=0.05)))
    monkeypatch.undo()
    _, other_report, _, _ = numerical.select_stage2_features_multi_model(
        **{**arguments, "policy": stricter}, checkpoint_dir=tmp_path / "numerical")
    assert other_report["input_fingerprint"] == report["input_fingerprint"]
    extras.update(statistical_report=other_report)
    _, _, stricter_audit = count.select_modifier_count(
        **{**arguments, "policy": stricter}, **extras)
    assert stricter_audit["input_fingerprint"] != audit["input_fingerprint"]


def test_estimator_policy_and_zero_discoveries(tmp_path):
    policy = small_policy()
    single = replace(policy, multi_model=replace(policy.multi_model, modifier_selection=(
        ModifierSelectionConfig(q=1e-20, estimator_policy="linear_interactions"))))
    arguments, extras = _selection_arguments(tmp_path, policy=single)
    kept, _, audit = count.select_modifier_count(**arguments, **extras)
    # Combined p-values are clamped at 1e-15, so q=1e-20 makes zero discoveries,
    # which is valid: nobody becomes a modifier.
    assert audit["discovered_feature_ids"] == [] and audit["chosen_modifier_count"] == 0
    assert not any("effect_modifier" in f["roles"] for f in kept)
    assert audit["chosen_estimator"] == "linear_interactions"
    dina = replace(policy, multi_model=replace(policy.multi_model, modifier_selection=(
        ModifierSelectionConfig(estimator_policy="dina"))))
    arguments, extras = _selection_arguments(tmp_path / "dina", policy=dina)
    with pytest.raises(ValueError, match="requires a binary outcome"):
        count.select_modifier_count(**arguments, **extras)


def test_concept_review_requires_balanced_evidence():
    with pytest.raises(ValueError, match="concept_review requires"):
        statistical_selection_config_from_mapping(
            {"multi_model": {"modifier_count": {"concept_review": True}}})
    config = statistical_selection_config_from_mapping({"multi_model": {
        "modifier_count": {"concept_review": True},
        "modifier_selection": {"rule": "balanced_evidence"},
    }})
    assert config.multi_model.modifier_count.concept_review is True
    assert config.multi_model.modifier_selection.resolved_estimator_policy == "nested_selection"
    default = statistical_selection_config_from_mapping(None).multi_model.modifier_selection
    assert (default.rule, default.q, default.weights, default.max_modifiers) == (
        "orthogonal_tests", 0.1, None, 64)
    assert default.resolved_estimator_policy == "ensemble"
    with pytest.raises(ValueError, match="unsupported modifier_selection fields"):
        statistical_selection_config_from_mapping(
            {"multi_model": {"modifier_selection": {"threshold": 0.1}}})
    with pytest.raises(ValueError, match="weights"):
        statistical_selection_config_from_mapping(
            {"multi_model": {"modifier_selection": {"weights": {"r_projection": 1.0}}}})


@pytest.mark.parametrize("binary", [True, False])
def test_ensemble_averages_components_and_continuous_outcomes_exclude_dina(tmp_path, binary):
    kw = final_inputs(binary)
    result = analysis.estimate_outer_fold(
        **kw, **final_candidates(kw), estimator="ensemble",
        statistical_policy=small_policy(), output_dir=tmp_path,
    )
    expected = ["causal_forest", "linear_interactions", *(["dina"] if binary else [])]
    assert result["ensemble"]["component_estimators"] == expected
    assert result["ensemble"]["fitted_estimators"] == expected
    assert result["model_family"] == "unweighted_estimator_ensemble"
    predictions = pd.read_csv(tmp_path / "predictions.csv")
    columns = [f"estimated_cate_{name}" for name in expected]
    assert ("estimated_cate_dina" in predictions) is binary
    assert ("dina_log_odds_ratio" in predictions) is binary
    np.testing.assert_allclose(
        predictions["estimated_cate"], predictions[columns].mean(axis=1), rtol=0, atol=1e-12)
    assert predictions["estimated_cate_lower_95"].isna().all()
    assert set(result["mean_estimated_cate_by_estimator"]) == set(expected)
    complete = json.loads((tmp_path / "complete.json").read_text())
    assert complete["outcome_model_contract"]["combination"] == (
        "unweighted_mean_of_fitted_component_cates")
    # Each component equals the same estimator fitted alone.
    alone = analysis.estimate_outer_fold(
        **kw, **final_candidates(kw), estimator="linear_interactions",
        statistical_policy=small_policy(), output_dir=tmp_path / "alone",
    )
    single = pd.read_csv(tmp_path / "alone" / "predictions.csv")
    np.testing.assert_allclose(
        predictions["estimated_cate_linear_interactions"], single["estimated_cate"])
    assert alone["mean_estimated_cate"] == pytest.approx(
        result["mean_estimated_cate_by_estimator"]["linear_interactions"])


def test_ensemble_without_modifiers_and_with_an_unestimable_component(tmp_path, monkeypatch):
    kw = final_inputs(True)
    kw["definitions"] = [{**kw["definitions"][0], "roles": ["confounder"]}]

    def unestimable(**kwargs):
        raise numerical.NotEstimable("simulated_dina_failure")

    monkeypatch.setattr(stage2_dina, "fit_effect", unestimable)
    result = analysis.estimate_outer_fold(
        **kw, **final_candidates(kw), estimator="ensemble",
        statistical_policy=small_policy(), output_dir=tmp_path,
    )
    assert result["effect_modifiers"] == 0
    assert result["ensemble"]["fitted_estimators"] == ["causal_forest", "linear_interactions"]
    assert result["ensemble"]["unavailable_estimators"]["dina"]["reason"] == (
        "simulated_dina_failure")
    predictions = pd.read_csv(tmp_path / "predictions.csv")
    assert "estimated_cate_dina" not in predictions
    np.testing.assert_allclose(
        predictions["estimated_cate"],
        predictions[["estimated_cate_causal_forest",
                     "estimated_cate_linear_interactions"]].mean(axis=1),
        rtol=0, atol=1e-12,
    )
