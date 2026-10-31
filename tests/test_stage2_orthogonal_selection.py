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
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oci.inference import plain_handoff_stage2_analysis as analysis
from oci.inference import stage2_modifier_count as count
from oci.inference import stage2_modifier_tests as modifier_tests
from oci.inference.plain_handoff_stage2 import (
    PlainHandoffStage2Config,
    Stage2ExtractionLLMConfig,
)
from oci.inference.stage2_diagnostics import write_heterogeneity_diagnostics
from tests.stage2_prompt_spy import install, prompt_inputs
from tests.test_stage2_final_modeling import candidates as final_candidates
from tests.test_stage2_final_modeling import inputs as final_inputs
from tests.test_stage2_multi_model import small_policy


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


def _run_fold(output, *, estimation_trees=12, nuisance_learner="super_learner"):
    dataset, definitions, split = _fold_inputs()
    return analysis.run_fold_analysis(
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
            estimation_trees=estimation_trees,
            nuisance_learner=nuisance_learner,
            min_informative_patients=2,
            min_propensity=0.05,
            max_propensity=0.95,
            statistical_selection=small_policy(),
        ),
    )


def _audits(directory: Path) -> dict[str, dict]:
    return {
        str(path.relative_to(directory)): json.loads(path.read_text())
        for path in sorted(directory.rglob("*.json"))
    }


def test_run_fold_analysis_selects_modifiers_by_orthogonal_tests_and_ensembles(
    tmp_path, monkeypatch
):
    install(monkeypatch)
    _, _, split = _fold_inputs()
    nested = []
    monkeypatch.setattr(count, "_score_prefix", lambda **k: nested.append(k))
    stage2_dir = tmp_path / "stage2"
    output = stage2_dir / "outer_001"
    result = _run_fold(output, nuisance_learner="elastic_net")
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
    # Training-row p-values are screening values; the held-out confirmation is inferential.
    assert tests["p_value_role"] == modifier_tests.SCREENING_P_VALUE_ROLE
    assert tests["p_value_role"].startswith("screening")
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
    testing = pd.read_csv(report["testing_nuisances"]["predictions_path"],
                          float_precision="round_trip")
    fitted = pd.read_csv(output / "estimation" / "nuisance_fit_predictions.csv",
                         float_precision="round_trip")
    assert testing["_oci_row_id"].tolist() == fitted["_oci_row_id"].tolist() == split[
        "fit_row_ids"]
    for column in ("propensity", "mu0", "mu1", "effect_eligible", "inner_fold"):
        np.testing.assert_array_equal(testing[column].to_numpy(), fitted[column].to_numpy())
    # Both are the raw out-of-fold predictions (training rows are not recalibrated).
    for column in ("propensity", "mu0", "mu1"):
        np.testing.assert_array_equal(fitted[column], fitted[f"{column}_uncalibrated"])
    # Final estimation scored the held-out rows with the stored selection-time
    # inner-fold models (2 folds x e, mu0, mu1) instead of refitting them, and
    # their fit audits are identical.
    store = result["estimation"]["nuisance_model_store"]
    assert store["status"] == "available"
    assert store["fingerprint"] == report["testing_nuisances"]["model_store"]["fingerprint"]
    assert result["estimation"]["nuisance_fit_audit_summary"]["model_sources"] == {
        "reused_stored_model": 6}
    selection_audits = _audits(Path(report["checkpoint_dir"]) / "testing_nuisance_fit_audits")
    estimation_audits = _audits(next((output / "estimation" / "nuisance_fit_audits").iterdir()))
    assert set(selection_audits) == set(estimation_audits) and len(selection_audits) == 6
    for name, audit in selection_audits.items():
        assert audit["model_source"] == "fitted"
        assert estimation_audits[name]["model_source"] == "reused_stored_model"
        for key in ("rows", "columns", "fit_audit"):
            assert estimation_audits[name][key] == audit[key], (name, key)

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


def _stored_nuisances(kw, store, **overrides):
    """The modifier tests' training-only call, which stores the inner-fold models."""
    fit_ids = kw["split"]["fit_row_ids"]
    candidate_fit = analysis._aligned_measurement_frame(
        kw["extracted_fit"], fit_ids, kw["definitions"], label="training")
    return analysis._outer_training_nuisances(**{
        "dataset": kw["dataset"],
        "candidate_fit": candidate_fit,
        "candidate_definitions": kw["definitions"],
        "fit_ids": fit_ids,
        "inner_splits": kw["split"]["inner_splits"],
        "treatment_column": "treatment",
        "outcome_column": "outcome",
        "binary": kw["outcome_type"] == "binary",
        "seed": kw["seed"],
        "checkpoint_dir": store.parent / "testing_checkpoints",
        "checkpoint_fingerprint": "testing",
        "audit_dir": store.parent / "testing_audits",
        "model_store_root": store,
        "model_store_mode": "save",
        **overrides,
    })


@pytest.mark.parametrize("binary", [True])
def test_final_estimation_reuses_stored_nuisance_models_without_refitting(
    tmp_path, monkeypatch, binary
):
    kw = final_inputs(binary)
    store = tmp_path / "selection" / "nuisance_models"
    saved = _stored_nuisances(kw, store)
    assert saved["model_store"]["status"] == "saved"
    root = Path(saved["model_store"]["root"])
    manifest = json.loads((root / analysis.NUISANCE_MODEL_MANIFEST).read_text())
    assert len(manifest["models"]) == 9  # 3 inner folds x (e, mu0, mu1)
    assert manifest["identity"]["libraries"]["scikit-learn"]

    refit = analysis.estimate_outer_fold(
        **kw, **final_candidates(kw), estimator="linear_interactions",
        statistical_policy=small_policy(), output_dir=tmp_path / "refit",
    )
    assert "nuisance_model_store" not in refit

    def no_refit(*args, **kwargs):
        raise AssertionError("a stored nuisance model was refitted")

    with monkeypatch.context() as patch:
        patch.setattr(analysis, "_fit_classifier", no_refit)
        patch.setattr(analysis, "_fit_regressor", no_refit)
        reused = analysis.estimate_outer_fold(
            **kw, **final_candidates(kw), estimator="linear_interactions",
            statistical_policy=small_policy(), output_dir=tmp_path / "reused",
            nuisance_model_store=store,
        )
    assert reused["nuisance_model_store"]["status"] == "available"
    assert reused["nuisance_fit_audit_summary"]["model_sources"] == {"reused_stored_model": 9}
    for name in ("predictions.csv", "nuisance_fit_predictions.csv"):
        pd.testing.assert_frame_equal(
            pd.read_csv(tmp_path / "reused" / name), pd.read_csv(tmp_path / "refit" / name))
    # Training predictions equal the stored models' own training out-of-fold values.
    fitted = pd.read_csv(tmp_path / "reused" / "nuisance_fit_predictions.csv",
                         float_precision="round_trip")
    for column, values in zip(("propensity", "mu0", "mu1"), saved["recalibrated"]["fit"]):
        np.testing.assert_array_equal(fitted[column], values)
    refit_audits = _audits(next((tmp_path / "refit" / "nuisance_fit_audits").iterdir()))
    reused_audits = _audits(next((tmp_path / "reused" / "nuisance_fit_audits").iterdir()))
    stored_audits = _audits(tmp_path / "selection" / "testing_audits")
    assert set(refit_audits) == set(reused_audits) == set(stored_audits)
    for name, audit in reused_audits.items():
        assert audit["fit_audit"] == refit_audits[name]["fit_audit"] == (
            stored_audits[name]["fit_audit"])

    # Fail closed: a corrupted model file is refitted; the others are reused.
    corrupted = next(iter(sorted(root.rglob("*.joblib"))))
    corrupted.write_bytes(corrupted.read_bytes() + b"tampered")
    partial = analysis.estimate_outer_fold(
        **kw, **final_candidates(kw), estimator="linear_interactions",
        statistical_policy=small_policy(), output_dir=tmp_path / "partial",
        nuisance_model_store=store,
    )
    assert partial["nuisance_fit_audit_summary"]["model_sources"] == {
        "fitted": 1, "reused_stored_model": 8}
    pd.testing.assert_frame_equal(
        pd.read_csv(tmp_path / "partial" / "predictions.csv"),
        pd.read_csv(tmp_path / "refit" / "predictions.csv"))

    # Changed inputs give another fingerprint: nothing is loaded.
    changed = {**kw, "seed": kw["seed"] + 1}
    absent = analysis.estimate_outer_fold(
        **changed, **final_candidates(kw), estimator="linear_interactions",
        statistical_policy=small_policy(), output_dir=tmp_path / "changed",
        nuisance_model_store=store,
    )
    assert absent["nuisance_model_store"]["status"] == "absent"
    assert "model_sources" not in absent["nuisance_fit_audit_summary"]
