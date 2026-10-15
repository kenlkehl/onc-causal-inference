"""P3-A items 1, 2 and 4: one recalibrated all-candidate nuisance construction everywhere,
and the prognostic score in the binary-outcome forest's X."""

import json

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from oci.inference import plain_handoff_stage2_analysis as analysis
from oci.inference import stage2_dina
from tests.test_stage2_final_modeling import inputs as final_inputs
from tests.test_stage2_multi_model import feature, small_policy
from tests.test_stage2_propensity_overlap import final_fixture


def _with_discarded_candidate(kw):
    all_fit = kw["extracted_fit"].copy()
    all_fit["discarded"] = np.arange(len(all_fit)) % 3
    all_heldout = kw["extracted_heldout"].copy()
    all_heldout["discarded"] = np.arange(len(all_heldout)) % 3
    kw["definitions"][0].update(nuisance_model_roles=["outcome"])
    return dict(
        nuisance_candidate_definitions=[feature("x"), feature("discarded")],
        nuisance_candidate_fit=all_fit,
        nuisance_candidate_heldout=all_heldout,
    )


@pytest.mark.parametrize("estimator", ["linear_interactions", "dina", "causal_forest"])
def test_training_and_heldout_nuisances_share_the_all_candidate_construction(
    tmp_path, monkeypatch, estimator
):
    kw = final_inputs()
    candidates = _with_discarded_candidate(kw)
    designs = []
    original = analysis._nuisance_prediction_futures

    def record(x_t, x_y, treatment, outcome, valid_t, valid_y, **kwargs):
        designs.append((kwargs.get("name", "full_training"), x_t.shape[1], x_y.shape[1],
                        valid_t.shape[1], len(valid_t)))
        return original(x_t, x_y, treatment, outcome, valid_t, valid_y, **kwargs)

    monkeypatch.setattr(analysis, "_nuisance_prediction_futures", record)
    captured = {}
    original_unpack = stage2_dina.unpack_nuisances

    def capture(value):
        captured.update({k: np.asarray(v, float) for k, v in value.items()})
        return original_unpack(value)

    monkeypatch.setattr(stage2_dina, "unpack_nuisances", capture)
    result = analysis.estimate_outer_fold(
        **kw, **candidates, estimator=estimator, statistical_policy=small_policy(),
        output_dir=tmp_path,
    )
    # Three inner cross-fits on the same 4-column all-candidate design. Each fold
    # model scores its 24 inner-validation rows plus the 24 outer-held-out rows;
    # there is no separate full-data fit and no refit on the (here empty)
    # selected treatment support.
    assert [d[0] for d in designs] == [f"cross_fit/fold_{k:03d}" for k in (1, 2, 3)]
    assert {d[1:] for d in designs} == {(4, 4, 4, 48)}
    routing = result["nuisance_routing"]
    assert routing["selected_task_supports_used_for_nuisances"] is False
    assert routing["heldout_candidates"] == routing["training_crossfit_candidates"] == 2
    assert routing["heldout_design"] == "average_of_the_inner_fold_all_candidate_models"
    assert routing["heldout_ensemble_models"] == 3
    assert "one construction" in result["methodology_notes"]["unified_nuisances"]
    fit = pd.read_csv(tmp_path / "nuisance_fit_predictions.csv")
    held = pd.read_csv(tmp_path / "predictions.csv")
    for frame in (fit, held):
        assert {"propensity", "mu0", "mu1", "propensity_uncalibrated", "mu0_uncalibrated",
                "mu1_uncalibrated"} <= set(frame.columns)
    assert sorted(fit.inner_fold.unique()) == [1, 2, 3]
    # Held-out raw nuisances are the probability-scale average of the three fold
    # models' held-out predictions (the last 24 entries of each fold checkpoint).
    checkpoints = next((tmp_path / "fit_checkpoints").iterdir())
    for target, column in (("propensity", "propensity_uncalibrated"),
                           ("mu0", "mu0_uncalibrated"), ("mu1", "mu1_uncalibrated")):
        fold_predictions = [
            json.loads((checkpoints / f"cross_fit/fold_{k:03d}/{target}.json").read_text())
            ["result"]["prediction"][-24:]
            for k in (1, 2, 3)
        ]
        np.testing.assert_allclose(held[column], np.mean(fold_predictions, axis=0))
    # P3-B artifacts: the exact candidate measurement frames and definitions.
    assert list(pd.read_csv(tmp_path / "nuisance_candidate_fit.csv").columns) == [
        "_oci_row_id", "x", "discarded"]
    assert len(pd.read_csv(tmp_path / "nuisance_candidate_heldout.csv")) == 24
    assert [f["name"] for f in json.loads(
        (tmp_path / "nuisance_candidate_definitions.json").read_text())["features"]] == [
        "x", "discarded"]
    # Every nuisance fit (3 inner folds x 3 models) persists its elastic-net
    # audit: standardization and refit iteration limits.
    audits = result["nuisance_fit_audit_summary"]
    assert audits["fits_audited"] == 9
    assert audits["standardized_fits"] == 9
    assert isinstance(audits["refit_iteration_limit_reached"], list)
    assert all(isinstance(f["refit_iterations"], int) for f in audits["fits"])
    calibration = result["nuisance_calibration"]
    assert calibration["predictions_used_downstream"] == "recalibrated"
    assert set(calibration["calibration_slope_summary"]) >= {
        "fit_out_of_fold_treatment", "heldout_treatment"}
    assert calibration["recalibration"]["propensity"]["method"].startswith("cross_fitted_platt")
    if estimator == "dina":
        # DINA trains and predicts with the same (recalibrated) construction.
        eligible_fit = fit.effect_eligible.to_numpy(bool)
        eligible_held = held.effect_eligible.to_numpy(bool)
        for name in ("propensity", "mu0", "mu1"):
            np.testing.assert_allclose(captured[f"training_{name}"], fit[name][eligible_fit])
            np.testing.assert_allclose(captured[f"validation_{name}"], held[name][eligible_held])
    if estimator == "linear_interactions":
        assert result["interaction_fit_audit"]["nuisance_adjustment"]["penalized"] is False
        assert "interaction_model_nuisance_adjustment" in result["methodology_notes"]
    if estimator == "causal_forest":
        assert result["prognostic_score_in_x"] is True
        assert result["constant_effect_design"] is False


def test_heldout_candidate_frame_must_cover_the_heldout_rows(tmp_path):
    kw = final_inputs()
    candidates = _with_discarded_candidate(kw)
    candidates["nuisance_candidate_heldout"] = candidates["nuisance_candidate_heldout"].iloc[1:]
    with pytest.raises(ValueError, match="do not cover exactly"):
        analysis.estimate_outer_fold(
            **kw, **candidates, estimator="linear_interactions",
            statistical_policy=small_policy(), output_dir=tmp_path,
        )


def test_cross_fitted_platt_recalibration_improves_an_overfit_propensity(
    tmp_path, monkeypatch
):
    """M-02: overfit nuisances are recalibrated using training OOF predictions only."""
    rng = np.random.default_rng(2026)
    # Five inner folds, as in the shipped configs. Held-out rows get the average
    # of the five fold models; the calibrator is learned from their OOF output.
    n_fit, n_held, p = 800, 800, 50
    n = n_fit + n_held
    x = rng.normal(size=(n, p))
    t = rng.binomial(1, expit(0.8 * x[:, 0] - 0.6 * x[:, 1] + 0.3 * x[:, 2:6].sum(1))).astype(float)
    y = rng.binomial(1, expit(-0.2 + 0.7 * x[:, 0] + 0.5 * t)).astype(float)
    frame = pd.DataFrame({"_oci_row_id": range(n), **{f"x{j}": x[:, j] for j in range(p)}})
    dataset = pd.DataFrame({"patient_id": range(n), "treatment": t, "outcome": y})
    definitions = [{**feature(f"x{j}"), "roles": ["confounder"],
                    "nuisance_model_roles": ["treatment", "outcome"]} for j in range(p)]

    def overfit_classifier(design, target, *, seed, learner=None):
        # Nearly unpenalized logistic regression: OOF slopes well below one.
        return LogisticRegression(C=1e4, max_iter=5000).fit(design, target.astype(int))

    monkeypatch.setattr(analysis, "_fit_classifier", overfit_classifier)

    class Forest:
        def __init__(self, **kwargs):
            pass

        def fit(self, x, t, y, W=None):
            pass

        def predict(self, x, **kwargs):
            return {"tau_pred": np.zeros(len(x))}

        def fit_audit(self):
            return {"outcome_model_contract": {}}

    monkeypatch.setattr(analysis, "CausalForestHead", Forest)
    split = {"fit_row_ids": list(range(n_fit)), "heldout_row_ids": list(range(n_fit, n))}
    result = analysis.estimate_outer_fold(
        dataset=dataset, extracted_fit=frame.iloc[:n_fit], extracted_heldout=frame.iloc[n_fit:],
        definitions=definitions, split=split, unit_id_column="patient_id",
        treatment_column="treatment", outcome_column="outcome", outcome_type="binary",
        inner_folds=5, seed=4, propensity_clip=0.02, estimation_trees=8, output_dir=tmp_path,
        nuisance_candidate_definitions=definitions, nuisance_candidate_fit=frame.iloc[:n_fit],
        nuisance_candidate_heldout=frame.iloc[n_fit:],
    )
    summary = result["nuisance_calibration"]["calibration_slope_summary"]
    held = summary["heldout_treatment"]
    before, after = held["before"]["calibration_slope"], held["after"]["calibration_slope"]
    assert before < 0.85
    assert abs(after - 1.0) < abs(before - 1.0)
    oof = summary["fit_out_of_fold_treatment"]
    assert oof["before"]["calibration_slope"] < 0.85
    assert abs(oof["after"]["calibration_slope"] - 1.0) < 0.15
    # Log loss also improves on the untouched held-out patients.
    calibration = result["nuisance_calibration"]
    assert (calibration["heldout_treatment"]["log_loss"]
            < calibration["before_recalibration"]["heldout_treatment"]["log_loss"])
    audit = calibration["recalibration"]
    assert audit["propensity"]["heldout"]["status"] == "fitted"
    assert set(audit["propensity"]["per_fold"]) == {"1", "2", "3", "4", "5"}
    assert "out-of-fold" in audit["description"]


def test_binary_forest_cate_varies_with_prognosis_under_a_constant_odds_ratio(tmp_path):
    """M-04: with no selected modifier, RD-scale CATEs still track baseline risk."""
    rng = np.random.default_rng(404)
    n_fit, n_held = 1200, 400
    n = n_fit + n_held
    c = rng.normal(size=n)
    noise = rng.normal(size=n)
    t = rng.binomial(1, expit(0.5 * c)).astype(float)
    baseline = -0.5 + 1.8 * c
    log_odds_ratio = 1.2  # constant on the logit scale
    y = rng.binomial(1, expit(baseline + log_odds_ratio * t)).astype(float)
    true_cate = expit(baseline + log_odds_ratio) - expit(baseline)
    frame = pd.DataFrame({"_oci_row_id": range(n), "c": c, "noise": noise})
    dataset = pd.DataFrame({"patient_id": range(n), "treatment": t, "outcome": y})
    definitions = [{**feature("c"), "roles": ["confounder"],
                    "nuisance_model_roles": ["treatment", "outcome"]}]
    candidates = [feature("c"), feature("noise")]
    split = {"fit_row_ids": list(range(n_fit)), "heldout_row_ids": list(range(n_fit, n))}
    result = analysis.estimate_outer_fold(
        dataset=dataset, extracted_fit=frame.iloc[:n_fit][["_oci_row_id", "c"]],
        extracted_heldout=frame.iloc[n_fit:][["_oci_row_id", "c"]], definitions=definitions,
        split=split, unit_id_column="patient_id", treatment_column="treatment",
        outcome_column="outcome", outcome_type="binary", inner_folds=3, seed=8,
        propensity_clip=0.02, estimation_trees=200, output_dir=tmp_path,
        nuisance_candidate_definitions=candidates, nuisance_candidate_fit=frame.iloc[:n_fit],
        nuisance_candidate_heldout=frame.iloc[n_fit:],
    )
    assert result["effect_modifiers"] == 0
    assert result["prognostic_score_in_x"] is True
    assert result["constant_effect_design"] is False
    assert result["selected_modifier_columns_in_x"] == 0
    assert "risk-difference" in result["prognostic_score"]["description"]
    cate = pd.read_csv(tmp_path / "predictions.csv").estimated_cate.to_numpy()
    # Before M-04 the forest's X was a constant column: one CATE for everyone.
    assert np.std(cate) > 0.02
    assert np.corrcoef(cate, true_cate[n_fit:])[0, 1] > 0.5


def test_continuous_outcome_forest_keeps_constant_design_without_modifiers(
    monkeypatch, tmp_path
):
    kwargs, calls = final_fixture(monkeypatch, tmp_path)
    kwargs["dataset"] = kwargs["dataset"].assign(outcome=np.linspace(0, 1, 24))
    designs = []

    class Forest:
        def __init__(self, **kw):
            pass

        def fit(self, x, t, y, W=None):
            designs.append(x.copy())

        def predict(self, x, **kw):
            return {"tau_pred": np.full(len(x), 0.4)}

        def fit_audit(self):
            return {"outcome_model_contract": {}}

    monkeypatch.setattr(analysis, "CausalForestHead", Forest)
    definitions = [{**kwargs["definitions"][0], "roles": ["confounder"]}]
    result = analysis.estimate_outer_fold(
        **{**kwargs, "definitions": definitions, "outcome_type": "continuous"})
    assert result["prognostic_score_in_x"] is False
    assert result["constant_effect_design"] is True
    assert designs and np.ptp(designs[0]) == 0
