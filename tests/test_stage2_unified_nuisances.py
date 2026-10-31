"""P3-A items 1, 2 and 4: one recalibrated all-candidate nuisance construction everywhere,
and the prognostic score in the binary-outcome forest's X."""

import json

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from oci.inference import plain_handoff_stage2_analysis as analysis
from tests.test_stage2_multi_model import feature, small_policy


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


def _calibration_fixture(n_fit=240, n_held=60, seed=11):
    rng = np.random.default_rng(seed)
    n = n_fit + n_held
    x = rng.normal(size=(n, 3))
    t = rng.binomial(1, expit(0.9 * x[:, 0])).astype(float)
    y = rng.binomial(1, expit(-0.3 + 0.8 * x[:, 1] + 0.6 * t)).astype(float)
    frame = pd.DataFrame({"_oci_row_id": range(n), **{f"x{j}": x[:, j] for j in range(3)}})
    definitions = [{**feature(f"x{j}"), "roles": ["confounder"],
                    "nuisance_model_roles": ["treatment", "outcome"]} for j in range(3)]
    split = {"fit_row_ids": list(range(n_fit)), "heldout_row_ids": list(range(n_fit, n))}
    split["inner_splits"] = analysis._fallback_inner_splits(range(n_fit), folds=3, seed=2)
    return dict(
        dataset=pd.DataFrame({"patient_id": range(n), "treatment": t, "outcome": y}),
        extracted_fit=frame.iloc[:n_fit], extracted_heldout=frame.iloc[n_fit:],
        definitions=definitions, split=split, unit_id_column="patient_id",
        treatment_column="treatment", outcome_column="outcome", outcome_type="binary",
        inner_folds=3, seed=4, propensity_clip=0.02, estimation_trees=8,
        nuisance_candidate_definitions=definitions, nuisance_candidate_fit=frame.iloc[:n_fit],
        nuisance_candidate_heldout=frame.iloc[n_fit:],
    )


def test_training_nuisances_are_raw_and_heldout_averages_own_fold_calibrations(tmp_path):
    from oci.utils.calibration import PlattRecalibration

    kw = _calibration_fixture()
    result = analysis.estimate_outer_fold(
        **kw, estimator="linear_interactions", statistical_policy=small_policy(),
        output_dir=tmp_path)
    fit = pd.read_csv(tmp_path / "nuisance_fit_predictions.csv", float_precision="round_trip")
    held = pd.read_csv(tmp_path / "predictions.csv", float_precision="round_trip")
    # Training out-of-fold nuisances are the raw fold-model predictions.
    for name in ("propensity", "mu0", "mu1"):
        np.testing.assert_array_equal(fit[name], fit[f"{name}_uncalibrated"])
    # Held-out values: each fold model's held-out predictions calibrated by a Platt
    # fit on that model's own validation rows (arm rows for mu), then averaged.
    checkpoints = next((tmp_path / "fit_checkpoints").iterdir())
    dataset, n_held = kw["dataset"], len(kw["split"]["heldout_row_ids"])
    audit = result["nuisance_calibration"]["recalibration"]
    fitted = 0
    for name, arm in (("propensity", None), ("mu0", 0), ("mu1", 1)):
        expected = []
        for k, inner in enumerate(kw["split"]["inner_splits"], 1):
            prediction = np.asarray(json.loads(
                (checkpoints / f"cross_fit/fold_{k:03d}/{name}.json").read_text()
            )["result"]["prediction"])
            rows = np.asarray(inner["heldout_row_ids"])
            own, heldout = prediction[:len(rows)], prediction[len(rows):]
            assert len(heldout) == n_held
            t = dataset.loc[rows, "treatment"].to_numpy()
            label = t if arm is None else dataset.loc[rows, "outcome"].to_numpy()
            keep = np.ones(len(rows), bool) if arm is None else t == arm
            calibrator = PlattRecalibration.fit(own[keep], label[keep])
            recorded = audit[name]["per_fold"][str(k)]
            assert recorded["fit_rows"] == int(keep.sum())
            assert recorded["status"] == calibrator.status
            assert recorded["slope"] == pytest.approx(calibrator.slope)
            fitted += calibrator.status == "fitted"
            expected.append(calibrator.transform(heldout))
        np.testing.assert_allclose(held[name], np.mean(expected, axis=0), rtol=0, atol=1e-12)
        assert audit[name]["method"] == "own_fold_platt_then_average"
    assert fitted >= 3  # at least the propensity calibrators are estimated
    summary = result["nuisance_calibration"]["calibration_slope_summary"]
    assert summary["fit_out_of_fold_treatment"]["before"] == (
        summary["fit_out_of_fold_treatment"]["after"])
