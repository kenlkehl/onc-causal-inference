import json

import numpy as np
import pytest
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from oci.inference.nuisance_diagnostics import calibration_diagnostics
from oci.utils.calibration import (
    BinaryProbabilityCalibrator,
    PlattRecalibration,
    binary_calibration_metrics,
    cross_fitted_platt,
)


def test_binary_calibration_metrics_reports_ece_and_brier():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])

    metrics = binary_calibration_metrics(y, p, prefix="propensity", n_bins=2)

    assert metrics["propensity_brier"] is not None
    assert metrics["propensity_log_loss"] is not None
    assert metrics["propensity_ece"] is not None
    assert metrics["propensity_prob_mean"] == 0.5


def test_temperature_isotonic_calibrator_shrinks_overconfident_probabilities():
    y = np.array([0] * 30 + [1] * 30)
    p = np.array([0.02] * 15 + [0.30] * 15 + [0.70] * 15 + [0.98] * 15)

    calibrator = BinaryProbabilityCalibrator.fit(p, y, method="temperature_isotonic")
    calibrated = calibrator.transform(p)

    assert calibrated.shape == p.shape
    assert np.all((calibrated > 0.0) & (calibrated < 1.0))
    assert calibrator.temperature > 0.0


# ---------------------------------------------------------------------------
# Platt recalibration and cross-fitting (M-02)
# ---------------------------------------------------------------------------


def _miscalibrated_sample(rng, n):
    true_logit = rng.normal(-0.4, 1.3, size=n)
    target = rng.binomial(1, expit(true_logit))
    # Overconfident and shifted: logit(p_true) = 0.3 + 0.6 * logit(observed).
    observed = expit((true_logit - 0.3) / 0.6)
    return observed, target


def test_platt_recovers_known_miscalibration():
    rng = np.random.RandomState(0)
    observed, target = _miscalibrated_sample(rng, 20_000)
    test_observed, test_target = _miscalibrated_sample(rng, 20_000)

    calibrator = PlattRecalibration.fit(observed, target)
    before = calibration_diagnostics(test_target, test_observed, binary=True)
    after = calibration_diagnostics(test_target, calibrator.transform(test_observed), binary=True)

    assert calibrator.status == "fitted"
    assert calibrator.rows == 20_000
    assert calibrator.slope == pytest.approx(0.6, abs=0.04)
    assert calibrator.intercept == pytest.approx(0.3, abs=0.05)
    assert before["calibration_slope"] == pytest.approx(0.6, abs=0.04)
    assert after["calibration_slope"] == pytest.approx(1.0, abs=0.05)
    assert after["calibration_intercept"] == pytest.approx(0.0, abs=0.05)
    assert after["log_loss"] < before["log_loss"]
    assert calibrator.public_dict() == {
        "method": "platt_logistic_on_logit",
        "intercept": calibrator.intercept,
        "slope": calibrator.slope,
        "status": "fitted",
        "rows": 20_000,
    }
    json.dumps(calibrator.public_dict())


def test_platt_fit_matches_unpenalized_logistic_regression():
    rng = np.random.RandomState(1)
    observed, target = _miscalibrated_sample(rng, 2_000)
    reference = LogisticRegression(penalty=None, tol=1e-12, max_iter=10_000).fit(
        logit(np.clip(observed, 1e-6, 1 - 1e-6)).reshape(-1, 1), target
    )

    calibrator = PlattRecalibration.fit(observed, target)

    assert calibrator.intercept == pytest.approx(reference.intercept_[0], abs=1e-6)
    assert calibrator.slope == pytest.approx(reference.coef_[0, 0], abs=1e-6)
    np.testing.assert_allclose(
        calibrator.transform(observed),
        np.clip(
            reference.predict_proba(logit(np.clip(observed, 1e-6, 1 - 1e-6)).reshape(-1, 1))[:, 1],
            1e-6,
            1 - 1e-6,
        ),
        atol=1e-6,
    )


def test_cross_fitted_platt_reduces_overfit_miscalibration_on_heldout_rows():
    rng = np.random.RandomState(5)
    rows, features, folds = 600, 40, 10

    def sample(n):
        design = rng.normal(size=(n, features))
        target = rng.binomial(1, expit(design[:, 0] - 0.8 * design[:, 1]))
        return design, target

    train_x, train_y = sample(rows)
    test_x, test_y = sample(5_000)

    def overfit_model():
        # Nearly unpenalized with 38 pure-noise features.
        return LogisticRegression(C=1e4, max_iter=10_000)

    fold_ids = np.empty(rows, dtype=int)
    oof = np.empty(rows, dtype=float)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=5)
    for fold, (fit_rows, predict_rows) in enumerate(splitter.split(train_x, train_y)):
        model = overfit_model().fit(train_x[fit_rows], train_y[fit_rows])
        oof[predict_rows] = model.predict_proba(train_x[predict_rows])[:, 1]
        fold_ids[predict_rows] = fold
    test_prediction = overfit_model().fit(train_x, train_y).predict_proba(test_x)[:, 1]

    calibrated_oof, calibrated_test, audit = cross_fitted_platt(
        oof, train_y, fold_ids, heldout_prob=test_prediction
    )

    slope_before = calibration_diagnostics(test_y, test_prediction, binary=True)[
        "calibration_slope"
    ]
    slope_after = calibration_diagnostics(test_y, calibrated_test, binary=True)["calibration_slope"]
    oof_before = calibration_diagnostics(train_y, oof, binary=True)["calibration_slope"]
    oof_after = calibration_diagnostics(train_y, calibrated_oof, binary=True)["calibration_slope"]
    assert slope_before < 0.85
    assert abs(slope_after - 1.0) < abs(slope_before - 1.0)
    assert abs(oof_after - 1.0) < abs(oof_before - 1.0)
    assert audit["method"] == "cross_fitted_platt_logistic_on_logit"
    assert audit["fit_rows"] == rows
    assert set(audit["per_fold"]) == {str(fold) for fold in range(folds)}
    assert all(item["status"] == "fitted" for item in audit["per_fold"].values())
    assert all(
        item["rows"] == rows - int(np.sum(fold_ids == int(fold)))
        for fold, item in audit["per_fold"].items()
    )
    assert audit["heldout"]["status"] == "fitted"
    assert audit["heldout"]["rows"] == rows
    assert audit["heldout"]["slope"] < 1.0
    json.dumps(audit)


def _random_calibration_problem(seed=7, rows=300, folds=5):
    rng = np.random.RandomState(seed)
    true_logit = rng.normal(size=rows)
    target = rng.binomial(1, expit(true_logit)).astype(float)
    prob = expit(1.8 * true_logit + rng.normal(scale=0.3, size=rows))
    fold_ids = np.arange(rows) % folds
    rng.shuffle(fold_ids)
    return prob, target, fold_ids


def test_cross_fitted_platt_is_fold_honest():
    prob, target, fold_ids = _random_calibration_problem()
    baseline, _, baseline_audit = cross_fitted_platt(prob, target, fold_ids)

    perturbed_target = target.copy()
    in_fold = fold_ids == 2
    perturbed_target[in_fold] = 1.0 - perturbed_target[in_fold]
    perturbed, _, perturbed_audit = cross_fitted_platt(prob, perturbed_target, fold_ids)

    np.testing.assert_array_equal(perturbed[in_fold], baseline[in_fold])
    assert perturbed_audit["per_fold"]["2"] == baseline_audit["per_fold"]["2"]
    # Every other fold's calibrator used fold 2's targets, so it moves.
    assert not np.allclose(perturbed[~in_fold], baseline[~in_fold])
    for fold in (0, 1, 3, 4):
        assert perturbed_audit["per_fold"][str(fold)] != baseline_audit["per_fold"][str(fold)]

    # Each fold's output is exactly the calibrator fit on the other folds.
    for fold in range(5):
        rows = fold_ids == fold
        expected = PlattRecalibration.fit(prob[~rows], target[~rows]).transform(prob[rows])
        np.testing.assert_array_equal(baseline[rows], expected)


def test_cross_fitted_platt_fit_mask_controls_fitting_but_not_application():
    prob, target, fold_ids = _random_calibration_problem(seed=11, rows=400)
    rng = np.random.RandomState(3)
    treatment = rng.binomial(1, 0.5, size=len(prob)).astype(bool)
    control = ~treatment
    heldout = np.linspace(0.05, 0.95, 7)
    # Treated outcomes are irrelevant to a control-arm calibrator.
    masked_target = np.where(control, target, np.nan)

    calibrated, calibrated_heldout, audit = cross_fitted_platt(
        prob, masked_target, fold_ids, heldout_prob=heldout, fit_mask=control
    )
    flipped_treated = np.where(control, target, 1.0 - target)
    same, same_heldout, _ = cross_fitted_platt(
        prob, flipped_treated, fold_ids, heldout_prob=heldout, fit_mask=control
    )

    assert np.isfinite(calibrated).all()
    assert audit["fit_rows"] == int(control.sum())
    for fold in range(5):
        rows = fold_ids == fold
        reference = PlattRecalibration.fit(prob[control & ~rows], target[control & ~rows])
        assert reference.status == "fitted"
        # Applied to every row of the fold, treated rows included.
        np.testing.assert_array_equal(calibrated[rows], reference.transform(prob[rows]))
        assert audit["per_fold"][str(fold)] == reference.public_dict()
        assert audit["per_fold"][str(fold)]["rows"] == int(np.sum(control & ~rows))
    full = PlattRecalibration.fit(prob[control], target[control])
    np.testing.assert_array_equal(calibrated_heldout, full.transform(heldout))
    assert audit["heldout"] == full.public_dict()
    np.testing.assert_array_equal(same, calibrated)
    np.testing.assert_array_equal(same_heldout, calibrated_heldout)


def test_cross_fitted_platt_without_heldout_and_with_string_folds():
    prob, target, fold_ids = _random_calibration_problem(seed=13)
    labels = np.asarray([f"fold_{fold}" for fold in fold_ids])

    numeric, numeric_heldout, numeric_audit = cross_fitted_platt(prob, target, fold_ids)
    named, _, named_audit = cross_fitted_platt(prob, target, labels)

    assert numeric_heldout is None
    assert numeric_audit["heldout"] is None
    np.testing.assert_array_equal(named, numeric)
    assert set(named_audit["per_fold"]) == {f"fold_{fold}" for fold in range(5)}


def test_platt_identity_fallbacks():
    rng = np.random.RandomState(2)
    prob = rng.uniform(0.05, 0.95, size=60)
    target = rng.binomial(1, prob).astype(float)

    too_few = PlattRecalibration.fit(prob[:10], target[:10], min_rows=20)
    single_class = PlattRecalibration.fit(prob, np.ones_like(prob))
    order = np.argsort(prob)
    separated_target = np.zeros_like(prob)
    separated_target[order[30:]] = 1.0
    separated = PlattRecalibration.fit(prob, separated_target)
    quasi_target = separated_target.copy()
    tied = prob.copy()
    tied[order[29]] = tied[order[30]]  # one tie at the class boundary
    quasi = PlattRecalibration.fit(tied, quasi_target)
    reversed_separation = PlattRecalibration.fit(prob, 1.0 - separated_target)
    no_finite = PlattRecalibration.fit(np.full(30, np.nan), np.ones(30))
    constant = PlattRecalibration.fit(np.full(60, 0.3), target)
    # Ranking anti-correlated with the outcome would need a negative slope.
    inverted = PlattRecalibration.fit(prob, rng.binomial(1, 1.0 - prob).astype(float))

    assert too_few.status == "identity_too_few_rows" and too_few.rows == 10
    assert single_class.status == "identity_single_class"
    assert separated.status == "identity_separated"
    assert quasi.status == "identity_separated"
    assert reversed_separation.status == "identity_separated"
    assert no_finite.status == "identity_no_finite_rows" and no_finite.rows == 0
    assert constant.status == "identity_constant_predictions"
    assert inverted.status == "identity_nonpositive_slope"
    assert PlattRecalibration().status == "identity_not_fitted"
    for calibrator in (
        too_few,
        single_class,
        separated,
        quasi,
        reversed_separation,
        no_finite,
        constant,
        inverted,
        PlattRecalibration(),
    ):
        assert (calibrator.intercept, calibrator.slope) == (0.0, 1.0)
        np.testing.assert_array_equal(
            calibrator.transform(np.array([0.0, 0.25, 1.0])),
            np.array([1e-6, 0.25, 1 - 1e-6]),
        )
        json.dumps(calibrator.public_dict())


def test_platt_excludes_nonfinite_rows_from_fitting():
    rng = np.random.RandomState(4)
    observed, target = _miscalibrated_sample(rng, 500)
    target = target.astype(float)
    noisy_prob = observed.copy()
    noisy_target = target.copy()
    noisy_prob[:20] = np.nan
    noisy_target[20:40] = np.nan

    calibrator = PlattRecalibration.fit(noisy_prob, noisy_target)
    reference = PlattRecalibration.fit(observed[40:], target[40:])

    assert calibrator == reference
    assert calibrator.rows == 460


def test_platt_validation_errors():
    calibrator = PlattRecalibration(intercept=0.1, slope=0.8, status="fitted", rows=50)
    with pytest.raises(ValueError, match="finite"):
        calibrator.transform(np.array([0.2, np.nan]))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        calibrator.transform(np.array([0.2, 1.2]))
    with pytest.raises(ValueError, match="slope must be positive"):
        PlattRecalibration(intercept=0.0, slope=-1.0, status="fitted")
    with pytest.raises(ValueError, match="identity_"):
        PlattRecalibration(status="unknown")
    with pytest.raises(ValueError, match="same length"):
        PlattRecalibration.fit(np.full(5, 0.5), np.ones(4))
    with pytest.raises(ValueError, match="only 0 and 1"):
        PlattRecalibration.fit(np.full(30, 0.5), np.full(30, 2.0))
    with pytest.raises(ValueError, match="one-dimensional"):
        PlattRecalibration.fit(np.full((30, 1), 0.5), np.ones((30, 1)))
    with pytest.raises(TypeError, match="min_rows"):
        PlattRecalibration.fit(np.full(30, 0.5), np.ones(30), min_rows=2.5)

    prob, target, fold_ids = _random_calibration_problem()
    with pytest.raises(ValueError, match="same length"):
        cross_fitted_platt(prob, target, fold_ids[:-1])
    with pytest.raises(TypeError, match="boolean"):
        cross_fitted_platt(prob, target, fold_ids, fit_mask=np.ones(len(prob)))
    with pytest.raises(ValueError, match="same length"):
        cross_fitted_platt(prob, target, fold_ids, fit_mask=np.ones(3, dtype=bool))
    with pytest.raises(ValueError, match="finite"):
        cross_fitted_platt(np.where(fold_ids == 0, np.nan, prob), target, fold_ids)
    with pytest.raises(ValueError, match="finite"):
        cross_fitted_platt(prob, target, fold_ids, heldout_prob=np.array([0.3, np.inf]))
    with pytest.raises(ValueError, match="fold_ids must be finite"):
        cross_fitted_platt(prob, target, np.where(fold_ids == 0, np.nan, fold_ids))
