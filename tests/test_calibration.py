import numpy as np

from oci.utils.calibration import (
    BinaryProbabilityCalibrator,
    binary_calibration_metrics,
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
