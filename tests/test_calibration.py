import json

import numpy as np
import pytest
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression

from oci.inference.nuisance_diagnostics import calibration_diagnostics
from oci.utils.calibration import (
    PlattRecalibration,
)


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
