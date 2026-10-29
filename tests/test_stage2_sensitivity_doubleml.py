"""Cross-check of the omitted-variable-bias sensitivity analysis against DoubleML.

DoubleML (Bach, Chernozhukov, Kurz and Spindler, JMLR 2022) implements the
bounds of Chernozhukov, Cinelli, Newey, Sharma and Syrgkanis ("Long story
short"). Given the same nuisance predictions, its interactive regression model
must reproduce OCI's estimate, bounds and robustness value. The module is
skipped when DoubleML is not installed; it is a test-only reference, not a
dependency.

DoubleML 0.11 forms the lower bound's confidence bound from the adjusted score
that the delta method assigns to the upper bound (its scaled score is the
negative influence function for this model), so the bound standard errors are
compared as a pair.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from scipy.stats import norm

from oci.inference import stage2_sensitivity as sensitivity

dml = pytest.importorskip("doubleml")
dml_utils = pytest.importorskip("doubleml.utils")

CLIP = 0.01
CF_Y, CF_D = 0.03, 0.05


def _cohort(seed: int = 0, n: int = 3000, *, binary: bool):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    e = expit(0.8 * x[:, 0] - 0.5 * x[:, 1])
    t = rng.binomial(1, e).astype(float)
    if binary:
        mu0 = expit(-0.5 + 0.7 * x[:, 0] + 0.4 * x[:, 2])
        mu1 = expit(0.1 + 0.7 * x[:, 0] + 0.4 * x[:, 2] + 0.3 * x[:, 1])
        y = rng.binomial(1, np.where(t == 1, mu1, mu0)).astype(float)
    else:
        mu0 = 0.7 * x[:, 0] + 0.4 * x[:, 2]
        mu1 = 0.5 + mu0 + 0.3 * x[:, 1]
        y = np.where(t == 1, mu1, mu0) + rng.normal(size=n)
    # Noisy "fitted" nuisances, as a cross-fitted learner would return them.
    e_hat = np.clip(expit(np.log(e / (1 - e)) + rng.normal(0, 0.2, n)), 0.02, 0.98)
    noise0, noise1 = rng.normal(0, 0.1, n), rng.normal(0, 0.1, n)
    mu0_hat = expit(np.log(mu0 / (1 - mu0)) + noise0) if binary else mu0 + noise0
    mu1_hat = expit(np.log(mu1 / (1 - mu1)) + noise1) if binary else mu1 + noise1
    return x, y, t, e_hat, mu0_hat, mu1_hat


def _irm(x, y, t, e, mu0, mu1, **kwargs):
    frame = pd.DataFrame(x, columns=[f"x{i}" for i in range(x.shape[1])])
    frame["y"], frame["d"] = y, t
    data = dml.DoubleMLData(frame, y_col="y", d_cols="d", x_cols=list(frame.columns[:-2]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = dml.DoubleMLIRM(
            data,
            ml_g=dml_utils.DMLDummyRegressor(),
            ml_m=dml_utils.DMLDummyClassifier(),
            n_folds=2,
            score="ATE",
            trimming_threshold=CLIP,
            **kwargs,
        )
        model.fit(
            external_predictions={
                "d": {"ml_g0": mu0[:, None], "ml_g1": mu1[:, None], "ml_m": e[:, None]}
            }
        )
    return model


def _scalar(value) -> float:
    return float(np.squeeze(np.asarray(value)))


@pytest.mark.parametrize("binary", [True])
def test_ate_bounds_and_robustness_value_match_doubleml(binary):
    x, y, t, e, mu0, mu1 = _cohort(binary=binary)
    n = len(y)
    model = _irm(x, y, t, e, mu0, mu1)
    # A one-sided level of Phi(1.96) gives DoubleML OCI's two-sided 95% critical value.
    level = float(norm.cdf(sensitivity.CONFIDENCE_Z))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.sensitivity_analysis(cf_y=CF_Y, cf_d=CF_D, rho=1.0, level=level)
    params = model.sensitivity_params
    elements = model.sensitivity_elements

    ours = sensitivity.ate_elements(y, t, e, mu0, mu1, clip=CLIP)
    bounds = sensitivity.ovb_bounds(ours, CF_Y, CF_D)
    robustness = sensitivity.robustness_values(ours)

    assert ours.theta == pytest.approx(_scalar(model.coef), rel=1e-10)
    # OCI's variances divide by n - 1, DoubleML's by n.
    assert ours.standard_error == pytest.approx(
        _scalar(model.se) * math.sqrt(n / (n - 1)), rel=1e-8)
    assert ours.sigma2 == pytest.approx(_scalar(elements["sigma2"]), rel=1e-10)
    assert ours.nu2 == pytest.approx(_scalar(elements["nu2"]), rel=1e-10)
    assert bounds["lower"] == pytest.approx(_scalar(params["theta"]["lower"]), rel=1e-9)
    assert bounds["upper"] == pytest.approx(_scalar(params["theta"]["upper"]), rel=1e-9)
    assert robustness["robustness_value"] == pytest.approx(_scalar(params["rv"]), abs=1e-6)

    ours_se = sorted([bounds["lower_standard_error"], bounds["upper_standard_error"]])
    theirs_se = sorted(
        _scalar(params["se"][side]) * math.sqrt(n / (n - 1)) for side in ("lower", "upper")
    )
    assert ours_se == pytest.approx(theirs_se, rel=1e-8)
