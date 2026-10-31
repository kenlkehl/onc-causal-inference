"""Reporting-only unmeasured-confounding sensitivity: E-values, OVB bounds, benchmarks."""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from oci.inference import stage2_diagnostics as diagnostics
from oci.inference import stage2_sensitivity as sensitivity

Z = sensitivity.CONFIDENCE_Z


# ---------------------------------------------------------------------------
# E-values


def test_evalue_matches_the_published_examples():
    # VanderWeele and Ding (2017): RR = 3.9 gives 7.26; RR = 1.06 gives 1.31.
    assert sensitivity.evalue(3.9) == pytest.approx(7.26, abs=0.005)
    assert sensitivity.evalue(1.06) == pytest.approx(1.31, abs=0.005)
    assert sensitivity.evalue(1.0) == 1.0
    # RR < 1 uses 1/RR.
    assert sensitivity.evalue(1 / 3.9) == pytest.approx(sensitivity.evalue(3.9))
    assert sensitivity.evalue(0.0) is None and sensitivity.evalue(float("nan")) is None


def _binary_population(rng, n):
    x = rng.normal(size=n)
    e = expit(0.8 * x)
    p0 = expit(-0.5 + 0.7 * x)
    p1 = expit(0.3 + 0.7 * x)
    return x, e, p0, p1


def test_binary_risk_ratio_uses_the_aipw_arm_means_with_a_delta_method_interval():
    rng = np.random.default_rng(0)
    x, e, p0, p1 = _binary_population(rng, 3000)
    t = rng.binomial(1, e).astype(float)
    y = rng.binomial(1, np.where(t == 1, p1, p0)).astype(float)
    m1, m0, if1, if0 = sensitivity.aipw_arm_means(y, t, e, p0, p1, clip=0.02)
    aipw = diagnostics.dr_scores(y, t, e, p0, p1, clip=0.02)
    assert m1 - m0 == pytest.approx(aipw.mean(), abs=1e-12)
    result = sensitivity.risk_ratio_evalue(m1, m0, if1, if0)
    # Independent delta method: gradient of log(m1/m0) and the arm-mean covariance.
    covariance = np.cov(np.vstack((if1, if0)), ddof=1) / len(y)
    gradient = np.array([1 / m1, -1 / m0])
    se = math.sqrt(gradient @ covariance @ gradient)
    assert result["risk_ratio"] == pytest.approx(m1 / m0)
    assert result["log_risk_ratio_standard_error"] == pytest.approx(se, rel=1e-10)
    low, high = result["risk_ratio_confidence_interval_95"]
    assert low == pytest.approx(m1 / m0 * math.exp(-Z * se))
    assert high == pytest.approx(m1 / m0 * math.exp(Z * se))
    assert result["evalue_point"] == pytest.approx(sensitivity.evalue(m1 / m0))
    assert result["evalue_confidence_limit"] == pytest.approx(
        sensitivity.evalue_confidence_limit(low, high))


# ---------------------------------------------------------------------------
# Omitted-variable-bias bounds


def _continuous_sample(rng, n=4000):
    x = rng.normal(size=n)
    e = expit(1.2 * x)
    t = rng.binomial(1, e).astype(float)
    mu0, mu1 = x, x + 0.4
    y = np.where(t == 1, mu1, mu0) + rng.normal(size=n)
    return y, t, e, mu0, mu1


def _doubleml_reference(y, d, m_hat, g0, g1, cf_y, cf_d, rho=1.0):
    """Independent transcription of the DoubleML IRM (ATE) sensitivity formulas.

    The adjusted scores use the delta-method orientation, psi = score - theta
    (see ``test_bound_standard_errors_follow_the_delta_method``).
    """

    n = len(y)
    psi_b = g1 - g0 + d * (y - g1) / m_hat - (1 - d) * (y - g0) / (1 - m_hat)
    theta = psi_b.mean()
    psi_scaled = psi_b - theta
    sigma2_element = (y - d * g1 - (1 - d) * g0) ** 2
    sigma2 = sigma2_element.mean()
    m_alpha = 1 / m_hat + 1 / (1 - m_hat)
    riesz = d / m_hat - (1 - d) / (1 - m_hat)
    nu2_element = 2 * m_alpha - riesz ** 2
    nu2 = nu2_element.mean()
    strength = abs(rho) * np.sqrt(cf_y * cf_d / (1 - cf_d))
    scaling = np.sqrt(sigma2 * nu2)
    max_bias = strength * scaling
    psi_max_bias = strength * (sigma2 * (nu2_element - nu2) + nu2 * (sigma2_element - sigma2)) / (
        2 * scaling)
    var_lower = np.mean((psi_scaled - psi_max_bias) ** 2) / n
    var_upper = np.mean((psi_scaled + psi_max_bias) ** 2) / n
    return {
        "theta": theta, "sigma2": sigma2, "nu2": nu2,
        "lower": theta - max_bias, "upper": theta + max_bias,
        "var_lower": var_lower, "var_upper": var_upper,
    }


@pytest.mark.parametrize("cf_y,cf_d,rho", [(0.1, 0.2, 0.5)])
def test_ovb_bounds_match_an_independent_doubleml_transcription(cf_y, cf_d, rho):
    y, t, e, mu0, mu1 = _continuous_sample(np.random.default_rng(5))
    elements = sensitivity.ate_elements(y, t, e, mu0, mu1, clip=1e-6)
    reference = _doubleml_reference(y, t, e, mu0, mu1, cf_y, cf_d, rho)
    ours = sensitivity.ovb_bounds(elements, cf_y, cf_d, rho=rho)
    assert elements.theta == pytest.approx(reference["theta"], rel=1e-12)
    assert elements.sigma2 == pytest.approx(reference["sigma2"], rel=1e-12)
    assert elements.nu2 == pytest.approx(reference["nu2"], rel=1e-12)
    assert ours["lower"] == pytest.approx(reference["lower"], rel=1e-12)
    assert ours["upper"] == pytest.approx(reference["upper"], rel=1e-12)
    # DoubleML averages the squared adjusted scores over n; Stage 2 uses n - 1,
    # matching its reported standard errors.
    n = len(y)
    for side in ("lower", "upper"):
        expected = math.sqrt(reference[f"var_{side}"] * n / (n - 1))
        assert ours[f"{side}_standard_error"] == pytest.approx(expected, rel=1e-10)
    if cf_y == cf_d == 0.0:
        se = np.std(diagnostics.dr_scores(y, t, e, mu0, mu1, clip=1e-6), ddof=1) / math.sqrt(n)
        assert ours["lower_confidence_bound"] == pytest.approx(elements.theta - 1.96 * se)
        assert ours["upper_confidence_bound"] == pytest.approx(elements.theta + 1.96 * se)


def test_ate_bound_contains_the_truth_at_the_true_confounding_strength():
    """With an omitted confounder U, the true (cf_y, cf_d) bound covers the true ATE."""

    rng = np.random.default_rng(8)
    n, tau, c = 400_000, 0.5, 0.8
    x, u = rng.normal(size=n), rng.normal(size=n)
    e_long = expit(0.6 * x + 0.8 * u)
    t = rng.binomial(1, e_long).astype(float)
    y = tau * t + x + c * u + rng.normal(size=n)
    # Short nuisances by Gauss-Hermite integration over U ~ N(0, 1).
    nodes, weights = np.polynomial.hermite_e.hermegauss(60)
    weights = weights / weights.sum()
    p = expit(0.6 * x[:, None] + 0.8 * nodes[None, :])
    e_short = p @ weights
    u_treated = (p * nodes) @ weights / e_short
    u_control = ((1 - p) * nodes) @ weights / (1 - e_short)
    mu1_short, mu0_short = tau + x + c * u_treated, x + c * u_control
    short = sensitivity.ate_elements(y, t, e_short, mu0_short, mu1_short, clip=1e-9)
    g_long = tau * t + x + c * u
    g_short = np.where(t == 1, mu1_short, mu0_short)
    alpha_long = t / e_long - (1 - t) / (1 - e_long)
    alpha_short = t / e_short - (1 - t) / (1 - e_short)
    cf_y = np.mean((g_long - g_short) ** 2) / np.mean((y - g_short) ** 2)
    cf_d = 1 - np.mean(alpha_short ** 2) / np.mean(alpha_long ** 2)
    bounds = sensitivity.ovb_bounds(short, cf_y, cf_d)
    assert short.theta - tau > 0.2  # the omitted confounder biases the short estimate
    assert bounds["lower"] <= tau <= bounds["upper"]
    # The bound is |rho| times the bias; here rho is the correlation of g - g_s and alpha - alpha_s.
    assert abs(short.theta - tau) <= bounds["max_bias"]


# ---------------------------------------------------------------------------
# Benchmark refits


DEFINITIONS = [
    {"feature_id": f"f{index}", "name": name, "value_type": "continuous"}
    for index, name in enumerate(("strong_confounder", "outcome_only", "noise"), start=1)
]


def _benchmark_inputs(tmp_path, rng=None, n=1500):
    rng = rng or np.random.default_rng(10)
    x1, x2, x3 = rng.normal(size=(3, n))
    t = rng.binomial(1, expit(1.0 * x1)).astype(int)
    y = 0.5 * t + 1.0 * x1 + 1.0 * x2 + rng.normal(size=n)
    dataset = pd.DataFrame({"treatment": t, "outcome": y})
    measurements = pd.DataFrame({
        "_oci_row_id": np.arange(n), "strong_confounder": x1, "outcome_only": x2, "noise": x3,
    })
    fit_ids, heldout_ids = list(range(1000)), list(range(1000, n))
    estimation = tmp_path / "outer_001" / "estimation"
    estimation.mkdir(parents=True)
    e = expit(x1)
    mu0, mu1 = x1 + x2, 0.5 + x1 + x2
    predictions = pd.DataFrame({
        "_oci_row_id": heldout_ids, "outer_fold": 1,
        "treatment": t[heldout_ids], "outcome": y[heldout_ids],
        "propensity": e[heldout_ids], "mu0": mu0[heldout_ids], "mu1": mu1[heldout_ids],
        "aipw_score": diagnostics.dr_scores(y[heldout_ids], t[heldout_ids], e[heldout_ids],
                                            mu0[heldout_ids], mu1[heldout_ids], clip=0.02),
        "estimated_cate": 0.5,
    })
    predictions.to_csv(estimation / "predictions.csv", index=False)
    inputs = {
        "dataset": dataset,
        "candidate_definitions": DEFINITIONS,
        "candidate_fit": measurements.iloc[fit_ids].reset_index(drop=True),
        "candidate_heldout": measurements.iloc[heldout_ids].reset_index(drop=True),
        "selected_definitions": [{**DEFINITIONS[0], "roles": ["confounder"]}],
        "split": {"fit_row_ids": fit_ids, "heldout_row_ids": heldout_ids},
        "treatment_column": "treatment",
        "outcome_column": "outcome",
        "outcome_type": "continuous",
        "seed": 7,
        "propensity_clip": 0.02,
    }
    return inputs, estimation, predictions


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot(root):
    """Digests of every file outside the benchmark directory."""

    return {
        str(path.relative_to(root)): _digest(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and "sensitivity_benchmarks" not in path.parts
    }


def test_benchmarks_rank_the_known_confounder_and_show_its_bias_direction(tmp_path, monkeypatch):
    inputs, estimation, predictions = _benchmark_inputs(tmp_path)
    (estimation / "diagnostics.json").write_text('{"ate_aipw": 0.5}')
    before = _snapshot(tmp_path)
    frozen_inputs = {key: inputs[key].copy() for key in ("candidate_fit", "candidate_heldout")}
    result = sensitivity.run_fold_benchmarks(**inputs, groups=2, estimation_dir=estimation)
    assert result["status"] == "ok"
    # Reporting only: predictions, fold diagnostics and the inputs are untouched.
    assert _snapshot(tmp_path) == before
    for key, frame in frozen_inputs.items():
        pd.testing.assert_frame_equal(inputs[key], frame)
    first = result["groups"][0]
    assert first["name"] == "strong_confounder" and first["selected_roles"] == ["confounder"]
    gains = first["gains"]
    # Omitting a variable that raises both treatment and outcome biases the ATE upward.
    assert gains["theta_short_minus_long"] > 0.3
    assert gains["cf_y"] > 0.1 and gains["cf_d"] > 0.05 and gains["rho"] > 0
    assert gains["sigma2_short"] > gains["sigma2_long"] and gains["nu2_short"] < gains["nu2_long"]
    assert all(group["name"] != "noise" or group["rank"] > 1 for group in result["groups"])
    written = json.loads((estimation / "sensitivity_benchmarks" / "benchmarks.json").read_text())
    assert written == result

    # Resume: a matching benchmark file is reused without refitting.
    def refuse(*args, **kwargs):
        raise AssertionError("refit on resume")

    monkeypatch.setattr(sensitivity, "_benchmark_fit", refuse)
    assert sensitivity.run_fold_benchmarks(**inputs, groups=2, estimation_dir=estimation) == result

    # The pooled section applies the gains to this fold's and the pooled ATE.
    section = sensitivity.unmeasured_confounding_section(
        predictions, outcome_type="continuous", propensity_clip=0.02,
        benchmarks_by_fold={"1": result},
    )
    fold = section["benchmarks"]["per_fold"]["1"]
    assert section["benchmarks"]["status"] == "ok" and fold["status"] == "ok"
    bounds = fold["groups"][0]["pooled_ate_bounds"]["rho_1"]
    ate = predictions["aipw_score"].mean()
    assert bounds["lower"] < ate < bounds["upper"]
    statement = section["benchmarks"]["pooled"]["rows"][0]["statement"]
    assert statement.startswith("A confounder as strong as strong_confounder (fold 1)")
    # A benchmark file for different held-out rows is not applied.
    stale = predictions.copy()
    stale.loc[0, "aipw_score"] = np.nan
    mismatch = sensitivity.unmeasured_confounding_section(
        stale, outcome_type="continuous", propensity_clip=0.02,
        benchmarks_by_fold={"1": result},
    )
    assert mismatch["benchmarks"]["per_fold"]["1"]["status"] == "unavailable"


# ---------------------------------------------------------------------------
# Section, diagnostics integration, configuration
