"""Reporting-only unmeasured-confounding sensitivity: E-values, OVB bounds, benchmarks."""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import brentq
from scipy.special import expit

from oci.inference import stage2_diagnostics as diagnostics
from oci.inference import stage2_sensitivity as sensitivity
from oci.inference.plain_handoff_stage2 import plain_stage2_config_from_mapping

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


def test_confidence_limit_evalue_uses_the_limit_closest_to_the_null():
    assert sensitivity.evalue_confidence_limit(0.8, 1.3) == 1.0
    assert sensitivity.evalue_confidence_limit(1.0, 2.0) == 1.0
    assert sensitivity.evalue_confidence_limit(1.8, 8.0) == pytest.approx(
        sensitivity.evalue(1.8))
    assert sensitivity.evalue_confidence_limit(0.2, 0.7) == pytest.approx(
        sensitivity.evalue(0.7))
    assert sensitivity.evalue_confidence_limit(None, 2.0) is None


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


def test_binary_risk_ratio_interval_covers_the_causal_risk_ratio():
    _, _, p0, p1 = _binary_population(np.random.default_rng(1), 2_000_000)
    truth = p1.mean() / p0.mean()
    rng = np.random.default_rng(2)
    covered, reps = 0, 200
    for _ in range(reps):
        x, e, p0, p1 = _binary_population(rng, 2000)
        t = rng.binomial(1, e).astype(float)
        y = rng.binomial(1, np.where(t == 1, p1, p0)).astype(float)
        result = sensitivity.risk_ratio_evalue(
            *sensitivity.aipw_arm_means(y, t, e, p0, p1, clip=0.02))
        low, high = result["risk_ratio_confidence_interval_95"]
        covered += low <= truth <= high
    assert 0.90 <= covered / reps <= 0.99


def test_overlap_arm_means_reproduce_the_reported_ato():
    rng = np.random.default_rng(3)
    x, e, p0, p1 = _binary_population(rng, 2000)
    t = rng.binomial(1, e).astype(float)
    y = rng.binomial(1, np.where(t == 1, p1, p0)).astype(float)
    m1, m0, if1, if0 = sensitivity.overlap_arm_means(y, t, e, p0, p1)
    reported = diagnostics.overlap_weighted_effects(y, t, e, p0, p1)["augmented"]
    assert m1 - m0 == pytest.approx(reported["estimate"], abs=1e-12)
    assert np.std(if1 - if0, ddof=1) / math.sqrt(len(y)) == pytest.approx(
        reported["standard_error"], rel=1e-12)
    assert sensitivity.risk_ratio_evalue(m1, m0, if1, if0)["status"] == "ok"


def test_nonpositive_arm_mean_makes_the_risk_ratio_unavailable():
    result = sensitivity.risk_ratio_evalue(0.2, -0.01, np.zeros(10), np.zeros(10))
    assert result["status"] == "unavailable" and "arm mean" in result["reason"]


def test_continuous_outcomes_use_the_approximate_standardized_difference_conversion():
    result = sensitivity.standardized_difference_evalue(0.5, 0.1, 2.0)
    d, se_d = 0.25, 0.05
    assert result["standardized_mean_difference"] == pytest.approx(d)
    assert result["risk_ratio"] == pytest.approx(math.exp(0.91 * d))
    assert result["risk_ratio_confidence_interval_95"] == pytest.approx(
        [math.exp(0.91 * (d - Z * se_d)), math.exp(0.91 * (d + Z * se_d))])
    assert result["evalue_point"] == pytest.approx(sensitivity.evalue(math.exp(0.91 * d)))
    assert result["evalue_confidence_limit"] == pytest.approx(
        sensitivity.evalue(math.exp(0.91 * (d - Z * se_d))))
    negative = sensitivity.standardized_difference_evalue(-0.5, 0.1, 2.0)
    assert negative["risk_ratio"] < 1
    assert negative["evalue_point"] == pytest.approx(result["evalue_point"])
    wide = sensitivity.standardized_difference_evalue(0.5, 1.0, 2.0)
    assert wide["evalue_confidence_limit"] == 1.0
    assert sensitivity.standardized_difference_evalue(0.5, 0.1, 0.0)["status"] == "unavailable"


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


@pytest.mark.parametrize("cf_y,cf_d,rho", [(0.03, 0.04, 1.0), (0.1, 0.2, 0.5), (0.0, 0.0, 1.0)])
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


def test_bound_standard_errors_follow_the_delta_method():
    # theta - bias has influence function psi - psi_bias, so its standard error
    # must track the sampling spread of the lower bound; skewed, arm-dependent
    # noise makes the two adjusted scores differ sharply.
    rng = np.random.default_rng(1)
    lower, upper, lower_se, upper_se = [], [], [], []
    for _ in range(600):
        x = rng.normal(size=600)
        e = expit(0.8 * x)
        t = rng.binomial(1, e).astype(float)
        mu0, mu1 = 0.5 * x, 1.0 + 0.5 * x
        y = np.where(t == 1, mu1, mu0) + (rng.exponential(1.0, 600) - 1.0) * (1 + 1.5 * t)
        bounds = sensitivity.ovb_bounds(
            sensitivity.ate_elements(y, t, e, mu0, mu1, clip=0.01), 0.5, 0.5)
        lower.append(bounds["lower"])
        upper.append(bounds["upper"])
        lower_se.append(bounds["lower_standard_error"])
        upper_se.append(bounds["upper_standard_error"])
    spread_lower, spread_upper = np.std(lower, ddof=1), np.std(upper, ddof=1)
    assert np.mean(lower_se) == pytest.approx(spread_lower, rel=0.15)
    assert np.mean(upper_se) == pytest.approx(spread_upper, rel=0.15)
    assert abs(np.mean(lower_se) - spread_lower) < abs(np.mean(upper_se) - spread_lower)


def test_robustness_values_solve_the_bound_equations():
    y, t, e, mu0, mu1 = _continuous_sample(np.random.default_rng(6))
    elements = sensitivity.ate_elements(y, t, e, mu0, mu1, clip=0.02)
    values = sensitivity.robustness_values(elements)
    k = abs(elements.theta) / math.sqrt(elements.sigma2 * elements.nu2)
    closed_form = (-k ** 2 + math.sqrt(k ** 4 + 4 * k ** 2)) / 2
    assert values["robustness_value"] == pytest.approx(closed_form, abs=1e-9)
    rv_brent = brentq(lambda x: sensitivity.ovb_bounds(elements, x, x)["lower"], 0, 0.9999)
    assert values["robustness_value"] == pytest.approx(rv_brent, abs=1e-8)
    rva = values["robustness_value_confidence"]
    assert 0 < rva < values["robustness_value"]
    assert sensitivity.ovb_bounds(elements, rva, rva)["lower_confidence_bound"] == pytest.approx(
        0.0, abs=1e-8)
    # A confidence interval that already includes zero gives RVa = 0.
    null = sensitivity.ate_elements(*_continuous_sample(np.random.default_rng(7), n=60),
                                    clip=0.02)
    shifted = sensitivity.OVBElements(
        0.01 * null.standard_error, null.psi, null.sigma2, null.nu2, null.psi_sigma2,
        null.psi_nu2)
    assert sensitivity.robustness_values(shifted)["robustness_value_confidence"] == 0.0
    # A negative estimate uses the upper bound.
    negative = sensitivity.OVBElements(-elements.theta, elements.psi, elements.sigma2,
                                       elements.nu2, elements.psi_sigma2, elements.psi_nu2)
    assert sensitivity.robustness_values(negative)["robustness_value"] == pytest.approx(
        closed_form, abs=1e-9)


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


def test_ato_elements_use_the_fixed_overlap_weight_representer():
    y, t, e, mu0, mu1 = _continuous_sample(np.random.default_rng(9), n=20_000)
    elements = sensitivity.ato_elements(y, t, e, mu0, mu1)
    reported = diagnostics.overlap_weighted_effects(y, t, e, mu0, mu1)["augmented"]
    assert elements.theta == pytest.approx(reported["estimate"], abs=1e-12)
    assert elements.standard_error == pytest.approx(reported["standard_error"], rel=1e-10)
    h = e * (1 - e)
    alpha = (t - e) / h.mean()
    assert elements.nu2 == pytest.approx(np.mean(2 * h / h.mean() ** 2 - alpha ** 2), rel=1e-12)
    # With the true propensity E[(D - e)^2] = E[h], so nu^2 is close to 1 / E[h].
    assert elements.nu2 == pytest.approx(1 / h.mean(), rel=0.03)
    summary = sensitivity.ovb_summary(elements)
    assert summary["status"] == "ok" and len(summary["contour"]) == 25
    assert {row["cf_y"] for row in summary["contour"]} == set(sensitivity.OVB_GRID)


def test_benchmark_gain_statistics_follow_the_doubleml_guide():
    psi = np.zeros(10)
    long = sensitivity.OVBElements(0.30, psi, 1.00, 4.0, psi, psi)
    short = sensitivity.OVBElements(0.40, psi, 1.10, 3.5, psi, psi)
    gains = sensitivity.benchmark_gains(long, short, outcome_variance=2.0)
    r2_long, r2_short = 1 - 1.00 / 2.0, 1 - 1.10 / 2.0
    assert gains["cf_y"] == pytest.approx((r2_long - r2_short) / (1 - r2_long))
    assert gains["cf_d"] == pytest.approx((1 - 3.5 / 4.0) / (3.5 / 4.0))
    assert gains["rho"] == pytest.approx(0.10 / math.sqrt(0.10 * 0.5))
    # rho is limited to [-1, 1]; a short model that fits better leaves rho unidentified.
    strong = sensitivity.benchmark_gains(
        long, sensitivity.OVBElements(-0.5, psi, 1.01, 3.99, psi, psi), outcome_variance=2.0)
    assert strong["rho"] == -1.0
    better = sensitivity.benchmark_gains(
        long, sensitivity.OVBElements(0.3, psi, 0.9, 3.5, psi, psi), outcome_variance=2.0)
    assert better["cf_y"] == 0.0 and better["rho"] is None and better["rho_reason"]


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


def test_benchmarks_disabled_and_failures_are_recorded_not_raised(tmp_path):
    inputs, estimation, _ = _benchmark_inputs(tmp_path, n=1200)
    disabled = sensitivity.run_fold_benchmarks(**inputs, groups=0, estimation_dir=estimation)
    assert disabled["status"] == "disabled"
    assert not (estimation / "sensitivity_benchmarks" / "fits").exists()

    class Config:
        class sensitivity:  # noqa: N801 - mimics the config attribute
            benchmark_groups = 3

    broken = {**inputs, "split": {**inputs["split"], "heldout_row_ids": [0, 1, 2]}}
    failed = sensitivity.benchmark_fold(config=Config, estimation_dir=estimation, **broken)
    assert failed["status"] == "error" and failed["requested_groups"] == 3
    recorded = sensitivity.load_fold_benchmarks(tmp_path / "outer_001")
    assert recorded["status"] == "error"
    assert "missing" in sensitivity.load_fold_benchmarks(tmp_path / "outer_404")


# ---------------------------------------------------------------------------
# Section, diagnostics integration, configuration


def _predictions(rng, n=1200, folds=2, binary=True):
    x = rng.normal(size=n)
    e = expit(0.8 * x)
    t = rng.binomial(1, e).astype(float)
    if binary:
        mu0, mu1 = expit(-0.5 + 0.7 * x), expit(0.3 + 0.7 * x)
        y = rng.binomial(1, np.where(t == 1, mu1, mu0)).astype(float)
    else:
        mu0, mu1 = x, x + 0.4
        y = np.where(t == 1, mu1, mu0) + rng.normal(size=n)
    return pd.DataFrame({
        "_oci_row_id": np.arange(n), "outer_fold": np.arange(n) % folds + 1,
        "treatment": t, "outcome": y, "propensity": e, "mu0": mu0, "mu1": mu1,
        "aipw_score": diagnostics.dr_scores(y, t, e, mu0, mu1, clip=0.02),
        "estimated_cate": mu1 - mu0,
    })


def test_diagnostics_report_the_section_without_changing_the_predictions(tmp_path):
    frame = _predictions(np.random.default_rng(11))
    original = frame.copy()
    path, summary = diagnostics.write_heterogeneity_diagnostics(
        output_dir=tmp_path, predictions=frame, propensity_clip=0.02, seed=1, n_bootstrap=10,
        outcome_type="binary",
    )
    pd.testing.assert_frame_equal(frame, original)
    full = json.loads(path.read_text())
    section = full["unmeasured_confounding_sensitivity"]
    assert section["status"] == "ok" and section["outcome_type"] == "binary"
    ate_ev = section["evalues"]["ate"]["pooled"]
    m1, m0, _, _ = sensitivity.aipw_arm_means(
        frame["outcome"], frame["treatment"], frame["propensity"], frame["mu0"], frame["mu1"],
        clip=0.02)
    assert ate_ev["risk_ratio"] == pytest.approx(m1 / m0)
    assert set(section["evalues"]["ato"]["per_fold"]) == {"1", "2"}
    ovb = section["omitted_variable_bias"]
    assert ovb["ate"]["pooled"]["estimate"] == pytest.approx(frame["aipw_score"].mean())
    assert ovb["ato"]["pooled"]["estimate"] == pytest.approx(
        full["overlap_weighted_ate"]["pooled"]["augmented"]["estimate"])
    # Missing benchmark files leave only the benchmarks unavailable.
    assert section["benchmarks"]["status"] == "unavailable"
    assert "missing" in section["benchmarks"]["per_fold"]["1"]["reason"]
    compact = summary["unmeasured_confounding_sensitivity"]
    assert compact["ate"]["evalue_point"] == pytest.approx(ate_ev["evalue_point"])
    assert compact["ate"]["robustness_value"] == pytest.approx(
        ovb["ate"]["pooled"]["robustness_value"])
    assert compact["benchmarks"]["strongest"] is None
    for text in ("evalue", "robustness_value", "benchmarks", "measurement_error"):
        assert section["interpretation"][text]
    assert any("10.7326/M16-2607" in reference for reference in section["references"])


def test_continuous_section_and_inferred_outcome_type():
    frame = _predictions(np.random.default_rng(12), binary=False)
    full = diagnostics.build_heterogeneity_diagnostics(
        frame, propensity_clip=0.02, seed=1, n_bootstrap=10)
    section = full["unmeasured_confounding_sensitivity"]
    assert section["outcome_type"] == "continuous"
    pooled = section["evalues"]["ate"]["pooled"]
    sd = np.std(frame["outcome"], ddof=1)
    assert pooled["standardized_mean_difference"] == pytest.approx(
        frame["aipw_score"].mean() / sd)
    assert pooled["scale"] == "approximate_risk_ratio_from_standardized_mean_difference"


def test_sensitivity_configuration_is_strict():
    raw = {"endpoint": "http://stage2.test/v1", "model": "m"}
    default = plain_stage2_config_from_mapping(raw, default_workers=1)
    assert default.sensitivity.benchmark_groups == 5
    assert default.public_dict()["sensitivity"] == {"benchmark_groups": 5}
    off = plain_stage2_config_from_mapping(
        {**raw, "sensitivity": {"benchmark_groups": 0}}, default_workers=1)
    assert off.sensitivity.benchmark_groups == 0
    restored = plain_stage2_config_from_mapping(
        {**raw, "sensitivity": off.public_dict()["sensitivity"]}, default_workers=1)
    assert restored.sensitivity == off.sensitivity
    for bad in ({"benchmark_groups": -1}, {"benchmark_groups": 51},
                {"benchmark_groups": True}, {"benchmark_groups": "5"},
                {"benchmark_groups": 2.0}, {"benchmark_group": 5}, [5]):
        with pytest.raises(ValueError, match="stage2.sensitivity"):
            plain_stage2_config_from_mapping({**raw, "sensitivity": bad}, default_workers=1)


def test_sensitivity_settings_are_outside_the_selection_identity():
    import copy
    from pathlib import Path

    from oci.inference.research_all_evidence_workflow import (
        _stage2_reselection_policy_fingerprint,
        compile_config,
    )

    path = Path(__file__).resolve().parents[1] / "example_configs/research_all_evidence_multi_model.json"
    raw = json.loads(path.read_text())
    config = compile_config(raw, config_dir=path.parent)
    changed = copy.deepcopy(raw)
    changed["stage2"]["sensitivity"] = {"benchmark_groups": 0}
    disabled = compile_config(changed, config_dir=path.parent)
    assert config.stage2.sensitivity.benchmark_groups == 5
    assert disabled.stage2.sensitivity.benchmark_groups == 0
    assert _stage2_reselection_policy_fingerprint(config) == (
        _stage2_reselection_policy_fingerprint(disabled))
