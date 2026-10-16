"""Reporting-only sensitivity of the Stage 2 estimates to unmeasured confounding.

Everything here runs after a fold's predictions are frozen. Nothing in
selection, estimation, or any other fold reads these results, and no oracle
column is read: the inputs are the cross-fitted prediction table, the fold's
nuisance-candidate measurements, and the dataset's treatment and outcome
columns.

**E-values** (VanderWeele and Ding 2017). For a risk ratio RR >= 1 the E-value
is ``RR + sqrt(RR * (RR - 1))``; for RR < 1 the formula is applied to 1/RR. The
E-value for the confidence interval is that of the limit closest to the null,
and 1 when the interval includes 1.

* Binary outcomes: RR is the causal risk ratio E[Y(1)] / E[Y(0)] formed from
  the same arm-mean estimates whose difference is the reported risk difference
  (the clipped AIPW arm means for the ATE; the augmented overlap-weighted arm
  means for the ATO). Its 95% CI is the delta-method interval on log RR from
  the arm-mean influence functions, with the cross-fitted nuisances held fixed.
* Continuous outcomes: d = estimate / SD(Y), with SD(Y) the marginal standard
  deviation of the observed outcome over the rows that enter the estimate, and
  RR is approximated by exp(0.91 * d), with the CI from d +/- 1.96 * SE(d). This
  conversion combines Chinn (2000), ln(OR) ~ 1.81 * d, with VanderWeele (2017),
  RR ~ sqrt(OR) for a common outcome, as the EValue R package does for
  continuous outcomes; it is not a verbatim formula from either paper.

**Omitted-variable-bias bounds** (Chernozhukov, Cinelli, Newey, Sharma and
Syrgkanis; formulas as in the DoubleML sensitivity guide). With
sigma^2 = E[(Y - g(D, X))^2] and nu^2 = E[alpha^2], estimated by
E_n[2 m(W, alpha) - alpha^2], the bias of the estimate is bounded by
``|rho| * sigma * nu * C_Y * C_D`` with ``C_Y^2 = cf_y`` and
``C_D^2 = cf_d / (1 - cf_d)``; bounds are ``theta -/+`` that amount, with
rho = 1 as the conservative choice. Confidence bounds come from the adjusted
scores ``psi -/+ psi_bias``, the delta-method influence functions of the lower
and upper bounds, with ``psi`` the centered score (DoubleML 0.11 pairs
``psi_bias`` with its Jacobian-scaled score ``-psi``, which exchanges its two
bound standard errors); the robustness value RV is the common value of cf_y
and cf_d at which the bound reaches zero effect, and RVa uses the confidence
bound instead (Cinelli and Hazlett 2020 introduced the robustness value).

* ATE: alpha = D / e - (1 - D) / (1 - e) with the configured clip applied to e,
  m(W, alpha) = 1 / e + 1 / (1 - e), over the effect-eligible rows.
* ATO: the overlap weights h(X) = e(X)(1 - e(X)) are treated as a fixed
  function of the measured covariates, so the target is the h-weighted average
  effect over the measured-covariate equipoise population. Then
  alpha = h / E[h] * (D / e - (1 - D) / (1 - e)) = (D - e) / E[h] and
  m(W, alpha) = h / E[h]^2. The bound does not describe how an omitted
  confounder would redefine that population.

**Benchmarking** (Cinelli and Hazlett 2020; gain statistics as in the DoubleML
guide). For the ATE, each fold refits its nuisances on the outer-training rows
without one discovered variable at a time ("short" models) and compares them
with a refit that keeps every variable ("long" reference). Both are elastic
nets fitted once on all outer-training rows, so the comparison does not mix the
configured nuisance learner with the omission. Held-out values give
``cf_y = (R^2 - R^2_short) / (1 - R^2)`` with ``R^2 = 1 - sigma^2 / Var(Y)``,
``cf_d = (1 - R^2_alpha) / R^2_alpha`` with ``R^2_alpha = nu^2_short / nu^2``,
and ``rho = (theta_short - theta) / sqrt((sigma^2_short - sigma^2) *
(nu^2 - nu^2_short))``; cf_y and cf_d are limited to [0, 1] and |rho| to 1.
The variables are ranked by the product of the L2 norms of their standardized
elastic-net coefficients in the long reference's propensity model and in its
two arm outcome models.

References (exactly as verified for this project):

* VanderWeele TJ, Ding P. Sensitivity analysis in observational research:
  introducing the E-value. Annals of Internal Medicine. 2017;167(4):268-274.
  doi:10.7326/M16-2607. PMID 28693043.
* Ding P, VanderWeele TJ. Sensitivity analysis without assumptions.
  Epidemiology. 2016;27(3):368-377.
* VanderWeele TJ. On a square-root transformation of the odds ratio for a
  common outcome. Epidemiology. 2017;28(6):e58-e60.
  doi:10.1097/EDE.0000000000000733.
* Chinn S. A simple method for converting an odds ratio to effect size for use
  in meta-analysis. Statistics in Medicine. 2000;19(22):3127-3131.
  PMID 11113947.
* Mathur MB, Ding P, Riddell CA, VanderWeele TJ. Website and R package for
  computing E-values. Epidemiology. 2018;29(5):e45-e47 (PMC6066405).
* Chernozhukov V, Cinelli C, Newey W, Sharma A, Syrgkanis V. Long story short:
  omitted variable bias in causal machine learning. NBER Working Paper 30302
  (2022); arXiv:2112.13398; The Review of Economics and Statistics,
  doi:10.1162/REST.a.1705.
* Cinelli C, Hazlett C. Making sense of sensitivity: extending omitted variable
  bias. Journal of the Royal Statistical Society Series B. 2020;82(1):39-67.
  doi:10.1111/rssb.12348.
* Bach P, Chernozhukov V, Kurz MS, Spindler M. DoubleML - an object-oriented
  implementation of double machine learning in Python. Journal of Machine
  Learning Research. 2022;23(53):1-6. Sensitivity user guide:
  https://docs.doubleml.org/stable/guide/sensitivity.html
* Li F, Morgan KL, Zaslavsky AM. Balancing covariates via propensity score
  weighting. Journal of the American Statistical Association.
  2018;113(521):390-400. doi:10.1080/01621459.2016.1260466.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

SECTION_SCHEMA_VERSION = "stage2_unmeasured_confounding_sensitivity_v1"
BENCHMARK_SCHEMA_VERSION = "stage2_sensitivity_benchmarks_v1"
BENCHMARK_DIRNAME = "sensitivity_benchmarks"
BENCHMARK_FILENAME = "benchmarks.json"
# (cf_y, cf_d) grid of the bound table; rho = 1.
OVB_GRID = (0.01, 0.02, 0.05, 0.10, 0.20)
OVB_RHO = 1.0
# One-sided critical value of each confidence bound, the value Stage 2 uses for
# its 95% intervals, so at zero confounding the confidence bounds equal the
# reported 95% confidence interval.
CONFIDENCE_Z = 1.96
# Approximate log-RR per standardized mean difference: 1.81 / 2, rounded.
SMD_TO_LOG_RR = 0.91
DEFAULT_BENCHMARK_GROUPS = 5
MAX_BENCHMARK_GROUPS = 50
BENCHMARK_SEED_OFFSET = 40_000
RV_SCAN_POINTS = 200
RV_UPPER = 1.0 - 1e-9

REFERENCES = (
    "VanderWeele TJ, Ding P. Sensitivity analysis in observational research: introducing "
    "the E-value. Annals of Internal Medicine. 2017;167(4):268-274. doi:10.7326/M16-2607. "
    "PMID 28693043.",
    "Ding P, VanderWeele TJ. Sensitivity analysis without assumptions. Epidemiology. "
    "2016;27(3):368-377.",
    "VanderWeele TJ. On a square-root transformation of the odds ratio for a common "
    "outcome. Epidemiology. 2017;28(6):e58-e60. doi:10.1097/EDE.0000000000000733.",
    "Chinn S. A simple method for converting an odds ratio to effect size for use in "
    "meta-analysis. Statistics in Medicine. 2000;19(22):3127-3131. PMID 11113947.",
    "Mathur MB, Ding P, Riddell CA, VanderWeele TJ. Website and R package for computing "
    "E-values. Epidemiology. 2018;29(5):e45-e47 (PMC6066405).",
    "Chernozhukov V, Cinelli C, Newey W, Sharma A, Syrgkanis V. Long story short: omitted "
    "variable bias in causal machine learning. NBER Working Paper 30302 (2022); "
    "arXiv:2112.13398; The Review of Economics and Statistics, doi:10.1162/REST.a.1705.",
    "Cinelli C, Hazlett C. Making sense of sensitivity: extending omitted variable bias. "
    "Journal of the Royal Statistical Society Series B. 2020;82(1):39-67. "
    "doi:10.1111/rssb.12348.",
    "Bach P, Chernozhukov V, Kurz MS, Spindler M. DoubleML - an object-oriented "
    "implementation of double machine learning in Python. Journal of Machine Learning "
    "Research. 2022;23(53):1-6. Sensitivity user guide: "
    "https://docs.doubleml.org/stable/guide/sensitivity.html",
    "Li F, Morgan KL, Zaslavsky AM. Balancing covariates via propensity score weighting. "
    "Journal of the American Statistical Association. 2018;113(521):390-400. "
    "doi:10.1080/01621459.2016.1260466.",
)

INTERPRETATIONS = {
    "evalue": (
        "The E-value is the minimum strength of association, on the risk-ratio scale, that "
        "an unmeasured confounder would need to have with both the treatment and the "
        "outcome, conditional on the measured covariates, to fully explain away the "
        "observed association (VanderWeele and Ding 2017). The confidence-limit E-value is "
        "that strength for the confidence limit closest to the null; it is 1 when the "
        "interval includes the null. Larger values mean more robust estimates; an E-value "
        "near 1 means weak confounding could explain the estimate."
    ),
    "evalue_binary": (
        "Binary outcome: the E-value is for the estimated causal risk ratio "
        "E[Y(1)]/E[Y(0)] built from the same arm means whose difference is the reported "
        "risk difference, with a delta-method 95% CI on log RR (nuisances held fixed)."
    ),
    "evalue_continuous": (
        "Continuous outcome: d = estimate / SD(Y), with SD(Y) the marginal standard "
        "deviation of the observed outcome over the rows that enter the estimate, and the "
        "risk ratio is approximated by exp(0.91 * d) with CI exp(0.91 * (d +/- 1.96 SE(d))). "
        "This approximate conversion combines Chinn (2000), ln(OR) ~ 1.81 d, with "
        "VanderWeele (2017), RR ~ sqrt(OR) for a common outcome, as the EValue R package "
        "does; it is not a verbatim formula from either paper."
    ),
    "omitted_variable_bias": (
        "Bounds on the estimate that omitted confounders of a given strength could produce "
        "(Chernozhukov, Cinelli, Newey, Sharma and Syrgkanis). cf_y is the share of the "
        "outcome's residual variation (given treatment and the measured covariates) that "
        "the omitted confounders would explain, a partial R^2; cf_d is the share of the "
        "variation of the Riesz representer alpha (for the ATE, the inverse-propensity "
        "weight D/e - (1-D)/(1-e)) that they would explain. The bound is "
        "theta +/- |rho| sigma nu sqrt(cf_y cf_d / (1 - cf_d)); rho = 1 assumes the "
        "confounding bias is as adversarial as possible, so the bounds are conservative. "
        "Confidence bounds add 1.96 standard errors of the adjusted scores on each side; "
        "at cf_y = cf_d = 0 they equal the reported 95% confidence interval."
    ),
    "robustness_value": (
        "The robustness value RV is the common value x = cf_y = cf_d at which the bound "
        "reaches zero effect: omitted confounders that explain a share x of the outcome's "
        "residual variation (partial R^2) and a share x of the Riesz representer's "
        "variation could explain the estimate away; weaker ones could not (Cinelli and "
        "Hazlett 2020). RVa is the same for the confidence bound, i.e. the strength at "
        "which the 95% confidence bound reaches zero; it is 0 when the confidence interval "
        "already includes zero."
    ),
    "ato_bound": (
        "The ATO bound treats the overlap weights h(X) = e(X)(1 - e(X)) as a fixed "
        "function of the measured covariates: it bounds the bias for the effect averaged "
        "over the measured-covariate equipoise population, not how an omitted confounder "
        "would redefine that population."
    ),
    "benchmarks": (
        "Benchmarks describe hypothetical confounders as strong as the discovered "
        "variables: each fold refits its outer-training nuisances without one variable "
        "and measures, on its held-out patients, how much of the outcome's residual "
        "variation (cf_y) and of the Riesz representer (cf_d) that variable explained "
        "beyond the others, and the bias rho it induced. An unmeasured confounder may be "
        "weaker or stronger than any measured one, so benchmarks can understate or "
        "overstate the real risk. Both the long reference and the short models are "
        "elastic nets fitted once on all outer-training rows (not the configured nuisance "
        "learner), and only the ATE is benchmarked: omitting a variable also changes the "
        "ATO's overlap weights, i.e. its target population."
    ),
    "measurement_error": (
        "Discovered confounders are measured by an extraction model. Measurement error in "
        "them leaves part of their confounding unadjusted, which acts like partial "
        "unmeasured confounding; these sensitivity analyses also cover that residual bias."
    ),
}

ESTIMANDS = ("ate", "ato")


@dataclass(frozen=True)
class SensitivityConfig:
    """``stage2.sensitivity``: reporting-only unmeasured-confounding analyses.

    ``benchmark_groups`` is how many discovered variables per fold are
    benchmarked by refitting the nuisances without them (0 disables the refits;
    E-values and bounds are always reported).
    """

    benchmark_groups: int = DEFAULT_BENCHMARK_GROUPS

    def validate(self) -> None:
        value = self.benchmark_groups
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= MAX_BENCHMARK_GROUPS
        ):
            raise ValueError(
                "stage2.sensitivity.benchmark_groups must be an integer from 0 to "
                f"{MAX_BENCHMARK_GROUPS}"
            )

    def public_dict(self) -> dict[str, Any]:
        return {"benchmark_groups": self.benchmark_groups}


def sensitivity_config_from_mapping(value: Mapping[str, Any] | None) -> SensitivityConfig:
    if value is None:
        return SensitivityConfig()
    if not isinstance(value, Mapping):
        raise ValueError("stage2.sensitivity must be an object")
    unknown = sorted(set(map(str, value)) - set(SensitivityConfig.__dataclass_fields__))
    if unknown:
        raise ValueError(f"stage2.sensitivity contains unsupported fields: {unknown}")
    config = SensitivityConfig(**dict(value))
    config.validate()
    return config


def _float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _unavailable(reason: str, **extra: Any) -> dict[str, Any]:
    return {"status": "unavailable", "reason": reason, **extra}


def _sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def eligible_row_ids_sha256(row_ids: Sequence[int]) -> str:
    """Identity of the held-out rows that enter a fold's ATE."""

    return _sha256(sorted(int(value) for value in row_ids))


# ---------------------------------------------------------------------------
# E-values


def evalue(risk_ratio: Any) -> float | None:
    """E-value for a risk ratio (VanderWeele and Ding 2017); RR < 1 uses 1/RR."""

    rr = _float(risk_ratio)
    if rr is None or rr <= 0.0:
        return None
    if rr < 1.0:
        rr = 1.0 / rr
    return float(rr + math.sqrt(rr * (rr - 1.0)))


def evalue_confidence_limit(lower: Any, upper: Any) -> float | None:
    """E-value of the confidence limit closest to the null; 1 if the CI includes 1."""

    low, high = _float(lower), _float(upper)
    if low is None or high is None or low <= 0.0 or high <= 0.0:
        return None
    if low <= 1.0 <= high:
        return 1.0
    return evalue(low if low > 1.0 else high)


def _risk_ratio_summary(rr: float, lower: float, upper: float, **extra: Any) -> dict[str, Any]:
    return {
        "status": "ok",
        "risk_ratio": float(rr),
        "risk_ratio_confidence_interval_95": [float(lower), float(upper)],
        "evalue_point": evalue(rr),
        "evalue_confidence_limit": evalue_confidence_limit(lower, upper),
        **extra,
    }


def _clean(*arrays: Any) -> tuple[np.ndarray, ...]:
    values = [np.asarray(array, dtype=float) for array in arrays]
    finite = np.logical_and.reduce([np.isfinite(value) for value in values])
    return tuple(value[finite] for value in values)


def aipw_arm_means(y: Any, t: Any, e: Any, mu0: Any, mu1: Any, *, clip: float):
    """Clipped AIPW arm means and their centered influence functions."""

    y, t, e, mu0, mu1 = (np.asarray(v, dtype=float) for v in (y, t, e, mu0, mu1))
    e = np.clip(e, clip, 1.0 - clip)
    phi1 = mu1 + t * (y - mu1) / e
    phi0 = mu0 + (1.0 - t) * (y - mu0) / (1.0 - e)
    m1, m0 = float(phi1.mean()), float(phi0.mean())
    return m1, m0, phi1 - m1, phi0 - m0


def overlap_arm_means(y: Any, t: Any, e: Any, mu0: Any, mu1: Any):
    """Augmented overlap-weighted arm means and their influence functions.

    Their difference is the augmented ATO of
    :func:`oci.inference.stage2_diagnostics.overlap_weighted_effects`, and the
    difference of the influence functions is that estimator's influence function.
    """

    y, t, e, mu0, mu1 = (np.asarray(v, dtype=float) for v in (y, t, e, mu0, mu1))
    h, w1, w0 = e * (1.0 - e), t * (1.0 - e), (1.0 - t) * e
    if h.sum() <= 0 or w1.sum() <= 0 or w0.sum() <= 0:
        raise ValueError("overlap weights are zero in one treatment arm")
    model1, model0 = np.sum(h * mu1) / h.sum(), np.sum(h * mu0) / h.sum()
    residual1 = np.sum(w1 * (y - mu1)) / w1.sum()
    residual0 = np.sum(w0 * (y - mu0)) / w0.sum()
    if1 = h * (mu1 - model1) / h.mean() + w1 * (y - mu1 - residual1) / w1.mean()
    if0 = h * (mu0 - model0) / h.mean() + w0 * (y - mu0 - residual0) / w0.mean()
    return float(model1 + residual1), float(model0 + residual0), if1, if0


def risk_ratio_evalue(m1: float, m0: float, if1: Any, if0: Any) -> dict[str, Any]:
    """Causal risk ratio m1/m0 with a delta-method CI on log RR, and its E-values."""

    if1, if0 = np.asarray(if1, dtype=float), np.asarray(if0, dtype=float)
    n = len(if1)
    if n < 2:
        return _unavailable("fewer than two rows")
    if not (m1 > 0.0 and m0 > 0.0):
        return _unavailable(
            "an estimated arm mean is not positive, so the risk ratio is undefined",
            arm_mean_treated=m1, arm_mean_control=m0,
        )
    log_rr = math.log(m1 / m0)
    influence = if1 / m1 - if0 / m0
    se = float(influence.std(ddof=1) / math.sqrt(n))
    return _risk_ratio_summary(
        math.exp(log_rr), math.exp(log_rr - CONFIDENCE_Z * se),
        math.exp(log_rr + CONFIDENCE_Z * se),
        scale="causal_risk_ratio",
        arm_mean_treated=m1, arm_mean_control=m0,
        log_risk_ratio_standard_error=se, rows=n,
    )


def standardized_difference_evalue(estimate: Any, standard_error: Any, sd: Any) -> dict[str, Any]:
    """Approximate E-value for a mean difference via d = estimate / SD(Y)."""

    estimate, se, sd = _float(estimate), _float(standard_error), _float(sd)
    if estimate is None or se is None or sd is None or sd <= 0.0:
        return _unavailable("the estimate, its standard error or SD(Y) is unavailable")
    d, se_d = estimate / sd, se / sd
    return _risk_ratio_summary(
        math.exp(SMD_TO_LOG_RR * d),
        math.exp(SMD_TO_LOG_RR * (d - CONFIDENCE_Z * se_d)),
        math.exp(SMD_TO_LOG_RR * (d + CONFIDENCE_Z * se_d)),
        scale="approximate_risk_ratio_from_standardized_mean_difference",
        standardized_mean_difference=d,
        standardized_mean_difference_standard_error=se_d,
        outcome_sd=sd,
        conversion="RR ~ exp(0.91 * d)",
    )


# ---------------------------------------------------------------------------
# Omitted-variable-bias bounds


@dataclass(frozen=True)
class OVBElements:
    """Estimate, centered influence function and the sensitivity elements."""

    theta: float
    psi: np.ndarray
    sigma2: float
    nu2: float
    psi_sigma2: np.ndarray
    psi_nu2: np.ndarray

    @property
    def rows(self) -> int:
        return int(len(self.psi))

    @property
    def standard_error(self) -> float:
        return float(np.sqrt(np.sum(self.psi ** 2) / (self.rows - 1) / self.rows))


def _sigma2(y: np.ndarray, t: np.ndarray, mu0: np.ndarray, mu1: np.ndarray):
    squared = (y - (t * mu1 + (1.0 - t) * mu0)) ** 2
    sigma2 = float(squared.mean())
    return sigma2, squared - sigma2


def ate_elements(y: Any, t: Any, e: Any, mu0: Any, mu1: Any, *, clip: float) -> OVBElements:
    """ATE: clipped AIPW psi; alpha = D/e - (1-D)/(1-e), m(W, alpha) = 1/e + 1/(1-e)."""

    y, t, e, mu0, mu1 = _clean(y, t, e, mu0, mu1)
    if len(y) < 4:
        raise ValueError("fewer than four rows")
    e = np.clip(e, clip, 1.0 - clip)
    score = mu1 - mu0 + t * (y - mu1) / e - (1.0 - t) * (y - mu0) / (1.0 - e)
    theta = float(score.mean())
    sigma2, psi_sigma2 = _sigma2(y, t, mu0, mu1)
    alpha = t / e - (1.0 - t) / (1.0 - e)
    nu_element = 2.0 * (1.0 / e + 1.0 / (1.0 - e)) - alpha ** 2
    nu2 = float(nu_element.mean())
    return OVBElements(theta, score - theta, sigma2, nu2, psi_sigma2, nu_element - nu2)


def ato_elements(y: Any, t: Any, e: Any, mu0: Any, mu1: Any) -> OVBElements:
    """ATO with fixed overlap weights h = e(1-e): alpha = (D - e)/E[h], m = h/E[h]^2."""

    y, t, e, mu0, mu1 = _clean(y, t, e, mu0, mu1)
    if len(y) < 4:
        raise ValueError("fewer than four rows")
    m1, m0, if1, if0 = overlap_arm_means(y, t, e, mu0, mu1)
    h = e * (1.0 - e)
    mean_h = float(h.mean())
    sigma2, psi_sigma2 = _sigma2(y, t, mu0, mu1)
    alpha = (t - e) / mean_h
    nu_element = 2.0 * h / mean_h ** 2 - alpha ** 2
    nu2 = float(nu_element.mean())
    return OVBElements(m1 - m0, if1 - if0, sigma2, nu2, psi_sigma2, nu_element - nu2)


def _confounding_strength(cf_y: float, cf_d: float, rho: float) -> float:
    if not (0.0 <= cf_y <= 1.0 and 0.0 <= cf_d < 1.0):
        raise ValueError("cf_y must lie in [0, 1] and cf_d in [0, 1)")
    return abs(rho) * math.sqrt(cf_y * cf_d / (1.0 - cf_d))


def ovb_bounds(
    elements: OVBElements, cf_y: float, cf_d: float, *, rho: float = OVB_RHO,
    z: float = CONFIDENCE_Z,
) -> dict[str, Any]:
    """Bounds theta -/+ |rho| sigma nu C_Y C_D and confidence bounds from adjusted scores."""

    if elements.sigma2 <= 0.0 or elements.nu2 <= 0.0:
        raise ValueError("sigma^2 and nu^2 must be positive")
    strength = _confounding_strength(float(cf_y), float(cf_d), float(rho))
    scale = math.sqrt(elements.sigma2 * elements.nu2)
    bias = strength * scale
    psi_bias = strength * (
        elements.sigma2 * elements.psi_nu2 + elements.nu2 * elements.psi_sigma2
    ) / (2.0 * scale)
    n = elements.rows
    se_lower = float(np.sqrt(np.sum((elements.psi - psi_bias) ** 2) / (n - 1) / n))
    se_upper = float(np.sqrt(np.sum((elements.psi + psi_bias) ** 2) / (n - 1) / n))
    lower, upper = elements.theta - bias, elements.theta + bias
    return {
        "cf_y": float(cf_y),
        "cf_d": float(cf_d),
        "rho": float(rho),
        "max_bias": float(bias),
        "lower": float(lower),
        "upper": float(upper),
        "lower_standard_error": se_lower,
        "upper_standard_error": se_upper,
        "lower_confidence_bound": float(lower - z * se_lower),
        "upper_confidence_bound": float(upper + z * se_upper),
    }


def _first_crossing(function, *, monotone: bool) -> float | None:
    """Smallest x in [0, RV_UPPER] with function(x) <= 0, given function(0) > 0."""

    if monotone:
        low, high = 0.0, RV_UPPER
        if function(high) > 0.0:
            return None
    else:
        grid = np.linspace(0.0, RV_UPPER, RV_SCAN_POINTS + 1)
        low = high = None
        for left, right in zip(grid[:-1], grid[1:]):
            if function(right) <= 0.0:
                low, high = float(left), float(right)
                break
        if low is None:
            return None
    for _ in range(100):
        middle = 0.5 * (low + high)
        if function(middle) > 0.0:
            low = middle
        else:
            high = middle
        if high - low < 1e-12:
            break
    return float(high)


def robustness_values(
    elements: OVBElements, *, null: float = 0.0, rho: float = OVB_RHO,
    z: float = CONFIDENCE_Z,
) -> dict[str, Any]:
    """RV (bound reaches ``null``) and RVa (confidence bound reaches ``null``) by bisection."""

    distance = elements.theta - null
    if distance == 0.0:
        return {"robustness_value": 0.0, "robustness_value_confidence": 0.0, "null": null}
    side = "lower" if distance > 0 else "upper"
    sign = 1.0 if distance > 0 else -1.0

    def gap(x: float, key: str) -> float:
        return sign * (ovb_bounds(elements, x, x, rho=rho, z=z)[key] - null)

    rv = _first_crossing(lambda x: gap(x, side), monotone=True)
    confidence_key = f"{side}_confidence_bound"
    if gap(0.0, confidence_key) <= 0.0:
        rva = 0.0
    else:
        rva = _first_crossing(lambda x: gap(x, confidence_key), monotone=False)
    return {"robustness_value": rv, "robustness_value_confidence": rva, "null": null}


def ovb_summary(elements: OVBElements, *, grid: Sequence[float] = OVB_GRID) -> dict[str, Any]:
    """Sensitivity elements, RV/RVa and the (cf_y, cf_d) bound table at rho = 1."""

    if elements.sigma2 <= 0.0 or elements.nu2 <= 0.0:
        return _unavailable(
            "sigma^2 or nu^2 is not positive", sigma2=elements.sigma2, nu2=elements.nu2,
        )
    contour = [ovb_bounds(elements, cf_y, cf_d) for cf_y in grid for cf_d in grid]
    for row in contour:
        row["confidence_bounds_exclude_zero"] = bool(
            row["lower_confidence_bound"] > 0.0 or row["upper_confidence_bound"] < 0.0
        )
    return {
        "status": "ok",
        "rows": elements.rows,
        "estimate": elements.theta,
        "standard_error": elements.standard_error,
        "sigma2": elements.sigma2,
        "nu2": elements.nu2,
        **robustness_values(elements),
        "contour": contour,
    }


# ---------------------------------------------------------------------------
# Benchmark gain statistics


def benchmark_gains(
    long: OVBElements, short: OVBElements, *, outcome_variance: float,
) -> dict[str, Any]:
    """cf_y, cf_d and rho implied by omitting a benchmark variable (DoubleML guide)."""

    if outcome_variance <= 0.0 or long.sigma2 <= 0.0 or long.nu2 <= 0.0 or short.nu2 <= 0.0:
        return _unavailable("outcome variance, sigma^2 or nu^2 is not positive")
    r2_long = 1.0 - long.sigma2 / outcome_variance
    r2_short = 1.0 - short.sigma2 / outcome_variance
    r2_alpha = short.nu2 / long.nu2
    cf_y = float(np.clip((r2_long - r2_short) / (1.0 - r2_long), 0.0, 1.0))
    cf_d = float(np.clip((1.0 - r2_alpha) / r2_alpha, 0.0, 1.0))
    delta = short.theta - long.theta
    var_g, var_alpha = short.sigma2 - long.sigma2, long.nu2 - short.nu2
    rho: float | None = None
    rho_reason = None
    if var_g > 0.0 and var_alpha > 0.0:
        rho = float(np.sign(delta) * min(abs(delta) / math.sqrt(var_g * var_alpha), 1.0))
    else:
        rho_reason = (
            "omitting the variable did not increase sigma^2 and decrease nu^2, so rho is "
            "not identified"
        )
    return {
        "status": "ok",
        "cf_y": cf_y,
        "cf_d": cf_d,
        "rho": rho,
        **({"rho_reason": rho_reason} if rho_reason else {}),
        "r2_outcome_long": float(r2_long),
        "r2_outcome_short": float(r2_short),
        "r2_riesz": float(r2_alpha),
        "theta_long": long.theta,
        "theta_short": short.theta,
        "theta_short_minus_long": float(delta),
        "sigma2_long": long.sigma2,
        "sigma2_short": short.sigma2,
        "nu2_long": long.nu2,
        "nu2_short": short.nu2,
    }


def implied_bounds(elements: OVBElements, gains: Mapping[str, Any]) -> dict[str, Any]:
    """Bounds of an estimate under a benchmark's (cf_y, cf_d), at rho = 1 and its rho."""

    cf_y, cf_d = float(gains["cf_y"]), float(gains["cf_d"])
    if cf_d >= 1.0:
        return {"status": "unbounded", "reason": "cf_d = 1 makes the bound infinite"}
    conservative = ovb_bounds(elements, cf_y, cf_d, rho=1.0)
    result = {
        "status": "ok",
        "rho_1": _pick_bounds(conservative),
    }
    if gains.get("rho") is not None:
        result["benchmark_rho"] = _pick_bounds(
            ovb_bounds(elements, cf_y, cf_d, rho=float(gains["rho"]))
        )
    return result


def _pick_bounds(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = ("rho", "max_bias", "lower", "upper", "lower_confidence_bound",
            "upper_confidence_bound")
    return {key: row[key] for key in keys}


# ---------------------------------------------------------------------------
# Pooled section (reads only frozen predictions and per-fold benchmark files)


def _numeric(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["outcome", "treatment", "propensity", "mu0", "mu1", "aipw_score"]
    return frame[columns].apply(pd.to_numeric, errors="coerce")


def _eligible(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[np.isfinite(frame["aipw_score"].to_numpy(dtype=float))]


def _elements(frame: pd.DataFrame, estimand: str, clip: float) -> OVBElements:
    if estimand == "ate":
        rows = _eligible(frame)
        return ate_elements(rows["outcome"], rows["treatment"], rows["propensity"],
                            rows["mu0"], rows["mu1"], clip=clip)
    return ato_elements(frame["outcome"], frame["treatment"], frame["propensity"],
                        frame["mu0"], frame["mu1"])


def _evalues(frame: pd.DataFrame, estimand: str, *, outcome_type: str, clip: float):
    if estimand == "ate":
        rows = _eligible(frame)
        y, t, e, m0, m1 = _clean(rows["outcome"], rows["treatment"], rows["propensity"],
                                 rows["mu0"], rows["mu1"])
    else:
        y, t, e, m0, m1 = _clean(frame["outcome"], frame["treatment"], frame["propensity"],
                                 frame["mu0"], frame["mu1"])
    if len(y) < 4:
        return _unavailable("fewer than four rows")
    if outcome_type == "binary":
        arms = (
            aipw_arm_means(y, t, e, m0, m1, clip=clip) if estimand == "ate"
            else overlap_arm_means(y, t, e, m0, m1)
        )
        return risk_ratio_evalue(*arms)
    elements = _elements(frame, estimand, clip)
    sd = float(np.std(y, ddof=1))
    return standardized_difference_evalue(elements.theta, elements.standard_error, sd)


def _guarded(function, *args: Any, **kwargs: Any) -> dict[str, Any]:
    """Run one analysis; too little data (ValueError) is unavailable, anything else an error."""

    try:
        return function(*args, **kwargs)
    except ValueError as error:
        return _unavailable(str(error))
    except Exception as error:  # reporting only: never fail a finished run
        LOGGER.warning("Stage 2 sensitivity analysis %s failed",
                       getattr(function, "__name__", "?"), exc_info=True)
        return {"status": "error", "reason": f"{type(error).__name__}: {error}"}


def _benchmark_rows(
    payload: Mapping[str, Any] | str | None, fold_frame: pd.DataFrame,
) -> dict[str, Any]:
    """Validate one fold's benchmark file against its frozen predictions."""

    if payload is None:
        return _unavailable("sensitivity benchmarks were not loaded")
    if isinstance(payload, str):
        return _unavailable(payload)
    status = payload.get("status")
    if status != "ok":
        return {"status": str(status or "unavailable"),
                "reason": str(payload.get("reason") or "benchmarks did not run")}
    if payload.get("schema_version") != BENCHMARK_SCHEMA_VERSION:
        return _unavailable("benchmark file has an unsupported schema version")
    eligible_ids = fold_frame.loc[
        np.isfinite(fold_frame["aipw_score"].to_numpy(dtype=float)), "_oci_row_id"
    ]
    if payload.get("heldout_eligible_row_ids_sha256") != eligible_row_ids_sha256(eligible_ids):
        return _unavailable(
            "benchmark file does not match this fold's effect-eligible held-out rows"
        )
    return dict(payload)


def unmeasured_confounding_section(
    predictions: pd.DataFrame,
    *,
    outcome_type: str,
    propensity_clip: float,
    benchmarks_by_fold: Mapping[str, Mapping[str, Any] | str] | None = None,
) -> dict[str, Any]:
    """E-values, OVB bounds with RV/RVa, and benchmark-implied bounds, per fold and pooled."""

    if outcome_type not in {"binary", "continuous"}:
        return _unavailable("outcome type is unknown")
    numeric = _numeric(predictions)
    frame = pd.concat(
        [predictions[["_oci_row_id", "outer_fold"]].reset_index(drop=True),
         numeric.reset_index(drop=True)], axis=1,
    )
    frame["outer_fold"] = pd.to_numeric(frame["outer_fold"]).astype(int)
    by_fold = {str(fold): part.reset_index(drop=True)
               for fold, part in frame.groupby("outer_fold", sort=True)}
    clip = float(propensity_clip)

    def per_estimand(function) -> dict[str, Any]:
        return {
            estimand: {
                "pooled": _guarded(function, frame, estimand),
                "per_fold": {fold: _guarded(function, part, estimand)
                             for fold, part in by_fold.items()},
            }
            for estimand in ESTIMANDS
        }

    evalues = per_estimand(
        lambda part, estimand: _evalues(part, estimand, outcome_type=outcome_type, clip=clip)
    )
    bounds = per_estimand(lambda part, estimand: ovb_summary(_elements(part, estimand, clip)))
    benchmarks = _guarded(_benchmark_section, frame, by_fold, benchmarks_by_fold or {}, clip)
    return {
        "schema_version": SECTION_SCHEMA_VERSION,
        "status": "ok",
        "role": "reporting only; never read by selection or estimation",
        "outcome_type": outcome_type,
        "interpretation": dict(INTERPRETATIONS),
        "references": list(REFERENCES),
        "evalues": {
            "interpretation": INTERPRETATIONS["evalue"],
            "scale": INTERPRETATIONS[f"evalue_{outcome_type}"],
            **evalues,
        },
        "omitted_variable_bias": {
            "interpretation": INTERPRETATIONS["omitted_variable_bias"],
            "robustness_value": INTERPRETATIONS["robustness_value"],
            "ato": {"population": INTERPRETATIONS["ato_bound"], **bounds["ato"]},
            "ate": bounds["ate"],
            "rho": OVB_RHO,
            "grid": [float(value) for value in OVB_GRID],
            "confidence_bound_critical_value": CONFIDENCE_Z,
            "null": 0.0,
            "riesz_representers": {
                "ate": "alpha = D/e - (1-D)/(1-e) with e clipped to [c, 1-c]; "
                       "m(W, alpha) = 1/e + 1/(1-e); effect-eligible rows",
                "ato": "alpha = h/E[h] (D/e - (1-D)/(1-e)) = (D - e)/E[h] with "
                       "h = e(1-e) held fixed; m(W, alpha) = h/E[h]^2; all held-out rows",
            },
            "pooling": "per fold on its held-out rows; pooled by stacking every fold's "
                       "held-out rows",
        },
        "benchmarks": benchmarks,
    }


def _benchmark_section(
    frame: pd.DataFrame,
    by_fold: Mapping[str, pd.DataFrame],
    benchmarks_by_fold: Mapping[str, Mapping[str, Any] | str],
    clip: float,
) -> dict[str, Any]:
    pooled_elements = _elements(frame, "ate", clip)
    per_fold: dict[str, Any] = {}
    pooled_rows: list[dict[str, Any]] = []
    for fold, part in by_fold.items():
        payload = _benchmark_rows(benchmarks_by_fold.get(fold), part)
        if payload.get("status") != "ok":
            per_fold[fold] = payload
            continue
        fold_elements = _elements(part, "ate", clip)
        groups = []
        for group in payload.get("groups") or []:
            gains = group.get("gains") or {}
            entry = {key: group.get(key) for key in (
                "rank", "feature_id", "name", "selected_roles", "importance",
                "propensity_importance", "outcome_importance", "columns",
            )}
            entry["gains"] = gains
            if gains.get("status") == "ok":
                entry["fold_ate_bounds"] = implied_bounds(fold_elements, gains)
                pooled = implied_bounds(pooled_elements, gains)
                entry["pooled_ate_bounds"] = pooled
                pooled_rows.append({
                    "fold": fold, "name": group.get("name"), "cf_y": gains["cf_y"],
                    "cf_d": gains["cf_d"], "rho": gains.get("rho"), "bounds": pooled,
                    "statement": _statement(group.get("name"), fold, pooled),
                })
            groups.append(entry)
        per_fold[fold] = {
            "status": "ok",
            "requested_groups": payload.get("requested_groups"),
            "ranked_groups": payload.get("ranked_groups"),
            "learner": payload.get("learner"),
            "importance": payload.get("importance"),
            "long_reference": payload.get("long_reference"),
            "fold_ate": {"estimate": fold_elements.theta, "sigma2": fold_elements.sigma2,
                         "nu2": fold_elements.nu2},
            "groups": groups,
        }
    statuses = {value.get("status") for value in per_fold.values()}
    pooled_rows.sort(key=lambda row: -_max_bias(row["bounds"]))
    return {
        "status": "ok" if "ok" in statuses else (
            "disabled" if statuses == {"disabled"} else "unavailable"),
        "interpretation": INTERPRETATIONS["benchmarks"],
        "estimand": "ate",
        "per_fold": per_fold,
        "pooled": {
            "applied_to": "the pooled ATE (each fold's benchmark gains applied to the "
                          "pooled sensitivity elements)",
            "estimate": pooled_elements.theta,
            "rows": pooled_rows,
        },
    }


def _max_bias(bounds: Mapping[str, Any]) -> float:
    value = (bounds.get("rho_1") or {}).get("max_bias")
    return float(value) if value is not None else math.inf


def _statement(name: Any, fold: str, bounds: Mapping[str, Any]) -> str:
    if bounds.get("status") != "ok":
        return (f"A confounder as strong as {name} (fold {fold}) would make the pooled ATE "
                "bound unbounded.")
    rho_1 = bounds["rho_1"]
    return (
        f"A confounder as strong as {name} (fold {fold}) would move the pooled ATE bound "
        f"to [{rho_1['lower']:.4g}, {rho_1['upper']:.4g}] at rho = 1 (95% confidence "
        f"bounds [{rho_1['lower_confidence_bound']:.4g}, "
        f"{rho_1['upper_confidence_bound']:.4g}])."
    )


def summarize_section(section: Any) -> dict[str, Any]:
    """Compact pooled summary for ``causal_estimate.json``."""

    section = section if isinstance(section, Mapping) else {}
    if section.get("status") != "ok":
        return {key: section.get(key) for key in ("status", "reason") if key in section}
    evalues = section.get("evalues") or {}
    bounds = section.get("omitted_variable_bias") or {}
    summary: dict[str, Any] = {"status": "ok"}
    for estimand in ESTIMANDS:
        ev = (evalues.get(estimand) or {}).get("pooled") or {}
        ovb = (bounds.get(estimand) or {}).get("pooled") or {}
        summary[estimand] = {
            "risk_ratio": ev.get("risk_ratio"),
            "risk_ratio_confidence_interval_95": ev.get("risk_ratio_confidence_interval_95"),
            "evalue_point": ev.get("evalue_point"),
            "evalue_confidence_limit": ev.get("evalue_confidence_limit"),
            "evalue_status": ev.get("status"),
            "evalue_scale": ev.get("scale"),
            "robustness_value": ovb.get("robustness_value"),
            "robustness_value_confidence": ovb.get("robustness_value_confidence"),
            "bound_status": ovb.get("status"),
        }
    benchmarks = section.get("benchmarks") or {}
    rows = (benchmarks.get("pooled") or {}).get("rows") or []
    summary["benchmarks"] = {
        "status": benchmarks.get("status"),
        "benchmarks": len(rows),
        "strongest": (
            {key: rows[0].get(key) for key in ("fold", "name", "cf_y", "cf_d", "rho",
                                               "statement")}
            if rows else None
        ),
    }
    return summary


# ---------------------------------------------------------------------------
# Fold-level benchmark refits (the only part that fits models)


def _coefficients(model: Any) -> np.ndarray | None:
    inner = getattr(model, "model_", None)
    coef = getattr(inner, "coef_", None)
    return None if coef is None else np.asarray(coef, dtype=float).reshape(-1)


def _benchmark_fit(x_train, target, x_heldout, *, binary: bool, seed: int) -> dict[str, Any]:
    """One elastic-net nuisance fit on outer-training rows, scored on held-out rows."""

    from .plain_handoff_stage2_analysis import (
        _fit_classifier,
        _fit_regressor,
        _predict_probability,
    )

    model = (
        _fit_classifier(x_train, target, seed=seed) if binary
        else _fit_regressor(x_train, target, seed=seed)
    )
    prediction = (
        _predict_probability(model, x_heldout) if binary
        else np.asarray(model.predict(x_heldout), dtype=float)
    )
    coefficients = _coefficients(model)
    return {
        "prediction": [float(value) for value in prediction],
        "coefficients": (
            [float(value) for value in coefficients]
            if coefficients is not None else [0.0] * int(np.asarray(x_train).shape[1])
        ),
        "fit_mode": str(getattr(model, "fit_mode_", type(model).__name__)),
    }


def _fit_set(
    directory: Path, name: str, fingerprint: str, design_train, design_heldout,
    treatment, outcome, *, binary: bool, seed: int,
):
    """Submit the propensity and both arm outcome fits of one design."""

    from functools import partial
    from .stage2_multi_model_selection import checkpoint_future

    futures = {"propensity": checkpoint_future(
        directory, f"{name}/propensity", fingerprint,
        partial(_benchmark_fit, design_train, treatment, design_heldout, binary=True,
                seed=seed),
    )}
    for arm in (0, 1):
        mask = treatment.astype(int) == arm
        if not mask.any():
            mask = np.ones(len(treatment), dtype=bool)
        futures[f"mu{arm}"] = checkpoint_future(
            directory, f"{name}/mu{arm}", fingerprint,
            partial(_benchmark_fit, design_train[mask], outcome[mask], design_heldout,
                    binary=binary, seed=seed + 1 + arm),
        )
    return futures


def _results(futures: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {key: future.result() for key, future in futures.items()}


def _group_norm(coefficients: Sequence[float], columns: slice) -> float:
    return float(np.linalg.norm(np.asarray(coefficients, dtype=float)[columns]))


def run_fold_benchmarks(
    *,
    dataset: pd.DataFrame,
    candidate_definitions: Sequence[Mapping[str, Any]],
    candidate_fit: pd.DataFrame,
    candidate_heldout: pd.DataFrame,
    selected_definitions: Sequence[Mapping[str, Any]],
    split: Mapping[str, Any],
    treatment_column: str,
    outcome_column: str,
    outcome_type: str,
    seed: int,
    propensity_clip: float,
    groups: int,
    estimation_dir: Path,
) -> dict[str, Any]:
    """Benchmark omitted-variable gains for one outer fold; writes ``benchmarks.json``.

    Runs after the fold's predictions are frozen. Training-row fits only;
    held-out labels are read to evaluate them. Reporting only.
    """

    from .plain_handoff_stage2_analysis import (
        _FeatureEncoder,
        _aligned_measurement_frame,
        _frame_fingerprint,
        _write_json,
    )

    root = Path(estimation_dir) / BENCHMARK_DIRNAME
    path = root / BENCHMARK_FILENAME
    base = {"schema_version": BENCHMARK_SCHEMA_VERSION, "requested_groups": int(groups)}
    if int(groups) == 0:
        result = {**base, "status": "disabled",
                  "reason": "stage2.sensitivity.benchmark_groups is 0"}
        _write_json(path, result)
        return result
    fit_ids = [int(value) for value in split["fit_row_ids"]]
    heldout_ids = [int(value) for value in split["heldout_row_ids"]]
    predictions = pd.read_csv(Path(estimation_dir) / "predictions.csv")
    if [int(value) for value in predictions["_oci_row_id"]] != heldout_ids:
        raise ValueError("fold predictions do not follow the held-out row order")
    eligible = np.isfinite(pd.to_numeric(predictions["aipw_score"], errors="coerce")
                           .to_numpy(dtype=float))
    eligible_ids = [row for row, keep in zip(heldout_ids, eligible) if keep]
    definitions = [dict(feature) for feature in candidate_definitions]
    if not definitions:
        result = {**base, "status": "unavailable", "reason": "no nuisance candidates"}
        _write_json(path, result)
        return result
    train = _aligned_measurement_frame(candidate_fit, fit_ids, definitions, label="training")
    heldout = _aligned_measurement_frame(
        candidate_heldout, heldout_ids, definitions, label="held-out"
    )
    labels = dataset[[treatment_column, outcome_column]]
    t_fit = labels.iloc[fit_ids][treatment_column].to_numpy(dtype=float)
    y_fit = labels.iloc[fit_ids][outcome_column].to_numpy(dtype=float)
    binary = str(outcome_type) == "binary"
    source = {
        name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
        for name in ("stage2_sensitivity.py", "plain_handoff_stage2_analysis.py",
                     "../models/elastic_net_nuisance.py")
    }
    fingerprint = _sha256({
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "groups": int(groups),
        "definitions": definitions,
        "fit_row_ids": fit_ids,
        "heldout_row_ids": heldout_ids,
        "eligible_row_ids": eligible_ids,
        "training_measurements": _frame_fingerprint(train),
        "heldout_measurements": _frame_fingerprint(heldout),
        "training_labels_sha256": hashlib.sha256(
            np.ascontiguousarray(np.vstack((t_fit, y_fit))).tobytes()
        ).hexdigest(),
        "outcome_type": str(outcome_type),
        "seed": int(seed),
        "propensity_clip": float(propensity_clip),
        "sources": source,
    })
    if path.is_file():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            if cached.get("input_fingerprint") == fingerprint and cached.get("status") == "ok":
                return cached
        except (OSError, ValueError):
            pass

    blocks, train_parts, heldout_parts, start = [], [], [], 0
    for feature in definitions:
        encoder = _FeatureEncoder([feature]).fit(train)
        x_train, x_heldout = encoder.transform(train), encoder.transform(heldout)
        width = int(x_train.shape[1])
        blocks.append((feature, slice(start, start + width), width))
        train_parts.append(x_train)
        heldout_parts.append(x_heldout)
        start += width
    design_train = np.hstack(train_parts) if train_parts else np.zeros((len(train), 0))
    design_heldout = np.hstack(heldout_parts) if heldout_parts else np.zeros((len(heldout), 0))
    fits_dir = root / "fits" / fingerprint[:20]
    fit_seed = int(seed) + BENCHMARK_SEED_OFFSET
    long_fits = _results(_fit_set(
        fits_dir, "long", fingerprint, design_train, design_heldout, t_fit, y_fit,
        binary=binary, seed=fit_seed,
    ))

    importance = []
    for feature, columns, width in blocks:
        if width == 0:
            continue
        treatment_norm = _group_norm(long_fits["propensity"]["coefficients"], columns)
        outcome_norm = math.hypot(
            _group_norm(long_fits["mu0"]["coefficients"], columns),
            _group_norm(long_fits["mu1"]["coefficients"], columns),
        )
        importance.append({
            "feature": feature, "columns": columns, "width": width,
            "propensity_importance": treatment_norm, "outcome_importance": outcome_norm,
            "importance": treatment_norm * outcome_norm,
        })
    ranked = sorted(
        (item for item in importance if item["importance"] > 0.0),
        key=lambda item: (-item["importance"],
                          -(item["propensity_importance"] + item["outcome_importance"]),
                          str(item["feature"]["name"])),
    )[: int(groups)]

    t_heldout = labels.iloc[heldout_ids][treatment_column].to_numpy(dtype=float)
    y_heldout = labels.iloc[heldout_ids][outcome_column].to_numpy(dtype=float)

    def elements(fits: Mapping[str, Mapping[str, Any]]) -> OVBElements:
        e, mu0, mu1 = (np.asarray(fits[key]["prediction"], dtype=float)[eligible]
                       for key in ("propensity", "mu0", "mu1"))
        return ate_elements(y_heldout[eligible], t_heldout[eligible], e, mu0, mu1,
                            clip=float(propensity_clip))

    long_elements = elements(long_fits)
    outcome_variance = float(np.var(y_heldout[eligible]))
    pending = []
    for item in ranked:
        keep = np.ones(design_train.shape[1], dtype=bool)
        keep[item["columns"]] = False
        tag = hashlib.sha256(str(item["feature"]["feature_id"]).encode()).hexdigest()[:12]
        pending.append((item, _fit_set(
            fits_dir, f"short_{tag}", fingerprint, design_train[:, keep],
            design_heldout[:, keep], t_fit, y_fit, binary=binary, seed=fit_seed,
        )))
    roles = {
        str(feature.get("feature_id")): [str(role) for role in feature.get("roles") or []]
        for feature in selected_definitions
    }
    group_results = []
    for rank, (item, futures) in enumerate(pending, start=1):
        feature = item["feature"]
        short = elements(_results(futures))
        group_results.append({
            "rank": rank,
            "feature_id": str(feature.get("feature_id")),
            "name": str(feature.get("name")),
            "selected_roles": roles.get(str(feature.get("feature_id")), []),
            "columns": item["width"],
            "importance": item["importance"],
            "propensity_importance": item["propensity_importance"],
            "outcome_importance": item["outcome_importance"],
            "gains": benchmark_gains(long_elements, short, outcome_variance=outcome_variance),
        })
    result = {
        **base,
        "status": "ok",
        "input_fingerprint": fingerprint,
        "heldout_eligible_row_ids_sha256": eligible_row_ids_sha256(eligible_ids),
        "estimand": "ate",
        "learner": (
            "elastic net (oci.models.elastic_net_nuisance) for the long reference and every "
            "short model: one fit per target on all outer-training rows, raw held-out "
            "predictions, propensity clipped at the configured clip"
        ),
        "importance": (
            "product of the L2 norms of the variable's standardized elastic-net coefficients "
            "in the long reference's propensity model and in its two arm outcome models "
            "(the outcome norm combines both arms); ties broken by the sum, then the name; "
            "variables with zero product are not benchmarked"
        ),
        "candidates": len(definitions),
        "ranked_groups": len(ranked),
        "long_reference": {
            "estimate": long_elements.theta, "sigma2": long_elements.sigma2,
            "nu2": long_elements.nu2, "rows": long_elements.rows,
            "fit_modes": {key: value["fit_mode"] for key, value in long_fits.items()},
        },
        "groups": group_results,
    }
    _write_json(path, result)
    return result


def benchmark_fold(*, config: Any, estimation_dir: Path, **inputs: Any) -> dict[str, Any]:
    """:func:`run_fold_benchmarks` with the configured group count; records failures.

    Reporting only: an error is written to ``benchmarks.json`` and logged, never
    raised, so it cannot change or stop the fold's estimation.
    """

    sensitivity = getattr(config, "sensitivity", None)
    groups = getattr(sensitivity, "benchmark_groups", DEFAULT_BENCHMARK_GROUPS)
    try:
        return run_fold_benchmarks(groups=groups, estimation_dir=estimation_dir, **inputs)
    except Exception as error:  # reporting only: never fail the fold
        LOGGER.warning("Stage 2 sensitivity benchmarks failed in %s", estimation_dir,
                       exc_info=True)
        result = {
            "schema_version": BENCHMARK_SCHEMA_VERSION,
            "status": "error",
            "reason": f"{type(error).__name__}: {error}",
            "requested_groups": groups,
        }
        try:
            from .plain_handoff_stage2_analysis import _write_json

            _write_json(Path(estimation_dir) / BENCHMARK_DIRNAME / BENCHMARK_FILENAME, result)
        except Exception:  # the diagnostics then report the file as missing
            LOGGER.warning("could not record the benchmark failure", exc_info=True)
        return result


def load_fold_benchmarks(fold_dir: Path) -> Mapping[str, Any] | str:
    """Read one fold's ``benchmarks.json``; a reason string if it is absent or unreadable."""

    path = Path(fold_dir) / "estimation" / BENCHMARK_DIRNAME / BENCHMARK_FILENAME
    if not path.is_file():
        return f"sensitivity benchmarks unavailable: {path} is missing"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return f"sensitivity benchmarks unreadable: {error}"
    return value if isinstance(value, Mapping) else "sensitivity benchmarks are not an object"


__all__ = [
    "BENCHMARK_FILENAME",
    "DEFAULT_BENCHMARK_GROUPS",
    "MAX_BENCHMARK_GROUPS",
    "OVBElements",
    "SensitivityConfig",
    "OVB_GRID",
    "REFERENCES",
    "aipw_arm_means",
    "ate_elements",
    "ato_elements",
    "benchmark_fold",
    "benchmark_gains",
    "evalue",
    "evalue_confidence_limit",
    "implied_bounds",
    "load_fold_benchmarks",
    "overlap_arm_means",
    "ovb_bounds",
    "ovb_summary",
    "risk_ratio_evalue",
    "robustness_values",
    "run_fold_benchmarks",
    "sensitivity_config_from_mapping",
    "standardized_difference_evalue",
    "summarize_section",
    "unmeasured_confounding_section",
]
