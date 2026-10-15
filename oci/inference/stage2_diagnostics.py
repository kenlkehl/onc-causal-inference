"""Oracle-free heterogeneity, overlap and balance diagnostics for Stage 2.

The diagnostics run after every outer fold's predictions are frozen. They read
only the pooled cross-fitted prediction table (held-out AIPW score ``psi``,
``estimated_cate``, ``propensity``, ``mu0``, ``mu1``, ``treatment`` and the
observed ``outcome``) and persisted per-fold measurement artifacts. They never
read the dataset, so oracle columns cannot enter; columns and features with
oracle-style names are dropped as a second guard.

Diagnostics (all reported per outer fold and pooled):

* BLP (Chernozhukov, Demirer, Duflo and Fernandez-Val): OLS of ``psi`` on
  ``[1, tau_hat - mean(tau_hat)]`` with HC3 standard errors.
* GATES: mean ``psi`` within quintiles of ``tau_hat`` formed inside each fold.
* CLAN: top- versus bottom-quintile means of the retained effect modifiers and
  confounders, measured on the fold's held-out patients.
* RATE (Yadlowsky et al.): AUTOC and Qini of the ``tau_hat`` ranking with
  seeded bootstrap standard errors.
* Overlap: overlap-weighted ATE (Li, Morgan and Zaslavsky; Hajek and augmented)
  with its effective sample size, inverse-propensity weight summaries (Kish ESS,
  maximum weight) and an AIPW trimming curve (Crump et al.).
* Balance: IPW-weighted standardized mean differences of every measured
  candidate, with a summary of variables whose ``|SMD| > 0.1``.

Pooling rules (also written into the output under ``pooling_rules``):

* BLP coefficients are pooled across folds by fixed-effect inverse-variance
  weighting. The outer folds have disjoint held-out rows, so the fold estimates
  are independent given the training data; the Chernozhukov et al. median
  adjustment targets repeated random splits of one sample and is not used. The
  median of the fold estimates is reported alongside as a robustness check.
* Means over patients (GATES, CLAN, overlap-weighted ATE, trimming curve,
  IPW weight summaries, held-out balance) are recomputed on the pooled held-out
  rows. Quintile groups are always formed within a fold, because ``tau_hat``
  comes from a different fitted model in each fold.
* RATE is recomputed on the pooled held-out rows, ranking patients by their
  within-fold percentile of ``tau_hat``; bootstrap resampling is stratified by
  fold.
* Training-row balance is per fold only (training sets overlap across folds);
  the pooled entry summarizes the fold tables.
* Folds whose ``tau_hat`` is constant cannot rank patients. Their BLP slope is
  not identified, and they are left out of the pooled GATES, CLAN and RATE.

Every function below the I/O layer is pure. The I/O layer
(:func:`write_heterogeneity_diagnostics`) never raises for a missing input: the
affected diagnostic is marked ``unavailable`` with a reason, and an unexpected
error in one diagnostic is recorded as ``status: "error"`` without failing the
finished Stage 2 run.
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

LOGGER = logging.getLogger(__name__)

SCHEMA_VERSION = "stage2_heterogeneity_diagnostics_v1"
DIAGNOSTICS_FILENAME = "heterogeneity_diagnostics.json"
TRIMMING_THRESHOLDS = (0.0, 0.01, 0.025, 0.05, 0.1, 0.15)
SMD_THRESHOLD = 0.1
CATE_GROUPS = 5
DEFAULT_BOOTSTRAP_REPLICATES = 500
MAX_CATEGORY_LEVELS = 20
NOT_DOCUMENTED_LEVEL = "not_documented"
EXTRACTION_FAILED_LEVEL = "extraction_failed"
# Companion 0/1 column marking cells whose extraction failed after all repairs
# (the value cell is NaN but is not "not documented").
EXTRACTION_FAILURE_SUFFIX = "__oci_extraction_failed"
OTHER_LEVEL = "other"
BOOTSTRAP_SEED_OFFSET = 50_000
# Mirrors ``oci.inference.applied.ORACLE_NAME_PREFIXES`` (not imported: that
# module pulls in torch).
ORACLE_NAME_PREFIXES = ("true_", "oracle_", "ground_truth")
CLAN_ROLES = frozenset({"effect_modifier", "confounder"})
ROW_ID = "_oci_row_id"

INTERPRETATIONS = {
    "blp": (
        "Best linear predictor of the held-out AIPW score psi on "
        "[1, tau_hat - mean(tau_hat)]. beta1 estimates the ATE among "
        "effect-eligible rows. beta2 tests whether the CATE ranking carries "
        "real heterogeneity: beta2 > 0 with a small one-sided p-value is "
        "evidence that tau_hat captures heterogeneity; beta2 near 1 means "
        "tau_hat is calibrated in scale; beta2 near 0 means no detectable "
        "heterogeneity signal; 0 < beta2 < 1 suggests the CATEs are too "
        "spread out, beta2 > 1 that they are too compressed."
    ),
    "gates": (
        "Group average treatment effects: mean psi within quintiles of tau_hat "
        "formed inside each outer fold (G1 lowest, G5 highest predicted "
        "effect). An increasing pattern, a positive linear trend and a "
        "positive top-minus-bottom difference indicate that the ranking "
        "separates patients with larger and smaller effects; similar group "
        "means indicate no detectable heterogeneity."
    ),
    "clan": (
        "Classification analysis: mean of each retained effect modifier and "
        "confounder (held-out measurements; continuous values or level "
        "indicators) in the top versus bottom tau_hat quintile. Large "
        "differences describe which measured characteristics distinguish "
        "predicted high- and low-effect patients. Descriptive, not causal, and "
        "not adjusted for multiple comparisons."
    ),
    "rate": (
        "Rank-weighted average treatment effects for prioritizing patients by "
        "tau_hat, computed from psi: TOC(u) is the ATE among the top u "
        "fraction minus the overall ATE. AUTOC weights every u equally "
        "(sensitive to the top of the ranking); Qini weights by u (more power "
        "for diffuse heterogeneity). Positive values whose confidence interval "
        "excludes 0 indicate useful prioritization; values near 0 indicate "
        "none."
    ),
    "overlap_weighted_ate": (
        "ATE in the overlap population (ATO; weights e(1-e)), which emphasizes "
        "patients in clinical equipoise. Hajek uses weighted outcome means "
        "only; augmented adds the cross-fitted outcome models. A large gap "
        "between the ATO and the AIPW ATE means the ATE depends on patients "
        "with extreme propensities, or that effects differ in the overlap "
        "region. The effective sample size is that of the overlap weights."
    ),
    "ipw_weights": (
        "Inverse-propensity weights 1/e (treated) and 1/(1-e) (controls) as "
        "used by the AIPW scores, with e clipped at propensity_clip. The Kish "
        "effective sample size (sum w)^2 / sum w^2 and the maximum weight show "
        "how much the AIPW ATE leans on a few patients; an ESS fraction well "
        "below 1 or a very large maximum weight signals weak overlap and high "
        "variance."
    ),
    "trimming_curve": (
        "AIPW ATE restricted to rows with alpha <= e <= 1 - alpha, using psi "
        "recomputed from the cross-fitted nuisances with the configured clip. "
        "alpha = 0 is the clipped, untrimmed ATE over all held-out rows "
        "(including rows outside configured eligibility bounds). Each alpha "
        "targets a different population; a stable curve suggests the ATE is "
        "not driven by poor-overlap patients, while drift signals dependence "
        "on extreme propensities."
    ),
    "balance": (
        "Standardized mean differences (treated minus control mean divided by "
        "the pooled unweighted SD) of measured candidates before and after "
        "inverse-propensity weighting. Training: each outer fold's training "
        "rows weighted by their inner-fold out-of-fold e. Held-out: outer "
        "held-out rows weighted by the outer-training e. A weighted |SMD| > "
        "0.1 flags a variable the propensity model does not balance, i.e. a "
        "likely miss in nuisance selection or a misspecified propensity model."
    ),
}

POOLING_RULES = {
    "blp": (
        "fixed-effect inverse-variance weighting of fold estimates (disjoint "
        "held-out rows); fold median reported as a robustness check; folds "
        "with constant tau_hat excluded"
    ),
    "gates": (
        "pooled held-out rows grouped by their within-fold tau_hat quintile; "
        "folds with constant tau_hat excluded"
    ),
    "clan": (
        "pooled held-out rows grouped by their within-fold tau_hat quintile; "
        "each variable uses the folds in which it was retained and measured"
    ),
    "rate": (
        "pooled held-out rows ranked by within-fold tau_hat percentile; "
        "bootstrap stratified by fold; folds with constant tau_hat excluded"
    ),
    "overlap": "recomputed on pooled held-out rows",
    "balance": (
        "training: per fold only (training sets overlap), pooled entry "
        "summarizes fold tables; held-out: recomputed on pooled held-out rows"
    ),
}


# ---------------------------------------------------------------------------
# Small numeric helpers


def _float(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _p_two_sided(z: float | None) -> float | None:
    return None if z is None else float(2.0 * norm.sf(abs(z)))


def _p_greater(z: float | None) -> float | None:
    return None if z is None else float(norm.sf(z))


def _interval(estimate: float | None, se: float | None) -> list[float] | None:
    if estimate is None or se is None:
        return None
    return [float(estimate - 1.96 * se), float(estimate + 1.96 * se)]


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return _float(numerator / denominator)


def _mean_with_se(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = _float(values.mean()) if n else None
    se = _float(values.std(ddof=1) / math.sqrt(n)) if n > 1 else None
    return {
        "rows": n,
        "estimate": mean,
        "standard_error": se,
        "confidence_interval_95": _interval(mean, se),
    }


def _is_constant(values: np.ndarray) -> bool:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return True
    scale = max(1.0, float(np.max(np.abs(values))))
    return float(np.ptp(values)) <= 1e-12 * scale


def _kish(weights: np.ndarray) -> float | None:
    weights = np.asarray(weights, dtype=float)
    denominator = float(np.sum(weights**2))
    return _float(np.sum(weights) ** 2 / denominator) if denominator > 0 else None


def _fold_key(fold: Any) -> tuple[int, int, str]:
    """Sort fold labels numerically ("2" before "10")."""

    text = str(fold)
    return (0, int(text), text) if text.isdigit() else (1, 0, text)


def unavailable(reason: str, **extra: Any) -> dict[str, Any]:
    """Return the standard marker for a diagnostic that cannot be computed."""

    return {"status": "unavailable", "reason": reason, **extra}


def is_oracle_name(name: Any) -> bool:
    return str(name).strip().lower().startswith(ORACLE_NAME_PREFIXES)


def json_safe(value: Any) -> Any:
    """Convert numpy scalars/arrays and non-finite floats to strict JSON values."""

    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return _float(value)
    return value


# ---------------------------------------------------------------------------
# Scores


def dr_scores(
    outcome: Any,
    treatment: Any,
    propensity: Any,
    mu0: Any,
    mu1: Any,
    *,
    clip: float,
) -> np.ndarray:
    """AIPW scores with clipped propensities (same formula as Stage 2 ``_dr_score``)."""

    y, t = np.asarray(outcome, dtype=float), np.asarray(treatment, dtype=float)
    m0, m1 = np.asarray(mu0, dtype=float), np.asarray(mu1, dtype=float)
    e = np.clip(np.asarray(propensity, dtype=float), clip, 1.0 - clip)
    with np.errstate(divide="ignore", invalid="ignore"):
        score = m1 - m0 + t * (y - m1) / e - (1.0 - t) * (y - m0) / (1.0 - e)
    return np.where(np.isfinite(score), score, np.nan)


# ---------------------------------------------------------------------------
# BLP


def best_linear_predictor(score: Any, cate: Any) -> dict[str, Any]:
    """Regress psi on [1, tau_hat - mean(tau_hat)] with HC3 standard errors."""

    import statsmodels.api as sm

    score, cate = np.asarray(score, dtype=float), np.asarray(cate, dtype=float)
    finite = np.isfinite(score) & np.isfinite(cate)
    score, cate = score[finite], cate[finite]
    n = len(score)
    if n < 4:
        return unavailable("fewer than 4 rows with finite aipw_score and estimated_cate", rows=n)
    if _is_constant(cate):
        return unavailable(
            "estimated_cate is constant, so the heterogeneity slope beta2 is not identified",
            rows=n,
            beta1_ate=_float(score.mean()),
            beta1_standard_error=_float(score.std(ddof=1) / math.sqrt(n)),
        )
    design = np.column_stack([np.ones(n), cate - cate.mean()])
    fit = sm.OLS(score, design).fit(cov_type="HC3")
    beta1, beta2 = (float(value) for value in fit.params)
    se1, se2 = (_float(value) for value in fit.bse)
    z2 = _ratio(beta2, se2) if se2 else None
    z2_one = _float((beta2 - 1.0) / se2) if se2 else None
    return {
        "status": "ok",
        "rows": n,
        "estimated_cate_mean": float(cate.mean()),
        "estimated_cate_sd": float(cate.std(ddof=1)),
        "beta1_ate": beta1,
        "beta1_standard_error": se1,
        "beta1_confidence_interval_95": _interval(beta1, se1),
        "beta2_heterogeneity": beta2,
        "beta2_standard_error": se2,
        "beta2_confidence_interval_95": _interval(beta2, se2),
        "beta2_z": z2,
        "beta2_p_value_one_sided_greater_than_0": _p_greater(z2),
        "beta2_p_value_two_sided_equal_1": _p_two_sided(z2_one),
        "covariance": "HC3",
    }


def inverse_variance_pool(estimates: Mapping[Any, tuple[Any, Any]]) -> dict[str, Any]:
    """Fixed-effect inverse-variance pooling of per-fold (estimate, SE) pairs."""

    usable = {
        str(fold): (float(estimate), float(se))
        for fold, (estimate, se) in estimates.items()
        if _float(estimate) is not None and _float(se) is not None and float(se) > 0
    }
    excluded = sorted((str(fold) for fold in estimates if str(fold) not in usable), key=_fold_key)
    if not usable:
        return unavailable(
            "no fold has a finite estimate with a positive standard error",
            excluded_folds=excluded,
        )
    values = np.array([pair[0] for pair in usable.values()])
    weights = np.array([1.0 / pair[1] ** 2 for pair in usable.values()])
    estimate = float(np.sum(weights * values) / np.sum(weights))
    se = float(1.0 / math.sqrt(np.sum(weights)))
    return {
        "status": "ok",
        "estimate": estimate,
        "standard_error": se,
        "confidence_interval_95": _interval(estimate, se),
        "z": estimate / se,
        "folds": sorted(usable, key=_fold_key),
        "excluded_folds": excluded,
        "fold_median": float(np.median(values)),
    }


def pool_blp(per_fold: Mapping[Any, Mapping[str, Any]]) -> dict[str, Any]:
    ok = {fold: result for fold, result in per_fold.items() if result.get("status") == "ok"}
    beta1 = inverse_variance_pool(
        {f: (r["beta1_ate"], r["beta1_standard_error"]) for f, r in ok.items()}
    )
    beta2 = inverse_variance_pool(
        {f: (r["beta2_heterogeneity"], r["beta2_standard_error"]) for f, r in ok.items()}
    )
    if beta2.get("status") != "ok":
        return unavailable(
            "no fold produced an identified BLP slope",
            folds_without_identified_slope=sorted(map(str, per_fold), key=_fold_key),
        )
    z_one = (beta2["estimate"] - 1.0) / beta2["standard_error"]
    return {
        "status": "ok",
        "method": "fixed_effect_inverse_variance",
        "folds": beta2["folds"],
        "excluded_folds": sorted((str(f) for f in per_fold if f not in ok), key=_fold_key),
        "beta1_ate": beta1.get("estimate"),
        "beta1_standard_error": beta1.get("standard_error"),
        "beta1_confidence_interval_95": beta1.get("confidence_interval_95"),
        "beta2_heterogeneity": beta2["estimate"],
        "beta2_standard_error": beta2["standard_error"],
        "beta2_confidence_interval_95": beta2["confidence_interval_95"],
        "beta2_z": beta2["z"],
        "beta2_p_value_one_sided_greater_than_0": _p_greater(beta2["z"]),
        "beta2_p_value_two_sided_equal_1": _p_two_sided(z_one),
        "beta2_fold_median": beta2["fold_median"],
    }


# ---------------------------------------------------------------------------
# GATES and CLAN


def cate_groups(cate: Any, *, groups: int = CATE_GROUPS, seed: int = 0) -> np.ndarray:
    """Assign 1..groups by ascending tau_hat (equal-size groups, seeded tie-break)."""

    cate = np.asarray(cate, dtype=float)
    n = len(cate)
    if n < groups:
        raise ValueError(f"need at least {groups} rows to form {groups} groups")
    tie_break = np.random.default_rng(seed).random(n)
    order = np.lexsort((tie_break, cate))
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(n)
    return ranks * groups // n + 1


def _ols_slope_hc3(score: np.ndarray, regressor: np.ndarray) -> dict[str, Any]:
    import statsmodels.api as sm

    fit = sm.OLS(score, np.column_stack([np.ones(len(score)), regressor])).fit(cov_type="HC3")
    slope, se = float(fit.params[1]), _float(fit.bse[1])
    z = _ratio(slope, se) if se else None
    return {
        "estimate": slope,
        "standard_error": se,
        "confidence_interval_95": _interval(slope, se),
        "p_value_one_sided_greater_than_0": _p_greater(z),
    }


def group_average_treatment_effects(
    score: Any, cate: Any, group: Any, *, groups: int = CATE_GROUPS
) -> dict[str, Any]:
    """Mean psi per tau_hat group, top-minus-bottom and a linear trend across groups."""

    score, cate = np.asarray(score, dtype=float), np.asarray(cate, dtype=float)
    group = np.asarray(group, dtype=int)
    rows = []
    for label in range(1, groups + 1):
        mask = group == label
        rows.append(
            {
                "group": label,
                "mean_estimated_cate": _float(cate[mask].mean()) if mask.any() else None,
                **_mean_with_se(score[mask]),
            }
        )
    top, bottom = rows[-1], rows[0]
    if top["rows"] < 2 or bottom["rows"] < 2:
        return unavailable("top or bottom group has fewer than 2 rows", groups=rows)
    difference = top["estimate"] - bottom["estimate"]
    se = _float(math.hypot(top["standard_error"] or 0.0, bottom["standard_error"] or 0.0))
    z = _ratio(difference, se) if se else None
    return {
        "status": "ok",
        "rows": len(score),
        "groups": rows,
        "top_minus_bottom": {
            "estimate": float(difference),
            "standard_error": se,
            "confidence_interval_95": _interval(difference, se),
            "p_value_two_sided": _p_two_sided(z),
            "p_value_one_sided_greater_than_0": _p_greater(z),
        },
        "linear_trend_per_group": {
            **_ols_slope_hc3(score, group.astype(float)),
            "covariance": "HC3",
        },
    }


def classification_analysis(
    encoded: pd.DataFrame,
    columns: Sequence[Mapping[str, Any]],
    group: Any,
    *,
    groups: int = CATE_GROUPS,
) -> dict[str, Any]:
    """Top- vs bottom-group means of encoded measurements (Welch-type SEs)."""

    group = np.asarray(group, dtype=int)
    top, bottom = group == groups, group == 1
    variables = []
    for meta in columns:
        values = encoded[meta["column"]].to_numpy(dtype=float)
        finite = np.isfinite(values)
        high, low = values[top & finite], values[bottom & finite]
        entry: dict[str, Any] = {
            **{key: meta[key] for key in ("column", "feature", "level", "kind")},
            "roles": list(meta.get("roles") or []),
            "top_rows": len(high),
            "bottom_rows": len(low),
        }
        if len(high) < 2 or len(low) < 2:
            variables.append({**entry, "status": "insufficient_rows"})
            continue
        difference = float(high.mean() - low.mean())
        se = _float(math.sqrt(high.var(ddof=1) / len(high) + low.var(ddof=1) / len(low)))
        z = _ratio(difference, se) if se else None
        variables.append(
            {
                **entry,
                "status": "ok",
                "top_mean": float(high.mean()),
                "bottom_mean": float(low.mean()),
                "difference": difference,
                "standard_error": se,
                "p_value_two_sided": _p_two_sided(z),
            }
        )
    return {
        "status": "ok",
        "top_group": groups,
        "bottom_group": 1,
        "variables": variables,
    }


# ---------------------------------------------------------------------------
# RATE


def rank_weighted_ate_point(score: Any, priority: Any) -> tuple[float, float]:
    """Return (AUTOC, Qini) for ranking by ``priority`` (higher first).

    Tied priorities are handled by averaging psi within each tie block, which
    equals the expectation over random tie-breaking.
    """

    score, priority = np.asarray(score, dtype=float), np.asarray(priority, dtype=float)
    n = len(score)
    order = np.argsort(-priority, kind="mergesort")
    ordered, ranked = score[order], priority[order]
    starts = np.r_[0, np.flatnonzero(np.diff(ranked) != 0) + 1]
    counts = np.diff(np.r_[starts, n])
    ordered = np.repeat(np.add.reduceat(ordered, starts) / counts, counts)
    positions = np.arange(1, n + 1)
    toc = np.cumsum(ordered) / positions - ordered.mean()
    return float(toc.mean()), float(np.mean(positions / n * toc))


def rank_weighted_average_treatment_effects(
    score: Any,
    priority: Any,
    *,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = 0,
    strata: Any = None,
) -> dict[str, Any]:
    """AUTOC and Qini with nonparametric (optionally stratified) bootstrap SEs."""

    score, priority = np.asarray(score, dtype=float), np.asarray(priority, dtype=float)
    strata = np.zeros(len(score), dtype=int) if strata is None else np.asarray(strata)
    finite = np.isfinite(score) & np.isfinite(priority)
    score, priority, strata = score[finite], priority[finite], strata[finite]
    n = len(score)
    if n < 4:
        return unavailable("fewer than 4 rows with finite aipw_score and estimated_cate", rows=n)
    if _is_constant(priority):
        return unavailable("estimated_cate is constant, so it cannot prioritize patients", rows=n)
    autoc, qini = rank_weighted_ate_point(score, priority)
    rng = np.random.default_rng(seed)
    blocks = [np.flatnonzero(strata == value) for value in np.unique(strata)]
    draws = np.empty((int(n_bootstrap), 2))
    for replicate in range(int(n_bootstrap)):
        index = np.concatenate([rng.choice(block, size=len(block)) for block in blocks])
        draws[replicate] = rank_weighted_ate_point(score[index], priority[index])
    result: dict[str, Any] = {
        "status": "ok",
        "rows": n,
        "bootstrap_replicates": int(n_bootstrap),
        "bootstrap_seed": int(seed),
    }
    for column, (name, estimate) in enumerate((("autoc", autoc), ("qini", qini))):
        se = _float(draws[:, column].std(ddof=1)) if n_bootstrap > 1 else None
        z = _ratio(estimate, se) if se else None
        result[name] = {
            "estimate": estimate,
            "standard_error": se,
            "confidence_interval_95": _interval(estimate, se),
            "p_value_one_sided_greater_than_0": _p_greater(z),
        }
    return result


# ---------------------------------------------------------------------------
# Overlap


def overlap_weighted_effects(
    outcome: Any, treatment: Any, propensity: Any, mu0: Any, mu1: Any
) -> dict[str, Any]:
    """Overlap-weighted ATE (ATO): Hajek and augmented, with linearization SEs.

    Standard errors treat the cross-fitted nuisances as fixed.
    """

    y, t = np.asarray(outcome, dtype=float), np.asarray(treatment, dtype=float)
    e = np.asarray(propensity, dtype=float)
    m0, m1 = np.asarray(mu0, dtype=float), np.asarray(mu1, dtype=float)
    finite = np.isfinite(y) & np.isfinite(t) & np.isfinite(e) & np.isfinite(m0) & np.isfinite(m1)
    y, t, e, m0, m1 = y[finite], t[finite], e[finite], m0[finite], m1[finite]
    n = len(y)
    tilt = e * (1.0 - e)
    w1, w0 = t * (1.0 - e), (1.0 - t) * e
    if n < 4 or w1.sum() <= 0 or w0.sum() <= 0 or tilt.sum() <= 0:
        return unavailable("overlap weights are zero in one treatment arm", rows=n)
    mean1, mean0 = np.sum(w1 * y) / w1.sum(), np.sum(w0 * y) / w0.sum()
    hajek = float(mean1 - mean0)
    influence_hajek = w1 * (y - mean1) / w1.mean() - w0 * (y - mean0) / w0.mean()
    residual1 = np.sum(w1 * (y - m1)) / w1.sum()
    residual0 = np.sum(w0 * (y - m0)) / w0.sum()
    model_part = np.sum(tilt * (m1 - m0)) / tilt.sum()
    augmented = float(model_part + residual1 - residual0)
    influence_augmented = (
        tilt * (m1 - m0 - model_part) / tilt.mean()
        + w1 * (y - m1 - residual1) / w1.mean()
        - w0 * (y - m0 - residual0) / w0.mean()
    )

    def summary(estimate: float, influence: np.ndarray) -> dict[str, Any]:
        se = _float(influence.std(ddof=1) / math.sqrt(n))
        return {
            "estimate": estimate,
            "standard_error": se,
            "confidence_interval_95": _interval(estimate, se),
        }

    treated, control = t == 1, t == 0
    ess_treated, ess_control = _kish(1.0 - e[treated]), _kish(e[control])
    ess_total = (ess_treated or 0.0) + (ess_control or 0.0)
    return {
        "status": "ok",
        "rows": n,
        "weights": "treated 1-e, controls e (target population density proportional to e(1-e))",
        "hajek": summary(hajek, influence_hajek),
        "augmented": summary(augmented, influence_augmented),
        "effective_sample_size": {
            "treated": ess_treated,
            "control": ess_control,
            "total": ess_total,
            "fraction_of_rows": ess_total / n,
        },
        "standard_error_method": "influence_function_with_cross_fitted_nuisances_fixed",
    }


def inverse_propensity_weight_summary(
    treatment: Any, propensity: Any, *, clip: float
) -> dict[str, Any]:
    """Kish ESS and maximum weight of the AIPW inverse-propensity weights."""

    t, raw = np.asarray(treatment, dtype=float), np.asarray(propensity, dtype=float)
    finite = np.isfinite(t) & np.isfinite(raw)
    t, raw = t[finite], raw[finite]
    n = len(t)
    if n < 2 or not (t == 1).any() or not (t == 0).any():
        return unavailable("both treatment arms are required", rows=n)
    e = np.clip(raw, clip, 1.0 - clip)
    weights = np.where(t == 1, 1.0 / e, 1.0 / (1.0 - e))
    with np.errstate(divide="ignore"):
        unclipped = np.where(t == 1, 1.0 / raw, 1.0 / (1.0 - raw))
    arms: dict[str, Any] = {}
    for name, mask in (("treated", t == 1), ("control", t == 0)):
        arm_weights = weights[mask]
        ess = _kish(arm_weights)
        arms[name] = {
            "rows": int(mask.sum()),
            "kish_effective_sample_size": ess,
            "kish_effective_sample_size_fraction": _ratio(ess, float(mask.sum())),
            "max_weight": float(arm_weights.max()),
            "max_weight_share": float(arm_weights.max() / arm_weights.sum()),
        }
    ess_total = (arms["treated"]["kish_effective_sample_size"] or 0.0) + (
        arms["control"]["kish_effective_sample_size"] or 0.0
    )
    return {
        "status": "ok",
        "rows": n,
        "propensity_clip": float(clip),
        "kish_effective_sample_size": ess_total,
        "kish_effective_sample_size_fraction": ess_total / n,
        "max_weight": float(weights.max()),
        "max_weight_unclipped": _float(unclipped.max()),
        "rows_clipped_low": int(np.sum(raw < clip)),
        "rows_clipped_high": int(np.sum(raw > 1.0 - clip)),
        "by_arm": arms,
    }


def aipw_trimming_curve(
    outcome: Any,
    treatment: Any,
    propensity: Any,
    mu0: Any,
    mu1: Any,
    *,
    clip: float,
    thresholds: Sequence[float] = TRIMMING_THRESHOLDS,
) -> dict[str, Any]:
    """AIPW ATE over rows with ``alpha <= e <= 1 - alpha`` for each alpha."""

    e = np.asarray(propensity, dtype=float)
    t = np.asarray(treatment, dtype=float)
    psi = dr_scores(outcome, treatment, propensity, mu0, mu1, clip=clip)
    usable = np.isfinite(psi) & np.isfinite(e)
    total = int(usable.sum())
    if total < 2:
        return unavailable("fewer than 2 rows with finite recomputed AIPW scores", rows=total)
    curve = []
    for alpha in thresholds:
        keep = usable & (e >= alpha) & (e <= 1.0 - alpha)
        curve.append(
            {
                "alpha": float(alpha),
                "propensity_range": [float(alpha), float(1.0 - alpha)],
                "fraction_of_rows": float(keep.sum() / total),
                "treated_rows": int(np.sum(keep & (t == 1))),
                "control_rows": int(np.sum(keep & (t == 0))),
                **_mean_with_se(psi[keep]),
            }
        )
    return {
        "status": "ok",
        "rows": total,
        "propensity_clip": float(clip),
        "bounds_are_inclusive": True,
        "curve": curve,
    }


# ---------------------------------------------------------------------------
# Measurement encoding and balance


def _is_continuous(feature: Mapping[str, Any]) -> bool:
    if str(feature.get("value_type") or "").strip().lower() != "continuous":
        return False
    plan = feature.get("harmonization_plan")
    if isinstance(plan, Mapping):
        target = str(plan.get("target_representation") or "").strip().lower()
        return target != "categorical"
    return True


def encode_measurements(
    frame: pd.DataFrame,
    definitions: Sequence[Mapping[str, Any]],
    *,
    max_levels: int = MAX_CATEGORY_LEVELS,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Encode measured features as named numeric columns for descriptive summaries.

    Continuous features give a numeric column (NaN when not documented or not
    numeric). Categorical values, and non-numeric values of continuous
    features, give one 0/1 indicator per level (the ``max_levels`` most frequent
    levels; the rest share an ``other`` level). Missing values give a
    ``not_documented`` indicator, except cells flagged in a
    ``<name>__oci_extraction_failed`` companion column, which give a separate
    ``extraction_failed`` indicator. Oracle-named features are skipped.
    """

    columns: dict[str, np.ndarray] = {}
    meta: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(column: str, values: Any, feature: Mapping[str, Any], kind: str, level: Any) -> None:
        columns[column] = np.asarray(values, dtype=float)
        meta.append(
            {
                "column": column,
                "feature": str(feature["name"]),
                "feature_id": str(feature.get("feature_id") or feature["name"]),
                "roles": sorted(str(role) for role in feature.get("roles") or []),
                "kind": kind,
                "level": level,
            }
        )

    for feature in definitions:
        name = str(feature.get("name") or "").strip()
        if not name or name in seen or is_oracle_name(name) or name not in frame.columns:
            continue
        seen.add(name)
        raw = frame[name]
        text = pd.Series(
            [None if pd.isna(value) else str(value).strip() for value in raw],
            index=raw.index,
            dtype=object,
        )
        failure_column = f"{name}{EXTRACTION_FAILURE_SUFFIX}"
        failed = (
            pd.to_numeric(frame[failure_column], errors="coerce").fillna(0).gt(0)
            if failure_column in frame.columns
            else pd.Series(False, index=frame.index)
        )
        missing = (text.isna() | text.eq("")) & ~failed
        observed = ~missing & ~failed
        if _is_continuous(feature):
            numeric = pd.to_numeric(text.where(observed), errors="coerce")
            if numeric.notna().any():
                add(name, numeric.to_numpy(dtype=float), feature, "continuous", None)
            categorical = observed & numeric.isna()
        else:
            categorical = observed
        if categorical.any():
            counts = text[categorical].value_counts()
            ordered = sorted(counts.index, key=lambda level: (-int(counts[level]), level))
            kept = ordered[:max_levels]
            for level in kept:
                add(
                    f"{name}={level}",
                    (categorical & text.eq(level)).to_numpy(dtype=float),
                    feature,
                    "indicator",
                    level,
                )
            if len(ordered) > len(kept):
                add(
                    f"{name}={OTHER_LEVEL}",
                    (categorical & ~text.isin(kept)).to_numpy(dtype=float),
                    feature,
                    "indicator",
                    OTHER_LEVEL,
                )
        if missing.any():
            add(
                f"{name}={NOT_DOCUMENTED_LEVEL}",
                missing.to_numpy(dtype=float),
                feature,
                "indicator",
                NOT_DOCUMENTED_LEVEL,
            )
        if failed.any():
            add(
                f"{name}={EXTRACTION_FAILED_LEVEL}",
                failed.to_numpy(dtype=float),
                feature,
                "indicator",
                EXTRACTION_FAILED_LEVEL,
            )
    encoded = pd.DataFrame(columns, index=frame.index)
    return encoded, meta


def stack_encoded(
    parts: Sequence[tuple[pd.DataFrame, Sequence[Mapping[str, Any]]]],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Row-stack fold-local encodings.

    An indicator column absent from a fold whose feature was measured there is
    0 (that level was not observed); columns of features not measured in a
    fold are NaN, so that fold does not contribute to them.
    """

    meta_by_column: dict[str, dict[str, Any]] = {}
    for _encoded, meta in parts:
        for item in meta:
            meta_by_column.setdefault(str(item["column"]), dict(item))
    columns = list(meta_by_column)
    frames = []
    for encoded, meta in parts:
        measured = {str(item["feature"]) for item in meta}
        frame = encoded.reindex(columns=columns).reset_index(drop=True)
        for column in columns:
            item = meta_by_column[column]
            unobserved_level = column not in encoded and item["kind"] == "indicator"
            if unobserved_level and item["feature"] in measured:
                frame[column] = 0.0
        frames.append(frame)
    if not frames:
        return pd.DataFrame(), []
    return pd.concat(frames, ignore_index=True), [meta_by_column[column] for column in columns]


def weighted_standardized_mean_differences(
    encoded: pd.DataFrame,
    columns: Sequence[Mapping[str, Any]],
    treatment: Any,
    propensity: Any,
    *,
    clip: float,
    threshold: float = SMD_THRESHOLD,
) -> dict[str, Any]:
    """Unweighted and IPW-weighted SMDs for every encoded column.

    The denominator is the pooled unweighted SD, sqrt((s1^2 + s0^2) / 2), for
    both versions so that they are directly comparable (Austin and Stuart,
    2015). Continuous columns use rows with an observed value.
    """

    t = np.asarray(treatment, dtype=float)
    e = np.clip(np.asarray(propensity, dtype=float), clip, 1.0 - clip)
    weights = np.where(t == 1, 1.0 / e, 1.0 / (1.0 - e))
    variables = []
    for meta in columns:
        values = encoded[meta["column"]].to_numpy(dtype=float)
        observed = np.isfinite(values) & np.isfinite(weights)
        treated, control = observed & (t == 1), observed & (t == 0)
        entry: dict[str, Any] = {
            **{key: meta[key] for key in ("column", "feature", "level", "kind")},
            "roles": list(meta.get("roles") or []),
            "treated_rows": int(treated.sum()),
            "control_rows": int(control.sum()),
        }
        if treated.sum() < 2 or control.sum() < 2:
            variables.append({**entry, "status": "insufficient_rows"})
            continue
        sd = math.sqrt((values[treated].var(ddof=1) + values[control].var(ddof=1)) / 2.0)
        if sd <= 1e-12:
            # Constant within each arm: either uninformative or perfectly
            # separating treatment (an unbounded SMD that weighting cannot fix).
            separated = abs(values[treated].mean() - values[control].mean()) > 1e-12
            variables.append({**entry, "status": "separated" if separated else "constant"})
            continue
        unweighted = (values[treated].mean() - values[control].mean()) / sd
        weighted_treated = np.average(values[treated], weights=weights[treated])
        weighted_control = np.average(values[control], weights=weights[control])
        variables.append(
            {
                **entry,
                "status": "ok",
                "treated_mean_weighted": float(weighted_treated),
                "control_mean_weighted": float(weighted_control),
                "smd_unweighted": float(unweighted),
                "smd_weighted": float((weighted_treated - weighted_control) / sd),
            }
        )
    evaluated = [item for item in variables if item["status"] == "ok"]
    flagged = sorted(
        (item for item in evaluated if abs(item["smd_weighted"]) > threshold),
        key=lambda item: (-abs(item["smd_weighted"]), item["column"]),
    )
    separated = [item for item in variables if item["status"] == "separated"]
    return {
        "status": "ok",
        "rows": len(t),
        "treated_rows": int(np.sum(t == 1)),
        "control_rows": int(np.sum(t == 0)),
        "propensity_clip": float(clip),
        "threshold": float(threshold),
        "features": len({str(item["feature"]) for item in columns}),
        "columns": len(columns),
        "columns_evaluated": len(evaluated),
        "columns_above_threshold_unweighted": sum(
            abs(item["smd_unweighted"]) > threshold for item in evaluated
        ),
        "columns_above_threshold_weighted": len(flagged),
        "max_abs_smd_unweighted": max(
            (abs(item["smd_unweighted"]) for item in evaluated), default=None
        ),
        "max_abs_smd_weighted": max(
            (abs(item["smd_weighted"]) for item in evaluated), default=None
        ),
        "features_above_threshold_weighted": sorted(
            {str(item["feature"]) for item in [*flagged, *separated]}
        ),
        "columns_perfectly_separating_treatment": [item["column"] for item in separated],
        "above_threshold_weighted": [
            {
                key: item[key]
                for key in ("column", "feature", "level", "smd_weighted", "smd_unweighted")
            }
            for item in flagged
        ],
        "variables": variables,
    }


def summarize_fold_balance(
    per_fold: Mapping[str, Mapping[str, Any]], *, threshold: float = SMD_THRESHOLD
) -> dict[str, Any]:
    """Summarize per-fold SMD tables whose rows cannot be pooled."""

    ok = {fold: table for fold, table in per_fold.items() if table.get("status") == "ok"}
    if not ok:
        return unavailable("no fold has a balance table")
    by_column: dict[str, dict[str, Any]] = {}
    separated: dict[str, list[str]] = {}
    for fold, table in ok.items():
        for item in table["variables"]:
            if item.get("status") == "separated":
                separated.setdefault(item["feature"], []).append(fold)
            if item.get("status") != "ok":
                continue
            record = by_column.setdefault(
                item["column"],
                {"column": item["column"], "feature": item["feature"], "level": item["level"],
                 "weighted": {}},
            )
            record["weighted"][fold] = item["smd_weighted"]
    flagged = []
    for record in by_column.values():
        values = record.pop("weighted")
        above = sorted((f for f, value in values.items() if abs(value) > threshold), key=_fold_key)
        record.update(
            folds_evaluated=len(values),
            folds_above_threshold=above,
            mean_smd_weighted=float(np.mean(list(values.values()))),
            max_abs_smd_weighted=float(max(abs(value) for value in values.values())),
        )
        if above:
            flagged.append(record)
    flagged.sort(
        key=lambda item: (
            -len(item["folds_above_threshold"]), -item["max_abs_smd_weighted"], item["column"]
        )
    )
    return {
        "status": "ok",
        "folds": sorted(ok, key=_fold_key),
        "threshold": float(threshold),
        "columns_evaluated": len(by_column),
        "columns_above_threshold_in_any_fold": len(flagged),
        "columns_above_threshold_by_fold": {
            fold: table["columns_above_threshold_weighted"] for fold, table in ok.items()
        },
        "max_abs_smd_weighted": max(
            (item["max_abs_smd_weighted"] for item in by_column.values()), default=None
        ),
        "features_above_threshold_in_any_fold": sorted(
            {item["feature"] for item in flagged}.union(separated)
        ),
        "features_perfectly_separating_treatment": {
            feature: sorted(set(folds), key=_fold_key)
            for feature, folds in sorted(separated.items())
        },
        "above_threshold_in_any_fold": flagged,
    }


# ---------------------------------------------------------------------------
# Assembly from in-memory inputs


@dataclass
class MeasurementInput:
    """Encoded-ready measurements for one fold (raw values keyed by ``_oci_row_id``)."""

    frame: pd.DataFrame
    definitions: list[dict[str, Any]]
    scope: str
    sources: dict[str, str] = field(default_factory=dict)
    propensity: pd.DataFrame | None = None  # training balance: _oci_row_id, treatment, propensity


@dataclass
class FoldArtifacts:
    clan: MeasurementInput | str
    balance_training: MeasurementInput | str
    balance_heldout: MeasurementInput | str


def _guarded(name: str, function: Callable[..., dict[str, Any]], *args: Any, **kwargs: Any):
    try:
        return function(*args, **kwargs)
    except Exception as error:  # a diagnostic must never fail a finished Stage 2 run
        LOGGER.warning("Stage 2 diagnostic %s failed", name, exc_info=True)
        return {"status": "error", "reason": f"{type(error).__name__}: {error}"}


def _join_rows(rows: pd.DataFrame, measurements: pd.DataFrame) -> pd.DataFrame:
    if ROW_ID not in measurements.columns:
        raise ValueError(f"measurement frame lacks {ROW_ID}")
    measurements = measurements.copy()
    measurements[ROW_ID] = pd.to_numeric(measurements[ROW_ID]).astype(int)
    if measurements[ROW_ID].duplicated().any():
        raise ValueError("measurement frame has duplicate row ids")
    joined = rows[[ROW_ID]].merge(measurements, on=ROW_ID, how="left", validate="one_to_one")
    missing = int(joined[ROW_ID].isin(measurements[ROW_ID]).eq(False).sum())
    if missing:
        raise ValueError(f"{missing} rows have no persisted measurements")
    return joined


def _clan_section(
    ranked: Mapping[str, pd.DataFrame],
    artifacts: Mapping[str, FoldArtifacts],
    *,
    groups: int,
    unranked: Mapping[str, str],
) -> dict[str, Any]:
    per_fold: dict[str, Any] = {fold: unavailable(reason) for fold, reason in unranked.items()}
    parts, labels = [], []
    for fold, rows in ranked.items():
        source = artifacts.get(fold)
        clan_input = source.clan if source is not None else "no fold artifacts were found"
        if isinstance(clan_input, str):
            per_fold[fold] = unavailable(clan_input)
            continue

        def compute(rows=rows, clan_input=clan_input):
            joined = _join_rows(rows, clan_input.frame)
            encoded, meta = encode_measurements(joined, clan_input.definitions)
            if not meta:
                return unavailable("no retained effect modifier or confounder was measured")
            result = classification_analysis(encoded, meta, rows["cate_group"], groups=groups)
            parts.append((encoded, meta))
            labels.append(rows["cate_group"].to_numpy())
            return {
                **result,
                "variable_scope": clan_input.scope,
                "sources": clan_input.sources,
            }

        per_fold[fold] = _guarded(f"clan fold {fold}", compute)
    if parts:
        stacked, meta = stack_encoded(parts)
        pooled = _guarded(
            "clan pooled",
            classification_analysis,
            stacked,
            meta,
            np.concatenate(labels),
            groups=groups,
        )
        pooled["folds"] = sorted(
            (fold for fold, result in per_fold.items() if result.get("status") == "ok"),
            key=_fold_key,
        )
    else:
        pooled = unavailable("no fold has CLAN measurements")
    return {"per_fold": dict(sorted(per_fold.items(), key=lambda item: _fold_key(item[0]))),
            "pooled": pooled}


def _balance_section(
    predictions_by_fold: Mapping[str, pd.DataFrame],
    artifacts: Mapping[str, FoldArtifacts],
    *,
    scope: str,
    clip: float,
) -> dict[str, Any]:
    per_fold: dict[str, Any] = {}
    parts, treatments, propensities = [], [], []
    scopes = set()
    for fold, rows in predictions_by_fold.items():
        source = artifacts.get(fold)
        item = (
            getattr(source, f"balance_{scope}") if source is not None
            else "no fold artifacts were found"
        )
        if isinstance(item, str):
            per_fold[fold] = unavailable(item)
            continue

        def compute(rows=rows, item=item):
            if scope == "training":
                weights_frame = item.propensity
                if weights_frame is None:
                    return unavailable("training propensities are missing")
                weights_frame = weights_frame.copy()
                weights_frame[ROW_ID] = pd.to_numeric(weights_frame[ROW_ID]).astype(int)
            else:
                weights_frame = rows
            joined = _join_rows(weights_frame[[ROW_ID]], item.frame)
            treatment = pd.to_numeric(weights_frame["treatment"]).to_numpy(dtype=float)
            propensity = pd.to_numeric(weights_frame["propensity"]).to_numpy(dtype=float)
            encoded, meta = encode_measurements(joined, item.definitions)
            if not meta:
                return unavailable("no candidate measurement columns were found")
            result = weighted_standardized_mean_differences(
                encoded, meta, treatment, propensity, clip=clip
            )
            parts.append((encoded, meta))
            treatments.append(treatment)
            propensities.append(propensity)
            scopes.add(item.scope)
            return {
                **result,
                "candidate_scope": item.scope,
                "complete": item.scope == "all_nuisance_candidates",
                "sources": item.sources,
            }

        per_fold[fold] = _guarded(f"balance {scope} fold {fold}", compute)
    if scope == "training":
        pooled = summarize_fold_balance(per_fold)
    elif parts:
        stacked, meta = stack_encoded(parts)
        pooled = _guarded(
            "balance heldout pooled",
            weighted_standardized_mean_differences,
            stacked,
            meta,
            np.concatenate(treatments),
            np.concatenate(propensities),
            clip=clip,
        )
    else:
        pooled = unavailable("no fold has held-out balance inputs")
    status = "ok" if any(r.get("status") == "ok" for r in per_fold.values()) else "unavailable"
    return {
        "status": status,
        "propensity_source": (
            "training inner-fold out-of-fold propensity (nuisance_fit_predictions.csv)"
            if scope == "training"
            else "outer-training propensity for held-out rows (predictions.csv)"
        ),
        "candidate_scope": (
            scopes.pop() if len(scopes) == 1 else ("mixed" if scopes else None)
        ),
        "per_fold": per_fold,
        ("pooled" if scope == "heldout" else "pooled_summary"): pooled,
    }


def build_heterogeneity_diagnostics(
    predictions: pd.DataFrame,
    *,
    propensity_clip: float,
    seed: int,
    fold_artifacts: Mapping[int, FoldArtifacts] | None = None,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_REPLICATES,
    groups: int = CATE_GROUPS,
) -> dict[str, Any]:
    """Compute every diagnostic from the pooled cross-fitted prediction table."""

    predictions = predictions.drop(
        columns=[column for column in predictions.columns if is_oracle_name(column)]
    ).reset_index(drop=True)
    required = {"outer_fold", "treatment", "outcome", "propensity", "mu0", "mu1",
                "aipw_score", "estimated_cate"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"cross-fitted predictions lack columns: {missing}")
    if ROW_ID not in predictions.columns:
        predictions[ROW_ID] = np.arange(len(predictions))
    artifacts = {str(fold): value for fold, value in (fold_artifacts or {}).items()}
    numeric = predictions[sorted(required - {"outer_fold"})].apply(pd.to_numeric, errors="coerce")
    frame = pd.concat([predictions[[ROW_ID, "outer_fold"]], numeric], axis=1)
    frame[ROW_ID] = pd.to_numeric(frame[ROW_ID]).astype(int)
    frame["outer_fold"] = pd.to_numeric(frame["outer_fold"]).astype(int)
    by_fold = {str(fold): part.reset_index(drop=True)
               for fold, part in frame.groupby("outer_fold", sort=True)}
    recomputed = dr_scores(frame["outcome"], frame["treatment"], frame["propensity"],
                           frame["mu0"], frame["mu1"], clip=propensity_clip)
    eligible = np.isfinite(frame["aipw_score"].to_numpy(dtype=float))
    discrepancy = (
        float(np.nanmax(np.abs(recomputed[eligible] - frame["aipw_score"].to_numpy()[eligible])))
        if eligible.any() else None
    )

    # Rows with an effect estimate, grouped within fold.
    ranked: dict[str, pd.DataFrame] = {}
    constant_folds, small_folds = [], []
    for fold, part in by_fold.items():
        rows = part[np.isfinite(part["aipw_score"]) & np.isfinite(part["estimated_cate"])]
        rows = rows.reset_index(drop=True)
        if len(rows) < max(groups, 4):
            small_folds.append(fold)
            continue
        if _is_constant(rows["estimated_cate"].to_numpy()):
            constant_folds.append(fold)
            continue
        rows["cate_group"] = cate_groups(rows["estimated_cate"], groups=groups,
                                         seed=seed + BOOTSTRAP_SEED_OFFSET + int(fold))
        rows["cate_percentile"] = rows["estimated_cate"].rank(pct=True, method="average")
        ranked[fold] = rows

    blp_per_fold = {
        fold: _guarded(f"blp fold {fold}", best_linear_predictor,
                       part["aipw_score"], part["estimated_cate"])
        for fold, part in by_fold.items()
    }
    gates_per_fold = {
        fold: _guarded(f"gates fold {fold}", group_average_treatment_effects,
                       rows["aipw_score"], rows["estimated_cate"], rows["cate_group"],
                       groups=groups)
        for fold, rows in ranked.items()
    }
    unranked = {
        **{fold: "estimated_cate is constant in this fold" for fold in constant_folds},
        **{fold: "too few rows with effect estimates" for fold in small_folds},
    }
    for fold, reason in unranked.items():
        gates_per_fold[fold] = unavailable(reason)
    gates_per_fold = dict(sorted(gates_per_fold.items(), key=lambda item: _fold_key(item[0])))
    rate_per_fold = {
        fold: _guarded(
            f"rate fold {fold}", rank_weighted_average_treatment_effects,
            part["aipw_score"], part["estimated_cate"], n_bootstrap=n_bootstrap,
            seed=seed + BOOTSTRAP_SEED_OFFSET + int(fold),
        )
        for fold, part in by_fold.items()
    }
    if ranked:
        pooled_rows = pd.concat(list(ranked.values()), ignore_index=True)
        gates_pooled = _guarded("gates pooled", group_average_treatment_effects,
                                pooled_rows["aipw_score"], pooled_rows["estimated_cate"],
                                pooled_rows["cate_group"], groups=groups)
        rate_pooled = _guarded(
            "rate pooled", rank_weighted_average_treatment_effects,
            pooled_rows["aipw_score"], pooled_rows["cate_percentile"],
            n_bootstrap=n_bootstrap, seed=seed + BOOTSTRAP_SEED_OFFSET,
            strata=pooled_rows["outer_fold"],
        )
        for pooled in (gates_pooled, rate_pooled):
            pooled["folds"] = sorted(ranked, key=_fold_key)
        rate_pooled["priority"] = "within_fold_percentile_of_estimated_cate"
    else:
        gates_pooled = unavailable("no fold has a non-constant estimated_cate")
        rate_pooled = unavailable("no fold has a non-constant estimated_cate")

    def overlap_parts(part: pd.DataFrame) -> dict[str, Any]:
        args = (part["outcome"], part["treatment"], part["propensity"], part["mu0"], part["mu1"])
        return {
            "overlap_weighted_ate": _guarded("ato", overlap_weighted_effects, *args),
            "ipw_weights": _guarded("ipw weights", inverse_propensity_weight_summary,
                                    part["treatment"], part["propensity"], clip=propensity_clip),
            "trimming_curve": _guarded("trimming", aipw_trimming_curve, *args,
                                       clip=propensity_clip),
        }

    overlap_by_fold = {fold: overlap_parts(part) for fold, part in by_fold.items()}
    overlap_pooled = overlap_parts(frame)

    diagnostics: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "oracle_columns_consumed": False,
        "inputs": {
            "rows": len(frame),
            "effect_estimation_rows": int(eligible.sum()),
            "outer_folds": sorted(by_fold, key=int),
            "propensity_clip": float(propensity_clip),
            "bootstrap_replicates": int(n_bootstrap),
            "bootstrap_seed_base": int(seed + BOOTSTRAP_SEED_OFFSET),
            "cate_groups": int(groups),
            "trimming_thresholds": [float(value) for value in TRIMMING_THRESHOLDS],
            "smd_threshold": SMD_THRESHOLD,
            "folds_with_constant_cate": sorted(constant_folds, key=int),
            "folds_with_too_few_effect_rows": sorted(small_folds, key=int),
            "score_columns": {
                "psi": "aipw_score", "cate": "estimated_cate", "propensity": "propensity",
                "mu0": "mu0", "mu1": "mu1", "treatment": "treatment", "outcome": "outcome",
            },
            "recomputed_psi_max_abs_difference_from_aipw_score": discrepancy,
        },
        "pooling_rules": dict(POOLING_RULES),
        "blp": {
            "interpretation": INTERPRETATIONS["blp"],
            "per_fold": blp_per_fold,
            "pooled": _guarded("blp pooled", pool_blp, blp_per_fold),
        },
        "gates": {
            "interpretation": INTERPRETATIONS["gates"],
            "per_fold": gates_per_fold,
            "pooled": gates_pooled,
        },
        "clan": {
            "interpretation": INTERPRETATIONS["clan"],
            **_clan_section(ranked, artifacts, groups=groups, unranked=unranked),
        },
        "rate": {
            "interpretation": INTERPRETATIONS["rate"],
            "per_fold": rate_per_fold,
            "pooled": rate_pooled,
        },
    }
    for key in ("overlap_weighted_ate", "ipw_weights", "trimming_curve"):
        diagnostics[key] = {
            "interpretation": INTERPRETATIONS[key],
            "per_fold": {fold: values[key] for fold, values in overlap_by_fold.items()},
            "pooled": overlap_pooled[key],
        }
    diagnostics["balance"] = {
        "interpretation": INTERPRETATIONS["balance"],
        "threshold": SMD_THRESHOLD,
        "training": _balance_section(by_fold, artifacts, scope="training", clip=propensity_clip),
        "heldout": _balance_section(by_fold, artifacts, scope="heldout", clip=propensity_clip),
    }
    return json_safe(diagnostics)


def _pick(mapping: Any, *keys: str) -> dict[str, Any]:
    mapping = mapping if isinstance(mapping, Mapping) else {}
    return {key: mapping.get(key) for key in keys if key in mapping}


def summarize_diagnostics(diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    """Compact pooled summary for ``causal_estimate.json``."""

    def pooled(name: str) -> Mapping[str, Any]:
        section = diagnostics.get(name)
        return section.get("pooled", {}) if isinstance(section, Mapping) else {}

    blp, gates, rate = pooled("blp"), pooled("gates"), pooled("rate")
    ato, ipw = pooled("overlap_weighted_ate"), pooled("ipw_weights")
    trimming = pooled("trimming_curve")
    balance = diagnostics.get("balance") or {}
    training = balance.get("training") or {}
    heldout = balance.get("heldout") or {}
    training_summary = training.get("pooled_summary") or {}
    heldout_pooled = heldout.get("pooled") or {}
    return {
        "schema_version": diagnostics.get("schema_version"),
        "blp_pooled": _pick(
            blp, "status", "reason", "folds", "beta1_ate", "beta1_standard_error",
            "beta2_heterogeneity", "beta2_standard_error",
            "beta2_p_value_one_sided_greater_than_0", "beta2_p_value_two_sided_equal_1",
        ),
        "gates_pooled": {
            **_pick(gates, "status", "reason", "folds"),
            "top_minus_bottom": _pick(
                gates.get("top_minus_bottom"), "estimate", "standard_error", "p_value_two_sided"
            ),
            "group_means": [group.get("estimate") for group in gates.get("groups") or []],
        },
        "rate_pooled": {
            **_pick(rate, "status", "reason"),
            "autoc": _pick(rate.get("autoc"), "estimate", "standard_error"),
            "qini": _pick(rate.get("qini"), "estimate", "standard_error"),
        },
        "overlap_weighted_ate_pooled": {
            **_pick(ato, "status", "reason"),
            "hajek": _pick(ato.get("hajek"), "estimate", "standard_error"),
            "augmented": _pick(ato.get("augmented"), "estimate", "standard_error"),
            "effective_sample_size": (ato.get("effective_sample_size") or {}).get("total"),
        },
        "ipw_weights_pooled": _pick(
            ipw, "status", "reason", "kish_effective_sample_size",
            "kish_effective_sample_size_fraction", "max_weight",
        ),
        "trimming_curve_pooled": [
            _pick(point, "alpha", "rows", "estimate", "standard_error")
            for point in trimming.get("curve") or []
        ],
        "balance_training": {
            **_pick(training, "status", "candidate_scope"),
            **_pick(training_summary, "columns_above_threshold_by_fold",
                    "features_above_threshold_in_any_fold", "max_abs_smd_weighted"),
        },
        "balance_heldout": {
            **_pick(heldout, "status", "candidate_scope"),
            **_pick(heldout_pooled, "columns_above_threshold_weighted",
                    "features_above_threshold_weighted", "max_abs_smd_weighted"),
        },
    }


# ---------------------------------------------------------------------------
# I/O layer


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_measurements(path: Path, failure_mask: Path | None = None) -> pd.DataFrame:
    """Read raw measurements, adding extraction-failure companions from a mask file."""

    # Strings keep categorical levels exactly as written ("1", not "1.0").
    frame = pd.read_csv(path, dtype=str, keep_default_na=True)
    if failure_mask is None or not failure_mask.is_file() or ROW_ID not in frame.columns:
        return frame
    mask = pd.read_csv(failure_mask)
    if ROW_ID not in mask.columns:
        return frame
    companions = {
        column: f"{column}{EXTRACTION_FAILURE_SUFFIX}"
        for column in mask.columns
        if column != ROW_ID and f"{column}{EXTRACTION_FAILURE_SUFFIX}" not in frame.columns
    }
    if not companions:
        return frame
    mask = mask[[ROW_ID, *companions]].rename(columns=companions)
    mask[ROW_ID] = pd.to_numeric(mask[ROW_ID]).astype(int)
    frame[ROW_ID] = pd.to_numeric(frame[ROW_ID]).astype(int)
    return frame.merge(mask, on=ROW_ID, how="left")


def _features(payload: Any, path: Path) -> list[dict[str, Any]]:
    features = payload.get("features") if isinstance(payload, Mapping) else None
    if not isinstance(features, list):
        raise ValueError(f"{path} has no features list")
    return [dict(feature) for feature in features if isinstance(feature, Mapping)]


def load_fold_artifacts(fold_dir: Path) -> FoldArtifacts:
    """Read the persisted per-fold inputs; missing inputs become reason strings."""

    fold_dir = Path(fold_dir)
    final_path = fold_dir / "final_definitions.json"
    retained: list[dict[str, Any]] | str
    try:
        retained = _features(_read_json(final_path), final_path)
    except (OSError, ValueError) as error:
        retained = f"retained definitions unavailable ({final_path.name}): {error}"

    candidate_fit = fold_dir / "estimation" / "nuisance_candidate_fit.csv"
    candidate_heldout = fold_dir / "estimation" / "nuisance_candidate_heldout.csv"
    candidate_definition_paths = (
        fold_dir / "estimation" / "nuisance_candidate_definitions.json",
        fold_dir / "selection" / "nuisance_candidates.json",
    )
    candidates: list[dict[str, Any]] | None = None
    candidates_path: Path | None = None
    for path in candidate_definition_paths:
        if path.is_file():
            try:
                candidates, candidates_path = _features(_read_json(path), path), path
                break
            except (OSError, ValueError):
                continue

    fit_mask = fold_dir / "extraction" / "failure_mask_fit.csv"
    heldout_mask = fold_dir / "extraction" / "failure_mask_heldout.csv"

    def measurement_input(
        all_path: Path,
        retained_path: Path,
        *,
        mask: Path,
        extra: Mapping[str, Path] | None = None,
    ) -> MeasurementInput | str:
        if all_path.is_file() and candidates is not None:
            return MeasurementInput(
                frame=_read_measurements(all_path, mask), definitions=candidates,
                scope="all_nuisance_candidates",
                sources={"measurements": str(all_path), "definitions": str(candidates_path),
                         **{k: str(v) for k, v in (extra or {}).items()}},
            )
        if isinstance(retained, str):
            return retained
        if not retained_path.is_file():
            return f"measurements unavailable: {retained_path} is missing"
        return MeasurementInput(
            frame=_read_measurements(retained_path, mask), definitions=retained,
            scope="retained_features_only",
            sources={
                "measurements": str(retained_path), "definitions": str(final_path),
                "full_candidate_measurements_missing": str(all_path),
                **{k: str(v) for k, v in (extra or {}).items()},
            },
        )

    heldout_retained = fold_dir / "extraction" / "heldout" / "harmonized.csv"
    if isinstance(retained, str):
        clan: MeasurementInput | str = retained
    elif not heldout_retained.is_file():
        clan = f"held-out measurements unavailable: {heldout_retained} is missing"
    else:
        clan = MeasurementInput(
            frame=_read_measurements(heldout_retained, heldout_mask),
            definitions=[
                feature for feature in retained
                if CLAN_ROLES.intersection(map(str, feature.get("roles") or []))
            ],
            scope="retained_effect_modifiers_and_confounders",
            sources={"measurements": str(heldout_retained), "definitions": str(final_path)},
        )

    nuisance_fit = fold_dir / "estimation" / "nuisance_fit_predictions.csv"
    training: MeasurementInput | str
    if not nuisance_fit.is_file():
        training = f"training propensities unavailable: {nuisance_fit} is missing"
    else:
        training = measurement_input(
            candidate_fit, fold_dir / "extraction" / "fit" / "extracted.csv",
            mask=fit_mask, extra={"propensity": nuisance_fit},
        )
        if isinstance(training, MeasurementInput):
            training.propensity = pd.read_csv(nuisance_fit)
            missing = {ROW_ID, "treatment", "propensity"} - set(training.propensity.columns)
            if missing:
                training = f"{nuisance_fit} lacks columns {sorted(missing)}"
    heldout = measurement_input(candidate_heldout, heldout_retained, mask=heldout_mask)
    return FoldArtifacts(clan=clan, balance_training=training, balance_heldout=heldout)


def write_heterogeneity_diagnostics(
    *,
    output_dir: Path,
    predictions: pd.DataFrame,
    propensity_clip: float,
    seed: int,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_REPLICATES,
) -> tuple[Path, dict[str, Any]]:
    """Write ``heterogeneity_diagnostics.json`` and return (path, compact summary).

    Never raises for diagnostic failures: they are recorded in the file.
    """

    output_dir = Path(output_dir)
    path = output_dir / DIAGNOSTICS_FILENAME
    try:
        folds = sorted({int(value) for value in pd.to_numeric(predictions["outer_fold"])})
        artifacts: dict[int, FoldArtifacts] = {}
        for fold in folds:
            try:
                artifacts[fold] = load_fold_artifacts(output_dir / f"outer_{fold:03d}")
            except Exception as error:  # a diagnostic must never fail a finished Stage 2 run
                LOGGER.warning("Stage 2 diagnostics could not read fold %s", fold, exc_info=True)
                reason = f"fold artifacts unreadable: {type(error).__name__}: {error}"
                artifacts[fold] = FoldArtifacts(reason, reason, reason)
        diagnostics = build_heterogeneity_diagnostics(
            predictions,
            propensity_clip=propensity_clip,
            seed=seed,
            fold_artifacts=artifacts,
            n_bootstrap=n_bootstrap,
        )
    except Exception as error:  # a diagnostic must never fail a finished Stage 2 run
        LOGGER.warning("Stage 2 heterogeneity diagnostics failed", exc_info=True)
        diagnostics = {
            "schema_version": SCHEMA_VERSION,
            "status": "error",
            "reason": f"{type(error).__name__}: {error}",
        }
    diagnostics = json_safe(diagnostics)
    diagnostics["predictions_rows"] = len(predictions)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    summary = (
        summarize_diagnostics(diagnostics)
        if diagnostics.get("status") != "error"
        else _pick(diagnostics, "schema_version", "status", "reason")
    )
    return path, json_safe({"path": str(path), **summary})


__all__ = [
    "CATE_GROUPS",
    "DIAGNOSTICS_FILENAME",
    "SCHEMA_VERSION",
    "SMD_THRESHOLD",
    "TRIMMING_THRESHOLDS",
    "FoldArtifacts",
    "MeasurementInput",
    "aipw_trimming_curve",
    "best_linear_predictor",
    "build_heterogeneity_diagnostics",
    "cate_groups",
    "classification_analysis",
    "dr_scores",
    "encode_measurements",
    "group_average_treatment_effects",
    "inverse_propensity_weight_summary",
    "inverse_variance_pool",
    "load_fold_artifacts",
    "overlap_weighted_effects",
    "pool_blp",
    "rank_weighted_ate_point",
    "rank_weighted_average_treatment_effects",
    "stack_encoded",
    "summarize_diagnostics",
    "summarize_fold_balance",
    "weighted_standardized_mean_differences",
    "write_heterogeneity_diagnostics",
]
