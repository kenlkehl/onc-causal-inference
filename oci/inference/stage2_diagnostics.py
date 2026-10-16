"""Oracle-free heterogeneity, overlap and balance diagnostics for Stage 2.

The diagnostics run after every outer fold's predictions are frozen. They read
only the pooled cross-fitted prediction table (held-out AIPW score ``psi``,
``estimated_cate``, ``propensity``, ``mu0``, ``mu1``, ``treatment`` and the
observed ``outcome``) and persisted per-fold measurement artifacts. They never
read the dataset, so oracle columns cannot enter; columns and features with
oracle-style names are dropped as a second guard. Nothing in selection or
estimation reads these results.

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
* Clip sensitivity: the AIPW ATE recomputed at propensity clips 0.01, 0.02
  and 0.05 (and the configured clip), a secondary sensitivity analysis.
* Overlap flags (reporting only): counts of held-out patients whose estimated
  propensity lies outside ``stage2.overlap_flag_bounds``.
* Unmeasured-confounding sensitivity (reporting only;
  :mod:`oci.inference.stage2_sensitivity`): E-values (VanderWeele and Ding) and
  omitted-variable-bias bounds with robustness values (Chernozhukov et al.;
  Cinelli and Hazlett) for the ATE and the ATO, and benchmark-implied ATE
  bounds from each fold's ``estimation/sensitivity_benchmarks/benchmarks.json``.
* Balance: IPW-weighted standardized mean differences of every measured
  candidate, with a summary of variables whose ``|SMD| > 0.1``.
* Modifier confirmation (reporting only): each fold's selected effect
  modifiers are re-tested on that fold's held-out patients with the same three
  component tests and Cauchy combination used for training selection
  (:mod:`oci.inference.stage2_modifier_tests`), with Benjamini-Hochberg within
  the fold's selected set. Sample splitting makes these tests valid conditional
  on the selection.
* CATE variance decomposition (reporting only): the share of each fold's
  held-out CATE variance (and of each ensemble component's) that a linear
  projection attributes to the prognostic score, to the selected modifiers'
  grouped design, to both jointly, or to neither.

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
* Modifier confirmation is computed per fold; the pooled entry counts selected
  and confirmed modifiers. Each fold defines its own variables, so the
  cross-fold summary groups them into concepts by lexical name and definition
  similarity (at most one variable per fold), aggregates each concept's
  held-out p-values (Meinshausen-Meier-Buhlmann) with Benjamini-Hochberg across
  concepts, and pools matched effect sizes by fixed effects.
* The unmeasured-confounding sensitivity is computed per fold and on the
  stacked held-out rows; benchmark gains are per fold and are applied to the
  fold's and the pooled ATE.
* The CATE variance decomposition is computed per fold; pooled shares are
  row-weighted means of the fold partitions, reported with the fold range.

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
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from .stage2_documented_unknown import DOCUMENTED_UNKNOWN, is_documented_unknown
from .stage2_sensitivity import (
    load_fold_benchmarks,
    summarize_section as summarize_sensitivity_section,
    unmeasured_confounding_section,
)

LOGGER = logging.getLogger(__name__)

SCHEMA_VERSION = "stage2_heterogeneity_diagnostics_v5_unmeasured_confounding"
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
        "spread out, beta2 > 1 that they are too compressed. The pooled beta2 "
        "with its one-sided p-value is the pre-specified primary heterogeneity "
        "test; per-fold estimates use only one fold's held-out rows and are "
        "low-powered."
    ),
    "gates": (
        "Group average treatment effects: mean psi within quintiles of tau_hat "
        "formed inside each outer fold (G1 lowest, G5 highest predicted "
        "effect), each reported with the group's mean tau_hat. An increasing "
        "pattern, a positive linear trend and a positive top-minus-bottom "
        "difference indicate that the ranking separates patients with larger and "
        "smaller effects; similar group means indicate no detectable "
        "heterogeneity. calibration is the HC3 slope of psi on the patient's "
        "group mean tau_hat: near 1, the group effects match the predicted group "
        "means; near 0, the predicted spread is not detectably real. GATES is a "
        "secondary analysis; the pooled BLP beta2 is the primary test."
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
        "none. RATE is a secondary analysis."
    ),
    "overlap_weighted_ate": (
        "ATE in the overlap population (ATO; weights e(1-e)), which emphasizes "
        "patients in clinical equipoise. Hajek uses weighted outcome means "
        "only; augmented adds the cross-fitted outcome models. A large gap "
        "between the ATO and the AIPW ATE means the ATE depends on patients "
        "with extreme propensities, or that effects differ in the overlap "
        "region. The effective sample size is that of the overlap weights."
    ),
    "clip_sensitivity": (
        "Secondary sensitivity analysis: the AIPW ATE over the effect-eligible rows "
        "recomputed with the propensity clipped to [c, 1 - c] for several c, using the "
        "same psi construction; configured marks the clip used for the reported ATE. "
        "Clipping changes the weights, not the population; estimates that move with c "
        "show dependence on patients with extreme propensities."
    ),
    "overlap_flags": (
        "Reporting only: held-out patients whose final estimated propensity lies outside "
        "stage2.overlap_flag_bounds (the outside_overlap column of "
        "cross_fitted_predictions.csv), per fold and pooled. The flag never removes "
        "patients or changes estimates."
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

CONFIRMATION_INTERPRETATION = (
    "Reporting only: no selection, estimation, ensemble or other fold reads these "
    "results. Each outer fold's selected effect modifiers are re-tested on that "
    "fold's held-out patients (propensity-eligible rows, the frozen held-out "
    "nuisances e_hat, mu0_hat, mu1_hat and the observed labels) with the same "
    "r_projection, aipw_projection and outcome_interaction tests and Cauchy "
    "combination used for training selection; Benjamini-Hochberg runs within the "
    "fold's selected set. confirmed means held-out combined q <= q. Because the "
    "held-out patients took no part in discovery or selection, these tests are "
    "valid conditional on the selection (sample splitting); training p-values are "
    "optimistic because the candidate list was built from the training patients. "
    "effects gives each modifier's held-out R-learner and AIPW projection "
    "coefficients (change in the absolute treatment effect, or in the AIPW "
    "pseudo-outcome, per unit of each design column, per level versus the reference "
    "for categorical modifiers, and per training SD for continuous ones) with HC3 "
    "standard errors and 95% intervals; together with the GATES group effects these "
    "are the honest uncertainty statements, since no per-patient CATE interval "
    "accounts for variable discovery, extraction and selection. Each fold defines and "
    "extracts its own variables; across_folds groups them into concepts by name and "
    "definition similarity, never tests one fold's variable on another fold's "
    "held-out rows, and aggregates each concept's held-out p-values across folds."
)
CONFIRMATION_POOLING_RULE = (
    "per fold; pooled counts of selected and confirmed modifiers; across folds, "
    "variables grouped into concepts (at most one per fold) by lexical name and "
    "definition similarity, with Meinshausen-Meier-Buhlmann aggregation of the "
    "held-out combined p-values (gamma = 0.5, p = 1 where not selected), "
    "Benjamini-Hochberg across concepts, and fixed-effect inverse-variance pooling "
    "of effect sizes (Cochran's Q, I^2, fold range) when every matched fold has the "
    "same value or category columns"
)

CATE_COMPONENT_COLUMNS = (
    "estimated_cate_causal_forest",
    "estimated_cate_linear_interactions",
    "estimated_cate_dina",
)
DECOMPOSITION_KEYS = (
    "r2_prognosis", "r2_modifiers", "r2_both",
    "unique_prognosis", "unique_modifiers", "shared", "unexplained",
)
DECOMPOSITION_INTERPRETATION = (
    "Reporting only: no selection or estimation step reads these results. Descriptive "
    "partition of the variance of each outer fold's held-out estimated_cate (and of each "
    "ensemble component's CATE column) by ordinary least squares with an intercept on "
    "(a) the prognostic score alone (logit of the held-out mu0 for binary outcomes, mu0 "
    "for continuous outcomes: the baseline-risk input of the causal forest's X and of the "
    "interaction model's adjustment), (b) the selected modifiers' grouped design alone "
    "(the modifier tests' encoding of the held-out measurements), and (c) both. "
    "unique_prognosis = R2(c) - R2(b); unique_modifiers = R2(c) - R2(a); shared = R2(a) + "
    "R2(b) - R2(c) (negative when one input suppresses the other); unexplained = 1 - "
    "R2(c). A large prognosis share explains wide CATE spread with few modifiers: on the "
    "risk-difference scale the effect varies with baseline risk even under a constant "
    "odds ratio. The partition is descriptive, not causal: it attributes predicted "
    "variation, not treatment-effect mechanisms. Nonlinear or interaction structure that "
    "a linear projection does not capture (for example a forest's splits) lands in "
    "unexplained, and many modifier columns relative to rows inflate R2(b)."
)
DECOMPOSITION_POOLING_RULE = (
    "per fold; pooled values are row-weighted means of the fold partitions, with the "
    "fold range; folds whose CATE is constant or whose design cannot be estimated are "
    "excluded"
)

POOLING_RULES = {
    "cate_variance_decomposition": DECOMPOSITION_POOLING_RULE,
    "clip_sensitivity": "recomputed on the pooled effect-eligible held-out rows",
    "overlap_flags": "counts per fold and on the pooled held-out rows",
    "unmeasured_confounding_sensitivity": (
        "E-values and omitted-variable-bias elements per fold on its held-out rows and "
        "pooled by stacking every fold's held-out rows; benchmark gains per fold (each "
        "fold's own variables), applied to that fold's ATE and to the pooled ATE"
    ),
    "modifier_confirmation": CONFIRMATION_POOLING_RULE,
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
    calibration = _gates_calibration(score, cate, group, rows)
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
        "calibration": calibration,
    }


def _gates_calibration(score, cate, group, rows) -> dict[str, Any]:
    """HC3 OLS of each patient's psi on [1, mean tau_hat of the patient's group].

    A slope near 1 means the group effects track the predicted group means; a
    slope near 0 means the predicted spread carries no detectable signal. This
    group-level summary is secondary to the BLP beta2.
    """

    means = {row["group"]: row["mean_estimated_cate"] for row in rows}
    predicted = np.array([means.get(int(label)) for label in group], dtype=float)
    if not np.isfinite(predicted).all() or _is_constant(predicted):
        return unavailable("group mean estimated CATEs are constant or missing")
    slope = _ols_slope_hc3(score, predicted)
    z_one = (
        _float((slope["estimate"] - 1.0) / slope["standard_error"])
        if slope["standard_error"] else None
    )
    return {
        "status": "ok",
        "method": "hc3_ols_of_psi_on_group_mean_estimated_cate",
        "slope": slope["estimate"],
        "standard_error": slope["standard_error"],
        "confidence_interval_95": slope["confidence_interval_95"],
        "p_value_one_sided_greater_than_0": slope["p_value_one_sided_greater_than_0"],
        "p_value_two_sided_equal_1": _p_two_sided(z_one),
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


DEFAULT_OVERLAP_FLAG_BOUNDS = (0.05, 0.95)
CLIP_SENSITIVITY_CLIPS = (0.01, 0.02, 0.05)
ESTIMANDS = ("ate", "ato")
ESTIMAND_DESCRIPTIONS = {
    "ate": {
        "label": "Average treatment effect (ATE)",
        "population": (
            "all effect-eligible held-out patients, each weighted equally (with "
            "stage2.min_propensity/max_propensity set, the patients whose estimated "
            "propensity lies within the bounds)"
        ),
        "method": (
            "cross-fitted AIPW: the mean of the held-out AIPW scores with propensities "
            "clipped to [c, 1 - c]; standard error sd(psi)/sqrt(n) with the cross-fitted "
            "nuisances held fixed"
        ),
    },
    "ato": {
        "label": "Overlap-weighted average treatment effect (ATO)",
        "population": (
            "patients in clinical equipoise: every held-out patient weighted by "
            "e(1 - e), so patients who would almost always or almost never be treated "
            "contribute little"
        ),
        "method": (
            "augmented overlap-weighted estimator (Li, Morgan and Zaslavsky) with the "
            "cross-fitted nuisances; influence-function standard error with the nuisances "
            "held fixed; no clipping or trimming is needed because the weights vanish at "
            "the extremes"
        ),
    },
}


def outside_overlap_flags(propensity: Any, bounds: Sequence[float]) -> np.ndarray:
    """True where the propensity lies outside ``[low, high]`` (reporting only)."""

    low, high = (float(value) for value in bounds)
    e = np.asarray(propensity, dtype=float)
    return np.isfinite(e) & ((e < low) | (e > high))


def overlap_flag_summary(
    predictions: pd.DataFrame, bounds: Sequence[float]
) -> dict[str, Any]:
    """Counts of held-out patients outside the overlap-flag bounds, per fold and pooled."""

    def counts(part: pd.DataFrame) -> dict[str, Any]:
        e = pd.to_numeric(part["propensity"], errors="coerce").to_numpy(dtype=float)
        flags = outside_overlap_flags(e, bounds)
        return {
            "rows": int(len(part)),
            "outside_overlap": int(flags.sum()),
            "fraction": float(flags.mean()) if len(part) else None,
            "below_lower_bound": int(np.sum(e < float(bounds[0]))),
            "above_upper_bound": int(np.sum(e > float(bounds[1]))),
        }

    per_fold = {
        str(int(fold)): counts(part)
        for fold, part in predictions.groupby("outer_fold", sort=True)
    } if "outer_fold" in predictions.columns else {}
    return {
        "status": "ok",
        "bounds": [float(bounds[0]), float(bounds[1])],
        "role": (
            "reporting only: outside_overlap marks held-out patients whose final estimated "
            "propensity lies outside the bounds; it never removes patients or changes "
            "estimates (eligibility trimming is the separate min_propensity/max_propensity)"
        ),
        "per_fold": dict(sorted(per_fold.items(), key=lambda item: _fold_key(item[0]))),
        "pooled": counts(predictions),
    }


def clip_sensitivity(
    outcome: Any, treatment: Any, propensity: Any, mu0: Any, mu1: Any, *,
    configured_clip: float, clips: Sequence[float] = CLIP_SENSITIVITY_CLIPS,
) -> dict[str, Any]:
    """The AIPW ATE recomputed at several propensity clips (same psi construction)."""

    values = sorted({float(c) for c in clips} | {float(configured_clip)})
    points = []
    for clip in values:
        psi = dr_scores(outcome, treatment, propensity, mu0, mu1, clip=clip)
        psi = psi[np.isfinite(psi)]
        summary = _mean_with_se(psi)
        points.append({
            "clip": clip,
            "configured": math.isclose(clip, float(configured_clip)),
            "rows": summary["rows"],
            "estimate": summary["estimate"],
            "standard_error": summary["standard_error"],
            "confidence_interval_95": summary["confidence_interval_95"],
        })
    if not points or points[0]["rows"] == 0:
        return unavailable("no effect-eligible rows")
    return {"status": "ok", "points": points}


def _aipw_summary(part: pd.DataFrame) -> dict[str, Any]:
    scores = pd.to_numeric(part["aipw_score"], errors="coerce").to_numpy(dtype=float)
    return _mean_with_se(scores[np.isfinite(scores)])


def co_primary_estimands(
    predictions: pd.DataFrame, *, propensity_clip: float, lead_estimand: str = "ate",
) -> dict[str, Any]:
    """The clipped AIPW ATE and the augmented ATO side by side, per fold and pooled.

    ``lead_estimand`` (pre-specified) changes only the order and labels.
    """

    if lead_estimand not in ESTIMANDS:
        raise ValueError("lead_estimand must be ate or ato")
    order = [lead_estimand, *(name for name in ESTIMANDS if name != lead_estimand)]

    def ato(part: pd.DataFrame) -> dict[str, Any]:
        values = [pd.to_numeric(part[c], errors="coerce").to_numpy(dtype=float)
                  for c in ("outcome", "treatment", "propensity", "mu0", "mu1")]
        result = overlap_weighted_effects(*values)
        if result.get("status") != "ok":
            return result
        return {
            "status": "ok", "rows": result["rows"], **result["augmented"],
            "effective_sample_size": result["effective_sample_size"]["total"],
            "effective_sample_size_fraction": result["effective_sample_size"]["fraction_of_rows"],
        }

    def ate(part: pd.DataFrame) -> dict[str, Any]:
        summary = _aipw_summary(part)
        return {"status": "ok" if summary["rows"] else "unavailable", **summary}

    folds = (
        {str(int(f)): part for f, part in predictions.groupby("outer_fold", sort=True)}
        if "outer_fold" in predictions.columns else {}
    )
    estimands = {}
    for name, compute in (("ate", ate), ("ato", ato)):
        estimands[name] = {
            **ESTIMAND_DESCRIPTIONS[name],
            "role": "lead co-primary estimand" if name == lead_estimand else "co-primary estimand",
            **({"propensity_clip": float(propensity_clip)} if name == "ate" else {}),
            **compute(predictions),
            "per_fold": {fold: compute(part) for fold, part in
                         sorted(folds.items(), key=lambda item: _fold_key(item[0]))},
        }
    gap = (
        estimands["ate"]["estimate"] - estimands["ato"]["estimate"]
        if estimands["ate"].get("estimate") is not None
        and estimands["ato"].get("estimate") is not None else None
    )
    return {
        "lead_estimand": lead_estimand,
        "order": order,
        "description": (
            "The ATE and the ATO are co-primary estimands. lead_estimand records which one "
            "the study pre-specified to lead with; it changes only this order and the role "
            "labels, never the computation. The ATE targets every effect-eligible patient; "
            "the ATO targets patients in clinical equipoise. A large gap means the effect "
            "differs between those populations or the ATE leans on patients with extreme "
            "propensities; the gap, the trimming curve and the clip sensitivity are "
            "secondary sensitivity analyses."
        ),
        **{name: estimands[name] for name in order},
        "ate_minus_ato": {"estimate": gap, "role": "secondary sensitivity analysis"},
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
    ``extraction_failed`` indicator. The reserved ``documented_unknown`` value
    gives its own ``documented_unknown`` indicator and is never a level or
    counted against ``max_levels``. Oracle-named features are skipped.
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
        unknown = pd.Series(
            [is_documented_unknown(value) for value in text.tolist()], index=text.index, dtype=bool,
        )
        observed = ~missing & ~failed & ~unknown
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
        if unknown.any():
            add(
                f"{name}={DOCUMENTED_UNKNOWN}",
                unknown.to_numpy(dtype=float),
                feature,
                "indicator",
                DOCUMENTED_UNKNOWN,
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
class ConfirmationInput:
    """A fold's selected effect modifiers and their held-out measurements."""

    definitions: list[dict[str, Any]]
    frame: pd.DataFrame | None
    sources: dict[str, str] = field(default_factory=dict)
    excluded_oracle_features: list[str] = field(default_factory=list)
    missing_measurement_columns: list[str] = field(default_factory=list)
    # Standard deviation of each continuous modifier's observed training values
    # (the fold's own outer-training measurements; no labels).
    training_scales: dict[str, float] = field(default_factory=dict)


@dataclass
class FoldArtifacts:
    clan: MeasurementInput | str
    balance_training: MeasurementInput | str
    balance_heldout: MeasurementInput | str
    confirmation: ConfirmationInput | str = "modifier confirmation inputs were not loaded"
    sensitivity_benchmarks: Mapping[str, Any] | str = "sensitivity benchmarks were not loaded"


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


def normalized_feature_name(name: Any) -> str:
    """Lower-case name with non-alphanumeric runs collapsed to ``_`` (cross-fold matching)."""

    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def resolve_confirmation_policy(policy: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the confirmation policy and fill outcome-type default weights."""

    from . import stage2_modifier_tests as modifier_tests

    outcome_type = str(policy.get("outcome_type"))
    if outcome_type not in {"binary", "continuous"}:
        raise ValueError("modifier confirmation requires a binary or continuous outcome_type")
    q = _float(policy.get("q"))
    if q is None or not 0 < q <= 1:
        raise ValueError("modifier confirmation q must be in (0, 1]")
    configured = policy.get("weights")
    weights = (
        modifier_tests.default_weights(outcome_type)
        if configured is None
        else {name: float(configured[name]) for name in modifier_tests.COMPONENTS}
    )
    min_count = policy.get("min_count", 5)
    if isinstance(min_count, bool) or not isinstance(min_count, (int, np.integer)) or min_count < 1:
        raise ValueError("modifier confirmation min_count must be a positive integer")
    return {
        "outcome_type": outcome_type,
        "q": q,
        "weights": weights,
        "weights_source": "outcome_type_default" if configured is None else "configured",
        "min_count": int(min_count),
        "family": "the fold's selected effect modifiers",
        "components": list(modifier_tests.COMPONENTS),
    }


def _effect_eligibility(rows: pd.DataFrame) -> np.ndarray:
    if "effect_eligible" in rows.columns:
        values = rows["effect_eligible"]
        if values.dtype == bool:
            return values.to_numpy(dtype=bool)
        return values.astype(str).str.strip().str.lower().isin({"true", "1", "1.0"}).to_numpy()
    return np.isfinite(pd.to_numeric(rows["aipw_score"], errors="coerce").to_numpy(dtype=float))


def confirm_fold_modifiers(
    rows: pd.DataFrame,
    confirmation: ConfirmationInput,
    *,
    policy: Mapping[str, Any],
    propensity_clip: float,
) -> dict[str, Any]:
    """Re-test one fold's selected modifiers on its held-out rows (``policy`` resolved)."""

    from . import stage2_modifier_tests as modifier_tests

    rows = rows.reset_index(drop=True)
    labels = {
        name: pd.to_numeric(rows[name], errors="coerce").to_numpy(dtype=float)
        for name in ("treatment", "outcome", "propensity", "mu0", "mu1")
    }
    finite = np.logical_and.reduce([np.isfinite(values) for values in labels.values()])
    eligible = _effect_eligibility(rows) & finite
    definitions = list(confirmation.definitions)
    result: dict[str, Any] = {
        "status": "ok",
        "selected_modifiers": len(definitions),
        "confirmed_modifiers": 0,
        "heldout_rows": len(rows),
        "eligible_rows": int(eligible.sum()),
        "rows_with_nonfinite_inputs": int((~finite).sum()),
        "measurement_source": dict(confirmation.sources),
        "excluded_oracle_named_features": list(confirmation.excluded_oracle_features),
        "missing_measurement_columns": list(confirmation.missing_measurement_columns),
        "modifiers": [],
    }
    if not definitions:
        return result
    if confirmation.frame is None:
        return unavailable("held-out measurements of the selected modifiers are missing")
    try:
        joined = _join_rows(rows, confirmation.frame).loc[finite].reset_index(drop=True)
        records, audit = modifier_tests.candidate_tests(
            joined,
            definitions,
            **{name: values[finite] for name, values in labels.items()},
            binary=policy["outcome_type"] == "binary",
            propensity_clip=propensity_clip,
            eligible=eligible[finite],
            min_count=policy["min_count"],
        )
    except ValueError as error:
        return unavailable(f"held-out modifier tests could not run: {error}")
    records, confirmed = modifier_tests.combine_and_select(
        records,
        weights=policy["weights"],
        q=policy["q"],
        binary=policy["outcome_type"] == "binary",
        family_ids=[record["feature_id"] for record in records],
        max_modifiers=None,
    )
    training = {
        str(feature.get("feature_id") or feature["name"]): feature.get("modifier_evidence")
        for feature in definitions
    }
    by_id = {str(feature.get("feature_id") or feature["name"]): feature for feature in definitions}
    scale = (
        "risk difference per unit" if policy["outcome_type"] == "binary"
        else "outcome difference per unit"
    )
    modifiers = []
    for record in records:
        components = record["components"]
        evidence = training.get(record["feature_id"])
        definition = by_id.get(record["feature_id"], {})
        modifiers.append({
            "feature_id": record["feature_id"],
            "name": record["name"],
            "normalized_name": normalized_feature_name(record["name"]),
            "value_type": definition.get("value_type"),
            "definition_text": _definition_text(definition),
            "effects": {
                "scale": scale,
                "training_sd": confirmation.training_scales.get(record["name"]),
                **{
                    name: projection_effects(
                        components[name], record.get("grouped_columns") or [],
                        training_sd=confirmation.training_scales.get(record["name"]),
                    )
                    for name in ("r_projection", "aipw_projection")
                },
            },
            "confirmed": bool(record["selected"]),
            "combined_p_value": record["combined_p_value"],
            "combined_q_value": record["combined_q_value"],
            "combination_weights": record["combination_weights"],
            "component_p_values": {
                name: components[name].get("p_value") for name in modifier_tests.COMPONENTS
            },
            "component_q_values": {
                name: components[name].get("q_value") for name in modifier_tests.COMPONENTS
            },
            "component_status": {
                name: (
                    "ok" if components[name].get("p_value") is not None
                    else str(components[name].get("reason") or "not_evaluable")
                )
                for name in modifier_tests.COMPONENTS
            },
            "labels": list(record["labels"]),
            "extreme_propensity_signal": bool(record["extreme_propensity_signal"]),
            "n_eligible": record["n_eligible"],
            "grouped_columns": list(record.get("grouped_columns") or []),
            "dropped_sparse_columns": list(record.get("dropped_sparse_columns") or []),
            "training_evidence": (
                {key: evidence.get(key) for key in
                 ("combined_p_value", "combined_q_value", "labels", "investigator_locked")}
                if isinstance(evidence, Mapping) else None
            ),
        })
    result.update(
        confirmed_modifiers=len(confirmed),
        tests_evaluable=bool(audit["evaluable"]),
        modifiers=modifiers,
    )
    return result


# Conservative: a false merge would combine p-values of different concepts, while
# a missed merge only forgoes aggregation. Lexical overlap separates concepts
# imperfectly: distinct variables sharing most words (for example brain versus
# liver metastases) can score about 0.65, and differently worded synonyms can
# score below 0.6. Complete linkage requires every cross-fold pair to pass.
CONCEPT_SIMILARITY_THRESHOLD = 0.7
CONCEPT_SIMILARITY_METHOD = (
    "mean of the TF-IDF cosine similarities of word 1-2 grams and character 3-5 grams "
    "(within word boundaries) of the normalized name, description and measurement "
    "definition, fitted on the selected variables of all folds; identical normalized "
    "names score 1; groups merge by complete linkage at a conservative threshold, so "
    "differently worded synonyms can stay separate"
)
MMB_GAMMA = 0.5


def _definition_text(feature: Mapping[str, Any]) -> str:
    parts = [feature.get("name"), feature.get("description"), feature.get("measurement_definition")]
    text = " ".join(str(part) for part in parts if part)
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _column_kind(column: str) -> tuple[str, str | None]:
    """Kind of a grouped-design column (value, level, fallback, missing, documented
    unknown, failure) and its level."""

    suffix = column.rsplit(":", 1)[-1]
    if suffix == "value":
        return "value", None
    if suffix == "missing":
        return "missing_indicator", None
    if suffix == "documented_unknown":
        return "documented_unknown_indicator", None
    if suffix == "extraction_failed":
        return "extraction_failure_indicator", None
    for prefix, kind in (("level=", "level"), ("fallback=", "fallback_level")):
        if suffix.startswith(prefix):
            return kind, suffix[len(prefix):]
    return "other", None


def projection_effects(
    component: Mapping[str, Any], grouped_columns: Sequence[str], *, training_sd: float | None
) -> dict[str, Any]:
    """Per-column projection coefficients with HC3 standard errors and 95% intervals.

    For ``r_projection`` a coefficient is the change in the absolute treatment
    effect per unit of the column; for ``aipw_projection`` the change in the
    AIPW pseudo-outcome. Continuous value columns also get the change per
    training SD.
    """

    if component.get("p_value") is None:
        return unavailable(str(component.get("reason") or "component not evaluable"))
    indices = [int(i) for i in component.get("tested_column_indices") or []]
    coefficients = list(component.get("coefficients") or [])
    errors = list(component.get("standard_errors") or [])
    if len(indices) != len(coefficients) or len(errors) != len(coefficients):
        return unavailable("component did not record per-column coefficients")
    columns = []
    for index, coefficient, error in zip(indices, coefficients, errors):
        name = str(grouped_columns[index]) if 0 <= index < len(grouped_columns) else str(index)
        kind, level = _column_kind(name)
        coefficient, error = _float(coefficient), _float(error)
        z = _ratio(coefficient, error) if error else None
        entry = {
            "column": name,
            "kind": kind,
            "level": level,
            "coefficient": coefficient,
            "standard_error": error,
            "confidence_interval_95": _interval(coefficient, error),
            "p_value_two_sided": _p_two_sided(z),
        }
        if kind == "value" and training_sd and coefficient is not None:
            entry["per_training_sd"] = {
                "coefficient": coefficient * training_sd,
                "standard_error": error * training_sd if error is not None else None,
                "confidence_interval_95": _interval(
                    coefficient * training_sd,
                    error * training_sd if error is not None else None,
                ),
            }
        columns.append(entry)
    return {"status": "ok", "covariance": "HC3", "columns": columns}


def concept_similarities(texts: Sequence[str], names: Sequence[str]) -> np.ndarray:
    """Deterministic lexical similarity matrix (see CONCEPT_SIMILARITY_METHOD)."""

    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = [text or name for text, name in zip(texts, names)]
    if len(texts) < 2:
        return np.ones((len(texts), len(texts)))
    matrices = []
    for analyzer, ngrams in (("word", (1, 2)), ("char_wb", (3, 5))):
        vectorizer = TfidfVectorizer(analyzer=analyzer, ngram_range=ngrams, sublinear_tf=True)
        vectors = vectorizer.fit_transform(texts)
        matrices.append(np.asarray((vectors @ vectors.T).todense(), dtype=float))
    similarity = np.mean(matrices, axis=0)
    normalized = [normalized_feature_name(name) for name in names]
    for i in range(len(texts)):
        for j in range(len(texts)):
            if normalized[i] == normalized[j]:
                similarity[i, j] = 1.0
    return similarity


def match_concepts(
    entries: Sequence[Mapping[str, Any]], *, threshold: float = CONCEPT_SIMILARITY_THRESHOLD
) -> list[dict[str, Any]]:
    """Group selected variables across folds; at most one variable per fold per concept.

    Greedy complete linkage: cross-fold pairs are taken in descending
    similarity and two groups merge only when every cross pair is at least
    ``threshold`` and no fold would appear twice. ``entries`` need ``fold``,
    ``name`` and ``definition_text``.
    """

    entries = list(entries)
    if not entries:
        return []
    similarity = concept_similarities(
        [str(e.get("definition_text") or "") for e in entries], [str(e["name"]) for e in entries]
    )
    group_of = list(range(len(entries)))
    members = {i: {i} for i in range(len(entries))}
    pairs = sorted(
        ((float(similarity[i, j]), i, j) for i in range(len(entries)) for j in range(i + 1, len(entries))
         if entries[i]["fold"] != entries[j]["fold"]),
        key=lambda item: (-item[0], item[1], item[2]),
    )
    for score, i, j in pairs:
        if score < threshold:
            break
        a, b = group_of[i], group_of[j]
        if a == b:
            continue
        folds_a = {entries[k]["fold"] for k in members[a]}
        folds_b = {entries[k]["fold"] for k in members[b]}
        if folds_a & folds_b:
            continue
        if min(similarity[x, y] for x in members[a] for y in members[b]) < threshold:
            continue
        members[a] |= members.pop(b)
        for k in members[a]:
            group_of[k] = a
    groups = sorted(
        (sorted(group, key=lambda k: _fold_key(entries[k]["fold"])) for group in members.values()),
        key=lambda group: (_fold_key(entries[group[0]]["fold"]), str(entries[group[0]]["name"])),
    )
    concepts = []
    for number, group in enumerate(groups, 1):
        pairwise = [
            float(similarity[x, y]) for a, x in enumerate(group) for y in group[a + 1:]
        ]
        concepts.append({
            "concept_id": f"concept_{number:03d}",
            "member_indices": group,
            "minimum_similarity": min(pairwise) if pairwise else None,
            "pairwise_similarities": {
                f"{entries[x]['fold']}:{entries[y]['fold']}": float(similarity[x, y])
                for a, x in enumerate(group) for y in group[a + 1:]
            },
        })
    return concepts


def mmb_quantile_aggregation(p_values: Sequence[float | None], *, gamma: float = MMB_GAMMA) -> float:
    """Meinshausen-Meier-Buhlmann fixed-gamma aggregation: min(1, quantile_gamma(p) / gamma).

    Missing p-values count as 1. With gamma = 0.5 this is min(1, 2 * median).
    """

    values = np.array([1.0 if p is None else float(p) for p in p_values], dtype=float)
    if not len(values):
        return 1.0
    return float(min(1.0, np.quantile(values, gamma) / gamma))


def fixed_effect_pool(estimates: Sequence[tuple[float, float]]) -> dict[str, Any]:
    """Inverse-variance pooled estimate with Cochran's Q, I^2 and the fold range."""

    usable = [(float(b), float(se)) for b, se in estimates
              if _float(b) is not None and _float(se) is not None and float(se) > 0]
    if len(usable) < 2:
        return unavailable("fewer than two folds have an estimate with a positive standard error")
    b = np.array([value for value, _ in usable])
    w = np.array([1.0 / se ** 2 for _, se in usable])
    pooled = float(np.sum(w * b) / np.sum(w))
    se = float(1.0 / math.sqrt(np.sum(w)))
    q = float(np.sum(w * (b - pooled) ** 2))
    df = len(usable) - 1
    from scipy.stats import chi2

    return {
        "status": "ok",
        "estimate": pooled,
        "standard_error": se,
        "confidence_interval_95": _interval(pooled, se),
        "folds": len(usable),
        "cochran_q": q,
        "q_degrees_of_freedom": df,
        "q_p_value": float(chi2.sf(q, df)),
        "i_squared": float(max(0.0, (q - df) / q)) if q > 0 else 0.0,
        "fold_range": [float(b.min()), float(b.max())],
    }


def _pooled_concept_effects(members: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pool matched folds' effects when every member has the same single or level columns."""

    signatures = []
    for member in members:
        effects = member.get("effects") or {}
        columns = (effects.get("r_projection") or {}).get("columns") or []
        kinds = sorted(
            (c["kind"], normalized_feature_name(c["level"] or ""))
            for c in columns if c["kind"] in {"value", "level"}
        )
        signatures.append(tuple(kinds))
    if len(members) < 2:
        return unavailable("selected in fewer than two folds")
    if not signatures[0] or len(set(signatures)) != 1:
        return unavailable(
            "folds differ in the modifier's columns (value type or category set), so the "
            "effects are reported per fold only"
        )
    pooled: dict[str, Any] = {"status": "ok", "columns": {}}
    for kind, level in signatures[0]:
        key = "value_per_training_sd" if kind == "value" else f"level={level}"
        pooled["columns"][key] = {}
        for component in ("r_projection", "aipw_projection"):
            estimates, missing_scale = [], False
            for member in members:
                columns = ((member.get("effects") or {}).get(component) or {}).get("columns") or []
                for column in columns:
                    if column["kind"] != kind or normalized_feature_name(column["level"] or "") != level:
                        continue
                    source = column.get("per_training_sd") if kind == "value" else column
                    if source:
                        estimates.append((source.get("coefficient"), source.get("standard_error")))
                    else:
                        missing_scale = True
            pooled["columns"][key][component] = (
                unavailable("a matched fold has no training SD for this continuous modifier, "
                            "so per-unit effects with possibly different units are not pooled")
                if missing_scale else fixed_effect_pool(estimates)
            )
    pooled["note"] = (
        "continuous modifiers are pooled per training SD (each fold's own definition and "
        "units); categorical levels are pooled by matching normalized level labels"
    )
    return pooled


def _concept_summary(ok: Mapping[str, Mapping[str, Any]], all_folds: Sequence[str], q: float):
    entries = [
        {**modifier, "fold": fold}
        for fold, result in ok.items()
        for modifier in result.get("modifiers") or []
    ]
    concepts = match_concepts(entries)
    aggregated = []
    for concept in concepts:
        members = [entries[k] for k in concept["member_indices"]]
        by_fold = {member["fold"]: member for member in members}
        p_by_fold = {
            fold: (by_fold[fold]["combined_p_value"] if fold in by_fold else None)
            for fold in all_folds
        }
        aggregated.append({
            "concept_id": concept["concept_id"],
            "members": [
                {key: member.get(key) for key in (
                    "fold", "feature_id", "name", "normalized_name", "definition_text",
                    "confirmed", "combined_p_value", "combined_q_value")}
                for member in members
            ],
            "selected_in_folds": [member["fold"] for member in members],
            "confirmed_in_folds": [member["fold"] for member in members if member["confirmed"]],
            "folds_selected": len(members),
            "folds_confirmed": sum(bool(member["confirmed"]) for member in members),
            "minimum_similarity": concept["minimum_similarity"],
            "pairwise_similarities": concept["pairwise_similarities"],
            "heldout_p_values_by_fold": p_by_fold,
            "aggregated_p_value": mmb_quantile_aggregation(list(p_by_fold.values())),
            "pooled_effects": _pooled_concept_effects(members),
        })
    from .stage2_elastic_net_selection import _benjamini_hochberg

    for item, q_value in zip(
        aggregated, _benjamini_hochberg([item["aggregated_p_value"] for item in aggregated])
    ):
        item["aggregated_q_value"] = q_value
        item["aggregated_supported"] = q_value is not None and q_value <= q
    aggregated.sort(key=lambda item: (item["aggregated_p_value"], item["concept_id"]))
    return {
        "note": (
            "Each fold defines its own variables. Variables are grouped into concepts by "
            "name and definition similarity (at most one variable per fold per concept); a "
            "concept selected in one fold is never tested on another fold's held-out rows, "
            "which were that fold's training data. Each fold's confirmation is valid "
            "conditional on that fold's own selection. The Meinshausen-Meier-Buhlmann "
            "aggregated p-value is min(1, 2 * median) over all folds, with p = 1 where the "
            "concept was not selected or not testable; Benjamini-Hochberg then runs across "
            "concepts. The aggregation treats the folds' tests as tests of one concept, "
            "which holds only approximately when folds define the variable differently."
        ),
        "similarity_method": CONCEPT_SIMILARITY_METHOD,
        "similarity_threshold": CONCEPT_SIMILARITY_THRESHOLD,
        "aggregation": "meinshausen_meier_buhlmann_fixed_gamma",
        "gamma": MMB_GAMMA,
        "folds": list(all_folds),
        "concepts": aggregated,
    }


def _confirmation_section(
    predictions: pd.DataFrame,
    artifacts: Mapping[str, FoldArtifacts],
    *,
    policy: Mapping[str, Any] | None,
    propensity_clip: float,
) -> dict[str, Any]:
    from . import stage2_modifier_tests as modifier_tests

    header = {
        "interpretation": CONFIRMATION_INTERPRETATION,
        "pooling_rule": CONFIRMATION_POOLING_RULE,
        "p_value_role": modifier_tests.CONFIRMATION_P_VALUE_ROLE,
    }
    if policy is None:
        return {**header, **unavailable("no modifier confirmation policy was supplied")}
    try:
        resolved = resolve_confirmation_policy(policy)
    except (KeyError, TypeError, ValueError) as error:
        return {**header, **unavailable(f"invalid modifier confirmation policy: {error}")}
    resolved["propensity_clip"] = float(propensity_clip)
    per_fold: dict[str, Any] = {}
    for fold, rows in predictions.groupby("outer_fold", sort=True):
        fold = str(int(fold))
        source = artifacts.get(fold)
        item = source.confirmation if source is not None else "no fold artifacts were found"
        if isinstance(item, str):
            per_fold[fold] = unavailable(item)
            continue
        per_fold[fold] = _guarded(
            f"modifier confirmation fold {fold}", confirm_fold_modifiers,
            rows, item, policy=resolved, propensity_clip=propensity_clip,
        )
    per_fold = dict(sorted(per_fold.items(), key=lambda entry: _fold_key(entry[0])))
    ok = {fold: result for fold, result in per_fold.items() if result.get("status") == "ok"}
    if not ok:
        return {
            **header, "policy": resolved, "per_fold": per_fold,
            **unavailable("no outer fold has modifier confirmation inputs"),
        }
    return {
        **header,
        "status": "ok",
        "policy": resolved,
        "per_fold": per_fold,
        "pooled": {
            "folds": sorted(ok, key=_fold_key),
            "selected_by_fold": {fold: r["selected_modifiers"] for fold, r in ok.items()},
            "confirmed_by_fold": {fold: r["confirmed_modifiers"] for fold, r in ok.items()},
            "total_selected": int(sum(r["selected_modifiers"] for r in ok.values())),
            "total_confirmed": int(sum(r["confirmed_modifiers"] for r in ok.values())),
        },
        "across_folds": _concept_summary(ok, sorted(per_fold, key=_fold_key), resolved["q"]),
    }


def _r_squared(target: np.ndarray, design: np.ndarray) -> float:
    """OLS R^2 with an intercept; an empty or constant design explains nothing."""

    design = np.asarray(design, dtype=float).reshape(len(target), -1)
    columns = [design[:, j] for j in range(design.shape[1]) if not _is_constant(design[:, j])]
    if not columns:
        return 0.0
    x = np.column_stack([np.ones(len(target)), *columns])
    beta = np.linalg.lstsq(x, target, rcond=None)[0]
    residual = target - x @ beta
    total = float(np.sum((target - target.mean()) ** 2))
    return float(min(max(1.0 - float(residual @ residual) / total, 0.0), 1.0))


def cate_variance_decomposition(
    cate: Any, prognosis: Any, modifiers: Any, *, min_rows: int = 10
) -> dict[str, Any]:
    """Partition CATE variance between the prognostic score and the modifier design.

    ``modifiers`` is an ``(n, p)`` design (``p`` may be 0). Returns the three
    R^2 values and the unique, shared and unexplained shares (see
    ``DECOMPOSITION_INTERPRETATION``).
    """

    cate = np.asarray(cate, dtype=float)
    prognosis = np.asarray(prognosis, dtype=float)
    modifiers = np.asarray(modifiers, dtype=float).reshape(len(cate), -1)
    finite = np.isfinite(cate) & np.isfinite(prognosis) & np.isfinite(modifiers).all(axis=1)
    cate, prognosis, modifiers = cate[finite], prognosis[finite], modifiers[finite]
    rows, columns = len(cate), modifiers.shape[1]
    if rows < max(int(min_rows), columns + 3):
        return unavailable("too few rows for the prognostic score and the modifier design",
                           rows=rows, modifier_columns=columns)
    if _is_constant(cate):
        return unavailable("the estimated CATE is constant, so there is no variance to "
                           "decompose", rows=rows, cate_variance=0.0)
    r2_prognosis = _r_squared(cate, prognosis)
    r2_modifiers = _r_squared(cate, modifiers)
    r2_both = _r_squared(cate, np.column_stack([prognosis, modifiers]))
    return {
        "status": "ok",
        "rows": rows,
        "modifier_columns": columns,
        "cate_variance": float(np.var(cate, ddof=1)),
        "cate_sd": float(np.std(cate, ddof=1)),
        "r2_prognosis": r2_prognosis,
        "r2_modifiers": r2_modifiers,
        "r2_both": r2_both,
        "unique_prognosis": r2_both - r2_modifiers,
        "unique_modifiers": r2_both - r2_prognosis,
        "shared": r2_prognosis + r2_modifiers - r2_both,
        "unexplained": 1.0 - r2_both,
    }


def _decompose_fold(
    rows: pd.DataFrame,
    confirmation: ConfirmationInput | str,
    *,
    columns: Sequence[str],
    binary: bool,
    min_count: int,
) -> dict[str, Any]:
    from . import stage2_modifier_tests as modifier_tests
    from .stage2_effect_estimators import prognostic_score

    if isinstance(confirmation, str):
        return unavailable(confirmation)
    rows = rows.reset_index(drop=True)
    mu0 = pd.to_numeric(rows["mu0"], errors="coerce").to_numpy(dtype=float)
    definitions = list(confirmation.definitions)
    if definitions and confirmation.frame is None:
        return unavailable("held-out measurements of the selected modifiers are missing")
    joined = _join_rows(rows, confirmation.frame) if definitions else None
    result: dict[str, Any] = {
        "status": "ok",
        "heldout_rows": len(rows),
        "selected_modifiers": [str(feature["name"]) for feature in definitions],
        "missing_measurement_columns": list(confirmation.missing_measurement_columns),
        "measurement_source": dict(confirmation.sources),
        "columns": {},
    }
    for column in columns:
        if column not in rows.columns:
            continue
        cate = pd.to_numeric(rows[column], errors="coerce").to_numpy(dtype=float)
        keep = np.isfinite(cate) & np.isfinite(mu0)
        if not keep.any():
            result["columns"][column] = unavailable("no held-out row has this CATE")
            continue
        design_parts, names, dropped = [], [], []
        for feature in definitions:
            design, feature_names, feature_dropped = modifier_tests.grouped_design(
                joined.loc[keep].reset_index(drop=True), feature, min_count=min_count
            )
            design_parts.append(design)
            names.extend(feature_names)
            dropped.extend(feature_dropped)
        design = (
            np.column_stack(design_parts) if design_parts
            else np.empty((int(keep.sum()), 0), dtype=float)
        )
        decomposition = cate_variance_decomposition(
            cate[keep], prognostic_score(mu0[keep], binary=binary), design
        )
        result["columns"][column] = {
            **decomposition, "modifier_design_columns": names, "dropped_sparse_columns": dropped,
        }
    return result


def _pool_decompositions(per_fold: Mapping[str, Any], column: str) -> dict[str, Any]:
    usable = {
        fold: result["columns"][column]
        for fold, result in per_fold.items()
        if result.get("status") == "ok"
        and (result.get("columns") or {}).get(column, {}).get("status") == "ok"
    }
    if not usable:
        return unavailable("no fold has a decomposable CATE in this column")
    weights = np.array([item["rows"] for item in usable.values()], dtype=float)
    pooled: dict[str, Any] = {
        "status": "ok",
        "folds": sorted(usable, key=_fold_key),
        "rows": int(weights.sum()),
        "range": {},
    }
    for key in DECOMPOSITION_KEYS:
        values = np.array([item[key] for item in usable.values()], dtype=float)
        pooled[key] = float(np.sum(weights * values) / weights.sum())
        pooled["range"][key] = [float(values.min()), float(values.max())]
    return pooled


def _decomposition_section(
    predictions: pd.DataFrame,
    artifacts: Mapping[str, FoldArtifacts],
    *,
    policy: Mapping[str, Any] | None,
) -> dict[str, Any]:
    header = {
        "interpretation": DECOMPOSITION_INTERPRETATION,
        "pooling_rule": DECOMPOSITION_POOLING_RULE,
        "prognostic_score": (
            "logit of the held-out mu0 (binary outcomes) or mu0 (continuous outcomes), "
            "as in the causal forest's X and the interaction model's adjustment"
        ),
    }
    if policy is None:
        return {**header, **unavailable(
            "no outcome type was supplied (the modifier confirmation policy is missing)")}
    try:
        resolved = resolve_confirmation_policy(policy)
    except (KeyError, TypeError, ValueError) as error:
        return {**header, **unavailable(f"invalid modifier confirmation policy: {error}")}
    if "mu0" not in predictions.columns:
        return {**header, **unavailable("cross-fitted predictions lack mu0")}
    columns = ["estimated_cate", *(c for c in CATE_COMPONENT_COLUMNS if c in predictions.columns)]
    per_fold: dict[str, Any] = {}
    for fold, rows in predictions.groupby("outer_fold", sort=True):
        fold = str(int(fold))
        source = artifacts.get(fold)
        item = source.confirmation if source is not None else "no fold artifacts were found"
        per_fold[fold] = _guarded(
            f"cate variance decomposition fold {fold}", _decompose_fold, rows, item,
            columns=columns, binary=resolved["outcome_type"] == "binary",
            min_count=resolved["min_count"],
        )
    per_fold = dict(sorted(per_fold.items(), key=lambda entry: _fold_key(entry[0])))
    pooled = {column: _pool_decompositions(per_fold, column) for column in columns}
    status = "ok" if any(item.get("status") == "ok" for item in pooled.values()) else "unavailable"
    return {
        **header,
        "status": status,
        **({} if status == "ok" else {"reason": "no fold has a decomposable CATE"}),
        "outcome_type": resolved["outcome_type"],
        "columns": columns,
        "per_fold": per_fold,
        "pooled": pooled,
    }


def build_heterogeneity_diagnostics(
    predictions: pd.DataFrame,
    *,
    propensity_clip: float,
    seed: int,
    fold_artifacts: Mapping[int, FoldArtifacts] | None = None,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_REPLICATES,
    groups: int = CATE_GROUPS,
    modifier_confirmation_policy: Mapping[str, Any] | None = None,
    overlap_flag_bounds: Sequence[float] = DEFAULT_OVERLAP_FLAG_BOUNDS,
    outcome_type: str | None = None,
) -> dict[str, Any]:
    """Compute every diagnostic from the pooled cross-fitted prediction table.

    ``modifier_confirmation_policy`` (``outcome_type``, ``q``, ``weights`` or
    None for the outcome-type defaults, ``min_count``) enables the held-out
    modifier confirmation; without it that section is ``unavailable``.
    ``outcome_type`` (else the policy's) sets the E-value scale; when neither is
    given, an outcome with only the values 0 and 1 is treated as binary.
    """

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
        eligible_part = part[np.isfinite(part["aipw_score"].to_numpy(dtype=float))]
        return {
            "overlap_weighted_ate": _guarded("ato", overlap_weighted_effects, *args),
            "ipw_weights": _guarded("ipw weights", inverse_propensity_weight_summary,
                                    part["treatment"], part["propensity"], clip=propensity_clip),
            "trimming_curve": _guarded("trimming", aipw_trimming_curve, *args,
                                       clip=propensity_clip),
            "clip_sensitivity": _guarded(
                "clip sensitivity", clip_sensitivity,
                *(eligible_part[c] for c in ("outcome", "treatment", "propensity", "mu0", "mu1")),
                configured_clip=propensity_clip,
            ),
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
    diagnostics["overlap_flags"] = {
        "interpretation": INTERPRETATIONS["overlap_flags"],
        **_guarded("overlap flags", overlap_flag_summary, predictions, overlap_flag_bounds),
    }
    for key in ("overlap_weighted_ate", "ipw_weights", "trimming_curve", "clip_sensitivity"):
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
    confirmation_rows = predictions.copy()
    confirmation_rows[ROW_ID] = pd.to_numeric(confirmation_rows[ROW_ID]).astype(int)
    confirmation_rows["outer_fold"] = pd.to_numeric(confirmation_rows["outer_fold"]).astype(int)
    diagnostics["modifier_confirmation"] = _guarded(
        "modifier confirmation", _confirmation_section, confirmation_rows, artifacts,
        policy=modifier_confirmation_policy, propensity_clip=propensity_clip,
    )
    diagnostics["cate_variance_decomposition"] = _guarded(
        "cate variance decomposition", _decomposition_section, confirmation_rows, artifacts,
        policy=modifier_confirmation_policy,
    )
    resolved_outcome_type = outcome_type or (modifier_confirmation_policy or {}).get(
        "outcome_type"
    )
    if resolved_outcome_type is None:
        observed = frame["outcome"].dropna().unique()
        resolved_outcome_type = (
            "binary" if len(observed) and set(observed) <= {0.0, 1.0} else "continuous"
        )
    diagnostics["unmeasured_confounding_sensitivity"] = _guarded(
        "unmeasured confounding sensitivity", unmeasured_confounding_section,
        confirmation_rows, outcome_type=str(resolved_outcome_type),
        propensity_clip=propensity_clip,
        benchmarks_by_fold={
            fold: value.sensitivity_benchmarks for fold, value in artifacts.items()
        },
    )
    diagnostics["primary_heterogeneity_test"] = primary_heterogeneity_test(diagnostics)
    return json_safe(diagnostics)


def _pick(mapping: Any, *keys: str) -> dict[str, Any]:
    mapping = mapping if isinstance(mapping, Mapping) else {}
    return {key: mapping.get(key) for key in keys if key in mapping}


PRIMARY_HETEROGENEITY_TEST = (
    "Pre-specified primary heterogeneity test: the pooled BLP beta2 (fixed-effect "
    "inverse-variance pooling of the outer folds), one-sided test of beta2 > 0. GATES "
    "and RATE are secondary; CLAN, the CATE variance decomposition and the modifier "
    "confirmation are descriptive. No diagnostic is used for selection or estimation."
)


def primary_heterogeneity_test(diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    """The pooled BLP beta2 > 0 test, labeled as the pre-specified primary test."""

    section = diagnostics.get("blp")
    pooled = section.get("pooled", {}) if isinstance(section, Mapping) else {}
    if not isinstance(pooled, Mapping) or pooled.get("status") != "ok":
        return {"description": PRIMARY_HETEROGENEITY_TEST, "pre_specified": True,
                **unavailable((pooled or {}).get("reason") or "pooled BLP is unavailable")}
    return {
        "status": "ok",
        "description": PRIMARY_HETEROGENEITY_TEST,
        "pre_specified": True,
        "test": "pooled_blp_beta2_greater_than_0_one_sided",
        "beta2": pooled.get("beta2_heterogeneity"),
        "standard_error": pooled.get("beta2_standard_error"),
        "confidence_interval_95": pooled.get("beta2_confidence_interval_95"),
        "p_value_one_sided": pooled.get("beta2_p_value_one_sided_greater_than_0"),
        "folds": pooled.get("folds"),
        "secondary_tests": ["gates", "rate"],
        "descriptive": ["clan", "cate_variance_decomposition", "modifier_confirmation"],
    }


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
        "primary_heterogeneity_test": primary_heterogeneity_test(diagnostics),
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
            "group_mean_estimated_cate": [
                group.get("mean_estimated_cate") for group in gates.get("groups") or []
            ],
            "calibration": _pick(
                gates.get("calibration"), "status", "slope", "standard_error",
                "p_value_one_sided_greater_than_0", "p_value_two_sided_equal_1",
            ),
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
        "clip_sensitivity_pooled": [
            _pick(point, "clip", "configured", "estimate", "standard_error")
            for point in pooled("clip_sensitivity").get("points") or []
        ],
        "overlap_flags_pooled": {
            **_pick(diagnostics.get("overlap_flags"), "status", "bounds"),
            **_pick((diagnostics.get("overlap_flags") or {}).get("pooled"),
                    "rows", "outside_overlap", "fraction"),
        },
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
        "modifier_confirmation": _confirmation_summary(diagnostics.get("modifier_confirmation")),
        "cate_variance_decomposition": _decomposition_summary(
            diagnostics.get("cate_variance_decomposition")
        ),
        "unmeasured_confounding_sensitivity": summarize_sensitivity_section(
            diagnostics.get("unmeasured_confounding_sensitivity")
        ),
    }


def _decomposition_summary(section: Any) -> dict[str, Any]:
    section = section if isinstance(section, Mapping) else {}
    pooled = section.get("pooled") or {}
    return {
        **_pick(section, "status", "reason"),
        **{
            column: _pick(value, "status", "folds", "rows", *DECOMPOSITION_KEYS)
            for column, value in pooled.items()
            if isinstance(value, Mapping)
        },
    }


def _confirmation_summary(section: Any) -> dict[str, Any]:
    section = section if isinstance(section, Mapping) else {}
    pooled = section.get("pooled") or {}
    concepts = (section.get("across_folds") or {}).get("concepts") or []
    return {
        **_pick(section, "status", "reason"),
        **_pick(pooled, "total_selected", "total_confirmed", "selected_by_fold",
                "confirmed_by_fold"),
        "q": (section.get("policy") or {}).get("q"),
        "concepts": [
            {
                "concept_id": concept.get("concept_id"),
                "names": [member.get("name") for member in concept.get("members") or []],
                "folds_selected": concept.get("folds_selected"),
                "folds_confirmed": concept.get("folds_confirmed"),
                "aggregated_p_value": concept.get("aggregated_p_value"),
                "aggregated_q_value": concept.get("aggregated_q_value"),
            }
            for concept in concepts
            if isinstance(concept, Mapping)
        ],
        "concepts_supported_after_aggregation": [
            concept.get("concept_id") for concept in concepts
            if isinstance(concept, Mapping) and concept.get("aggregated_supported")
        ],
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
    confirmation = _confirmation_input(
        retained, final_path,
        (candidate_heldout, heldout_retained),
        mask=heldout_mask,
        training_paths=(candidate_fit, fold_dir / "extraction" / "fit" / "extracted.csv"),
    )
    return FoldArtifacts(
        clan=clan, balance_training=training, balance_heldout=heldout,
        confirmation=confirmation,
        sensitivity_benchmarks=load_fold_benchmarks(fold_dir),
    )


def _confirmation_input(
    retained: list[dict[str, Any]] | str,
    final_path: Path,
    measurement_paths: Sequence[Path],
    *,
    mask: Path,
    training_paths: Sequence[Path] = (),
) -> ConfirmationInput | str:
    """The fold's selected modifiers and the first held-out file that measures them."""

    result = _confirmation_measurements(retained, final_path, measurement_paths, mask=mask)
    if isinstance(result, ConfirmationInput) and result.definitions:
        result.training_scales = _training_scales(result.definitions, training_paths)
    return result


def _training_scales(
    definitions: Sequence[Mapping[str, Any]], paths: Sequence[Path]
) -> dict[str, float]:
    """SD of each continuous modifier's observed outer-training values (first file that has it)."""

    scales: dict[str, float] = {}
    wanted = [str(f["name"]) for f in definitions if _is_continuous(f)]
    for path in paths:
        if not wanted or not Path(path).is_file():
            continue
        try:
            frame = pd.read_csv(path, usecols=lambda column: column in wanted)
        except (OSError, ValueError):
            continue
        for name in list(wanted):
            if name in frame.columns:
                values = pd.to_numeric(frame[name], errors="coerce").dropna()
                if len(values) > 1 and float(values.std(ddof=1)) > 0:
                    scales[name] = float(values.std(ddof=1))
                wanted.remove(name)
    return scales


def _confirmation_measurements(
    retained: list[dict[str, Any]] | str,
    final_path: Path,
    measurement_paths: Sequence[Path],
    *,
    mask: Path,
) -> ConfirmationInput | str:

    if isinstance(retained, str):
        return retained
    selected = [
        feature for feature in retained
        if "effect_modifier" in set(map(str, feature.get("roles") or []))
    ]
    excluded = sorted(str(f["name"]) for f in selected if is_oracle_name(f.get("name")))
    selected = [feature for feature in selected if not is_oracle_name(feature.get("name"))]
    sources = {"definitions": str(final_path)}
    if not selected:
        return ConfirmationInput(definitions=[], frame=None, sources=sources,
                                 excluded_oracle_features=excluded)
    names = [str(feature["name"]) for feature in selected]
    fallback: tuple[Path, pd.DataFrame, list[str]] | None = None
    for path in measurement_paths:
        if not path.is_file():
            continue
        frame = _read_measurements(path, mask)
        frame = frame.drop(columns=[c for c in frame.columns if is_oracle_name(c)])
        missing = [name for name in names if name not in frame.columns]
        if not missing:
            return ConfirmationInput(
                definitions=selected, frame=frame,
                sources={**sources, "measurements": str(path)},
                excluded_oracle_features=excluded,
            )
        if fallback is None:
            fallback = (path, frame, missing)
    if fallback is None:
        return (
            "held-out measurements unavailable: "
            + ", ".join(str(path) for path in measurement_paths) + " are missing"
        )
    path, frame, missing = fallback
    return ConfirmationInput(
        definitions=selected, frame=frame,
        sources={**sources, "measurements": str(path)},
        excluded_oracle_features=excluded, missing_measurement_columns=missing,
    )


def write_heterogeneity_diagnostics(
    *,
    output_dir: Path,
    predictions: pd.DataFrame,
    propensity_clip: float,
    seed: int,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_REPLICATES,
    modifier_confirmation_policy: Mapping[str, Any] | None = None,
    overlap_flag_bounds: Sequence[float] = DEFAULT_OVERLAP_FLAG_BOUNDS,
    outcome_type: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Write ``heterogeneity_diagnostics.json`` and return (path, compact summary).

    Never raises for diagnostic failures: they are recorded in the file. Only
    ``heterogeneity_diagnostics.json`` is written; no other artifact changes.
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
                artifacts[fold] = FoldArtifacts(reason, reason, reason, reason, reason)
        diagnostics = build_heterogeneity_diagnostics(
            predictions,
            propensity_clip=propensity_clip,
            seed=seed,
            fold_artifacts=artifacts,
            n_bootstrap=n_bootstrap,
            modifier_confirmation_policy=modifier_confirmation_policy,
            overlap_flag_bounds=overlap_flag_bounds,
            outcome_type=outcome_type,
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
    "ConfirmationInput",
    "FoldArtifacts",
    "MeasurementInput",
    "aipw_trimming_curve",
    "best_linear_predictor",
    "build_heterogeneity_diagnostics",
    "cate_groups",
    "cate_variance_decomposition",
    "clip_sensitivity",
    "co_primary_estimands",
    "classification_analysis",
    "confirm_fold_modifiers",
    "dr_scores",
    "encode_measurements",
    "group_average_treatment_effects",
    "inverse_propensity_weight_summary",
    "inverse_variance_pool",
    "load_fold_artifacts",
    "normalized_feature_name",
    "outside_overlap_flags",
    "overlap_flag_summary",
    "overlap_weighted_effects",
    "pool_blp",
    "rank_weighted_ate_point",
    "rank_weighted_average_treatment_effects",
    "resolve_confirmation_policy",
    "stack_encoded",
    "summarize_diagnostics",
    "summarize_fold_balance",
    "weighted_standardized_mean_differences",
    "write_heterogeneity_diagnostics",
]
