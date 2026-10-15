"""Pre-specified modifier tests on one honest nuisance construction.

Every candidate variable is tested on its grouped design ``Z`` (the centered
value or the categorical levels, plus its missingness and extraction-failure
indicators), jointly, by three component tests:

* ``r_projection`` (absolute scale, overlap-weighted): ordinary least squares
  of ``Y - m_hat`` on ``[T - e_hat, (T - e_hat) * Z]`` without a penalty, HC3
  covariance, and a joint Wald test of the ``Z`` interaction block.
* ``aipw_projection`` (absolute scale, population-weighted): ordinary least
  squares of the AIPW pseudo-outcome ``psi`` on ``[1, Z]`` with HC3 covariance
  and a joint Wald test of the ``Z`` block.
* ``outcome_interaction`` (multiplicative for binary outcomes, additive for
  continuous outcomes): the adjusted interaction test
  ``Y ~ T + f(e_hat) + g(m_hat) + Z + T:Z`` versus the model without ``T:Z``
  (likelihood-ratio test for binary outcomes, partial F test otherwise).

Here ``m_hat = e_hat * mu1_hat + (1 - e_hat) * mu0_hat``. The component p-values
are combined with a weighted Cauchy combination and Benjamini-Hochberg runs
across candidates. Every function is pure: callers supply measurements, labels
and the cross-fitted nuisance predictions, so the same code serves training
selection and post-hoc held-out confirmation.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from . import stage2_interaction_screen as interaction_screen
from .stage2_elastic_net_selection import _benjamini_hochberg
from .stage2_statistical_selection import (
    _binary_nested_p_value,
    _continuous_nested_p_value,
    _encode_feature,
    _rank_safe_columns,
)

SCHEMA_VERSION = "stage2_orthogonal_modifier_tests_v1"
COMPONENTS = ("r_projection", "aipw_projection", "outcome_interaction")
ABSOLUTE_COMPONENTS = ("r_projection", "aipw_projection")
BINARY_WEIGHTS = {"r_projection": 0.25, "aipw_projection": 0.25, "outcome_interaction": 0.5}
CONTINUOUS_WEIGHTS = {name: 1.0 / 3.0 for name in COMPONENTS}
P_VALUE_CLAMP = 1e-15
LEVERAGE_LIMIT = 1.0 - 1e-8
_UNRELIABLE_WARNINGS = ("converg", "separation", "overflow", "divide by zero", "invalid value")

COMPONENT_DESCRIPTIONS = {
    "r_projection": (
        "OLS of (Y - m_hat) on [(T - e_hat), (T - e_hat) * Z] without a penalty; HC3 "
        "covariance; joint Wald chi-square test of the Z interaction block. Absolute "
        "scale, overlap-weighted (rows weigh in proportion to (T - e_hat)^2)."
    ),
    "aipw_projection": (
        "OLS of the AIPW pseudo-outcome psi = mu1_hat - mu0_hat + T(Y - mu1_hat)/e_hat "
        "- (1 - T)(Y - mu0_hat)/(1 - e_hat) (propensity clipped as in the AIPW ATE) on "
        "[1, Z]; HC3 covariance; joint Wald chi-square test of the Z block. Absolute "
        "scale, population-weighted."
    ),
    "outcome_interaction": (
        "Y ~ T + f(e_hat) + g(m_hat) + Z + T:Z versus the same model without T:Z, with "
        "natural-spline adjustment scores of the testing nuisances; likelihood-ratio "
        "chi-square test (logistic link) for binary outcomes, partial F test for "
        "continuous outcomes. Multiplicative scale for binary outcomes, additive for "
        "continuous outcomes."
    ),
}

INTERPRETATIONS = {
    "components": (
        "One p-value per component test of the candidate's grouped design Z (value or "
        "categorical levels plus missingness and extraction-failure indicators), "
        "tested jointly. status 'not_evaluable' gives the reason a component has no "
        "p-value (rank-deficient design, a row with leverage one, too few eligible "
        "rows, or an unreliable logistic fit)."
    ),
    "statistic": "Wald chi-square (projections), likelihood-ratio chi-square or partial F.",
    "degrees_of_freedom": "Number of independent Z columns tested jointly (F: [numerator, residual]).",
    "combined_p_value": (
        "Weighted Cauchy combination T = sum_k w_k tan((0.5 - p_k) pi), p = 0.5 - "
        "arctan(T)/pi over the components that have p-values, with weights renormalized "
        "over them. Exact under independence and approximately valid under arbitrary "
        "dependence between the components."
    ),
    "combined_q_value": (
        "Benjamini-Hochberg q-value of the combined p-value across the tested family "
        "(training: all candidates not locked by the investigator; confirmation: the "
        "fold's selected modifiers)."
    ),
    "component_q_value": (
        "Benjamini-Hochberg q-value of one component across the same family; used only "
        "for labels."
    ),
    "selected": (
        "Training selection: combined q-value at most q, ordered by combined p-value, "
        "subject to max_modifiers; investigator-locked modifiers are always included."
    ),
    "labels": (
        "absolute: r_projection or aipw_projection passes its own BH at q. "
        "multiplicative (binary outcomes) or additive (continuous outcomes): "
        "outcome_interaction passes. combined_only: selected although no single "
        "component passes. Labels describe the scale of the evidence, not causal roles."
    ),
    "extreme_propensity_signal": (
        "aipw_projection passes and r_projection does not. The population-weighted "
        "projection gives patients with extreme propensities far more weight than the "
        "overlap-weighted one, so the signal may rest on a few poorly overlapping "
        "patients."
    ),
    "n_rows": "Rows supplied to the tests (outer-training rows, or held-out rows for confirmation).",
    "n_eligible": "Propensity-eligible rows used by every component test.",
    "training_p_value_optimism": (
        "The candidate list was built from the same outer-training patients, so "
        "training p-values are somewhat optimistic; outer-fold confirmation repeats "
        "the tests on held-out patients."
    ),
}


def default_weights(outcome_type: str) -> dict[str, float]:
    """Scale-balanced component weights for the outcome type."""

    if outcome_type == "binary":
        return dict(BINARY_WEIGHTS)
    if outcome_type == "continuous":
        return dict(CONTINUOUS_WEIGHTS)
    raise ValueError("modifier tests require a binary or continuous outcome")


def _cauchy_term(p: float) -> float:
    """tan((0.5 - p) pi), evaluated as a cotangent near the poles for precision."""

    p = min(max(float(p), P_VALUE_CLAMP), 1.0 - P_VALUE_CLAMP)
    if p <= 0.5:
        return 1.0 / math.tan(p * math.pi)
    return -1.0 / math.tan((1.0 - p) * math.pi)


def cauchy_combination(
    p_values: Mapping[str, float | None], weights: Mapping[str, float]
) -> dict[str, Any]:
    """Weighted Cauchy combination over the components that have p-values.

    ``T = sum_k w_k tan((0.5 - p_k) pi)`` and ``p = 0.5 - arctan(T) / pi``, with
    p-values clamped to ``[1e-15, 1 - 1e-15]`` and the weights renormalized over
    the available components (Liu and Xie 2020).
    """

    available = [
        name for name in weights
        if p_values.get(name) is not None
        and math.isfinite(float(p_values[name]))
        and float(weights[name]) > 0
    ]
    if not available:
        return {"p_value": None, "statistic": None, "weights": {}, "components": []}
    total = math.fsum(float(weights[name]) for name in available)
    used = {name: float(weights[name]) / total for name in available}
    statistic = math.fsum(used[name] * _cauchy_term(p_values[name]) for name in available)
    # atan2(1, T) / pi equals 0.5 - arctan(T) / pi without cancellation for large T.
    p_value = math.atan2(1.0, statistic) / math.pi
    return {
        "p_value": float(min(max(p_value, 0.0), 1.0)),
        "statistic": float(statistic),
        "weights": used,
        "components": available,
    }


def grouped_design(
    frame: pd.DataFrame, feature: Mapping[str, Any], *, min_count: int
) -> tuple[np.ndarray, list[str], list[str]]:
    """The candidate's grouped design Z, fitted on the supplied rows.

    Indicator columns (0/1) with fewer than ``min_count`` rows on either side are
    dropped and reported: they cannot be estimated reliably and give rows with
    leverage one under HC3.
    """

    design = _encode_feature(frame.reset_index(drop=True), feature)
    columns, names, dropped = [], [], []
    for index, name in enumerate(design.main_names):
        column = design.main[:, index]
        if np.isin(column, (0.0, 1.0)).all():
            ones = int(column.sum())
            if min(ones, len(column) - ones) < int(min_count):
                dropped.append(str(name))
                continue
        columns.append(column)
        names.append(str(name))
    matrix = np.column_stack(columns) if columns else np.empty((len(frame), 0), dtype=float)
    return matrix, names, dropped


def json_ready(value: Any) -> Any:
    """Plain JSON values: numpy scalars/arrays converted, non-finite floats to None."""

    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_ready(item) for item in (value.tolist() if isinstance(value, np.ndarray) else value)]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def _not_evaluable(reason: str, **details: Any) -> tuple[None, dict[str, Any]]:
    return None, {"status": "not_evaluable", "reason": reason, **details}


def hc3_wald_test(
    target: np.ndarray, design: np.ndarray, tested: Sequence[int]
) -> tuple[float | None, dict[str, Any]]:
    """OLS with HC3 covariance and a joint Wald chi-square test of ``tested`` columns."""

    target = np.asarray(target, dtype=float)
    design = np.asarray(design, dtype=float)
    tested = [int(index) for index in tested]
    rows, columns = design.shape
    if not tested:
        return _not_evaluable("no_independent_candidate_columns")
    if rows - columns < 1:
        return _not_evaluable("insufficient_residual_degrees_of_freedom", rows=rows)
    try:
        gram_inverse = np.linalg.inv(design.T @ design)
    except np.linalg.LinAlgError:
        return _not_evaluable("singular_design")
    beta = gram_inverse @ (design.T @ target)
    residual = target - design @ beta
    leverage = np.einsum("ij,jk,ik->i", design, gram_inverse, design)
    if not np.isfinite(leverage).all() or float(np.max(leverage)) >= LEVERAGE_LIMIT:
        return _not_evaluable("row_with_leverage_one", rows=rows)
    omega = (residual / (1.0 - leverage)) ** 2
    covariance = gram_inverse @ (design.T @ (design * omega[:, None])) @ gram_inverse
    coefficients = beta[tested]
    block = covariance[np.ix_(tested, tested)]
    try:
        statistic = float(coefficients @ np.linalg.solve(block, coefficients))
    except np.linalg.LinAlgError:
        return _not_evaluable("singular_hc3_covariance", rows=rows)
    if not math.isfinite(statistic) or statistic < 0:
        return _not_evaluable("nonfinite_wald_statistic", rows=rows)
    p_value = float(stats.chi2.sf(statistic, len(tested)))
    return p_value, {
        "status": "ok",
        "test": "hc3_wald_chi_square",
        "statistic": statistic,
        "degrees_of_freedom": len(tested),
        "rows": rows,
        "coefficients": coefficients.tolist(),
        "standard_errors": np.sqrt(np.clip(np.diag(block), 0.0, None)).tolist(),
    }


def r_projection_test(
    outcome_residual: np.ndarray, treatment_residual: np.ndarray, z: np.ndarray
) -> tuple[float | None, dict[str, Any]]:
    """R-learner projection: (Y - m) on [(T - e), (T - e) Z]; Wald test of the Z block."""

    treatment_residual = np.asarray(treatment_residual, dtype=float)
    base = treatment_residual.reshape(-1, 1)
    full, kept = _rank_safe_columns(base, treatment_residual[:, None] * np.asarray(z, float))
    p_value, details = hc3_wald_test(outcome_residual, full, range(1, full.shape[1]))
    return p_value, {**details, "tested_column_indices": kept}


def aipw_projection_test(psi: np.ndarray, z: np.ndarray) -> tuple[float | None, dict[str, Any]]:
    """DR-learner projection: psi on [1, Z]; Wald test of the Z block."""

    psi = np.asarray(psi, dtype=float)
    full, kept = _rank_safe_columns(np.ones((len(psi), 1)), np.asarray(z, float))
    p_value, details = hc3_wald_test(psi, full, range(1, full.shape[1]))
    return p_value, {**details, "tested_column_indices": kept}


def outcome_interaction_test(
    outcome: np.ndarray,
    treatment: np.ndarray,
    adjustment_base: np.ndarray,
    z: np.ndarray,
    *,
    binary: bool,
) -> tuple[float | None, dict[str, Any]]:
    """Adjusted interaction test of T:Z with Z main effects in both models.

    ``adjustment_base`` is ``stage2_interaction_screen.adjustment_design``'s
    intercept/T/nuisance-score matrix. Fits that raise convergence or separation
    warnings give no p-value, as in the multi-model univariable family.
    """

    z = np.asarray(z, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    reduced, _ = _rank_safe_columns(adjustment_base, z)
    interactions = treatment[:, None] * z
    nested = _binary_nested_p_value if binary else _continuous_nested_p_value
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        p_value, details = nested(np.asarray(outcome, dtype=float), reduced, interactions)
    unreliable = sorted({
        str(warning.message) for warning in caught
        if any(term in str(warning.message).lower() for term in _UNRELIABLE_WARNINGS)
    })
    if p_value is not None and unreliable:
        return _not_evaluable("unreliable_fit_warning", warnings=unreliable[:3])
    return p_value, dict(details)


def testing_scores(
    *,
    treatment: Any,
    outcome: Any,
    propensity: Any,
    mu0: Any,
    mu1: Any,
    propensity_clip: float,
) -> dict[str, np.ndarray]:
    """m_hat = e mu1 + (1 - e) mu0 and the clipped AIPW pseudo-outcome psi."""

    t, y, e, m0, m1 = (
        np.asarray(value, dtype=float) for value in (treatment, outcome, propensity, mu0, mu1)
    )
    if any(value.ndim != 1 or value.shape != t.shape for value in (y, e, m0, m1)):
        raise ValueError("modifier-test labels and nuisances must be aligned vectors")
    if not all(np.isfinite(value).all() for value in (t, y, e, m0, m1)):
        raise ValueError("modifier-test labels and nuisances must be finite")
    if np.any((e < 0) | (e > 1)):
        raise ValueError("modifier-test propensities must be probabilities")
    if not 0.0 < float(propensity_clip) < 0.5:
        raise ValueError("propensity_clip must be between 0 and 0.5")
    clipped = np.clip(e, propensity_clip, 1.0 - propensity_clip)
    psi = m1 - m0 + t * (y - m1) / clipped - (1.0 - t) * (y - m0) / (1.0 - clipped)
    return {"marginal": e * m1 + (1.0 - e) * m0, "psi": psi}


def candidate_tests(
    frame: pd.DataFrame,
    definitions: Sequence[Mapping[str, Any]],
    *,
    treatment: Any,
    outcome: Any,
    propensity: Any,
    mu0: Any,
    mu1: Any,
    binary: bool,
    propensity_clip: float,
    eligible: Any = None,
    min_count: int = 5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the three component tests for every candidate on the eligible rows.

    ``frame`` rows align with the label and nuisance vectors. Returns one record
    per definition (in order) and an audit of the shared adjustment design.
    """

    t = np.asarray(treatment, dtype=float)
    if len(frame) != len(t):
        raise ValueError("modifier-test measurements must align with labels")
    scores = testing_scores(
        treatment=treatment, outcome=outcome, propensity=propensity, mu0=mu0, mu1=mu1,
        propensity_clip=propensity_clip,
    )
    keep = np.ones(len(t), dtype=bool) if eligible is None else np.asarray(eligible, dtype=bool)
    if keep.shape != t.shape:
        raise ValueError("eligibility must align with the labels")
    y, e = np.asarray(outcome, dtype=float)[keep], np.asarray(propensity, dtype=float)[keep]
    t_keep, marginal, psi = t[keep], scores["marginal"][keep], scores["psi"][keep]
    rows = frame.reset_index(drop=True).loc[keep].reset_index(drop=True)
    evaluable = len(t_keep) >= 4 and len(np.unique(t_keep)) == 2
    base, adjustment_audit = (None, None)
    if evaluable:
        base, adjustment_audit = interaction_screen.adjustment_design(
            t_keep, e, marginal, binary=binary
        )
    records = []
    for feature in definitions:
        record: dict[str, Any] = {
            "feature_id": str(feature.get("feature_id") or feature["name"]),
            "name": str(feature["name"]),
            "n_rows": len(t),
            "n_eligible": int(keep.sum()),
        }
        if not evaluable:
            reason = "insufficient_eligible_rows_or_treatment_arms"
            record.update(grouped_columns=[], dropped_sparse_columns=[], components={
                name: {"status": "not_evaluable", "reason": reason, "p_value": None}
                for name in COMPONENTS
            })
            records.append(record)
            continue
        z, names, dropped = grouped_design(rows, feature, min_count=min_count)
        results = {
            "r_projection": r_projection_test(y - marginal, t_keep - e, z),
            "aipw_projection": aipw_projection_test(psi, z),
            "outcome_interaction": outcome_interaction_test(
                y, t_keep, base, z, binary=binary
            ),
        }
        components = {}
        for name in COMPONENTS:
            p_value, details = results[name]
            if z.shape[1] == 0:
                p_value, details = _not_evaluable("no_estimable_candidate_columns")
            components[name] = {**details, "p_value": p_value}
        record.update(grouped_columns=names, dropped_sparse_columns=dropped,
                      components=components)
        records.append(record)
    return json_ready(records), json_ready({
        "rows": len(t),
        "eligible_rows": int(keep.sum()),
        "evaluable": bool(evaluable),
        "outcome_interaction_adjustment": adjustment_audit,
        "sparse_indicator_min_count": int(min_count),
        "component_descriptions": dict(COMPONENT_DESCRIPTIONS),
    })


def _passes(q_value: float | None, q: float) -> bool:
    return q_value is not None and q_value <= q


def combine_and_select(
    records: Sequence[Mapping[str, Any]],
    *,
    weights: Mapping[str, float],
    q: float,
    binary: bool,
    family_ids: Sequence[str] | None = None,
    max_modifiers: int | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Cauchy-combine, apply BH across the family, select, and label.

    Returns updated records (same order) and the selected feature IDs ordered
    by combined p-value. Records outside ``family_ids`` get combined p-values
    but no q-values and are never selected here.
    """

    records = [dict(record) for record in records]
    family = (
        {record["feature_id"] for record in records}
        if family_ids is None else {str(value) for value in family_ids}
    )
    for record in records:
        components = {name: dict(record["components"][name]) for name in COMPONENTS}
        combined = cauchy_combination(
            {name: components[name].get("p_value") for name in COMPONENTS}, weights
        )
        record.update(
            components=components,
            combined_p_value=combined["p_value"],
            combined_statistic=combined["statistic"],
            combination_weights=combined["weights"],
            combined_components=combined["components"],
            bh_family=record["feature_id"] in family,
        )
    members = [index for index, record in enumerate(records) if record["bh_family"]]
    combined_q = _benjamini_hochberg([records[i]["combined_p_value"] for i in members])
    component_q = {
        name: _benjamini_hochberg([records[i]["components"][name].get("p_value") for i in members])
        for name in COMPONENTS
    }
    for position, index in enumerate(members):
        records[index]["combined_q_value"] = combined_q[position]
        for name in COMPONENTS:
            records[index]["components"][name]["q_value"] = component_q[name][position]
    discoveries = sorted(
        (i for i in members if _passes(records[i].get("combined_q_value"), q)),
        key=lambda i: (records[i]["combined_p_value"], i),
    )
    capped = discoveries if max_modifiers is None else discoveries[: int(max_modifiers)]
    rank = {index: position for position, index in enumerate(capped, 1)}
    scale_label = "multiplicative" if binary else "additive"
    for index, record in enumerate(records):
        record.setdefault("combined_q_value", None)
        passes = {}
        for name in COMPONENTS:
            component = record["components"][name]
            component.setdefault("q_value", None)
            component["passes"] = _passes(component["q_value"], q)
            passes[name] = component["passes"]
        selected = index in rank
        labels = []
        if selected:
            if any(passes[name] for name in ABSOLUTE_COMPONENTS):
                labels.append("absolute")
            if passes["outcome_interaction"]:
                labels.append(scale_label)
            if not labels:
                labels.append("combined_only")
        record.update(
            selected=selected,
            selection_rank=rank.get(index),
            excluded_by_cap=index in discoveries and index not in rank,
            labels=labels,
            extreme_propensity_signal=bool(
                passes["aipw_projection"] and not passes["r_projection"]
            ),
            q_threshold=float(q),
        )
    return records, [records[index]["feature_id"] for index in capped]


def stouffer_combination(p_values: Sequence[float | None]) -> dict[str, Any]:
    """Unweighted Stouffer combination of one-sided z-scores z_k = Phi^-1(1 - p_k)."""

    usable = [
        min(max(float(p), P_VALUE_CLAMP), 1.0 - P_VALUE_CLAMP)
        for p in p_values if p is not None and math.isfinite(float(p))
    ]
    if not usable:
        return {"p_value": None, "z": None, "components": 0}
    z_scores = stats.norm.isf(np.asarray(usable))
    z = float(np.sum(z_scores) / math.sqrt(len(z_scores)))
    return {
        "p_value": float(stats.norm.sf(z)),
        "z": z,
        "z_scores": z_scores.tolist(),
        "components": len(usable),
    }


def compact_evidence(record: Mapping[str, Any], *, rule: str) -> dict[str, Any]:
    """The per-modifier summary carried in final_definitions.json."""

    return {
        "rule": rule,
        "combined_p_value": record.get("combined_p_value"),
        "combined_q_value": record.get("combined_q_value"),
        "component_p_values": {
            name: record["components"][name].get("p_value") for name in COMPONENTS
        },
        "component_q_values": {
            name: record["components"][name].get("q_value") for name in COMPONENTS
        },
        "labels": list(record.get("labels") or []),
        "extreme_propensity_signal": bool(record.get("extreme_propensity_signal")),
        "selection_rank": record.get("selection_rank"),
        "investigator_locked": bool(record.get("investigator_locked")),
    }


__all__ = [
    "ABSOLUTE_COMPONENTS",
    "COMPONENTS",
    "COMPONENT_DESCRIPTIONS",
    "INTERPRETATIONS",
    "SCHEMA_VERSION",
    "aipw_projection_test",
    "candidate_tests",
    "cauchy_combination",
    "combine_and_select",
    "compact_evidence",
    "default_weights",
    "grouped_design",
    "hc3_wald_test",
    "json_ready",
    "outcome_interaction_test",
    "r_projection_test",
    "stouffer_combination",
    "testing_scores",
]
