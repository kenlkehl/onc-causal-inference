"""Feature encoding and nested likelihood-ratio tests shared by Stage 2 numerical selection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .stage2_extraction_failures import failure_flags, has_failure_indicator


def _feature_strategy(feature: Mapping[str, Any]) -> str:
    plan = feature.get("harmonization_plan")
    if isinstance(plan, Mapping):
        target = str(plan.get("target_representation") or "").strip().lower()
        if target in {"continuous", "categorical"}:
            return target
    value_type = str(feature.get("value_type") or "ambiguous").strip().lower()
    if value_type == "continuous":
        return (
            "continuous_with_categorical_fallback"
            if feature.get("modeling_strategy") == "continuous_with_categorical_fallback"
            or isinstance(feature.get("harmonization_fallback"), Mapping)
            else "continuous"
        )
    return "categorical"


def _declared_categories(feature: Mapping[str, Any]) -> list[str]:
    plan = feature.get("harmonization_plan")
    if isinstance(plan, Mapping) and str(plan.get("target_representation") or "") == "categorical":
        values = plan.get("canonical_categories") or []
    else:
        values = feature.get("categories_or_unit") or []
    return list(dict.fromkeys(str(value) for value in values if str(value).strip()))


def _nonconstant(
    matrix: np.ndarray,
    names: Sequence[str],
) -> tuple[np.ndarray, list[str], list[int]]:
    if matrix.size == 0:
        return np.empty((len(matrix), 0), dtype=float), [], []
    keep = [
        index
        for index in range(matrix.shape[1])
        if np.isfinite(matrix[:, index]).all()
        and float(np.ptp(matrix[:, index])) > 1e-12
    ]
    return matrix[:, keep], [str(names[index]) for index in keep], keep


@dataclass(frozen=True)
class _FeatureDesign:
    main: np.ndarray
    main_names: tuple[str, ...]
    interaction: np.ndarray
    interaction_names: tuple[str, ...]
    categorical_levels: tuple[str, ...] = ()
    categorical_reference_level: str | None = None


def _encode_feature(
    frame: pd.DataFrame,
    feature: Mapping[str, Any],
) -> _FeatureDesign:
    """Fit a deterministic reference encoding on one inner-training partition."""

    name = str(feature["name"])
    series = (
        frame[name].reset_index(drop=True)
        if name in frame
        else pd.Series([None] * len(frame), dtype=object)
    )
    strategy = _feature_strategy(feature)
    # Failed extractions are not "not documented"; they get their own column.
    failed = failure_flags(frame, feature).reset_index(drop=True)
    main_columns: list[np.ndarray] = []
    main_names: list[str] = []
    interaction_columns: list[np.ndarray] = []
    interaction_names: list[str] = []
    categorical_levels: tuple[str, ...] = ()
    categorical_reference_level: str | None = None

    if strategy in {"continuous", "continuous_with_categorical_fallback"}:
        numeric = pd.to_numeric(series, errors="coerce")
        numeric_observed = numeric.notna()
        median = float(numeric.loc[numeric_observed].median()) if numeric_observed.any() else 0.0
        centered = numeric.fillna(median).to_numpy(dtype=float) - median
        main_columns.append(centered)
        main_names.append(f"{name}:value")
        interaction_columns.append(centered)
        interaction_names.append(f"{name}:value")

        if strategy == "continuous_with_categorical_fallback":
            fallback = series.notna() & numeric.isna()
            observed_fallback = sorted(str(value) for value in series.loc[fallback].unique())
            for category in observed_fallback:
                column = (fallback & (series.astype(str) == category)).to_numpy(dtype=float)
                main_columns.append(column)
                main_names.append(f"{name}:fallback={category}")
                interaction_columns.append(column)
                interaction_names.append(f"{name}:fallback={category}")
        missing = (series.isna() & ~failed).to_numpy(dtype=float)
        main_columns.append(missing)
        main_names.append(f"{name}:missing")
    else:
        missing = series.isna()
        observed = sorted(str(value) for value in series.loc[~missing].unique())
        declared = _declared_categories(feature)
        reference = next((value for value in declared if value in observed), None)
        if reference is None and observed:
            reference = observed[0]
        levels = list(dict.fromkeys([*declared, *observed]))
        if reference is not None:
            levels = [reference, *(value for value in levels if value != reference)]
        categorical_levels = tuple(levels)
        categorical_reference_level = reference
        # The first observed level is the fold-fitted reference. Every other
        # observed level contributes an interaction contrast, yielding the
        # full-rank K-1 omnibus test in a model that also contains treatment.
        normalized = series.astype(str)
        for category in levels[1:]:
            column = ((~missing) & (normalized == category)).to_numpy(dtype=float)
            main_columns.append(column)
            main_names.append(f"{name}:level={category}")
            interaction_columns.append(column)
            interaction_names.append(f"{name}:level={category}")
        main_columns.append((missing & ~failed).to_numpy(dtype=float))
        main_names.append(f"{name}:missing")
    if has_failure_indicator(feature):
        main_columns.append(failed.to_numpy(dtype=float))
        main_names.append(f"{name}:extraction_failed")

    raw_main = (
        np.column_stack(main_columns).astype(float, copy=False)
        if main_columns
        else np.empty((len(frame), 0), dtype=float)
    )
    raw_interaction = (
        np.column_stack(interaction_columns).astype(float, copy=False)
        if interaction_columns
        else np.empty((len(frame), 0), dtype=float)
    )
    main, kept_main_names, _ = _nonconstant(raw_main, main_names)
    interaction, kept_interaction_names, _ = _nonconstant(
        raw_interaction,
        interaction_names,
    )
    return _FeatureDesign(
        main=main,
        main_names=tuple(kept_main_names),
        interaction=interaction,
        interaction_names=tuple(kept_interaction_names),
        categorical_levels=categorical_levels,
        categorical_reference_level=categorical_reference_level,
    )


def _rank_safe_columns(
    base: np.ndarray,
    additions: np.ndarray,
) -> tuple[np.ndarray, list[int]]:
    """Append only columns that increase rank, preserving deterministic order."""

    current = np.asarray(base, dtype=float)
    rank = int(np.linalg.matrix_rank(current))
    kept: list[int] = []
    for index in range(additions.shape[1]):
        candidate = np.column_stack([current, additions[:, index]])
        next_rank = int(np.linalg.matrix_rank(candidate))
        if next_rank > rank:
            current = candidate
            rank = next_rank
            kept.append(index)
    return current, kept


def _binary_nested_p_value(
    target: np.ndarray,
    reduced: np.ndarray,
    additions: np.ndarray,
) -> tuple[float | None, dict[str, Any]]:
    target = np.asarray(target, dtype=float)
    if len(np.unique(target)) != 2:
        return None, {"status": "not_evaluable", "reason": "target_has_one_class"}
    full, kept = _rank_safe_columns(reduced, additions)
    degrees = full.shape[1] - reduced.shape[1]
    if degrees < 1:
        return None, {"status": "not_evaluable", "reason": "no_independent_candidate_columns"}
    try:
        import statsmodels.api as sm

        reduced_fit = sm.GLM(target, reduced, family=sm.families.Binomial()).fit(disp=0)
        full_fit = sm.GLM(target, full, family=sm.families.Binomial()).fit(disp=0)
        likelihood_ratio = max(0.0, 2.0 * float(full_fit.llf - reduced_fit.llf))
        p_value = float(stats.chi2.sf(likelihood_ratio, degrees))
        if not math.isfinite(p_value):
            raise ValueError("nonfinite likelihood-ratio p-value")
    except Exception as exc:
        return None, {
            "status": "not_evaluable",
            "reason": f"{type(exc).__name__}: {exc}",
        }
    return p_value, {
        "status": "ok",
        "test": "likelihood_ratio_chi_square",
        "statistic": likelihood_ratio,
        "degrees_of_freedom": int(degrees),
        "tested_column_indices": kept,
    }


def _continuous_nested_p_value(
    target: np.ndarray,
    reduced: np.ndarray,
    additions: np.ndarray,
) -> tuple[float | None, dict[str, Any]]:
    target = np.asarray(target, dtype=float)
    full, kept = _rank_safe_columns(reduced, additions)
    degrees = full.shape[1] - reduced.shape[1]
    residual_degrees = len(target) - full.shape[1]
    if degrees < 1 or residual_degrees < 1:
        return None, {
            "status": "not_evaluable",
            "reason": "insufficient_independent_columns_or_residual_degrees_of_freedom",
        }
    try:
        reduced_residual = target - reduced @ np.linalg.lstsq(reduced, target, rcond=None)[0]
        full_residual = target - full @ np.linalg.lstsq(full, target, rcond=None)[0]
        reduced_ss = float(reduced_residual @ reduced_residual)
        full_ss = float(full_residual @ full_residual)
        if full_ss <= 1e-15:
            statistic = math.inf if reduced_ss > full_ss + 1e-15 else 0.0
        else:
            numerator = max(0.0, (reduced_ss - full_ss) / degrees)
            statistic = numerator / (full_ss / residual_degrees)
        p_value = float(stats.f.sf(statistic, degrees, residual_degrees))
        if not math.isfinite(p_value):
            raise ValueError("nonfinite partial-F p-value")
    except Exception as exc:
        return None, {
            "status": "not_evaluable",
            "reason": f"{type(exc).__name__}: {exc}",
        }
    return p_value, {
        "status": "ok",
        "test": "partial_f",
        "statistic": statistic,
        "degrees_of_freedom": [int(degrees), int(residual_degrees)],
        "tested_column_indices": kept,
    }


def _feature_key(feature: Mapping[str, Any]) -> str:
    return str(feature.get("feature_id") or feature["name"])


def _annotated_interaction_test(
    test: Mapping[str, Any],
    candidate: _FeatureDesign,
) -> dict[str, Any]:
    result = dict(test)
    indices = [int(value) for value in result.get("tested_column_indices") or []]
    interaction_names = list(candidate.interaction_names)
    result.update(
        {
            "candidate_interaction_columns": interaction_names,
            "tested_interaction_columns": [
                interaction_names[index]
                for index in indices
                if 0 <= index < len(interaction_names)
            ],
            "categorical_levels": list(candidate.categorical_levels),
            "categorical_reference_level": candidate.categorical_reference_level,
            "categorical_parameterization": (
                "all_estimable_nonreference_level_interactions_in_one_omnibus_test"
                if candidate.categorical_levels
                else None
            ),
        }
    )
    return result


def _modifier_test_chunk(
    frame: pd.DataFrame,
    treatment: np.ndarray,
    outcome: np.ndarray,
    reduced_with_treatment: np.ndarray,
    features: Sequence[Mapping[str, Any]],
    *,
    binary_outcome: bool,
    p_value_threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature in features:
        candidate = _encode_feature(frame, feature)
        reduced, _ = _rank_safe_columns(reduced_with_treatment, candidate.main)
        interactions = treatment.reshape(-1, 1) * candidate.interaction
        if binary_outcome:
            p_value, test = _binary_nested_p_value(outcome, reduced, interactions)
        else:
            p_value, test = _continuous_nested_p_value(outcome, reduced, interactions)
        selected = bool(
            p_value is not None and p_value < float(p_value_threshold)
        )
        rows.append(
            {
                "feature_id": _feature_key(feature),
                "name": str(feature["name"]),
                "interaction_p_value": p_value,
                "interaction_test": _annotated_interaction_test(test, candidate),
                "vote": selected,
            }
        )
    return rows
