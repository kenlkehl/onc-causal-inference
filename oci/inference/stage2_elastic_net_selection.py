"""Fold-honest statistical evidence for Stage 2 causal-role adjudication.

For confounders, this module records both multivariable grouped elastic-net
support in treatment/outcome nuisance models and candidate-wise treatment and
outcome association tests.  For effect modifiers, it records both
candidate-augmented held-out R-loss comparisons and a joint multivariable
grouped elastic-net interaction model.  All four views use only the current
outer-training partition and inner-fold honest evaluation.

The numerical unions remain provisional for compatibility and model setup.
They are packaged as fallible evidence for the downstream LLM role adjudicator,
which makes the final confounder/effect-modifier assignments.  No pairwise
clustering or latent construction occurs here.
"""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

from .nuisance_diagnostics import (
    calibration_diagnostics,
    overlap_diagnostics,
    propensity_eligibility,
    validate_propensity_bounds,
)

from .stage2_statistical_selection import (
    _binary_nested_p_value,
    _continuous_nested_p_value,
    _encode_feature as _encode_univariable_feature,
    _rank_safe_columns as _univariable_rank_safe_columns,
)
from .stage2_multi_model_config import Stage2MultiModelConfig, multi_model_config_from_mapping

LOGGER = logging.getLogger(__name__)

SCHEMA_VERSION = "stage2_all_evidence_statistical_components_v2_calibration_overlap"
TEMPORAL_SCOPE = "pre_index_treatment"


@dataclass(frozen=True)
class Stage2ElasticNetSelectionConfig:
    """Scientific and numerical policy for deterministic grouped selection."""

    min_propensity: float | None = None
    max_propensity: float | None = None
    l1_ratio: float = 0.8
    nuisance_selection_rule: str = "any_inner_fold_union"
    modifier_selection_rule: str = "any_inner_fold_union"
    # Retained only so older configuration files remain loadable. Selection no
    # longer uses frequency thresholds; public_dict deliberately omits them.
    nuisance_selection_frequency: float = 0.6
    modifier_selection_frequency: float = 0.6
    internal_cv_folds: int = 3
    regularization_grid_size: int = 16
    minimum_log10_alpha: float = -5.0
    maximum_log10_alpha: float = -1.0
    coefficient_tolerance: float = 1e-7
    optimization_tolerance: float = 1e-6
    categorical_min_count: int = 5
    max_iter: int = 5_000
    one_standard_error_rule: bool = True
    nuisance_prediction_one_standard_error_rule: bool = False
    modifier_one_standard_error_rule: bool = False
    # Univariable tests are descriptive inputs to the final role adjudicator.
    # These thresholds create fold-level evidence flags; neither is a hard
    # selection gate.
    univariable_confounder_p_value_threshold: float = 0.05
    univariable_confounder_q_value_threshold: float = 0.10
    # Retained only so older configuration files remain loadable. Nuisance
    # prediction is now performed by the same grouped elastic-net family used
    # for screening; these forest settings no longer affect computation.
    nuisance_forest_trees: int = 200
    nuisance_forest_min_samples_leaf: int = 10
    modifier_top_n_per_inner_fold: int = 10
    modifier_ridge_alpha: float = 10.0
    modifier_continuous_winsor_quantile: float = 0.005
    # Retained only so configurations written for earlier R-learner selectors
    # remain loadable. They do not gate the per-fold top-N union.
    modifier_min_fold_r_loss_improvement: float = 0.0
    modifier_min_mean_r_loss_improvement: float = 0.0
    modifier_min_positive_fold_fraction: float = 0.4
    # Opt-in task selection; omit the legacy value from checkpoint policy JSON.
    selection_mode: str = "llm_roles"
    multi_model: Stage2MultiModelConfig = field(default_factory=Stage2MultiModelConfig)

    def validate(self) -> None:
        validate_propensity_bounds(self.min_propensity, self.max_propensity)
        if not isinstance(self.selection_mode, str) or self.selection_mode not in {
            "llm_roles", "independent_tasks", "multi_model"
        }:
            raise ValueError(
                "stage2.statistical_selection.selection_mode must be "
                "llm_roles, independent_tasks, or multi_model"
            )
        if not isinstance(self.multi_model, Stage2MultiModelConfig):
            raise ValueError("multi_model must be a Stage2MultiModelConfig object")
        self.multi_model.validate()
        for name in ("nuisance_selection_rule", "modifier_selection_rule"):
            if getattr(self, name) != "any_inner_fold_union":
                raise ValueError(
                    f"stage2.statistical_selection.{name} must be "
                    "'any_inner_fold_union'"
                )
        for name in (
            "l1_ratio",
            "nuisance_selection_frequency",
            "modifier_selection_frequency",
            "modifier_min_positive_fold_fraction",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 < float(value) <= 1.0
            ):
                raise ValueError(f"stage2.statistical_selection.{name} must be in (0, 1]")
        for name in (
            "internal_cv_folds",
            "regularization_grid_size",
            "categorical_min_count",
            "max_iter",
            "modifier_top_n_per_inner_fold",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(
                    f"stage2.statistical_selection.{name} must be a positive integer"
                )
        if self.internal_cv_folds < 2:
            raise ValueError(
                "stage2.statistical_selection.internal_cv_folds must be at least 2"
            )
        if self.regularization_grid_size < 3:
            raise ValueError(
                "stage2.statistical_selection.regularization_grid_size must be at least 3"
            )
        for name in (
            "minimum_log10_alpha",
            "maximum_log10_alpha",
            "coefficient_tolerance",
            "optimization_tolerance",
            "modifier_ridge_alpha",
            "modifier_continuous_winsor_quantile",
            "modifier_min_fold_r_loss_improvement",
            "modifier_min_mean_r_loss_improvement",
            "univariable_confounder_p_value_threshold",
            "univariable_confounder_q_value_threshold",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(
                    f"stage2.statistical_selection.{name} must be a finite number"
                )
        if self.minimum_log10_alpha >= self.maximum_log10_alpha:
            raise ValueError(
                "stage2.statistical_selection.minimum_log10_alpha must be smaller than "
                "maximum_log10_alpha"
            )
        if self.coefficient_tolerance < 0.0:
            raise ValueError(
                "stage2.statistical_selection.coefficient_tolerance must be nonnegative"
            )
        if self.optimization_tolerance <= 0.0:
            raise ValueError(
                "stage2.statistical_selection.optimization_tolerance must be positive"
            )
        if self.modifier_min_mean_r_loss_improvement < 0.0:
            raise ValueError(
                "stage2.statistical_selection.modifier_min_mean_r_loss_improvement "
                "must be nonnegative"
            )
        if self.modifier_min_fold_r_loss_improvement < 0.0:
            raise ValueError(
                "stage2.statistical_selection.modifier_min_fold_r_loss_improvement "
                "must be nonnegative"
            )
        if self.modifier_ridge_alpha <= 0.0:
            raise ValueError(
                "stage2.statistical_selection.modifier_ridge_alpha must be positive"
            )
        if not 0.0 <= self.modifier_continuous_winsor_quantile < 0.25:
            raise ValueError(
                "stage2.statistical_selection.modifier_continuous_winsor_quantile "
                "must be in [0, 0.25)"
            )
        for name in (
            "univariable_confounder_p_value_threshold",
            "univariable_confounder_q_value_threshold",
        ):
            if not 0.0 < float(getattr(self, name)) <= 1.0:
                raise ValueError(
                    f"stage2.statistical_selection.{name} must be in (0, 1]"
                )
        for name in (
            "one_standard_error_rule",
            "nuisance_prediction_one_standard_error_rule",
            "modifier_one_standard_error_rule",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(
                    f"stage2.statistical_selection.{name} must be boolean"
                )

    def public_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for name in ("min_propensity", "max_propensity"):
            if result[name] is None:
                result.pop(name)
        if self.selection_mode == "llm_roles":
            result.pop("selection_mode")
        if self.selection_mode != "multi_model":
            result.pop("multi_model")
        else:
            result["multi_model"] = self.multi_model.public_dict()
        for retired in (
            "nuisance_selection_frequency",
            "modifier_selection_frequency",
            "nuisance_forest_trees",
            "nuisance_forest_min_samples_leaf",
            "modifier_min_fold_r_loss_improvement",
            "modifier_min_mean_r_loss_improvement",
            "modifier_min_positive_fold_fraction",
        ):
            result.pop(retired, None)
        return result


def statistical_selection_config_from_mapping(
    value: Mapping[str, Any] | None,
) -> Stage2ElasticNetSelectionConfig:
    if value is not None and not isinstance(value, Mapping):
        raise ValueError("stage2.statistical_selection must be an object")
    raw = dict(value or {})
    known = set(Stage2ElasticNetSelectionConfig.__dataclass_fields__)
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(
            "stage2.statistical_selection contains unsupported fields: " f"{unknown}"
        )
    if "multi_model" in raw:
        raw["multi_model"] = multi_model_config_from_mapping(raw["multi_model"])
    config = Stage2ElasticNetSelectionConfig(**raw)
    config.validate()
    return config


def _feature_key(feature: Mapping[str, Any]) -> str:
    return str(feature.get("feature_id") or feature["name"])


def _feature_strategy(feature: Mapping[str, Any]) -> str:
    value_type = str(feature.get("value_type") or "ambiguous").strip().lower()
    if value_type == "ordinal":
        return "ordinal"
    plan = feature.get("harmonization_plan")
    if isinstance(plan, Mapping):
        target = str(plan.get("target_representation") or "").strip().lower()
        if target in {"continuous", "categorical"}:
            return target
    if value_type == "continuous":
        if (
            feature.get("modeling_strategy") == "continuous_with_categorical_fallback"
            or isinstance(feature.get("harmonization_fallback"), Mapping)
        ):
            return "continuous_with_categorical_fallback"
        return "continuous"
    return "categorical"


@dataclass(frozen=True)
class _EncodedDesign:
    train: np.ndarray
    valid: np.ndarray
    column_feature_ids: tuple[str, ...]
    column_names: tuple[str, ...]


def _append_variable_column(
    train_columns: list[np.ndarray],
    valid_columns: list[np.ndarray],
    column_feature_ids: list[str],
    column_names: list[str],
    *,
    feature_id: str,
    name: str,
    train_values: np.ndarray,
    valid_values: np.ndarray,
) -> None:
    train_values = np.asarray(train_values, dtype=float)
    valid_values = np.asarray(valid_values, dtype=float)
    if not np.isfinite(train_values).all() or not np.isfinite(valid_values).all():
        raise ValueError(f"nonfinite encoded values for Stage 2 feature {feature_id!r}")
    if len(train_values) == 0:
        return
    mean = float(np.mean(train_values))
    scale = float(np.std(train_values, ddof=0))
    if not math.isfinite(scale) or scale <= 1e-12:
        return
    # Every penalized column has unit training-fold variance.  Without this,
    # an indicator for a rare category is charged much more heavily than a
    # standardized continuous measurement.
    train_columns.append((train_values - mean) / scale)
    valid_columns.append((valid_values - mean) / scale)
    column_feature_ids.append(feature_id)
    column_names.append(name)


def _ordered_values(
    train_series: pd.Series,
    valid_series: pd.Series,
    definition: Mapping[str, Any],
) -> tuple[pd.Series, pd.Series]:
    """Map an ordinal measurement to one ordered numerical score."""

    train_numeric = pd.to_numeric(train_series, errors="coerce")
    valid_numeric = pd.to_numeric(valid_series, errors="coerce")
    categories = definition.get("categories_or_unit")
    if isinstance(categories, Sequence) and not isinstance(categories, (str, bytes)):
        category_order = {
            str(category).strip().casefold(): float(index)
            for index, category in enumerate(categories)
        }
        train_text = train_series.astype(str).str.strip().str.casefold()
        valid_text = valid_series.astype(str).str.strip().str.casefold()
        train_numeric = train_numeric.where(
            train_numeric.notna(), train_text.map(category_order)
        )
        valid_numeric = valid_numeric.where(
            valid_numeric.notna(), valid_text.map(category_order)
        )
    return train_numeric, valid_numeric


def _encode_design(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    definitions: Sequence[Mapping[str, Any]],
    *,
    categorical_min_count: int,
) -> _EncodedDesign:
    """Fit a feature-group-preserving design on one training partition."""

    train_columns: list[np.ndarray] = []
    valid_columns: list[np.ndarray] = []
    column_feature_ids: list[str] = []
    column_names: list[str] = []
    for definition in definitions:
        feature_id = _feature_key(definition)
        name = str(definition["name"])
        train_series = (
            train[name].reset_index(drop=True)
            if name in train
            else pd.Series([None] * len(train), dtype=object)
        )
        valid_series = (
            valid[name].reset_index(drop=True)
            if name in valid
            else pd.Series([None] * len(valid), dtype=object)
        )
        strategy = _feature_strategy(definition)
        if strategy in {
            "continuous",
            "continuous_with_categorical_fallback",
            "ordinal",
        }:
            if strategy == "ordinal":
                train_numeric, valid_numeric = _ordered_values(
                    train_series,
                    valid_series,
                    definition,
                )
            else:
                train_numeric = pd.to_numeric(train_series, errors="coerce")
                valid_numeric = pd.to_numeric(valid_series, errors="coerce")
            observed = train_numeric.dropna()
            median = float(observed.median()) if len(observed) else 0.0
            _append_variable_column(
                train_columns,
                valid_columns,
                column_feature_ids,
                column_names,
                feature_id=feature_id,
                name=(
                    f"{name}:ordered_score"
                    if strategy == "ordinal"
                    else f"{name}:value"
                ),
                train_values=train_numeric.fillna(median).to_numpy(),
                valid_values=valid_numeric.fillna(median).to_numpy(),
            )
            if strategy == "continuous_with_categorical_fallback":
                train_fallback = train_series.notna() & train_numeric.isna()
                valid_fallback = valid_series.notna() & valid_numeric.isna()
                counts = train_series.loc[train_fallback].astype(str).value_counts()
                frequent = sorted(
                    str(level)
                    for level, count in counts.items()
                    if int(count) >= int(categorical_min_count)
                )
                train_text = train_series.astype(str)
                valid_text = valid_series.astype(str)
                for level in frequent:
                    _append_variable_column(
                        train_columns,
                        valid_columns,
                        column_feature_ids,
                        column_names,
                        feature_id=feature_id,
                        name=f"{name}:fallback={level}",
                        train_values=(train_fallback & (train_text == level)).to_numpy(),
                        valid_values=(valid_fallback & (valid_text == level)).to_numpy(),
                    )
            _append_variable_column(
                train_columns,
                valid_columns,
                column_feature_ids,
                column_names,
                feature_id=feature_id,
                name=f"{name}:missing",
                train_values=(
                    train_numeric.isna().to_numpy()
                    if strategy == "ordinal"
                    else train_series.isna().to_numpy()
                ),
                valid_values=(
                    valid_numeric.isna().to_numpy()
                    if strategy == "ordinal"
                    else valid_series.isna().to_numpy()
                ),
            )
            continue

        train_missing = train_series.isna()
        valid_missing = valid_series.isna()
        train_text = train_series.astype(str)
        valid_text = valid_series.astype(str)
        counts = train_text.loc[~train_missing].value_counts()
        frequent = sorted(
            str(level)
            for level, count in counts.items()
            if int(count) >= int(categorical_min_count)
        )
        rare_train = (~train_missing) & (~train_text.isin(frequent))
        rare_valid = (~valid_missing) & (~valid_text.isin(frequent))
        levels = list(frequent)
        if bool(rare_train.any()):
            levels.append("__OTHER__")
        # Reference contrasts are standardized individually and penalized as
        # one feature group.  The group penalty, rather than an arbitrary
        # individual dummy coefficient, determines whether the factor survives.
        for level in levels[1:]:
            if level == "__OTHER__":
                train_values, valid_values = rare_train.to_numpy(), rare_valid.to_numpy()
            else:
                train_values = ((~train_missing) & (train_text == level)).to_numpy()
                valid_values = ((~valid_missing) & (valid_text == level)).to_numpy()
            _append_variable_column(
                train_columns,
                valid_columns,
                column_feature_ids,
                column_names,
                feature_id=feature_id,
                name=f"{name}:level={level}",
                train_values=train_values,
                valid_values=valid_values,
            )
        _append_variable_column(
            train_columns,
            valid_columns,
            column_feature_ids,
            column_names,
            feature_id=feature_id,
            name=f"{name}:missing",
            train_values=train_missing.to_numpy(),
            valid_values=valid_missing.to_numpy(),
        )
    return _EncodedDesign(
        train=(
            np.column_stack(train_columns).astype(float, copy=False)
            if train_columns
            else np.empty((len(train), 0), dtype=float)
        ),
        valid=(
            np.column_stack(valid_columns).astype(float, copy=False)
            if valid_columns
            else np.empty((len(valid), 0), dtype=float)
        ),
        column_feature_ids=tuple(column_feature_ids),
        column_names=tuple(column_names),
    )


@dataclass(frozen=True)
class _PenalizedFit:
    coefficients: np.ndarray
    train_prediction: np.ndarray
    valid_prediction: np.ndarray
    regularization: float | None
    cv_folds: int
    status: str
    iterations: int
    converged: bool


@dataclass(frozen=True)
class _GroupStructure:
    starts: np.ndarray
    sizes: np.ndarray
    weights: np.ndarray
    feature_ids: tuple[str, ...]


@dataclass(frozen=True)
class _SolverState:
    coefficients: np.ndarray
    intercept: float
    iterations: int
    converged: bool


def _group_structure(column_feature_ids: Sequence[str]) -> _GroupStructure:
    if not column_feature_ids:
        return _GroupStructure(
            starts=np.asarray([], dtype=int),
            sizes=np.asarray([], dtype=int),
            weights=np.asarray([], dtype=float),
            feature_ids=(),
        )
    starts = [0]
    feature_ids = [str(column_feature_ids[0])]
    seen = {feature_ids[0]}
    for position, raw_feature_id in enumerate(column_feature_ids[1:], start=1):
        feature_id = str(raw_feature_id)
        if feature_id == feature_ids[-1]:
            continue
        if feature_id in seen:
            raise ValueError("group elastic-net columns must be contiguous by feature")
        starts.append(position)
        feature_ids.append(feature_id)
        seen.add(feature_id)
    start_values = np.asarray(starts, dtype=int)
    sizes = np.diff(np.append(start_values, len(column_feature_ids))).astype(int)
    return _GroupStructure(
        starts=start_values,
        sizes=sizes,
        weights=np.sqrt(sizes.astype(float)),
        feature_ids=tuple(feature_ids),
    )


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _spectral_norm_squared(
    design: np.ndarray,
    *,
    fit_intercept: bool,
    iterations: int = 30,
) -> float:
    """Estimate ||[1, X]||_2^2 deterministically by power iteration."""

    columns = int(design.shape[1]) + int(fit_intercept)
    if columns == 0 or len(design) == 0:
        return 0.0
    vector = np.random.default_rng(0).normal(size=columns)
    vector /= float(np.linalg.norm(vector))
    for _ in range(iterations):
        if fit_intercept:
            projected = vector[0] + design @ vector[1:]
            back = np.concatenate(
                ([float(np.sum(projected))], design.T @ projected)
            )
        else:
            projected = design @ vector
            back = design.T @ projected
        norm = float(np.linalg.norm(back))
        if not math.isfinite(norm) or norm <= 1e-15:
            return 0.0
        vector = back / norm
    if fit_intercept:
        projected = vector[0] + design @ vector[1:]
    else:
        projected = design @ vector
    return float(np.dot(projected, projected))


def _group_ridge_prox(
    values: np.ndarray,
    *,
    step: float,
    regularization: float,
    group_ratio: float,
    groups: _GroupStructure,
) -> np.ndarray:
    if len(values) == 0:
        return values.copy()
    squared = np.square(values)
    norms = np.sqrt(np.add.reduceat(squared, groups.starts))
    thresholds = (
        float(step)
        * float(regularization)
        * float(group_ratio)
        * groups.weights
    )
    scales = np.maximum(0.0, 1.0 - thresholds / np.maximum(norms, 1e-30))
    scales /= 1.0 + (
        float(step) * float(regularization) * (1.0 - float(group_ratio))
    )
    return values * np.repeat(scales, groups.sizes)


def _fit_group_elastic_net(
    train_x: np.ndarray,
    train_y: np.ndarray,
    *,
    groups: _GroupStructure,
    regularization: float,
    group_ratio: float,
    binary: bool,
    fit_intercept: bool,
    max_iter: int,
    tolerance: float,
    initial: _SolverState | None = None,
) -> _SolverState:
    """Fit a convex group-lasso-plus-ridge model with accelerated proximal steps."""

    train_x = np.asarray(train_x, dtype=float)
    train_y = np.asarray(train_y, dtype=float)
    rows, columns = train_x.shape
    if initial is not None and len(initial.coefficients) == columns:
        coefficients = initial.coefficients.copy()
        intercept = float(initial.intercept) if fit_intercept else 0.0
    else:
        coefficients = np.zeros(columns, dtype=float)
        if fit_intercept and binary:
            mean = float(np.clip(np.mean(train_y), 1e-6, 1.0 - 1e-6))
            intercept = float(math.log(mean / (1.0 - mean)))
        elif fit_intercept:
            intercept = float(np.mean(train_y))
        else:
            intercept = 0.0
    spectral_squared = _spectral_norm_squared(
        train_x,
        fit_intercept=fit_intercept,
    )
    loss_curvature = 0.25 if binary else 1.0
    lipschitz = max(
        1.01 * loss_curvature * spectral_squared / max(rows, 1),
        1e-12,
    )
    step = 1.0 / lipschitz
    extrapolated = coefficients.copy()
    extrapolated_intercept = intercept
    momentum = 1.0
    converged = False
    iterations_used = 0
    for iteration in range(1, int(max_iter) + 1):
        linear = train_x @ extrapolated + extrapolated_intercept
        residual = _sigmoid(linear) - train_y if binary else linear - train_y
        gradient = train_x.T @ residual / rows
        intercept_gradient = float(np.mean(residual)) if fit_intercept else 0.0
        updated = _group_ridge_prox(
            extrapolated - step * gradient,
            step=step,
            regularization=regularization,
            group_ratio=group_ratio,
            groups=groups,
        )
        updated_intercept = (
            extrapolated_intercept - step * intercept_gradient
            if fit_intercept
            else 0.0
        )
        delta = math.sqrt(
            float(np.dot(updated - coefficients, updated - coefficients))
            + (updated_intercept - intercept) ** 2
        )
        scale = 1.0 + math.sqrt(
            float(np.dot(coefficients, coefficients)) + intercept**2
        )
        iterations_used = iteration
        if delta <= float(tolerance) * scale:
            coefficients = updated
            intercept = updated_intercept
            converged = True
            break
        next_momentum = (1.0 + math.sqrt(1.0 + 4.0 * momentum**2)) / 2.0
        acceleration = (momentum - 1.0) / next_momentum
        next_extrapolated = updated + acceleration * (updated - coefficients)
        next_extrapolated_intercept = updated_intercept + acceleration * (
            updated_intercept - intercept
        )
        # Adaptive restart avoids oscillation in highly correlated clinical groups.
        restart_direction = float(
            np.dot(extrapolated - updated, updated - coefficients)
        ) + (extrapolated_intercept - updated_intercept) * (
            updated_intercept - intercept
        )
        coefficients = updated
        intercept = updated_intercept
        if restart_direction > 0.0:
            extrapolated = updated.copy()
            extrapolated_intercept = updated_intercept
            momentum = 1.0
        else:
            extrapolated = next_extrapolated
            extrapolated_intercept = next_extrapolated_intercept
            momentum = next_momentum
    return _SolverState(
        coefficients=np.asarray(coefficients, dtype=float),
        intercept=float(intercept),
        iterations=iterations_used,
        converged=converged,
    )


def _state_prediction(
    state: _SolverState,
    design: np.ndarray,
    *,
    binary: bool,
) -> np.ndarray:
    linear = np.asarray(design, dtype=float) @ state.coefficients + state.intercept
    if binary:
        return np.clip(_sigmoid(linear), 1e-6, 1.0 - 1e-6)
    return np.asarray(linear, dtype=float)


def _constant_fit(
    train_y: np.ndarray,
    valid_rows: int,
    columns: int,
    *,
    binary: bool,
    status: str,
) -> _PenalizedFit:
    mean = float(np.mean(train_y)) if len(train_y) else 0.0
    if binary:
        mean = float(np.clip(mean, 1e-6, 1.0 - 1e-6))
    return _PenalizedFit(
        coefficients=np.zeros(columns, dtype=float),
        train_prediction=np.full(len(train_y), mean, dtype=float),
        valid_prediction=np.full(valid_rows, mean, dtype=float),
        regularization=None,
        cv_folds=0,
        status=status,
        iterations=0,
        converged=True,
    )


def _group_elastic_net_cv(
    train_x: np.ndarray,
    train_y: np.ndarray,
    valid_x: np.ndarray,
    column_feature_ids: Sequence[str],
    *,
    config: Stage2ElasticNetSelectionConfig,
    seed: int,
    binary: bool,
    fit_intercept: bool,
    one_standard_error_rule: bool,
) -> _PenalizedFit:
    train_y = np.asarray(train_y, dtype=float)
    if binary:
        class_counts = np.bincount(train_y.astype(int), minlength=2)
        cv_folds = min(int(config.internal_cv_folds), int(class_counts.min()))
        splitter: Any = StratifiedKFold(
            n_splits=max(cv_folds, 2),
            shuffle=True,
            random_state=seed,
        )
    else:
        cv_folds = min(int(config.internal_cv_folds), len(train_y))
        splitter = KFold(
            n_splits=max(cv_folds, 2),
            shuffle=True,
            random_state=seed,
        )
    if train_x.shape[1] == 0 or cv_folds < 2:
        if not fit_intercept:
            return _PenalizedFit(
                coefficients=np.zeros(train_x.shape[1], dtype=float),
                train_prediction=np.zeros(len(train_y), dtype=float),
                valid_prediction=np.zeros(len(valid_x), dtype=float),
                regularization=None,
                cv_folds=0,
                status="empty_design",
                iterations=0,
                converged=True,
            )
        return _constant_fit(
            train_y,
            len(valid_x),
            train_x.shape[1],
            binary=binary,
            status="insufficient_rows_or_empty_design",
        )
    groups = _group_structure(column_feature_ids)
    alphas = np.logspace(
        float(config.minimum_log10_alpha),
        float(config.maximum_log10_alpha),
        int(config.regularization_grid_size),
    )
    losses = np.empty((cv_folds, len(alphas)), dtype=float)
    split_iterator = (
        splitter.split(train_x, train_y.astype(int))
        if binary
        else splitter.split(train_x)
    )
    for fold_index, (fit, heldout) in enumerate(split_iterator):
        initial: _SolverState | None = None
        for alpha_index in range(len(alphas) - 1, -1, -1):
            state = _fit_group_elastic_net(
                train_x[fit],
                train_y[fit],
                groups=groups,
                regularization=float(alphas[alpha_index]),
                group_ratio=float(config.l1_ratio),
                binary=binary,
                fit_intercept=fit_intercept,
                max_iter=int(config.max_iter),
                tolerance=float(config.optimization_tolerance),
                initial=initial,
            )
            initial = state
            prediction = _state_prediction(
                state,
                train_x[heldout],
                binary=binary,
            )
            losses[fold_index, alpha_index] = (
                log_loss(train_y[heldout], prediction, labels=[0, 1])
                if binary
                else mean_squared_error(train_y[heldout], prediction)
            )
    means = np.mean(losses, axis=0)
    errors = np.std(losses, axis=0, ddof=1) / math.sqrt(losses.shape[0])
    best = int(np.argmin(means))
    chosen_index = best
    if one_standard_error_rule:
        eligible = np.flatnonzero(means <= means[best] + errors[best])
        if len(eligible):
            chosen_index = int(eligible[-1])
    chosen_alpha = float(alphas[chosen_index])
    state = _fit_group_elastic_net(
        train_x,
        train_y,
        groups=groups,
        regularization=chosen_alpha,
        group_ratio=float(config.l1_ratio),
        binary=binary,
        fit_intercept=fit_intercept,
        max_iter=int(config.max_iter),
        tolerance=float(config.optimization_tolerance),
    )
    return _PenalizedFit(
        coefficients=state.coefficients,
        train_prediction=_state_prediction(state, train_x, binary=binary),
        valid_prediction=_state_prediction(state, valid_x, binary=binary),
        regularization=chosen_alpha,
        cv_folds=cv_folds,
        status="ok" if state.converged else "maximum_iterations_reached",
        iterations=state.iterations,
        converged=state.converged,
    )


def _logistic_elastic_net(
    train_x: np.ndarray,
    train_y: np.ndarray,
    valid_x: np.ndarray,
    column_feature_ids: Sequence[str],
    *,
    config: Stage2ElasticNetSelectionConfig,
    seed: int,
    one_standard_error_rule: bool | None = None,
) -> _PenalizedFit:
    train_y = np.asarray(train_y, dtype=int)
    if train_x.shape[1] == 0 or len(np.unique(train_y)) < 2:
        return _constant_fit(
            train_y,
            len(valid_x),
            train_x.shape[1],
            binary=True,
            status="constant_or_empty_design",
        )
    use_one_standard_error_rule = (
        config.one_standard_error_rule
        if one_standard_error_rule is None
        else bool(one_standard_error_rule)
    )
    return _group_elastic_net_cv(
        train_x,
        train_y,
        valid_x,
        column_feature_ids,
        config=config,
        seed=seed,
        binary=True,
        fit_intercept=True,
        one_standard_error_rule=use_one_standard_error_rule,
    )


def _squared_error_elastic_net(
    train_x: np.ndarray,
    train_y: np.ndarray,
    valid_x: np.ndarray,
    column_feature_ids: Sequence[str],
    *,
    config: Stage2ElasticNetSelectionConfig,
    seed: int,
    fit_intercept: bool = True,
    one_standard_error_rule: bool | None = None,
) -> _PenalizedFit:
    use_one_standard_error_rule = (
        config.one_standard_error_rule
        if one_standard_error_rule is None
        else bool(one_standard_error_rule)
    )
    return _group_elastic_net_cv(
        train_x,
        train_y,
        valid_x,
        column_feature_ids,
        config=config,
        seed=seed,
        binary=False,
        fit_intercept=fit_intercept,
        one_standard_error_rule=use_one_standard_error_rule,
    )


def _selected_feature_ids(
    coefficients: np.ndarray,
    column_feature_ids: Sequence[str],
    *,
    tolerance: float,
) -> tuple[list[str], dict[str, float]]:
    squared_magnitudes: dict[str, float] = {}
    for coefficient, feature_id in zip(coefficients, column_feature_ids):
        key = str(feature_id)
        squared_magnitudes[key] = squared_magnitudes.get(key, 0.0) + float(
            coefficient
        ) ** 2
    magnitudes = {
        feature_id: math.sqrt(value)
        for feature_id, value in squared_magnitudes.items()
    }
    selected = sorted(
        feature_id
        for feature_id, magnitude in magnitudes.items()
        if magnitude > float(tolerance)
    )
    return selected, dict(sorted(magnitudes.items()))


def _selected_in_any_inner_fold(votes: Mapping[str, int]) -> set[str]:
    return {key for key, value in votes.items() if int(value) >= 1}


def _loss(observed: np.ndarray, predicted: np.ndarray, *, binary: bool) -> float:
    if binary:
        return float(
            log_loss(observed, np.clip(predicted, 1e-6, 1 - 1e-6), labels=[0, 1])
        )
    return float(mean_squared_error(observed, predicted))


def _safe_auroc(observed: np.ndarray, predicted: np.ndarray) -> float | None:
    observed = np.asarray(observed)
    predicted = np.asarray(predicted, dtype=float)
    mask = np.isfinite(observed) & np.isfinite(predicted)
    if int(np.sum(mask)) < 2 or len(np.unique(observed[mask])) < 2:
        return None
    return float(roc_auc_score(observed[mask].astype(int), predicted[mask]))


def _rank_safe_columns(
    base: np.ndarray,
    additions: np.ndarray,
) -> tuple[np.ndarray, list[int]]:
    """Append columns only when they add rank, preserving source order."""

    current = np.asarray(base, dtype=float)
    additions = np.asarray(additions, dtype=float)
    if additions.ndim != 2:
        raise ValueError("Stage 2 modifier additions must be a two-dimensional matrix")
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


def _rank_safe_column_pairs(
    train_base: np.ndarray,
    valid_base: np.ndarray,
    train_additions: np.ndarray,
    valid_additions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Append train-rank-safe columns and mirror them in validation data."""

    train_result, kept = _rank_safe_columns(train_base, train_additions)
    valid_result = np.column_stack(
        [
            np.asarray(valid_base, dtype=float),
            np.asarray(valid_additions, dtype=float)[:, kept],
        ]
    )
    return train_result, valid_result, kept


def _winsorize_modifier_design(
    design: _EncodedDesign,
    *,
    quantile: float,
) -> _EncodedDesign:
    """Fold-locally bound continuous candidate values against outliers."""

    if float(quantile) <= 0.0 or design.train.shape[1] == 0:
        return design
    train = np.asarray(design.train, dtype=float).copy()
    valid = np.asarray(design.valid, dtype=float).copy()
    for index, name in enumerate(design.column_names):
        if not str(name).endswith(":value"):
            continue
        lower, upper = np.quantile(
            train[:, index],
            [float(quantile), 1.0 - float(quantile)],
        )
        if not math.isfinite(float(lower)) or not math.isfinite(float(upper)):
            continue
        train[:, index] = np.clip(train[:, index], lower, upper)
        valid[:, index] = np.clip(valid[:, index], lower, upper)
        mean = float(np.mean(train[:, index]))
        scale = float(np.std(train[:, index], ddof=0))
        if scale > 1e-12 and math.isfinite(scale):
            train[:, index] = (train[:, index] - mean) / scale
            valid[:, index] = (valid[:, index] - mean) / scale
    return _EncodedDesign(
        train=train,
        valid=valid,
        column_feature_ids=design.column_feature_ids,
        column_names=design.column_names,
    )


def _crossfit_indices(
    treatment: np.ndarray,
    *,
    requested_folds: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create deterministic subfolds contained in one modifier-fit set."""

    treatment = np.asarray(treatment, dtype=int)
    if len(treatment) < 2:
        return []
    class_counts = np.bincount(treatment, minlength=2)
    if len(np.unique(treatment)) == 2 and int(class_counts.min()) >= 2:
        fold_count = min(int(requested_folds), int(class_counts.min()))
        splitter: Any = StratifiedKFold(
            n_splits=fold_count,
            shuffle=True,
            random_state=seed,
        )
        return [
            (np.asarray(fit, dtype=int), np.asarray(heldout, dtype=int))
            for fit, heldout in splitter.split(np.zeros(len(treatment)), treatment)
        ]
    fold_count = min(int(requested_folds), len(treatment))
    if fold_count < 2:
        return []
    splitter = KFold(n_splits=fold_count, shuffle=True, random_state=seed)
    return [
        (np.asarray(fit, dtype=int), np.asarray(heldout, dtype=int))
        for fit, heldout in splitter.split(np.zeros(len(treatment)))
    ]


def _cross_fitted_base_nuisances(
    *,
    treatment_design: np.ndarray,
    treatment_column_feature_ids: Sequence[str],
    outcome_design: np.ndarray,
    outcome_column_feature_ids: Sequence[str],
    treatment: np.ndarray,
    outcome: np.ndarray,
    binary_outcome: bool,
    config: Stage2ElasticNetSelectionConfig,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Cross-fit grouped elastic-net base nuisances inside a modifier fold."""

    splits = _crossfit_indices(
        treatment,
        requested_folds=int(config.internal_cv_folds),
        seed=seed,
    )
    base_e = np.full(len(treatment), np.nan, dtype=float)
    base_m = np.full(len(outcome), np.nan, dtype=float)
    for subfold, (fit, heldout) in enumerate(splits, start=1):
        treatment_fit = _logistic_elastic_net(
            treatment_design[fit],
            treatment[fit],
            treatment_design[heldout],
            treatment_column_feature_ids,
            config=config,
            seed=seed + 100 * subfold,
            one_standard_error_rule=(
                config.nuisance_prediction_one_standard_error_rule
            ),
        )
        outcome_fit = (
            _logistic_elastic_net(
                outcome_design[fit],
                outcome[fit],
                outcome_design[heldout],
                outcome_column_feature_ids,
                config=config,
                seed=seed + 100 * subfold + 1,
                one_standard_error_rule=(
                    config.nuisance_prediction_one_standard_error_rule
                ),
            )
            if binary_outcome
            else _squared_error_elastic_net(
                outcome_design[fit],
                outcome[fit],
                outcome_design[heldout],
                outcome_column_feature_ids,
                config=config,
                seed=seed + 100 * subfold + 1,
                one_standard_error_rule=(
                    config.nuisance_prediction_one_standard_error_rule
                ),
            )
        )
        base_e[heldout] = treatment_fit.valid_prediction
        base_m[heldout] = outcome_fit.valid_prediction
    if splits and (np.isnan(base_e).any() or np.isnan(base_m).any()):
        raise ValueError("nested nuisance subfolds did not predict every fit row")
    return base_e, base_m, splits


def _nuisance_calibration_prediction(
    *,
    target: np.ndarray,
    base_train_prediction: np.ndarray,
    candidate_train: np.ndarray,
    base_valid_prediction: np.ndarray,
    candidate_valid: np.ndarray,
    candidate_column_feature_ids: Sequence[str],
    binary: bool,
    config: Stage2ElasticNetSelectionConfig,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Augment one base nuisance with a candidate using grouped elastic net."""

    base_train = np.asarray(base_train_prediction, dtype=float)
    base_valid = np.asarray(base_valid_prediction, dtype=float)
    if binary:
        base_train = np.log(
            np.clip(base_train, 1e-6, 1 - 1e-6)
            / np.clip(1.0 - base_train, 1e-6, 1 - 1e-6)
        )
        base_valid = np.log(
            np.clip(base_valid, 1e-6, 1 - 1e-6)
            / np.clip(1.0 - base_valid, 1e-6, 1 - 1e-6)
        )
        base_name = "base_logit"
    else:
        base_name = "base_prediction"
    train_additions = np.column_stack([base_train, candidate_train])
    valid_additions = np.column_stack([base_valid, candidate_valid])
    _, _, kept = _rank_safe_column_pairs(
        np.ones((len(target), 1), dtype=float),
        np.ones((len(base_valid), 1), dtype=float),
        train_additions,
        valid_additions,
    )
    input_names = [base_name] + [
        f"candidate_column_{index}" for index in range(candidate_train.shape[1])
    ]
    group_ids = ["__base_nuisance_prediction__"] + list(
        map(str, candidate_column_feature_ids)
    )
    train_x = train_additions[:, kept]
    valid_x = valid_additions[:, kept]
    kept_group_ids = [group_ids[index] for index in kept]
    try:
        fit = (
            _logistic_elastic_net(
                train_x,
                target,
                valid_x,
                kept_group_ids,
                config=config,
                seed=seed,
                one_standard_error_rule=(
                    config.nuisance_prediction_one_standard_error_rule
                ),
            )
            if binary
            else _squared_error_elastic_net(
                train_x,
                target,
                valid_x,
                kept_group_ids,
                config=config,
                seed=seed,
                one_standard_error_rule=(
                    config.nuisance_prediction_one_standard_error_rule
                ),
            )
        )
        prediction = np.asarray(fit.valid_prediction, dtype=float)
        if binary:
            prediction = np.clip(prediction, 1e-6, 1 - 1e-6)
        selected, magnitudes = _selected_feature_ids(
            fit.coefficients,
            kept_group_ids,
            tolerance=float(config.coefficient_tolerance),
        )
        return prediction, {
            "status": fit.status,
            "model_family": "group_elastic_net",
            "regularization": fit.regularization,
            "input_columns": [input_names[index] for index in kept],
            "selected_input_groups": selected,
            "input_group_l2_norms": magnitudes,
        }
    except Exception as exc:
        prediction = np.asarray(base_valid_prediction, dtype=float)
        if binary:
            prediction = np.clip(prediction, 1e-6, 1 - 1e-6)
        return prediction, {
            "status": f"base_prediction_fallback: {type(exc).__name__}: {exc}",
            "model_family": "group_elastic_net",
            "input_columns": [input_names[index] for index in kept],
        }


def _candidate_r_learner_test(
    *,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    treatment_train: np.ndarray,
    treatment_valid: np.ndarray,
    outcome_train: np.ndarray,
    outcome_valid: np.ndarray,
    base_e_train: np.ndarray,
    base_e_valid: np.ndarray,
    base_m_train: np.ndarray,
    base_m_valid: np.ndarray,
    augmentation_splits: Sequence[tuple[np.ndarray, np.ndarray]],
    feature: Mapping[str, Any],
    binary_outcome: bool,
    config: Stage2ElasticNetSelectionConfig,
    seed: int,
) -> dict[str, Any]:
    """Score one candidate using elastic-net nuisances and held-out R-loss."""

    train_eligible = propensity_eligibility(
        base_e_train, config.min_propensity, config.max_propensity
    )
    strategy = _feature_strategy(feature)
    result: dict[str, Any] = {
        "feature_id": _feature_key(feature),
        "name": str(feature["name"]),
        "candidate_strategy": strategy,
        "categorical_interactions_are_grouped": strategy == "categorical",
        "missingness_interactions": False,
    }
    design = _encode_design(
        train,
        valid,
        [feature],
        categorical_min_count=int(config.categorical_min_count),
    )
    design = _winsorize_modifier_design(
        design,
        quantile=float(config.modifier_continuous_winsor_quantile),
    )
    candidate_interaction_indices = [
        index
        for index, name in enumerate(design.column_names)
        if not str(name).endswith(":missing")
    ]
    interaction_indices: list[int] = []
    excluded_low_support: list[int] = []
    for index in candidate_interaction_indices:
        name = str(design.column_names[index])
        if ":level=" not in name and ":fallback=" not in name:
            interaction_indices.append(index)
            continue
        values = design.train[:, index]
        present = values == float(np.max(values))
        treated = int(np.sum(present & train_eligible & (treatment_train == 1)))
        untreated = int(np.sum(present & train_eligible & (treatment_train == 0)))
        if min(treated, untreated) >= int(config.categorical_min_count):
            interaction_indices.append(index)
        else:
            excluded_low_support.append(index)
    result["encoded_main_columns"] = list(map(str, design.column_names))
    result["candidate_interaction_columns"] = [
        str(design.column_names[index]) for index in candidate_interaction_indices
    ]
    result["excluded_interaction_columns_low_treatment_arm_support"] = [
        str(design.column_names[index]) for index in excluded_low_support
    ]
    if not augmentation_splits:
        return {
            **result,
            "status": "not_evaluable",
            "reason": "insufficient_rows_for_nested_cross_fitting",
        }
    if not interaction_indices:
        return {
            **result,
            "status": "not_evaluable",
            "reason": "no_nonconstant_candidate_interaction_columns",
        }

    e_train = np.full(len(train), np.nan, dtype=float)
    m_train = np.full(len(train), np.nan, dtype=float)
    treatment_calibration_statuses: list[str] = []
    outcome_calibration_statuses: list[str] = []
    for subfold, (fit, heldout) in enumerate(augmentation_splits, start=1):
        e_train[heldout], e_status = _nuisance_calibration_prediction(
            target=treatment_train[fit],
            base_train_prediction=base_e_train[fit],
            candidate_train=design.train[fit],
            base_valid_prediction=base_e_train[heldout],
            candidate_valid=design.train[heldout],
            candidate_column_feature_ids=design.column_feature_ids,
            binary=True,
            config=config,
            seed=seed + 100 * subfold,
        )
        m_train[heldout], m_status = _nuisance_calibration_prediction(
            target=outcome_train[fit],
            base_train_prediction=base_m_train[fit],
            candidate_train=design.train[fit],
            base_valid_prediction=base_m_train[heldout],
            candidate_valid=design.train[heldout],
            candidate_column_feature_ids=design.column_feature_ids,
            binary=binary_outcome,
            config=config,
            seed=seed + 100 * subfold + 1,
        )
        treatment_calibration_statuses.append(str(e_status["status"]))
        outcome_calibration_statuses.append(str(m_status["status"]))
    if np.isnan(e_train).any() or np.isnan(m_train).any():
        return {
            **result,
            "status": "not_evaluable",
            "reason": "candidate_nuisance_cross_fitting_was_incomplete",
        }
    e_valid, e_valid_status = _nuisance_calibration_prediction(
        target=treatment_train,
        base_train_prediction=base_e_train,
        candidate_train=design.train,
        base_valid_prediction=base_e_valid,
        candidate_valid=design.valid,
        candidate_column_feature_ids=design.column_feature_ids,
        binary=True,
        config=config,
        seed=seed + 10_000,
    )
    m_valid, m_valid_status = _nuisance_calibration_prediction(
        target=outcome_train,
        base_train_prediction=base_m_train,
        candidate_train=design.train,
        base_valid_prediction=base_m_valid,
        candidate_valid=design.valid,
        candidate_column_feature_ids=design.column_feature_ids,
        binary=binary_outcome,
        config=config,
        seed=seed + 10_001,
    )

    train_keep = propensity_eligibility(base_e_train, config.min_propensity, config.max_propensity)
    valid_keep = propensity_eligibility(base_e_valid, config.min_propensity, config.max_propensity)
    result["propensity_overlap"] = {
        "train": overlap_diagnostics(base_e_train, config.min_propensity, config.max_propensity),
        "valid": overlap_diagnostics(base_e_valid, config.min_propensity, config.max_propensity),
        "eligibility_source": "base_nuisance_common_to_all_candidates",
    }
    if (
        train_keep.sum() < 2
        or not valid_keep.any()
        or len(np.unique(treatment_train[train_keep])) < 2
    ):
        return {
            **result,
            "status": "not_evaluable",
            "reason": "insufficient_propensity_eligible_rows",
        }
    train, valid = train.iloc[np.flatnonzero(train_keep)], valid.iloc[np.flatnonzero(valid_keep)]
    design = replace(design, train=design.train[train_keep], valid=design.valid[valid_keep])
    treatment_train, outcome_train = treatment_train[train_keep], outcome_train[train_keep]
    treatment_valid, outcome_valid = treatment_valid[valid_keep], outcome_valid[valid_keep]
    e_train, m_train = e_train[train_keep], m_train[train_keep]
    e_valid, m_valid = e_valid[valid_keep], m_valid[valid_keep]
    base_e_valid, base_m_valid = base_e_valid[valid_keep], base_m_valid[valid_keep]
    rt_train = np.asarray(treatment_train - e_train, dtype=float)
    rt_valid = np.asarray(treatment_valid - e_valid, dtype=float)
    ry_train = np.asarray(outcome_train - m_train, dtype=float)
    ry_valid = np.asarray(outcome_valid - m_valid, dtype=float)
    reduced_train, reduced_valid, kept_main = _rank_safe_column_pairs(
        np.ones((len(train), 1), dtype=float),
        np.ones((len(valid), 1), dtype=float),
        design.train,
        design.valid,
    )
    reduced_train, reduced_valid, kept_treatment = _rank_safe_column_pairs(
        reduced_train,
        reduced_valid,
        rt_train.reshape(-1, 1),
        rt_valid.reshape(-1, 1),
    )
    if not kept_treatment:
        return {
            **result,
            "status": "not_evaluable",
            "reason": "residualized_treatment_has_no_independent_variation",
        }
    interaction_train = (
        rt_train.reshape(-1, 1) * design.train[:, interaction_indices]
    )
    interaction_valid = (
        rt_valid.reshape(-1, 1) * design.valid[:, interaction_indices]
    )
    full_train, full_valid, kept_interactions = _rank_safe_column_pairs(
        reduced_train,
        reduced_valid,
        interaction_train,
        interaction_valid,
    )
    if not kept_interactions:
        return {
            **result,
            "status": "not_evaluable",
            "reason": "no_independent_residual_treatment_interaction_columns",
        }
    try:
        reduced_model = Ridge(
            alpha=float(config.modifier_ridge_alpha),
            fit_intercept=False,
        )
        full_model = Ridge(
            alpha=float(config.modifier_ridge_alpha),
            fit_intercept=False,
        )
        reduced_model.fit(reduced_train, ry_train)
        full_model.fit(full_train, ry_train)
        reduced_prediction = reduced_model.predict(reduced_valid)
        full_prediction = full_model.predict(full_valid)
        reduced_loss = float(mean_squared_error(ry_valid, reduced_prediction))
        full_loss = float(mean_squared_error(ry_valid, full_prediction))
        improvement = float(reduced_loss - full_loss)
        relative_improvement = float(improvement / max(reduced_loss, 1e-15))
        if not all(
            math.isfinite(value)
            for value in (reduced_loss, full_loss, improvement, relative_improvement)
        ):
            raise ValueError("nonfinite held-out R-loss")
    except Exception as exc:
        return {
            **result,
            "status": "not_evaluable",
            "reason": f"{type(exc).__name__}: {exc}",
        }
    interaction_names = [
        str(design.column_names[index]) for index in interaction_indices
    ]
    return {
        **result,
        "status": "ok",
        "encoded_main_columns": [
            str(design.column_names[index]) for index in kept_main
        ],
        "tested_interaction_columns": [
            interaction_names[index] for index in kept_interactions
        ],
        "interaction_degrees_of_freedom": len(kept_interactions),
        "ridge_alpha": float(config.modifier_ridge_alpha),
        "constant_effect_coefficient": float(reduced_model.coef_[-1]),
        "heldout_reduced_r_loss": reduced_loss,
        "heldout_full_r_loss": full_loss,
        "heldout_r_loss_improvement": improvement,
        "heldout_relative_r_loss_improvement": relative_improvement,
        "candidate_augmented_nuisance_models": {
            "model_family": "group_elastic_net",
            "training_predictions_are_cross_fitted": True,
            "crossfit_folds": len(augmentation_splits),
            "treatment_calibration_statuses": treatment_calibration_statuses,
            "outcome_calibration_statuses": outcome_calibration_statuses,
            "heldout_treatment_calibration_status": e_valid_status["status"],
            "heldout_outcome_calibration_status": m_valid_status["status"],
            "heldout_base_treatment_log_loss": float(
                log_loss(treatment_valid, base_e_valid, labels=[0, 1])
            ),
            "heldout_augmented_treatment_log_loss": float(
                log_loss(treatment_valid, e_valid, labels=[0, 1])
            ),
            "heldout_base_outcome_loss": _loss(
                outcome_valid, base_m_valid, binary=binary_outcome
            ),
            "heldout_augmented_outcome_loss": _loss(
                outcome_valid, m_valid, binary=binary_outcome
            ),
        },
    }


def _rank_modifier_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    evaluable = [
        row
        for row in rows
        if row.get("status") == "ok"
        and row.get("heldout_r_loss_improvement") is not None
    ]
    return [
        {
            "rank": rank,
            "feature_id": str(row["feature_id"]),
            "name": str(row["name"]),
            "heldout_r_loss_improvement": float(
                row["heldout_r_loss_improvement"]
            ),
            "heldout_relative_r_loss_improvement": float(
                row["heldout_relative_r_loss_improvement"]
            ),
            "heldout_reduced_r_loss": float(row["heldout_reduced_r_loss"]),
            "heldout_full_r_loss": float(row["heldout_full_r_loss"]),
        }
        for rank, row in enumerate(
            sorted(
                evaluable,
                key=lambda row: (
                    -float(row["heldout_r_loss_improvement"]),
                    str(row["feature_id"]),
                ),
            ),
            start=1,
        )
    ]


def _benjamini_hochberg(values: Sequence[float | None]) -> list[float | None]:
    """Return multiplicity-adjusted q-values while preserving missing entries."""

    valid = [
        (index, float(value))
        for index, value in enumerate(values)
        if value is not None and math.isfinite(float(value))
    ]
    result: list[float | None] = [None] * len(values)
    if not valid:
        return result
    ordered = sorted(valid, key=lambda item: (item[1], item[0]))
    adjusted = [0.0] * len(ordered)
    running = 1.0
    total = len(ordered)
    for reverse_position in range(total - 1, -1, -1):
        _index, p_value = ordered[reverse_position]
        rank = reverse_position + 1
        running = min(running, float(p_value) * total / rank)
        adjusted[reverse_position] = min(1.0, running)
    for (index, _p_value), q_value in zip(ordered, adjusted):
        result[index] = float(q_value)
    return result


def _rank_p_values(
    rows: Sequence[Mapping[str, Any]],
    key: str,
) -> list[dict[str, Any]]:
    evaluable = [row for row in rows if row.get(key) is not None]
    return [
        {
            "rank": rank,
            "feature_id": str(row["feature_id"]),
            "name": str(row["name"]),
            key: float(row[key]),
        }
        for rank, row in enumerate(
            sorted(
                evaluable,
                key=lambda row: (float(row[key]), str(row["feature_id"])),
            ),
            start=1,
        )
    ]


def _confounder_univariable_rows(
    *,
    frame: pd.DataFrame,
    treatment: np.ndarray,
    outcome: np.ndarray,
    definitions: Sequence[Mapping[str, Any]],
    binary_outcome: bool,
    p_value_threshold: float,
    q_value_threshold: float,
) -> list[dict[str, Any]]:
    """Compute simple candidate-wise treatment and outcome associations."""

    intercept = np.ones((len(frame), 1), dtype=float)
    outcome_adjustment, _ = _univariable_rank_safe_columns(
        intercept,
        np.asarray(treatment, dtype=float).reshape(-1, 1),
    )
    rows: list[dict[str, Any]] = []
    for feature in definitions:
        design = _encode_univariable_feature(frame, feature)
        treatment_p, treatment_test = _binary_nested_p_value(
            treatment,
            intercept,
            design.main,
        )
        outcome_test_function = (
            _binary_nested_p_value if binary_outcome else _continuous_nested_p_value
        )
        outcome_p, outcome_test = outcome_test_function(
            outcome,
            intercept,
            design.main,
        )
        adjusted_outcome_p, adjusted_outcome_test = outcome_test_function(
            outcome,
            outcome_adjustment,
            design.main,
        )
        rows.append(
            {
                "feature_id": _feature_key(feature),
                "name": str(feature["name"]),
                "treatment_p_value": treatment_p,
                "outcome_p_value": outcome_p,
                "outcome_adjusted_for_treatment_p_value": adjusted_outcome_p,
                "treatment_test": treatment_test,
                "outcome_test": outcome_test,
                "outcome_adjusted_for_treatment_test": adjusted_outcome_test,
            }
        )
    for p_key in (
        "treatment_p_value",
        "outcome_p_value",
        "outcome_adjusted_for_treatment_p_value",
    ):
        q_key = p_key.replace("p_value", "q_value")
        for row, q_value in zip(
            rows,
            _benjamini_hochberg([row.get(p_key) for row in rows]),
        ):
            row[q_key] = q_value
    for row in rows:
        row["nominal_joint_support"] = bool(
            row["treatment_p_value"] is not None
            and row["outcome_adjusted_for_treatment_p_value"] is not None
            and float(row["treatment_p_value"]) < float(p_value_threshold)
            and float(row["outcome_adjusted_for_treatment_p_value"])
            < float(p_value_threshold)
        )
        row["multiplicity_adjusted_joint_support"] = bool(
            row["treatment_q_value"] is not None
            and row["outcome_adjusted_for_treatment_q_value"] is not None
            and float(row["treatment_q_value"]) < float(q_value_threshold)
            and float(row["outcome_adjusted_for_treatment_q_value"])
            < float(q_value_threshold)
        )
    return rows


def _joint_modifier_elastic_net_fold(
    *,
    context: Mapping[str, Any],
    definitions: Sequence[Mapping[str, Any]],
    config: Stage2ElasticNetSelectionConfig,
    seed: int,
) -> dict[str, Any]:
    """Fit one multivariable group-elastic-net R-loss interaction screen."""

    context = dict(context)
    trim = {}
    for part in ("train", "valid"):
        p = np.asarray(context[f"base_e_{part}"], dtype=float)
        keep = propensity_eligibility(p, config.min_propensity, config.max_propensity)
        trim[part] = overlap_diagnostics(p, config.min_propensity, config.max_propensity)
        context[part] = context[part].iloc[np.flatnonzero(keep)].reset_index(drop=True)
        for prefix in ("base_e", "base_m", "treatment", "outcome"):
            context[f"{prefix}_{part}"] = np.asarray(context[f"{prefix}_{part}"])[keep]
    train = context["train"]
    valid = context["valid"]
    if len(train) < 2 or not len(valid) or len(np.unique(context["treatment_train"])) < 2:
        raise ValueError(
            "propensity bounds leave insufficient rows or treatment arms for modifier selection"
        )
    treatment_train = np.asarray(context["treatment_train"], dtype=float)
    treatment_valid = np.asarray(context["treatment_valid"], dtype=float)
    outcome_train = np.asarray(context["outcome_train"], dtype=float)
    outcome_valid = np.asarray(context["outcome_valid"], dtype=float)
    design = _winsorize_modifier_design(
        _encode_design(
            train,
            valid,
            definitions,
            categorical_min_count=int(config.categorical_min_count),
        ),
        quantile=float(config.modifier_continuous_winsor_quantile),
    )
    interaction_indices: list[int] = []
    excluded_low_support: list[str] = []
    for index, name in enumerate(design.column_names):
        rendered_name = str(name)
        if rendered_name.endswith(":missing"):
            continue
        if ":level=" in rendered_name or ":fallback=" in rendered_name:
            values = design.train[:, index]
            present = values == float(np.max(values))
            treated = int(np.sum(present & (treatment_train == 1)))
            untreated = int(np.sum(present & (treatment_train == 0)))
            if min(treated, untreated) < int(config.categorical_min_count):
                excluded_low_support.append(rendered_name)
                continue
        interaction_indices.append(index)

    interaction_feature_ids = tuple(
        str(design.column_feature_ids[index]) for index in interaction_indices
    )
    interaction_names = tuple(
        str(design.column_names[index]) for index in interaction_indices
    )
    candidate_train = design.train[:, interaction_indices]
    candidate_valid = design.valid[:, interaction_indices]
    residual_treatment_train = treatment_train - np.asarray(
        context["base_e_train"], dtype=float
    )
    residual_treatment_valid = treatment_valid - np.asarray(
        context["base_e_valid"], dtype=float
    )
    residual_outcome_train = outcome_train - np.asarray(
        context["base_m_train"], dtype=float
    )
    residual_outcome_valid = outcome_valid - np.asarray(
        context["base_m_valid"], dtype=float
    )
    weights = np.square(residual_treatment_train)
    denominator = float(np.sum(weights))
    centers = (
        np.average(candidate_train, axis=0, weights=weights)
        if candidate_train.shape[1] and denominator > 1e-12
        else np.zeros(candidate_train.shape[1], dtype=float)
    )
    centered_train = candidate_train - centers
    centered_valid = candidate_valid - centers
    constant_effect = (
        float(
            np.sum(residual_treatment_train * residual_outcome_train)
            / denominator
        )
        if denominator > 1e-12
        else 0.0
    )
    modifier_target = (
        residual_outcome_train - residual_treatment_train * constant_effect
    )
    modifier_fit = _squared_error_elastic_net(
        residual_treatment_train.reshape(-1, 1) * centered_train,
        modifier_target,
        residual_treatment_valid.reshape(-1, 1) * centered_valid,
        interaction_feature_ids,
        config=config,
        seed=seed,
        fit_intercept=False,
        one_standard_error_rule=bool(config.modifier_one_standard_error_rule),
    )
    selected_ids, magnitudes = _selected_feature_ids(
        modifier_fit.coefficients,
        interaction_feature_ids,
        tolerance=float(config.coefficient_tolerance),
    )
    reduced_prediction = residual_treatment_valid * constant_effect
    full_prediction = reduced_prediction + modifier_fit.valid_prediction
    reduced_loss = float(
        mean_squared_error(residual_outcome_valid, reduced_prediction)
    )
    full_loss = float(mean_squared_error(residual_outcome_valid, full_prediction))
    return {
        "propensity_overlap": trim,
        "fit_rows": len(train),
        "heldout_rows": len(valid),
        "encoded_candidate_columns": int(design.train.shape[1]),
        "tested_interaction_columns": list(interaction_names),
        "excluded_missingness_interactions": True,
        "excluded_interaction_columns_low_treatment_arm_support": (excluded_low_support),
        "constant_effect": constant_effect,
        "status": modifier_fit.status,
        "regularization_alpha": modifier_fit.regularization,
        "internal_cv_folds": modifier_fit.cv_folds,
        "selected_feature_ids": selected_ids,
        "feature_group_l2_norms": magnitudes,
        "solver_iterations": modifier_fit.iterations,
        "solver_converged": modifier_fit.converged,
        "heldout_reduced_r_loss": reduced_loss,
        "heldout_full_r_loss": full_loss,
        "heldout_r_loss_improvement": reduced_loss - full_loss,
    }


def select_stage2_features_elastic_net(
    *,
    dataset: pd.DataFrame,
    extracted_fit: pd.DataFrame,
    definitions: Sequence[Mapping[str, Any]],
    inner_splits: Sequence[Mapping[str, Any]],
    treatment_column: str,
    outcome_column: str,
    outcome_type: str,
    seed: int,
    policy: Stage2ElasticNetSelectionConfig,
    checkpoint_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], list[Any]]:
    """Select nuisance inputs and effect modifiers inside one outer fold."""

    policy.validate()
    if policy.selection_mode == "multi_model":
        from .stage2_multi_model_selection import select_stage2_features_multi_model

        return select_stage2_features_multi_model(
            dataset=dataset, extracted_fit=extracted_fit, definitions=definitions,
            inner_splits=inner_splits, treatment_column=treatment_column,
            outcome_column=outcome_column, outcome_type=outcome_type,
            seed=seed, policy=policy, checkpoint_dir=checkpoint_dir,
        )
    original = [copy.deepcopy(dict(feature)) for feature in definitions]
    if not original:
        report = {
            "schema_version": SCHEMA_VERSION,
            "temporal_scope": TEMPORAL_SCOPE,
            "status": "complete_no_candidates",
            "policy": policy.public_dict(),
            "decisions": [],
        }
        return [], report, [], []
    folds = list(inner_splits)
    if not folds:
        raise ValueError("Stage 2 group-elastic-net selection requires inner folds")
    if str(outcome_type) not in {"binary", "continuous"}:
        raise ValueError("Stage 2 outcome_type must be binary or continuous")
    by_id = {_feature_key(feature): feature for feature in original}
    if len(by_id) != len(original):
        raise ValueError(
            "Stage 2 group-elastic-net selection requires unique feature IDs"
        )
    extracted_by_id = extracted_fit.set_index("_oci_row_id", drop=False)
    treatment_votes = {feature_id: 0 for feature_id in by_id}
    outcome_votes = {feature_id: 0 for feature_id in by_id}
    screen_folds: list[dict[str, Any]] = []
    binary_outcome = str(outcome_type) == "binary"
    screen_t_observed: list[np.ndarray] = []
    screen_t_predicted: list[np.ndarray] = []
    screen_y_observed: list[np.ndarray] = []
    screen_y_predicted: list[np.ndarray] = []
    univariable_nominal_votes = {feature_id: 0 for feature_id in by_id}
    univariable_adjusted_votes = {feature_id: 0 for feature_id in by_id}
    univariable_folds: list[dict[str, Any]] = []

    for position, split in enumerate(folds, start=1):
        train_ids = [int(value) for value in split.get("fit_row_ids") or []]
        valid_ids = [int(value) for value in split.get("heldout_row_ids") or []]
        if not train_ids or not valid_ids:
            raise ValueError("each Stage 2 inner fold must have fit and heldout rows")
        train = extracted_by_id.loc[train_ids].reset_index(drop=True)
        valid = extracted_by_id.loc[valid_ids].reset_index(drop=True)
        design = _encode_design(
            train,
            valid,
            original,
            categorical_min_count=int(policy.categorical_min_count),
        )
        t_train = dataset.iloc[train_ids][treatment_column].to_numpy(dtype=float)
        t_valid = dataset.iloc[valid_ids][treatment_column].to_numpy(dtype=float)
        y_train = dataset.iloc[train_ids][outcome_column].to_numpy(dtype=float)
        y_valid = dataset.iloc[valid_ids][outcome_column].to_numpy(dtype=float)
        univariable_rows = _confounder_univariable_rows(
            frame=train,
            treatment=t_train,
            outcome=y_train,
            definitions=original,
            binary_outcome=binary_outcome,
            p_value_threshold=float(
                policy.univariable_confounder_p_value_threshold
            ),
            q_value_threshold=float(
                policy.univariable_confounder_q_value_threshold
            ),
        )
        for row in univariable_rows:
            feature_id = str(row["feature_id"])
            univariable_nominal_votes[feature_id] += int(
                bool(row["nominal_joint_support"])
            )
            univariable_adjusted_votes[feature_id] += int(
                bool(row["multiplicity_adjusted_joint_support"])
            )
        univariable_folds.append(
            {
                "inner_fold": int(split.get("inner_fold", position)),
                "fit_rows": len(train_ids),
                "evaluation_scope": "inner_fold_fit_rows_only",
                "tests": univariable_rows,
                "treatment_p_value_ranking": _rank_p_values(
                    univariable_rows, "treatment_p_value"
                ),
                "outcome_p_value_ranking": _rank_p_values(
                    univariable_rows, "outcome_p_value"
                ),
                "outcome_adjusted_for_treatment_p_value_ranking": (
                    _rank_p_values(
                        univariable_rows,
                        "outcome_adjusted_for_treatment_p_value",
                    )
                ),
            }
        )
        treatment_fit = _logistic_elastic_net(
            design.train,
            t_train,
            design.valid,
            design.column_feature_ids,
            config=policy,
            seed=seed + 100 * position,
        )
        outcome_fit = (
            _logistic_elastic_net(
                design.train,
                y_train,
                design.valid,
                design.column_feature_ids,
                config=policy,
                seed=seed + 100 * position + 1,
            )
            if binary_outcome
            else _squared_error_elastic_net(
                design.train,
                y_train,
                design.valid,
                design.column_feature_ids,
                config=policy,
                seed=seed + 100 * position + 1,
            )
        )
        treatment_selected, treatment_magnitudes = _selected_feature_ids(
            treatment_fit.coefficients,
            design.column_feature_ids,
            tolerance=float(policy.coefficient_tolerance),
        )
        outcome_selected, outcome_magnitudes = _selected_feature_ids(
            outcome_fit.coefficients,
            design.column_feature_ids,
            tolerance=float(policy.coefficient_tolerance),
        )
        for feature_id in treatment_selected:
            treatment_votes[feature_id] += 1
        for feature_id in outcome_selected:
            outcome_votes[feature_id] += 1
        screen_t_observed.append(t_valid)
        screen_t_predicted.append(treatment_fit.valid_prediction)
        screen_y_observed.append(y_valid)
        screen_y_predicted.append(outcome_fit.valid_prediction)
        screen_folds.append(
            {
                "inner_fold": int(split.get("inner_fold", position)),
                "fit_rows": len(train_ids),
                "heldout_rows": len(valid_ids),
                "encoded_columns": int(design.train.shape[1]),
                "treatment": {
                    "calibration": calibration_diagnostics(
                        t_valid, treatment_fit.valid_prediction, binary=True
                    ),
                    "status": treatment_fit.status,
                    "selected_feature_ids": treatment_selected,
                    "feature_group_l2_norms": treatment_magnitudes,
                    "regularization_alpha": treatment_fit.regularization,
                    "internal_cv_folds": treatment_fit.cv_folds,
                    "solver_iterations": treatment_fit.iterations,
                    "solver_converged": treatment_fit.converged,
                    "heldout_log_loss": float(
                        log_loss(
                            t_valid,
                            treatment_fit.valid_prediction,
                            labels=[0, 1],
                        )
                    ),
                    "heldout_auroc": _safe_auroc(
                        t_valid,
                        treatment_fit.valid_prediction,
                    ),
                },
                "outcome": {
                    "calibration": calibration_diagnostics(
                        y_valid, outcome_fit.valid_prediction, binary=binary_outcome
                    ),
                    "status": outcome_fit.status,
                    "selected_feature_ids": outcome_selected,
                    "feature_group_l2_norms": outcome_magnitudes,
                    "regularization": outcome_fit.regularization,
                    "internal_cv_folds": outcome_fit.cv_folds,
                    "solver_iterations": outcome_fit.iterations,
                    "solver_converged": outcome_fit.converged,
                    "heldout_loss": _loss(
                        y_valid,
                        outcome_fit.valid_prediction,
                        binary=binary_outcome,
                    ),
                    "heldout_auroc": (
                        _safe_auroc(y_valid, outcome_fit.valid_prediction)
                        if binary_outcome
                        else None
                    ),
                },
            }
        )

    any_treatment = _selected_in_any_inner_fold(treatment_votes)
    any_outcome = _selected_in_any_inner_fold(outcome_votes)
    locked_confounders = {
        _feature_key(feature)
        for feature in original
        if feature.get("configured_explicit_feature") is True
        and "confounder" in set(map(str, feature.get("roles") or []))
    }
    confounder_union = any_treatment | any_outcome | locked_confounders

    nuisance_definitions = [
        by_id[feature_id] for feature_id in by_id if feature_id in confounder_union
    ]
    if policy.selection_mode == "independent_tasks":
        # Cross-fold screen votes are final routing evidence, not a feature
        # gate for another fold's nuisance predictions. Fit both nuisances on
        # all candidates, with independently CV-selected regularization.
        nuisance_definitions = original
    treatment_definitions = nuisance_definitions
    outcome_definitions = nuisance_definitions
    all_fit_ids = sorted(
        {
            int(value)
            for split in folds
            for key in ("fit_row_ids", "heldout_row_ids")
            for value in split.get(key) or []
        }
    )
    id_position = {row_id: position for position, row_id in enumerate(all_fit_ids)}
    oof_e = np.full(len(all_fit_ids), np.nan, dtype=float)
    oof_m = np.full(len(all_fit_ids), np.nan, dtype=float)
    nuisance_folds: list[dict[str, Any]] = []
    modifier_contexts: list[dict[str, Any]] = []
    for position, split in enumerate(folds, start=1):
        train_ids = [int(value) for value in split.get("fit_row_ids") or []]
        valid_ids = [int(value) for value in split.get("heldout_row_ids") or []]
        train = extracted_by_id.loc[train_ids].reset_index(drop=True)
        valid = extracted_by_id.loc[valid_ids].reset_index(drop=True)
        t_design = _encode_design(
            train,
            valid,
            treatment_definitions,
            categorical_min_count=int(policy.categorical_min_count),
        )
        y_design = _encode_design(
            train,
            valid,
            outcome_definitions,
            categorical_min_count=int(policy.categorical_min_count),
        )
        t_train = dataset.iloc[train_ids][treatment_column].to_numpy(dtype=float)
        t_valid = dataset.iloc[valid_ids][treatment_column].to_numpy(dtype=float)
        y_train = dataset.iloc[train_ids][outcome_column].to_numpy(dtype=float)
        y_valid = dataset.iloc[valid_ids][outcome_column].to_numpy(dtype=float)
        treatment_nuisance_fit = _logistic_elastic_net(
            t_design.train,
            t_train,
            t_design.valid,
            t_design.column_feature_ids,
            config=policy,
            seed=seed + 10_000 + position,
            one_standard_error_rule=(
                policy.nuisance_prediction_one_standard_error_rule
            ),
        )
        outcome_nuisance_fit = (
            _logistic_elastic_net(
                y_design.train,
                y_train,
                y_design.valid,
                y_design.column_feature_ids,
                config=policy,
                seed=seed + 20_000 + position,
                one_standard_error_rule=(
                    policy.nuisance_prediction_one_standard_error_rule
                ),
            )
            if binary_outcome
            else _squared_error_elastic_net(
                y_design.train,
                y_train,
                y_design.valid,
                y_design.column_feature_ids,
                config=policy,
                seed=seed + 20_000 + position,
                one_standard_error_rule=(
                    policy.nuisance_prediction_one_standard_error_rule
                ),
            )
        )
        e_valid = treatment_nuisance_fit.valid_prediction
        m_valid = outcome_nuisance_fit.valid_prediction
        treatment_nuisance_selected, _ = _selected_feature_ids(
            treatment_nuisance_fit.coefficients,
            t_design.column_feature_ids,
            tolerance=float(policy.coefficient_tolerance),
        )
        outcome_nuisance_selected, _ = _selected_feature_ids(
            outcome_nuisance_fit.coefficients,
            y_design.column_feature_ids,
            tolerance=float(policy.coefficient_tolerance),
        )
        nested_e_train, nested_m_train, augmentation_splits = (
            _cross_fitted_base_nuisances(
                treatment_design=t_design.train,
                treatment_column_feature_ids=t_design.column_feature_ids,
                outcome_design=y_design.train,
                outcome_column_feature_ids=y_design.column_feature_ids,
                treatment=t_train,
                outcome=y_train,
                binary_outcome=binary_outcome,
                config=policy,
                seed=seed + 30_000 + 1_000 * position,
            )
        )
        locations = [id_position[row_id] for row_id in valid_ids]
        if np.isfinite(oof_e[locations]).any() or np.isfinite(oof_m[locations]).any():
            raise ValueError("Stage 2 inner folds predict an outer-training row more than once")
        oof_e[locations] = e_valid
        oof_m[locations] = m_valid
        nuisance_folds.append(
            {
                "inner_fold": int(split.get("inner_fold", position)),
                "fit_rows": len(train_ids),
                "heldout_rows": len(valid_ids),
                "treatment_features": len(treatment_definitions),
                "outcome_features": len(outcome_definitions),
                "treatment_encoded_columns": int(t_design.train.shape[1]),
                "outcome_encoded_columns": int(y_design.train.shape[1]),
                "treatment_status": treatment_nuisance_fit.status,
                "outcome_status": outcome_nuisance_fit.status,
                "treatment_regularization_alpha": (treatment_nuisance_fit.regularization),
                "outcome_regularization_alpha": outcome_nuisance_fit.regularization,
                "treatment_selected_feature_ids": treatment_nuisance_selected,
                "outcome_selected_feature_ids": outcome_nuisance_selected,
                "heldout_treatment_log_loss": float(log_loss(t_valid, e_valid, labels=[0, 1])),
                "heldout_treatment_auroc": _safe_auroc(t_valid, e_valid),
                "heldout_outcome_loss": _loss(y_valid, m_valid, binary=binary_outcome),
                "heldout_outcome_auroc": (
                    _safe_auroc(y_valid, m_valid) if binary_outcome else None
                ),
                "calibration": {
                    "heldout_treatment": calibration_diagnostics(t_valid, e_valid, binary=True),
                    "heldout_outcome": calibration_diagnostics(
                        y_valid, m_valid, binary=binary_outcome
                    ),
                    "nested_training_treatment": calibration_diagnostics(
                        t_train, nested_e_train, binary=True
                    ),
                    "nested_training_outcome": calibration_diagnostics(
                        y_train, nested_m_train, binary=binary_outcome
                    ),
                },
                "propensity_overlap": overlap_diagnostics(
                    e_valid, policy.min_propensity, policy.max_propensity
                ),
                "nested_modifier_training_crossfit_folds": len(augmentation_splits),
            }
        )
        modifier_contexts.append(
            {
                "split": split,
                "train": train,
                "valid": valid,
                "treatment_train": t_train,
                "treatment_valid": t_valid,
                "outcome_train": y_train,
                "outcome_valid": y_valid,
                "base_e_train": nested_e_train,
                "base_e_valid": e_valid,
                "base_m_train": nested_m_train,
                "base_m_valid": m_valid,
                "augmentation_splits": augmentation_splits,
            }
        )
    if np.isnan(oof_e).any() or np.isnan(oof_m).any():
        raise ValueError(
            "Stage 2 inner splits must provide one out-of-fold nuisance prediction "
            "for every outer-training row"
        )
    t_all = dataset.iloc[all_fit_ids][treatment_column].to_numpy(dtype=float)
    y_all = dataset.iloc[all_fit_ids][outcome_column].to_numpy(dtype=float)

    LOGGER.info(
        "Stage 2 held-out nuisance calibration treatment=%s outcome=%s overlap=%s",
        calibration_diagnostics(t_all, oof_e, binary=True),
        calibration_diagnostics(y_all, oof_m, binary=binary_outcome),
        overlap_diagnostics(oof_e, policy.min_propensity, policy.max_propensity),
    )
    modifier_votes = {feature_id: 0 for feature_id in by_id}
    modifier_r_loss_improvements = {feature_id: [] for feature_id in by_id}
    modifier_folds: list[dict[str, Any]] = []
    modifier_elastic_net_votes = {feature_id: 0 for feature_id in by_id}
    modifier_elastic_net_folds: list[dict[str, Any]] = []
    for position, context in enumerate(modifier_contexts, start=1):
        split = context["split"]
        train = context["train"]
        valid = context["valid"]
        elastic_net_fold = _joint_modifier_elastic_net_fold(
            context=context,
            definitions=original,
            config=policy,
            seed=seed + 50_000 * position,
        )
        elastic_net_fold["inner_fold"] = int(
            split.get("inner_fold", position)
        )
        for feature_id in elastic_net_fold["selected_feature_ids"]:
            modifier_elastic_net_votes[str(feature_id)] += 1
        modifier_elastic_net_folds.append(elastic_net_fold)
        tests = [
            _candidate_r_learner_test(
                train=train,
                valid=valid,
                treatment_train=context["treatment_train"],
                treatment_valid=context["treatment_valid"],
                outcome_train=context["outcome_train"],
                outcome_valid=context["outcome_valid"],
                base_e_train=context["base_e_train"],
                base_e_valid=context["base_e_valid"],
                base_m_train=context["base_m_train"],
                base_m_valid=context["base_m_valid"],
                augmentation_splits=context["augmentation_splits"],
                feature=feature,
                binary_outcome=binary_outcome,
                config=policy,
                seed=seed + 100_000 * position + candidate_position,
            )
            for candidate_position, feature in enumerate(original, start=1)
        ]
        ranking = _rank_modifier_rows(tests)
        selected_ids = [
            str(row["feature_id"])
            for row in ranking[: int(policy.modifier_top_n_per_inner_fold)]
        ]
        selected_set = set(selected_ids)
        rank_by_id = {
            str(row["feature_id"]): int(row["rank"])
            for row in ranking
        }
        for row in tests:
            feature_id = str(row["feature_id"])
            row["rank"] = rank_by_id.get(feature_id)
            row["selected_top_n"] = feature_id in selected_set
            improvement = row.get("heldout_r_loss_improvement")
            if improvement is not None:
                modifier_r_loss_improvements[feature_id].append(float(improvement))
        for feature_id in selected_ids:
            modifier_votes[feature_id] += 1
        modifier_folds.append(
            {
                "inner_fold": int(split.get("inner_fold", position)),
                "fit_rows": len(train),
                "heldout_rows": len(valid),
                "model_family": "candidate_augmented_univariable_r_learner",
                "base_nuisance_model_family": "group_elastic_net",
                "candidate_nuisance_model_family": "group_elastic_net",
                "base_nuisance_predictions_for_fit_are_nested_cross_fitted": True,
                "candidate_nuisance_augmentations_are_cross_fitted": True,
                "modifier_evaluation_rows_are_untouched_inner_heldout": True,
                "candidate_models": len(tests),
                "evaluable_candidates": len(ranking),
                "requested_top_n": int(policy.modifier_top_n_per_inner_fold),
                "selected_feature_ids": selected_ids,
                "selected_count": len(selected_ids),
                "heldout_r_loss_improvement_ranking": ranking,
                "tests": tests,
            }
        )
        LOGGER.info(
            "Stage 2 modifier screen inner_fold=%s candidates=%s "
            "evaluable=%s selected=%s top_n=%s",
            int(split.get("inner_fold", position)),
            len(tests),
            len(ranking),
            len(selected_ids),
            int(policy.modifier_top_n_per_inner_fold),
        )

    modifier_union = _selected_in_any_inner_fold(modifier_votes)
    locked_modifiers = {
        _feature_key(feature)
        for feature in original
        if feature.get("configured_explicit_feature") is True
        and "effect_modifier" in set(map(str, feature.get("roles") or []))
    }
    modifier_union.update(locked_modifiers)

    selected: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for feature in original:
        feature_id = _feature_key(feature)
        nuisance_roles = (
            ["treatment", "outcome"] if feature_id in confounder_union else []
        )
        roles: list[str] = []
        if nuisance_roles:
            roles.append("confounder")
        if feature_id in modifier_union:
            roles.append("effect_modifier")
        configured = feature.get("configured_explicit_feature") is True
        if configured:
            configured_roles = list(dict.fromkeys(map(str, feature.get("roles") or [])))
            roles = configured_roles
            if "confounder" in configured_roles:
                nuisance_roles = ["treatment", "outcome"]
        retained = bool(roles)
        decisions.append(
            {
                "feature_id": feature_id,
                "name": str(feature["name"]),
                "configured_explicit_feature": configured,
                "treatment_votes": int(treatment_votes[feature_id]),
                "outcome_votes": int(outcome_votes[feature_id]),
                "modifier_votes": int(modifier_votes[feature_id]),
                "univariable_nominal_confounder_votes": int(
                    univariable_nominal_votes[feature_id]
                ),
                "univariable_adjusted_confounder_votes": int(
                    univariable_adjusted_votes[feature_id]
                ),
                "multivariable_modifier_elastic_net_votes": int(
                    modifier_elastic_net_votes[feature_id]
                ),
                "treatment_selection_frequency": float(
                    treatment_votes[feature_id] / len(folds)
                ),
                "outcome_selection_frequency": float(outcome_votes[feature_id] / len(folds)),
                "modifier_selection_frequency": float(
                    modifier_votes[feature_id] / len(folds)
                ),
                "modifier_max_heldout_r_loss_improvement": (
                    float(max(modifier_r_loss_improvements[feature_id]))
                    if modifier_r_loss_improvements[feature_id]
                    else None
                ),
                "modifier_mean_heldout_r_loss_improvement": (
                    float(np.mean(modifier_r_loss_improvements[feature_id]))
                    if modifier_r_loss_improvements[feature_id]
                    else None
                ),
                "nuisance_model_roles": nuisance_roles,
                "roles": roles,
                "retained": retained,
                "selection_source": (
                    "investigator_locked"
                    if configured
                    else (
                        "group_elastic_net_nuisance_and_candidate_augmented_"
                        "rlearner_any_inner_fold_union"
                    )
                ),
            }
        )
        if retained:
            updated = copy.deepcopy(feature)
            updated["roles"] = roles
            updated["nuisance_model_roles"] = nuisance_roles
            updated["selection_source"] = (
                "investigator_locked"
                if configured
                else (
                    "group_elastic_net_nuisance_and_candidate_augmented_"
                    "rlearner_any_inner_fold_union"
                )
            )
            selected.append(updated)

    report = {
        "schema_version": SCHEMA_VERSION,
        "temporal_scope": TEMPORAL_SCOPE,
        "status": "complete",
        "selection_method": (
            "any_inner_fold_union_group_elastic_net_nuisance_and_top_n_"
            "candidate_augmented_univariable_rlearners"
        ),
        "penalized_nuisance_screen_model_family": "group_lasso_plus_ridge",
        # Backward-compatible report key; group elastic-net penalization applies
        # to nuisance selection and prediction, while the R-effect comparison
        # itself is ridge stabilized.
        "penalized_model_family": "group_lasso_plus_ridge",
        "encoding": {
            "ordinal": "single_training_standardized_ordered_score",
            "nominal": "training_standardized_reference_contrasts",
            "missing_indicator": "same_penalty_group_as_measurement",
            "group_weight": "square_root_of_encoded_group_rank",
        },
        "latent_construction": "disabled",
        "pairwise_association_screen": "disabled",
        "policy": policy.public_dict(),
        "inner_folds": len(folds),
        "nuisance_screen": {
            "folds": screen_folds,
            "selection_rule": "selected_in_any_inner_fold_for_either_task",
            "required_votes": 1,
            "treatment_votes": dict(sorted(treatment_votes.items())),
            "outcome_votes": dict(sorted(outcome_votes.items())),
            "stable_treatment_feature_ids": sorted(any_treatment),
            "stable_outcome_feature_ids": sorted(any_outcome),
            "union_confounder_feature_ids": sorted(confounder_union),
            "intersection_is_not_a_selection_gate": True,
            "union_is_used_by_both_nuisance_models": True,
            "overall_treatment_auroc": _safe_auroc(
                np.concatenate(screen_t_observed),
                np.concatenate(screen_t_predicted),
            ),
            "overall_outcome_auroc": (
                _safe_auroc(
                    np.concatenate(screen_y_observed),
                    np.concatenate(screen_y_predicted),
                )
                if binary_outcome
                else None
            ),
        },
        "confounder_univariable_screen": {
            "objective": (
                "candidate-wise association with treatment and outcome, including "
                "outcome association adjusted only for treatment"
            ),
            "role": "selection_evidence_only",
            "hard_selection_gate": False,
            "fit_scope": "inner_fold_fit_rows_only",
            "categorical_tests_are_omnibus": True,
            "p_value_threshold": float(policy.univariable_confounder_p_value_threshold),
            "q_value_threshold": float(policy.univariable_confounder_q_value_threshold),
            "multiplicity_adjustment": "benjamini_hochberg_within_inner_fold_endpoint",
            "nominal_joint_support_votes": dict(sorted(univariable_nominal_votes.items())),
            "multiplicity_adjusted_joint_support_votes": dict(
                sorted(univariable_adjusted_votes.items())
            ),
            "folds": univariable_folds,
        },
        "cross_fitted_nuisance_models": {
            "model_family": "group_elastic_net",
            "penalty": "group_lasso_plus_ridge",
            "l1_ratio": float(policy.l1_ratio),
            "regularization_selected_by_internal_cv": True,
            "one_standard_error_rule": bool(policy.nuisance_prediction_one_standard_error_rule),
            "treatment_feature_ids": sorted(confounder_union),
            "outcome_feature_ids": sorted(confounder_union),
            "folds": nuisance_folds,
            "overall_treatment_log_loss": float(log_loss(t_all, oof_e, labels=[0, 1])),
            "overall_outcome_loss": _loss(y_all, oof_m, binary=binary_outcome),
            "overall_treatment_auroc": _safe_auroc(t_all, oof_e),
            "overall_outcome_auroc": (_safe_auroc(y_all, oof_m) if binary_outcome else None),
            "propensity_min": float(np.min(oof_e)),
            "propensity_max": float(np.max(oof_e)),
            "predictions_are_inner_fold_out_of_fold": True,
            "calibration": {
                "treatment": calibration_diagnostics(t_all, oof_e, binary=True),
                "outcome": calibration_diagnostics(y_all, oof_m, binary=binary_outcome),
            },
            "propensity_overlap": overlap_diagnostics(
                oof_e, policy.min_propensity, policy.max_propensity
            ),
            "predictions": [
                {
                    "_oci_row_id": int(row_id),
                    "treatment": float(t),
                    "outcome": float(y),
                    "propensity": float(e),
                    "outcome_prediction": float(m),
                    "effect_eligible": bool(keep),
                }
                for row_id, t, y, e, m, keep in zip(
                    all_fit_ids,
                    t_all,
                    y_all,
                    oof_e,
                    oof_m,
                    propensity_eligibility(oof_e, policy.min_propensity, policy.max_propensity),
                )
            ],
        },
        "effect_modifier_screen": {
            "objective": (
                "candidate-wise held-out squared R-loss gain after augmenting "
                "both elastic-net nuisance models with the candidate"
            ),
            "model_family": "candidate_augmented_univariable_r_learner",
            "nuisance_model_family": "group_elastic_net",
            "nuisance_augmentation": (
                "cross-fitted candidate-specific grouped elastic-net calibration "
                "for treatment and outcome"
            ),
            "r_learner_regularization": {
                "family": "ridge",
                "alpha": float(policy.modifier_ridge_alpha),
            },
            "continuous_candidate_winsorization": {
                "fit_scope": "modifier_inner_fold_training_rows_only",
                "two_sided_quantile": float(policy.modifier_continuous_winsor_quantile),
            },
            "r_learner_comparison": (
                "reduced residual-outcome model has candidate main effects and a "
                "constant residual-treatment effect; full model jointly adds all "
                "residual-treatment-by-candidate terms"
            ),
            "categorical_interaction_handling": (
                "all estimable nonreference-level interactions enter the full "
                "candidate model together and receive one held-out R-loss score"
            ),
            "categorical_interaction_minimum_rows_per_treatment_arm": int(
                policy.categorical_min_count
            ),
            "candidate_scope": "one_candidate_group_per_model",
            "training_rows": (
                "inner-fold fit rows with base and candidate-augmented elastic-net "
                "nuisance predictions cross-fitted entirely inside those rows"
            ),
            "evaluation_rows": (
                "untouched inner-heldout rows excluded from nuisance augmentation "
                "and R-learner fitting"
            ),
            "folds": modifier_folds,
            "selection_rule": ("union_of_top_n_heldout_r_loss_gains_from_each_inner_fold"),
            "top_n_per_inner_fold": int(policy.modifier_top_n_per_inner_fold),
            "required_votes": 1,
            "votes": dict(sorted(modifier_votes.items())),
            "r_loss_improvement_threshold_is_not_a_selection_gate": True,
            "non_evaluable_candidates_are_not_ranked": True,
            "missingness_interactions": False,
            "stable_effect_modifier_feature_ids": sorted(modifier_union),
        },
        "multivariable_modifier_elastic_net_screen": {
            "objective": (
                "joint grouped elastic-net treatment-interaction selection under " "squared R-loss"
            ),
            "role": "selection_evidence_only",
            "hard_selection_gate": False,
            "model_family": "group_elastic_net_r_learner",
            "candidate_scope": "all_candidate_groups_in_one_model_per_inner_fold",
            "nuisance_model_family": "group_elastic_net",
            "training_nuisance_predictions_are_nested_cross_fitted": True,
            "evaluation_rows_are_untouched_inner_heldout": True,
            "one_standard_error_rule": bool(policy.modifier_one_standard_error_rule),
            "missingness_interactions": False,
            "votes": dict(sorted(modifier_elastic_net_votes.items())),
            "selected_in_any_inner_fold_feature_ids": sorted(
                _selected_in_any_inner_fold(modifier_elastic_net_votes)
            ),
            "mean_heldout_r_loss_improvement": float(
                np.mean(
                    [float(row["heldout_r_loss_improvement"]) for row in modifier_elastic_net_folds]
                )
            ),
            "positive_heldout_r_loss_improvement_fraction": float(
                np.mean(
                    np.asarray(
                        [
                            float(row["heldout_r_loss_improvement"])
                            for row in modifier_elastic_net_folds
                        ]
                    )
                    > 0.0
                )
            ),
            "folds": modifier_elastic_net_folds,
        },
        "decisions": decisions,
        "retained_feature_ids": [_feature_key(feature) for feature in selected],
        "measurement_dependency_feature_ids": [_feature_key(feature) for feature in selected],
    }
    if policy.selection_mode == "independent_tasks":
        from .stage2_taskwise_policy import finalize_taskwise_report

        selected, report = finalize_taskwise_report(original, report)
    return selected, report, [copy.deepcopy(feature) for feature in selected], []


__all__ = [
    "SCHEMA_VERSION",
    "TEMPORAL_SCOPE",
    "Stage2ElasticNetSelectionConfig",
    "select_stage2_features_elastic_net",
    "statistical_selection_config_from_mapping",
]
