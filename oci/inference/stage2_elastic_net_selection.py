"""Grouped elastic-net primitives and statistical policy for Stage 2 selection.

The deterministic design encoder, the proximal-gradient group elastic-net
solver, and the cross-fitting helpers here are shared by the multi-model
numerical evidence (``stage2_multi_model_selection``), the DINA adapters,
matched-batch evidence, and estimand-informed ontology search.  All fits use
only the current outer-training partition and inner-fold honest evaluation.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold

from .nuisance_diagnostics import (
    propensity_eligibility,
    validate_propensity_bounds,
)

# Re-exported for ``stage2_multi_model_selection``, which reaches these
# univariable nested tests through this module.
from .stage2_statistical_selection import (
    _binary_nested_p_value,
    _continuous_nested_p_value,
    _encode_feature as _encode_univariable_feature,
    _rank_safe_columns as _univariable_rank_safe_columns,
)
from .stage2_multi_model_config import Stage2MultiModelConfig, multi_model_config_from_mapping
from .stage2_extraction_failures import failure_flags, has_failure_indicator


TEMPORAL_SCOPE = "pre_index_treatment"


@dataclass(frozen=True)
class Stage2ElasticNetSelectionConfig:
    """Scientific and numerical policy for multi-model Stage 2 selection."""

    min_propensity: float | None = None
    max_propensity: float | None = None
    l1_ratio: float = 0.8
    internal_cv_folds: int = 3
    minimum_log10_alpha: float = -5.0
    maximum_log10_alpha: float = -1.0
    coefficient_tolerance: float = 1e-7
    optimization_tolerance: float = 1e-6
    categorical_min_count: int = 5
    max_iter: int = 5_000
    modifier_one_standard_error_rule: bool = False
    modifier_ridge_alpha: float = 10.0
    modifier_continuous_winsor_quantile: float = 0.005
    multi_model: Stage2MultiModelConfig = field(default_factory=Stage2MultiModelConfig)

    def validate(self) -> None:
        validate_propensity_bounds(self.min_propensity, self.max_propensity)
        if not isinstance(self.multi_model, Stage2MultiModelConfig):
            raise ValueError("multi_model must be a Stage2MultiModelConfig object")
        self.multi_model.validate()
        value = self.l1_ratio
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 < float(value) <= 1.0
        ):
            raise ValueError("stage2.statistical_selection.l1_ratio must be in (0, 1]")
        for name in ("internal_cv_folds", "categorical_min_count", "max_iter"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(
                    f"stage2.statistical_selection.{name} must be a positive integer"
                )
        if self.internal_cv_folds < 2:
            raise ValueError(
                "stage2.statistical_selection.internal_cv_folds must be at least 2"
            )
        for name in (
            "minimum_log10_alpha",
            "maximum_log10_alpha",
            "coefficient_tolerance",
            "optimization_tolerance",
            "modifier_ridge_alpha",
            "modifier_continuous_winsor_quantile",
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
        if self.modifier_ridge_alpha <= 0.0:
            raise ValueError(
                "stage2.statistical_selection.modifier_ridge_alpha must be positive"
            )
        if not 0.0 <= self.modifier_continuous_winsor_quantile < 0.25:
            raise ValueError(
                "stage2.statistical_selection.modifier_continuous_winsor_quantile "
                "must be in [0, 0.25)"
            )
        if not isinstance(self.modifier_one_standard_error_rule, bool):
            raise ValueError(
                "stage2.statistical_selection.modifier_one_standard_error_rule must be boolean"
            )

    def public_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for name in ("min_propensity", "max_propensity"):
            if result[name] is None:
                result.pop(name)
        result["multi_model"] = self.multi_model.public_dict()
        return result


def statistical_selection_config_from_mapping(
    value: Mapping[str, Any] | None,
) -> Stage2ElasticNetSelectionConfig:
    if value is not None and not isinstance(value, Mapping):
        raise ValueError("stage2.statistical_selection must be an object")
    raw = dict(value or {})
    if "selection_mode" in raw and raw.pop("selection_mode") != "multi_model":
        raise ValueError(
            "stage2.statistical_selection.selection_mode: only 'multi_model' is "
            "supported; llm_roles and independent_tasks were removed"
        )
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


def _append_failure_column(
    train_columns: list[np.ndarray],
    valid_columns: list[np.ndarray],
    column_feature_ids: list[str],
    column_names: list[str],
    *,
    definition: Mapping[str, Any],
    feature_id: str,
    name: str,
    train_failed: pd.Series,
    valid_failed: pd.Series,
) -> None:
    """Grouped extraction-failure indicator (dropped when constant in training)."""

    if not has_failure_indicator(definition):
        return
    _append_variable_column(
        train_columns,
        valid_columns,
        column_feature_ids,
        column_names,
        feature_id=feature_id,
        name=f"{name}:extraction_failed",
        train_values=train_failed.to_numpy(dtype=float),
        valid_values=valid_failed.to_numpy(dtype=float),
    )


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
        # Extraction failures (null after all repairs) are not "not documented":
        # they get their own indicator in the parent feature's column group.
        train_failed = failure_flags(train, definition).reset_index(drop=True)
        valid_failed = failure_flags(valid, definition).reset_index(drop=True)
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
                    (train_numeric.isna() if strategy == "ordinal" else train_series.isna())
                    & ~train_failed
                ).to_numpy(),
                valid_values=(
                    (valid_numeric.isna() if strategy == "ordinal" else valid_series.isna())
                    & ~valid_failed
                ).to_numpy(),
            )
            _append_failure_column(
                train_columns, valid_columns, column_feature_ids, column_names,
                definition=definition, feature_id=feature_id, name=name,
                train_failed=train_failed, valid_failed=valid_failed,
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
            train_values=(train_missing & ~train_failed).to_numpy(),
            valid_values=(valid_missing & ~valid_failed).to_numpy(),
        )
        _append_failure_column(
            train_columns, valid_columns, column_feature_ids, column_names,
            definition=definition, feature_id=feature_id, name=name,
            train_failed=train_failed, valid_failed=valid_failed,
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
class _GroupStructure:
    starts: np.ndarray
    sizes: np.ndarray
    weights: np.ndarray
    feature_ids: tuple[str, ...]
    penalty_factors: np.ndarray


@dataclass(frozen=True)
class _SolverState:
    coefficients: np.ndarray
    intercept: float
    iterations: int
    converged: bool


def _group_structure(
    column_feature_ids: Sequence[str],
    *,
    unpenalized_feature_ids: Sequence[str] = (),
) -> _GroupStructure:
    """Build feature groups, explicitly exempting any supplied IDs from both penalties."""
    if not column_feature_ids:
        return _GroupStructure(
            starts=np.asarray([], dtype=int),
            sizes=np.asarray([], dtype=int),
            weights=np.asarray([], dtype=float),
            feature_ids=(),
            penalty_factors=np.asarray([], dtype=float),
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
    unpenalized = set(unpenalized_feature_ids)
    return _GroupStructure(
        starts=start_values,
        sizes=sizes,
        weights=np.sqrt(sizes.astype(float)),
        feature_ids=tuple(feature_ids),
        penalty_factors=np.asarray([key not in unpenalized for key in feature_ids], dtype=float),
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
        * groups.penalty_factors
    )
    scales = np.maximum(0.0, 1.0 - thresholds / np.maximum(norms, 1e-30))
    scales /= 1.0 + (
        float(step) * float(regularization) * (1.0 - float(group_ratio))
        * groups.penalty_factors
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


def _loss(observed: np.ndarray, predicted: np.ndarray, *, binary: bool) -> float:
    if binary:
        return float(
            log_loss(observed, np.clip(predicted, 1e-6, 1 - 1e-6), labels=[0, 1])
        )
    return float(mean_squared_error(observed, predicted))


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


__all__ = [
    "TEMPORAL_SCOPE",
    "Stage2ElasticNetSelectionConfig",
    "statistical_selection_config_from_mapping",
]
