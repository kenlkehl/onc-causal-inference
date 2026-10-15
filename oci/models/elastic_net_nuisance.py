"""Clone-safe elastic-net estimators for causal nuisance functions.

The causal workflows repeatedly fit nuisance models inside an outer/inner
cross-fitting scheme.  These wrappers keep the model family and tuning rule
identical in direct Stage 2 fits and in estimators cloned by EconML.  They also
handle the small-fold edge cases where an internal regularization CV would not
have enough observations from both classes.

Both wrappers standardize their design internally.  A ``StandardScaler`` is fit
on exactly the rows passed to ``fit`` (labels are never used) and the same
fitted scaler is applied at prediction time.  Because every cross-fitting fold
fits its own clone, each fold also fits its own scaler, so held-out rows never
inform the scaling used to predict them.  The elastic-net penalty and the saga
or coordinate-descent convergence are therefore invariant to the measurement
scale of each column.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.linear_model import ElasticNetCV, LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


_STANDARDIZATION_FIT_SCOPE = (
    "rows passed to fit (unsupervised; refit per cross-fitting fold)"
)


def _fit_standardizer(design: Any) -> tuple[StandardScaler, Any]:
    """Fit a label-free scaler on the fit rows and return the scaled design.

    Sparse designs are scaled without centering so they stay sparse.  Columns
    with zero variance keep scale one (sklearn's documented behavior), so a
    constant column becomes zero after centering instead of producing NaN.
    """

    scaler = StandardScaler(with_mean=not sparse.issparse(design), with_std=True)
    return scaler, scaler.fit_transform(design)


def _apply_standardizer(scaler: StandardScaler, design: Any) -> Any:
    if scaler.with_mean and sparse.issparse(design):
        # The scaler was fit on a dense design and centers columns; densify
        # instead of failing so sparse and dense inputs predict identically.
        design = design.toarray()
    return scaler.transform(design)


def _has_varying_column(scaler: StandardScaler, standardized: Any) -> bool:
    """Whether any column carries signal after label-free standardization.

    Centered constant columns are identically zero.  saga stops after one
    epoch when every coefficient is zero and unchanged, which would leave the
    intercept unconverged, so such designs use the intercept-only fit.
    """

    if scaler.with_mean:
        constant = np.asarray(scaler.scale_, dtype=float) != np.sqrt(
            np.asarray(scaler.var_, dtype=float)
        )
        return not bool(np.all(constant))
    return bool(standardized.count_nonzero() > 0)


def _standardization_audit(scaler: StandardScaler | None) -> dict[str, Any]:
    zero_variance_columns = None
    if scaler is not None:
        # sklearn replaces the scale of (near-)constant columns by one.
        zero_variance_columns = int(
            np.count_nonzero(
                np.asarray(scaler.scale_, dtype=float)
                != np.sqrt(np.asarray(scaler.var_, dtype=float))
            )
        )
    return {
        "method": "StandardScaler",
        "fit_scope": _STANDARDIZATION_FIT_SCOPE,
        "applied": scaler is not None,
        "with_mean": bool(scaler.with_mean) if scaler is not None else None,
        "zero_variance_columns": zero_variance_columns,
    }


def _positive_int(value: Any, *, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonzero_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result == 0:
        raise ValueError(f"{name} must be nonzero")
    return result


class ElasticNetLogisticClassifier(ClassifierMixin, BaseEstimator):
    """Standardized logistic elastic net with deterministic, fold-adaptive CV."""

    def __init__(
        self,
        *,
        l1_ratio: float = 0.8,
        cv_folds: int = 3,
        regularization_grid_size: int = 16,
        minimum_log10_c: float = -2.0,
        maximum_log10_c: float = 4.0,
        max_iter: int = 5_000,
        tolerance: float = 1e-4,
        random_state: int = 42,
        n_jobs: int = 1,
    ) -> None:
        self.l1_ratio = l1_ratio
        self.cv_folds = cv_folds
        self.regularization_grid_size = regularization_grid_size
        self.minimum_log10_c = minimum_log10_c
        self.maximum_log10_c = maximum_log10_c
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _validated_parameters(self) -> tuple[float, int, np.ndarray, int, float, int, int]:
        ratio = _finite_float(self.l1_ratio, name="l1_ratio")
        if not 0.0 < ratio <= 1.0:
            raise ValueError("l1_ratio must be in (0, 1]")
        folds = _positive_int(self.cv_folds, name="cv_folds", minimum=2)
        grid_size = _positive_int(
            self.regularization_grid_size,
            name="regularization_grid_size",
            minimum=3,
        )
        minimum = _finite_float(self.minimum_log10_c, name="minimum_log10_c")
        maximum = _finite_float(self.maximum_log10_c, name="maximum_log10_c")
        if maximum <= minimum:
            raise ValueError("maximum_log10_c must exceed minimum_log10_c")
        max_iter = _positive_int(self.max_iter, name="max_iter")
        tolerance = _finite_float(self.tolerance, name="tolerance")
        if tolerance <= 0.0:
            raise ValueError("tolerance must be positive")
        seed = _positive_int(self.random_state, name="random_state", minimum=0)
        jobs = _nonzero_int(self.n_jobs, name="n_jobs")
        return (
            ratio,
            folds,
            np.logspace(minimum, maximum, grid_size),
            max_iter,
            tolerance,
            seed,
            jobs,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ElasticNetLogisticClassifier":
        (
            ratio,
            requested_folds,
            grid,
            max_iter,
            tolerance,
            seed,
            jobs,
        ) = self._validated_parameters()
        design, target = check_X_y(
            X,
            y,
            accept_sparse=True,
            dtype=float,
            ensure_min_features=0,
        )
        target = np.asarray(target).reshape(-1)
        values, counts = np.unique(target, return_counts=True)
        if not set(values.tolist()).issubset({0, 1}):
            raise ValueError("elastic-net logistic target must contain only 0 and 1")
        self.n_features_in_ = int(design.shape[1])
        self.classes_ = np.asarray([0, 1], dtype=int)
        self.constant_probability_ = None
        self.model_ = None
        self.scaler_ = None
        self.selected_c_ = None
        self.cv_n_iter_ = None
        self.refit_n_iter_ = None
        self.fit_mode_ = "constant"
        self.constant_reason_ = None
        self.effective_cv_folds_ = 0
        constant_probability = float(np.clip(np.mean(target), 1e-6, 1 - 1e-6))
        if len(values) < 2 or design.shape[1] == 0:
            self.constant_reason_ = "single_class" if len(values) < 2 else "no_features"
            self.constant_probability_ = constant_probability
            return self
        self.scaler_, standardized = _fit_standardizer(design)
        if not _has_varying_column(self.scaler_, standardized):
            self.constant_reason_ = "no_varying_features"
            self.constant_probability_ = constant_probability
            return self

        folds = min(requested_folds, int(counts.min()))
        common = {
            "penalty": "elasticnet",
            "solver": "saga",
            "fit_intercept": True,
            "max_iter": max_iter,
            "tol": tolerance,
            "random_state": seed,
            "n_jobs": jobs,
        }
        labels = target.astype(int)
        if folds >= 2:
            self.fit_mode_ = "cross_validated"
            self.effective_cv_folds_ = int(folds)
            splitter = StratifiedKFold(
                n_splits=folds,
                shuffle=True,
                random_state=seed,
            )
            # Score the grid with sklearn's own CV path, then refit
            # explicitly so the refit's iteration count is auditable.  sklearn
            # 1.6 needs an array (not a list) of l1 ratios when refit=False.
            search = LogisticRegressionCV(
                Cs=grid,
                cv=splitter,
                scoring="neg_log_loss",
                refit=False,
                l1_ratios=np.asarray([ratio], dtype=float),
                **common,
            )
            search.fit(standardized, labels)
            positive_class = search.classes_[-1]
            # Shapes: scores (folds, Cs, l1_ratios); coefficient paths
            # (folds, Cs, l1_ratios, features + intercept).
            fold_scores = np.asarray(search.scores_[positive_class], dtype=float)[
                :, :, 0
            ]
            # Same rule as LogisticRegressionCV(refit=True): maximize the
            # neg_log_loss summed over folds; argmax keeps the first tie.
            best_index = int(np.argmax(fold_scores.sum(axis=0)))
            self.selected_c_ = float(search.Cs_[best_index])
            self.cv_n_iter_ = np.asarray(search.n_iter_, dtype=int).reshape(-1)
            coefficient_start = np.mean(
                np.asarray(search.coefs_paths_[positive_class], dtype=float)[
                    :, best_index, 0, :
                ],
                axis=0,
            )
            self.model_ = LogisticRegression(
                C=self.selected_c_,
                l1_ratio=ratio,
                warm_start=True,
                **common,
            )
            # sklearn's internal refit starts from the fold-averaged
            # coefficients at the selected C; seeding the warm start the same
            # way reproduces that refit exactly.
            self.model_.coef_ = coefficient_start[:-1].reshape(1, -1)
            self.model_.intercept_ = coefficient_start[-1:].copy()
        else:
            self.fit_mode_ = "fixed_grid_midpoint"
            self.selected_c_ = float(grid[len(grid) // 2])
            self.model_ = LogisticRegression(
                C=self.selected_c_,
                l1_ratio=ratio,
                **common,
            )
        self.model_.fit(standardized, labels)
        self.refit_n_iter_ = int(np.asarray(self.model_.n_iter_).reshape(-1).max())
        self.classes_ = np.asarray(self.model_.classes_, dtype=int)
        return self

    def fit_audit(self) -> dict[str, Any]:
        """Return JSON-safe details from the fitted nuisance clone."""

        check_is_fitted(
            self,
            ("classes_", "model_", "fit_mode_", "effective_cv_folds_"),
        )
        max_iter = int(self.max_iter)
        selected_regularization = None
        if self.model_ is not None:
            selected_regularization = {
                "parameter": "C",
                "value": float(self.selected_c_),
            }
        cv_iterations = (
            int(self.cv_n_iter_.max())
            if self.cv_n_iter_ is not None and self.cv_n_iter_.size
            else None
        )
        refit_iterations = (
            int(self.refit_n_iter_) if self.refit_n_iter_ is not None else None
        )
        cv_limit_reached = bool(
            cv_iterations is not None and cv_iterations >= max_iter
        )
        refit_limit_reached = bool(
            refit_iterations is not None and refit_iterations >= max_iter
        )
        observed = [
            value for value in (cv_iterations, refit_iterations) if value is not None
        ]
        return {
            "estimator": (
                "oci.models.elastic_net_nuisance."
                "ElasticNetLogisticClassifier"
            ),
            "fit_mode": str(self.fit_mode_),
            "n_features": int(self.n_features_in_),
            "requested_cv_folds": int(self.cv_folds),
            "effective_cv_folds": int(self.effective_cv_folds_),
            "selected_regularization": selected_regularization,
            "constant_prediction": (
                float(self.constant_probability_)
                if self.constant_probability_ is not None
                else None
            ),
            "constant_reason": self.constant_reason_,
            "standardization": _standardization_audit(self.scaler_),
            "optimization": {
                "configured_max_iter": max_iter,
                # Maximum over every CV path fit and the final refit.
                "maximum_iterations_observed": max(observed) if observed else 0,
                "iteration_limit_reached": cv_limit_reached or refit_limit_reached,
                "cv_maximum_iterations_observed": cv_iterations,
                "cv_iteration_limit_reached": cv_limit_reached,
                "refit_iterations": refit_iterations,
                "refit_iteration_limit_reached": refit_limit_reached,
            },
        }

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ("classes_", "model_"))
        design = check_array(
            X,
            accept_sparse=True,
            dtype=float,
            ensure_min_features=0,
        )
        if self.model_ is not None:
            return np.asarray(
                self.model_.predict_proba(_apply_standardizer(self.scaler_, design)),
                dtype=float,
            )
        probability = np.full(
            design.shape[0],
            float(self.constant_probability_),
            dtype=float,
        )
        return np.column_stack([1.0 - probability, probability])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class ElasticNetRegressor(RegressorMixin, BaseEstimator):
    """Standardized squared-error elastic net with deterministic, adaptive CV."""

    def __init__(
        self,
        *,
        l1_ratio: float = 0.8,
        cv_folds: int = 3,
        regularization_grid_size: int = 16,
        minimum_log10_alpha: float = -5.0,
        maximum_log10_alpha: float = -1.0,
        max_iter: int = 5_000,
        tolerance: float = 1e-4,
        random_state: int = 42,
        n_jobs: int = 1,
    ) -> None:
        self.l1_ratio = l1_ratio
        self.cv_folds = cv_folds
        self.regularization_grid_size = regularization_grid_size
        self.minimum_log10_alpha = minimum_log10_alpha
        self.maximum_log10_alpha = maximum_log10_alpha
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ElasticNetRegressor":
        ratio = _finite_float(self.l1_ratio, name="l1_ratio")
        if not 0.0 < ratio <= 1.0:
            raise ValueError("l1_ratio must be in (0, 1]")
        requested_folds = _positive_int(self.cv_folds, name="cv_folds", minimum=2)
        grid_size = _positive_int(
            self.regularization_grid_size,
            name="regularization_grid_size",
            minimum=3,
        )
        minimum = _finite_float(
            self.minimum_log10_alpha,
            name="minimum_log10_alpha",
        )
        maximum = _finite_float(
            self.maximum_log10_alpha,
            name="maximum_log10_alpha",
        )
        if maximum <= minimum:
            raise ValueError("maximum_log10_alpha must exceed minimum_log10_alpha")
        max_iter = _positive_int(self.max_iter, name="max_iter")
        tolerance = _finite_float(self.tolerance, name="tolerance")
        if tolerance <= 0.0:
            raise ValueError("tolerance must be positive")
        seed = _positive_int(self.random_state, name="random_state", minimum=0)
        jobs = _nonzero_int(self.n_jobs, name="n_jobs")

        design, target = check_X_y(
            X,
            y,
            accept_sparse=False,
            dtype=float,
            ensure_min_features=0,
        )
        target = np.asarray(target, dtype=float).reshape(-1)
        self.n_features_in_ = int(design.shape[1])
        self.constant_mean_ = None
        self.model_ = None
        self.scaler_ = None
        self.fit_mode_ = "constant"
        self.constant_reason_ = None
        self.effective_cv_folds_ = 0
        if design.shape[1] == 0 or len(target) < 3 or float(np.var(target)) <= 1e-15:
            self.constant_reason_ = (
                "no_features"
                if design.shape[1] == 0
                else "too_few_rows" if len(target) < 3 else "constant_target"
            )
            self.constant_mean_ = float(np.mean(target))
            return self
        self.scaler_, standardized = _fit_standardizer(design)
        if not _has_varying_column(self.scaler_, standardized):
            self.constant_reason_ = "no_varying_features"
            self.constant_mean_ = float(np.mean(target))
            return self

        folds = min(requested_folds, len(target))
        self.fit_mode_ = "cross_validated"
        self.effective_cv_folds_ = int(folds)
        splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
        self.model_ = ElasticNetCV(
            l1_ratio=ratio,
            alphas=np.logspace(minimum, maximum, grid_size),
            fit_intercept=True,
            max_iter=max_iter,
            tol=tolerance,
            cv=splitter,
            n_jobs=jobs,
            random_state=seed,
            selection="cyclic",
        )
        self.model_.fit(standardized, target)
        return self

    def fit_audit(self) -> dict[str, Any]:
        """Return JSON-safe details from the fitted nuisance clone."""

        check_is_fitted(
            self,
            ("model_", "fit_mode_", "effective_cv_folds_"),
        )
        max_iter = int(self.max_iter)
        selected_regularization = None
        refit_iterations = None
        duality_gap = None
        if self.model_ is not None:
            selected_regularization = {
                "parameter": "alpha",
                "value": float(self.model_.alpha_),
            }
            # ElasticNetCV.n_iter_ belongs to the final refit at the selected
            # alpha; sklearn does not expose the per-fold path iterations.
            refit_iterations = int(np.asarray(self.model_.n_iter_).reshape(-1).max())
            duality_gap = float(np.asarray(self.model_.dual_gap_).reshape(-1).max())
        refit_limit_reached = bool(
            refit_iterations is not None and refit_iterations >= max_iter
        )
        return {
            "estimator": "oci.models.elastic_net_nuisance.ElasticNetRegressor",
            "fit_mode": str(self.fit_mode_),
            "n_features": int(self.n_features_in_),
            "requested_cv_folds": int(self.cv_folds),
            "effective_cv_folds": int(self.effective_cv_folds_),
            "selected_regularization": selected_regularization,
            "constant_prediction": (
                float(self.constant_mean_)
                if self.constant_mean_ is not None
                else None
            ),
            "constant_reason": self.constant_reason_,
            "standardization": _standardization_audit(self.scaler_),
            "optimization": {
                "configured_max_iter": max_iter,
                # Only the refit is observable (see cv_* keys below).
                "maximum_iterations_observed": (
                    refit_iterations if refit_iterations is not None else 0
                ),
                "iteration_limit_reached": refit_limit_reached,
                "cv_maximum_iterations_observed": None,
                "cv_iteration_limit_reached": None,
                "cv_iterations_note": (
                    "sklearn ElasticNetCV does not expose per-fold path iterations"
                ),
                "refit_iterations": refit_iterations,
                "refit_iteration_limit_reached": refit_limit_reached,
                "duality_gap": duality_gap,
            },
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "model_")
        design = check_array(
            X,
            accept_sparse=False,
            dtype=float,
            ensure_min_features=0,
        )
        if self.model_ is not None:
            return np.asarray(
                self.model_.predict(_apply_standardizer(self.scaler_, design)),
                dtype=float,
            )
        return np.full(design.shape[0], float(self.constant_mean_), dtype=float)


__all__ = ["ElasticNetLogisticClassifier", "ElasticNetRegressor"]
