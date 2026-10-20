"""Stacked super-learner estimators for Stage 2 nuisance functions.

The estimators combine a fixed library of base learners with non-negative
weights that sum to one (van der Laan, Polley and Hubbard 2007):

* ``elastic_net``: the standardized elastic-net wrappers of
  :mod:`oci.models.elastic_net_nuisance`, with their internal penalty CV;
* ``gradient_boosting``: scikit-learn histogram gradient boosting with the
  fixed settings in :data:`GRADIENT_BOOSTING_SETTINGS`;
* ``random_forest``: a scikit-learn random forest with the fixed settings in
  :data:`RANDOM_FOREST_SETTINGS`.

Inside the training rows, ``cv_folds``-fold internal cross-validation
(stratified for binary targets) gives out-of-fold predictions of every base
learner. The weights minimize the log loss of the combined probability (binary
targets) or the squared error of the combined prediction (continuous targets,
non-negative least squares normalized to sum to one). Learners with positive
weight are then refitted on all training rows. The estimators are
scikit-learn compatible, clone-safe and picklable; ``fit_audit()`` reports the
weights, the internal-CV losses (including an honest stack CV loss with weights
refitted on the other internal folds), each refitted learner's own audit and
the seeds.

Thread budget: ``n_jobs`` (default 1) is passed to the random forest; the
gradient-boosting OpenMP threads follow the caller's ``threadpoolctl`` limit.
Stage 2 fits these models inside its modeling pool, where every job is limited
to one thread.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
from scipy.optimize import minimize, nnls
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from .elastic_net_nuisance import ElasticNetLogisticClassifier, ElasticNetRegressor

LEARNERS = ("elastic_net", "gradient_boosting", "random_forest")
DEFAULT_CV_FOLDS = 5
PROBABILITY_CLIP = 1e-6
WEIGHT_TOLERANCE = 1e-8
# Pre-specified gradient-boosting settings: shallow trees, a small learning
# rate and L2 shrinkage; early stopping on an internal validation fraction when
# the training rows allow it (EARLY_STOPPING_MIN_ROWS, and for binary targets
# EARLY_STOPPING_MIN_CLASS_ROWS in the smaller class), otherwise max_iter.
GRADIENT_BOOSTING_SETTINGS = {
    "max_depth": 3,
    "learning_rate": 0.05,
    "max_iter": 300,
    "l2_regularization": 1.0,
    "min_samples_leaf": 20,
    "validation_fraction": 0.15,
    "n_iter_no_change": 20,
}
EARLY_STOPPING_MIN_ROWS = 100
EARLY_STOPPING_MIN_CLASS_ROWS = 10
# Pre-specified random-forest settings.
RANDOM_FOREST_SETTINGS = {
    "n_estimators": 500,
    "min_samples_leaf": 5,
    "max_features": "sqrt",
}
# Each component draws its own seed from the estimator's random_state. The
# elastic net uses random_state itself, as in the elastic-net-only learner.
SEED_OFFSETS = {"elastic_net": 0, "gradient_boosting": 1, "random_forest": 2, "internal_cv": 3}


@dataclass(frozen=True)
class SuperLearnerConfig:
    """``stage2.super_learner``: internal CV folds and the base-learner library."""

    cv_folds: int = DEFAULT_CV_FOLDS
    learners: tuple[str, ...] = LEARNERS

    def validate(self) -> None:
        if isinstance(self.cv_folds, bool) or not isinstance(self.cv_folds, int) or self.cv_folds < 2:
            raise ValueError("stage2.super_learner.cv_folds must be an integer of at least 2")
        if (
            not isinstance(self.learners, tuple)
            or not self.learners
            or any(not isinstance(name, str) or name not in LEARNERS for name in self.learners)
            or len(set(self.learners)) != len(self.learners)
            or self.learners != tuple(name for name in LEARNERS if name in self.learners)
        ):
            raise ValueError(
                "stage2.super_learner.learners must be a nonempty unique list from "
                f"{list(LEARNERS)} in that order"
            )

    def public_dict(self) -> dict[str, Any]:
        return {**asdict(self), "learners": list(self.learners)}


def super_learner_config_from_mapping(value: Mapping[str, Any] | None) -> SuperLearnerConfig:
    if value is None:
        return SuperLearnerConfig()
    if not isinstance(value, Mapping):
        raise ValueError("stage2.super_learner must be an object")
    raw = dict(value)
    unknown = sorted(set(raw) - set(SuperLearnerConfig.__dataclass_fields__))
    if unknown:
        raise ValueError(f"stage2.super_learner contains unsupported fields: {unknown}")
    if "learners" in raw:
        names = raw["learners"]
        if not isinstance(names, (list, tuple)) or any(
            not isinstance(name, str) or name not in LEARNERS for name in names
        ) or len(set(names)) != len(names):
            raise ValueError(
                f"stage2.super_learner.learners must list unique values from {list(LEARNERS)}"
            )
        raw["learners"] = tuple(name for name in LEARNERS if name in names)
    config = SuperLearnerConfig(**raw)
    config.validate()
    return config


def learner_spec(nuisance_learner: str, config: SuperLearnerConfig | None) -> dict[str, Any]:
    """The JSON learner description passed to the nuisance fits and fingerprints."""

    if nuisance_learner == "elastic_net":
        return {"name": "elastic_net"}
    if nuisance_learner != "super_learner":
        raise ValueError("stage2.nuisance_learner must be super_learner or elastic_net")
    config = config or SuperLearnerConfig()
    config.validate()
    return {"name": "super_learner", "cv_folds": config.cv_folds, "learners": list(config.learners)}


def _seed(random_state: int, component: str) -> int:
    return int((int(random_state) + SEED_OFFSETS[component]) % (2**32 - 1))


def _validated_learners(learners: Any) -> tuple[str, ...]:
    names = tuple(learners)
    if not names or len(set(names)) != len(names) or any(name not in LEARNERS for name in names):
        raise ValueError(f"learners must be a nonempty unique subset of {list(LEARNERS)}")
    return tuple(name for name in LEARNERS if name in names)


def _validated_folds(cv_folds: Any) -> int:
    if isinstance(cv_folds, bool) or not isinstance(cv_folds, (int, np.integer)) or cv_folds < 2:
        raise ValueError("cv_folds must be an integer of at least 2")
    return int(cv_folds)


def _clip(probability: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(probability, dtype=float), PROBABILITY_CLIP, 1.0 - PROBABILITY_CLIP)


def _log_loss(target: np.ndarray, probability: np.ndarray) -> float:
    p = _clip(probability)
    return float(-np.mean(target * np.log(p) + (1.0 - target) * np.log1p(-p)))


def _squared_error(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean((np.asarray(target, float) - np.asarray(prediction, float)) ** 2))


def _sparsify(weights: np.ndarray) -> np.ndarray:
    weights = np.where(np.asarray(weights, float) > WEIGHT_TOLERANCE, weights, 0.0)
    return weights / weights.sum()


def _log_loss_weights(predictions: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Simplex weights minimizing the log loss of the combined probability (SLSQP)."""

    k = predictions.shape[1]
    if k == 1:
        return np.ones(1), {"method": "single_learner"}
    probabilities = _clip(predictions)

    def objective(weights):
        return _log_loss(target, probabilities @ weights)

    def gradient(weights):
        p = _clip(probabilities @ weights)
        return -((target / p - (1.0 - target) / (1.0 - p)) @ probabilities) / len(target)

    result = minimize(
        objective,
        np.full(k, 1.0 / k),
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * k,
        constraints=({"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0),
                      "jac": lambda w: np.ones(k)},),
        options={"maxiter": 500, "ftol": 1e-12},
    )
    weights = np.clip(np.asarray(result.x, dtype=float), 0.0, None)
    audit = {"method": "slsqp_simplex_log_loss", "converged": bool(result.success),
             "iterations": int(result.nit)}
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        best = int(np.argmin([_log_loss(target, predictions[:, j]) for j in range(k)]))
        weights = np.eye(k)[best]
        audit["fallback"] = "best_single_learner"
    return _sparsify(weights), audit


def _least_squares_weights(predictions: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Non-negative least squares weights, normalized to sum to one."""

    k = predictions.shape[1]
    if k == 1:
        return np.ones(1), {"method": "single_learner"}
    weights, _ = nnls(np.asarray(predictions, float), np.asarray(target, float))
    audit = {"method": "nnls_normalized"}
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        best = int(np.argmin([_squared_error(target, predictions[:, j]) for j in range(k)]))
        weights = np.eye(k)[best]
        audit["fallback"] = "best_single_learner"
    return _sparsify(weights), audit


class _SuperLearnerBase(BaseEstimator):
    """Shared stacking logic; subclasses define the target type."""

    _binary = True

    def __init__(
        self,
        *,
        learners: tuple[str, ...] = LEARNERS,
        cv_folds: int = DEFAULT_CV_FOLDS,
        random_state: int = 42,
        n_jobs: int = 1,
    ) -> None:
        self.learners = learners
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.n_jobs = n_jobs

    # -- base learners -------------------------------------------------
    def _make(self, name: str, rows: int, smallest_class: int) -> Any:
        jobs = int(self.n_jobs)
        if name == "elastic_net":
            cls = ElasticNetLogisticClassifier if self._binary else ElasticNetRegressor
            return cls(random_state=_seed(self.random_state, name), n_jobs=jobs)
        if name == "gradient_boosting":
            early = rows >= EARLY_STOPPING_MIN_ROWS and (
                not self._binary or smallest_class >= EARLY_STOPPING_MIN_CLASS_ROWS
            )
            cls = HistGradientBoostingClassifier if self._binary else HistGradientBoostingRegressor
            return cls(**GRADIENT_BOOSTING_SETTINGS, early_stopping=early,
                       random_state=_seed(self.random_state, name))
        cls = RandomForestClassifier if self._binary else RandomForestRegressor
        return cls(**RANDOM_FOREST_SETTINGS, n_jobs=jobs, random_state=_seed(self.random_state, name))

    def _predict_one(self, model: Any, design: np.ndarray) -> np.ndarray:
        if not self._binary:
            return np.asarray(model.predict(design), dtype=float)
        probabilities = np.asarray(model.predict_proba(design), dtype=float)
        classes = list(model.classes_)
        return probabilities[:, classes.index(1)] if 1 in classes else np.zeros(len(design))

    def _fit_one(self, name: str, design: np.ndarray, target: np.ndarray) -> Any:
        counts = np.unique(target, return_counts=True)[1]
        return self._make(name, len(target), int(counts.min()) if self._binary else 0).fit(
            design, target.astype(int) if self._binary else target
        )

    @staticmethod
    def _learner_audit(name: str, model: Any) -> dict[str, Any]:
        if name == "elastic_net":
            return model.fit_audit()
        if name == "gradient_boosting":
            return {
                "estimator": type(model).__name__,
                "settings": dict(GRADIENT_BOOSTING_SETTINGS),
                "early_stopping": bool(model.early_stopping),
                "n_iter": int(model.n_iter_),
            }
        return {"estimator": type(model).__name__, "settings": dict(RANDOM_FOREST_SETTINGS),
                "n_jobs": int(model.n_jobs)}

    @staticmethod
    def _compact_internal_audit(name: str, model: Any) -> dict[str, Any]:
        if name == "elastic_net":
            audit = model.fit_audit()
            return {
                "selected_regularization": audit["selected_regularization"],
                "iteration_limit_reached": audit["optimization"]["iteration_limit_reached"],
            }
        if name == "gradient_boosting":
            return {"n_iter": int(model.n_iter_), "early_stopping": bool(model.early_stopping)}
        return {}

    # -- fitting -------------------------------------------------------
    def _loss(self, target, prediction):
        return _log_loss(target, prediction) if self._binary else _squared_error(target, prediction)

    def _weights(self, predictions, target):
        return (_log_loss_weights if self._binary else _least_squares_weights)(predictions, target)

    def _internal_splits(self, target: np.ndarray, folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
        seed = _seed(self.random_state, "internal_cv")
        splitter = (
            StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
            if self._binary else KFold(n_splits=folds, shuffle=True, random_state=seed)
        )
        return list(splitter.split(np.zeros(len(target)), target))

    def _stack(self, design: np.ndarray, target: np.ndarray, learners: tuple[str, ...]) -> None:
        requested = _validated_folds(self.cv_folds)
        if self._binary:
            available = int(np.unique(target, return_counts=True)[1].min())
            reason = "smaller class has fewer rows than cv_folds"
        else:
            available = len(target) // 2
            reason = "fewer than two rows per internal fold"
        folds = min(requested, available)
        self.effective_cv_folds_ = int(folds) if folds >= 2 else 0
        self.cv_folds_reason_ = reason if folds < requested else None
        if folds < 2:
            # No internal CV is possible: use the elastic net (or the first learner).
            chosen = "elastic_net" if "elastic_net" in learners else learners[0]
            self.fit_mode_ = "single_learner_fallback"
            self.weights_ = {name: float(name == chosen) for name in learners}
            return
        splits = self._internal_splits(target, folds)
        out_of_fold = np.full((len(target), len(learners)), np.nan)
        self.internal_cv_audit_ = {name: [] for name in learners}
        for fit_rows, hold_rows in splits:
            for column, name in enumerate(learners):
                model = self._fit_one(name, design[fit_rows], target[fit_rows])
                out_of_fold[hold_rows, column] = self._predict_one(model, design[hold_rows])
                self.internal_cv_audit_[name].append(self._compact_internal_audit(name, model))
        weights, optimizer = self._weights(out_of_fold, target)
        self.weights_ = {name: float(weight) for name, weight in zip(learners, weights)}
        self.weight_optimizer_ = optimizer
        self.learner_cv_loss_ = {
            name: self._loss(target, out_of_fold[:, column]) for column, name in enumerate(learners)
        }
        # Honest stack loss: weights refitted without each internal fold.
        stacked = np.empty(len(target))
        for fit_rows, hold_rows in splits:
            fold_weights, _ = self._weights(out_of_fold[fit_rows], target[fit_rows])
            stacked[hold_rows] = out_of_fold[hold_rows] @ fold_weights
        self.stack_cv_loss_ = self._loss(target, stacked)
        self.fit_mode_ = "stacked"

    def _fit_super_learner(self, design: np.ndarray, target: np.ndarray) -> None:
        learners = _validated_learners(self.learners)
        _validated_folds(self.cv_folds)
        self.learners_ = learners
        self.weights_ = {}
        self.models_ = {}
        self.learner_cv_loss_ = {}
        self.stack_cv_loss_ = None
        self.weight_optimizer_ = None
        self.internal_cv_audit_ = {}
        self.effective_cv_folds_ = 0
        self.cv_folds_reason_ = None
        if len(learners) == 1:
            self.fit_mode_ = "single_learner"
            self.weights_ = {learners[0]: 1.0}
        else:
            self._stack(design, target, learners)
        for name, weight in self.weights_.items():
            if weight > 0:
                self.models_[name] = self._fit_one(name, design, target)

    def _combined(self, design: np.ndarray) -> np.ndarray:
        parts = [weight * self._predict_one(self.models_[name], design)
                 for name, weight in self.weights_.items() if weight > 0]
        combined = parts[0] if len(parts) == 1 else np.sum(parts, axis=0)
        return _clip(combined) if self._binary and len(parts) > 1 else combined

    # -- audit ---------------------------------------------------------
    def fit_audit(self) -> dict[str, Any]:
        """Return JSON-safe details of the fitted stack."""

        check_is_fitted(self, ("fit_mode_", "weights_"))
        elastic_flags = [
            fold["iteration_limit_reached"]
            for fold in self.internal_cv_audit_.get("elastic_net", [])
        ]
        base = {name: self._learner_audit(name, model) for name, model in self.models_.items()}
        if "elastic_net" in base:
            elastic_flags.append(base["elastic_net"]["optimization"]["iteration_limit_reached"])
        return {
            "estimator": f"oci.models.super_learner_nuisance.{type(self).__name__}",
            "fit_mode": str(self.fit_mode_),
            "n_features": int(self.n_features_in_),
            "rows": int(self.n_rows_),
            "learners": list(self.learners_),
            "requested_cv_folds": int(self.cv_folds),
            "effective_cv_folds": int(self.effective_cv_folds_),
            "cv_folds_reason": self.cv_folds_reason_,
            "loss": "log_loss" if self._binary else "mean_squared_error",
            "stack_weights": dict(self.weights_),
            "learner_cv_loss": dict(self.learner_cv_loss_),
            "stack_cv_loss": self.stack_cv_loss_,
            "stack_cv_loss_method": (
                "weights_refitted_on_the_other_internal_folds"
                if self.stack_cv_loss_ is not None else None
            ),
            "weight_optimizer": self.weight_optimizer_,
            "refitted_learners": sorted(self.models_, key=LEARNERS.index),
            "base_learners": base,
            "internal_cv": dict(self.internal_cv_audit_),
            "elastic_net_iteration_limit_reached": (
                bool(any(elastic_flags)) if "elastic_net" in self.learners_ else None
            ),
            "seeds": {
                name: _seed(self.random_state, name)
                for name in (*self.learners_, "internal_cv")
            },
            "settings": {
                "gradient_boosting": dict(GRADIENT_BOOSTING_SETTINGS),
                "early_stopping_min_rows": EARLY_STOPPING_MIN_ROWS,
                "early_stopping_min_class_rows": EARLY_STOPPING_MIN_CLASS_ROWS,
                "random_forest": dict(RANDOM_FOREST_SETTINGS),
                "n_jobs": int(self.n_jobs),
            },
        }


class SuperLearnerClassifier(ClassifierMixin, _SuperLearnerBase):
    """Stacked binary nuisance model; ``predict_proba`` returns combined probabilities."""

    _binary = True

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SuperLearnerClassifier":
        design, target = check_X_y(X, y, dtype=float, ensure_min_features=0)
        target = np.asarray(target, dtype=float).reshape(-1)
        values = np.unique(target)
        if not set(values.tolist()).issubset({0.0, 1.0}):
            raise ValueError("super-learner classifier target must contain only 0 and 1")
        if len(values) < 2 or design.shape[1] == 0:
            raise ValueError("super-learner classifier needs both classes and at least one feature")
        self.n_features_in_ = int(design.shape[1])
        self.n_rows_ = int(len(target))
        self.classes_ = np.asarray([0, 1], dtype=int)
        self._fit_super_learner(design, target)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ("models_", "weights_"))
        probability = self._combined(check_array(X, dtype=float, ensure_min_features=0))
        return np.column_stack([1.0 - probability, probability])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class SuperLearnerRegressor(RegressorMixin, _SuperLearnerBase):
    """Stacked continuous nuisance model."""

    _binary = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SuperLearnerRegressor":
        design, target = check_X_y(X, y, dtype=float, ensure_min_features=0, y_numeric=True)
        if design.shape[1] == 0:
            raise ValueError("super-learner regressor needs at least one feature")
        self.n_features_in_ = int(design.shape[1])
        self.n_rows_ = int(len(target))
        self._fit_super_learner(design, np.asarray(target, dtype=float).reshape(-1))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ("models_", "weights_"))
        return np.asarray(self._combined(check_array(X, dtype=float, ensure_min_features=0)), float)


__all__ = [
    "DEFAULT_CV_FOLDS",
    "GRADIENT_BOOSTING_SETTINGS",
    "LEARNERS",
    "RANDOM_FOREST_SETTINGS",
    "SuperLearnerClassifier",
    "SuperLearnerConfig",
    "SuperLearnerRegressor",
    "learner_spec",
    "super_learner_config_from_mapping",
]
