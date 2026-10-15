# oci/models/causal_forest_head.py
"""Causal Forest head for ITE estimation from neural network features."""

import logging
import math
from typing import Optional, Dict, Any
import numpy as np
from scipy.stats import norm

from .elastic_net_nuisance import ElasticNetLogisticClassifier, ElasticNetRegressor

try:
    from econml.dml import CausalForestDML

    ECONML_AVAILABLE = True
except ImportError:
    ECONML_AVAILABLE = False
    CausalForestDML = None

logger = logging.getLogger(__name__)

# Explicit copy of EconML 0.16's ``CausalForestDML.tune(params="auto")`` grid.
# Passing it explicitly makes the searched parameters part of the audit instead
# of an implicit library default.
CAUSAL_FOREST_TUNING_GRID: Dict[str, list] = {
    "min_weight_fraction_leaf": [0.0001, 0.01],
    "max_depth": [3, 5, None],
    "min_var_fraction_leaf": [0.001, 0.01],
}

# EconML's doubly robust ATE floors e(1-e) at 0.01; the held-out scores use the
# same floor so training-sample and held-out DR estimates are comparable.
DR_PROPENSITY_VARIANCE_FLOOR = 0.01


def summarize_doubly_robust_scores(scores: Any, alpha: float = 0.05) -> Dict[str, Any]:
    """Mean, standard error and normal CI of per-row doubly robust (AIPW) scores."""

    values = np.asarray(scores, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n == 0:
        return {
            "ate": None,
            "ate_stderr": None,
            "ate_ci_lower": None,
            "ate_ci_upper": None,
            "n": 0,
        }
    ate = float(np.mean(values))
    stderr = float(np.std(values, ddof=1) / math.sqrt(n)) if n > 1 else None
    z = float(norm.ppf(1.0 - alpha / 2.0))
    return {
        "ate": ate,
        "ate_stderr": stderr,
        "ate_ci_lower": None if stderr is None else ate - z * stderr,
        "ate_ci_upper": None if stderr is None else ate + z * stderr,
        "n": n,
    }


def _normalize_outcome_type(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("outcome_type must be a string")
    normalized = value.strip().lower()
    if normalized not in {"binary", "continuous"}:
        raise ValueError("outcome_type must be exactly 'binary' or 'continuous'")
    return normalized


def _validate_max_features(value: Any, *, name: str, allow_none: bool) -> None:
    if value is None:
        if allow_none:
            return
        raise ValueError(f"{name} cannot be None")
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} cannot be boolean")
    if isinstance(value, str):
        if not value.strip():
            raise ValueError(f"{name} cannot be empty")
        return
    if isinstance(value, (int, np.integer)):
        if int(value) < 1:
            raise ValueError(f"integer {name} must be positive")
        return
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if not np.isfinite(numeric) or not 0 < numeric <= 1:
            raise ValueError(f"fractional {name} must be finite and in (0, 1]")
        return
    raise TypeError(f"{name} must be a supported scalar")


def tune_causal_forest_model(
    model: Any,
    Y: np.ndarray,
    T: np.ndarray,
    X: Optional[np.ndarray],
    W: Optional[np.ndarray] = None,
    params: Any = None,
) -> bool:
    """Tune an EconML CausalForestDML model, falling back to configured parameters."""
    if params is None:
        params = {name: list(values) for name, values in CAUSAL_FOREST_TUNING_GRID.items()}
    try:
        logger.info("Tuning CausalForestDML hyperparameters with params=%r", params)
        model.tune(Y=Y, T=T, X=X, W=W, params=params)
        logger.info("CausalForestDML hyperparameter tuning complete")
        return True
    except Exception as exc:
        logger.warning(
            "CausalForestDML hyperparameter tuning failed; fitting with configured "
            "hyperparameters. Error: %s",
            exc,
        )
        return False


class CausalForestHead:
    """
    Causal Forest head for ITE estimation from neural features.

    Unlike end-to-end neural causal heads, this uses
    econml's CausalForestDML to estimate treatment effects.

    The causal forest provides:
    - Doubly-robust estimation (robust to misspecification of either propensity or outcome)
    - Honest trees for unbiased effect estimates
    - Built-in confidence intervals
    - Direct estimation of τ(X) = E[Y(1) - Y(0) | X]

    References:
        Athey, Tibshirani, Wager (2019). Generalized Random Forests. Annals of Statistics.
        Chernozhukov et al. (2018). Double/Debiased Machine Learning. Econometrica.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 5,
        max_features: str = "sqrt",
        honest: bool = True,
        inference: bool = True,
        random_state: int = 42,
        tune_model: bool = False,
        subforest_size: int = 4,
        nuisance_l1_ratio: float = 0.8,
        nuisance_cv_folds: int = 3,
        nuisance_regularization_grid_size: int = 16,
        nuisance_minimum_log10_c: float = -2.0,
        nuisance_maximum_log10_c: float = 4.0,
        nuisance_minimum_log10_alpha: float = -5.0,
        nuisance_maximum_log10_alpha: float = -1.0,
        nuisance_max_iter: int = 5_000,
        nuisance_tolerance: float = 1e-4,
        n_jobs: int = -1,
        outcome_type: Optional[str] = None,
    ):
        """
        Initialize Causal Forest head.

        Args:
            n_estimators: Number of trees in the forest (must be divisible by 4)
            max_depth: Maximum depth of trees (None = unlimited)
            min_samples_leaf: Minimum samples per leaf
            max_features: Feature subset strategy for splitting
            honest: Use honest estimation (sample splitting within trees)
            inference: Enable inference for confidence intervals
            random_state: Random seed for reproducibility
            tune_model: Run EconML's tuning step over ``CAUSAL_FOREST_TUNING_GRID``
                before fitting. Disabled by default because tuning silently
                replaces the configured forest settings; when enabled, every
                tuned parameter and its effective value is recorded in
                ``fit_audit()``.
            subforest_size: Number of trees in each inference subforest.
            nuisance_l1_ratio: Elastic-net L1 share for both nuisance tasks.
            nuisance_cv_folds: Internal folds used to select regularization.
            nuisance_regularization_grid_size: Number of regularization values.
            n_jobs: Operational CPU parallelism for the causal forest and
                nuisance estimators. It is recorded in the fit audit.
            outcome_type: ``"binary"`` selects logistic elastic net and
                EconML's discrete-outcome probability contract;
                ``"continuous"`` selects squared-error elastic net. ``None``
                preserves the historical convenience-path default of binary.

        Note: All nuisance functions are cross-validated elastic nets.
        The heterogeneous-effect model remains an honest causal forest.
        """
        if not ECONML_AVAILABLE:
            raise ImportError(
                "econml is required for CausalForestHead. " "Install with: pip install econml"
            )

        if isinstance(n_estimators, bool) or not isinstance(n_estimators, int):
            raise TypeError("n_estimators must be an integer")
        if n_estimators < 1:
            raise ValueError("n_estimators must be positive")
        if isinstance(subforest_size, bool) or not isinstance(subforest_size, int):
            raise TypeError("subforest_size must be an integer")
        if subforest_size < 1:
            raise ValueError("subforest_size must be positive")
        if inference and n_estimators % subforest_size:
            raise ValueError(
                "n_estimators must be divisible by subforest_size when inference is enabled"
            )
        if max_depth is not None and (
            isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 1
        ):
            raise ValueError("max_depth must be None or a positive integer")
        if (
            isinstance(min_samples_leaf, bool)
            or not isinstance(min_samples_leaf, int)
            or min_samples_leaf < 1
        ):
            raise ValueError("min_samples_leaf must be a positive integer")
        for name, value in (
            ("honest", honest),
            ("inference", inference),
            ("tune_model", tune_model),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be boolean")
        if isinstance(random_state, bool) or not isinstance(random_state, int):
            raise TypeError("random_state must be an integer")
        for name, value, minimum in (
            ("nuisance_cv_folds", nuisance_cv_folds, 2),
            (
                "nuisance_regularization_grid_size",
                nuisance_regularization_grid_size,
                3,
            ),
            ("nuisance_max_iter", nuisance_max_iter, 1),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or int(value) < minimum
            ):
                raise ValueError(f"{name} must be an integer of at least {minimum}")
        for name, value in (
            ("nuisance_l1_ratio", nuisance_l1_ratio),
            ("nuisance_minimum_log10_c", nuisance_minimum_log10_c),
            ("nuisance_maximum_log10_c", nuisance_maximum_log10_c),
            ("nuisance_minimum_log10_alpha", nuisance_minimum_log10_alpha),
            ("nuisance_maximum_log10_alpha", nuisance_maximum_log10_alpha),
            ("nuisance_tolerance", nuisance_tolerance),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(float(value))
            ):
                raise ValueError(f"{name} must be finite")
        if not 0.0 < float(nuisance_l1_ratio) <= 1.0:
            raise ValueError("nuisance_l1_ratio must be in (0, 1]")
        if float(nuisance_tolerance) <= 0.0:
            raise ValueError("nuisance_tolerance must be positive")
        if float(nuisance_minimum_log10_c) >= float(nuisance_maximum_log10_c):
            raise ValueError(
                "nuisance_minimum_log10_c must be smaller than "
                "nuisance_maximum_log10_c"
            )
        if float(nuisance_minimum_log10_alpha) >= float(
            nuisance_maximum_log10_alpha
        ):
            raise ValueError(
                "nuisance_minimum_log10_alpha must be smaller than "
                "nuisance_maximum_log10_alpha"
            )
        if isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or n_jobs == 0:
            raise ValueError("n_jobs must be a nonzero integer")
        _validate_max_features(
            max_features,
            name="max_features",
            allow_none=False,
        )
        normalized_outcome_type = _normalize_outcome_type(
            "binary" if outcome_type is None else outcome_type
        )

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        # EconML defaults, fixed explicitly because tuning may change them.
        self.min_weight_fraction_leaf = 0.0
        self.min_var_fraction_leaf = None
        self.honest = honest
        self.inference = inference
        self.random_state = random_state
        self.tune_model = bool(tune_model)
        self.subforest_size = subforest_size
        self.nuisance_l1_ratio = float(nuisance_l1_ratio)
        self.nuisance_cv_folds = int(nuisance_cv_folds)
        self.nuisance_regularization_grid_size = int(
            nuisance_regularization_grid_size
        )
        self.nuisance_minimum_log10_c = float(nuisance_minimum_log10_c)
        self.nuisance_maximum_log10_c = float(nuisance_maximum_log10_c)
        self.nuisance_minimum_log10_alpha = float(nuisance_minimum_log10_alpha)
        self.nuisance_maximum_log10_alpha = float(nuisance_maximum_log10_alpha)
        self.nuisance_max_iter = int(nuisance_max_iter)
        self.nuisance_tolerance = float(nuisance_tolerance)
        self.n_jobs = n_jobs
        self.outcome_type = normalized_outcome_type
        self.discrete_outcome = normalized_outcome_type == "binary"
        self.runtime_mode = "elastic_net_nuisance_v1"

        # The CausalForestDML model (created during fit)
        self.model = None
        self._fitted = False
        self.tuning_attempted_ = False
        self.tuning_succeeded_ = None
        self.effective_forest_parameters_ = None
        self.effective_nuisance_parameters_ = None

    def _configured_forest_parameters(self) -> Dict[str, Any]:
        return {
            "n_estimators": int(self.n_estimators),
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "min_weight_fraction_leaf": self.min_weight_fraction_leaf,
            "min_var_fraction_leaf": self.min_var_fraction_leaf,
            "honest": bool(self.honest),
            "inference": bool(self.inference),
            "subforest_size": int(self.subforest_size),
            "random_state": int(self.random_state),
            "discrete_outcome": bool(self.discrete_outcome),
        }

    def _outcome_model_contract(self) -> Dict[str, Any]:
        binary = self.outcome_type == "binary"
        return {
            "outcome_type": self.outcome_type,
            "discrete_outcome": bool(self.discrete_outcome),
            "model_class": (
                "oci.models.elastic_net_nuisance.ElasticNetLogisticClassifier"
                if binary
                else "oci.models.elastic_net_nuisance.ElasticNetRegressor"
            ),
            "prediction_interface": "predict_proba" if binary else "predict",
            "criterion": "log_loss" if binary else "squared_error",
            "penalty": "elastic_net",
        }

    def _elastic_classifier_kwargs(self) -> Dict[str, Any]:
        return {
            "l1_ratio": float(self.nuisance_l1_ratio),
            "cv_folds": int(self.nuisance_cv_folds),
            "regularization_grid_size": int(
                self.nuisance_regularization_grid_size
            ),
            "minimum_log10_c": float(self.nuisance_minimum_log10_c),
            "maximum_log10_c": float(self.nuisance_maximum_log10_c),
            "max_iter": int(self.nuisance_max_iter),
            "tolerance": float(self.nuisance_tolerance),
            "random_state": int(self.random_state),
            "n_jobs": int(self.n_jobs),
        }

    def _elastic_regressor_kwargs(self) -> Dict[str, Any]:
        return {
            "l1_ratio": float(self.nuisance_l1_ratio),
            "cv_folds": int(self.nuisance_cv_folds),
            "regularization_grid_size": int(
                self.nuisance_regularization_grid_size
            ),
            "minimum_log10_alpha": float(self.nuisance_minimum_log10_alpha),
            "maximum_log10_alpha": float(self.nuisance_maximum_log10_alpha),
            "max_iter": int(self.nuisance_max_iter),
            "tolerance": float(self.nuisance_tolerance),
            "random_state": int(self.random_state),
            "n_jobs": int(self.n_jobs),
        }

    def _configured_nuisance_parameters(self) -> Dict[str, Any]:
        return {
            "model_family": "elastic_net",
            "treatment_model": self._elastic_classifier_kwargs(),
            "outcome_model": (
                self._elastic_classifier_kwargs()
                if self.discrete_outcome
                else self._elastic_regressor_kwargs()
            ),
            "outcome_model_contract": self._outcome_model_contract(),
        }

    def _operational_parameters(self) -> Dict[str, Any]:
        return {"n_jobs": int(self.n_jobs)}

    @staticmethod
    def _estimator_parameter(estimator: Any, name: str) -> Any:
        get_params = getattr(estimator, "get_params", None)
        if callable(get_params):
            parameters = get_params(deep=False)
            if isinstance(parameters, dict) and name in parameters:
                value = parameters[name]
                return value.item() if isinstance(value, np.generic) else value
        parameters = getattr(estimator, "kwargs", None)
        if isinstance(parameters, dict) and name in parameters:
            value = parameters[name]
            return value.item() if isinstance(value, np.generic) else value
        if hasattr(estimator, name):
            value = getattr(estimator, name)
            return value.item() if isinstance(value, np.generic) else value
        raise TypeError(f"nuisance estimator does not expose configured parameter {name}")

    def _effective_nuisance_parameters(
        self,
        *,
        model_t: Any,
        model_y: Any,
    ) -> Dict[str, Any]:
        expected_outcome_class = (
            ElasticNetLogisticClassifier
            if self.discrete_outcome
            else ElasticNetRegressor
        )
        if type(model_t) is not ElasticNetLogisticClassifier:
            raise RuntimeError("treatment nuisance model is not an elastic net")
        if type(model_y) is not expected_outcome_class:
            raise RuntimeError(
                "outcome nuisance model class does not match the elastic-net outcome type"
            )
        result = {
            "model_family": "elastic_net",
            "treatment_model": model_t.get_params(deep=False),
            "outcome_model": model_y.get_params(deep=False),
            "outcome_model_contract": self._outcome_model_contract(),
        }
        if result != self._configured_nuisance_parameters():
            raise RuntimeError(
                "effective nuisance elastic-net parameters differ from configured settings"
            )
        return result

    def _effective_forest_parameters(self) -> Dict[str, Any]:
        """Return the post-tuning forest parameters that affect final fitting."""

        if self.model is None:
            raise RuntimeError("causal forest model has not been created")
        effective: Dict[str, Any] = {}
        for name in self._configured_forest_parameters():
            if not hasattr(self.model, name):
                raise RuntimeError(
                    "causal-forest runtime does not expose configured "
                    f"attribute {name}; refusing to substitute the requested value"
                )
            value = getattr(self.model, name)
            if isinstance(value, np.generic):
                value = value.item()
            if value is not None and not isinstance(value, (bool, int, float, str)):
                raise TypeError(f"effective causal-forest parameter {name} is not scalar")
            effective[name] = value
        return effective

    def _create_model(self):
        """Create an unfitted CausalForestDML with elastic-net nuisances."""

        model_t = ElasticNetLogisticClassifier(**self._elastic_classifier_kwargs())
        outcome_model_class = (
            ElasticNetLogisticClassifier
            if self.discrete_outcome
            else ElasticNetRegressor
        )
        model_y = outcome_model_class(
            **(
                self._elastic_classifier_kwargs()
                if self.discrete_outcome
                else self._elastic_regressor_kwargs()
            )
        )
        logger.info(
            "Using cross-validated elastic nets for treatment and %s outcome "
            "nuisance estimation",
            self.outcome_type,
        )

        model = CausalForestDML(
            model_t=model_t,
            model_y=model_y,
            discrete_outcome=self.discrete_outcome,
            discrete_treatment=True,  # Binary treatment indicator
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            min_var_fraction_leaf=self.min_var_fraction_leaf,
            honest=self.honest,
            inference=self.inference,
            subforest_size=self.subforest_size,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.effective_nuisance_parameters_ = self._effective_nuisance_parameters(
            model_t=model_t,
            model_y=model_y,
        )
        if self._estimator_parameter(model, "n_jobs") != int(self.n_jobs):
            raise RuntimeError(
                "effective causal-forest n_jobs differs from the operational setting"
            )
        return model

    def fit(
        self,
        X: Optional[np.ndarray],
        T: np.ndarray,
        Y: np.ndarray,
        W: Optional[np.ndarray] = None,
        propensity: Optional[np.ndarray] = None,
        outcome_pred: Optional[np.ndarray] = None,
    ) -> "CausalForestHead":
        """
        Fit causal forest on extracted features.

        Args:
            X: Effect-modifier feature matrix, shape (n_samples, n_features), or None
            W: Optional control/confounder feature matrix passed to EconML W
            T: Binary treatment indicator, shape (n_samples,)
            Y: Binary or continuous outcome, as declared by ``outcome_type``.
            propensity: Optional propensity scores from neural network P(T=1|X)
            outcome_pred: Optional outcome predictions from neural network E[Y|X]

        Returns:
            self
        """
        n_samples = X.shape[0] if X is not None else len(Y)
        x_dim = X.shape[1] if X is not None else 0
        w_msg = f", W={W.shape[1]} controls" if W is not None else ""
        logger.info(
            f"Fitting CausalForestDML on {n_samples} samples with X={x_dim} features{w_msg}"
        )

        # Ensure arrays are the right shape
        T = np.asarray(T).flatten()
        Y = np.asarray(Y).flatten()
        if len(T) != len(Y):
            raise ValueError("treatment and outcome must have the same number of rows")
        if not np.isfinite(T).all() or not np.isfinite(Y).all():
            raise ValueError("treatment and outcome must be finite")
        if set(np.unique(T).tolist()) != {0, 1}:
            raise ValueError("causal forest treatment must contain exactly 0 and 1")
        if self.discrete_outcome and set(np.unique(Y).tolist()) != {0, 1}:
            raise ValueError(
                "binary causal forest outcome must contain exactly both 0 and 1"
            )
        self.model = self._create_model()
        self.tuning_attempted_ = bool(self.tune_model)
        self.tuning_succeeded_ = None
        if self.tune_model:
            self.tuning_succeeded_ = tune_causal_forest_model(
                self.model,
                Y=Y,
                T=T,
                X=X,
                W=W,
            )
            if not self.tuning_succeeded_:
                logger.info("Rebuilding CausalForestDML after failed tuning attempt")
                self.model = self._create_model()
        else:
            logger.info("Skipping CausalForestDML tuning; using fixed configuration")

        # CausalForestDML expects T and Y as one-dimensional arrays.
        self.model.fit(Y=Y, T=T, X=X, W=W)
        self.effective_forest_parameters_ = self._effective_forest_parameters()
        self.n_fit_rows_ = int(len(Y))
        if (
            not self.tune_model
            and self.effective_forest_parameters_ != self._configured_forest_parameters()
        ):
            raise RuntimeError(
                "effective causal-forest parameters differ from the fixed "
                "scientific configuration"
            )
        self._fitted = True

        logger.info("CausalForestDML fitting complete")
        return self

    def fit_audit(self) -> Dict[str, Any]:
        """Return actual tuning status and effective post-tuning parameters."""

        if (
            not self._fitted
            or self.effective_forest_parameters_ is None
            or self.effective_nuisance_parameters_ is None
        ):
            raise RuntimeError("CausalForestHead must be fit before requesting its audit")
        return {
            "configuration_mode": self.runtime_mode,
            "outcome_model_contract": self._outcome_model_contract(),
            "configured_parameters": self._configured_forest_parameters(),
            "configured_nuisance_parameters": (self._configured_nuisance_parameters()),
            "operational_parameters": self._operational_parameters(),
            "tuning_configured": bool(self.tune_model),
            "tuning_attempted": bool(self.tuning_attempted_),
            "tuning_succeeded": self.tuning_succeeded_,
            "tuning_failure_fell_back_to_configured_parameters": bool(
                self.tuning_attempted_ and self.tuning_succeeded_ is False
            ),
            "tuning_params": (
                {name: list(values) for name, values in CAUSAL_FOREST_TUNING_GRID.items()}
                if self.tune_model
                else None
            ),
            "tuned_parameter_changes": self._tuned_parameter_changes(),
            "effective_parameters": dict(self.effective_forest_parameters_),
            "effective_nuisance_parameters": dict(self.effective_nuisance_parameters_),
            "fitted_nuisance_models": self._fitted_nuisance_model_audit(),
        }

    def _tuned_parameter_changes(self) -> Dict[str, Dict[str, Any]]:
        """Configured versus effective value of every parameter tuning changed."""

        configured = self._configured_forest_parameters()
        effective = dict(self.effective_forest_parameters_ or {})
        return {
            name: {"configured": configured[name], "effective": effective.get(name)}
            for name in configured
            if effective.get(name) != configured[name]
        }

    def _fitted_nuisance_model_audit(self) -> Dict[str, Any]:
        """Audit every EconML-fitted nuisance clone and its cross-fit position."""

        if self.model is None:
            raise RuntimeError("causal forest model has not been fit")

        def visit(value: Any, path: tuple[int, ...] = ()):
            if isinstance(value, (list, tuple)):
                for index, child in enumerate(value):
                    yield from visit(child, (*path, index))
                return
            audit_method = getattr(value, "fit_audit", None)
            if not callable(audit_method):
                raise RuntimeError(
                    "fitted nuisance estimator does not expose fit_audit; "
                    f"type={type(value).__module__}.{type(value).__name__}"
                )
            yield path, dict(audit_method())

        result: Dict[str, Any] = {}
        for role, attribute in (
            ("treatment", "models_t"),
            ("outcome", "models_y"),
        ):
            if not hasattr(self.model, attribute):
                raise RuntimeError(
                    f"fitted causal forest does not expose {attribute} for audit"
                )
            entries = []
            for path, audit in visit(getattr(self.model, attribute)):
                entries.append({"crossfit_path": list(path), **audit})
            if not entries:
                raise RuntimeError(f"fitted causal forest has no {role} nuisance models")
            result[role] = entries
        return result

    def predict(
        self, X: Optional[np.ndarray], return_ci: bool = True, alpha: float = 0.05
    ) -> Dict[str, np.ndarray]:
        """
        Predict ITE with optional confidence intervals.

        Args:
            X: Effect-modifier feature matrix, shape (n_samples, n_features), or None
            return_ci: Whether to return confidence intervals
            alpha: Significance level for confidence intervals (default 0.05 = 95% CI)

        Returns:
            Dictionary with predictions:
                - tau_pred: Point estimates of τ(X), shape (n_samples,)
                - tau_lower: Lower CI bound (if return_ci and inference enabled)
                - tau_upper: Upper CI bound (if return_ci and inference enabled)
        """
        if not self._fitted:
            raise RuntimeError("CausalForestHead must be fitted before predicting")

        # Point estimates
        tau_pred = self.model.effect(X, T0=0, T1=1).flatten()

        result = {"tau_pred": tau_pred}

        # Confidence intervals (if available)
        if return_ci and self.inference:
            try:
                # Get inference object
                inference_result = self.model.effect_inference(X)
                ci = inference_result.conf_int(alpha=alpha)
                result["tau_lower"] = ci[0].flatten()
                result["tau_upper"] = ci[1].flatten()
                # EconML 0.16 inference results expose the pointwise CATE
                # standard error as ``stderr``.
                result["tau_std"] = np.asarray(inference_result.stderr, dtype=float).flatten()
            except Exception as e:
                logger.warning(f"Could not compute confidence intervals: {e}")

        return result

    def training_doubly_robust_ate(self, alpha: float = 0.05) -> Dict[str, Any]:
        """EconML's doubly robust ATE over the forest's own fitting rows.

        CausalForestDML combines out-of-bag forest predictions with its
        cross-fitted elastic-net residuals, so this is an honest AIPW-type
        estimate for the fitting sample (not the held-out rows).
        """

        if not self._fitted:
            raise RuntimeError("CausalForestHead must be fitted before requesting the ATE")
        ate = float(np.ravel(self.model.ate_)[0])
        stderr = float(np.ravel(self.model.ate_stderr_)[0])
        z = float(norm.ppf(1.0 - alpha / 2.0))
        return {
            "ate": ate,
            "ate_stderr": stderr,
            "ate_ci_lower": ate - z * stderr,
            "ate_ci_upper": ate + z * stderr,
            "n": int(getattr(self, "n_fit_rows_", 0)),
            "estimand": "fitting_sample_ate",
            "method": "econml_causal_forest_dml_doubly_robust_ate",
        }

    def nuisance_predictions(
        self,
        X: Optional[np.ndarray],
        W: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict e(X, W) and m(X, W) for new rows from the fitted nuisance clones.

        Each EconML cross-fit clone was trained on part of the fitting rows
        only, so for rows outside the fitting sample the clone average is an
        out-of-sample nuisance prediction. The design matches EconML's
        first-stage input ``[X, W]``.
        """

        if not self._fitted:
            raise RuntimeError("CausalForestHead must be fitted before predicting nuisances")
        n_rows = X.shape[0] if X is not None else (W.shape[0] if W is not None else 0)
        if X is None:
            design = W if W is not None else np.ones((n_rows, 1))
        else:
            design = np.hstack([X, W]) if W is not None else X

        def flat(models: Any) -> list:
            if isinstance(models, (list, tuple)):
                return [leaf for child in models for leaf in flat(child)]
            return [models]

        treatment_models = flat(self.model.models_t)
        outcome_models = flat(self.model.models_y)
        if not treatment_models or not outcome_models:
            raise RuntimeError("fitted causal forest exposes no nuisance models")
        e_hat = np.mean(
            [np.asarray(model.predict_proba(design))[:, 1] for model in treatment_models],
            axis=0,
        )
        if self.discrete_outcome:
            m_hat = np.mean(
                [np.asarray(model.predict_proba(design))[:, 1] for model in outcome_models],
                axis=0,
            )
        else:
            m_hat = np.mean(
                [
                    np.asarray(model.predict(design), dtype=float).reshape(-1)
                    for model in outcome_models
                ],
                axis=0,
            )
        return e_hat, m_hat

    def doubly_robust_scores(
        self,
        X: Optional[np.ndarray],
        T: np.ndarray,
        Y: np.ndarray,
        W: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Per-row AIPW scores on rows outside the fitting sample.

        ``psi = tau(X) + (Y - m - tau (T - e)) (T - e) / max(e (1 - e), 0.01)``
        with the forest's tau and its out-of-sample elastic-net nuisances. The
        mean over held-out rows is a cross-fitted doubly robust ATE.
        """

        if not self._fitted:
            raise RuntimeError("CausalForestHead must be fitted before scoring")
        tau = self.model.effect(X, T0=0, T1=1).flatten()
        e_hat, m_hat = self.nuisance_predictions(X, W)
        t = np.asarray(T, dtype=float).reshape(-1)
        y = np.asarray(Y, dtype=float).reshape(-1)
        t_resid = t - e_hat
        variance = np.maximum(e_hat * (1.0 - e_hat), DR_PROPENSITY_VARIANCE_FLOOR)
        return tau + (y - m_hat - tau * t_resid) * t_resid / variance
