# cdt/models/causal_forest_head.py
"""Causal Forest head for ITE estimation from neural network features."""

import logging
from typing import Optional, Dict, Any
import numpy as np

try:
    from econml.dml import CausalForestDML
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    ECONML_AVAILABLE = True
except ImportError:
    ECONML_AVAILABLE = False
    CausalForestDML = None

logger = logging.getLogger(__name__)


class CausalForestHead:
    """
    Causal Forest head for ITE estimation from neural features.

    Unlike neural causal heads (DragonNet, RLearner), this uses
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
        random_state: int = 42
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

        Note: Nuisance functions (propensity, outcome) are estimated using sklearn
        random forests on the neural network's learned features.
        """
        if not ECONML_AVAILABLE:
            raise ImportError(
                "econml is required for CausalForestHead. "
                "Install with: pip install econml"
            )

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.honest = honest
        self.inference = inference
        self.random_state = random_state

        # The CausalForestDML model (created during fit)
        self.model = None
        self._fitted = False

    def fit(
        self,
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
        propensity: Optional[np.ndarray] = None,
        outcome_pred: Optional[np.ndarray] = None
    ) -> 'CausalForestHead':
        """
        Fit causal forest on extracted features.

        Args:
            X: Feature matrix from neural network, shape (n_samples, n_features)
            T: Binary treatment indicator, shape (n_samples,)
            Y: Binary outcome indicator, shape (n_samples,)
            propensity: Optional propensity scores from neural network P(T=1|X)
            outcome_pred: Optional outcome predictions from neural network E[Y|X]

        Returns:
            self
        """
        logger.info(f"Fitting CausalForestDML on {X.shape[0]} samples with {X.shape[1]} features")

        # Ensure arrays are the right shape
        T = np.asarray(T).flatten()
        Y = np.asarray(Y).flatten()

        # Create nuisance models using sklearn (reliable with cross-fitting)
        # Note: CausalForestDML uses cross-fitting internally, so we can't easily
        # use pre-computed neural network predictions. Instead, we use sklearn
        # models that work well with the cross-fitting procedure.
        # The neural network's contribution is the learned feature representation X.
        model_t = RandomForestClassifier(
            n_estimators=max(50, self.n_estimators // 2),
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=-1
        )
        logger.info("Using random forest for propensity estimation (on neural features)")

        model_y = RandomForestRegressor(
            n_estimators=max(50, self.n_estimators // 2),
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=-1
        )
        logger.info("Using random forest for outcome estimation (on neural features)")

        # Create CausalForestDML with discrete_treatment=True for binary treatment
        self.model = CausalForestDML(
            model_t=model_t,
            model_y=model_y,
            discrete_treatment=True,  # Binary treatment indicator
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            honest=self.honest,
            inference=self.inference,
            random_state=self.random_state,
            n_jobs=-1
        )

        # Fit the model
        # CausalForestDML expects T as 1D and Y as 1D
        self.model.fit(Y=Y, T=T, X=X)
        self._fitted = True

        logger.info("CausalForestDML fitting complete")
        return self

    def predict(
        self,
        X: np.ndarray,
        return_ci: bool = True,
        alpha: float = 0.05
    ) -> Dict[str, np.ndarray]:
        """
        Predict ITE with optional confidence intervals.

        Args:
            X: Feature matrix, shape (n_samples, n_features)
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
        tau_pred = self.model.effect(X).flatten()

        result = {
            'tau_pred': tau_pred
        }

        # Confidence intervals (if available)
        if return_ci and self.inference:
            try:
                # Get inference object
                inference_result = self.model.effect_inference(X)
                ci = inference_result.conf_int(alpha=alpha)
                result['tau_lower'] = ci[0].flatten()
                result['tau_upper'] = ci[1].flatten()
                result['tau_std'] = inference_result.std_point.flatten() if hasattr(inference_result, 'std_point') else None
            except Exception as e:
                logger.warning(f"Could not compute confidence intervals: {e}")

        return result

    def effect_summary(
        self,
        X: np.ndarray,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Get summary statistics of treatment effects.

        Args:
            X: Feature matrix
            alpha: Significance level for CIs

        Returns:
            Dictionary with summary statistics
        """
        preds = self.predict(X, return_ci=True, alpha=alpha)
        tau = preds['tau_pred']

        summary = {
            'ate': np.mean(tau),
            'ate_std': np.std(tau),
            'tau_min': np.min(tau),
            'tau_max': np.max(tau),
            'tau_median': np.median(tau),
            'n_samples': len(tau),
            'n_positive_effect': np.sum(tau > 0),
            'n_negative_effect': np.sum(tau < 0),
        }

        if 'tau_lower' in preds and preds['tau_lower'] is not None:
            # Proportion of significant effects (CI doesn't include 0)
            significant = (preds['tau_lower'] > 0) | (preds['tau_upper'] < 0)
            summary['n_significant'] = np.sum(significant)
            summary['pct_significant'] = np.mean(significant) * 100

        return summary

    def get_state(self) -> Dict[str, Any]:
        """Get serializable state for checkpointing."""
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features,
            'honest': self.honest,
            'inference': self.inference,
            'random_state': self.random_state,
            'fitted': self._fitted
        }


class _FixedPropensityModel:
    """
    Dummy model that returns pre-computed propensity scores.

    Used when we want to use neural network's propensity predictions
    as the nuisance function in CausalForestDML.
    """

    def __init__(self, propensity_scores: np.ndarray):
        self.propensity_scores = propensity_scores.flatten()
        self._idx = 0

    def fit(self, X, y, **kwargs):
        return self

    def predict_proba(self, X):
        """Return propensity as probability."""
        n = X.shape[0]
        # Return as 2-column array [P(T=0), P(T=1)]
        p = self.propensity_scores[self._idx:self._idx + n]
        self._idx += n
        return np.column_stack([1 - p, p])

    def predict(self, X):
        """Return binary predictions."""
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)


class _FixedOutcomeModel:
    """
    Dummy model that returns pre-computed outcome predictions.

    Used when we want to use neural network's outcome predictions
    as the nuisance function in CausalForestDML.
    """

    def __init__(self, outcome_preds: np.ndarray):
        self.outcome_preds = outcome_preds.flatten()
        self._idx = 0

    def fit(self, X, y, **kwargs):
        return self

    def predict(self, X):
        """Return outcome predictions."""
        n = X.shape[0]
        preds = self.outcome_preds[self._idx:self._idx + n]
        self._idx += n
        return preds
