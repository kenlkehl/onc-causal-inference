"""Pure numerical components for the downstream effect experiment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import rankdata


EPS = 1e-6


def clip_probability(value: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(value, dtype=float), EPS, 1.0 - EPS)


def oracle_nuisances(e: np.ndarray, mu0: np.ndarray, mu1: np.ndarray) -> dict[str, np.ndarray]:
    e = clip_probability(e)
    mu0 = clip_probability(mu0)
    mu1 = clip_probability(mu1)
    v0, v1 = mu0 * (1.0 - mu0), mu1 * (1.0 - mu1)
    a = e * v1 / (e * v1 + (1.0 - e) * v0)
    return {
        "e": e,
        "m": (1.0 - e) * mu0 + e * mu1,
        "a": a,
        "nu": a * logit(mu1) + (1.0 - a) * logit(mu0),
    }


@dataclass(frozen=True)
class Standardizer:
    center: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Standardizer":
        x = np.asarray(x, dtype=float)
        center = x.mean(axis=0) if x.shape[1] else np.empty(0)
        scale = x.std(axis=0, ddof=1) if x.shape[1] else np.empty(0)
        scale[scale < 1e-12] = 1.0
        return cls(center=center, scale=scale)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=float) - self.center) / self.scale


@dataclass(frozen=True)
class DinaFit:
    coefficients: np.ndarray
    standardizer: Standardizer
    converged: bool
    gradient_max: float

    def predict_log_or(self, x: np.ndarray) -> np.ndarray:
        z = self.standardizer.transform(x)
        return self.coefficients[0] + z @ self.coefficients[1:]


def fit_dina(
    x: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    a: np.ndarray,
    nu: np.ndarray,
) -> DinaFit:
    """Fit Gao-Hastie's counterfactual log-OR model on a fixed support."""

    standardizer = Standardizer.fit(x)
    z = standardizer.transform(x)
    design = np.column_stack([np.ones(len(z)), z])
    design *= (np.asarray(treatment, dtype=float) - a)[:, None]
    outcome = np.asarray(outcome, dtype=float)

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = nu + design @ theta
        probability = expit(eta)
        loss = float(np.logaddexp(0.0, eta).sum() - outcome @ eta)
        gradient = design.T @ (probability - outcome)
        return loss, gradient

    result = minimize(
        fun=lambda theta: objective(theta)[0],
        x0=np.zeros(design.shape[1]),
        jac=lambda theta: objective(theta)[1],
        method="L-BFGS-B",
        options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 20_000},
    )
    gradient_max = float(np.max(np.abs(objective(result.x)[1])))
    converged = bool(result.success or gradient_max <= 1e-5)
    if not converged:
        raise RuntimeError(f"DINA optimization failed: {result.message}; gradient={gradient_max:g}")
    return DinaFit(result.x, standardizer, converged, gradient_max)


def counterfactuals_from_rd(
    tau_rd: np.ndarray, e: np.ndarray, m: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mu0 = clip_probability(m - e * tau_rd)
    mu1 = clip_probability(m + (1.0 - e) * tau_rd)
    return mu0, mu1


def counterfactuals_from_log_or(
    tau_log_or: np.ndarray, a: np.ndarray, nu: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    return expit(nu - a * tau_log_or), expit(nu + (1.0 - a) * tau_log_or)


def prediction_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    residual = truth - prediction
    denominator = float(np.sum((truth - truth.mean()) ** 2))
    constant_prediction = float(np.std(prediction)) < 1e-14
    return {
        "pearson": 0.0 if constant_prediction else float(np.corrcoef(truth, prediction)[0, 1]),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "r2": float(1.0 - np.sum(residual**2) / denominator),
        "spearman": 0.0
        if constant_prediction
        else float(np.corrcoef(rankdata(truth), rankdata(prediction))[0, 1]),
    }
