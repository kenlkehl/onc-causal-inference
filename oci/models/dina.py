"""Gao--Hastie Bernoulli DINA objectives shared by discovery and estimation.

The learned effect is a conditional log odds ratio. Probability differences
must be obtained through ``counterfactuals`` before evaluating an R-loss.
Callers own patient splitting, nuisance cross-fitting, and feature encoding.
"""

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logit

SCHEMA_VERSION = "bernoulli_dina_v1"
EPS = 1e-6


def nuisances(e, mu0, mu1):
    arrays = [np.asarray(v, dtype=float).reshape(-1) for v in (e, mu0, mu1)]
    if len({v.shape for v in arrays}) != 1 or any(not np.isfinite(v).all() for v in arrays):
        raise ValueError("DINA nuisance probabilities must be finite and row aligned")
    if any(np.any((v < 0) | (v > 1)) for v in arrays):
        raise ValueError("DINA nuisance probabilities must be in [0, 1]")
    e, mu0, mu1 = [np.clip(v, EPS, 1 - EPS) for v in arrays]
    v0, v1 = mu0 * (1 - mu0), mu1 * (1 - mu1)
    a = e * v1 / (e * v1 + (1 - e) * v0)
    return {"e": e, "mu0": mu0, "mu1": mu1, "a": a, "nu": a * logit(mu1) + (1 - a) * logit(mu0)}


def loss(y, t, a, nu, delta):
    eta = np.asarray(nu) + (np.asarray(t) - np.asarray(a)) * np.asarray(delta)
    return np.logaddexp(0.0, eta) - np.asarray(y) * eta


def counterfactuals(delta, a, nu):
    delta, a, nu = np.asarray(delta), np.asarray(a), np.asarray(nu)
    return expit(nu - a * delta), expit(nu + (1 - a) * delta)


def constant_effect(y, t, a, nu):
    result = minimize_scalar(
        lambda d: float(loss(y, t, a, nu, d).mean()), bounds=(-40.0, 40.0), method="bounded"
    )
    if not result.success:
        raise ValueError("constant DINA fit did not converge")
    return float(result.x)


def score(y, t, a, nu, *, delta=None):
    """Score and Fisher centering weights about a training-fitted constant.

    Held-out callers MUST supply the constant fitted on the training partition.
    """
    if delta is None:
        delta = constant_effect(y, t, a, nu)
    u = np.asarray(t) - np.asarray(a)
    p = expit(np.asarray(nu) + u * delta)
    return u * (np.asarray(y) - p), u**2 * p * (1 - p), float(delta)


@dataclass
class DinaModel:
    coefficients: np.ndarray
    intercept: float
    iterations: int
    converged: bool
    group_ids: tuple
    regularization: float

    @property
    def coef_(self):
        return self.coefficients

    def predict(self, x):
        return np.asarray(x @ self.coefficients).reshape(-1) + self.intercept

    def selected_groups(self, tolerance=1e-7):
        return sorted({g for g, b in zip(self.group_ids, self.coefficients) if abs(b) > tolerance})


def fit(
    x,
    y,
    t,
    a,
    nu,
    *,
    groups=None,
    regularization=0.0,
    group_ratio=0.0,
    max_iter=5000,
    tolerance=1e-7,
):
    """Convex Bernoulli likelihood plus group lasso/ridge; constant unpenalized.

    Each clinical feature must use one group ID across all encoded levels and
    missingness columns. Sparse text columns may each form a singleton group.
    """
    x = sparse.csr_matrix(x, dtype=float) if sparse.issparse(x) else np.asarray(x, float)
    y, t, a, nu = [np.asarray(v, float).reshape(-1) for v in (y, t, a, nu)]
    n, p = x.shape
    if not np.isfinite(x.data if sparse.issparse(x) else x).all():
        raise ValueError("DINA design must be finite")
    if not n or any(len(v) != n or not np.isfinite(v).all() for v in (y, t, a, nu)):
        raise ValueError("DINA fitting arrays must be finite and row aligned")
    if not np.isin(y, [0, 1]).all() or not np.isin(t, [0, 1]).all():
        raise ValueError("Bernoulli DINA requires binary outcome and treatment")
    if regularization < 0 or not 0 <= group_ratio <= 1:
        raise ValueError("invalid DINA regularization")
    group_ids = tuple(map(str, range(p))) if groups is None else tuple(map(str, groups))
    if len(group_ids) != p:
        raise ValueError("one DINA group ID is required per encoded column")
    group_columns = {}
    for column, group in enumerate(group_ids):
        group_columns.setdefault(group, []).append(column)
    indices = [np.asarray(columns, dtype=int) for columns in group_columns.values()]
    u = t - a
    design = (
        sparse.hstack([u[:, None], x.multiply(u[:, None])], format="csr")
        if sparse.issparse(x)
        else np.column_stack([u, x * u[:, None]])
    )
    theta = np.zeros(p + 1)
    theta[0] = constant_effect(y, t, a, nu)

    def objective(b):
        eta = nu + np.asarray(design @ b).reshape(-1)
        ridge = regularization * (1 - group_ratio)
        grad = np.asarray(design.T @ (expit(eta) - y)).reshape(-1) / n
        grad[1:] += ridge * b[1:]
        return float(np.mean(np.logaddexp(0, eta) - y * eta) + 0.5 * ridge * (b[1:] @ b[1:])), grad

    if group_ratio == 0 or regularization == 0:
        result = minimize(
            objective,
            theta,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": max_iter, "ftol": 1e-12, "gtol": tolerance},
        )
        if not result.success and np.max(np.abs(objective(result.x)[1])) > 10 * tolerance:
            raise ValueError(f"DINA optimizer failed: {result.message}")
        return DinaModel(
            result.x[1:], float(result.x[0]), result.nit, True, group_ids, regularization
        )

    # Backtracking proximal gradient avoids a dense Gram matrix for text features.
    z, momentum, step = theta.copy(), 1.0, 1.0
    converged = False
    for iteration in range(1, max_iter + 1):
        value, grad = objective(z)
        for _ in range(60):
            candidate = z - step * grad
            for ix in indices:
                j = ix + 1
                norm = np.linalg.norm(candidate[j])
                threshold = step * regularization * group_ratio * np.sqrt(len(ix))
                candidate[j] *= max(0.0, 1 - threshold / max(norm, 1e-30))
            diff = candidate - z
            if objective(candidate)[0] <= value + grad @ diff + (diff @ diff) / (2 * step) + 1e-12:
                break
            step *= 0.5
        else:
            raise ValueError("DINA line search failed")
        if np.linalg.norm(candidate - theta) <= tolerance * (1 + np.linalg.norm(theta)):
            theta, converged = candidate, True
            break
        next_momentum = (1 + np.sqrt(1 + 4 * momentum**2)) / 2
        extrapolated = candidate + (momentum - 1) / next_momentum * (candidate - theta)
        if (z - candidate) @ (candidate - theta) > 0:
            extrapolated, next_momentum = candidate.copy(), 1.0
        theta, z, momentum = candidate, extrapolated, next_momentum
    if not converged:
        raise ValueError("group DINA optimizer did not converge")
    return DinaModel(theta[1:], float(theta[0]), iteration, True, group_ids, regularization)


def torch_loss(y, t, a, nu, delta, *, reduction="mean"):
    """Differentiable Bernoulli objective for staged and joint neural heads."""
    import torch.nn.functional as functional

    return functional.binary_cross_entropy_with_logits(nu + (t - a) * delta, y, reduction=reduction)
