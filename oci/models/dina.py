"""Gao--Hastie Bernoulli DINA objectives shared by discovery and estimation.

The learned effect is a conditional log odds ratio. Probability differences
must be obtained through ``counterfactuals`` before evaluating an R-loss.
Callers own patient splitting, nuisance cross-fitting, and feature encoding.
"""

from dataclasses import dataclass, field

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
    diagnostics: dict = field(default_factory=dict)

    @property
    def coef_(self):
        return self.coefficients

    def predict(self, x):
        return np.asarray(x @ self.coefficients).reshape(-1) + self.intercept

    def selected_groups(self, tolerance=1e-7):
        return sorted({g for g, b in zip(self.group_ids, self.coefficients) if abs(b) > tolerance})


class DinaConvergenceError(ValueError):
    """A numerical fit failure, distinct from invalid data or configuration."""

    def __init__(self, message, diagnostics=None):
        super().__init__(message)
        self.diagnostics = dict(diagnostics or {})


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
    initial=None,
):
    """Convex Bernoulli likelihood plus group lasso/ridge; constant unpenalized.

    Each clinical feature must use one group ID across all encoded levels and
    missingness columns. Sparse text columns may each form a singleton group.
    A warm start must come from the same training rows and column encoding;
    callers must reset it between CV partitions.
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
    if not np.isfinite(regularization) or regularization < 0 or not np.isfinite(group_ratio) or not 0 <= group_ratio <= 1:
        raise ValueError("invalid DINA regularization")
    if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("DINA max_iter must be a positive integer")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("DINA tolerance must be finite and positive")
    group_ids = tuple(map(str, range(p))) if groups is None else tuple(map(str, groups))
    if len(group_ids) != p:
        raise ValueError("one DINA group ID is required per encoded column")
    group_index = {group: i for i, group in enumerate(dict.fromkeys(group_ids))}
    column_groups = np.asarray([group_index[group] for group in group_ids], dtype=int)
    weights = np.sqrt(np.bincount(column_groups, minlength=len(group_index)))
    thresholds = regularization * group_ratio * weights

    def group_norms(values):
        return np.sqrt(np.bincount(column_groups, weights=values**2, minlength=len(weights)))

    def stationarity(b, grad):
        # Minimum group-wise subgradient norm (KKT residual). Unlike a small
        # coefficient step, this cannot report convergence just because the
        # line search chose a very small step size.
        norms = group_norms(b[1:])
        active = norms > 0
        scale = np.divide(thresholds, norms, out=np.zeros_like(norms), where=active)
        residual = group_norms(grad[1:] + scale[column_groups] * b[1:])
        residual[~active] = np.maximum(0.0, residual[~active] - thresholds[~active])
        return max(abs(float(grad[0])), float(np.max(residual, initial=0.0)))
    u = t - a
    design = (
        sparse.hstack([u[:, None], x.multiply(u[:, None])], format="csr")
        if sparse.issparse(x)
        else np.column_stack([u, x * u[:, None]])
    )
    theta = np.zeros(p + 1)
    if initial is None:
        theta[0] = constant_effect(y, t, a, nu)
    else:
        if (not isinstance(initial, DinaModel) or not initial.converged
                or initial.group_ids != group_ids or initial.coefficients.shape != (p,)
                or not np.isfinite(initial.coefficients).all() or not np.isfinite(initial.intercept)):
            raise ValueError("DINA warm start requires a converged model with matching encoded groups")
        theta[0], theta[1:] = initial.intercept, initial.coefficients

    def objective(b):
        eta = nu + np.asarray(design @ b).reshape(-1)
        ridge = regularization * (1 - group_ratio)
        grad = np.asarray(design.T @ (expit(eta) - y)).reshape(-1) / n
        grad[1:] += ridge * b[1:]
        return float(np.mean(np.logaddexp(0, eta) - y * eta) + 0.5 * ridge * (b[1:] @ b[1:])), grad

    def finish(b, iterations, *, converged, reason, step=None, backtracks=0):
        value, grad = objective(b)
        value += float(thresholds @ group_norms(b[1:]))
        residual = stationarity(b, grad)
        diagnostics = {
            "converged": bool(converged), "reason": str(reason), "iterations": int(iterations),
            "max_iter": int(max_iter), "tolerance": float(tolerance),
            "regularization": float(regularization), "group_ratio": float(group_ratio),
            "rows": n, "columns": p, "groups": len(weights), "warm_started": initial is not None,
            "objective": float(value) if np.isfinite(value) else None,
            "stationarity_residual": float(residual) if np.isfinite(residual) else None,
            "step_size": float(step) if step is not None else None, "backtracks": int(backtracks),
        }
        if not converged:
            raise DinaConvergenceError(
                f"DINA optimizer did not converge ({reason}; iterations={iterations}/{max_iter}, "
                f"regularization={regularization:g}, group_ratio={group_ratio:g}, "
                f"stationarity_residual={residual:.6g}, tolerance={tolerance:g})", diagnostics)
        return DinaModel(b[1:].copy(), float(b[0]), int(iterations), True, group_ids,
                         regularization, diagnostics)

    if group_ratio == 0 or regularization == 0:
        result = minimize(
            objective,
            theta,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": max_iter, "ftol": 1e-12, "gtol": tolerance},
        )
        return finish(result.x, result.nit,
            converged=bool(result.success or np.max(np.abs(objective(result.x)[1])) <= 10 * tolerance),
            reason=result.message)

    # Backtracking proximal gradient avoids a dense Gram matrix for text features.
    z, momentum, step = theta.copy(), 1.0, 1.0
    backtracks = 0
    for iteration in range(1, max_iter + 1):
        value, grad = objective(z)
        # Let the step recover after conservative backtracking as curvature
        # changes, while retaining the same quadratic upper-bound check.
        if iteration > 1:
            step *= 1.1
        for _ in range(60):
            candidate = z - step * grad
            norms = group_norms(candidate[1:])
            scales = np.maximum(0.0, 1 - step * thresholds / np.maximum(norms, 1e-30))
            candidate[1:] *= scales[column_groups]
            diff = candidate - z
            candidate_value, candidate_grad = objective(candidate)
            if candidate_value <= value + grad @ diff + (diff @ diff) / (2 * step) + 1e-12:
                break
            step *= 0.5
            backtracks += 1
        else:
            return finish(theta, iteration, converged=False, reason="line_search_failed",
                          step=step, backtracks=backtracks)
        if stationarity(candidate, candidate_grad) <= tolerance:
            return finish(candidate, iteration, converged=True, reason="stationarity_tolerance",
                          step=step, backtracks=backtracks)
        next_momentum = (1 + np.sqrt(1 + 4 * momentum**2)) / 2
        extrapolated = candidate + (momentum - 1) / next_momentum * (candidate - theta)
        if (z - candidate) @ (candidate - theta) > 0:
            extrapolated, next_momentum = candidate.copy(), 1.0
        theta, z, momentum = candidate, extrapolated, next_momentum
    return finish(theta, iteration, converged=False, reason="iteration_limit",
                  step=step, backtracks=backtracks)


def torch_loss(y, t, a, nu, delta, *, reduction="mean"):
    """Differentiable Bernoulli objective for staged and joint neural heads."""
    import torch.nn.functional as functional

    return functional.binary_cross_entropy_with_logits(nu + (t - a) * delta, y, reduction=reduction)
