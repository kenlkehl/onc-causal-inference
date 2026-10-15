"""Small probability calibration helpers for nuisance models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss


_DISABLED = {"", "none", "off", "false", "disabled", "no"}


def clip_probability(values: Any, eps: float = 1e-4) -> np.ndarray:
    probs = np.asarray(values, dtype=float)
    probs = np.where(np.isfinite(probs), probs, 0.5)
    return np.clip(probs, eps, 1.0 - eps)


def binary_calibration_metrics(
    y_true: Any,
    prob: Any,
    *,
    prefix: str = "",
    n_bins: int = 10,
) -> Dict[str, Optional[float]]:
    y = np.asarray(y_true, dtype=float)
    p = clip_probability(prob)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]
    out_prefix = f"{prefix}_" if prefix else ""
    metrics: Dict[str, Optional[float]] = {
        f"{out_prefix}brier": None,
        f"{out_prefix}log_loss": None,
        f"{out_prefix}ece": None,
        f"{out_prefix}prob_mean": None,
        f"{out_prefix}prob_std": None,
    }
    if y.size == 0:
        return metrics
    metrics[f"{out_prefix}prob_mean"] = float(np.mean(p))
    metrics[f"{out_prefix}prob_std"] = float(np.std(p))
    try:
        metrics[f"{out_prefix}brier"] = float(brier_score_loss(y, p))
    except Exception:
        pass
    try:
        metrics[f"{out_prefix}log_loss"] = float(log_loss(y, p, labels=[0, 1]))
    except Exception:
        pass
    bins = np.linspace(0.0, 1.0, max(2, int(n_bins)) + 1)
    bin_ids = np.digitize(p, bins[1:-1], right=False)
    ece = 0.0
    for bin_id in range(len(bins) - 1):
        in_bin = bin_ids == bin_id
        if not np.any(in_bin):
            continue
        weight = float(np.mean(in_bin))
        ece += weight * abs(float(np.mean(p[in_bin])) - float(np.mean(y[in_bin])))
    metrics[f"{out_prefix}ece"] = float(ece)
    return metrics


@dataclass
class BinaryProbabilityCalibrator:
    """Temperature and/or isotonic calibrator for binary probabilities."""

    method: str = "temperature_isotonic"
    temperature: float = 1.0
    isotonic: Optional[IsotonicRegression] = None

    @classmethod
    def fit(
        cls,
        prob: Any,
        target: Any,
        *,
        method: Optional[str] = "temperature_isotonic",
    ) -> "BinaryProbabilityCalibrator":
        value = "none" if method is None else str(method).strip().lower()
        if value in _DISABLED:
            return cls(method="none")
        if value not in {"temperature", "isotonic", "temperature_isotonic"}:
            raise ValueError(
                "nuisance calibration must be one of 'none', 'temperature', "
                "'isotonic', or 'temperature_isotonic'"
            )
        p = clip_probability(prob)
        y = np.asarray(target, dtype=float)
        mask = np.isfinite(y) & np.isfinite(p)
        p = p[mask]
        y = y[mask]
        calibrator = cls(method=value)
        if p.size < 8 or np.unique(y).size < 2:
            return calibrator

        working = p
        if "temperature" in value:
            calibrator.temperature = _fit_temperature(p, y)
            working = _apply_temperature(p, calibrator.temperature)
        if "isotonic" in value and np.unique(working).size >= 3:
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(working, y)
            calibrator.isotonic = iso
        return calibrator

    def transform(self, prob: Any) -> np.ndarray:
        p = clip_probability(prob)
        if self.method in _DISABLED or self.method == "none":
            return p
        if "temperature" in self.method:
            p = _apply_temperature(p, self.temperature)
        if self.isotonic is not None:
            p = self.isotonic.predict(p)
        return clip_probability(p)

    def metadata(self, prefix: str) -> Dict[str, Any]:
        return {
            f"{prefix}_calibration_method": self.method,
            f"{prefix}_temperature": float(self.temperature),
            f"{prefix}_isotonic": self.isotonic is not None,
        }


def _logit(prob: np.ndarray) -> np.ndarray:
    p = clip_probability(prob)
    return np.log(p / (1.0 - p))


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(value, -50.0, 50.0)))


def _apply_temperature(prob: np.ndarray, temperature: float) -> np.ndarray:
    t = max(float(temperature), 1e-3)
    return clip_probability(_sigmoid(_logit(prob) / t))


def _fit_temperature(prob: np.ndarray, target: np.ndarray) -> float:
    logits = _logit(prob)
    y = np.asarray(target, dtype=float)
    best_t = 1.0
    best_loss = math.inf
    for t in np.exp(np.linspace(math.log(0.2), math.log(10.0), 80)):
        calibrated = _sigmoid(logits / float(t))
        try:
            loss = float(log_loss(y, calibrated, labels=[0, 1]))
        except Exception:
            continue
        if loss < best_loss:
            best_loss = loss
            best_t = float(t)
    return best_t


# ---------------------------------------------------------------------------
# Two-parameter logistic (Platt) recalibration on the logit scale.
# ---------------------------------------------------------------------------

_PLATT_EPS = 1e-6
_PLATT_MAX_NEWTON_STEPS = 100


def _platt_clip(prob: np.ndarray) -> np.ndarray:
    return np.clip(prob, _PLATT_EPS, 1.0 - _PLATT_EPS)


def _platt_validated_probability(prob: Any, *, name: str) -> np.ndarray:
    values = np.asarray(prob, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite probabilities")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError(f"{name} must lie in [0, 1]")
    return values


def _platt_min_rows(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError("min_rows must be an integer")
    if int(value) < 2:
        raise ValueError("min_rows must be at least 2")
    return int(value)


def _one_dimensional(values: Any, *, name: str, dtype: Any = None) -> np.ndarray:
    array = np.asarray(values) if dtype is None else np.asarray(values, dtype=dtype)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


def _fit_logistic_on_logit(
    score: np.ndarray, target: np.ndarray
) -> tuple[Optional[np.ndarray], str]:
    """Unpenalized MLE of ``target ~ 1 + score`` by damped Newton steps.

    The caller rules out complete and quasi-complete separation first, so the
    log-likelihood is strictly concave with a finite maximizer.
    """

    design = np.column_stack([np.ones(len(score)), score])
    rows = float(len(score))

    def loss(beta: np.ndarray) -> float:
        z = design @ beta
        return float(np.mean(np.logaddexp(0.0, z) - target * z))

    beta = np.array([0.0, 1.0])
    current = loss(beta)
    for _ in range(_PLATT_MAX_NEWTON_STEPS):
        mu = expit(design @ beta)
        gradient = design.T @ (mu - target) / rows
        hessian = (design.T * (mu * (1.0 - mu))) @ design / rows
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            return None, "identity_optimizer_failed"
        if not np.all(np.isfinite(step)):
            return None, "identity_optimizer_failed"
        decrease = float(gradient @ step)
        scale = 1.0
        while True:
            candidate = beta - scale * step
            value = loss(candidate)
            if np.isfinite(value) and value <= current - 1e-4 * scale * decrease:
                break
            scale *= 0.5
            if scale < 1e-12:
                break
        if scale < 1e-12:
            # No descent is possible: the current point is stationary to
            # machine precision or the problem is numerically degenerate.
            if float(np.max(np.abs(gradient))) <= 1e-8:
                return beta, "fitted"
            return None, "identity_optimizer_failed"
        beta, current = candidate, value
        if float(np.max(np.abs(scale * step))) <= 1e-12:
            break
    mu = expit(design @ beta)
    gradient = design.T @ (mu - target) / rows
    if float(np.max(np.abs(gradient))) > 1e-8:
        return None, "identity_optimizer_failed"
    return beta, "fitted"


@dataclass(frozen=True)
class PlattRecalibration:
    """Logistic recalibration ``sigmoid(a + b * logit(p))`` of probabilities.

    ``status`` is ``"fitted"`` when ``(a, b)`` were estimated; any other value
    starts with ``"identity_"`` and states why the map is the identity.
    """

    intercept: float = 0.0
    slope: float = 1.0
    status: str = "identity_not_fitted"
    rows: int = 0

    def __post_init__(self) -> None:
        if self.status == "fitted":
            if not (np.isfinite(self.intercept) and np.isfinite(self.slope)):
                raise ValueError("fitted Platt parameters must be finite")
            if self.slope <= 0.0:
                raise ValueError("fitted Platt slope must be positive")
        elif not str(self.status).startswith("identity_"):
            raise ValueError("Platt status must be 'fitted' or start with 'identity_'")

    @classmethod
    def fit(cls, prob: Any, target: Any, *, min_rows: int = 20) -> "PlattRecalibration":
        """Fit ``target ~ 1 + logit(clip(prob))`` by unpenalized maximum likelihood.

        Rows with a nonfinite probability or target are excluded.  The identity
        map is returned when the slope is not estimable or would invert the
        ranking (nonpositive slope).
        """

        minimum = _platt_min_rows(min_rows)
        p = _one_dimensional(prob, name="prob", dtype=float)
        y = _one_dimensional(target, name="target", dtype=float)
        if p.shape != y.shape:
            raise ValueError("prob and target must have the same length")
        usable = np.isfinite(p) & np.isfinite(y)
        p, y = p[usable], y[usable]
        if np.any((p < 0.0) | (p > 1.0)):
            raise ValueError("prob must lie in [0, 1]")
        if not np.isin(y, (0.0, 1.0)).all():
            raise ValueError("target must contain only 0 and 1")
        rows = len(y)
        if rows == 0:
            return cls(status="identity_no_finite_rows", rows=0)
        if rows < minimum:
            return cls(status="identity_too_few_rows", rows=rows)
        if np.unique(y).size < 2:
            return cls(status="identity_single_class", rows=rows)
        score = logit(_platt_clip(p))
        if float(np.ptp(score)) <= 1e-12:
            return cls(status="identity_constant_predictions", rows=rows)
        positive, negative = score[y == 1.0], score[y == 0.0]
        # A single score separates the classes when every positive scores at
        # least as high as every negative (or vice versa); the MLE then does
        # not exist.
        if positive.min() >= negative.max() or negative.min() >= positive.max():
            return cls(status="identity_separated", rows=rows)
        beta, status = _fit_logistic_on_logit(score, y)
        if beta is None:
            return cls(status=status, rows=rows)
        intercept, slope = float(beta[0]), float(beta[1])
        if not (np.isfinite(intercept) and np.isfinite(slope)):
            return cls(status="identity_nonfinite_estimate", rows=rows)
        if slope <= 0.0:
            return cls(status="identity_nonpositive_slope", rows=rows)
        return cls(intercept=intercept, slope=slope, status="fitted", rows=rows)

    def transform(self, prob: Any) -> np.ndarray:
        """Recalibrate finite probabilities; the identity map only clips."""

        p = _platt_clip(_platt_validated_probability(prob, name="prob"))
        if self.status != "fitted":
            return p
        return _platt_clip(expit(self.intercept + self.slope * logit(p)))

    def public_dict(self) -> Dict[str, Any]:
        return {
            "method": "platt_logistic_on_logit",
            "intercept": float(self.intercept),
            "slope": float(self.slope),
            "status": str(self.status),
            "rows": int(self.rows),
        }


def cross_fitted_platt(
    oof_prob: Any,
    target: Any,
    fold_ids: Any,
    *,
    heldout_prob: Any = None,
    fit_mask: Any = None,
    min_rows: int = 20,
) -> tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """Cross-fitted Platt recalibration of out-of-fold predictions.

    For each distinct fold id ``k`` a calibrator is fit on rows with
    ``fit_mask & (fold_ids != k)`` and applied to ALL rows with
    ``fold_ids == k`` regardless of ``fit_mask`` (for example, a control-arm
    outcome calibrator is fit on control rows but applied to every row).  No
    row's calibrated value therefore depends on its own fold's targets.

    ``heldout_prob`` (optional) is recalibrated by a calibrator fit on all
    ``fit_mask`` rows.

    Returns ``(calibrated_oof, calibrated_heldout_or_None, audit)``.
    """

    minimum = _platt_min_rows(min_rows)
    p = _one_dimensional(oof_prob, name="oof_prob", dtype=float)
    y = _one_dimensional(target, name="target", dtype=float)
    folds = _one_dimensional(fold_ids, name="fold_ids")
    if not (p.shape == y.shape == folds.shape):
        raise ValueError("oof_prob, target, and fold_ids must have the same length")
    if folds.dtype.kind == "f" and not np.all(np.isfinite(folds)):
        raise ValueError("fold_ids must be finite")
    if folds.dtype.kind == "O" and any(value is None for value in folds):
        raise ValueError("fold_ids must not contain None")
    if fit_mask is None:
        mask = np.ones(len(p), dtype=bool)
    else:
        mask = _one_dimensional(fit_mask, name="fit_mask")
        if mask.dtype != bool:
            raise TypeError("fit_mask must be a boolean array")
        if mask.shape != p.shape:
            raise ValueError("fit_mask must have the same length as oof_prob")
    heldout = (
        None
        if heldout_prob is None
        else _one_dimensional(heldout_prob, name="heldout_prob", dtype=float)
    )
    _platt_validated_probability(p, name="oof_prob")
    if heldout is not None:
        _platt_validated_probability(heldout, name="heldout_prob")

    calibrated = np.empty(len(p), dtype=float)
    per_fold: Dict[str, Any] = {}
    for fold in np.unique(folds):
        in_fold = folds == fold
        calibrator = PlattRecalibration.fit(
            p[mask & ~in_fold], y[mask & ~in_fold], min_rows=minimum
        )
        calibrated[in_fold] = calibrator.transform(p[in_fold])
        per_fold[str(fold)] = calibrator.public_dict()

    calibrated_heldout = None
    heldout_audit = None
    if heldout is not None:
        calibrator = PlattRecalibration.fit(p[mask], y[mask], min_rows=minimum)
        calibrated_heldout = calibrator.transform(heldout)
        heldout_audit = calibrator.public_dict()
    audit = {
        "method": "cross_fitted_platt_logistic_on_logit",
        "fit_rows": int(mask.sum()),
        "min_rows": minimum,
        "per_fold": per_fold,
        "heldout": heldout_audit,
    }
    return calibrated, calibrated_heldout, audit
