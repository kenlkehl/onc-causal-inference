"""Small probability calibration helpers for nuisance models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
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
