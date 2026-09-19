"""Held-out nuisance diagnostics and a shared, inclusive overlap policy.

Calibration is descriptive only: it never recalibrates predictions or uses
held-out outcomes to choose propensity eligibility.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import log_loss, roc_auc_score


def validate_propensity_bounds(minimum: float | None, maximum: float | None) -> None:
    for name, value in (("min_propensity", minimum), ("max_propensity", maximum)):
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError(f"{name} must be null or a finite number in [0, 1]")
    if minimum is not None and maximum is not None and minimum >= maximum:
        raise ValueError("min_propensity must be less than max_propensity")


def propensity_eligibility(prediction, minimum=None, maximum=None) -> np.ndarray:
    validate_propensity_bounds(minimum, maximum)
    p = np.asarray(prediction, dtype=float)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("propensity predictions must be finite probabilities")
    return ((p >= minimum) if minimum is not None else np.ones(p.shape, dtype=bool)) & (
        (p <= maximum) if maximum is not None else np.ones(p.shape, dtype=bool)
    )


def overlap_diagnostics(prediction, minimum=None, maximum=None) -> dict[str, Any]:
    p = np.asarray(prediction, dtype=float)
    keep = propensity_eligibility(p, minimum, maximum)
    n = len(p)
    low, high = int(np.sum(p < 0.10)), int(np.sum(p > 0.90))
    return {
        "min_propensity": minimum,
        "max_propensity": maximum,
        "bounds_are_inclusive": True,
        "rows": n,
        "eligible_rows": int(keep.sum()),
        "excluded_rows": int((~keep).sum()),
        "excluded_below_min": int(np.sum(p < minimum)) if minimum is not None else 0,
        "excluded_above_max": int(np.sum(p > maximum)) if maximum is not None else 0,
        "below_0_10_rows": low,
        "above_0_90_rows": high,
        "outside_0_10_0_90_fraction": (low + high) / n if n else None,
        "propensity_min": float(p.min()) if n else None,
        "propensity_max": float(p.max()) if n else None,
        "quantiles": (
            {
                str(q): float(np.quantile(p, q))
                for q in (0, 0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99, 1)
            }
            if n
            else {}
        ),
    }


def calibration_diagnostics(observed, predicted, *, binary: bool) -> dict[str, Any]:
    y, p = np.asarray(observed, dtype=float), np.asarray(predicted, dtype=float)
    if y.shape != p.shape:
        raise ValueError("calibration observations and predictions must align")
    finite = np.isfinite(y) & np.isfinite(p)
    y, p = y[finite], p[finite]
    result: dict[str, Any] = {
        "rows": len(y),
        "nonfinite_rows": int((~finite).sum()),
        "outcome_type": "binary" if binary else "continuous",
    }
    if not len(y):
        return {**result, "status": "no_rows"}
    result.update(
        status="ok",
        observed_mean=float(y.mean()),
        predicted_mean=float(p.mean()),
        mean_prediction_error=float(np.mean(p - y)),
        mse=float(np.mean((p - y) ** 2)),
    )
    if not binary:
        result["rmse"] = float(np.sqrt(result["mse"]))
        if np.ptp(p) > 1e-12:
            intercept, slope = np.linalg.lstsq(
                np.column_stack([np.ones(len(p)), p]), y, rcond=None
            )[0]
            result.update(calibration_intercept=float(intercept), calibration_slope=float(slope))
        else:
            result.update(
                calibration_intercept=None,
                calibration_slope=None,
                calibration_status="constant_predictions",
            )
        return result
    if not np.isin(y, [0, 1]).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("binary calibration requires binary outcomes and probabilities")
    result.update(
        brier_score=result["mse"],
        log_loss=float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])),
        auroc=float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
    )
    bins = np.minimum((p * 10).astype(int), 9)
    result["reliability_bins"] = [
        {
            "lower": b / 10,
            "upper": (b + 1) / 10,
            "rows": int(np.sum(bins == b)),
            "predicted_mean": float(p[bins == b].mean()) if np.any(bins == b) else None,
            "observed_fraction": float(y[bins == b].mean()) if np.any(bins == b) else None,
        }
        for b in range(10)
    ]
    result["expected_calibration_error_10_bins"] = float(
        sum(
            b["rows"] * abs(b["predicted_mean"] - b["observed_fraction"])
            for b in result["reliability_bins"]
            if b["rows"]
        )
        / len(y)
    )
    result.update(calibration_intercept=None, calibration_slope=None)
    if len(np.unique(y)) < 2 or np.ptp(p) < 1e-12:
        result["calibration_status"] = "single_class_or_constant_predictions"
        return result
    # Complete or quasi-complete separation has no finite unpenalized MLE.
    # Detect it directly rather than trusting a small optimizer gradient.
    if p[y == 1].min() >= p[y == 0].max() or p[y == 0].min() >= p[y == 1].max():
        result["calibration_status"] = "nonconverged_or_separated"
        return result
    x = np.column_stack([np.ones(len(p)), logit(np.clip(p, 1e-6, 1 - 1e-6))])

    def objective(beta):
        z = x @ beta
        return float(np.mean(np.logaddexp(0, z) - y * z)), x.T @ (expit(z) - y) / len(y)

    fit = minimize(
        objective,
        [0.0, 1.0],
        jac=True,
        method="L-BFGS-B",
        bounds=[(-30, 30), (-30, 30)],
        options={"ftol": 1e-12, "gtol": 1e-8},
    )
    # Separation or failed optimization does not produce an interpretable slope.
    if fit.success and np.max(np.abs(fit.x)) < 29.99:
        result.update(
            calibration_intercept=float(fit.x[0]),
            calibration_slope=float(fit.x[1]),
            calibration_status="ok",
        )
    else:
        result["calibration_status"] = "nonconverged_or_separated"
    return result
