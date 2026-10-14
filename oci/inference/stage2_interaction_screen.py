"""Shared adjustment design for conventional treatment-by-candidate tests.

The supplied nuisance predictions must be out-of-fold within the evidence
training partition. These are ordinary nested GLM/F tests with learned
adjustment scores, not orthogonal tests or calibrated post-selection inference.
"""

import numpy as np

from .stage2_statistical_selection import _rank_safe_columns

ADJUSTMENT_VERSION = "crossfit_score_adjustment_v1"
PROBABILITY_CLIP = 1e-6


def _score_basis(values, *, probability):
    """A linear term and one natural-cubic term at 10/50/90% quantiles."""
    values = np.asarray(values, dtype=float)
    if probability:
        if np.any((values < 0) | (values > 1)):
            raise ValueError("interaction adjustment probabilities must lie in [0, 1]")
        values = np.clip(values, PROBABILITY_CLIP, 1 - PROBABILITY_CLIP)
        values = np.log(values) - np.log1p(-values)
    center, scale = float(np.mean(values)), float(np.std(values))
    if scale <= 1e-12:
        return np.empty((len(values), 0)), {"constant": True}
    z = (values - center) / scale
    knots = np.quantile(z, [0.1, 0.5, 0.9])
    columns = [z]
    if np.min(np.diff(knots)) > 1e-8:

        def divided(j):
            return (np.maximum(z - knots[j], 0) ** 3 - np.maximum(z - knots[2], 0) ** 3) / (
                knots[2] - knots[j]
            )

        nonlinear = (divided(0) - divided(1)) / (knots[2] - knots[0]) ** 2
        columns.append(nonlinear - nonlinear.mean())
    return np.column_stack(columns), {
        "constant": False,
        "center": center,
        "scale": scale,
        "knots": knots.tolist(),
        "columns": len(columns),
    }


def adjustment_design(treatment, propensity, outcome_prediction, *, binary):
    """Build one common intercept/T/score matrix for reduced and full models."""
    t, e, m = (np.asarray(x, dtype=float) for x in (treatment, propensity, outcome_prediction))
    if any(
        x.ndim != 1 or x.shape != t.shape or not np.isfinite(x).all() for x in (t, e, m)
    ) or not len(t):
        raise ValueError("interaction adjustment requires aligned finite nuisance arrays")
    base = np.column_stack((np.ones(len(t)), t))
    details = {}
    for name, values, probability in (("propensity", e, True), ("outcome_prediction", m, binary)):
        basis, audit = _score_basis(values, probability=probability)
        base, kept = _rank_safe_columns(base, basis)
        details[name] = {
            **audit,
            "transform": "logit" if probability else "identity",
            "retained_basis_columns": kept,
        }
    return base, {
        "version": ADJUSTMENT_VERSION,
        "nuisance_predictions": "cross_fitted_within_evidence_training_partition",
        "basis": "natural_cubic_spline_with_10_50_90_percentile_knots",
        "probability_clip": PROBABILITY_CLIP,
        "shared_reduced_full_adjustment": True,
        "base_columns": base.shape[1],
        "scores": details,
        "p_value_calibration": "nominal_conditional_on_learned_scores_not_post_selection_inference",
    }
