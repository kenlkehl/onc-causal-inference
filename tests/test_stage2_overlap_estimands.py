"""Co-primary ATE/ATO estimands, the reporting-only overlap flag, and clip sensitivity."""

from __future__ import annotations


import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from oci.inference import stage2_diagnostics as diagnostics


def _population(rng, n):
    x = rng.normal(size=n)
    e = expit(1.5 * x)
    tau = 0.5 + x ** 2
    return x, e, tau


def _predictions(rng, n=1000, *, folds=2, clip=0.02):
    x, e, tau = _population(rng, n)
    t = rng.binomial(1, e).astype(float)
    mu0, mu1 = x, x + tau
    y = np.where(t == 1, mu1, mu0) + rng.normal(size=n)
    return pd.DataFrame({
        "_oci_row_id": np.arange(n), "outer_fold": np.arange(n) % folds + 1,
        "treatment": t, "outcome": y, "propensity": e, "mu0": mu0, "mu1": mu1,
        "effect_eligible": True, "aipw_score": diagnostics.dr_scores(y, t, e, mu0, mu1, clip=clip),
        "estimated_cate": tau,
    })


def test_ato_recovers_the_overlap_weighted_truth_with_calibrated_intervals():
    _, e, tau = _population(np.random.default_rng(0), 2_000_000)
    ato_truth = float(np.sum(e * (1 - e) * tau) / np.sum(e * (1 - e)))
    ate_truth = float(np.mean(tau))
    assert ate_truth - ato_truth > 0.3  # the two populations differ
    rng = np.random.default_rng(1)
    covered, estimates, reps = 0, [], 200
    for _ in range(reps):
        frame = _predictions(rng)
        result = diagnostics.co_primary_estimands(frame, propensity_clip=0.02)["ato"]
        low, high = result["confidence_interval_95"]
        covered += low <= ato_truth <= high
        estimates.append(result["estimate"])
    assert np.mean(estimates) == pytest.approx(ato_truth, abs=0.03)
    assert 0.90 <= covered / reps <= 0.99
