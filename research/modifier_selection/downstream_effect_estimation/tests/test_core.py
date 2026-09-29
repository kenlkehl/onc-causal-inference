"""Deterministic tests of scale conversion and DINA fitting."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.special import expit, logit

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from core import (  # noqa: E402
    counterfactuals_from_log_or,
    counterfactuals_from_rd,
    fit_dina,
    oracle_nuisances,
    prediction_metrics,
)


class CoreTests(unittest.TestCase):
    def test_constant_effect_has_zero_heterogeneity_correlation(self) -> None:
        metrics = prediction_metrics(np.array([-1.0, 0.0, 1.0]), np.repeat(0.25, 3))
        self.assertEqual(metrics["pearson"], 0.0)
        self.assertEqual(metrics["spearman"], 0.0)

    def test_counterfactual_round_trip(self) -> None:
        e = np.array([0.2, 0.5, 0.8])
        mu0 = np.array([0.1, 0.4, 0.7])
        mu1 = np.array([0.2, 0.3, 0.9])
        nuisance = oracle_nuisances(e, mu0, mu1)
        rd0, rd1 = counterfactuals_from_rd(mu1 - mu0, e, nuisance["m"])
        lo0, lo1 = counterfactuals_from_log_or(
            logit(mu1) - logit(mu0), nuisance["a"], nuisance["nu"]
        )
        np.testing.assert_allclose(rd0, mu0)
        np.testing.assert_allclose(rd1, mu1)
        np.testing.assert_allclose(lo0, mu0)
        np.testing.assert_allclose(lo1, mu1)

    def test_dina_recovers_large_sample_linear_log_or(self) -> None:
        rng = np.random.default_rng(20260929)
        n = 30_000
        x = rng.normal(size=(n, 1))
        e = np.full(n, 0.5)
        mu0 = expit(-0.3 + 0.2 * x[:, 0])
        tau = 0.4 + 0.7 * x[:, 0]
        mu1 = expit(logit(mu0) + tau)
        nuisance = oracle_nuisances(e, mu0, mu1)
        treatment = rng.binomial(1, e)
        outcome = rng.binomial(1, np.where(treatment == 1, mu1, mu0))
        fit = fit_dina(x, treatment, outcome, nuisance["a"], nuisance["nu"])
        prediction = fit.predict_log_or(x)
        self.assertGreater(np.corrcoef(tau, prediction)[0, 1], 0.999)
        self.assertLess(np.sqrt(np.mean((tau - prediction) ** 2)), 0.06)


if __name__ == "__main__":
    unittest.main()
