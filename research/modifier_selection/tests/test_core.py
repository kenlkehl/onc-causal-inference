"""Fast deterministic tests for the standalone Note 030 implementation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from oci_ridge import rank_safe_columns, ridge_interaction_gain  # noqa: E402


class CoreTests(unittest.TestCase):
    def test_rank_safe_columns_drops_duplicates(self) -> None:
        base = np.ones((4, 1))
        additions = np.column_stack([np.arange(4), np.arange(4), np.ones(4)])
        result, kept = rank_safe_columns(base, additions)
        self.assertEqual(kept, [0])
        self.assertEqual(result.shape, (4, 2))

    def test_interaction_signal_has_positive_gain(self) -> None:
        rng = np.random.default_rng(20260928)
        x = rng.normal(size=(600, 1))
        a = rng.binomial(1, 0.5, size=600)
        ra = a - 0.5
        ry = ra * x[:, 0] + rng.normal(scale=0.05, size=600)
        gain = ridge_interaction_gain(
            x[:400], x[400:], ra[:400], ra[400:], ry[:400], ry[400:]
        )
        self.assertGreater(gain, 0.1)


if __name__ == "__main__":
    unittest.main()
