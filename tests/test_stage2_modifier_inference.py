"""Held-out effect sizes, cross-fold concept matching, and multi-split aggregation."""

from __future__ import annotations


import numpy as np
import pytest

from oci.inference import stage2_diagnostics as diagnostics


def test_mmb_aggregation_and_fixed_effect_pooling_arithmetic():
    assert diagnostics.mmb_quantile_aggregation([0.01, 0.02, 0.03, None, None]) == pytest.approx(0.06)
    assert diagnostics.mmb_quantile_aggregation([0.01, None, None, None, None]) == 1.0
    assert diagnostics.mmb_quantile_aggregation([0.001] * 5) == pytest.approx(0.002)
    assert diagnostics.mmb_quantile_aggregation([0.2] * 5) == pytest.approx(0.4)
    assert diagnostics.mmb_quantile_aggregation([]) == 1.0
    pooled = diagnostics.fixed_effect_pool([(1.0, 0.1), (2.0, 0.1)])
    assert pooled["estimate"] == pytest.approx(1.5)
    assert pooled["standard_error"] == pytest.approx(0.1 / np.sqrt(2))
    assert pooled["cochran_q"] == pytest.approx(50.0)
    assert pooled["i_squared"] == pytest.approx(49 / 50)
    assert pooled["fold_range"] == [1.0, 2.0]
    homogeneous = diagnostics.fixed_effect_pool([(1.0, 0.1), (1.0, 0.2), (1.0, 0.3)])
    assert homogeneous["cochran_q"] == pytest.approx(0.0) and homogeneous["i_squared"] == 0.0
    assert diagnostics.fixed_effect_pool([(1.0, 0.1)])["status"] == "unavailable"
