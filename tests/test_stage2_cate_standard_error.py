"""B2: the forest CATE standard error reaches Stage 2 fold outputs."""

import numpy as np
import pandas as pd

from oci.inference import plain_handoff_stage2_analysis as analysis
from oci.models.causal_forest_head import CausalForestHead
from tests.test_stage2_propensity_overlap import final_fixture


def test_stage2_forest_fold_writes_finite_cate_standard_errors(monkeypatch, tmp_path):
    kwargs, _ = final_fixture(monkeypatch, tmp_path)
    # Keep the fixture's stubbed nuisances but fit the real causal forest.
    monkeypatch.setattr(analysis, "CausalForestHead", CausalForestHead)

    analysis.estimate_outer_fold(**kwargs)
    frame = pd.read_csv(tmp_path / "predictions.csv")

    eligible = frame[frame["effect_eligible"]]
    standard_error = eligible["estimated_cate_standard_error"].to_numpy(dtype=float)
    assert len(standard_error) and np.isfinite(standard_error).all()
    assert (standard_error > 0).all()
    width = (eligible["estimated_cate_upper_95"] - eligible["estimated_cate_lower_95"]).to_numpy()
    np.testing.assert_allclose(width, 2 * 1.959964 * standard_error, rtol=1e-4)
