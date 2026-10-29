"""B12: synthetic DGP sampling, oracle ITE scale and CLI defaults (no LLM calls)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from synthetic_data import generator as g
from synthetic_data.config import SyntheticDataConfig

SOURCE = (
    Path(__file__).resolve().parents[1]
    / "synthetic_data/example_synthetic_datasets/"
    "five_confounders_five_effect_modifiers_nsclc_with_structured/metadata.json"
)


def _saved_dgp():
    return json.loads(SOURCE.read_text())


@pytest.mark.parametrize("outcome_type", ["continuous"])
def test_scaffold_stores_the_ite_on_the_outcome_scale(outcome_type):
    metadata = _saved_dgp()
    config = SyntheticDataConfig.from_dict(metadata["config"])
    config.outcome_type = outcome_type
    # The saved config predates the setting and loads as "normal"; the
    # positivity assertions below exercise the log-normal draw.
    config.continuous_covariate_distribution = "positive_lognormal"
    features = metadata["features"]
    stats = metadata["summary_statistics"]

    np.random.seed(5)
    records = [
        g._build_patient_scaffold_record(
            i, config, features, stats, metadata["treatment_equation"], metadata["outcome_equation"]
        )
        for i in range(40)
    ]
    for record in records:
        characteristics = {f["name"]: record[f"true_{f['name']}"] for f in features}
        lp = [
            g._compute_logit(characteristics, features, stats, metadata["outcome_equation"], t)
            for t in (0, 1)
        ]
        if outcome_type == "continuous":
            assert record["true_ite"] == pytest.approx(lp[1] - lp[0])
            assert record["true_y0"] == pytest.approx(lp[0])
            assert record["true_y1"] == pytest.approx(lp[1])
            for column in ("true_ite_prob", "true_y0_prob", "true_y1_prob", "true_outcome_prob"):
                assert column not in record
        else:
            probability = [1.0 / (1.0 + np.exp(-value)) for value in lp]
            assert record["true_ite_prob"] == pytest.approx(probability[1] - probability[0])
            assert record["true_ite"] == pytest.approx(record["true_ite_prob"])
            assert record["true_y1"] == pytest.approx(record["true_y1_prob"])
        assert record["true_baseline_nlr"] > 0
        assert record["true_creatinine_clearance"] > 0

    stats_out = g.oracle_dataset_statistics(pd.DataFrame(records))
    assert "mean_ite" in stats_out
    assert ("mean_ite_prob" in stats_out) is (outcome_type == "binary")
