"""B12: synthetic DGP sampling, oracle ITE scale and CLI defaults (no LLM calls)."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from synthetic_data import cli as synthetic_cli
from synthetic_data import generator as g
from synthetic_data.config import SyntheticDataConfig

SOURCE = (
    Path(__file__).resolve().parents[1]
    / "synthetic_data/example_synthetic_datasets/"
    "five_confounders_five_effect_modifiers_nsclc_with_structured/metadata.json"
)


def _saved_dgp():
    return json.loads(SOURCE.read_text())


def test_unicode_hyphen_category_is_sampled_with_its_proportion():
    metadata = _saved_dgp()
    feature = next(f for f in metadata["features"] if f["name"] == "prior_platinum_therapy")
    # The definition uses U+2011 while the summary statistics use "-".
    assert "First‑line" in feature["categories"]
    assert "First-line" in metadata["summary_statistics"]["prior_platinum_therapy"]["proportions"]

    np.random.seed(0)
    draws = [
        g._sample_patient_characteristics([feature], metadata["summary_statistics"])[
            "prior_platinum_therapy"
        ]
        for _ in range(4000)
    ]
    counts = pd.Series(draws).value_counts(normalize=True)
    # The sampled value keeps the definition's label so equation dummies match.
    assert counts.get("First‑line", 0.0) == pytest.approx(0.35, abs=0.03)
    assert counts.get("None", 0.0) == pytest.approx(0.30, abs=0.03)


def test_coefficient_names_are_matched_across_hyphen_spellings():
    valid = g._get_valid_coefficient_names(
        [
            {
                "name": "prior",
                "type": "categorical",
                "categories": ["None", "First‑line"],
                "roles": ["confounder"],
            }
        ]
    )
    equation = g._validate_equation_coefficients(
        {"coefficients": {"prior_First-line": 0.4}, "interactions": []},
        valid,
        "outcome_equation",
    )
    assert equation["coefficients"] == {"prior_First‑line": 0.4}


def test_positive_continuous_covariates_cannot_go_negative_and_keep_moments():
    feature = [{"name": "baseline_nlr", "type": "continuous", "roles": ["effect_modifier"]}]
    stats = {"baseline_nlr": {"mean": 4.5, "std": 2.5}}

    np.random.seed(1)
    values = np.array(
        [g._sample_patient_characteristics(feature, stats)["baseline_nlr"] for _ in range(20000)]
    )
    assert values.min() > 0
    assert values.mean() == pytest.approx(4.5, rel=0.03)
    assert values.std() == pytest.approx(2.5, rel=0.05)

    # The legacy Normal draw is still available and does produce negatives.
    np.random.seed(1)
    legacy = np.array(
        [
            g._sample_patient_characteristics(
                feature, stats, continuous_distribution="normal"
            )["baseline_nlr"]
            for _ in range(20000)
        ]
    )
    assert (legacy < 0).mean() > 0.02

    # Non-positive means are not forced positive.
    np.random.seed(2)
    centered = [
        g._sample_patient_characteristics(
            [{"name": "delta", "type": "continuous", "roles": ["confounder"]}],
            {"delta": {"mean": 0.0, "std": 1.0}},
        )["delta"]
        for _ in range(500)
    ]
    assert min(centered) < 0

    with pytest.raises(ValueError, match="positive_lognormal"):
        replace(SyntheticDataConfig(), continuous_covariate_distribution="truncated").validate()


@pytest.mark.parametrize("outcome_type", ["binary", "continuous"])
def test_scaffold_stores_the_ite_on_the_outcome_scale(outcome_type):
    metadata = _saved_dgp()
    config = SyntheticDataConfig.from_dict(metadata["config"])
    config.outcome_type = outcome_type
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


def test_feature_names_cannot_shadow_oracle_columns():
    config = SyntheticDataConfig()
    features = [{"name": "ite", "type": "continuous", "roles": ["confounder"]}]
    with pytest.raises(ValueError, match="collides with the oracle column"):
        g._build_patient_scaffold_record(
            0,
            config,
            features,
            {"ite": {"mean": 1.0, "std": 0.1}},
            {"intercept": 0.0, "coefficients": {}},
            {"intercept": 0.0, "coefficients": {}, "treatment_coefficient": 0.2},
        )


def _run_cli(monkeypatch, tmp_path, *args):
    captured = {}

    def fake_generate(config, num_workers, show_progress, dgp_metadata_path):
        captured["config"] = config
        frame = pd.DataFrame({"treatment_indicator": [0, 1], "outcome_indicator": [0, 1]})
        return frame, {}

    monkeypatch.setattr(synthetic_cli, "generate_synthetic_dataset", fake_generate)
    monkeypatch.setattr(
        sys,
        "argv",
        ["synthetic", "--output-dir", str(tmp_path / "out"), *args],
    )
    synthetic_cli.main()
    return captured["config"]


def test_cli_treatment_effect_defaults_and_overrides_follow_the_config(tmp_path, monkeypatch):
    default = _run_cli(monkeypatch, tmp_path)
    assert default.treatment_effect_prob == SyntheticDataConfig.treatment_effect_prob == 0.20
    assert default.target_control_outcome_rate == SyntheticDataConfig.target_control_outcome_rate

    saved = SyntheticDataConfig(treatment_effect_prob=0.5, target_control_outcome_rate=0.3)
    config_path = tmp_path / "saved.json"
    saved.to_json(str(config_path))
    kept = _run_cli(monkeypatch, tmp_path, "--config", str(config_path))
    assert kept.treatment_effect_prob == 0.5
    assert kept.target_control_outcome_rate == 0.3
    # An explicit value equal to the old CLI default still overrides the file.
    overridden = _run_cli(
        monkeypatch, tmp_path, "--config", str(config_path), "--treatment-effect", "0.2"
    )
    assert overridden.treatment_effect_prob == 0.2


def test_cli_treatment_effect_help_states_log_odds_units(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["synthetic", "--help"])
    with pytest.raises(SystemExit):
        synthetic_cli.main()
    help_text = " ".join(capsys.readouterr().out.split())
    assert "log-odds units for binary outcomes" in help_text
    assert "not a probability-scale effect" in help_text
