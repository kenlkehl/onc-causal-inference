"""Standalone ``oci run`` contracts: outer splits, oracle isolation, DR ATE, CLI."""

from __future__ import annotations

import hashlib
import json
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from oci.config import (
    AppliedInferenceConfig,
    ExplicitFeatureExtractionConfig,
    ExplicitFeatureForestConfig,
    ExplicitFeatureSpec,
    ExperimentConfig,
    ModelArchitectureConfig,
)
from oci.inference import applied
from oci.inference import agentic_explicit_feature_forest as aeff
from oci.inference.agentic_attention_variable_forest import AgenticAttentionVariableForestRunner


def _frame(n: int = 240, seed: int = 0, continuous: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    confounder = rng.normal(size=n)
    modifier = rng.normal(size=n)
    treatment = rng.binomial(1, 1.0 / (1.0 + np.exp(-0.5 * confounder)))
    linear = 0.3 * confounder + treatment * (1.0 + 0.8 * modifier)
    outcome = (
        linear + rng.normal(scale=0.5, size=n)
        if continuous
        else rng.binomial(1, 1.0 / (1.0 + np.exp(-linear)))
    )
    true_ite = 1.0 + 0.8 * modifier
    return pd.DataFrame(
        {
            "clinical_text": [f"note {i}" for i in range(n)],
            "treatment_indicator": treatment,
            "outcome_indicator": outcome,
            "explicit_feat_conf": confounder,
            "explicit_feat_conf_missing": False,
            "explicit_feat_mod": modifier,
            "explicit_feat_mod_missing": False,
            "true_ite": true_ite,
            "true_treatment_prob": 1.0 / (1.0 + np.exp(-0.5 * confounder)),
            "true_conf": confounder,
        }
    )


def _config(**overrides) -> AppliedInferenceConfig:
    forest = overrides.pop(
        "forest",
        ExplicitFeatureForestConfig(n_estimators=40, min_samples_leaf=5),
    )
    values = dict(
        seed=11,
        outcome_type="binary",
        cv_folds=3,
        architecture=ModelArchitectureConfig(
            model_type="explicit_feature_forest",
            explicit_feature_forest=forest,
        ),
        explicit_features=ExplicitFeatureExtractionConfig(
            enabled=True,
            features=[
                ExplicitFeatureSpec(name="conf", type="continuous", roles=["confounder"]),
                ExplicitFeatureSpec(name="mod", type="continuous", roles=["effect_modifier"]),
            ],
        ),
    )
    values.update(overrides)
    return AppliedInferenceConfig(**values)


# ---------------------------------------------------------------- SA-28 splits


def test_avf_and_aeff_use_the_shared_split_policy(tmp_path):
    frame = _frame()
    config = _config(
        cv_folds=3,
        architecture=ModelArchitectureConfig(model_type="agentic_attention_variable_forest"),
    )
    runner = AgenticAttentionVariableForestRunner(
        dataset=frame,
        config=config,
        output_path=tmp_path / "predictions.parquet",
        device=torch.device("cpu"),
        proposal_agent=object(),
        extraction_provider=object(),
    )
    assert not [column for column in runner.dataset.columns if column.startswith("true_")]
    expected = applied.standalone_outer_splits(runner.dataset, config)
    for (fold, train, test), (efold, etrain, etest) in zip(runner._analysis_splits(), expected):
        assert fold == efold and np.array_equal(train, etrain) and np.array_equal(test, etest)

    runner.config = _config(cv_folds=1)
    with pytest.raises(ValueError, match="No held-out outer split"):
        runner._analysis_splits()

    # The AEFF splitter no longer falls back to unstratified KFold for
    # continuous outcomes.
    continuous = _frame(continuous=True)
    splits = aeff._make_splits(continuous, _config(outcome_type="continuous"), 3, 4)
    expected = applied.stratified_fold_indices(
        continuous,
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        outcome_type="continuous",
        n_splits=3,
        seed=4,
    )
    assert all(np.array_equal(a[1], b[1]) for a, b in zip(splits, expected))


# ------------------------------------------- SA-29 oracle isolation, SA-25 ATE


def test_explicit_feature_forest_run_isolates_oracle_and_reports_dr_ate(tmp_path, monkeypatch):
    seen_columns = []

    def fake_extraction(dataset, config, output_path):
        seen_columns.append(list(dataset.columns))
        return dataset, ["explicit_feat_conf", "explicit_feat_mod"]

    monkeypatch.setattr(applied, "_run_explicit_feature_extraction", fake_extraction)
    output_path = tmp_path / "applied_inference" / "predictions.parquet"
    applied.run_applied_inference(
        dataset=_frame(),
        config=_config(),
        output_path=output_path,
        device=torch.device("cpu"),
    )

    assert seen_columns and not any(c.startswith("true_") for c in seen_columns[0])
    predictions = pd.read_parquet(output_path)
    assert not [c for c in predictions.columns if applied.is_oracle_name(c)]
    assert predictions["_oci_row_id"].tolist() == list(range(len(predictions)))
    assert np.isfinite(predictions["pred_dr_score"]).all()
    assert np.isfinite(predictions["pred_ite_stderr"]).all()
    assert (predictions["pred_ite_stderr"] > 0).all()

    summary = json.loads((output_path.parent / "ate_summary.json").read_text())
    assert summary["estimator"] == "cross_fitted_aipw"
    assert summary["honest_outer_holdout"] is True
    assert summary["n"] == len(predictions)
    assert summary["ate"] == pytest.approx(predictions["pred_dr_score"].mean())
    assert summary["ate_stderr"] > 0

    oracle = json.loads((output_path.parent / "posthoc_oracle_metrics.json").read_text())
    assert oracle["status"] == "complete"
    assert oracle["predictions_sha256"] == hashlib.sha256(output_path.read_bytes()).hexdigest()
    assert oracle["oracle_ite_column"] == "true_ite"
    assert oracle["metrics"]["n"] == len(predictions)
    assert np.isfinite(oracle["metrics"]["ite_pearson"])
    assert "propensity_pearson" in oracle["metrics"]
    assert len(oracle["by_outer_fold"]) == 3

    log = pd.read_csv(output_path.parent / "training_log.csv")
    assert {
        "ate_dr_heldout",
        "ate_dr_heldout_stderr",
        "ate_dr_fit_sample",
        "mean_cate_heldout",
    } <= set(log.columns)
    assert "ate_estimate" not in log.columns
    audits = json.loads((output_path.parent / "causal_forest_fit_audits.json").read_text())
    assert [item["outer_fold"] for item in audits] == [1, 2, 3]
    assert all(item["fit_audit"]["tuning_configured"] is False for item in audits)
    assert all(item["fit_audit"]["fitted_nuisance_models"]["treatment"] for item in audits)


def test_single_oracle_name_rule_is_shared_by_the_runners():
    from oci.inference import multi_model_agentic_forest as mmaf

    metrics = {"true_ite_mae": 1.0, "oracle_corr": 2.0, "ground_truth_x": 3.0, "r_loss": 4.0}
    assert aeff._non_oracle_metrics(metrics) == {"r_loss": 4.0}
    assert mmaf._agent_visible_metrics(metrics) == {"r_loss": 4.0}
    frame = pd.DataFrame(columns=["true_y0", "oracle_x", "ground_truth", "x"])
    assert list(mmaf._without_oracle_columns(frame).columns) == ["x"]


# ----------------------------------------------------- B11 forest configuration


# ------------------------------------------------------------- B10 CLI choices


@pytest.mark.parametrize(
    ("outcome_type", "objective", "accepted"),
    [("binary", "dina", True)],
)
def test_cli_effect_objective_choices_match_config_validation(
    tmp_path, monkeypatch, capsys, outcome_type, objective, accepted
):
    from oci import cli

    dataset = tmp_path / "data.parquet"
    _frame(n=12).to_parquet(dataset)
    config = ExperimentConfig(
        output_dir=str(tmp_path / "out"),
        applied_inference=AppliedInferenceConfig(
            dataset_path=str(dataset),
            outcome_type=outcome_type,
            architecture=ModelArchitectureConfig(model_type="agentic_attention_variable_forest"),
        ),
    )
    config_path = tmp_path / "config.json"
    config.to_json(str(config_path))
    ran = []

    class FakeRunner:
        def __init__(self, cfg):
            ran.append(cfg)

        def run(self):
            return {}

    monkeypatch.setattr(cli, "ExperimentRunner", FakeRunner)
    monkeypatch.setattr(
        sys, "argv", ["oci", "run", "--config", str(config_path), "--effect-objective", objective]
    )
    status = cli.main()
    if accepted:
        assert status == 0
        effect = ran[0].applied_inference.architecture.agentic_attention_variable_forest
        assert effect.effect_objective == objective
    else:
        assert status == 1 and not ran
        assert "Invalid --effect-objective" in capsys.readouterr().out


def test_agentic_example_config_extracts_from_long_notes_without_stopping():
    from pathlib import Path

    from oci.extraction.explicit_features import build_extraction_prompt

    path = (
        Path(__file__).resolve().parents[1]
        / "example_configs"
        / "agentic_explicit_feature_forest_config.json"
    )
    features = ExperimentConfig.from_json(str(path)).applied_inference.explicit_features
    # Bundled synthetic notes average about 50,000 characters (up to 130,000).
    note = "Progress note: ECOG 1, stable disease. " * 3_500
    prompt = build_extraction_prompt(
        note,
        [
            ExplicitFeatureSpec(
                name="ecog", type="categorical", categories=["0", "1", "2"],
                roles=["confounder"],
            )
        ],
        max_text_length=features.extraction_max_text_length,
        context_strategy=features.extraction_context_strategy,
    )
    assert len(note) > 130_000
    assert note in prompt
