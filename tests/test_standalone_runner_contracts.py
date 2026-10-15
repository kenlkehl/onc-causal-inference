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


def test_stratified_folds_balance_treatment_by_outcome_and_follow_the_seed():
    frame = _frame()
    kwargs = dict(
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        outcome_type="binary",
        n_splits=3,
    )
    folds = applied.stratified_fold_indices(frame, seed=5, **kwargs)
    again = applied.stratified_fold_indices(frame, seed=5, **kwargs)
    other = applied.stratified_fold_indices(frame, seed=6, **kwargs)

    assert all(np.array_equal(a[1], b[1]) for a, b in zip(folds, again))
    assert not all(np.array_equal(a[1], b[1]) for a, b in zip(folds, other))
    strata = frame["treatment_indicator"].astype(str) + frame["outcome_indicator"].astype(str)
    overall = strata.value_counts(normalize=True)
    for _, test_idx in folds:
        share = strata.iloc[test_idx].value_counts(normalize=True)
        assert (share.reindex(overall.index) - overall).abs().max() < 0.05
    assert sorted(np.concatenate([test for _, test in folds]).tolist()) == list(range(len(frame)))


def test_continuous_outcomes_are_stratified_and_tiny_arms_fail_closed():
    frame = _frame(continuous=True)
    folds = applied.stratified_fold_indices(
        frame,
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        outcome_type="continuous",
        n_splits=4,
        seed=1,
    )
    treated_share = frame["treatment_indicator"].mean()
    for _, test_idx in folds:
        assert abs(frame["treatment_indicator"].iloc[test_idx].mean() - treated_share) < 0.05

    tiny = pd.DataFrame({"t": [1, 0, 0, 0, 0, 0], "y": [1, 0, 1, 0, 1, 0]})
    with pytest.raises(ValueError, match="smallest treatment arm"):
        applied.stratified_fold_indices(
            tiny,
            treatment_column="t",
            outcome_column="y",
            outcome_type="binary",
            n_splits=3,
            seed=0,
        )


def test_outer_splits_use_configured_seed_and_fail_closed_without_holdout():
    frame = _frame()
    first = applied.standalone_outer_splits(frame, _config(seed=1))
    second = applied.standalone_outer_splits(frame, _config(seed=2))
    assert [fold for fold, _, _ in first] == [1, 2, 3]
    assert not all(np.array_equal(a[2], b[2]) for a, b in zip(first, second))

    with pytest.raises(ValueError, match="explicit integer seed"):
        applied.standalone_outer_splits(frame, _config(seed=None))

    with pytest.raises(ValueError, match="allow_non_honest_outer_split"):
        applied.standalone_outer_splits(frame, _config(cv_folds=1))

    allowed = _config(
        cv_folds=1,
        forest=ExplicitFeatureForestConfig(allow_non_honest_outer_split=True),
    )
    ((fold, train_idx, test_idx),) = applied.standalone_outer_splits(frame, allowed)
    assert fold == 1 and np.array_equal(train_idx, test_idx)

    split_frame = frame.assign(split=np.where(np.arange(len(frame)) % 4 == 0, "test", "train"))
    ((_, train_idx, test_idx),) = applied.standalone_outer_splits(split_frame, _config(cv_folds=1))
    assert set(train_idx).isdisjoint(test_idx)
    assert (split_frame["split"].iloc[test_idx] == "test").all()


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


def test_oracle_ite_column_resolution_respects_outcome_scale():
    assert applied.resolve_oracle_ite_column(["true_ite", "true_ite_prob"], "binary") == "true_ite"
    assert applied.resolve_oracle_ite_column(["true_ite_prob"], "binary") == "true_ite_prob"
    assert applied.resolve_oracle_ite_column(["true_ite_prob"], "continuous") is None
    assert applied.resolve_oracle_ite_column(["true_ite"], "continuous") == "true_ite"


def test_posthoc_oracle_metrics_refuse_leaked_oracle_columns(tmp_path):
    path = tmp_path / "predictions.parquet"
    pd.DataFrame(
        {"_oci_row_id": [0, 1], "pred_ite_prob": [0.1, 0.2], "true_ite": [0.0, 0.0]}
    ).to_parquet(path)
    oracle = pd.DataFrame({"_oci_row_id": [0, 1], "true_ite": [0.1, 0.3]})
    with pytest.raises(RuntimeError, match="oracle columns"):
        applied.write_posthoc_oracle_metrics(path, oracle, outcome_type="binary")


def test_single_oracle_name_rule_is_shared_by_the_runners():
    from oci.inference import multi_model_agentic_forest as mmaf

    metrics = {"true_ite_mae": 1.0, "oracle_corr": 2.0, "ground_truth_x": 3.0, "r_loss": 4.0}
    assert aeff._non_oracle_metrics(metrics) == {"r_loss": 4.0}
    assert mmaf._agent_visible_metrics(metrics) == {"r_loss": 4.0}
    frame = pd.DataFrame(columns=["true_y0", "oracle_x", "ground_truth", "x"])
    assert list(mmaf._without_oracle_columns(frame).columns) == ["x"]


# ----------------------------------------------------- B11 forest configuration


def test_explicit_forest_config_disables_tuning_and_audits_it():
    from oci.inference.applied_explicit_feature_forest import fit_predict_explicit_forest

    assert ExplicitFeatureForestConfig().tune_model is False
    assert ExplicitFeatureForestConfig().allow_non_honest_outer_split is False
    with pytest.raises(TypeError, match="tune_model"):
        ExplicitFeatureForestConfig(tune_model="yes")

    frame = _frame(n=300)
    x = frame[["explicit_feat_mod"]].to_numpy()
    w = frame[["explicit_feat_conf"]].to_numpy()
    t = frame["treatment_indicator"].to_numpy()
    y = frame["outcome_indicator"].to_numpy()
    for tune in (False, True):
        fitted = fit_predict_explicit_forest(
            X_train=x[:200], W_train=w[:200], T_train=t[:200], Y_train=y[:200],
            X_test=x[200:], W_test=w[200:], T_test=t[200:], Y_test=y[200:],
            cf_config=ExplicitFeatureForestConfig(
                n_estimators=40, min_samples_leaf=5, tune_model=tune
            ),
            outcome_type="binary",
            random_state=3,
        )
        audit = fitted["fit_audit"]
        assert audit["tuning_configured"] is tune
        assert {"min_weight_fraction_leaf", "min_var_fraction_leaf", "max_depth"} <= set(
            audit["effective_parameters"]
        )
        if not tune:
            assert audit["tuned_parameter_changes"] == {}


# ------------------------------------------------------------- B10 CLI choices


@pytest.mark.parametrize(
    ("outcome_type", "objective", "accepted"),
    [
        ("binary", "auto", True),
        ("binary", "dina", True),
        ("binary", "squared_r_loss", False),
        ("continuous", "pseudo_outcome_mse", True),
        ("continuous", "dina", False),
    ],
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


def test_agentic_explicit_feature_forest_smoke_persists_audits_and_hides_oracle(
    tmp_path, monkeypatch
):
    from oci.config import AgenticFeatureSearchConfig

    contexts = []

    class StubAgent:
        def propose(self, context):
            contexts.append(json.dumps(context, default=str))
            return []

    class StubProvider:
        def ensure_features(self, dataset, specs):
            assert not [c for c in dataset.columns if applied.is_oracle_name(c)]
            return dataset

    monkeypatch.setattr(aeff, "make_feature_search_agent", lambda config: StubAgent())
    monkeypatch.setattr(
        aeff, "make_explicit_feature_extraction_provider", lambda **kwargs: StubProvider()
    )
    config = _config(
        architecture=ModelArchitectureConfig(
            model_type="agentic_explicit_feature_forest",
            explicit_feature_forest=ExplicitFeatureForestConfig(
                n_estimators=40, min_samples_leaf=5
            ),
            agentic_feature_search=AgenticFeatureSearchConfig(
                outer_folds=3, inner_folds=2, max_iterations=1, agent_model_name="stub"
            ),
        ),
    )
    output_path = tmp_path / "applied_inference" / "predictions.parquet"
    applied.run_applied_inference(
        dataset=_frame(), config=config, output_path=output_path, device=torch.device("cpu")
    )

    artifacts = output_path.parent / "agentic_feature_search"
    audits = json.loads((artifacts / "causal_forest_fit_audits.json").read_text())
    assert [item["outer_fold"] for item in audits] == [1, 2, 3]
    assert all(item["fit_audit"]["tuning_configured"] is False for item in audits)
    for name in ("inner_cv_metrics.csv", "outer_cv_metrics.csv"):
        columns = pd.read_csv(artifacts / name).columns
        assert not [c for c in columns if applied.is_oracle_name(c)]
        assert "ate_dr_heldout" in columns and "ate_estimate" not in columns
    assert contexts and not any("true_" in context for context in contexts)
    summary = json.loads((output_path.parent / "ate_summary.json").read_text())
    assert summary["n"] == len(pd.read_parquet(output_path)) and summary["ate_stderr"] > 0
    assert json.loads((output_path.parent / "posthoc_oracle_metrics.json").read_text())[
        "status"
    ] == "complete"


def test_dispatch_fails_closed_before_extraction(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        applied, "_run_explicit_feature_extraction", lambda *args: calls.append(args)
    )
    for config, message in (
        (_config(cv_folds=1), "No held-out outer split"),
        (_config(seed=None), "explicit integer seed"),
    ):
        with pytest.raises(ValueError, match=message):
            applied.run_applied_inference(
                dataset=_frame(),
                config=config,
                output_path=tmp_path / "predictions.parquet",
                device=torch.device("cpu"),
            )
    assert not calls


def test_explicit_forest_without_effect_modifiers_estimates_one_constant_effect():
    from oci.inference.applied_explicit_feature_forest import fit_predict_explicit_forest

    rng = np.random.default_rng(0)
    w = rng.normal(size=(240, 2))
    t = rng.binomial(1, 1 / (1 + np.exp(-w[:, 0])))
    y = rng.binomial(1, 1 / (1 + np.exp(-(0.5 * w[:, 0] + 0.8 * t))))
    fitted = fit_predict_explicit_forest(
        X_train=None, W_train=w[:160], T_train=t[:160], Y_train=y[:160],
        X_test=None, W_test=w[160:], T_test=t[160:], Y_test=y[160:],
        cf_config=ExplicitFeatureForestConfig(n_estimators=40, min_samples_leaf=5),
        outcome_type="binary", random_state=3,
    )
    assert fitted["tau"].shape == (80,)
    assert np.allclose(fitted["tau"], fitted["tau"][0])
    assert np.isfinite(fitted["ate_metrics"]["ate_dr_heldout"])


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


@pytest.mark.parametrize("workers,accepted", [("0", False), ("-2", False), ("3", True)])
def test_cli_workers_override_rejects_nonpositive_values(
    tmp_path, monkeypatch, capsys, workers, accepted
):
    from oci import cli

    dataset = tmp_path / "data.parquet"
    _frame(n=12).to_parquet(dataset)
    config = ExperimentConfig(
        output_dir=str(tmp_path / "out"),
        num_workers=2,
        applied_inference=AppliedInferenceConfig(dataset_path=str(dataset)),
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
        sys, "argv", ["oci", "run", "--config", str(config_path), "--workers", workers]
    )
    status = cli.main()
    if accepted:
        assert status == 0 and ran[0].num_workers == int(workers)
    else:
        assert status == 1 and not ran
        assert "--workers must be a positive integer" in capsys.readouterr().out
