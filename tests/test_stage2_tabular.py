"""Supplied covariates reach the shared estimators without text models or LLMs."""

from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from oci.inference import research_all_evidence_workflow as workflow
from oci.inference.plain_handoff_stage2 import PlainHandoffStage2
from oci.inference.stage2_preflight import preflight_stage2
from oci.inference.stage2_tabular import INPUT_FILE, prepare_tabular_inputs


def inputs(tmp_path, *, outcome_type="binary"):
    rng = np.random.default_rng(28)
    n = 120
    age = rng.normal(65, 10, n)
    ecog = np.tile(np.arange(3), n // 3)
    treatment = rng.binomial(1, 0.5, n)
    outcome = (
        rng.binomial(1, 0.3 + 0.1 * treatment + 0.05 * ecog)
        if outcome_type == "binary"
        else (age - 65) / 10 + treatment * (1 + 0.2 * ecog) + rng.normal(size=n)
    )
    data = pd.DataFrame(
        {
            "id": [f"p{i}" for i in range(n)],
            "t": treatment,
            "y": outcome,
            "Age (years)": age,
            "ecog": ecog.astype(float),
            "true_ite": np.ones(n),
            "unused": ["DO_NOT_MODEL"] * n,
        }
    )
    data.loc[0, "Age (years)"] = np.nan
    data.loc[1, "ecog"] = np.nan
    path = tmp_path / "cohort.csv"
    data.to_csv(path, index=False)
    raw = {
        "dataset": str(path),
        "output_dir": str(tmp_path / "run"),
        "stage1_template": str(tmp_path / "no-stage1.json"),
        "neural_query_template": str(tmp_path / "no-neural-queries.json"),
        "columns": {"unit_id": "id", "treatment": "t", "outcome": "y"},
        "science": {"outer_folds": 2, "inner_folds": 2, "seed": 7, "outcome_type": outcome_type},
        "run": {"mode": "tabular"},
        "tabular": {
            "features": [
                {"column": "Age (years)", "value_type": "continuous", "roles": ["confounder"]},
                {
                    "column": "ecog",
                    "value_type": "ordinal",
                    "categories": [0, 1, 2],
                    "roles": "both",
                },
            ]
        },
    }
    return raw, workflow.compile_config(raw, config_dir=tmp_path), data


def forbid(*args, **kwargs):
    pytest.fail("tabular execution attempted discovery, extraction, or selection")


def test_config_cli_roundtrip_and_readonly_preflight(tmp_path, monkeypatch, capsys):
    raw, config, data = inputs(tmp_path)
    assert config.mode == "tabular" and config.components == ("stage2",)
    assert not config.stage2.endpoint and not config.stage2.model
    assert config.stage2.nuisance_learner == "super_learner"
    monkeypatch.setattr(workflow, "_load_stage1_template", forbid)
    monkeypatch.setattr(workflow, "_load_neural_query_template", forbid)
    monkeypatch.setattr(PlainHandoffStage2, "__init__", forbid)
    report = preflight_stage2(config)
    assert report["read_only"] and report["rows"] == len(data)
    assert report["estimator"] == "ensemble" and not report["llm_required"]
    assert not config.output_dir.exists()
    saved = tmp_path / "saved.json"
    saved.write_text(json.dumps(config.as_dict(), default=str))
    assert workflow.main(["--config", str(saved), "--tabular-only", "--preflight-only"]) == 0
    assert json.loads(capsys.readouterr().out)["llm_required"] is False
    assert not config.output_dir.exists()
    for flag in ([], ["--tabular-only"], ["--stage2-only"]):
        args = workflow.build_parser().parse_args(["--config", str(saved), *flag])
        values, directory = workflow._raw_config_from_args(args)
        assert workflow.compile_config(values, config_dir=directory) == config
    with pytest.raises(ValueError, match="fixed"):
        preflight_stage2(config, reselect=True)
    with pytest.raises(SystemExit):
        workflow.build_parser().parse_args(["--tabular-only", "--explicit-only"])


@pytest.mark.parametrize(
    "change,match",
    [("bad_roles", "unsupported roles"), ("oracle", "oracle")],
)
def test_reject_incompatible_config(tmp_path, change, match):
    raw, _, _ = inputs(tmp_path)
    if change == "no_features":
        del raw["tabular"]
    elif change == "empty_features":
        raw["tabular"]["features"] = []
    elif change == "endpoint":
        raw["stage2"] = {"endpoint": "http://unused.test/v1", "model": "unused"}
    elif change == "notes":
        raw["notes"] = {"path": "notes.csv"}
    elif change == "no_roles":
        del raw["tabular"]["features"][0]["roles"]
    elif change == "bad_roles":
        raw["tabular"]["features"][0]["roles"] = ["instrument"]
    elif change == "duplicate_names":
        raw["tabular"]["features"][1]["column"] = "age_years"
    elif change == "no_categories":
        del raw["tabular"]["features"][1]["categories"]
    elif change == "reserved_category":
        raw["tabular"]["features"][1]["categories"] = ["0", "documented_unknown"]
    elif change == "oracle":
        raw["tabular"]["features"][0]["column"] = "true_age"
    elif change == "stage1":
        raw["science"]["stage1_architectures"] = ["bow_nuisance"]
    else:
        raw["tabular"]["features"][0]["value_type"] = "ambiguous"
    with pytest.raises(ValueError, match=match):
        workflow.compile_config(raw, config_dir=tmp_path)


@pytest.mark.parametrize(
    "change,match",
    [("label_predictor", "distinct")],
)
def test_reject_invalid_table_before_writing(tmp_path, change, match):
    _, config, data = inputs(tmp_path)
    if change == "missing_column":
        data = data.drop(columns="ecog")
    elif change == "duplicate_id":
        data.loc[1, "id"] = data.loc[0, "id"]
    elif change == "missing_id":
        data.loc[1, "id"] = None
    elif change == "one_arm":
        data["t"] = 0
    elif change == "bad_outcome":
        data["y"] = data["y"].astype(float)
        data.loc[1, "y"] = np.inf
    elif change == "not_binary":
        data.loc[1, "y"] = 2
    elif change == "bad_number":
        data["Age (years)"] = data["Age (years)"].astype(object)
        data.loc[1, "Age (years)"] = "young"
    elif change == "infinite":
        data.loc[1, "Age (years)"] = np.inf
    elif change == "undeclared":
        data.loc[1, "ecog"] = 4
    elif change == "label_predictor":
        config = replace(config, treatment_column="ecog")
    elif change == "few_folds":
        config = replace(config, inner_folds=1)
    with pytest.raises(ValueError, match=match):
        prepare_tabular_inputs(config, data, persist=True)
    assert not config.output_dir.exists()


@pytest.mark.parametrize("change", ["corrupt"])
def test_resume_rejects_changed_inputs_even_if_complete(tmp_path, change):
    _, config, data = inputs(tmp_path)
    workflow.ResearchAllEvidenceWorkflow(config, component_runners={"stage2": lambda *_: {}}).run()
    manifest = config.output_dir / "stage2" / INPUT_FILE
    if change == "value":
        data.loc[5, "ecog"] = 2 - data.loc[5, "ecog"]
        data.to_csv(config.dataset, index=False)
    elif change == "order":
        data.iloc[::-1].to_csv(config.dataset, index=False)
    elif change == "schema":
        config = replace(
            config,
            tabular=replace(
                config.tabular,
                features=(
                    replace(config.tabular.features[0], roles=("effect_modifier",)),
                    config.tabular.features[1],
                ),
            ),
        )
    elif change == "seed":
        config = replace(config, seed=config.seed + 1)
    elif change == "missing":
        manifest.unlink()
    else:
        saved = json.loads(manifest.read_text())
        saved["splits"][0]["fit_row_ids"].reverse()
        manifest.write_text(json.dumps(saved))
    for action in (
        lambda: preflight_stage2(config),
        lambda: workflow.ResearchAllEvidenceWorkflow(config).run(),
    ):
        with pytest.raises(RuntimeError, match="inputs changed|provenance"):
            action()


@pytest.mark.parametrize(
    "outcome_type,nuisance",
    [("binary", "elastic_net")],
)
def test_real_ensemble_without_llms_and_resume(tmp_path, monkeypatch, outcome_type, nuisance):
    from tests.test_stage2_multi_model import small_policy
    from oci.inference import plain_handoff_stage2_analysis as analysis
    from oci.inference import plain_handoff_stage2 as extraction

    _, config, data = inputs(tmp_path, outcome_type=outcome_type)
    config = replace(
        config,
        stage2=replace(
            config.stage2,
            statistical_selection=small_policy(),
            nuisance_learner=nuisance,
            estimation_trees=12,
            super_learner=replace(config.stage2.super_learner, cv_folds=2),
        ),
    )
    monkeypatch.setattr(workflow, "_load_stage1_template", forbid)
    monkeypatch.setattr(workflow, "_load_neural_query_template", forbid)
    monkeypatch.setattr(workflow, "run_plain_handoff_stage2", forbid)
    monkeypatch.setattr(extraction, "launch_managed_vllm_servers", forbid)
    monkeypatch.setattr(PlainHandoffStage2, "__init__", forbid)
    runner = workflow.ResearchAllEvidenceWorkflow(config)
    assert runner.run()["status"] == "complete"
    directory = config.output_dir / "stage2"
    predictions = pd.read_csv(directory / "cross_fitted_predictions.csv")
    assert list(predictions["_oci_row_id"]) == list(range(len(data)))
    assert predictions[["aipw_score", "estimated_cate"]].notna().all().all()
    components = [
        f"estimated_cate_{name}"
        for name in ("causal_forest", "linear_interactions", "dina")
        if f"estimated_cate_{name}" in predictions
    ]
    np.testing.assert_allclose(predictions["estimated_cate"], predictions[components].mean(axis=1))
    if outcome_type == "continuous":
        assert "estimated_cate_dina" not in predictions
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["llm_requests"] == 0
    assert summary["causal_estimate"]["rows"] == len(data)
    heterogeneity = json.loads((directory / "heterogeneity_diagnostics.json").read_text())
    assert heterogeneity["clan"]["pooled"]["status"] == "ok"
    assert heterogeneity["balance"]["training"]["candidate_scope"] == "all_nuisance_candidates"
    for fold in (1, 2):
        outer = directory / f"outer_{fold:03d}"
        defs = json.loads((outer / "final_definitions.json").read_text())["features"]
        assert [item["name"] for item in defs] == ["age_years", "ecog"]
        assert defs[0]["roles"] == ["confounder"]
        assert defs[1]["roles"] == ["confounder", "effect_modifier"]
        assert not (outer / "extraction").exists()
        assert not (outer / "selection").exists()
    assert not (config.output_dir / "components").exists()
    assert not (config.output_dir / "handoff").exists()
    checkpoint = directory / "outer_001" / "estimation" / "complete.json"
    original = checkpoint.read_bytes(), checkpoint.stat().st_mtime_ns
    with monkeypatch.context() as reuse:
        # No fresh nuisance or effect fit is reached for identical input.
        reuse.setattr(analysis, "_outer_training_nuisances", forbid)
        assert runner.run()["status"] == "complete"
        preflight_stage2(config)
    assert (checkpoint.read_bytes(), checkpoint.stat().st_mtime_ns) == original
    # Completed runs still visit the scientific checkpoint when model settings change.
    changed = replace(config, stage2=replace(config.stage2, estimation_trees=16))
    assert workflow.ResearchAllEvidenceWorkflow(changed).run()["status"] == "complete"
    assert checkpoint.read_bytes() != original[0]


def test_binary_categories_and_training_only_imputation(tmp_path):
    from oci.inference.plain_handoff_stage2_analysis import _FeatureEncoder

    raw, _, data = inputs(tmp_path)
    raw["tabular"]["features"][1] = {
        "column": "ecog",
        "value_type": "binary",
        "roles": "effect_modifier",
    }
    data["ecog"] = data["ecog"].clip(upper=1)
    config = workflow.compile_config(raw, config_dir=tmp_path)
    measured, _ = prepare_tabular_inputs(config, data, persist=False)
    assert set(measured["ecog"].dropna()) == {"0", "1"}
    definitions = [feature.definition(1) for feature in config.tabular.features]
    encoder = _FeatureEncoder(definitions).fit(measured.iloc[2:20])
    encoded = encoder.transform(measured.iloc[:2])
    assert np.isfinite(encoded).all()
    assert encoded[0, 0] == 0  # Training median imputes missing age.
    assert encoded[0, 1] == 1  # Missingness is retained as a separate indicator.
    assert encoder.encodings[0][2][0] == measured.iloc[2:20]["age_years"].median()
