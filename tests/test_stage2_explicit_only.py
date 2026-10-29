"""Explicit ontologies enter the real Stage 2 pipeline without Stage 1 artifacts."""

from dataclasses import replace
from contextlib import contextmanager
import json
import re

import numpy as np
import pandas as pd
import pytest

from oci.inference import research_all_evidence_workflow as workflow
from oci.inference.plain_handoff_stage2 import PlainHandoffStage2, run_plain_handoff_stage2
from oci.inference.stage2_explicit_inputs import INPUT_FILE
from oci.inference.stage2_preflight import preflight_stage2


def inputs(tmp_path, *, outcome_type="binary"):
    rng = np.random.default_rng(28)
    n = 80
    ecog = np.tile(np.arange(4), n // 4)
    treatment = rng.binomial(1, 0.5, n)
    outcome = (
        rng.binomial(1, 0.3 + 0.1 * treatment + 0.05 * ecog)
        if outcome_type == "binary"
        else ecog + treatment * (1 + 0.2 * ecog) + rng.normal(size=n)
    )
    data = pd.DataFrame(
        {
            "patient_id": [f"p{i}" for i in range(n)],
            "clinical_text": [f"Pretreatment ECOG {value}." for value in ecog],
            "treatment_indicator": treatment,
            "outcome_indicator": outcome,
        }
    )
    path = tmp_path / "cohort.parquet"
    data.to_parquet(path, index=False)
    raw = {
        "dataset": str(path),
        "output_dir": str(tmp_path / "run"),
        # These deliberately do not exist. This mode must never read a Stage 1 template.
        "stage1_template": str(tmp_path / "no-stage1.json"),
        "neural_query_template": str(tmp_path / "no-neural-queries.json"),
        "science": {"outer_folds": 2, "inner_folds": 2, "seed": 7, "outcome_type": outcome_type},
        "run": {"mode": "explicit_only"},
        "stage2": {
            "endpoint": "http://primary.test/v1",
            "model": "test-primary",
            "extraction_llm": {"endpoint": "http://extract.test/v1", "model": "test-extractor"},
            "explicit_features": [
                {
                    "name": "ecog",
                    "description": "Pretreatment ECOG performance status.",
                    "value_type": "ordinal",
                    "categories_or_unit": ["0", "1", "2", "3"],
                    "measurement_definition": "Use the explicitly documented pretreatment ECOG score.",
                    "missing_value_rule": "Return null if undocumented.",
                    "roles": ["confounder", "effect_modifier"],
                }
            ],
        },
    }
    return raw, workflow.compile_config(raw, config_dir=tmp_path), data


def forbid(*args, **kwargs):
    pytest.fail("explicit-only execution attempted Stage 1 or discovery")


@pytest.mark.parametrize("change", ["cohort"])
def test_completed_run_rejects_input_or_provenance_drift(tmp_path, change):
    _, config, data = inputs(tmp_path)
    workflow.ResearchAllEvidenceWorkflow(
        config,
        component_runners={"stage2": lambda *_: {}},
    ).run()
    if change == "cohort":
        data.loc[0, "clinical_text"] = "Changed baseline record."
        data.to_parquet(config.dataset, index=False)
    elif change == "order":
        data.iloc[::-1].to_parquet(config.dataset, index=False)
    elif change == "ontology":
        feature = replace(config.stage2.explicit_features[0], roles=("confounder",))
        config = replace(config, stage2=replace(config.stage2, explicit_features=(feature,)))
    elif change == "seed":
        config = replace(config, seed=config.seed + 1)
    else:
        manifest = config.output_dir / "stage2" / INPUT_FILE
        if change == "missing":
            manifest.unlink()
        else:
            value = json.loads(manifest.read_text())
            value["splits"][0]["heldout_row_ids"].reverse()
            manifest.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="inputs changed|provenance"):
        workflow.ResearchAllEvidenceWorkflow(config).run()
    with pytest.raises(RuntimeError, match="inputs changed|provenance"):
        preflight_stage2(config)


def test_modes_cannot_share_output_directory(tmp_path):
    raw, config, _ = inputs(tmp_path)
    workflow.ResearchAllEvidenceWorkflow(
        config,
        component_runners={"stage2": lambda *_: {}},
    ).run()
    raw["run"]["mode"] = "stage2"
    ordinary = workflow.compile_config(raw, config_dir=tmp_path)
    with pytest.raises(ValueError, match="cannot switch"):
        workflow.ResearchAllEvidenceWorkflow(ordinary)
    raw["output_dir"] = str(tmp_path / "legacy")
    legacy = workflow.compile_config(raw, config_dir=tmp_path)
    legacy.output_dir.mkdir()
    (legacy.output_dir / "run_config.json").write_text(json.dumps(legacy.as_dict(), default=str))
    raw["run"]["mode"] = "explicit_only"
    with pytest.raises(ValueError, match="cannot switch"):
        workflow.ResearchAllEvidenceWorkflow(workflow.compile_config(raw, config_dir=tmp_path))


@pytest.mark.parametrize(
    "outcome_type,serving",
    [("binary", "external")],
)
def test_explicit_only_extracts_estimates_and_reuses_measurements(
    tmp_path,
    monkeypatch,
    outcome_type,
    serving,
):
    from tests.stage2_prompt_spy import install, prompt_inputs
    from tests.test_plain_handoff_stage2 import _CharacterChatTokenizer
    from tests.test_stage2_multi_model import small_policy
    from oci.inference.stage2_multi_model_config import ModifierSelectionConfig

    install(monkeypatch)
    _, config, data = inputs(tmp_path, outcome_type=outcome_type)
    policy = small_policy()
    policy = replace(
        policy,
        multi_model=replace(
            policy.multi_model,
            modifier_selection=ModifierSelectionConfig(estimator_policy="linear_interactions"),
        ),
    )
    config = replace(
        config,
        stage2=replace(
            config.stage2,
            statistical_selection=policy,
            nuisance_learner="elastic_net",
            estimation_trees=10,
            min_informative_patients=2,
            max_review_rounds=1,
        ),
    )
    launches = []
    if serving == "managed":
        from oci.inference import plain_handoff_stage2 as stage2_module
        from oci.inference.vllm_server_pool import ManagedVLLMConfig

        config = replace(
            config,
            stage2=replace(
                config.stage2,
                endpoint="",
                vllm=ManagedVLLMConfig(server_count=1, gpus=("cuda:0",)),
                extraction_llm=replace(
                    config.stage2.extraction_llm,
                    endpoint="",
                    vllm=ManagedVLLMConfig(server_count=1, gpus=("cuda:1",), base_port=8110),
                ),
            ),
        )

        @contextmanager
        def launch(**kwargs):
            launches.append(kwargs["model"])
            yield ("http://managed.test/v1",)

        monkeypatch.setattr(stage2_module, "launch_managed_vllm_servers", launch)
    extraction_calls = []

    def extract(messages, request_config):
        body = prompt_inputs(messages)
        assert body["job"] == "extract_stage2_patient_variables"
        assert request_config.model == "test-extractor"
        assert len(body["patients"]) == 1
        patient = body["patients"][0]
        value = re.search(r"ECOG (\d)", patient["text"]).group(1)
        extraction_calls.append(patient["row_id"])
        return json.dumps({body["features"][0]["name"]: value})

    monkeypatch.setattr(workflow, "_load_stage1_template", forbid)
    monkeypatch.setattr(PlainHandoffStage2, "_load_or_compile_evidence", forbid)
    monkeypatch.setattr(PlainHandoffStage2, "_interpret_batch", forbid)
    real_run = run_plain_handoff_stage2

    def run_with_test_llms(**kwargs):
        return real_run(
            **kwargs,
            completion=forbid,
            extraction_completion=extract,
            extraction_tokenizer=_CharacterChatTokenizer(),
        )

    monkeypatch.setattr(workflow, "run_plain_handoff_stage2", run_with_test_llms)
    runner = workflow.ResearchAllEvidenceWorkflow(config)
    runner.run()
    if serving == "managed":
        assert launches == ["test-extractor"]
    stage2 = config.output_dir / "stage2"
    estimates = json.loads((stage2 / "causal_estimate.json").read_text())
    assert estimates["rows"] == len(data)
    predictions = pd.read_csv(stage2 / "cross_fitted_predictions.csv")
    assert sorted(predictions["_oci_row_id"]) == list(range(len(data)))
    assert predictions["aipw_score"].notna().all()
    assert (stage2 / "heterogeneity_diagnostics.json").is_file()
    assert extraction_calls
    for fold in (1, 2):
        outer = stage2 / f"outer_{fold:03d}"
        feature = json.loads((outer / "feature_definitions.json").read_text())["features"][0]
        assert feature["configured_explicit_feature"] is True
        assert feature["roles"] == ["confounder", "effect_modifier"]
        assert (
            feature["measurement_definition"]
            == config.stage2.explicit_features[0].measurement_definition
        )
        assert (outer / "input_packets.jsonl").read_text() == ""
    before = len(extraction_calls)
    workflow._invalidate_component_for_rerun(runner, "stage2")
    runner.run()
    assert len(extraction_calls) == before
    preflight_stage2(config)
    if outcome_type == "binary" and serving == "external":
        changed_policy = replace(
            policy,
            multi_model=replace(
                policy.multi_model,
                modifier_selection=replace(policy.multi_model.modifier_selection, q=0.2),
            ),
        )
        changed = replace(
            config, stage2=replace(config.stage2, statistical_selection=changed_policy)
        )
        assert preflight_stage2(changed, reselect=True)["migration"]["dry_run"]
        workflow.prepare_stage2_reselection(config=changed)
        workflow.ResearchAllEvidenceWorkflow(changed).run()
        workflow.finalize_stage2_reselection(config=changed)
        assert len(extraction_calls) == before
        assert (stage2 / INPUT_FILE).is_file()
    assert not (config.output_dir / "components").exists()
    assert not (config.output_dir / "handoff").exists()
