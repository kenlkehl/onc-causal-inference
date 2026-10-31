from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import oci.inference.research_all_evidence_workflow as all_evidence_workflow
import oci.inference.multi_model_forest_stage1 as multi_model_stage1

from oci.inference.plain_handoff_stage2_evidence import SUPPORTED_STAGE2_ARCHITECTURES
from oci.inference.stage1_architectures import DEFAULT_STAGE1_ARCHITECTURES
from oci.inference.stage1_checkpointing import row_id_identity
from oci.inference.research_all_evidence_workflow import (
    STAGE1_COMPONENT_ORDER as COMPONENT_ORDER,
    ResearchAllEvidenceWorkflow,
    Stage1RunContext,
    _required_stage2_architectures,
    _stage1_context_specs,
    compile_config,
    iter_stage1_handoff,
)


def _inputs(tmp_path: Path, *, components=COMPONENT_ORDER):
    dataset = tmp_path / "cohort.parquet"
    pd.DataFrame(
        {
            "patient_id": ["a", "b", "c", "d", "e", "f"],
            "clinical_text": ["one", "two", "three", "four", "five", "six"],
            "treatment_indicator": [0, 1, 0, 1, 0, 1],
            "outcome_indicator": [0, 0, 1, 1, 0, 1],
        }
    ).to_parquet(dataset, index=False)
    stage1_template = tmp_path / "stage1.json"
    stage1_template.write_text(
        json.dumps(
            {
                "config": {
                    "architecture": {
                        "model_type": "multi_model_forest",
                        "multi_model_forest": {
                            "feature_discovery_methods": [
                                "bow",
                                "htr",
                                "embedding_contrast",
                            ]
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    query_template = tmp_path / "queries.json"
    query_template.write_text(
        json.dumps({"query_inner_folds": 2}),
        encoding="utf-8",
    )
    raw = {
        "dataset": str(dataset),
        "output_dir": str(tmp_path / "output"),
        "stage1_template": str(stage1_template),
        "neural_query_template": str(query_template),
        "science": {
            "outer_folds": 2,
            "inner_folds": 2,
            "seed": 7,
            "stage1": {"training": {"epochs": 3}},
        },
        "run": {"components": list(components), "workers": 2},
        "models": {
            "htr": "test-htr-model",
            "embeddings": "test-embedding-model",
        },
    }
    return raw, compile_config(raw, config_dir=tmp_path)


def test_completed_components_are_reported_already_complete_without_revalidation(tmp_path):
    _raw, config = _inputs(tmp_path)
    calls: list[str] = []

    def runner(name):
        def run(_context, component_dir):
            calls.append(name)
            artifact = component_dir / "result.txt"
            artifact.write_text(name, encoding="utf-8")
            return {"artifacts": [str(artifact)]}

        return run

    runners = {name: runner(name) for name in COMPONENT_ORDER}
    workflow = ResearchAllEvidenceWorkflow(config, component_runners=runners)
    first = workflow.run()

    assert first["status"] == "complete"
    assert calls == list(COMPONENT_ORDER)
    for name in COMPONENT_ORDER:
        assert (workflow._component_dir(name) / "complete.json").is_file()

    calls.clear()
    second = workflow.run()
    assert second["status"] == "complete"
    assert calls == []
    assert all(
        record["status"] == "already_complete"
        for record in second["components"].values()
    )

    (workflow._component_dir("tfidf") / "complete.json").unlink()
    third = workflow.run()
    assert third["status"] == "complete"
    assert calls == ["tfidf"]


TEXT_MODEL_RERUN_CONTEXTS = 2 * (1 + 2)


def _text_model_rerun_fixture(tmp_path: Path, monkeypatch):
    """Launch a text_models-only run whose contexts fit two checkpointed units.

    Returns the study configuration, its path, the log of computed units, and
    a function that runs ``main`` on the configuration with extra flags.
    """

    raw, _config = _inputs(tmp_path, components=("text_models",))
    raw["run"]["workers"] = 1
    config_path = tmp_path / "study.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    computed: list[tuple[str, str, int]] = []

    def run_context(*, config, context, handoff_dir, checkpoint_store, **_unused):
        epochs = int(config.training.epochs)
        rows = {
            "fit": row_id_identity(context["train_idx"]),
            "heldout": row_id_identity(context["heldout_idx"]),
        }
        fitted = {}
        for unit in ("bow/fit", "htr/fit"):

            def compute(unit=unit):
                computed.append((str(context["scope_id"]), unit, epochs))
                return {"epochs": epochs}

            fitted[unit] = checkpoint_store.run(
                unit, compute, parameters={"epochs": epochs}, rows=rows
            ).value["epochs"]
        # Written beside the checkpoints, like worker_artifacts/**/predictions.parquet.
        artifact = Path(handoff_dir) / "worker_artifacts" / "fitted.json"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(json.dumps(fitted), encoding="utf-8")
        return {
            "outer_fold": context["outer_fold"],
            "inner_fold": context.get("inner_fold"),
            "scope": context["scope"],
            "fitted": fitted,
        }

    monkeypatch.setattr(multi_model_stage1, "run_multi_model_forest_handoff_context", run_context)
    monkeypatch.setattr(all_evidence_workflow, "_configure_logging", lambda *_args: None)

    def launch(*extra: str) -> None:
        assert all_evidence_workflow.main(["--config", str(config_path), *extra]) == 0

    return raw, config_path, computed, launch


def _text_model_rows(output_dir: Path) -> list[dict]:
    path = output_dir / "components" / "text_models" / "evidence.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_rerun_text_models_recomputes_checkpoint_units_whose_identity_changed(
    tmp_path: Path, monkeypatch
):
    raw, config_path, computed, launch = _text_model_rerun_fixture(tmp_path, monkeypatch)
    output_dir = tmp_path / "output"
    component_dir = output_dir / "components" / "text_models"

    def set_epochs(epochs: int) -> None:
        raw["science"]["stage1"]["training"]["epochs"] = epochs
        config_path.write_text(json.dumps(raw), encoding="utf-8")

    launch()
    computed.clear()
    set_epochs(4)
    launch()  # Completion markers record no identity; an ordinary resume skips.
    assert computed == []

    launch("--rerun", "text_models")
    assert len(computed) == 2 * TEXT_MODEL_RERUN_CONTEXTS
    assert {epochs for *_unused, epochs in computed} == {4}
    assert all(
        row["fitted"] == {"bow/fit": 4, "htr/fit": 4} for row in _text_model_rows(output_dir)
    )

    # The earlier settings' units are still valid and are restored, and each
    # context's evidence and side files are rebuilt from them rather than left
    # from the settings that ran last.
    computed.clear()
    set_epochs(3)
    launch("--rerun", "text_models")
    assert computed == []
    assert all(
        row["fitted"] == {"bow/fit": 3, "htr/fit": 3} for row in _text_model_rows(output_dir)
    )
    artifacts = sorted(component_dir.glob("*/worker_artifacts/fitted.json"))
    assert len(artifacts) == TEXT_MODEL_RERUN_CONTEXTS
    assert all(
        json.loads(path.read_text(encoding="utf-8")) == {"bow/fit": 3, "htr/fit": 3}
        for path in artifacts
    )


def test_legacy_handoff_references_sources_without_copy_or_architecture_expansion(
    tmp_path: Path,
):
    _raw, config = _inputs(tmp_path, components=())
    context = ResearchAllEvidenceWorkflow(config)._resolved_context()
    assert context.config.stage1_architectures is None
    source = context.component_dir("text_models") / "evidence.jsonl"
    source.parent.mkdir(parents=True)
    source_row = {
        "outer_fold": 1,
        "inner_fold": None,
        "scope": "full_outer_train",
        "importance": {"views": []},
    }
    source.write_text(json.dumps(source_row) + "\n", encoding="utf-8")
    handoff_dir = context.component_dir("handoff")
    handoff_dir.mkdir(parents=True)

    result = all_evidence_workflow._handoff_component(context, handoff_dir)

    assert result["rows"] == 1
    assert sorted(path.name for path in handoff_dir.iterdir()) == [
        "evidence.jsonl",
        "index.json",
    ]
    assert not (context.output_dir / "stage1_architectures").exists()
    index = json.loads((handoff_dir / "index.json").read_text(encoding="utf-8"))
    assert index["source_storage"] == "referenced_without_copy"
    assert index["architecture_materialization"] == "skipped_for_legacy_raw_handoff"
    assert index["sources"] == {
        "text_models": "../components/text_models/evidence.jsonl"
    }
    rows = list(iter_stage1_handoff(context.output_dir))
    assert rows == [
        {
            "source": "text_models",
            "outer_fold": 1,
            "inner_fold": None,
            "scope": "full_outer_train",
            "evidence": source_row,
        }
    ]


def test_other_evidence_families_reuse_visible_tfidf_splits(tmp_path):
    _raw, config = _inputs(tmp_path, components=("tfidf", "text_models"))
    split_path = config.output_dir / "components" / "tfidf" / "split_provenance.jsonl"
    split_path.parent.mkdir(parents=True)
    split_path.write_text(
        json.dumps(
            {
                "outer_fold": 1,
                "fit_row_ids": [5, 3, 1, 0],
                "heldout_row_ids": [2, 4],
                "inner_splits": [
                    {
                        "inner_fold": 1,
                        "fit_row_ids": [5, 1],
                        "heldout_row_ids": [3, 0],
                    },
                    {
                        "inner_fold": 2,
                        "fit_row_ids": [3, 0],
                        "heldout_row_ids": [5, 1],
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = pd.read_parquet(config.dataset)
    context = Stage1RunContext(
        config=config,
        dataset=dataset,
        applied_config=None,
        neural_query_config=None,
    )

    specs = _stage1_context_specs(context)

    assert [spec["scope_id"] for spec in specs] == [
        "outer_001_full",
        "outer_001_inner_001",
        "outer_001_inner_002",
    ]
    assert specs[0]["train_idx"].tolist() == [5, 3, 1, 0]
    assert specs[1]["train_idx"].tolist() == [5, 1]


@pytest.mark.parametrize("value", [0])
def test_invalid_contexts_per_gpu_fails_before_training(tmp_path, value):
    raw, _ = _inputs(tmp_path)
    raw["run"]["contexts_per_gpu"] = value
    with pytest.raises(ValueError, match="contexts_per_gpu must be a positive integer"):
        compile_config(raw, config_dir=tmp_path)


@pytest.mark.parametrize("enable_optional", [False])
def test_research_config_requires_only_its_enabled_stage2_architectures(
    tmp_path, enable_optional,
):
    raw, config = _inputs(tmp_path, components=())
    if enable_optional:
        raw["science"]["stage1"]["architecture"] = {
            "multi_model_forest": {
                "embedding_contrast": {"retrieval_tfidf_enabled": True},
                "tfidf_topic": {"orphan_ngram_enabled": True},
            }
        }
        config = compile_config(raw, config_dir=tmp_path)
    context = ResearchAllEvidenceWorkflow(config)._resolved_context()
    model = context.applied_config.architecture.multi_model_forest

    assert model.embedding_contrast.retrieval_tfidf_enabled is enable_optional
    assert model.tfidf_topic.orphan_ngram_enabled is enable_optional
    expected = (
        SUPPORTED_STAGE2_ARCHITECTURES if enable_optional else DEFAULT_STAGE1_ARCHITECTURES
    )
    assert _required_stage2_architectures(context) == expected


def _completed_stage2_reselection_fixture(tmp_path: Path):
    raw, _config = _inputs(tmp_path, components=())
    raw["run"]["mode"] = "stage2"
    raw["stage2"] = {
        "endpoint": "http://stage2.test/v1",
        "model": "primary-model",
        "extraction_llm": {
            "endpoint": "http://extract.test/v1",
            "model": "extractor-model",
            "workers": 1,
        },
        "max_review_rounds": 2,
    }
    config = compile_config(raw, config_dir=tmp_path)
    stage2_dir = config.output_dir / "stage2"
    stage2_dir.mkdir(parents=True)
    (stage2_dir / "model_identity.json").write_text(
        json.dumps(
            {
                "schema_version": "stage2_endpoint_model_identity_v1",
                "primary": {"selected_model": "primary-model"},
                "extraction": {"selected_model": "extractor-model"},
            }
        ),
        encoding="utf-8",
    )
    (stage2_dir / "complete.json").write_text(
        json.dumps({"status": "complete", "phase": "causal_estimation"}),
        encoding="utf-8",
    )
    for name in (
        "config.json",
        "summary.json",
        "features_by_outer_fold.jsonl",
        "cross_fitted_predictions.csv",
        "causal_estimate.json",
        "heterogeneity_diagnostics.json",
        "posthoc_oracle_ite_metrics.json",
    ):
        (stage2_dir / name).write_text("{}\n", encoding="utf-8")

    dataset = pd.read_parquet(config.dataset).reset_index(drop=True)
    for outer_fold in range(1, config.outer_folds + 1):
        outer_dir = stage2_dir / f"outer_{outer_fold:03d}"
        outer_dir.mkdir(parents=True)
        packets = [
            {
                "packet_id": f"outer-{outer_fold}",
                "outer_fold": outer_fold,
                "architecture": "tfidf_topic_contrast",
            }
        ]
        (outer_dir / "input_packets.jsonl").write_text(
            "".join(json.dumps(packet) + "\n" for packet in packets),
            encoding="utf-8",
        )
        source_features = [
            {
                "feature_id": f"outer_{outer_fold:03d}_feature_001",
                "name": "baseline_status",
                "value_type": "binary",
                "categories_or_unit": ["Present", "Absent"],
                "definition": "Pretreatment baseline status.",
                "extraction_guidance": "Extract only pretreatment status.",
                "roles": [],
            }
        ]
        (outer_dir / "feature_definitions.json").write_text(
            json.dumps({"outer_fold": outer_fold, "features": source_features}),
            encoding="utf-8",
        )
        definition_fingerprint = all_evidence_workflow._stage2_value_fingerprint(
            all_evidence_workflow._feature_definition_input_value(
                config=config.stage2,
                clinical_question=config.clinical_question,
                outer_fold=outer_fold,
                discovery_packets=packets,
                seed=config.seed,
            )
        )
        (outer_dir / "definitions_complete.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "evidence_input_fingerprint": definition_fingerprint,
                }
            ),
            encoding="utf-8",
        )

        fit_ids = [0, 1, 2, 3]
        matrix = pd.DataFrame(
            {
                "_oci_row_id": fit_ids,
                "baseline_status": ["Present", "Absent", "Present", "Absent"],
            }
        )
        matrix_path = outer_dir / "extraction" / "all_candidates_fit" / "extracted.csv"
        matrix_path.parent.mkdir(parents=True)
        matrix.to_csv(matrix_path, index=False)
        inner_splits = [
            {"fit_row_ids": [0, 1], "heldout_row_ids": [2, 3]},
            {"fit_row_ids": [2, 3], "heldout_row_ids": [0, 1]},
        ]
        treatment_outcome = dataset.iloc[fit_ids][
            [config.treatment_column, config.outcome_column]
        ].reset_index(drop=True)
        selection_value = {
            "schema_version": (
                all_evidence_workflow.STAGE2_ROLE_SELECTION_SCHEMA_VERSION
            ),
            "extracted_fit_fingerprint": all_evidence_workflow._frame_fingerprint(matrix),
            "treatment_outcome_fingerprint": (
                all_evidence_workflow._frame_fingerprint(treatment_outcome)
            ),
            "definitions": source_features,
            "inner_splits": inner_splits,
            "outcome_type": config.outcome_type,
            "preselection_matrix_path": "extraction/all_candidates_fit/extracted.csv",
        }
        selection_fingerprint = all_evidence_workflow._stage2_value_fingerprint(
            selection_value
        )
        selection_dir = outer_dir / "selection"
        selection_dir.mkdir()
        (selection_dir / "input.json").write_text(
            json.dumps(
                {**selection_value, "input_fingerprint": selection_fingerprint}
            ),
            encoding="utf-8",
        )
        (selection_dir / "complete.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "schema_version": selection_value["schema_version"],
                    "input_fingerprint": selection_fingerprint,
                }
            ),
            encoding="utf-8",
        )
        (selection_dir / "elastic_net_selection.json").write_text(
            json.dumps({"schema_version": selection_value["schema_version"]}),
            encoding="utf-8",
        )
        convergence = {
            "schema_version": "stage2_ontology_supervisor_convergence_v1",
            "status": "converged",
            "converged": True,
            "review_rounds": 1,
            "maximum_review_rounds": 2,
            "continued_with_latest_ontology": False,
            "history": [{"round": 1}],
        }
        (outer_dir / "final_definitions.json").write_text(
            json.dumps(
                {
                    "schema_version": selection_value["schema_version"],
                    "features": source_features,
                    "measurement_dependencies": source_features,
                    "all_candidate_features": 1,
                    "review_rounds": 1,
                    "evaluation_rounds": 1,
                    "review_converged": True,
                    "review_convergence": convergence,
                    "ontology_refinement_rounds": 0,
                    "harmonization_validation_fallbacks": [],
                }
            ),
            encoding="utf-8",
        )
        (outer_dir / "complete.json").write_text(
            json.dumps({"status": "complete", "phase": "causal_estimation"}),
            encoding="utf-8",
        )
        estimation_dir = outer_dir / "estimation"
        estimation_dir.mkdir()
        (estimation_dir / "complete.json").write_text(
            json.dumps({"status": "complete"}),
            encoding="utf-8",
        )
        heldout_dir = outer_dir / "extraction" / "heldout"
        heldout_dir.mkdir()
        pd.DataFrame(
            {
                "_oci_row_id": [4, 5],
                "baseline_status": ["Present", "Absent"],
            }
        ).to_csv(heldout_dir / "extracted.csv", index=False)
        (heldout_dir / "complete.json").write_text("{}\n", encoding="utf-8")
        interpretation_dir = outer_dir / "interpretations" / "cached"
        interpretation_dir.mkdir(parents=True)
        (interpretation_dir / "complete.json").write_text("{}\n", encoding="utf-8")
    return config


def test_stage2_reselection_archives_downstream_and_freezes_preselection(tmp_path):
    config = _completed_stage2_reselection_fixture(tmp_path)

    state = all_evidence_workflow.prepare_stage2_reselection(config=config)

    stage2_dir = config.output_dir / "stage2"
    assert state["status"] == "prepared"
    assert not (stage2_dir / "complete.json").exists()
    archive_dir = stage2_dir / state["archive_path"] / "artifacts"
    assert (archive_dir / "complete.json").is_file()
    for outer_fold in range(1, config.outer_folds + 1):
        outer_dir = stage2_dir / f"outer_{outer_fold:03d}"
        assert (outer_dir / "feature_definitions.json").is_file()
        assert (outer_dir / "definitions_complete.json").is_file()
        assert (outer_dir / "interpretations" / "cached" / "complete.json").is_file()
        assert (
            outer_dir / "extraction" / "all_candidates_fit" / "extracted.csv"
        ).is_file()
        assert (outer_dir / "preselection" / "complete.json").is_file()
        snapshot = json.loads(
            (outer_dir / "preselection" / "input.json").read_text(encoding="utf-8")
        )
        heldout_cache = snapshot["heldout_measurement_cache"]
        assert heldout_cache["schema_version"] == (
            all_evidence_workflow.HELDOUT_MEASUREMENT_CACHE_SCHEMA_VERSION
        )
        assert heldout_cache["heldout_row_ids"] == [4, 5]
        assert heldout_cache["measurement_definitions"][0]["name"] == (
            "baseline_status"
        )
        assert not (outer_dir / "selection").exists()
        assert (archive_dir / outer_dir.name / "selection").is_dir()
        assert (
            archive_dir
            / outer_dir.name
            / "extraction"
            / "heldout"
            / "extracted.csv"
        ).is_file()

    (stage2_dir / "config.json").write_text(
        json.dumps({"new_reselection_checkpoint": True}), encoding="utf-8"
    )
    resumed = all_evidence_workflow.prepare_stage2_reselection(config=config)
    assert resumed["reselection_id"] == state["reselection_id"]
    assert json.loads((stage2_dir / "config.json").read_text(encoding="utf-8")) == {
        "new_reselection_checkpoint": True
    }
    assert len(list((stage2_dir / "reselection_archives").iterdir())) == 1

    (stage2_dir / "complete.json").write_text(
        json.dumps({"status": "complete", "phase": "causal_estimation"}),
        encoding="utf-8",
    )
    finalized = all_evidence_workflow.finalize_stage2_reselection(config=config)
    assert finalized is not None
    assert finalized["status"] == "complete"
    repeated = all_evidence_workflow.prepare_stage2_reselection(config=config)
    assert repeated["reselection_id"] == state["reselection_id"]
    assert len(list((stage2_dir / "reselection_archives").iterdir())) == 1


_SENTINEL_KEYS = {
    "stage2": "sk-sentinel-stage2-primary",
    "extraction": "sk-sentinel-stage2-extraction",
    "extraction_runtime": "sk-sentinel-stage2-extraction-runtime",
    "consolidation_runtime": "sk-sentinel-stage2-consolidation-runtime",
    "stage1_vllm": "sk-sentinel-stage1-vllm",
    "stage1_agent": "sk-sentinel-stage1-agent",
}


def _api_key_values(value):
    from oci.extraction.llm_routing import is_api_key_field

    if isinstance(value, dict):
        for key, child in value.items():
            if is_api_key_field(key):
                yield child
            else:
                yield from _api_key_values(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _api_key_values(child)


def _run_with_sentinel_keys(tmp_path):
    raw, _config = _inputs(tmp_path)
    raw["science"]["stage1"] = {
        "training": {"epochs": 3},
        "explicit_features": {"vllm_api_key": _SENTINEL_KEYS["stage1_vllm"]},
        "architecture": {
            "agentic_feature_search": {"agent_api_key": _SENTINEL_KEYS["stage1_agent"]}
        },
    }
    raw["stage2"] = {
        "endpoint": "http://stage2.test/v1",
        "model": "primary-model",
        "api_key": _SENTINEL_KEYS["stage2"],
        "extraction_llm": {
            "endpoint": "http://extract.test/v1",
            "model": "extractor-model",
            "api_key": _SENTINEL_KEYS["extraction"],
            "runtime_endpoint": "http://extract-runtime.test/v1",
            "runtime_model": "replacement-extractor",
            "runtime_api_key": _SENTINEL_KEYS["extraction_runtime"],
        },
        "selection_consolidation": {
            "runtime_llm_endpoint": "http://consolidate.test/v1",
            "runtime_llm_model": "replacement-primary",
            "runtime_llm_api_key": _SENTINEL_KEYS["consolidation_runtime"],
        },
    }
    config = compile_config(raw, config_dir=tmp_path)
    ResearchAllEvidenceWorkflow(config)._resolved_context()
    return config


def test_persisted_configs_never_contain_api_keys(tmp_path):
    from oci.config import ExperimentConfig
    from oci.experiments.runner import ExperimentRunner

    config = _run_with_sentinel_keys(tmp_path)
    sentinels = set(_SENTINEL_KEYS.values())
    # Every key field of the Stage 2 tree (and the Stage 1 overrides) and of the
    # Stage 1 model configuration carries a sentinel, so a new key field would
    # be caught below.
    assert sorted(_api_key_values(config.as_dict())) == sorted(sentinels)
    applied = all_evidence_workflow._load_stage1_template(config)
    experiment = ExperimentConfig.from_dict(
        {"output_dir": str(config.output_dir / "oci_run"), "device": "cpu",
         "applied_inference": applied}
    )
    assert sorted(_api_key_values(experiment.to_dict())) == sorted(
        [_SENTINEL_KEYS["stage1_vllm"], _SENTINEL_KEYS["stage1_agent"]]
    )
    # The other persisted copies: Stage 2's config.json and `oci run`'s config.json.
    all_evidence_workflow._write_json(
        config.output_dir / "stage2" / "config.json", config.stage2.public_dict()
    )
    ExperimentRunner(experiment)._save_config()

    written = sorted(path for path in config.output_dir.rglob("*") if path.is_file())
    names = {path.relative_to(config.output_dir).as_posix() for path in written}
    assert {
        "run_config.json",
        "resolved_stage1_model_config.json",
        "resolved_neural_query_config.json",
        "stage2/config.json",
        "oci_run/config.json",
    } <= names
    for path in written:
        text = path.read_text(encoding="utf-8", errors="replace")
        leaked = sorted(sentinel for sentinel in sentinels if sentinel in text)
        assert not leaked, f"{path} contains {leaked}"
    saved = json.loads((config.output_dir / "run_config.json").read_text(encoding="utf-8"))
    assert set(_api_key_values(saved)) == {"<redacted>"}


def test_stage2_only_runs_from_saved_handoff_and_resumes(tmp_path, monkeypatch):
    raw, _config = _inputs(tmp_path, components=())
    raw["run"]["mode"] = "stage2"
    raw["stage2"] = {
        "endpoint": "http://stage2.test/v1",
        "model": "test-model",
    }
    config = compile_config(raw, config_dir=tmp_path)
    handoff = config.output_dir / "handoff" / "evidence.jsonl"
    handoff.parent.mkdir(parents=True)
    handoff.write_text('{"source":"tfidf"}\n', encoding="utf-8")
    (handoff.parent / "complete.json").write_text("{}", encoding="utf-8")
    workflow = ResearchAllEvidenceWorkflow(config)

    calls = []

    def run_stage2(*, output_dir, **kwargs):
        calls.append(kwargs)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "result.txt").write_text("done", encoding="utf-8")
        (output_dir / "causal_estimate.json").write_text("{}", encoding="utf-8")
        (output_dir / "cross_fitted_predictions.csv").write_text(
            "_oci_row_id,aipw_score\n", encoding="utf-8"
        )
        (output_dir / "posthoc_oracle_ite_metrics.json").write_text(
            '{"available":false}\n', encoding="utf-8"
        )
        return {
            "phase": "causal_estimation",
            "artifacts": [str(output_dir / "result.txt")],
        }

    monkeypatch.setattr(all_evidence_workflow, "run_plain_handoff_stage2", run_stage2)

    first = workflow.run()
    assert first["mode"] == "stage2"
    assert (config.output_dir / "stage2" / "result.txt").read_text() == "done"
    assert (config.output_dir / "stage2" / "complete.json").is_file()

    second = workflow.run()
    assert second["components"]["stage2"]["status"] == "already_complete"
    assert len(calls) == 1


def test_architecture_selector_cannot_be_added_to_existing_legacy_output(tmp_path):
    raw, _config = _inputs(tmp_path)
    completion = tmp_path / "output" / "components" / "tfidf" / "complete.json"
    completion.parent.mkdir(parents=True)
    completion.write_text('{"status":"complete"}', encoding="utf-8")
    raw["science"]["stage1_architectures"] = ["tfidf_topics"]
    selected = compile_config(raw, config_dir=tmp_path)

    with pytest.raises(ValueError, match="fresh output directory"):
        ResearchAllEvidenceWorkflow(selected)
