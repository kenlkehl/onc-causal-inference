from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import oci.inference.research_all_evidence_workflow as all_evidence_workflow
import oci.inference.multi_model_forest_stage1 as multi_model_stage1

from oci.inference.embedding_contrast_discovery import (
    EmbeddingContrastEvidenceGenerator,
    _retrieval_tfidf_terms,
)
from oci.inference.all_evidence_fusion import HTR_NEURAL, MATCHED_PAIR_UPLIFT
from oci.inference.plain_handoff_stage2_evidence import SUPPORTED_STAGE2_ARCHITECTURES
from oci.inference.research_all_evidence_workflow import (
    COMPONENT_ORDER,
    ResearchAllEvidenceWorkflow,
    Stage1RunContext,
    _context_execution_lanes,
    _lane_cpu_worker_budgets,
    _raw_config_from_args,
    _required_stage2_architectures,
    _stage1_context_specs,
    build_parser,
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


def test_multi_model_forest_defaults_to_the_full_stage1_architecture_set():
    from oci.config import MultiModelForestConfig

    config = MultiModelForestConfig()

    assert config.feature_discovery_methods == ["bow", "htr", "embedding_contrast"]
    assert config.bow_discovery_enabled is True
    assert config.htr_evidence_enabled is True
    assert config.embedding_contrast.enabled is True
    assert config.matched_pair_uplift_enabled is True
    assert config.matched_pair_bow_enabled is True
    assert config.matched_pair_htr_enabled is True


def test_generic_applied_dispatch_rejects_retired_multi_model_orchestration(tmp_path):
    from oci.inference.applied import run_applied_inference

    config = SimpleNamespace(
        architecture=SimpleNamespace(model_type="multi_model_forest")
    )
    with pytest.raises(RuntimeError, match="ResearchAllEvidenceWorkflow"):
        run_applied_inference(
            dataset=pd.DataFrame(),
            config=config,
            output_path=tmp_path / "predictions.parquet",
            device=multi_model_stage1.torch.device("cpu"),
        )


def test_embedding_retrieval_tfidf_producer_keeps_both_semantic_tails():
    rows = _retrieval_tfidf_terms(
        [
            {"text": "oxygen dependence severe dyspnea"},
            {"text": "oxygen requirement at baseline"},
        ],
        [
            {"text": "active runner excellent stamina"},
            {"text": "daily exercise without limitation"},
        ],
        ngram_range=(1, 2),
        max_features=100,
        top_terms_per_side=8,
    )

    assert rows
    assert {row["polarity"] for row in rows} == {"positive", "negative"}
    assert any(row["term"] == "oxygen" and row["tfidf_contrast"] > 0 for row in rows)
    assert any(row["term"] == "exercise" and row["tfidf_contrast"] < 0 for row in rows)


def test_compile_config_keeps_run_and_science_in_one_file(tmp_path):
    _raw, config = _inputs(tmp_path, components=("text_models",))

    assert config.dataset == tmp_path / "cohort.parquet"
    assert config.output_dir == tmp_path / "output"
    assert config.outer_folds == 2
    assert config.inner_folds == 2
    assert config.seed == 7
    assert config.workers == 2
    assert config.components == ("text_models",)
    assert config.stage1_overrides["training"]["epochs"] == 3


def test_resolved_context_uses_multi_model_forest_embedding_config(tmp_path):
    raw, _config = _inputs(tmp_path, components=())
    raw["science"]["stage1"]["architecture"] = {
        "multi_model_forest": {
            "embedding_contrast": {"max_chunks": 128},
        }
    }
    config = compile_config(raw, config_dir=tmp_path)

    context = ResearchAllEvidenceWorkflow(config)._resolved_context()
    architecture = context.applied_config.architecture
    generator = EmbeddingContrastEvidenceGenerator(
        config=context.applied_config,
        output_dir=tmp_path / "embedding",
    )

    assert architecture.model_type == "multi_model_forest"
    assert architecture.multi_model_forest.embedding_contrast.max_chunks == 128
    assert generator.embedding_config is architecture.multi_model_forest.embedding_contrast
    assert generator.embedding_config.cache_dir == str(
        config.output_dir / "components" / "embedding_cache" / "cache"
    )
    assert generator.embedding_config.model_name == "test-embedding-model"


def test_disable_htr_keeps_bow_and_embedding_without_scheduling_htr(
    tmp_path,
    monkeypatch,
):
    raw, _config = _inputs(tmp_path, components=("text_models",))
    raw["science"]["htr_enabled"] = False
    raw["science"]["stage1"]["architecture"] = {
        "multi_model_forest": {
            "feature_discovery_methods": ["bow", "htr", "embedding_contrast"],
        }
    }
    config = compile_config(raw, config_dir=tmp_path)
    context = ResearchAllEvidenceWorkflow(config)._resolved_context()
    model_config = context.applied_config.architecture.multi_model_forest

    assert config.htr_enabled is False
    assert model_config.feature_discovery_methods == ["bow", "embedding_contrast"]
    assert model_config.bow_discovery_enabled is True
    assert model_config.embedding_contrast.enabled is True
    assert model_config.htr_evidence_enabled is False
    assert model_config.htr_evidence_disable_reason == "disabled by research workflow option"
    assert model_config.matched_pair_htr_enabled is False
    required = _required_stage2_architectures(context)
    assert HTR_NEURAL not in required
    assert MATCHED_PAIR_UPLIFT in required

    def run_contexts(**kwargs):
        assert kwargs["plan"].htr_enabled is False
        assert kwargs["config"].architecture.multi_model_forest.htr_evidence_enabled is False
        return [{"outer_fold": 1, "scope": "full_outer_train"}]

    monkeypatch.setattr(
        multi_model_stage1,
        "run_multi_model_forest_handoff_contexts",
        run_contexts,
    )
    row = all_evidence_workflow._run_one_text_model_context(
        dataset=context.dataset,
        applied_config=context.applied_config,
        spec={
            "scope_id": "outer_001_full",
            "fold_key": 1,
            "outer_fold": 1,
            "scope": "full_outer_train",
        },
        context_dir=tmp_path / "no_htr_context",
        device="cpu",
        cpu_workers=1,
    )

    assert row == {"outer_fold": 1, "scope": "full_outer_train"}


def test_text_model_context_serializes_nonfinite_diagnostics_as_null(
    tmp_path,
    monkeypatch,
    caplog,
):
    raw, _config = _inputs(tmp_path, components=("text_models",))
    config = compile_config(raw, config_dir=tmp_path)
    context = ResearchAllEvidenceWorkflow(config)._resolved_context()

    def run_contexts(**_kwargs):
        return [
            {
                "outer_fold": 1,
                "scope": "full_outer_train",
                "metrics": {
                    "undefined_correlation": float("nan"),
                    "infinite_statistic": np.float64("inf"),
                },
                "importance": {
                    "scores": np.asarray([1.25, float("-inf")]),
                },
            }
        ]

    monkeypatch.setattr(
        multi_model_stage1,
        "run_multi_model_forest_handoff_contexts",
        run_contexts,
    )
    context_dir = tmp_path / "nonfinite_context"
    row = all_evidence_workflow._run_one_text_model_context(
        dataset=context.dataset,
        applied_config=context.applied_config,
        spec={
            "scope_id": "outer_001_full",
            "fold_key": 1,
            "outer_fold": 1,
            "scope": "full_outer_train",
        },
        context_dir=context_dir,
        device="cpu",
        cpu_workers=1,
    )

    assert row["metrics"] == {
        "undefined_correlation": None,
        "infinite_statistic": None,
    }
    assert row["importance"]["scores"] == [1.25, None]
    assert json.loads((context_dir / "evidence.json").read_text(encoding="utf-8")) == row
    assert (context_dir / "complete.json").is_file()
    assert "converted 3 non-finite evidence value(s) to JSON null" in caplog.text


def test_text_model_context_resumes_from_final_substep_checkpoint(tmp_path, monkeypatch):
    raw, _config = _inputs(tmp_path, components=("text_models",))
    config = compile_config(raw, config_dir=tmp_path)
    context = ResearchAllEvidenceWorkflow(config)._resolved_context()
    calls = 0

    def run_contexts(**_kwargs):
        nonlocal calls
        calls += 1
        return [{"outer_fold": 1, "scope": "full_outer_train", "metrics": {}}]

    monkeypatch.setattr(
        multi_model_stage1,
        "run_multi_model_forest_handoff_contexts",
        run_contexts,
    )
    context_dir = tmp_path / "resumable_context"
    spec = {
        "scope_id": "outer_001_full",
        "fold_key": 1,
        "outer_fold": 1,
        "scope": "full_outer_train",
        "train_idx": np.asarray([0, 1, 2, 3]),
        "heldout_idx": np.asarray([4, 5]),
    }

    first = all_evidence_workflow._run_one_text_model_context(
        dataset=context.dataset,
        applied_config=context.applied_config,
        spec=spec,
        context_dir=context_dir,
        device="cpu",
        cpu_workers=1,
    )
    (context_dir / "complete.json").unlink()
    second = all_evidence_workflow._run_one_text_model_context(
        dataset=context.dataset,
        applied_config=context.applied_config,
        spec=spec,
        context_dir=context_dir,
        device="cpu",
        cpu_workers=1,
    )

    assert calls == 1
    assert second == first
    progress = json.loads(
        (
            context_dir
            / "checkpoints"
            / "v1"
            / "checkpoint_progress.json"
        ).read_text(encoding="utf-8")
    )
    assert progress["units"]["assembly/handoff_context"]["status"] == "reused"


def test_text_model_handoff_row_omits_empty_htr_evidence():
    from oci.inference.multi_model_agentic_forest import (
        _agentic_discovery_handoff_row,
        _build_evidence_digest_agent_context,
    )

    result = {
        "metrics": {"feature_discovery_methods": ["bow", "embedding_contrast"]},
        "importance": {},
        "embedding_contrast_evidence": {"available": True},
        "htr_evidence": {},
        "context": {"feature_discovery_methods": ["bow", "embedding_contrast"]},
    }

    row = _agentic_discovery_handoff_row(
        result,
        fold_key=1,
        outer_fold=1,
        scope="full_outer_train",
        n_rows=10,
    )

    assert "htr_evidence" not in row
    assert row["embedding_contrast_evidence"] == {"available": True}

    context = _build_evidence_digest_agent_context(
        outer_fold=1,
        feature_discovery_methods=["bow", "embedding_contrast"],
        max_proposals=10,
        clinical_question="test",
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        outcome_type="binary",
        current_features=[],
        metrics={},
        importance={},
        clinical_text_examples=[],
        embedding_evidence={"available": True},
        htr_evidence={},
    )
    digest = context["evidence_digest"]
    assert "htr_blurbs" not in digest["confounders"]
    assert "htr_blurbs" not in digest["effect_modifiers"]
    assert all(not key.startswith("htr_") for key in digest["prompt_compaction"])


def test_simple_runner_uses_processes_for_tfidf_contexts_by_default(tmp_path):
    raw, config = _inputs(tmp_path, components=("tfidf",))

    default_mapping = all_evidence_workflow._load_stage1_template(config)
    default_multi_model = default_mapping["architecture"]["multi_model_forest"]

    assert default_multi_model["outer_parallel_backend"] == "processes"
    assert default_multi_model["cpus_total"] == config.workers

    raw["science"]["stage1"]["architecture"] = {
        "multi_model_forest": {"outer_parallel_backend": "threads"}
    }
    overridden = compile_config(raw, config_dir=tmp_path)
    overridden_mapping = all_evidence_workflow._load_stage1_template(overridden)

    assert (
        overridden_mapping["architecture"]["multi_model_forest"][
            "outer_parallel_backend"
        ]
        == "threads"
    )


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


def test_partial_component_is_reentered_after_interruption(tmp_path):
    _raw, config = _inputs(
        tmp_path,
        components=("embedding_cache", "text_models"),
    )
    calls: list[str] = []

    def embedding(_context, component_dir):
        calls.append("embedding_cache")
        (component_dir / "cache.npy").write_bytes(b"cache")

    def fail_text(_context, component_dir):
        calls.append("text_models_failed")
        (component_dir / "partial.txt").write_text("keep me", encoding="utf-8")
        raise RuntimeError("interrupted model")

    workflow = ResearchAllEvidenceWorkflow(
        config,
        component_runners={
            "embedding_cache": embedding,
            "text_models": fail_text,
        },
    )
    with pytest.raises(RuntimeError, match="interrupted model"):
        workflow.run()

    progress = json.loads(workflow.progress_path.read_text(encoding="utf-8"))
    assert progress["status"] == "failed"
    assert (workflow._component_dir("embedding_cache") / "complete.json").is_file()
    assert not (workflow._component_dir("text_models") / "complete.json").exists()

    def finish_text(_context, component_dir):
        calls.append("text_models_resumed")
        assert (component_dir / "partial.txt").read_text(encoding="utf-8") == "keep me"
        (component_dir / "result.txt").write_text("done", encoding="utf-8")

    resumed = ResearchAllEvidenceWorkflow(
        config,
        component_runners={
            "embedding_cache": embedding,
            "text_models": finish_text,
        },
    ).run()
    assert resumed["status"] == "complete"
    assert calls == ["embedding_cache", "text_models_failed", "text_models_resumed"]


def test_stage2_handoff_reader_uses_the_obvious_output_path(tmp_path):
    _raw, config = _inputs(tmp_path, components=())
    handoff = config.output_dir / "handoff" / "evidence.jsonl"
    handoff.parent.mkdir(parents=True)
    handoff.write_text(
        json.dumps({"source": "tfidf", "outer_fold": 1, "evidence": {"x": 1}}) + "\n",
        encoding="utf-8",
    )

    assert list(iter_stage1_handoff(config.output_dir)) == [
        {"source": "tfidf", "outer_fold": 1, "evidence": {"x": 1}}
    ]


def test_rerun_tfidf_invalidates_exact_context_metadata_but_keeps_products(tmp_path: Path):
    component_dir = tmp_path / "components" / "tfidf"
    context_dir = (
        component_dir
        / "stage1_tfidf_topics"
        / "contexts"
        / "outer_001_inner_001"
    )
    context_dir.mkdir(parents=True)
    component_complete = component_dir / "complete.json"
    worker_complete = context_dir / "complete.json"
    context_metadata = context_dir / "context_metadata.json"
    fitted_product = context_dir / "fitted_context.json"
    for path in (component_complete, worker_complete, context_metadata, fitted_product):
        path.write_text("{}\n", encoding="utf-8")
    workflow = SimpleNamespace(_component_dir=lambda _name: component_dir)

    all_evidence_workflow._invalidate_component_for_rerun(workflow, "tfidf")

    assert not component_complete.exists()
    assert not worker_complete.exists()
    assert not context_metadata.exists()
    assert fitted_product.exists()


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
    obsolete_copies = [
        handoff_dir / f"{source_name}.jsonl"
        for source_name in ("text_models", "tfidf", "neural_queries")
    ]
    for obsolete_copy in obsolete_copies:
        obsolete_copy.write_text("obsolete duplicate\n", encoding="utf-8")

    result = all_evidence_workflow._handoff_component(context, handoff_dir)

    assert result["rows"] == 1
    assert all(not obsolete_copy.exists() for obsolete_copy in obsolete_copies)
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


def test_text_model_contexts_are_independently_resumable(tmp_path, monkeypatch):
    _raw, config = _inputs(tmp_path, components=("text_models",))
    config = replace(config, workers=1)
    context = Stage1RunContext(
        config=config,
        dataset=pd.read_parquet(config.dataset),
        applied_config=None,
        neural_query_config=None,
    )
    calls: list[str] = []

    def fit_one(*, spec, context_dir, **_unused):
        calls.append(str(spec["scope_id"]))
        context_dir.mkdir(parents=True, exist_ok=True)
        row = {
            "outer_fold": spec["outer_fold"],
            "inner_fold": spec["inner_fold"],
            "scope": spec["scope"],
        }
        (context_dir / "evidence.json").write_text(json.dumps(row), encoding="utf-8")
        (context_dir / "complete.json").write_text("{}", encoding="utf-8")
        return row

    monkeypatch.setattr(all_evidence_workflow, "_run_one_text_model_context", fit_one)
    component_dir = context.component_dir("text_models")
    first = all_evidence_workflow._text_models_component(context, component_dir)
    expected_contexts = 2 * (1 + 2)

    assert first["contexts"] == expected_contexts
    assert len(calls) == expected_contexts

    calls.clear()
    second = all_evidence_workflow._text_models_component(context, component_dir)
    assert second["contexts"] == expected_contexts
    assert calls == []

    marker = component_dir / "outer_001_inner_001" / "complete.json"
    marker.unlink()
    all_evidence_workflow._text_models_component(context, component_dir)
    assert calls == ["outer_001_inner_001"]


def test_gpu_context_lanes_use_every_gpu_without_device_overlap():
    pending = [{"scope_id": f"context_{index:02d}"} for index in range(11)]

    lanes = _context_execution_lanes(
        pending,
        devices=(
            "cuda:0",
            "cuda:1",
            "cuda:2",
        ),
        workers=10,
    )

    assert [device for device, _specs in lanes] == ["cuda:0", "cuda:1", "cuda:2"]
    assert [len(specs) for _device, specs in lanes] == [4, 4, 3]
    scheduled = [spec["scope_id"] for _device, specs in lanes for spec in specs]
    assert sorted(scheduled) == sorted(spec["scope_id"] for spec in pending)
    assert _lane_cpu_worker_budgets(len(lanes), workers=10) == [4, 3, 3]


def test_cpu_worker_budget_is_divided_without_being_duplicated():
    assert _lane_cpu_worker_budgets(3, workers=10) == [4, 3, 3]
    assert sum(_lane_cpu_worker_budgets(7, workers=23)) == 23
    assert _lane_cpu_worker_budgets(0, workers=32) == []


def test_handoff_context_config_parallelizes_bow_folds_with_threads(tmp_path):
    _raw, config = _inputs(tmp_path, components=("text_models",))
    context = ResearchAllEvidenceWorkflow(config)._resolved_context()

    optimized = multi_model_stage1.config_for_multi_model_forest_handoff(
        context.applied_config
    )
    text_config = optimized.architecture.multi_model_agentic_forest
    primary_text_config = optimized.architecture.multi_model_forest

    assert text_config.fold_parallelism == "auto"
    assert text_config.bow_fold_parallelism == "auto"
    assert text_config.bow_parallel_backend == "threads"
    assert primary_text_config.fold_parallelism == "auto"
    assert primary_text_config.bow_fold_parallelism == "auto"
    assert primary_text_config.bow_parallel_backend == "threads"


def test_handoff_context_uses_stage1_runner_and_exact_heldout_rows(tmp_path, monkeypatch):
    _raw, config = _inputs(tmp_path, components=("text_models",))
    context = ResearchAllEvidenceWorkflow(config)._resolved_context()
    captured: dict[str, object] = {}

    class FakeStage1Runner:
        def __init__(self, *, dataset, config, output_path, device, gpu_ids, num_workers):
            captured["runner"] = "stage1"
            captured["output_path"] = output_path
            captured["num_workers"] = num_workers

        def build_discovery_handoff_row(self, **kwargs):
            captured.update(kwargs)
            return {
                "outer_fold": kwargs["outer_fold"],
                "scope": kwargs["scope"],
                "importance": {"matched_pair_uplift": {"views": []}},
            }

    monkeypatch.setattr(
        multi_model_stage1,
        "MultiModelForestStage1Runner",
        FakeStage1Runner,
    )
    rows = multi_model_stage1._run_handoff_context_shard(
        dataset=context.dataset,
        config=context.applied_config,
        shard=[
            {
                "fold_key": 1001,
                "outer_fold": 1,
                "scope": "candidate_consistency_inner_train",
                "train_idx": np.asarray([0, 1, 2, 3]),
                "heldout_idx": np.asarray([4, 5]),
                "inner_fold": 1,
                "heldout_rows": 2,
            }
        ],
        handoff_dir=tmp_path / "handoff",
        shard_index=0,
        device="cpu",
        gpu_ids=None,
        num_workers=3,
    )

    assert captured["runner"] == "stage1"
    assert captured["num_workers"] == 3
    assert np.asarray(captured["train_idx"]).tolist() == [0, 1, 2, 3]
    assert np.asarray(captured["heldout_idx"]).tolist() == [4, 5]
    assert rows[0]["importance"]["matched_pair_uplift"] == {"views": []}


def test_default_research_config_requires_all_ten_stage2_architectures(tmp_path):
    _raw, config = _inputs(tmp_path, components=())
    context = ResearchAllEvidenceWorkflow(config)._resolved_context()

    assert _required_stage2_architectures(context) == SUPPORTED_STAGE2_ARCHITECTURES


def test_handoff_context_receives_its_full_lane_cpu_budget(tmp_path, monkeypatch):
    captured: list[int] = []

    def run_shard(**kwargs):
        captured.append(int(kwargs["num_workers"]))
        return []

    monkeypatch.setattr(
        multi_model_stage1,
        "_run_handoff_context_shard",
        run_shard,
    )
    plan = multi_model_stage1.resolve_multi_model_forest_stage1_parallel_plan(
        cpus_total=4,
        num_workers=4,
        gpu_ids=[0],
        htr_jobs_per_gpu=1,
        htr_enabled=True,
        embedding_enabled=True,
    )

    rows = multi_model_stage1.run_multi_model_forest_handoff_contexts(
        dataset=pd.DataFrame({"clinical_text": ["one"]}),
        config=None,
        contexts=[{"fold_key": 1}],
        handoff_dir=tmp_path,
        plan=plan,
        base_device=multi_model_stage1.torch.device("cpu"),
    )

    assert rows == []
    assert captured == [4]
    assert multi_model_stage1._handoff_cpu_worker_budgets(23, 7) == [4, 4, 3, 3, 3, 3, 3]


def test_cpu_context_lanes_are_bounded_by_workers():
    pending = [{"scope_id": f"context_{index:02d}"} for index in range(7)]

    lanes = _context_execution_lanes(
        pending,
        devices=("cpu",),
        workers=3,
    )

    assert [device for device, _specs in lanes] == ["cpu", "cpu", "cpu"]
    assert [len(specs) for _device, specs in lanes] == [3, 2, 2]


def test_context_lanes_spread_larger_full_contexts_across_devices():
    pending = [
        {"scope_id": "outer_1", "train_idx": list(range(100))},
        {"scope_id": "inner_1", "train_idx": list(range(60))},
        {"scope_id": "outer_2", "train_idx": list(range(100))},
        {"scope_id": "inner_2", "train_idx": list(range(60))},
    ]

    lanes = _context_execution_lanes(
        pending,
        devices=("cuda:0", "cuda:1"),
        workers=1,
    )

    assert [
        [spec["scope_id"] for spec in specs]
        for _device, specs in lanes
    ] == [["outer_1", "inner_1"], ["outer_2", "inner_2"]]


def test_phase_flags_are_mutually_exclusive():
    parser = build_parser()

    assert parser.parse_args(["--disable-htr"]).disable_htr is True

    with pytest.raises(SystemExit):
        parser.parse_args(["--stage1-only", "--stage2-only"])


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
                all_evidence_workflow.LEGACY_STAGE2_SELECTION_SCHEMA_VERSION
            ),
            "extracted_fit_fingerprint": "legacy-in-memory-frame-fingerprint",
            "treatment_outcome_fingerprint": (
                all_evidence_workflow._frame_fingerprint(treatment_outcome)
            ),
            "definitions": source_features,
            "inner_splits": inner_splits,
            "outcome_type": config.outcome_type,
            "confounder_p_value_threshold": 0.05,
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
        (selection_dir / "statistical_selection.json").write_text(
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


def test_stage2_reselection_backfills_heldout_cache_for_prepared_migration(tmp_path):
    config = _completed_stage2_reselection_fixture(tmp_path)
    state = all_evidence_workflow.prepare_stage2_reselection(config=config)
    stage2_dir = config.output_dir / "stage2"
    snapshot_path = stage2_dir / "outer_001" / "preselection" / "input.json"
    completion_path = stage2_dir / "outer_001" / "preselection" / "complete.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot.pop("input_fingerprint")
    snapshot.pop("heldout_measurement_cache")
    legacy_fingerprint = all_evidence_workflow._stage2_value_fingerprint(snapshot)
    snapshot_path.write_text(
        json.dumps({**snapshot, "input_fingerprint": legacy_fingerprint}),
        encoding="utf-8",
    )
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["input_fingerprint"] = legacy_fingerprint
    completion.pop("heldout_measurement_cache_schema_version", None)
    completion_path.write_text(json.dumps(completion), encoding="utf-8")

    resumed = all_evidence_workflow.prepare_stage2_reselection(config=config)

    upgraded = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert resumed["reselection_id"] == state["reselection_id"]
    assert upgraded["heldout_measurement_cache"]["heldout_row_ids"] == [4, 5]
    assert upgraded["input_fingerprint"] == (
        all_evidence_workflow._stage2_value_fingerprint(
            {
                key: value
                for key, value in upgraded.items()
                if key != "input_fingerprint"
            }
        )
    )


def test_stage2_reselection_fails_before_moving_incompatible_definitions(tmp_path):
    config = _completed_stage2_reselection_fixture(tmp_path)
    stage2_dir = config.output_dir / "stage2"
    definitions_complete = stage2_dir / "outer_001" / "definitions_complete.json"
    definitions_complete.write_text(
        json.dumps({"status": "complete", "evidence_input_fingerprint": "wrong"}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="feature definitions do not match"):
        all_evidence_workflow.prepare_stage2_reselection(config=config)

    assert (stage2_dir / "complete.json").is_file()
    assert (stage2_dir / "outer_001" / "selection").is_dir()
    assert not (stage2_dir / "reselection_state.json").exists()


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


def test_completed_stage2_with_legacy_infrastructure_missingness_is_repaired(
    tmp_path,
    monkeypatch,
):
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
    stage2_dir = config.output_dir / "stage2"
    stage2_dir.mkdir(parents=True)
    for name, content in (
        ("causal_estimate.json", "{}"),
        ("cross_fitted_predictions.csv", "_oci_row_id,aipw_score\n"),
        ("posthoc_oracle_ite_metrics.json", '{"available":false}\n'),
    ):
        (stage2_dir / name).write_text(content, encoding="utf-8")
    (stage2_dir / "complete.json").write_text(
        json.dumps({"phase": "causal_estimation"}),
        encoding="utf-8",
    )
    failure = (
        stage2_dir
        / "outer_001"
        / "extraction"
        / "heldout"
        / "batches"
        / "batch_00001"
        / "extraction_failure.json"
    )
    failure.parent.mkdir(parents=True)
    failure.write_text(
        json.dumps(
            {
                "resolution": "conservative_all_null",
                "validation_error": "Stage2RequestExhaustedError: endpoint unavailable",
            }
        ),
        encoding="utf-8",
    )
    calls = []

    def repair_stage2(*, output_dir, **kwargs):
        calls.append(kwargs)
        failure.rename(
            failure.with_name("superseded_infrastructure_extraction_failure.json")
        )
        return {"phase": "causal_estimation"}

    monkeypatch.setattr(all_evidence_workflow, "run_plain_handoff_stage2", repair_stage2)

    result = ResearchAllEvidenceWorkflow(config).run()

    assert result["components"]["stage2"]["status"] == "complete"
    assert len(calls) == 1
    assert (stage2_dir / "superseded_infrastructure_complete.json").is_file()


def test_definition_only_stage2_marker_is_continued_after_upgrade(tmp_path, monkeypatch):
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
    stage2_dir = config.output_dir / "stage2"
    stage2_dir.mkdir(parents=True)
    (stage2_dir / "complete.json").write_text(
        json.dumps({"status": "complete", "component": "stage2"}),
        encoding="utf-8",
    )
    calls = []

    def finish_stage2(*, output_dir, **kwargs):
        calls.append(kwargs)
        (output_dir / "causal_estimate.json").write_text("{}", encoding="utf-8")
        (output_dir / "cross_fitted_predictions.csv").write_text(
            "_oci_row_id,aipw_score\n", encoding="utf-8"
        )
        (output_dir / "posthoc_oracle_ite_metrics.json").write_text(
            '{"available":false}\n', encoding="utf-8"
        )
        return {"phase": "causal_estimation"}

    monkeypatch.setattr(all_evidence_workflow, "run_plain_handoff_stage2", finish_stage2)

    result = ResearchAllEvidenceWorkflow(config).run()

    assert result["components"]["stage2"]["status"] == "complete"
    assert len(calls) == 1


def test_completed_legacy_stage2_without_oracle_metrics_is_backfilled(
    tmp_path,
    monkeypatch,
):
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
    stage2_dir = config.output_dir / "stage2"
    stage2_dir.mkdir(parents=True)
    (stage2_dir / "causal_estimate.json").write_text("{}", encoding="utf-8")
    (stage2_dir / "cross_fitted_predictions.csv").write_text(
        "_oci_row_id,aipw_score,estimated_cate\n", encoding="utf-8"
    )
    (stage2_dir / "complete.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "component": "stage2",
                "phase": "causal_estimation",
            }
        ),
        encoding="utf-8",
    )
    calls = []

    def backfill_stage2(*, output_dir, **kwargs):
        calls.append(kwargs)
        (output_dir / "posthoc_oracle_ite_metrics.json").write_text(
            '{"available":false}\n', encoding="utf-8"
        )
        return {"phase": "causal_estimation"}

    monkeypatch.setattr(all_evidence_workflow, "run_plain_handoff_stage2", backfill_stage2)

    result = ResearchAllEvidenceWorkflow(config).run()

    assert result["components"]["stage2"]["status"] == "complete"
    assert len(calls) == 1


def test_stage2_endpoint_makes_full_run_the_default(tmp_path):
    raw, _config = _inputs(tmp_path)
    raw["stage2"] = {
        "endpoint": "http://stage2.test/v1",
    }

    config = compile_config(raw, config_dir=tmp_path)

    assert config.mode == "full"
    assert config.components == (*COMPONENT_ORDER, "stage2")
    assert config.stage2 is not None
    assert config.stage2.endpoint == "http://stage2.test/v1"
    assert config.stage2.model == ""


def test_saved_run_config_can_start_stage2_with_endpoint_only(tmp_path):
    _raw, original = _inputs(tmp_path)
    saved_config = tmp_path / "run_config.json"
    saved_config.write_text(
        json.dumps(original.as_dict(), default=str),
        encoding="utf-8",
    )
    args = build_parser().parse_args(
        [
            "--config",
            str(saved_config),
            "--stage2-only",
            "--stage2-endpoint",
            "http://stage2.test/v1",
            "--stage2-extraction-feature-batch-size",
            "7",
            "--stage2-cluster-similarity-threshold",
            "0.7",
            "--stage2-max-tokens",
            "150000",
            "--stage2-extraction-max-tokens",
            "70000",
            "--stage2-extraction-chunk-size-tokens",
            "45000",
            "--stage2-extraction-context-window-tokens",
            "140000",
            "--stage2-extraction-context-margin-tokens",
            "1500",
        ]
    )

    raw, config_dir = _raw_config_from_args(args)
    resumed = compile_config(raw, config_dir=config_dir)

    assert resumed.mode == "stage2"
    assert resumed.components == ("stage2",)
    assert resumed.stage2 is not None
    assert resumed.stage2.endpoint == "http://stage2.test/v1"
    assert resumed.stage2.model == ""
    assert resumed.stage2.extraction_feature_batch_size == 7
    assert resumed.stage2.agentic_selection.cluster_similarity_threshold == 0.7
    assert resumed.stage2.max_tokens == 150_000
    assert resumed.stage2.extraction_max_tokens == 70_000
    assert resumed.stage2.extraction_chunk_size_tokens == 45_000
    assert resumed.stage2.extraction_context_window_tokens == 140_000
    assert resumed.stage2.extraction_context_margin_tokens == 1_500
    assert resumed.clinical_question == original.clinical_question
    assert resumed.unit_id_column == original.unit_id_column
    assert resumed.htr_enabled == original.htr_enabled


def test_reselection_migrates_retired_screen_fields_in_saved_full_run_config(tmp_path):
    _raw, original = _inputs(tmp_path)
    saved = original.as_dict()
    saved["components"] = [*saved["components"], "stage2"]
    saved["mode"] = "full"
    saved["stage2"] = {
        "endpoint": "http://stage2.test/v1",
        "model": "primary-model",
        "extraction_llm": {
            "endpoint": "http://extract.test/v1",
            "model": "extractor-model",
        },
        "selection_workers": 8,
        "confounder_p_value_threshold": 0.05,
        "confounder_min_inner_fold_fraction": 0.75,
        "effect_modifier_p_value_threshold": 0.05,
        "effect_modifier_min_inner_fold_fraction": 0.75,
    }
    saved_path = tmp_path / "completed_run_config.json"
    saved_path.write_text(json.dumps(saved, default=str), encoding="utf-8")
    args = build_parser().parse_args(
        [
            "--config",
            str(saved_path),
            "--stage2-only",
            "--stage2-reselect",
        ]
    )

    raw, config_dir = _raw_config_from_args(args)
    resumed = compile_config(raw, config_dir=config_dir)

    assert resumed.components == ("stage2",)
    assert resumed.stage2 is not None
    assert resumed.stage2.agentic_selection.cluster_similarity_threshold == 0.60
    assert resumed.stage2.input_temporal_scope == "pre_index_treatment"
    assert not set(raw["stage2"]).intersection(
        all_evidence_workflow.RETIRED_STAGE2_SCREEN_CONFIG_KEYS
    )


def test_old_stage2_command_is_rejected_instead_of_silently_ignored(tmp_path):
    raw, _config = _inputs(tmp_path)
    raw["stage2"] = {"command": ["old-bundle-stage2"]}

    with pytest.raises(ValueError, match="stage2.command is not used"):
        compile_config(raw, config_dir=tmp_path)


def test_stage1_architecture_selector_is_scientific_and_prunes_producers(tmp_path):
    raw, _legacy = _inputs(tmp_path)
    raw["science"]["stage1_architectures"] = ["bow_nuisance"]
    config = compile_config(raw, config_dir=tmp_path)
    workflow = ResearchAllEvidenceWorkflow(config)

    assert config.stage1_architectures == ("bow_nuisance",)
    assert workflow.components == ("text_models", "handoff")
    assert config.components == COMPONENT_ORDER


def test_cli_architecture_selector_populates_science_config(tmp_path):
    raw, _config = _inputs(tmp_path)
    config_path = tmp_path / "run.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    args = build_parser().parse_args(
        [
            "--config",
            str(config_path),
            "--architectures",
            "tfidf_topics,bow_nuisance",
        ]
    )

    overridden, config_dir = _raw_config_from_args(args)
    config = compile_config(overridden, config_dir=config_dir)

    assert config.stage1_architectures == ("bow_nuisance", "tfidf_topics")


def test_legacy_run_config_omits_unset_architecture_selector(tmp_path):
    _raw, config = _inputs(tmp_path)

    assert "stage1_architectures" not in config.as_dict()


def test_architecture_selector_cannot_be_added_to_existing_legacy_output(tmp_path):
    raw, _config = _inputs(tmp_path)
    completion = tmp_path / "output" / "components" / "tfidf" / "complete.json"
    completion.parent.mkdir(parents=True)
    completion.write_text('{"status":"complete"}', encoding="utf-8")
    raw["science"]["stage1_architectures"] = ["tfidf_topics"]
    selected = compile_config(raw, config_dir=tmp_path)

    with pytest.raises(ValueError, match="fresh output directory"):
        ResearchAllEvidenceWorkflow(selected)


def test_selected_direct_architecture_must_not_be_disabled(tmp_path):
    raw, _config = _inputs(tmp_path)
    raw["science"]["stage1_architectures"] = ["htr_neural"]
    raw["science"]["htr_enabled"] = False
    selected = compile_config(raw, config_dir=tmp_path)
    workflow = ResearchAllEvidenceWorkflow(selected)

    with pytest.raises(ValueError, match="disabled by their direct implementation"):
        workflow._resolved_context()
