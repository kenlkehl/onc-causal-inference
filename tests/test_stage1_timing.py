"""Reporting-only Stage 1 wall time per architecture and context (S1-29)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oci.config import AppliedInferenceConfig
from oci.evaluation.stage1 import evaluate_stage1_architectures
from oci.inference import multi_model_forest_stage1
from oci.inference import research_all_evidence_workflow as workflow
from oci.inference.stage1_architecture_artifacts import (
    materialize_stage1_architecture_artifacts,
)
from oci.inference.stage1_timing import (
    SHARED_ATTRIBUTION,
    STAGE1_TIMING_SCHEMA_VERSION,
    SectionTimer,
    read_timing_record,
)
from tests.test_tfidf_topic_stage1_end_to_end import _config as _tfidf_config
from tests.test_tfidf_topic_stage1_end_to_end import _run as _tfidf_run


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class _Store:
    def __init__(self) -> None:
        self.reused_units: list[str] = []


def test_section_timer_attributes_laps_and_flags_restored_sections():
    clock = _Clock()
    store = _Store()
    timer = SectionTimer(checkpoint_store=store, clock=clock)
    clock.now = 2.0
    timer.lap("bow_nuisance_fits", attribution=SHARED_ATTRIBUTION, consumers=["b", "a"])
    clock.now = 5.0
    store.reused_units.append("bow/effect/view_000_pseudo_outcome")
    timer.lap("bow_effect_fits", attribution="bow_r_loss")
    clock.now = 6.0
    timer.lap("bow_effect_fits", attribution="bow_r_loss")
    clock.now = 6.5
    record = timer.record(component="text_models", context={"scope_id": "s", "outer_fold": 1})

    entries = {entry["unit"]: entry for entry in record["entries"]}
    assert entries["bow_nuisance_fits"]["seconds"] == 2.0
    assert entries["bow_nuisance_fits"]["consumers"] == ["a", "b"]
    assert entries["bow_effect_fits"]["seconds"] == 4.0
    assert entries["bow_effect_fits"]["laps"] == 2
    assert entries["context_overhead"]["seconds"] == pytest.approx(0.5)
    assert record["wall_seconds"] == 6.5
    assert record["status"] == "partially_restored"
    assert record["restored_checkpoint_units"] == ["bow/effect/view_000_pseudo_outcome"]
    with pytest.raises(ValueError, match="changed its attribution"):
        timer.lap("bow_effect_fits", attribution="bow_nuisance")


def _spec() -> dict:
    return {
        "scope_id": "outer_001_full",
        "scope": "full_outer_train",
        "fold_key": 1,
        "outer_fold": 1,
        "inner_fold": None,
        "train_idx": np.arange(6),
        "heldout_idx": np.arange(6, 8),
    }


def test_text_model_timing_is_recorded_and_survives_resume(tmp_path, monkeypatch):
    calls = []

    def handoff(**kwargs):
        calls.append(kwargs["context"]["scope_id"])
        kwargs["timer"].lap("bow_nuisance_fits", attribution=SHARED_ATTRIBUTION)
        kwargs["timer"].lap("bow_effect_fits", attribution="bow_r_loss")
        return {"outer_fold": 1, "scope": "full_outer_train", "importance": {}}

    monkeypatch.setattr(multi_model_forest_stage1, "run_multi_model_forest_handoff_context", handoff)
    dataset = pd.DataFrame({"clinical_text": ["a"] * 8})
    context_dir = tmp_path / "text_models" / "outer_001_full"
    kwargs = dict(
        dataset=dataset,
        applied_config=AppliedInferenceConfig(seed=3),
        spec=_spec(),
        context_dir=context_dir,
        device="cpu",
        cpu_workers=1,
    )

    first_row = workflow._run_one_text_model_context(**kwargs)
    first = read_timing_record(context_dir / "timing.json")
    assert first["status"] == "measured"
    assert first["schema_version"] == STAGE1_TIMING_SCHEMA_VERSION
    assert {entry["unit"] for entry in first["entries"]} >= {"bow_nuisance_fits", "bow_effect_fits"}

    # Resume: the context checkpoint is reused and the original timing kept.
    second_row = workflow._run_one_text_model_context(**kwargs)
    assert calls == ["outer_001_full"]
    assert second_row == first_row
    assert read_timing_record(context_dir / "timing.json") == first

    # A context restored from a checkpoint without timing is marked unknown.
    (context_dir / "timing.json").unlink()
    workflow._run_one_text_model_context(**kwargs)
    restored = read_timing_record(context_dir / "timing.json")
    assert restored["status"] == "unknown"
    assert restored["entries"] == []
    assert calls == ["outer_001_full"]

    workflow._write_component_context_timing(
        tmp_path / "text_models",
        "text_models",
        [_spec(), {**_spec(), "scope_id": "outer_002_full", "outer_fold": 2}],
    )
    lines = (tmp_path / "text_models" / "timing.jsonl").read_text(encoding="utf-8").splitlines()
    statuses = [json.loads(line)["status"] for line in lines]
    assert statuses == ["unknown", "unknown"]


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_tfidf_contexts_record_timing_that_resume_keeps(tmp_path):
    config = _tfidf_config(orphan=False)
    _tfidf_run(tmp_path, config)
    records = [
        json.loads(line)
        for line in (tmp_path / "timing.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 6
    assert all(record["status"] == "measured" for record in records)
    units = {entry["unit"]: entry["attribution"] for entry in records[0]["entries"]}
    assert units["tfidf_nuisance_stacks"] == SHARED_ATTRIBUTION
    assert units["tfidf_topic_screens_and_banks"] == "tfidf_topics"

    _tfidf_run(tmp_path, config)
    again = [
        json.loads(line)
        for line in (tmp_path / "timing.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert again == records


def test_evaluation_reports_per_architecture_and_shared_time(tmp_path):
    run_dir = tmp_path / "run"
    dataset = pd.DataFrame({"clinical_text": ["x"] * 4, "true_age": [1.0, 2.0, 3.0, 4.0]})
    dataset.to_parquet(tmp_path / "dataset.parquet", index=False)
    run_dir.mkdir()
    (run_dir / "run_config.json").write_text(
        json.dumps({"dataset": str(tmp_path / "dataset.parquet")}), encoding="utf-8"
    )
    raw = {
        "source": "text_models",
        "outer_fold": 1,
        "inner_fold": None,
        "scope": "full_outer_train",
        "evidence": {
            "importance": {
                "views": [
                    {
                        "view_name": "v",
                        "treatment_positive": [{"feature": "age", "score": 1.0}],
                        "outcome_positive": [],
                        "pseudo_target_positive": [],
                    }
                ]
            }
        },
    }
    evidence_path = run_dir / "components" / "text_models" / "evidence.jsonl"
    evidence_path.parent.mkdir(parents=True)
    evidence_path.write_text(json.dumps(raw) + "\n", encoding="utf-8")
    measured = {
        "schema_version": STAGE1_TIMING_SCHEMA_VERSION,
        "component": "text_models",
        "scope_id": "outer_001_full",
        "outer_fold": 1,
        "inner_fold": None,
        "scope": "full_outer_train",
        "status": "measured",
        "wall_seconds": 10.0,
        "entries": [
            {"unit": "bow_nuisance_fits", "attribution": "shared",
             "consumers": ["bow_nuisance", "bow_r_loss"], "seconds": 6.0},
            {"unit": "bow_importance_nuisance_terms", "attribution": "bow_nuisance",
             "consumers": ["bow_nuisance"], "seconds": 3.0},
            {"unit": "bow_effect_fits", "attribution": "bow_r_loss",
             "consumers": ["bow_r_loss"], "seconds": 1.0},
        ],
    }
    unknown = {
        "schema_version": STAGE1_TIMING_SCHEMA_VERSION,
        "component": "text_models",
        "scope_id": "outer_001_inner_001",
        "outer_fold": 1,
        "inner_fold": 1,
        "status": "unknown",
        "reason": "restored from a checkpoint written before timing was recorded",
        "entries": [],
    }
    (run_dir / "components" / "text_models" / "timing.jsonl").write_text(
        json.dumps(measured) + "\n" + json.dumps(unknown) + "\n", encoding="utf-8"
    )
    # A component completed before timing existed is reported as unknown.
    (run_dir / "components" / "tfidf").mkdir()
    (run_dir / "components" / "tfidf" / "complete.json").write_text("{}", encoding="utf-8")
    materialize_stage1_architecture_artifacts(
        output_dir=run_dir,
        raw_handoff_rows=[raw],
        selected_architectures=("bow_nuisance",),
        source_artifacts={"text_models": evidence_path},
        selection_mode="explicit",
    )

    result = evaluate_stage1_architectures(run_dir=run_dir, null_draws=0)
    output = Path(result["output_dir"])

    timing = result["timing"]
    assert timing["architectures"]["bow_nuisance"]["exclusive_seconds"] == 3.0
    assert timing["architectures"]["bow_nuisance"]["timed_contexts"] == 1
    assert timing["shared"]["text_models:bow_nuisance_fits"]["seconds"] == 6.0
    assert timing["context_status_counts"] == {"measured": 1, "unknown": 2}
    assert [item["scope_id"] for item in timing["unknown_contexts"]] == [
        "outer_001_inner_001",
        None,
    ]
    assert timing["unknown_contexts"][1]["component"] == "tfidf"
    table = pd.read_csv(output / "timing.csv")
    assert set(table["status"]) == {"measured", "unknown"}
    assert table.loc[table["unit"] == "bow_effect_fits", "seconds"].item() == 1.0
    metrics = [
        json.loads(line)
        for line in (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    wall = [row for row in metrics if row["metric"] == "wall_seconds"]
    assert sorted((row["inner_fold"] is None, row["value"]) for row in wall) == [
        (False, None),
        (True, 3.0),
    ]
    assert json.loads((output / "summary.json").read_text(encoding="utf-8"))["timing"] == timing
    manifest = json.loads((output / "evaluation_manifest.json").read_text(encoding="utf-8"))
    assert list(manifest["consumed_timing_sha256"]) == ["components/text_models/timing.jsonl"]


def test_embedding_cache_component_records_shared_timing(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from oci.inference import embedding_contrast_discovery

    class Generator:
        def __init__(self, **_kwargs):
            self._chunk_cache_reused = True

        def prepare(self, _dataset):
            return None

    monkeypatch.setattr(embedding_contrast_discovery, "EmbeddingContrastEvidenceGenerator", Generator)
    context = SimpleNamespace(
        applied_config=None, config=SimpleNamespace(devices=("cpu",)), dataset=pd.DataFrame()
    )

    workflow._embedding_cache_component(context, tmp_path)

    record = read_timing_record(tmp_path / "timing.json")
    assert record["cache_reused"] is True
    assert [entry["unit"] for entry in record["entries"]][0] == "embedding_cache"
    assert record["entries"][0]["attribution"] == SHARED_ATTRIBUTION
    assert "neural_query_moments" in record["entries"][0]["consumers"]
    assert (tmp_path / "timing.jsonl").is_file()
