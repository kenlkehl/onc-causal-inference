"""Reporting-only Stage 1 wall time per architecture and context (S1-29)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from oci.config import AppliedInferenceConfig
from oci.inference import multi_model_forest_stage1
from oci.inference import research_all_evidence_workflow as workflow
from oci.inference.stage1_timing import (
    SHARED_ATTRIBUTION,
    STAGE1_TIMING_SCHEMA_VERSION,
    read_timing_record,
)


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
