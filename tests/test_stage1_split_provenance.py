"""Every Stage 1 lane uses the canonical stratified split provenance (B7)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oci.inference.research_all_evidence_workflow import (
    ResearchAllEvidenceWorkflow,
    _stage1_context_specs,
    compile_config,
)
from oci.inference.tfidf_topic_stage1 import build_tfidf_topic_split_provenance


def _raw_config(tmp_path: Path) -> dict:
    rng = np.random.default_rng(3)
    rows = 40
    treatment = np.tile([0, 1], rows // 2)
    outcome = (rng.random(rows) < 0.3 + 0.4 * treatment).astype(int)
    dataset = tmp_path / "cohort.parquet"
    pd.DataFrame(
        {
            "patient_id": [f"p{index}" for index in range(rows)],
            "clinical_text": [f"note {index} words" for index in range(rows)],
            "treatment_indicator": treatment,
            "outcome_indicator": outcome,
        }
    ).to_parquet(dataset, index=False)
    stage1_template = tmp_path / "stage1.json"
    stage1_template.write_text(
        json.dumps({"config": {"architecture": {"model_type": "multi_model_forest"}}}),
        encoding="utf-8",
    )
    query_template = tmp_path / "queries.json"
    query_template.write_text(json.dumps({"query_inner_folds": 2}), encoding="utf-8")
    return {
        "dataset": str(dataset),
        "output_dir": str(tmp_path / "output"),
        "stage1_template": str(stage1_template),
        "neural_query_template": str(query_template),
        "science": {"outer_folds": 3, "inner_folds": 2, "seed": 13},
        "run": {"components": ["text_models"], "workers": 1},
        "models": {"htr": "test-htr-model", "embeddings": "test-embedding-model"},
    }


def test_legacy_component_without_tfidf_uses_canonical_stratified_splits(tmp_path):
    config = compile_config(_raw_config(tmp_path), config_dir=tmp_path)
    assert config.stage1_architectures is None
    context = ResearchAllEvidenceWorkflow(config)._resolved_context()
    provenance_path = context.component_dir("tfidf") / "split_provenance.jsonl"
    assert not provenance_path.exists()

    specs = _stage1_context_specs(context)

    assert provenance_path.is_file()
    expected = build_tfidf_topic_split_provenance(context.dataset, context.applied_config)
    assert expected[0]["split_method"] == "stratified_joint_treatment_outcome"
    expected_specs = []
    for outer in expected:
        expected_specs.append(
            (outer["outer_fold"], None, outer["fit_row_ids"], outer["heldout_row_ids"])
        )
        for inner in outer["inner_splits"]:
            expected_specs.append(
                (
                    outer["outer_fold"],
                    inner["inner_fold"],
                    inner["fit_row_ids"],
                    inner["heldout_row_ids"],
                )
            )
    assert [
        (
            spec["outer_fold"],
            spec["inner_fold"],
            spec["train_idx"].tolist(),
            spec["heldout_idx"].tolist(),
        )
        for spec in specs
    ] == expected_specs
    # A later TF-IDF run or Stage 2 reads the very same file.
    assert [json.loads(line)["fit_row_ids"] for line in provenance_path.open()] == [
        outer["fit_row_ids"] for outer in expected
    ]


def test_empty_split_provenance_fails_closed(tmp_path):
    config = compile_config(_raw_config(tmp_path), config_dir=tmp_path)
    context = ResearchAllEvidenceWorkflow(config)._resolved_context()
    provenance_path = context.component_dir("tfidf") / "split_provenance.jsonl"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text("", encoding="utf-8")

    with pytest.raises(RuntimeError, match="contains no outer folds"):
        _stage1_context_specs(context)
