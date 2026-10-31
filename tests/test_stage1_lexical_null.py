"""Random-chunk null for the Stage 1 lexical recovery metric (S1-29)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from oci.evaluation.stage1 import evaluate_stage1_architectures
from oci.inference.stage1_architecture_artifacts import (
    materialize_stage1_architecture_artifacts,
)


def _run(tmp_path: Path) -> tuple[Path, Path]:
    run_dir = tmp_path / "run"
    rows = 40
    rng = np.random.default_rng(5)
    texts = [" ".join(rng.choice(["note", "routine", "visit", "ecog", "pain"], size=30))
             for _ in range(rows)]
    dataset = pd.DataFrame(
        {
            "clinical_text": texts,
            "treatment_indicator": [0, 1] * (rows // 2),
            "outcome_indicator": [0, 0, 1, 1] * (rows // 4),
            "true_ecog_ps": rng.integers(0, 3, size=rows),
        }
    )
    dataset_path = tmp_path / "dataset.parquet"
    dataset.to_parquet(dataset_path, index=False)
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({"features": [{"name": "ecog_ps"}]}), encoding="utf-8")
    run_dir.mkdir()
    (run_dir / "run_config.json").write_text(
        json.dumps({"dataset": str(dataset_path), "seed": 11}), encoding="utf-8"
    )
    split = run_dir / "components" / "tfidf" / "split_provenance.jsonl"
    split.parent.mkdir(parents=True)
    split.write_text(
        json.dumps(
            {
                "outer_fold": 1,
                "fit_row_ids": list(range(30)),
                "heldout_row_ids": list(range(30, 40)),
                "inner_splits": [
                    {"inner_fold": 1, "fit_row_ids": list(range(15)),
                     "heldout_row_ids": list(range(15, 30))}
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    evidence_path = run_dir / "components" / "text_models" / "evidence.jsonl"
    evidence_path.parent.mkdir(parents=True)
    raw_rows = [
        {
            "source": "text_models",
            "outer_fold": 1,
            "inner_fold": inner,
            "scope": scope,
            "evidence": {
                "importance": {
                    "views": [
                        {
                            "view_name": "word_linear",
                            "treatment_positive": [
                                {"feature": term, "score": 1.0} for term in terms
                            ],
                            "outcome_positive": [],
                            "pseudo_target_positive": [],
                        }
                    ]
                }
            },
        }
        for inner, scope, terms in (
            (None, "full_outer_train", ["ecog ps", "routine visit"]),
            (1, "candidate_consistency_inner_train", ["pain"]),
        )
    ]
    evidence_path.write_text(
        "".join(json.dumps(row) + "\n" for row in raw_rows), encoding="utf-8"
    )
    materialize_stage1_architecture_artifacts(
        output_dir=run_dir,
        raw_handoff_rows=raw_rows,
        selected_architectures=("bow_nuisance",),
        source_artifacts={"text_models": evidence_path},
        selection_mode="explicit",
    )
    return run_dir, metadata


def test_split_and_timing_inputs_are_frozen_before_the_oracle_dataset_loads(
    tmp_path, monkeypatch
):
    from oci.evaluation import stage1 as stage1_evaluation

    run_dir, metadata = _run(tmp_path)
    split = run_dir / "components" / "tfidf" / "split_provenance.jsonl"
    original_split = split.read_text(encoding="utf-8")
    timing = run_dir / "components" / "text_models" / "timing.jsonl"
    timing.write_text(
        json.dumps(
            {
                "schema_version": "stage1_architecture_timing_v1",
                "component": "text_models",
                "scope_id": "outer_001_full",
                "outer_fold": 1,
                "inner_fold": None,
                "status": "measured",
                "wall_seconds": 2.0,
                "entries": [
                    {"unit": "bow_importance_nuisance_terms", "attribution": "bow_nuisance",
                     "consumers": ["bow_nuisance"], "seconds": 2.0}
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    original_load = stage1_evaluation._load_dataset

    def load_after_tampering(path):
        # Anything read after this point would see the tampered files.
        split.write_text(original_split.replace('"fit_row_ids": [0', '"fit_row_ids": [39'),
                         encoding="utf-8")
        timing.write_text("", encoding="utf-8")
        return original_load(path)

    baseline = evaluate_stage1_architectures(
        run_dir=run_dir, metadata_path=metadata, null_draws=20, output_dir=tmp_path / "a"
    )
    split.write_text(original_split, encoding="utf-8")
    timing_text = json.dumps(baseline["timing"])
    monkeypatch.setattr(stage1_evaluation, "_load_dataset", load_after_tampering)
    tampered = evaluate_stage1_architectures(
        run_dir=run_dir, metadata_path=metadata, null_draws=20, output_dir=tmp_path / "b"
    )

    assert json.dumps(tampered["timing"]) == timing_text
    assert tampered["timing"]["architectures"]["bow_nuisance"]["exclusive_seconds"] == 2.0
    pd.testing.assert_frame_equal(
        pd.read_csv(tmp_path / "a" / "lexical_recovery_null.csv"),
        pd.read_csv(tmp_path / "b" / "lexical_recovery_null.csv"),
    )
