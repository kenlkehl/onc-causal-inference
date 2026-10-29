from __future__ import annotations

import json

import pandas as pd
import pytest

import oci.inference.plain_handoff_stage2 as stage2_workflow

from oci.inference.plain_handoff_stage2 import (
    PlainHandoffStage2,
    PlainHandoffStage2Config,
    Stage2ExtractionLLMConfig,
)


def _runner(**config_overrides) -> PlainHandoffStage2:
    return PlainHandoffStage2(
        config=PlainHandoffStage2Config(
            endpoint="http://stage2.test/v1",
            model="test-model",
            required_architectures=(),
            **config_overrides,
        ),
        clinical_question="Identify pretreatment confounders and effect modifiers.",
        completion=lambda _messages, _config: "{}",
    )


def _split_rows():
    return [
        {
            "outer_fold": outer_fold,
            "fit_row_ids": [row for row in range(8) if row not in heldout],
            "heldout_row_ids": heldout,
            "inner_splits": [
                {"inner_fold": 1, "fit_row_ids": fit[:3], "heldout_row_ids": fit[3:]},
                {"inner_fold": 2, "fit_row_ids": fit[3:], "heldout_row_ids": fit[:3]},
            ],
        }
        for outer_fold, heldout in ((1, [0, 1, 2, 3]), (2, [4, 5, 6, 7]))
        for fit in ([row for row in range(8) if row not in heldout],)
    ]


def _raw_handoff(tmp_path, splits, *, mutate=None):
    rows = []
    for split in splits:
        contexts = [(None, "full_outer_train", split)] + [
            (inner["inner_fold"], "candidate_selection_inner_fit", inner)
            for inner in split["inner_splits"]
        ]
        for inner_fold, scope, rows_for_context in contexts:
            payload = {
                "outer_fold": split["outer_fold"], "inner_fold": inner_fold, "scope": scope,
                "fit_row_ids": list(rows_for_context["fit_row_ids"]),
                "heldout_row_ids": list(rows_for_context["heldout_row_ids"]),
            }
            rows.append({"source": "tfidf", "outer_fold": split["outer_fold"],
                         "inner_fold": inner_fold, "scope": scope, "evidence": payload})
        rows.append({
            "source": "text_models", "outer_fold": split["outer_fold"], "inner_fold": None,
            "scope": "full_outer_train",
            "evidence": {"metrics": {
                "handoff_fit_rows": len(split["fit_row_ids"]),
                "handoff_heldout_rows": len(split["heldout_row_ids"]),
            }},
        })
    if mutate is not None:
        mutate(rows)
    path = tmp_path / "handoff" / "evidence.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _loaded_splits(tmp_path, rows=None):
    path = tmp_path / "split_provenance.jsonl"
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in (rows or _split_rows())), encoding="utf-8"
    )
    return stage2_workflow._load_stage2_splits(
        provenance_path=path, dataset_rows=8, outer_fold_ids=[1, 2],
    )


def test_stage2_splits_are_verified_through_materialized_handoff_sources(tmp_path):
    source = tmp_path / "components" / "tfidf" / "evidence.jsonl"
    source.parent.mkdir(parents=True)
    producer_rows = [
        {"outer_fold": split["outer_fold"], "inner_fold": None, "scope": "full_outer_train",
         "fit_row_ids": split["fit_row_ids"], "heldout_row_ids": split["heldout_row_ids"]}
        for split in _split_rows()
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in producer_rows), encoding="utf-8")
    handoff = tmp_path / "handoff" / "evidence.jsonl"
    handoff.parent.mkdir(parents=True)
    handoff.write_text(
        "".join(
            json.dumps({"source": "stage1_architecture", "outer_fold": fold, "inner_fold": None,
                        "scope": "full_outer_train",
                        "evidence": {"architecture": "tfidf_topics", "occurrence": {}}}) + "\n"
            for fold in (1, 2)
        ),
        encoding="utf-8",
    )
    (handoff.parent / "index.json").write_text(
        json.dumps({"sources": {"tfidf": "../components/tfidf/evidence.jsonl"}}), encoding="utf-8"
    )
    splits = _loaded_splits(tmp_path)
    stage2_workflow._verify_stage2_splits_against_handoff(
        splits, handoff_path=handoff, outer_fold_ids=[1, 2],
    )
    reshuffled = _split_rows()
    reshuffled[0]["fit_row_ids"], reshuffled[0]["heldout_row_ids"] = [0, 5, 6, 7], [4, 1, 2, 3]
    reshuffled[1]["fit_row_ids"], reshuffled[1]["heldout_row_ids"] = [4, 1, 2, 3], [0, 5, 6, 7]
    with pytest.raises(RuntimeError, match="disagrees with the Stage 1 handoff"):
        stage2_workflow._verify_stage2_splits_against_handoff(
            _loaded_splits(tmp_path, reshuffled), handoff_path=handoff, outer_fold_ids=[1, 2],
        )
    source.unlink()
    with pytest.raises(FileNotFoundError, match="component evidence"):
        stage2_workflow._verify_stage2_splits_against_handoff(
            splits, handoff_path=handoff, outer_fold_ids=[1, 2],
        )


def test_dataset_backed_stage2_refuses_to_resample_missing_splits(tmp_path):
    handoff = _raw_handoff(tmp_path, _split_rows())
    dataset = pd.DataFrame({
        "patient_id": [f"p{row}" for row in range(8)],
        "clinical_text": ["ECOG 1."] * 8,
        "treatment_indicator": [0, 1] * 4,
        "outcome_indicator": [0, 0, 1, 1] * 2,
    })
    runner = _runner(
        extraction_llm=Stage2ExtractionLLMConfig(endpoint="http://extract.test/v1", model="small"),
    )
    runner._load_or_compile_evidence = lambda **_kwargs: (
        [{"outer_fold": 1, "packet_id": "a"}, {"outer_fold": 2, "packet_id": "b"}], {},
    )
    with pytest.raises(FileNotFoundError, match="will not recreate outer folds"):
        runner.run(
            handoff_path=handoff, output_dir=tmp_path / "stage2", dataset=dataset,
            split_provenance_path=tmp_path / "missing_split_provenance.jsonl",
        )
