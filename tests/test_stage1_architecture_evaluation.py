from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oci.evaluation.stage1 import (
    _association_metrics,
    _lexical_metrics,
    _materialize_or_load_manifest,
    _native_evidence_metrics,
    _outer_heldout,
    _stability_metrics,
    evaluate_stage1_architectures,
)
from oci.inference.stage1_architecture_artifacts import (
    materialize_stage1_architecture_artifacts,
)


def test_posthoc_evaluation_uses_frozen_architecture_artifacts(tmp_path: Path):
    run_dir = tmp_path / "run"
    dataset_path = tmp_path / "dataset.parquet"
    metadata_path = tmp_path / "metadata.json"
    rows = 12
    age = np.arange(rows, dtype=float)
    dataset = pd.DataFrame(
        {
            "patient_id": [f"p{index}" for index in range(rows)],
            "clinical_text": [f"age marker {index}" for index in range(rows)],
            "treatment_indicator": [0, 1] * (rows // 2),
            "outcome_indicator": [0, 0, 1, 1] * (rows // 4),
            "true_age": age,
            "true_marker": ["low"] * 6 + ["high"] * 6,
            "true_ite_prob": np.linspace(-0.2, 0.3, rows),
        }
    )
    dataset.to_parquet(dataset_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "features": [
                    {
                        "name": "age",
                        "type": "continuous",
                        "description": "patient age in years",
                        "roles": ["confounder"],
                    },
                    {
                        "name": "marker",
                        "type": "categorical",
                        "categories": ["low", "high"],
                        "roles": ["effect_modifier"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    run_dir.mkdir()
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "dataset": str(dataset_path),
                "treatment_column": "treatment_indicator",
                "outcome_column": "outcome_indicator",
                "outcome_type": "binary",
            }
        ),
        encoding="utf-8",
    )
    evidence_path = run_dir / "components" / "text_models" / "evidence.jsonl"
    evidence_path.parent.mkdir(parents=True)
    raw = {
        "source": "text_models",
        "outer_fold": 1,
        "inner_fold": None,
        "scope": "full_outer_train",
        "evidence": {
            "importance": {
                "views": [
                    {
                        "view_name": "word_linear",
                        "treatment_positive": [{"feature": "age marker", "score": 2.0}],
                        "outcome_positive": [],
                        "pseudo_target_positive": [],
                    }
                ]
            }
        },
    }
    evidence_path.write_text(json.dumps(raw) + "\n", encoding="utf-8")
    score_path = (
        run_dir
        / "components"
        / "text_models"
        / "outer_001_full"
        / "worker_artifacts"
        / "fold_000001"
        / "predictions.parquet"
    )
    score_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "_oci_row_id": np.arange(rows),
            "outer_fold": 1,
            "inner_fold": None,
            "scope": "full_outer_train",
            "split_role": "test_outer_train_fit",
            "source_name": "bow__linear__nuisance",
            "architecture": "bow_nuisance",
            "e_hat": np.asarray(dataset["treatment_indicator"], dtype=float) * 0.8 + 0.1,
            "m_hat": np.asarray(dataset["outcome_indicator"], dtype=float) * 0.8 + 0.1,
            "age_score": age,
        }
    ).to_parquet(score_path, index=False)
    materialize_stage1_architecture_artifacts(
        output_dir=run_dir,
        raw_handoff_rows=[raw],
        selected_architectures=("bow_nuisance",),
        source_artifacts={"text_models": evidence_path},
        selection_mode="explicit",
    )

    result = evaluate_stage1_architectures(
        run_dir=run_dir,
        metadata_path=metadata_path,
    )

    output = Path(result["output_dir"])
    metrics = [
        json.loads(line)
        for line in (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(
        row["metric"] == "best_score_abs_spearman"
        and row["target"] == "true_age"
        and row["value"] == 1.0
        for row in metrics
    )
    assert any(
        row["metric"] == "best_treatment_orientation_free_auc" and row["value"] == 1.0
        for row in metrics
    )
    manifest = json.loads((output / "evaluation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["oracle_columns_available_to_stage1"] is False
    assert manifest["stage1_artifacts_frozen_before_oracle_load"] is True
    assert manifest["stage1_evidence_sha256"]["bow_nuisance"]


def test_compact_occurrence_metrics_use_multiplicity_and_reference_summaries():
    evidence = [
        {
            "outer_fold": 1,
            "inner_fold": 1,
            "occurrence": {
                "text": "pretreatment performance status",
                "evidence_kind": "neural_query_ngram",
                "axes": ["residual_effect"],
                "polarity": "unsigned",
                "source_families": ["neural_query_moments"],
                "architecture": "neural_query_moments",
                "reference": {"inner_fold": 1},
                "reference_summaries": [
                    {
                        "inner_fold": 1,
                        "query_id": "treatment_query",
                        "bank": "treatment",
                        "row_id": 7,
                        "occurrence_count": 1,
                    },
                    {
                        "inner_fold": 2,
                        "query_id": "outcome_query",
                        "bank": "outcome",
                        "row_id": 8,
                        "occurrence_count": 1,
                    },
                ],
                "raw_occurrence_count": 2,
                "details": {
                    "query_id": "treatment_query",
                    "bank": "treatment",
                },
                "scores": {},
                "patient_row_id": 7,
            },
        }
    ]

    metrics = [
        *_lexical_metrics("neural_query_moments", evidence, []),
        *_stability_metrics("neural_query_moments", evidence),
        *_native_evidence_metrics("neural_query_moments", evidence),
    ]
    by_name = {row["metric"]: row for row in metrics}

    assert by_name["occurrence_count"]["value"] == 2.0
    assert by_name["occurrence_count"]["n"] == 2
    assert by_name["mean_inner_fold_jaccard"]["value"] == 1.0
    assert by_name["mean_inner_fold_jaccard"]["n"] == 1
    assert by_name["represented_query_count"]["value"] == 2.0
    assert by_name["represented_query_count"]["n"] == 2
    assert by_name["query_bank_coverage"]["value"] == 2.0 / 3.0
    assert by_name["witness_patient_coverage"]["value"] == 2.0


def test_outer_heldout_rejects_contextless_external_tfidf_rows():
    inner_context = pd.DataFrame(
        {"_oci_row_id": [1], "prediction_scope": ["external_heldout"]}
    )
    canonical_outer = inner_context.assign(
        outer_fold=1,
        honest_outer_holdout=True,
    )

    assert _outer_heldout(inner_context).empty
    assert _outer_heldout(canonical_outer)["_oci_row_id"].tolist() == [1]


def test_legacy_backfill_resolves_source_artifacts_from_handoff_index(tmp_path: Path):
    run_dir = tmp_path / "run"
    source = run_dir / "components" / "text_models" / "evidence.jsonl"
    handoff_dir = run_dir / "handoff"
    source.parent.mkdir(parents=True)
    handoff_dir.mkdir(parents=True)
    source_row = {
        "outer_fold": 1,
        "inner_fold": None,
        "scope": "full_outer_train",
        "importance": {
            "views": [
                {
                    "view_name": "word_linear",
                    "treatment_positive": [{"feature": "performance status", "score": 1.0}],
                    "outcome_positive": [],
                    "pseudo_target_positive": [],
                }
            ]
        },
    }
    source.write_text(json.dumps(source_row) + "\n", encoding="utf-8")
    handoff_row = {
        "source": "text_models",
        "outer_fold": 1,
        "inner_fold": None,
        "scope": "full_outer_train",
        "evidence": source_row,
    }
    (handoff_dir / "evidence.jsonl").write_text(
        json.dumps(handoff_row) + "\n",
        encoding="utf-8",
    )
    (handoff_dir / "index.json").write_text(
        json.dumps(
            {
                "source_storage": "referenced_without_copy",
                "sources": {
                    "text_models": "../components/text_models/evidence.jsonl"
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = _materialize_or_load_manifest(run_dir)
    evidence_row = json.loads(
        (run_dir / "stage1_architectures" / "bow_nuisance" / "evidence.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )

    assert manifest["source_artifacts"]["text_models"]["path"] == str(
        source.relative_to(run_dir)
    )
    assert evidence_row["lineage"]["source_artifact"] == manifest["source_artifacts"][
        "text_models"
    ]


@pytest.mark.parametrize(
    ("outcome_type", "oracle_columns", "expected_target", "applicable"),
    [
        ("binary", ("true_ite_prob",), "true_ite_prob", True),
        ("binary", ("true_ite", "true_ite_prob"), "true_ite", True),
        ("continuous", ("true_ite",), "true_ite", True),
        ("continuous", ("true_ite_prob",), None, False),
    ],
)
def test_true_ite_metrics_resolve_the_outcome_scale_oracle_column(
    outcome_type, oracle_columns, expected_target, applicable
):
    rows = np.arange(12)
    scores = pd.DataFrame(
        {
            "_oci_row_id": rows,
            "scope": "full_outer_train",
            "inner_fold": None,
            "tau_hat": rows / 10.0,
        }
    )
    dataset = pd.DataFrame({"_oci_row_id": rows})
    for column in oracle_columns:
        # true_ite tracks tau_hat exactly; the probability column is reversed,
        # so the test also shows which column was correlated.
        dataset[column] = rows if column == "true_ite" else -rows

    metrics = [
        row
        for row in _association_metrics(
            "bow_r_loss", scores, dataset, [], outcome_type=outcome_type
        )
        if row["metric_family"] == "true_ite"
    ]

    assert {row["metric"] for row in metrics} == {
        "best_tau_abs_pearson",
        "best_tau_abs_spearman",
    }
    for row in metrics:
        assert row["target"] == expected_target
        if applicable:
            assert row["value"] == pytest.approx(1.0)
        else:
            assert row["value"] is None
            assert "lacks outcome-scale true_ite" in row["reason"]
