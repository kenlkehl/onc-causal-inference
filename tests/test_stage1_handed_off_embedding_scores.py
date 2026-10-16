"""Evaluation scores the embedding directions actually handed to Stage 2 (S1-04)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oci.config import (
    AppliedInferenceConfig,
    BoWViewConfig,
    ClusterLocalEmbeddingScientificConfig,
    EmbeddingContrastDiscoveryConfig,
    ModelArchitectureConfig,
    MultiModelForestConfig,
)
from oci.evaluation.stage1 import evaluate_stage1_architectures
from oci.inference import embedding_contrast_discovery
from oci.inference import multi_model_forest_stage1
from oci.inference.multi_model_forest_stage1 import run_multi_model_forest_handoff_context
from oci.inference.stage1_architecture_artifacts import (
    EMBEDDING_DIRECTION_SCORE_SOURCES,
    embedding_direction_score_column,
    materialize_stage1_architecture_artifacts,
)
from oci.models import concept_embedding_cache

VOCAB = [
    "fatigue", "cough", "stage", "four", "ecog", "one", "nausea", "pain",
    "egfr", "alk", "renal", "liver", "marker", "high", "low", "dyspnea",
]
_VECTORS = {word: np.random.default_rng(index).normal(size=8) for index, word in enumerate(VOCAB)}


class _BagOfWordsEncoder:
    max_seq_length = None

    def get_sentence_embedding_dimension(self) -> int:
        return 8

    def encode(self, chunks, **_kwargs):
        rows = []
        for chunk in chunks:
            vector = sum((_VECTORS.get(word, np.zeros(8)) for word in chunk.split()), np.zeros(8))
            rows.append(vector / max(float(np.linalg.norm(vector)), 1e-9))
        return np.asarray(rows, dtype=np.float32)


def _cluster_config() -> ClusterLocalEmbeddingScientificConfig:
    return ClusterLocalEmbeddingScientificConfig(
        requested_cluster_count=3,
        maximum_components_per_family=2,
        loading_evidence_capacity=None,
        minimum_cluster_size=8,
        minimum_group_size=3,
        minimum_cell_size=1,
        minimum_distinct_local_clusters_per_family=2,
        minimum_numerical_rank_per_family=2,
        computation_dtype="float32",
        normalize_patient_embeddings=True,
        normalization_epsilon=1e-12,
        zero_vector_policy="reject",
        kmeans_init="k-means++",
        kmeans_max_iter=100,
        kmeans_batch_size_lower_bound=16,
        kmeans_batch_size_upper_bound=64,
        kmeans_verbose=0,
        kmeans_compute_labels=True,
        kmeans_tol=0.0,
        kmeans_max_no_improvement=10,
        kmeans_init_size=None,
        kmeans_n_init=3,
        kmeans_reassignment_ratio=0.01,
        svd_full_matrices=False,
        svd_compute_uv=True,
        svd_hermitian=False,
        svd_sign_canonicalization_policy="largest_absolute_coordinate_positive_v1",
        svd_rank_tolerance_dtype="float32",
        svd_rank_tolerance_multiplier=1.0,
    )


@pytest.fixture
def embedding_context(tmp_path, monkeypatch):
    monkeypatch.setattr(
        concept_embedding_cache, "load_sentence_transformer", lambda *a, **k: _BagOfWordsEncoder()
    )
    monkeypatch.setattr(
        embedding_contrast_discovery,
        "load_sentence_transformer",
        lambda *a, **k: _BagOfWordsEncoder(),
    )
    rng = np.random.default_rng(0)
    n = 96
    outcome = rng.integers(0, 2, size=n)
    dataset = pd.DataFrame(
        {
            "clinical_text": [
                " ".join(rng.choice(VOCAB, size=16))
                + (" marker high" if outcome[index] else " marker low")
                for index in range(n)
            ],
            "treatment_indicator": np.tile([0, 1], n // 2),
            "outcome_indicator": outcome,
        }
    )
    config = AppliedInferenceConfig(
        seed=7,
        outcome_type="binary",
        architecture=ModelArchitectureConfig(
            model_type="multi_model_forest",
            multi_model_forest=MultiModelForestConfig(
                nuisance_folds=2,
                effect_folds=2,
                top_n_features=5,
                feature_discovery_methods=["bow", "embedding_contrast"],
                matched_pair_uplift_enabled=False,
                bow_views=[BoWViewConfig(name="tiny", min_df=1, max_df=1.0)],
                embedding_contrast=EmbeddingContrastDiscoveryConfig(
                    model_name="bag-of-words-test-encoder",
                    cache_dir=str(tmp_path / "cache"),
                    max_seq_length=None,
                    chunk_size_words=4,
                    chunk_overlap_words=1,
                    top_k_chunks_per_tail=3,
                    include_bow_phrases_as_concepts=False,
                    cluster_local_scientific=_cluster_config(),
                    batch_size=8,
                ),
            ),
        ),
    )
    return dataset, config


def test_text_model_context_scores_heldout_rows_on_the_handed_off_directions(
    tmp_path, embedding_context, monkeypatch
):
    dataset, config = embedding_context
    captured: dict = {}
    original = embedding_contrast_discovery.EmbeddingContrastEvidenceGenerator.build_evidence_with_directions

    def capture(self, **kwargs):
        evidence, directions = original(self, **kwargs)
        captured["generator"] = self
        captured["directions"] = directions
        return evidence, directions

    monkeypatch.setattr(
        embedding_contrast_discovery.EmbeddingContrastEvidenceGenerator,
        "build_evidence_with_directions",
        capture,
    )
    rows = np.arange(len(dataset))
    row = run_multi_model_forest_handoff_context(
        dataset=dataset,
        config=config,
        context={
            "train_idx": rows[:72],
            "heldout_idx": rows[72:],
            "fold_key": 1,
            "outer_fold": 1,
            "scope": "full_outer_train",
        },
        handoff_dir=tmp_path / "handoff",
        device="cpu",
        num_workers=1,
    )
    predictions = pd.read_parquet(
        tmp_path / "handoff" / "worker_artifacts" / "fold_000001" / "predictions.parquet"
    )

    # Only the handed-off sources score embedding architectures.
    embedding_rows = predictions[predictions["source_name"].str.startswith("embedding__")]
    assert set(embedding_rows["source_name"]) == set(EMBEDDING_DIRECTION_SCORE_SOURCES.values())
    assert set(embedding_rows["split_role"]) == {"test_outer_train_fit"}
    assert set(embedding_rows["_oci_row_id"]) == set(range(72, 96))

    contrasts = row["embedding_contrast_evidence"]["contrasts"]
    directions = captured["directions"]
    generator = captured["generator"]
    assert [item["name"] for item in directions] == [record["name"] for record in contrasts]
    for item in directions:
        record = contrasts[item["contrast_index"]]
        # The captured vector is the one that ranked the handed-off chunks.
        for chunk in record["positive_aligned_chunks"]:
            matrix = generator.chunk_matrices([chunk["row_id"]])[0]
            assert float(matrix[chunk["chunk_index"]] @ item["direction"]) == pytest.approx(
                chunk["score"], abs=1e-6
            )
        frame = embedding_rows[
            embedding_rows["source_name"] == EMBEDDING_DIRECTION_SCORE_SOURCES[item["architecture"]]
        ].sort_values("_oci_row_id")
        expected_max = [
            float(np.max(matrix @ item["direction"]))
            for matrix in generator.chunk_matrices(frame["_oci_row_id"].tolist())
        ]
        column = embedding_direction_score_column(item["name"], "max_chunk_cosine")
        assert frame[column].to_numpy() == pytest.approx(expected_max, abs=1e-6)
    assert {item["architecture"] for item in directions} == set(EMBEDDING_DIRECTION_SCORE_SOURCES)


def test_duplicate_embedding_direction_pipeline_is_gone():
    runner = multi_model_forest_stage1.MultiModelForestStage1Runner
    for name in (
        "_embedding_feature_bundle",
        "_embedding_directions",
        "_cluster_embedding_directions",
        "_svd_cluster_components",
        "_chunk_similarity_features",
    ):
        assert not hasattr(runner, name)
    assert not hasattr(multi_model_forest_stage1, "_weighted_binary_direction")


def _evaluation_run(tmp_path: Path, *, handed_off: bool, legacy: bool) -> Path:
    run_dir = tmp_path / "run"
    rows = 12
    marker = np.asarray([0, 1] * (rows // 2), dtype=float)
    dataset = pd.DataFrame(
        {
            "clinical_text": [f"note {index} marker" for index in range(rows)],
            "treatment_indicator": [0, 1] * (rows // 2),
            "outcome_indicator": [0, 0, 1, 1] * (rows // 4),
            "true_marker": marker,
        }
    )
    dataset_path = tmp_path / "dataset.parquet"
    dataset.to_parquet(dataset_path, index=False)
    (tmp_path / "metadata.json").write_text(
        json.dumps({"features": [{"name": "marker", "type": "continuous"}]}),
        encoding="utf-8",
    )
    run_dir.mkdir()
    (run_dir / "run_config.json").write_text(
        json.dumps({"dataset": str(dataset_path), "seed": 3}), encoding="utf-8"
    )
    evidence_path = run_dir / "components" / "text_models" / "evidence.jsonl"
    evidence_path.parent.mkdir(parents=True)
    raw = {
        "source": "text_models",
        "outer_fold": 1,
        "inner_fold": None,
        "scope": "full_outer_train",
        "evidence": {
            "embedding_contrast_evidence": {
                "contrasts": [
                    {
                        "name": "treatment",
                        "contrast_family": "marginal",
                        "positive_aligned_chunks": [
                            {"row_id": 0, "chunk_index": 0, "score": 0.9, "text": "marker strong"}
                        ],
                        "negative_aligned_chunks": [],
                    },
                    {
                        "name": "dina_score",
                        "contrast_family": "dina_score",
                        "positive_aligned_chunks": [
                            {"row_id": 1, "chunk_index": 0, "score": 0.8, "text": "marker weak"}
                        ],
                        "negative_aligned_chunks": [],
                    },
                ]
            }
        },
    }
    evidence_path.write_text(json.dumps(raw) + "\n", encoding="utf-8")
    frames = []
    heldout = np.arange(rows)
    if handed_off:
        frames.append(
            pd.DataFrame(
                {
                    "_oci_row_id": heldout,
                    "outer_fold": 1,
                    "inner_fold": None,
                    "scope": "full_outer_train",
                    "split_role": "test_outer_train_fit",
                    "source_name": EMBEDDING_DIRECTION_SCORE_SOURCES["embedding_whole_cohort"],
                    "architecture": "embedding_whole_cohort",
                    embedding_direction_score_column("treatment", "max_chunk_cosine"): marker,
                    embedding_direction_score_column("treatment", "mean_chunk_cosine"): marker / 2,
                    embedding_direction_score_column("dina_score", "max_chunk_cosine"): -marker,
                    embedding_direction_score_column("dina_score", "mean_chunk_cosine"): marker,
                }
            )
        )
    if legacy:
        frames.append(
            pd.DataFrame(
                {
                    "_oci_row_id": heldout,
                    "outer_fold": 1,
                    "inner_fold": None,
                    "scope": "full_outer_train",
                    "split_role": "test_outer_train_fit",
                    "source_name": "embedding__global_marginal_treatment__embedding__"
                    "global_treatment_contrast__mean_cosine",
                    "architecture": "embedding_whole_cohort",
                    "score": np.linspace(0, 1, rows),
                }
            )
        )
    score_path = (
        run_dir / "components" / "text_models" / "outer_001_full" / "worker_artifacts"
        / "fold_000001" / "predictions.parquet"
    )
    score_path.parent.mkdir(parents=True)
    pd.concat(frames, ignore_index=True).to_parquet(score_path, index=False)
    materialize_stage1_architecture_artifacts(
        output_dir=run_dir,
        raw_handoff_rows=[raw],
        selected_architectures=("embedding_whole_cohort",),
        source_artifacts={"text_models": evidence_path},
        selection_mode="explicit",
    )
    return run_dir


def _metrics(result) -> list[dict]:
    path = Path(result["output_dir"]) / "metrics.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_evaluation_scores_handed_off_direction_columns_and_ignores_legacy_rows(tmp_path):
    run_dir = _evaluation_run(tmp_path, handed_off=True, legacy=True)

    result = evaluate_stage1_architectures(
        run_dir=run_dir, metadata_path=tmp_path / "metadata.json", null_draws=10
    )
    metrics = _metrics(result)

    association = [
        row for row in metrics
        if row["metric_family"] == "common_oracle_association" and row["target"] == "true_marker"
    ]
    assert association and all(row["score"].startswith("direction__") for row in association)
    assert association[0]["value"] == pytest.approx(1.0)
    coverage = [row for row in metrics if row["metric"] == "handed_off_direction_score_coverage"]
    assert [row["value"] for row in coverage] == [1.0]
    assert coverage[0]["n"] == 2


def test_evaluation_reports_legacy_duplicate_scores_as_unscored(tmp_path):
    run_dir = _evaluation_run(tmp_path, handed_off=False, legacy=True)

    result = evaluate_stage1_architectures(
        run_dir=run_dir, metadata_path=tmp_path / "metadata.json", null_draws=0
    )
    metrics = _metrics(result)

    unscored = [row for row in metrics if row["metric"] == "best_score_association"]
    assert len(unscored) == 1
    assert "retired duplicate direction pipeline" in unscored[0]["reason"]
    assert not any(row["score"] == "score" for row in metrics)
    manifest = json.loads(
        (Path(result["output_dir"]) / "evaluation_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["legacy_embedding_score_artifacts_ignored"]["embedding_whole_cohort"]
