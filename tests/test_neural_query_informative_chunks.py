"""Neural queries are fitted, scored, and retrieved on informative chunks only."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from oci.config import BoWViewConfig, TfidfNuisanceStackScientificConfig
from oci.inference import neural_cohort_witness as witness
from oci.inference import neural_query_discovery_runtime as runtime
from oci.inference import research_all_evidence_workflow as workflow
from oci.inference.neural_query_agentic_forest import (
    NeuralQueryAgenticForestConfig,
)
from oci.inference.neural_query_evidence_contract import query_chunk_eligibility

SOFT_HYPHENS = "­" * 300


def test_initialization_pool_excludes_patients_without_chunks(monkeypatch):
    captured = {}
    original = witness._torch_spherical_kmeans

    def spy(flat_chunks, **kwargs):
        captured["flat"] = flat_chunks.detach().cpu().numpy()
        return original(flat_chunks, **kwargs)

    monkeypatch.setattr(witness, "_torch_spherical_kmeans", spy)
    rng = np.random.default_rng(0)
    real = [rng.normal(size=(3, 6)).astype(np.float32) for _ in range(12)]
    matrices = real + [np.zeros((0, 6), dtype=np.float32) for _ in range(12)]
    target = np.tile([0.0, 1.0], 12)
    config = witness.NeuralCohortWitnessConfig(
        n_prototypes=2, initial_pool_size=4, epochs=2, kmeans_iterations=3
    )
    result = witness.fit_soft_target_queries(
        matrices, target, binary=True, config=config, seed=3, device="cpu"
    )
    assert captured["flat"].shape == (36, 6)
    assert np.all(np.linalg.norm(captured["flat"], axis=1) > 0)
    np.testing.assert_allclose(np.linalg.norm(result["initial_queries"], axis=1), 1.0, rtol=1e-5)
    activations = result["train_activations"]
    np.testing.assert_allclose(
        activations[12:], np.broadcast_to(activations[:12].mean(axis=0), (12, 2)), rtol=1e-5
    )
    assert np.isfinite(result["loss_history"]).all()
    with pytest.raises(ValueError, match="nonzero chunk"):
        witness.fit_soft_target_queries(
            matrices[12:], target[12:], binary=True, config=config, seed=3, device="cpu"
        )


def test_workflow_fits_and_scores_queries_on_informative_chunks(tmp_path, monkeypatch):
    seen = {}

    def discover(**kwargs):
        seen["chunks"] = [np.array(matrix) for matrix in kwargs["chunks"]]
        rows = len(kwargs["row_ids"])
        return {
            "banks": {
                bank: {
                    "queries": np.eye(2, dtype=np.float32),
                    "train_activations": np.zeros((rows, 2), dtype=np.float32),
                    "records": [
                        {"query_id": f"{bank}_query_{index:03d}"} for index in (1, 2)
                    ],
                    "consensus": {"ranking": {}},
                }
                for bank in runtime.BANKS
            }
        }

    class Generator:
        def chunk_matrices(self, rows):
            return [np.eye(2, dtype=np.float32) for _ in rows]

        def chunk_texts(self, rows):
            return [[SOFT_HYPHENS, f"clinical passage for row {row}"] for row in rows]

    monkeypatch.setattr(runtime, "fit_context_query_discovery", discover)
    fold_dir = tmp_path / "outer_001_full"
    evidence = workflow._run_one_neural_query_context(
        dataset=pd.DataFrame(
            {"text": [f"note {i}" for i in range(8)], "t": [0, 1] * 4, "y": [1, 0] * 4}
        ),
        config=SimpleNamespace(
            text_column="text", treatment_column="t", outcome_column="y",
            outcome_type="binary", seed=1,
        ),
        applied_config=SimpleNamespace(
            architecture=SimpleNamespace(
                multi_model_forest=SimpleNamespace(
                    bow_views=[], nuisance_folds=2,
                    tfidf_topic=SimpleNamespace(
                        nuisance_stack_scientific=TfidfNuisanceStackScientificConfig()
                    ),
                )
            )
        ),
        neural_query_config=NeuralQueryAgenticForestConfig(
            evidence_top_patients=1, evidence_background_patients=None,
            evidence_ngram_stop_words=None,
        ),
        generator=Generator(),
        spec={"scope_id": "outer_001_full", "outer_fold": 1, "fold_key": 1,
              "scope": "full_outer_train", "train_idx": list(range(6)),
              "heldout_idx": [6, 7]},
        fold_dir=fold_dir,
        device="cpu",
    )

    for matrix in seen["chunks"]:
        np.testing.assert_array_equal(matrix, [[0.0, 1.0]])
    scores = pd.read_parquet(fold_dir / "scores.parquet")
    heldout = scores[scores["split_role"] == "heldout_query_projection"]
    # Query 1 points at the soft-hyphen chunk, which is never scored.
    np.testing.assert_allclose(heldout["outcome__query_001"], 0.0, atol=1e-6)
    np.testing.assert_allclose(heldout["outcome__query_002"], 1.0, atol=1e-6)
    for query in evidence["evidence"]:
        assert [chunk["chunk_index"] for chunk in query["top_chunks"]] == [1]
        assert query["chunk_eligibility"] == query_chunk_eligibility(12)


def test_real_context_fit_finds_the_signal_chunk_and_cites_only_fit_rows(tmp_path):
    # Each note has a constant chunk, a chunk that carries the signal behind
    # treatment and outcome, and an uninformative chunk.
    rng = np.random.default_rng(0)
    n, fit_rows = 120, list(range(96))
    signal = rng.random(n) < 0.5
    treatment = (rng.random(n) < np.where(signal, 0.8, 0.2)).astype(int)
    outcome = (rng.random(n) < np.where(signal, 0.7, 0.3)).astype(int)
    chunk_texts = [
        [
            "patient seen in clinic for follow up visit",
            "egfr mutation detected on tumor testing"
            if value
            else "routine laboratory values reviewed",
            SOFT_HYPHENS,
        ]
        for value in signal
    ]
    basis = np.eye(8, dtype=np.float32)

    class Generator:
        def chunk_matrices(self, rows):
            return [
                (
                    np.stack([basis[0], basis[1] if signal[row] else basis[2], basis[3]])
                    + np.random.default_rng(row).normal(scale=0.05, size=(3, 8))
                ).astype(np.float32)
                for row in rows
            ]

        def chunk_texts(self, rows):
            return [chunk_texts[row] for row in rows]

    fold_dir = tmp_path / "outer_001_full"
    evidence = workflow._run_one_neural_query_context(
        dataset=pd.DataFrame(
            {"text": [" ".join(texts[:2]) for texts in chunk_texts], "t": treatment, "y": outcome}
        ),
        config=SimpleNamespace(
            text_column="text", treatment_column="t", outcome_column="y",
            outcome_type="binary", seed=1,
        ),
        applied_config=SimpleNamespace(
            architecture=SimpleNamespace(
                multi_model_forest=SimpleNamespace(
                    bow_views=[BoWViewConfig(name="unigram", min_df=1, ngram_range_max=1)],
                    nuisance_folds=2,
                    tfidf_topic=SimpleNamespace(
                        nuisance_stack_scientific=TfidfNuisanceStackScientificConfig()
                    ),
                )
            )
        ),
        neural_query_config=NeuralQueryAgenticForestConfig(
            treatment_query_count=2, outcome_query_count=2, effect_query_count=2,
            query_inner_folds=2, initial_pool_size=4, query_epochs=3, final_refit_epochs=2,
            kmeans_iterations=3, consensus_kmeans_n_init=1, evidence_top_patients=3,
            evidence_background_patients=None, evidence_ngram_stop_words=None,
        ),
        generator=Generator(),
        spec={"scope_id": "outer_001_full", "outer_fold": 1, "fold_key": 1,
              "scope": "full_outer_train", "train_idx": fit_rows,
              "heldout_idx": list(range(96, n))},
        fold_dir=fold_dir,
        device="cpu",
    )

    complete = json.loads((fold_dir / "complete.json").read_text(encoding="utf-8"))
    assert all((fold_dir / name).is_file() for name in complete["artifacts"])
    queries = np.load(fold_dir / "queries.npz")
    scores = pd.read_parquet(fold_dir / "scores.parquet")
    heldout = scores[scores["split_role"] == "heldout_query_projection"]
    assert scores.loc[scores.index.difference(heldout.index), "_oci_row_id"].tolist() == fit_rows
    assert heldout["_oci_row_id"].tolist() == list(range(96, n))
    records = json.loads((fold_dir / "query_records.json").read_text(encoding="utf-8"))
    for bank in runtime.BANKS:
        norms = np.linalg.norm(queries[f"{bank}_queries"], axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-5)
        columns = [f"{bank}__query_001", f"{bank}__query_002"]
        assert np.isfinite(scores[columns].to_numpy()).all()
        assert records["ranking"]["banks"][bank]["groups_without_heldout_statistics"] == 0
        assert [record["rank"] for record in records["banks"][bank]] == [1, 2]
        if bank != "effect":  # The data have no effect modification.
            correlations = [np.corrcoef(heldout[column], signal[96:])[0, 1] for column in columns]
            assert max(np.abs(correlations)) > 0.9
    for query in evidence["evidence"]:
        assert {chunk["_oci_row_id"] for chunk in query["top_chunks"]} <= set(fit_rows)
        assert all(chunk["chunk_index"] != 2 for chunk in query["top_chunks"])
        assert len(query["top_contrastive_ngrams"]) <= 30
    for bank in ("treatment", "outcome"):
        terms = {
            term["term"]
            for query in evidence["evidence"]
            if query["bank"] == bank
            for term in query["top_contrastive_ngrams"]
        }
        assert "egfr" in terms
