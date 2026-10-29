"""Neural queries are fitted, scored, and retrieved on informative chunks only."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from oci.config import TfidfNuisanceStackScientificConfig
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
