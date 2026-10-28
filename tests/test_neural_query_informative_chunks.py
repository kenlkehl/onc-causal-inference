"""Neural queries are fitted, scored, and retrieved on informative chunks only."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from oci.config import TfidfNuisanceStackScientificConfig
from oci.inference import embedding_contrast_discovery
from oci.inference import neural_cohort_witness as witness
from oci.inference import neural_query_agentic_forest as forest
from oci.inference import neural_query_discovery_runtime as runtime
from oci.inference import research_all_evidence_workflow as workflow
from oci.inference.neural_query_agentic_forest import (
    NeuralQueryAgenticForestConfig,
    build_query_evidence,
    informative_chunk_matrices,
)
from oci.inference.neural_query_evidence_contract import query_chunk_eligibility
from oci.models import concept_embedding_utils
from oci.models.concept_embedding_utils import (
    MIN_INFORMATIVE_CHUNK_ALNUM_CHARS,
    informative_chunk_text,
)

SOFT_HYPHENS = "­" * 300


@pytest.mark.parametrize(
    "text,expected",
    [
        (SOFT_HYPHENS, False),
        ("―" * 70, False),
        ("", False),
        ("3/4 (Severe)", False),
        ("45/100", False),
        ("ECOG 1, stable", False),
        ("ECOG 1, stable disease", True),
        ("Patient reports worsening dyspnea.", True),
    ],
)
def test_informative_chunk_rule(text, expected):
    assert MIN_INFORMATIVE_CHUNK_ALNUM_CHARS == 12
    assert informative_chunk_text(text) is expected


def test_embedding_and_neural_query_lanes_share_one_rule():
    assert (
        embedding_contrast_discovery.informative_chunk_text
        is concept_embedding_utils.informative_chunk_text
    )
    assert forest.informative_chunk_text is concept_embedding_utils.informative_chunk_text


def test_filtered_matrices_drop_ineligible_chunks_and_empty_patients_take_the_mean():
    mixed = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    clean = np.array([[0.6, 0.8]], dtype=np.float32)
    empty = np.array([[1.0, 0.0]], dtype=np.float32)
    filtered = informative_chunk_matrices(
        [mixed, clean, empty],
        [[SOFT_HYPHENS, "real clinical sentence"], ["another clinical note"], [""]],
    )
    np.testing.assert_array_equal(filtered[0], mixed[1:])
    assert filtered[1] is clean
    assert filtered[2].shape == (0, 2) and filtered[2].dtype == np.float32

    activations = witness.soft_retrieval_activations(
        filtered,
        np.array([[1.0, 0.0]], dtype=np.float32),
        temperature=0.05,
        device="cpu",
        patient_batch_size=8,
    )
    # The soft-hyphen chunk matched the query exactly; without it the patient's
    # only chunk is orthogonal. A patient without chunks takes the others' mean.
    np.testing.assert_allclose(activations[:, 0], [0.0, 0.6, 0.3], atol=1e-6)

    only_empty = informative_chunk_matrices([empty], [["45/100"]])
    assert witness.soft_retrieval_activations(
        only_empty, np.array([[1.0, 0.0]], dtype=np.float32),
        temperature=0.05, device="cpu", patient_batch_size=8,
    ).tolist() == [[0.0]]
    with pytest.raises(ValueError, match="chunk counts are inconsistent"):
        informative_chunk_matrices([mixed], [["one text"]])


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


def _evidence_kwargs():
    config = NeuralQueryAgenticForestConfig(
        evidence_top_patients=1, evidence_background_patients=None,
        evidence_ngram_stop_words=None, evidence_ngram_range_max=1,
    )
    return config, dict(
        bank="outcome", queries=np.array([[1.0, 0.0]], dtype=np.float32),
        query_records=[{"query_id": "outcome_query_001"}], row_ids=[0, 1, 2],
        chunk_matrices=[
            # The soft-hyphen chunk is the best match overall.
            np.array([[1.0, 0.0], [0.8, 0.6]], dtype=np.float32),
            # Would rank first, but has no informative chunk.
            np.array([[0.99, 0.141]], dtype=np.float32),
            np.array([[0.6, 0.8]], dtype=np.float32),
        ],
        all_chunk_texts=[
            [SOFT_HYPHENS, "informative witness passage"],
            ["3/4 (Severe)"],
            ["background patient passage"],
        ],
        device="cpu", seed=7,
    )


def test_witnesses_and_background_use_informative_chunks_only(monkeypatch):
    seen = {}
    original = forest._contrastive_ngrams

    def spy(foreground, background, **kwargs):
        seen["foreground"], seen["background"] = list(foreground), list(background)
        return original(foreground, background, **kwargs)

    monkeypatch.setattr(forest, "_contrastive_ngrams", spy)
    config, kwargs = _evidence_kwargs()
    evidence = build_query_evidence(config=config, **kwargs)[0]
    assert [(c["_oci_row_id"], c["chunk_index"]) for c in evidence["top_chunks"]] == [(0, 1)]
    assert evidence["top_chunks"][0]["similarity"] == pytest.approx(0.8)
    assert seen == {
        "foreground": ["informative witness passage"],
        "background": ["background patient passage"],
    }
    assert evidence["chunk_eligibility"] == query_chunk_eligibility(12)
    assert SOFT_HYPHENS not in json.dumps(evidence)


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


def test_patients_without_chunks_do_not_change_query_statistics():
    rng = np.random.default_rng(1)
    matrices = [rng.normal(size=(4, 6)).astype(np.float32) for _ in range(40)]
    target = np.tile([0.0, 1.0], 20)
    queries = rng.normal(size=(3, 6)).astype(np.float32)
    kwargs = dict(temperature=0.05, device="cpu", patient_batch_size=7)
    with_empty = witness.soft_retrieval_activations(
        matrices + [np.zeros((0, 6), dtype=np.float32)] * 4, queries, **kwargs
    )
    without = witness.soft_retrieval_activations(matrices, queries, **kwargs)
    # The mean spans every batch, not only the batch holding the empty rows.
    np.testing.assert_allclose(with_empty[:40], without, rtol=1e-6)
    np.testing.assert_allclose(with_empty[40:], np.broadcast_to(without.mean(0), (4, 3)), rtol=1e-5)
    scores = witness.standardized_direct_target_contrasts
    z_with = scores(with_empty, np.r_[target, [0.0, 1.0, 1.0, 1.0]], binary=True)
    z_without = scores(without, target, binary=True)
    np.testing.assert_allclose(
        z_with["standardized_scores"], z_without["standardized_scores"], rtol=0.05
    )
