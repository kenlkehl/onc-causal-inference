from dataclasses import replace
import json

import numpy as np
import pytest

from oci.inference.neural_query_agentic_forest import (
    NeuralQueryAgenticForestConfig,
    NeuralQueryEvidenceCapacityOverflowError,
    build_query_evidence,
)
from oci.inference.neural_query_evidence_contract import (
    query_retrieval_policy,
    query_term_normalization,
    validate_query_retrieval_policy,
)


def fixture():
    config = NeuralQueryAgenticForestConfig(
        evidence_top_patients=1, evidence_ngram_top_patients=1,
        evidence_background_patients=None, evidence_top_ngrams=None,
        evidence_ngram_stop_words=None, evidence_ngram_range_min=1, evidence_ngram_range_max=1,
    )
    return config, dict(
        bank="effect", queries=np.array([[1., 0.]], dtype=np.float32),
        query_records=[{"query_id": "effect_query_001"}], row_ids=[0, 2],
        chunk_matrices=[
            np.array([[1., 0.], [.8, .6], [0., 1.]], dtype=np.float32),
            np.array([[.4, .916515], [0., 1.]], dtype=np.float32),
        ],
        all_chunk_texts=[
            ["matching_signal " + "complete " * 400, "secondrank_signal", "unrelated_history"],
            ["HELDOUT_SENTINEL"],
            ["background_signal", "background_unrelated_history"],
        ], device="cpu", seed=7,
    )


def test_selects_matching_chunks_and_recomputes_terms_from_both_selected_arms():
    config, kwargs = fixture()
    evidence = build_query_evidence(config=config, **kwargs)[0]
    assert [r["chunk_index"] for r in evidence["top_chunks"]] == [0]
    assert evidence["top_chunks"][0]["text"] == kwargs["all_chunk_texts"][0][0]
    serialized = json.dumps(evidence)
    for excluded in ["secondrank_signal", "unrelated_history", "HELDOUT_SENTINEL"]:
        assert excluded not in serialized
    assert evidence["retrieval_policy"] == query_retrieval_policy(1, 1)
    assert evidence["term_normalization"] == query_term_normalization(config)
    assert evidence["term_normalization"]["stop_words"] is None
    assert query_term_normalization() == query_term_normalization(NeuralQueryAgenticForestConfig())


def test_capacity_limits_apply_to_selected_chunks_and_full_selected_text():
    config, kwargs = fixture()
    build_query_evidence(config=replace(config, evidence_chunks_per_patient_per_query=1), **kwargs)
    with pytest.raises(NeuralQueryEvidenceCapacityOverflowError, match="selected neural-query"):
        build_query_evidence(config=replace(config, evidence_retrieval_top_k=2,
                                           evidence_chunks_per_patient_per_query=1), **kwargs)
    with pytest.raises(ValueError, match="refusing silent text truncation"):
        build_query_evidence(config=replace(config, evidence_excerpt_chars=10), **kwargs)


def test_phrases_use_their_own_patient_count_and_leave_the_handed_off_chunks_unchanged():
    config = NeuralQueryAgenticForestConfig(
        evidence_top_patients=1, evidence_ngram_top_patients=2,
        evidence_background_patients=None, evidence_ngram_stop_words=None,
        evidence_ngram_range_min=1, evidence_ngram_range_max=1,
    )
    kwargs = dict(
        bank="treatment", queries=np.array([[1., 0.]], dtype=np.float32),
        query_records=[{"query_id": "treatment_query_001"}], row_ids=[0, 1, 2],
        chunk_matrices=[
            np.array([[1., 0.]], dtype=np.float32),
            np.array([[.9, .43589]], dtype=np.float32),
            np.array([[0., 1.]], dtype=np.float32),
        ],
        all_chunk_texts=[["first_witness_signal"], ["second_witness_signal"], ["background_signal"]],
        device="cpu", seed=7,
    )
    evidence = build_query_evidence(config=config, **kwargs)[0]
    assert [chunk["_oci_row_id"] for chunk in evidence["top_chunks"]] == [0]
    assert {term["term"] for term in evidence["top_contrastive_ngrams"]} == {
        "first_witness_signal", "second_witness_signal",
    }
    assert evidence["term_selection"]["foreground_patients"] == 2
    assert evidence["retrieval_policy"] == query_retrieval_policy(1, 2)

    narrower = build_query_evidence(config=replace(config, evidence_ngram_top_patients=1), **kwargs)[0]
    assert [chunk["_oci_row_id"] for chunk in narrower["top_chunks"]] == [0]
    assert {term["term"] for term in narrower["top_contrastive_ngrams"]} == {"first_witness_signal"}
    with pytest.raises(ValueError, match="evidence_ngram_top_patients"):
        replace(config, evidence_ngram_top_patients=0).validate()


def test_stage2_accepts_the_current_and_legacy_retrieval_policies_only():
    current = query_retrieval_policy(1, 30)
    # The exact policy of handoffs whose phrases came from the handed-off chunks.
    legacy = {
        "version": "ranked_query_chunks_v1",
        "patient_selection": "highest_maximum_chunk_cosine",
        "chunk_selection": "highest_cosine_per_patient",
        "chunks_per_patient": 1,
        "contrastive_terms_source": "selected_foreground_and_background_chunks",
    }
    validate_query_retrieval_policy(current)
    validate_query_retrieval_policy(legacy)
    for policy in (
        {**current, "contrastive_terms_patients": 0},
        {**current, "contrastive_terms_source": legacy["contrastive_terms_source"]},
        {**current, "version": "ranked_query_chunks_v9"},
        {**legacy, "contrastive_terms_patients": 30},
    ):
        with pytest.raises(ValueError):
            validate_query_retrieval_policy(policy)
