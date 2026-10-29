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
)


def fixture():
    config = NeuralQueryAgenticForestConfig(
        evidence_top_patients=1, evidence_background_patients=None,
        evidence_top_ngrams=None, evidence_ngram_stop_words=None,
        evidence_ngram_range_min=1, evidence_ngram_range_max=1,
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
    assert evidence["retrieval_policy"] == query_retrieval_policy(1)
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
