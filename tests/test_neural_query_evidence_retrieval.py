from dataclasses import replace
import copy
import json

import numpy as np
import pytest

from oci.inference.neural_query_agentic_forest import (
    NeuralQueryAgenticForestConfig,
    NeuralQueryEvidenceCapacityOverflowError,
    build_query_evidence,
)
from oci.inference.neural_query_evidence_contract import query_retrieval_policy, query_term_normalization
from oci.inference.plain_handoff_stage2_evidence import (
    compile_stage2_handoff_evidence,
    extract_stage1_architecture_occurrences,
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


def raw_row(evidence):
    return {"source": "neural_queries", "outer_fold": 1,
            "scope": "full_outer_train", "evidence": {"evidence": [evidence]}}


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


def test_explicit_top_k_retains_second_match_without_importing_entire_history():
    config, kwargs = fixture()
    evidence = build_query_evidence(config=replace(config, evidence_retrieval_top_k=2), **kwargs)[0]
    assert [r["chunk_index"] for r in evidence["top_chunks"]] == [0, 1]
    terms = {r["term"] for r in evidence["top_contrastive_ngrams"]}
    assert "secondrank_signal" in terms
    assert "unrelated_history" not in terms
    assert evidence["retrieval_policy"] == query_retrieval_policy(2)


def test_background_contrast_does_not_use_unselected_patient_history():
    config, kwargs = fixture()
    baseline = build_query_evidence(config=config, **kwargs)[0]
    kwargs["all_chunk_texts"][2][1] = "matching_signal " * 1000
    changed = build_query_evidence(config=config, **kwargs)[0]
    assert changed == baseline


def test_capacity_limits_apply_to_selected_chunks_and_full_selected_text():
    config, kwargs = fixture()
    build_query_evidence(config=replace(config, evidence_chunks_per_patient_per_query=1), **kwargs)
    with pytest.raises(NeuralQueryEvidenceCapacityOverflowError, match="selected neural-query"):
        build_query_evidence(config=replace(config, evidence_retrieval_top_k=2,
                                           evidence_chunks_per_patient_per_query=1), **kwargs)
    with pytest.raises(ValueError, match="refusing silent text truncation"):
        build_query_evidence(config=replace(config, evidence_excerpt_chars=10), **kwargs)


@pytest.mark.parametrize("value", [None, 0, -1, 1.5, True])
def test_retrieval_top_k_is_a_positive_integer(value):
    with pytest.raises(ValueError, match="evidence_retrieval_top_k"):
        replace(NeuralQueryAgenticForestConfig(), evidence_retrieval_top_k=value).validate()


def test_compiler_rejects_legacy_evidence_and_false_top_k_provenance(tmp_path):
    config, kwargs = fixture()
    evidence = build_query_evidence(config=config, **kwargs)[0]
    options = dict(handoff_path=tmp_path / "handoff.jsonl", max_cards_per_outer_fold=16)
    compiled = compile_stage2_handoff_evidence([raw_row(evidence)], **options)
    assert compiled.cards_by_outer_fold[1]
    legacy = copy.deepcopy(evidence)
    legacy.pop("retrieval_policy")
    with pytest.raises(ValueError, match="legacy all-history"):
        compile_stage2_handoff_evidence([raw_row(legacy)], **options)
    excess = copy.deepcopy(evidence)
    excess["top_chunks"].append({**excess["top_chunks"][0], "chunk_index": 1})
    with pytest.raises(ValueError, match="exceeds its declared retrieval top-k"):
        compile_stage2_handoff_evidence([raw_row(excess)], **options)


def test_canonical_handoff_preserves_policy_and_cannot_bypass_legacy_rejection(tmp_path):
    config, kwargs = fixture()
    evidence = build_query_evidence(config=config, **kwargs)[0]
    occurrences = extract_stage1_architecture_occurrences([raw_row(evidence)])[1]
    for occurrence in occurrences:
        if occurrence["evidence_kind"] == "lexical_term":
            assert occurrence["details"]["term_normalization"] == evidence["term_normalization"]
    canonical = [
        {"source": "stage1_architecture", "outer_fold": 1,
         "evidence": {"architecture": "neural_query_moments", "occurrence": occurrence}}
        for occurrence in occurrences
    ]
    options = dict(handoff_path=tmp_path / "handoff.jsonl", max_cards_per_outer_fold=16)
    compile_stage2_handoff_evidence(canonical, **options)
    canonical[0]["evidence"]["occurrence"]["details"].pop("retrieval_policy")
    with pytest.raises(ValueError, match="legacy all-history"):
        compile_stage2_handoff_evidence(canonical, **options)
