from __future__ import annotations


import pytest


from oci.inference.all_evidence_fusion import PRIMARY_SOURCE_FAMILIES
from oci.inference.plain_handoff_stage2_evidence import (
    SUPPORTED_STAGE2_ARCHITECTURES,
)
from oci.inference.stage1_architectures import (
    STAGE1_ARCHITECTURES,
    canonicalize_stage1_architectures,
    resolve_support_services,
    selected_components,
)

EXPECTED_ARCHITECTURES = (
    "bow_nuisance",
    "bow_r_loss",
    "matched_pair_uplift",
    "htr_neural",
    "embedding_whole_cohort",
    "embedding_clustered",
    "tfidf_semantic_retrieval_contrasts",
    "tfidf_topics",
    "tfidf_orphan_ngrams",
    "neural_query_moments",
)


def test_registry_is_the_single_canonical_ten_architecture_contract():
    assert STAGE1_ARCHITECTURES == EXPECTED_ARCHITECTURES
    assert PRIMARY_SOURCE_FAMILIES is STAGE1_ARCHITECTURES
    assert SUPPORTED_STAGE2_ARCHITECTURES is STAGE1_ARCHITECTURES
    assert len(resolve_support_services(STAGE1_ARCHITECTURES)) == len(
        set(resolve_support_services(STAGE1_ARCHITECTURES))
    )


def test_selection_is_validated_and_canonicalized():
    assert canonicalize_stage1_architectures("tfidf_topics,bow_nuisance") == (
        "bow_nuisance",
        "tfidf_topics",
    )
    assert canonicalize_stage1_architectures("all") == STAGE1_ARCHITECTURES
    assert selected_components(("neural_query_moments",)) == (
        "embedding_cache",
        "neural_queries",
        "handoff",
    )
    with pytest.raises(ValueError, match="duplicate"):
        canonicalize_stage1_architectures("bow_nuisance,bow_nuisance")
    with pytest.raises(ValueError, match="unknown"):
        canonicalize_stage1_architectures("retired_agent")
