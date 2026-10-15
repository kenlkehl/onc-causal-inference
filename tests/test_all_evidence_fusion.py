from __future__ import annotations

import pytest

from oci.inference.all_evidence_fusion import (
    ALL_SOURCE_FAMILIES,
    BOW_NUISANCE,
    EMBEDDING_CLUSTERED,
    EMBEDDING_WHOLE_COHORT,
    EXACT_INNER_RECURRENCE_VERSION,
    PRIMARY_SOURCE_FAMILIES,
    SPARSE_QUERY_MOMENTS,
    TFIDF_ORPHAN_NGRAMS,
    TFIDF_TOPICS,
    _compact_bow_rows,
    _compact_tfidf_evidence,
    _compact_topic_terms,
    _embedding_family,
    _normalize_evidence_text,
)


def _tfidf_payload() -> dict:
    topic = lambda topic_id, phrase: {  # noqa: E731
        "topic_id": topic_id,
        "terms": [{"term": phrase, "loading": 0.8, "signed_score": 2.0}],
    }
    return {
        "outer_fold": 1,
        "discovery": {
            "topic_banks": {
                "treatment": {"topics": [topic("treatment_1", "prerun calibration pattern")]},
                "outcome": {"topics": [topic("outcome_1", "prerun load pattern")]},
                "effect": {"topics": [topic("effect_1", "prerun alloy phase phrase")]},
            },
            "effect_orphan_ngram_branch": {
                "selected_cluster_ids": ["orphan_1"],
                "selected_clusters": [
                    {
                        "cluster_id": "orphan_1",
                        "terms": [{"term": "unmodeled baseline phrase", "fit_rank": 23}],
                    },
                    {
                        "cluster_id": "orphan_2",
                        "terms": [{"term": "unselected phrase", "fit_rank": 1}],
                    },
                ],
            },
        },
    }


def _recurrence(**overrides) -> dict:
    value = {
        "schema_version": EXACT_INNER_RECURRENCE_VERSION,
        "normalization": "unicode_nfkc_casefold_nonword_to_space_exact_match",
        "inner_fold_count": 4,
        "latent_topic_ids_compared_across_folds": False,
        "minimum_inner_fold_support": 2,
        "groups": [
            {
                "source_family": TFIDF_TOPICS,
                "role": "effect_modifier",
                "discovered_recurrent_term_count": 3,
                "retained_term_count": 2,
                "terms": [
                    {"term": "recurrent  phrase", "inner_fold_support_count": 3, "occurrence_count": 5},
                    {"term": "recurrent phrase", "inner_fold_support_count": 2, "occurrence_count": 2},
                ],
            }
        ],
    }
    value.update(overrides)
    return value


def test_source_families_extend_primary_architectures_with_sparse_query_moments():
    assert set(PRIMARY_SOURCE_FAMILIES) == set(ALL_SOURCE_FAMILIES) - {SPARSE_QUERY_MOMENTS}


def test_tfidf_compaction_keeps_topic_banks_and_selected_orphan_clusters():
    blocks = _compact_tfidf_evidence(_tfidf_payload())

    assert [(families, role, block["kind"]) for families, role, block in blocks] == [
        ((TFIDF_TOPICS,), "confounder", "tfidf_topic"),
        ((TFIDF_TOPICS,), "confounder", "tfidf_topic"),
        ((TFIDF_TOPICS,), "effect_modifier", "tfidf_topic"),
        ((TFIDF_ORPHAN_NGRAMS,), "effect_modifier", "tfidf_orphan_ngram_cluster"),
    ]
    assert blocks[0][2]["terms"] == [
        {"term": "prerun calibration pattern", "loading": 0.8, "signed_score": 2.0}
    ]
    assert blocks[-1][2] == {
        "kind": "tfidf_orphan_ngram_cluster",
        "cluster_id": "orphan_1",
        "terms": [{"term": "unmodeled baseline phrase", "fit_rank": 23}],
    }
    with pytest.raises(ValueError, match="topic_banks"):
        _compact_tfidf_evidence({"discovery": {}})


def test_exact_inner_recurrence_is_deduplicated_and_closed_schema():
    payload = _tfidf_payload()
    payload["discovery"]["exact_inner_recurrence"] = _recurrence()
    families, role, block = _compact_tfidf_evidence(payload)[-1]

    assert families == (TFIDF_TOPICS,)
    assert role == "effect_modifier"
    assert block["kind"] == "exact_inner_normalized_term_recurrence"
    assert block["terms"] == [
        {
            "term": "recurrent phrase",
            "inner_fold_support_count": 3,
            "inner_fold_support_fraction": 0.75,
            "occurrence_count": 5,
        }
    ]

    payload["discovery"]["exact_inner_recurrence"] = _recurrence(
        latent_topic_ids_compared_across_folds=True
    )
    with pytest.raises(ValueError, match="latent topic IDs"):
        _compact_tfidf_evidence(payload)


def test_bow_topic_and_text_helpers_drop_empty_and_non_finite_values():
    assert _compact_bow_rows(
        [
            {"feature": "  baseline   marker ", "coefficient": 0.123456789, "rank": 2},
            {"term": "", "coefficient": 1.0},
            {"phrase": "second marker", "importance": float("nan")},
            "not a row",
        ]
    ) == [
        {"term": "baseline marker", "coefficient": 0.12345679, "rank": 2},
        {"term": "second marker"},
    ]
    assert _compact_bow_rows(None) == []
    assert _compact_topic_terms(["plain term", {"ngram": "scored term", "tfidf_contrast": 0.4}]) == [
        {"term": "plain term"},
        {"term": "scored term", "tfidf_contrast": 0.4},
    ]
    assert _normalize_evidence_text(" a\n\tb ") == "a b"
    assert _normalize_evidence_text(None) == ""


def test_embedding_family_routes_cluster_contrasts():
    assert _embedding_family({"contrast_family": "cluster_local_residualized"}) == EMBEDDING_CLUSTERED
    assert _embedding_family({"name": "whole cohort direction"}) == EMBEDDING_WHOLE_COHORT
    assert BOW_NUISANCE in PRIMARY_SOURCE_FAMILIES
