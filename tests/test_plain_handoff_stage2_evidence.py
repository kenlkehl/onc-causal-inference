from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from oci.inference.neural_query_evidence_contract import query_retrieval_policy

from oci.inference.all_evidence_fusion import (
    BOW_NUISANCE,
    BOW_R_LOSS,
    EMBEDDING_CLUSTERED,
    EMBEDDING_WHOLE_COHORT,
    HTR_NEURAL,
    MATCHED_PAIR_UPLIFT,
    NEURAL_QUERY_MOMENTS,
    TFIDF_ORPHAN_NGRAMS,
    TFIDF_SEMANTIC_RETRIEVAL,
    TFIDF_TOPICS,
)
from oci.inference.plain_handoff_stage2_evidence import (
    EVIDENCE_COMPILER_VERSION,
    SUPPORTED_STAGE2_ARCHITECTURES,
    compile_stage2_handoff_evidence,
)


def _htr_row(*, outer_fold: int, inner_fold: int, row_id: int) -> dict:
    return {
        "source": "text_models",
        "outer_fold": outer_fold,
        "inner_fold": inner_fold,
        "scope": "candidate_consistency_inner_train",
        "evidence": {
            "htr_evidence": {
                "effect": {
                    "attention": [
                        {
                            "row_id": row_id,
                            "chunk_index": 91,
                            "chunk_text": "Pretreatment ECOG performance status was 2.",
                            "highlighted_chunk_text": (
                                "Pretreatment **ECOG performance status** was 2."
                            ),
                            "attended_token_summary": "ECOG performance status; ECOG 2",
                            "attention": 0.8,
                            "stage": "effect_modifier",
                        }
                    ]
                }
            }
        },
    }


def test_compiler_is_fold_local_and_preserves_exact_deduplication_lineage(
    tmp_path: Path,
):
    compiled = compile_stage2_handoff_evidence(
        [
            _htr_row(outer_fold=1, inner_fold=1, row_id=3),
            _htr_row(outer_fold=1, inner_fold=2, row_id=4),
            _htr_row(outer_fold=2, inner_fold=1, row_id=3),
        ],
        handoff_path=tmp_path / "handoff" / "evidence.jsonl",
        max_cards_per_outer_fold=16,
        max_packet_chars=2_000,
    )

    fold_one = compiled.summary["outer_folds"]["1"]
    fold_two = compiled.summary["outer_folds"]["2"]
    assert fold_one["raw_occurrences"] == 2
    assert fold_one["exact_members"] == 1
    assert fold_one["exact_duplicate_occurrences_removed"] == 1
    assert fold_two["raw_occurrences"] == 1
    assert fold_two["exact_members"] == 1
    assert {packet["outer_fold"] for packet in compiled.packets} == {1, 2}

    member = compiled.members_by_outer_fold[1][0]
    assert member["raw_occurrence_count"] == 2
    assert {row["inner_fold"] for row in member["raw_references"]} == {1, 2}
    assert {row["row_id"] for row in member["raw_references"]} == {3, 4}
    card = compiled.cards_by_outer_fold[1][0]
    assert card["support"]["inner_folds"] == [1, 2]
    assert card["support"]["raw_occurrence_count"] == 2
    assert compiled.lineage_by_outer_fold[1][0]["member_ids"] == [member["member_id"]]

    rendered = json.dumps(compiled.packets, sort_keys=True)
    assert '"row_id"' not in rendered
    assert '"handoff_row"' not in rendered
    assert (
        max(
            len(json.dumps(packet, separators=(",", ":"), sort_keys=True))
            for packet in compiled.packets
        )
        <= 2_000
    )


def test_neural_query_ngrams_are_compacted_before_exact_member_aggregation(
    tmp_path: Path,
):
    rows = [
        {
            "source": "neural_queries",
            "outer_fold": 1,
            "inner_fold": None,
            "scope": "full_outer_train",
            "evidence": {
                "evidence": [
                    {
                        "retrieval_policy": query_retrieval_policy(1),
                        "query_id": "treatment_query",
                        "bank": "treatment",
                        "fit_standardized_score": 2.0,
                        "top_contrastive_ngrams": [
                            {"term": "performance status", "loading": 0.6},
                            {"term": "treatment wording", "loading": 0.4},
                        ],
                    },
                    {
                        "retrieval_policy": query_retrieval_policy(1),
                        "query_id": "outcome_query",
                        "bank": "outcome",
                        "fit_standardized_score": 3.0,
                        "top_contrastive_ngrams": [
                            {"term": "performance status", "loading": 0.8},
                            {"term": "outcome wording", "loading": 0.5},
                        ],
                    },
                ]
            },
        }
    ]

    compiled = compile_stage2_handoff_evidence(
        rows,
        handoff_path=tmp_path / "handoff" / "evidence.jsonl",
        max_cards_per_outer_fold=16,
        max_packet_chars=10_000,
        required_architectures=(NEURAL_QUERY_MOMENTS,),
    )

    fold = compiled.summary["outer_folds"]["1"]
    assert fold["raw_occurrences"] == 4
    assert fold["compact_occurrence_records"] == 3
    assert fold["exact_members"] == 3
    assert fold["exact_duplicate_occurrences_removed"] == 1
    assert fold["source_family_occurrences"][NEURAL_QUERY_MOMENTS] == 4
    assert fold["architecture_occurrences"][NEURAL_QUERY_MOMENTS] == 4
    repeated = [
        member
        for member in compiled.members_by_outer_fold[1]
        if member["raw_occurrence_count"] == 2
    ]
    assert len(repeated) == 1
    assert {row["query_id"] for row in repeated[0]["raw_references"]} == {
        "treatment_query",
        "outcome_query",
    }
    assert {
        row["query_id"]: row["scores"]["fit_standardized_score"]
        for row in repeated[0]["raw_references"]
    } == {"treatment_query": 2.0, "outcome_query": 3.0}
    assert sum(
        row["occurrence_count"] for row in repeated[0]["raw_references"]
    ) == 2
    repeated_card_id = next(
        row["card_id"]
        for row in compiled.lineage_by_outer_fold[1]
        if repeated[0]["member_id"] in row["member_ids"]
    )
    repeated_card = next(
        card for card in compiled.cards_by_outer_fold[1]
        if card["card_id"] == repeated_card_id
    )
    assert repeated_card["score_summary"]["loading"]["median"] == 0.7
    assert repeated_card["score_summary"]["fit_standardized_score"]["median"] == 2.5


def test_compiler_reuses_fusion_tfidf_allowlist_and_drops_large_score_arrays(
    tmp_path: Path,
):
    sentinel = "SHOULD_NOT_CROSS_THE_FUSION_BOUNDARY"
    rows = [
        {
            "source": "tfidf",
            "outer_fold": 1,
            "inner_fold": None,
            "scope": "full_outer_train",
            "evidence": {
                "discovery": {
                    "topic_banks": {
                        "treatment": {
                            "topics": [
                                {
                                    "topic_id": "topic-1",
                                    "terms": [
                                        {"term": "ECOG performance status", "loading": 0.9},
                                        {"term": "functional limitation", "loading": 0.7},
                                    ],
                                }
                            ],
                            "selected_terms": [sentinel] * 2_000,
                        },
                        "outcome": {"topics": []},
                        "effect": {"topics": []},
                    }
                }
            },
        }
    ]

    compiled = compile_stage2_handoff_evidence(
        rows,
        handoff_path=tmp_path / "handoff" / "evidence.jsonl",
        max_cards_per_outer_fold=16,
        max_packet_chars=2_000,
    )

    rendered = json.dumps(compiled.packets, sort_keys=True)
    assert "ECOG performance status" in rendered
    assert sentinel not in rendered
    assert compiled.packets[0]["content"]["source_families"] == ["tfidf_topics"]


def test_compiler_preserves_all_ten_architectures_as_independent_interpretation_lanes(
    tmp_path: Path,
):
    rows = [
        {
            "source": "text_models",
            "outer_fold": 1,
            "scope": "full_outer_train",
            "evidence": {
                "importance": {
                    "views": [
                        {
                            "view_name": "word_linear",
                            "treatment_positive": [{"feature": "treatment wording", "score": 2.0}],
                            "outcome_negative": [{"feature": "outcome wording", "score": -1.5}],
                            "pseudo_target_positive": [
                                {"feature": "residual effect wording", "score": 1.2}
                            ],
                        }
                    ],
                    "matched_pair_uplift": {
                        "views": [
                            {
                                "view_name": "pair_word_linear",
                                "uplift_delta_logit_positive": [
                                    {"feature": "treated::pair response wording", "score": 1.7}
                                ],
                                "ridge_delta_probability_negative": [
                                    {"feature": "control::pair control wording", "score": -0.8}
                                ],
                            }
                        ]
                    },
                },
                "embedding_contrast_evidence": {
                    "contrasts": [
                        {
                            "name": "global treatment contrast",
                            "contrast_family": "marginal",
                            "positive_aligned_chunks": [
                                {
                                    "row_id": 1,
                                    "chunk_index": 0,
                                    "text": "whole cohort semantic witness",
                                }
                            ],
                            "tfidf_retrieval_terms": [
                                {
                                    "term": "whole lexical projection",
                                    "polarity": "positive",
                                    "tfidf_contrast": 0.4,
                                }
                            ],
                        },
                        {
                            "name": "cluster local residual contrast",
                            "contrast_family": "cluster_local_residualized",
                            "negative_aligned_chunks": [
                                {
                                    "row_id": 2,
                                    "chunk_index": 0,
                                    "text": "cluster local semantic witness",
                                }
                            ],
                            "tfidf_retrieval_terms": [
                                {
                                    "term": "cluster lexical projection",
                                    "polarity": "negative",
                                    "tfidf_contrast": -0.5,
                                }
                            ],
                        },
                    ]
                },
                "htr_evidence": {
                    "nuisance": {
                        "attention": [
                            {"row_id": 3, "chunk_text": "HTR nuisance witness"}
                        ]
                    },
                    "effect": {
                        "attention": [
                            {"row_id": 4, "chunk_text": "duplicated canonical HTR effect"}
                        ]
                    },
                    "effect_variants": {
                        "pseudo_outcome_mse": {
                            "attention": [
                                {"row_id": 4, "chunk_text": "HTR pseudo outcome witness"}
                            ]
                        },
                        "squared_r_loss": {
                            "attention": [
                                {"row_id": 5, "chunk_text": "HTR squared R loss witness"}
                            ]
                        },
                    },
                    "pair_uplift": {
                        "attention": [
                            {"row_id": 6, "chunk_text": "HTR matched pair witness"}
                        ]
                    },
                },
            },
        },
        {
            "source": "tfidf",
            "outer_fold": 1,
            "scope": "full_outer_train",
            "evidence": {
                "discovery": {
                    "topic_banks": {
                        "treatment": {
                            "topics": [
                                {
                                    "topic_id": "treatment_topic_1",
                                    "terms": [{"term": "topic treatment wording", "loading": 0.9}],
                                }
                            ]
                        },
                        "outcome": {"topics": []},
                        "effect": {"topics": []},
                    },
                    "topic_score_tests": {
                        "effect_orphan_ngram_branch": {
                            "selected_cluster_ids": ["orphan_1"],
                            "selected_clusters": [
                                {
                                    "cluster_id": "orphan_1",
                                    "term_scores": [
                                        {"term": "orphan residual wording", "fit_rank": 7}
                                    ],
                                }
                            ],
                        }
                    },
                }
            },
        },
        {
            "source": "neural_queries",
            "outer_fold": 1,
            "scope": "full_outer_train",
            "evidence": {
                "evidence": [
                    {
                        "retrieval_policy": query_retrieval_policy(1),
                        "query_id": "effect_query_1",
                        "bank": "effect",
                        "top_chunks": [
                            {
                                "_oci_row_id": 7,
                                "chunk_index": 0,
                                "text": "learned neural query witness",
                            }
                        ],
                    }
                ]
            },
        },
    ]

    compiled = compile_stage2_handoff_evidence(
        rows,
        handoff_path=tmp_path / "handoff" / "evidence.jsonl",
        max_cards_per_outer_fold=64,
        max_packet_chars=4_000,
        required_architectures=SUPPORTED_STAGE2_ARCHITECTURES,
    )

    expected = {
        BOW_NUISANCE,
        BOW_R_LOSS,
        HTR_NEURAL,
        MATCHED_PAIR_UPLIFT,
        EMBEDDING_WHOLE_COHORT,
        EMBEDDING_CLUSTERED,
        TFIDF_SEMANTIC_RETRIEVAL,
        TFIDF_TOPICS,
        TFIDF_ORPHAN_NGRAMS,
        NEURAL_QUERY_MOMENTS,
    }
    assert {packet["architecture"] for packet in compiled.packets} == expected
    assert set(compiled.summary["outer_folds"]["1"]["architecture_packets"]) == expected
    assert set(compiled.summary["required_architectures"]) == expected

    pair_cards = [
        packet["content"]
        for packet in compiled.packets
        if packet["architecture"] == MATCHED_PAIR_UPLIFT
    ]
    assert pair_cards
    assert all("matched_pair" in card["evidence_axes"] for card in pair_cards)
    rendered_pairs = json.dumps(pair_cards)
    assert "treated::" not in rendered_pairs
    assert "control::" not in rendered_pairs
    assert "HTR matched pair witness" in rendered_pairs

    htr_rendered = json.dumps(
        [
            packet["content"]
            for packet in compiled.packets
            if packet["architecture"] == HTR_NEURAL
        ]
    )
    assert "HTR pseudo outcome witness" in htr_rendered
    assert "HTR squared R loss witness" in htr_rendered
    assert "duplicated canonical HTR effect" not in htr_rendered

    orphan_rendered = json.dumps(
        [
            packet["content"]
            for packet in compiled.packets
            if packet["architecture"] == TFIDF_ORPHAN_NGRAMS
        ]
    )
    assert "orphan residual wording" in orphan_rendered


def test_compiler_rejects_a_missing_enabled_architecture_before_interpretation(
    tmp_path: Path,
):
    rows = [
        {
            "source": "tfidf",
            "outer_fold": 1,
            "scope": "full_outer_train",
            "evidence": {
                "discovery": {
                    "topic_banks": {
                        "treatment": {
                            "topics": [
                                {
                                    "topic_id": "topic_1",
                                    "terms": [{"term": "treatment wording"}],
                                }
                            ]
                        }
                    }
                }
            },
        }
    ]

    with pytest.raises(ValueError, match="missing enabled Stage 1 architectures"):
        compile_stage2_handoff_evidence(
            rows,
            handoff_path=tmp_path / "handoff" / "evidence.jsonl",
            max_cards_per_outer_fold=16,
            max_packet_chars=4_000,
            required_architectures=(TFIDF_TOPICS, MATCHED_PAIR_UPLIFT),
        )


def test_compiler_memory_maps_existing_stage1_embeddings_for_semantic_cards(
    tmp_path: Path,
):
    cache = tmp_path / "components" / "embedding_cache" / "cache" / "test-cache"
    cache.mkdir(parents=True)
    vectors = np.zeros((20, 4), dtype=np.float16)
    vectors[:10, 0] = 1.0
    vectors[10:, 1] = 1.0
    np.save(cache / "chunk_embeddings.npy", vectors)
    np.save(cache / "offsets.npy", np.asarray([0, 20], dtype=np.int64))
    (cache / "metadata.json").write_text(
        json.dumps(
            {
                "sentence_model_name": "cached-test-embedder",
                "normalize_embeddings": True,
            }
        ),
        encoding="utf-8",
    )
    chunks = [
        {
            "row_id": 0,
            "chunk_index": index,
            "text": f"Pretreatment clinical evidence chunk {index}",
            "similarity": 0.9 - 0.01 * index,
        }
        for index in range(20)
    ]
    rows = [
        {
            "source": "text_models",
            "outer_fold": 1,
            "inner_fold": None,
            "scope": "full_outer_train",
            "evidence": {
                "embedding_contrast_evidence": {
                    "contrasts": [
                        {
                            "name": "treatment assignment contrast",
                            "positive_aligned_chunks": chunks,
                        }
                    ]
                }
            },
        }
    ]

    compiled = compile_stage2_handoff_evidence(
        rows,
        handoff_path=tmp_path / "handoff" / "evidence.jsonl",
        max_cards_per_outer_fold=16,
        max_packet_chars=2_000,
        seed=7,
    )

    assert compiled.summary["embedding_cache_model"] == "cached-test-embedder"
    assert compiled.summary["outer_folds"]["1"]["exact_members"] == 20
    # The cap is 16, but the cache contains only two distinct semantic vectors.
    assert compiled.summary["outer_folds"]["1"]["cards"] == 2
    assert all(
        card["semantic_grouping"] == "cached_embedding" for card in compiled.cards_by_outer_fold[1]
    )
    assert sum(len(lineage["member_ids"]) for lineage in compiled.lineage_by_outer_fold[1]) == 20
    assert compiled.summary["schema_version"] == EVIDENCE_COMPILER_VERSION
