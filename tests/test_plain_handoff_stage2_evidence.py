from __future__ import annotations

import json
from pathlib import Path

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
    SUPPORTED_STAGE2_ARCHITECTURES,
    compile_stage2_handoff_evidence,
)
from oci.inference.stage1_architectures import DEFAULT_STAGE1_ARCHITECTURES


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


def _all_architecture_raw_rows() -> list[dict]:
    """Raw omitted-selector handoff rows covering all ten Stage 1 architectures."""

    return [
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
                        "schema_version": "matched_pair_conditional_logit_v1",
                        "views": [
                            {
                                "view_name": "matched_pair__word_linear",
                                "score_polarity": {"effect_modification": "signed"},
                                "effect_modification_positive": [
                                    {"feature": "pair response wording", "score": 1.7, "rank": 1}
                                ],
                                "effect_modification_negative": [
                                    {"feature": "pair attenuation wording", "score": -0.8, "rank": 2}
                                ],
                            }
                        ],
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


@pytest.mark.parametrize("include_optional", [False], ids=["default_eight"])
def test_compiler_preserves_enabled_architectures_as_independent_interpretation_lanes(
    tmp_path: Path, include_optional: bool,
):
    rows = _all_architecture_raw_rows()

    required = SUPPORTED_STAGE2_ARCHITECTURES if include_optional else DEFAULT_STAGE1_ARCHITECTURES
    if not include_optional:
        # Mirror disabled producers: no retrieval terms, and only a disabled
        # orphan diagnostic. The compiler sees the actual raw handoff, without
        # an included-architecture filter hiding missing lanes.
        for contrast in rows[0]["evidence"]["embedding_contrast_evidence"]["contrasts"]:
            contrast.pop("tfidf_retrieval_terms")
        rows[1]["evidence"]["discovery"]["effect_orphan_ngram_branch"] = {
            "status": "disabled", "clusters": [], "selected_clusters": [],
            "selected_cluster_ids": [], "selection_count": 0,
        }

    compiled = compile_stage2_handoff_evidence(
        rows,
        handoff_path=tmp_path / "handoff" / "evidence.jsonl",
        max_cards_per_outer_fold=64,
        max_packet_chars=4_000,
        required_architectures=required,
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
    if not include_optional:
        expected -= {TFIDF_SEMANTIC_RETRIEVAL, TFIDF_ORPHAN_NGRAMS}
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
    assert "pair response wording" in rendered_pairs
    assert "pair attenuation wording" in rendered_pairs
    assert "pair_side" not in rendered_pairs
    assert {"positive", "negative"} <= {
        polarity for card in pair_cards for polarity in card["polarities"]
    }

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
    assert ("orphan residual wording" in orphan_rendered) is include_optional


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
