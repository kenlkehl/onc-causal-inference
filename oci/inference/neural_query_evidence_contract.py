"""Lightweight provenance contract for query-ranked evidence selection."""

from collections.abc import Mapping
from typing import Any


NEURAL_QUERY_RETRIEVAL_VERSION = "ranked_query_chunks_v1"
# Consensus query groups are ordered by the median absolute inner-validation
# z-statistic of their members, then by the median absolute training score,
# then by the medoid candidate ID. Groups without validation statistics follow.
NEURAL_QUERY_RANKING_VERSION = "heldout_abs_z_median_v1"
NEURAL_QUERY_RANKING_RULE = (
    "median_abs_inner_validation_z_desc, then median_abs_train_score_desc, "
    "then medoid_candidate_id; groups without validation statistics last"
)
# Per-context artifacts (evidence.json, query_records.json, complete.json).
NEURAL_QUERY_CONTEXT_SCHEMA_VERSION = "neural_query_context_v3_informative_chunks"


def query_term_normalization(config: Any = None) -> dict[str, Any]:
    """Record the word normalization used before constructing contrast n-grams.

    Defaults describe historical handoffs that predate this metadata. New
    producers always pass their actual configuration, including custom settings.
    """
    defaults = {
        "analyzer": "word", "strip_accents": None, "lowercase": True,
        "stop_words": "english", "token_pattern": r"(?u)\b\w\w+\b",
    }
    return {
        key: getattr(config, f"evidence_ngram_{key}", value)
        for key, value in defaults.items()
    }


def query_retrieval_policy(chunks_per_patient: int) -> dict[str, Any]:
    return {
        "version": NEURAL_QUERY_RETRIEVAL_VERSION,
        "patient_selection": "highest_maximum_chunk_cosine",
        "chunk_selection": "highest_cosine_per_patient",
        "chunks_per_patient": chunks_per_patient,
        "contrastive_terms_source": "selected_foreground_and_background_chunks",
    }


def query_chunk_eligibility(min_alphanumeric_characters: int) -> dict[str, Any]:
    """Record which chunks the queries were fitted, scored, and retrieved on."""
    return {
        "rule": "min_alphanumeric_characters",
        "min_alphanumeric_characters": int(min_alphanumeric_characters),
        "applies_to": ["query_fitting", "heldout_scoring", "witness_selection"],
        "patients_without_eligible_chunks": "mean_activation_never_witness",
    }


def query_term_selection(top_k: int | None, positive_terms: int) -> dict[str, Any]:
    """Record how the reported terms were chosen from every positive-contrast term."""
    return {
        "rule": "highest_positive_tfidf_contrast_then_term",
        "top_k": None if top_k is None else int(top_k),
        "positive_terms": int(positive_terms),
    }


def validate_query_retrieval_policy(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(
            "Neural-query evidence has no ranked-chunk retrieval provenance. "
            "Regenerate its evidence from the saved query vectors and chunk cache; "
            "legacy all-history evidence and its contrastive terms cannot be reused."
        )
    count = value.get("chunks_per_patient")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("neural-query retrieval chunks_per_patient must be a positive integer")
    if dict(value) != query_retrieval_policy(count):
        raise ValueError("Unsupported neural-query retrieval policy; rebuild query evidence")
