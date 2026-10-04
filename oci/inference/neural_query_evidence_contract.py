"""Lightweight provenance contract for query-ranked evidence selection."""

from collections.abc import Mapping
from typing import Any


NEURAL_QUERY_RETRIEVAL_VERSION = "ranked_query_chunks_v1"


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
