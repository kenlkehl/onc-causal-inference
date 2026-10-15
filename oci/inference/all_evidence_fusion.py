"""Compaction helpers for Stage 1 discovery evidence.

The Stage 2 evidence compiler (``plain_handoff_stage2_evidence``) uses these
deterministic helpers to reduce raw Stage 1 discovery payloads (bag-of-words
rows, embedding contrasts, TF-IDF topic banks and orphan n-gram clusters) to
compact, JSON-safe evidence rows.  They do not fit models, call an agent, or
know any dataset-specific clinical variables.
"""

from __future__ import annotations

import math
import re
from typing import Any, Mapping

from .stage1_architectures import (
    BOW_NUISANCE,
    BOW_R_LOSS,
    EMBEDDING_CLUSTERED,
    EMBEDDING_WHOLE_COHORT,
    HTR_NEURAL,
    MATCHED_PAIR_UPLIFT,
    NEURAL_QUERY_MOMENTS,
    STAGE1_ARCHITECTURES,
    TFIDF_ORPHAN_NGRAMS,
    TFIDF_SEMANTIC_RETRIEVAL,
    TFIDF_TOPICS,
)

EXACT_INNER_RECURRENCE_VERSION = "exact_inner_normalized_term_recurrence_v2"

SPARSE_QUERY_MOMENTS = "sparse_query_moments"

PRIMARY_SOURCE_FAMILIES = STAGE1_ARCHITECTURES
# Sparse query moments are an accepted auxiliary evidence family, not one of
# the ten architectures promised by the researcher all-evidence path.
ALL_SOURCE_FAMILIES = (
    *PRIMARY_SOURCE_FAMILIES,
    SPARSE_QUERY_MOMENTS,
)
_SOURCE_FAMILY_SET = frozenset(ALL_SOURCE_FAMILIES)

_VALID_ROLES = frozenset({"confounder", "effect_modifier"})


def _compact_exact_inner_recurrence(
    value: Any,
) -> list[tuple[tuple[str, ...], str, dict[str, Any]]]:
    if value is None:
        return []
    if not isinstance(value, Mapping):
        raise ValueError("exact-inner recurrence evidence must be an object")
    expected_top_fields = {
        "schema_version",
        "normalization",
        "inner_fold_count",
        "latent_topic_ids_compared_across_folds",
        "minimum_inner_fold_support",
        "groups",
    }
    if set(value) != expected_top_fields:
        raise ValueError("exact-inner recurrence evidence must use the closed schema")
    if value.get("schema_version") != EXACT_INNER_RECURRENCE_VERSION:
        raise ValueError("exact-inner recurrence evidence has an unsupported schema")
    if value.get("normalization") != "unicode_nfkc_casefold_nonword_to_space_exact_match":
        raise ValueError("exact-inner recurrence evidence has an unsupported normalization")
    if value.get("latent_topic_ids_compared_across_folds") is not False:
        raise ValueError("exact-inner recurrence must not align latent topic IDs")
    if value.get("minimum_inner_fold_support") != 2:
        raise ValueError("exact-inner recurrence minimum support must equal two folds")
    try:
        inner_fold_count = int(value.get("inner_fold_count"))
    except (TypeError, ValueError) as exc:
        raise ValueError("exact-inner recurrence inner_fold_count must be an integer") from exc
    if inner_fold_count < 2:
        raise ValueError("exact-inner recurrence requires at least two inner folds")
    groups = value.get("groups")
    if not isinstance(groups, list):
        raise ValueError("exact-inner recurrence groups must be a list")
    output: list[tuple[tuple[str, ...], str, dict[str, Any]]] = []
    for group_index, group in enumerate(groups):
        if not isinstance(group, Mapping):
            raise ValueError(f"exact-inner recurrence groups[{group_index}] must be an object")
        if set(group) != {
            "source_family",
            "role",
            "discovered_recurrent_term_count",
            "retained_term_count",
            "terms",
        }:
            raise ValueError("exact-inner recurrence group must use the closed schema")
        family = str(group.get("source_family") or "")
        role = str(group.get("role") or "")
        if family not in _SOURCE_FAMILY_SET or role not in _VALID_ROLES:
            raise ValueError("exact-inner recurrence group has an invalid family or role")
        raw_terms = group.get("terms")
        if not isinstance(raw_terms, list):
            raise ValueError("exact-inner recurrence group terms must be a list")
        try:
            discovered_count = int(group.get("discovered_recurrent_term_count"))
            retained_count = int(group.get("retained_term_count"))
        except (TypeError, ValueError) as exc:
            raise ValueError("exact-inner recurrence group counts must be integers") from exc
        if (
            discovered_count < retained_count
            or retained_count != len(raw_terms)
        ):
            raise ValueError("exact-inner recurrence group accounting is inconsistent")
        compact_terms: list[dict[str, Any]] = []
        seen_terms: set[str] = set()
        for term_index, raw_term in enumerate(raw_terms):
            if not isinstance(raw_term, Mapping):
                raise ValueError(f"exact-inner recurrence terms[{term_index}] must be an object")
            term = _normalize_evidence_text(raw_term.get("term"))
            if not term or term in seen_terms:
                continue
            seen_terms.add(term)
            try:
                support_count = int(raw_term.get("inner_fold_support_count"))
                occurrence_count = int(raw_term.get("occurrence_count"))
            except (TypeError, ValueError) as exc:
                raise ValueError("exact-inner recurrence support counts must be integers") from exc
            if not 2 <= support_count <= inner_fold_count or occurrence_count < support_count:
                raise ValueError("exact-inner recurrence contains inconsistent support counts")
            compact_terms.append(
                {
                    "term": term,
                    "inner_fold_support_count": support_count,
                    "inner_fold_support_fraction": round(support_count / inner_fold_count, 8),
                    "occurrence_count": occurrence_count,
                }
            )
        if compact_terms:
            output.append(
                (
                    (family,),
                    role,
                    {
                        "kind": "exact_inner_normalized_term_recurrence",
                        "normalization_version": EXACT_INNER_RECURRENCE_VERSION,
                        "inner_fold_count": inner_fold_count,
                        "discovered_recurrent_term_count": discovered_count,
                        "retained_term_count": len(compact_terms),
                        "terms": compact_terms,
                    },
                )
            )
    return output


def _compact_bow_rows(value: Any) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    if not isinstance(value, (list, tuple)):
        return output
    for row in value:
        if not isinstance(row, Mapping):
            continue
        feature = _normalize_evidence_text(
            row.get("feature") or row.get("term") or row.get("phrase"),
        )
        if not feature:
            continue
        item: dict[str, Any] = {"term": feature}
        for key in (
            "coefficient",
            "importance",
            "score",
            "signed_score",
            "frequency",
            "source_count",
            "rank",
        ):
            scalar = _finite_scalar(row.get(key))
            if scalar is not None:
                item[key] = scalar
        output.append(item)
    return output


def _embedding_family(contrast: Mapping[str, Any]) -> str:
    descriptor = " ".join(
        str(contrast.get(key) or "").lower()
        for key in (
            "name",
            "contrast_family",
            "direction_source",
            "role_hint",
            "cluster_component_index",
        )
    )
    return EMBEDDING_CLUSTERED if "cluster" in descriptor else EMBEDDING_WHOLE_COHORT


def _compact_tfidf_evidence(
    payload: Mapping[str, Any],
) -> list[tuple[tuple[str, ...], str, dict[str, Any]]]:
    discovery = (
        payload.get("discovery") if isinstance(payload.get("discovery"), Mapping) else payload
    )
    banks = discovery.get("topic_banks") if isinstance(discovery, Mapping) else None
    if not isinstance(banks, Mapping):
        raise ValueError("tfidf_topics payload is missing discovery.topic_banks")
    output: list[tuple[tuple[str, ...], str, dict[str, Any]]] = []
    for bank in ("treatment", "outcome", "effect"):
        bank_payload = banks.get(bank)
        if not isinstance(bank_payload, Mapping):
            continue
        role = "effect_modifier" if bank == "effect" else "confounder"
        topics = bank_payload.get("topics")
        if not isinstance(topics, (list, tuple)):
            continue
        for topic in topics:
            if not isinstance(topic, Mapping):
                continue
            terms = _compact_topic_terms(topic.get("terms"))
            if not terms:
                continue
            output.append(
                (
                    (TFIDF_TOPICS,),
                    role,
                    {
                        "kind": "tfidf_topic",
                        "bank": bank,
                        "topic_id": _normalize_evidence_text(topic.get("topic_id")),
                        "terms": terms,
                    },
                )
            )

    orphan = _find_orphan_branch(discovery)
    if isinstance(orphan, Mapping):
        clusters = orphan.get("selected_clusters") or orphan.get("clusters") or []
        selected_ids = {str(value) for value in orphan.get("selected_cluster_ids") or []}
        if isinstance(clusters, (list, tuple)):
            for cluster in clusters:
                if not isinstance(cluster, Mapping):
                    continue
                cluster_id = str(cluster.get("cluster_id") or cluster.get("topic_id") or "")
                if selected_ids and cluster_id not in selected_ids:
                    continue
                terms = _compact_topic_terms(
                    cluster.get("terms")
                    or cluster.get("member_terms")
                    or cluster.get("supporting_terms")
                    or cluster.get("term_scores")
                )
                if not terms:
                    continue
                output.append(
                    (
                        (TFIDF_ORPHAN_NGRAMS,),
                        "effect_modifier",
                        {
                            "kind": "tfidf_orphan_ngram_cluster",
                            "cluster_id": _normalize_evidence_text(cluster_id),
                            "terms": terms,
                        },
                    )
                )
    output.extend(_compact_exact_inner_recurrence(discovery.get("exact_inner_recurrence")))
    return output


def _find_orphan_branch(discovery: Mapping[str, Any]) -> Any:
    branch = discovery.get("effect_orphan_ngram_branch")
    return branch if isinstance(branch, Mapping) else None


def _compact_topic_terms(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    output: list[dict[str, Any]] = []
    for raw_term in value:
        row = raw_term if isinstance(raw_term, Mapping) else {"term": raw_term}
        term = _normalize_evidence_text(
            row.get("term") or row.get("feature") or row.get("ngram"),
        )
        if not term:
            continue
        item: dict[str, Any] = {"term": term}
        for key in (
            "tfidf_contrast",
            "loading",
            "signed_score",
            "fit_signed_score",
            "standardized_score",
            "rank",
            "fit_rank",
        ):
            scalar = _finite_scalar(row.get(key))
            if scalar is not None:
                item[key] = scalar
        output.append(item)
    return output


def _normalize_evidence_text(value: Any) -> str:
    """Normalize evidence text without shortening or dropping its suffix."""

    return re.sub(r"\s+", " ", str(value or "")).strip()


def _finite_scalar(value: Any) -> int | float | bool | None:
    if value is None or isinstance(value, (Mapping, list, tuple, set)):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return round(value, 8) if math.isfinite(value) else None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return round(numeric, 8) if math.isfinite(numeric) else None


__all__ = [
    "ALL_SOURCE_FAMILIES",
    "PRIMARY_SOURCE_FAMILIES",
    "BOW_NUISANCE",
    "BOW_R_LOSS",
    "EMBEDDING_CLUSTERED",
    "EMBEDDING_WHOLE_COHORT",
    "EXACT_INNER_RECURRENCE_VERSION",
    "HTR_NEURAL",
    "MATCHED_PAIR_UPLIFT",
    "NEURAL_QUERY_MOMENTS",
    "SPARSE_QUERY_MOMENTS",
    "TFIDF_ORPHAN_NGRAMS",
    "TFIDF_SEMANTIC_RETRIEVAL",
    "TFIDF_TOPICS",
]
