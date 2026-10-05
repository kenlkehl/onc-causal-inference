"""Honest held-out topic and n-gram score tests for TF-IDF/NMF banks.

The score layer is deliberately upstream of agents and CATE models.  Its
primary topic hypothesis asks whether each fitted one-dimensional NMF
patient-topic score has held-out association with the bank-specific target.
It also tests the topic's configured complete set of supplied TF-IDF terms
jointly and tests every supplied n-gram individually:

* treatment: treatment minus the fit prevalence;
* outcome: outcome minus the fit mean;
* effect: the orthogonal constant-effect cohort contribution.

Every fitted topic and every one of its supplied n-grams is tested.  Topic and
n-gram hypotheses are calibrated as two explicit, complete families.  All
topic definitions and scaling are fit-side.  Inner-held-out labels enter only
the score statistic.  Full outer contexts must not call this module with their
held-out labels.
"""

from __future__ import annotations

import copy
import re
from dataclasses import asdict
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from scipy import sparse
from scipy.stats import chi2, norm
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from ..config import TfidfTopicDiscoveryConfig
from .sparse_text_modeling import _make_bow_vectorizer


TOPIC_SCORE_TEST_SCHEMA_VERSION = "tfidf_topic_and_ngram_score_test_v6"
_BANKS = ("treatment", "outcome", "effect")

_ORPHAN_STOPWORDS = {
    "a",
    "an",
    "and",
    "for",
    "from",
    "in",
    "of",
    "on",
    "the",
    "to",
    "with",
}


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    """Return monotone Benjamini-Hochberg adjusted p-values."""
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1:
        raise ValueError("p_values must be one-dimensional")
    if not len(values):
        return np.zeros(0, dtype=float)
    values = np.clip(np.nan_to_num(values, nan=1.0, posinf=1.0, neginf=1.0), 0.0, 1.0)
    order = np.argsort(values, kind="stable")
    ranked = values[order]
    adjusted_ranked = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    return adjusted


def _constant_effect(
    treatment: np.ndarray,
    outcome: np.ndarray,
    propensity: np.ndarray,
    outcome_prediction: np.ndarray,
) -> float:
    u = np.asarray(treatment, dtype=float) - np.asarray(propensity, dtype=float)
    v = np.asarray(outcome, dtype=float) - np.asarray(outcome_prediction, dtype=float)
    denominator = float(np.dot(u, u))
    return 0.0 if denominator <= 1e-12 else float(np.dot(u, v) / denominator)


def _bank_contribution(
    *,
    bank: str,
    fit_treatment: np.ndarray,
    fit_outcome: np.ndarray,
    heldout_treatment: np.ndarray,
    heldout_outcome: np.ndarray,
    fit_propensity: np.ndarray,
    fit_outcome_prediction: np.ndarray,
    heldout_propensity: np.ndarray,
    heldout_outcome_prediction: np.ndarray,
    dina_nuisance=None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Return held-out score contribution and feature-centering weights."""
    if bank == "treatment":
        baseline = float(np.mean(fit_treatment))
        contribution = np.asarray(heldout_treatment, dtype=float) - baseline
        weights = np.ones_like(contribution)
        definition = {
            "null_model": "fit_treatment_prevalence",
            "fit_baseline": baseline,
            "row_contribution": "heldout_treatment - fit_treatment_prevalence",
        }
    elif bank == "outcome":
        baseline = float(np.mean(fit_outcome))
        contribution = np.asarray(heldout_outcome, dtype=float) - baseline
        weights = np.ones_like(contribution)
        definition = {
            "null_model": "fit_outcome_mean",
            "fit_baseline": baseline,
            "row_contribution": "heldout_outcome - fit_outcome_mean",
        }
    elif bank == "effect" and dina_nuisance is not None:
        from ..models import dina

        dn, dv = dina_nuisance
        constant = dina.constant_effect(fit_outcome, fit_treatment, dn["a"], dn["nu"])
        contribution, weights, _ = dina.score(
            heldout_outcome, heldout_treatment, dv["a"], dv["nu"], delta=constant
        )
        definition = {
            "null_model": "fit_oof_constant_dina_log_odds_ratio",
            "fit_baseline": constant,
            "row_contribution": "(T-a)*(Y-expit(nu+(T-a)*delta0))",
            "centering": "DINA Fisher information",
            "effect_scale": "conditional_log_odds_ratio",
        }
    elif bank == "effect":
        constant = _constant_effect(
            fit_treatment,
            fit_outcome,
            fit_propensity,
            fit_outcome_prediction,
        )
        u = np.asarray(heldout_treatment, dtype=float) - np.asarray(
            heldout_propensity, dtype=float
        )
        v = np.asarray(heldout_outcome, dtype=float) - np.asarray(
            heldout_outcome_prediction, dtype=float
        )
        contribution = u * (v - constant * u)
        weights = np.square(u)
        definition = {
            "null_model": "fit_oof_constant_residual_effect",
            "fit_baseline": constant,
            "row_contribution": (
                "treatment_residual * (outcome_residual - "
                "fit_constant_effect * treatment_residual)"
            ),
        }
    else:
        raise ValueError(f"Unknown topic bank: {bank}")
    if not np.isfinite(contribution).all() or not np.isfinite(weights).all():
        raise ValueError(f"{bank} score contribution contains non-finite values")
    return contribution, weights, definition


def _topic_columns(
    topic: Mapping[str, Any],
    vocabulary: Mapping[str, int],
    *,
    terms_per_topic: int,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    records = [dict(record) for record in topic.get("terms", [])]
    if int(terms_per_topic) < 1:
        raise ValueError("terms_per_topic must be positive")
    if int(topic.get("terms_per_topic", terms_per_topic)) != int(terms_per_topic):
        raise ValueError(
            f"Topic {topic.get('topic_id')} changed its configured term capacity"
        )
    if len(records) != int(terms_per_topic):
        raise ValueError(
            f"Topic {topic.get('topic_id')} must contain exactly "
            f"{int(terms_per_topic)} configured term records"
        )
    missing = [str(record.get("term")) for record in records if str(record.get("term")) not in vocabulary]
    if missing:
        raise ValueError(
            f"Topic {topic.get('topic_id')} terms are absent from the common vocabulary: "
            f"{missing[:3]}"
        )
    return records, [int(vocabulary[str(record["term"])]) for record in records]


def _orphan_tokens(value: Any) -> Tuple[str, ...]:
    return tuple(
        token
        for token in re.findall(r"[a-z0-9]+", str(value or "").lower())
        if token not in _ORPHAN_STOPWORDS
    )


def _is_contiguous_subsequence(shorter: Sequence[str], longer: Sequence[str]) -> bool:
    if not shorter or len(shorter) > len(longer):
        return False
    width = len(shorter)
    return any(tuple(longer[start : start + width]) == tuple(shorter) for start in range(len(longer) - width + 1))


def _presence_jaccard(csc: sparse.csc_matrix, left: int, right: int) -> float:
    left_rows = csc.indices[csc.indptr[left] : csc.indptr[left + 1]]
    right_rows = csc.indices[csc.indptr[right] : csc.indptr[right + 1]]
    if not len(left_rows) and not len(right_rows):
        return 1.0
    intersection = int(np.intersect1d(left_rows, right_rows, assume_unique=True).size)
    union = int(len(left_rows) + len(right_rows) - intersection)
    return float(intersection / union) if union else 0.0


def build_fit_side_orphan_ngram_clusters(
    *,
    fit_matrix: sparse.spmatrix,
    feature_names: Sequence[str],
    effect_scores: Any,
    represented_topic_terms: Sequence[str],
    config: TfidfTopicDiscoveryConfig,
) -> Dict[str, Any]:
    """Build bounded raw effect n-gram groups without held-out labels.

    The effect score frame and topic summaries are both fitted inside the
    current inner-fit rows.  Nested phrases with near-identical document
    support collapse before a word/character/co-occurrence neighborhood is
    formed.  Greedy bounded neighborhoods prevent transitive graph chains from
    creating an oversized prompt or quadratic test.
    """
    x = sparse.csr_matrix(fit_matrix, dtype=float)
    names = [str(name) for name in feature_names]
    if x.shape[1] != len(names):
        raise ValueError("fit_matrix columns must align with feature_names")
    name_to_index = {name: index for index, name in enumerate(names)}
    frame = effect_scores.copy()
    required = {"feature", "signed_score", "unsigned_score"}
    if not required <= set(frame.columns):
        raise ValueError(
            "effect_scores must include feature, signed_score, and unsigned_score"
        )
    frame["feature"] = frame["feature"].astype(str)
    frame["fit_rank"] = np.arange(1, len(frame) + 1, dtype=int)
    if "eligible" in frame.columns:
        frame = frame.loc[frame["eligible"].astype(bool)].copy()
    frame = frame.loc[
        frame["signed_score"].astype(float).abs()
        >= float(config.orphan_ngram_min_abs_fit_score)
    ].copy()
    frame = frame.loc[frame["feature"].isin(name_to_index)].copy()
    represented = {str(term) for term in represented_topic_terms}
    represented_rows = frame.loc[frame["feature"].isin(represented)].copy()
    frame = frame.loc[~frame["feature"].isin(represented)].copy()
    if frame.empty:
        return {
            "candidate_count_before_topic_exclusion": int(len(represented_rows)),
            "represented_topic_term_exclusion_count": int(len(represented_rows)),
            "candidate_count_before_nested_deduplication": 0,
            "deduplicated_alias_count": 0,
            "representative_count": 0,
            "clusters": [],
        }
    frame = frame.sort_values(
        ["fit_rank", "unsigned_score", "feature"],
        ascending=[True, False, True],
        kind="stable",
    ).reset_index(drop=True)
    csc = x.tocsc(copy=True)
    csc.data = np.ones_like(csc.data)
    kept: List[Dict[str, Any]] = []
    aliases: Dict[str, List[Dict[str, Any]]] = {}
    kept_by_token: Dict[str, set[int]] = {}
    for row in frame.to_dict(orient="records"):
        term = str(row["feature"])
        column = int(name_to_index[term])
        tokens = _orphan_tokens(term)
        possible: set[int] = set()
        for token in set(tokens):
            possible.update(kept_by_token.get(token, set()))
        alias_owner: int | None = None
        for kept_index in sorted(possible):
            owner = kept[kept_index]
            owner_tokens = tuple(owner["tokens"])
            if not (
                _is_contiguous_subsequence(tokens, owner_tokens)
                or _is_contiguous_subsequence(owner_tokens, tokens)
            ):
                continue
            if _presence_jaccard(
                csc,
                column,
                int(owner["column_index"]),
            ) < float(
                config.orphan_semantic_clustering_scientific.alias_jaccard_threshold
            ):
                continue
            alias_owner = kept_index
            break
        record = {
            "term": term,
            "tokens": list(tokens),
            "column_index": column,
            "fit_rank": int(row["fit_rank"]),
            "fit_signed_score": float(row["signed_score"]),
            "fit_unsigned_score": float(abs(float(row["signed_score"]))),
            "combined_importance": float(row.get("combined_importance", 0.0)),
            "support_control": int(row.get("support_control", 0)),
            "support_treated": int(row.get("support_treated", 0)),
            "nuisance_source_agreement": float(
                row.get("nuisance_source_agreement", 0.0)
            ),
            "subsample_selection_stability": float(
                row.get("subsample_selection_stability", 0.0)
            ),
            "subsample_sign_agreement": float(
                row.get("subsample_sign_agreement", 0.0)
            ),
            "tail_contrast_sign_agreement": float(
                row.get("tail_contrast_sign_agreement", 0.0)
            ),
        }
        if alias_owner is not None:
            owner_term = str(kept[alias_owner]["term"])
            aliases.setdefault(owner_term, []).append(record)
            continue
        kept_index = len(kept)
        kept.append(record)
        for token in set(tokens):
            kept_by_token.setdefault(token, set()).add(kept_index)

    if not kept:
        clusters: List[Dict[str, Any]] = []
    elif len(kept) == 1:
        clusters = [{"members": [kept[0]], "within_seed_similarity": [1.0]}]
    else:
        terms = [str(record["term"]) for record in kept]
        semantic = config.orphan_semantic_clustering_scientific
        word = _make_bow_vectorizer(
            asdict(semantic.word_vectorizer)
        ).fit_transform(terms)
        char = _make_bow_vectorizer(
            asdict(semantic.char_vectorizer)
        ).fit_transform(terms)
        occurrence = x[:, [int(record["column_index"]) for record in kept]].T.tocsr()
        occurrence.data = np.ones_like(occurrence.data)
        combined = sparse.hstack(
            [
                np.sqrt(float(semantic.word_similarity_weight))
                * normalize(
                    word,
                    norm=semantic.row_normalization_norm,
                    axis=1,
                    copy=True,
                ),
                np.sqrt(float(semantic.char_similarity_weight))
                * normalize(
                    char,
                    norm=semantic.row_normalization_norm,
                    axis=1,
                    copy=True,
                ),
                np.sqrt(float(semantic.occurrence_similarity_weight))
                * normalize(
                    occurrence,
                    norm=semantic.row_normalization_norm,
                    axis=1,
                    copy=True,
                ),
            ],
            format="csr",
        )
        neighbor_count = min(
            len(kept), int(config.orphan_ngram_cluster_neighbors) + 1
        )
        distances, neighbors = NearestNeighbors(
            n_neighbors=neighbor_count,
            radius=1.0,
            algorithm=semantic.neighbor_algorithm,
            leaf_size=30,
            metric=semantic.neighbor_metric,
            p=2,
            metric_params=None,
            n_jobs=1,
        ).fit(combined).kneighbors(combined, return_distance=True)
        available = set(range(len(kept)))
        clusters = []
        maximum = int(config.orphan_ngram_cluster_max_terms)
        threshold = float(config.orphan_ngram_cluster_similarity_threshold)
        for seed in range(len(kept)):
            if seed not in available:
                continue
            ranked_neighbors = sorted(
                (
                    (float(1.0 - distance), int(index))
                    for distance, index in zip(distances[seed], neighbors[seed])
                    if int(index) in available
                    and (
                        int(index) == seed
                        or float(1.0 - distance) >= threshold
                    )
                ),
                key=lambda item: (-item[0], kept[item[1]]["fit_rank"], item[1]),
            )
            chosen = [index for _similarity, index in ranked_neighbors[:maximum]]
            if seed not in chosen:
                chosen.insert(0, seed)
                chosen = chosen[:maximum]
            chosen = list(dict.fromkeys(chosen))
            for index in chosen:
                available.discard(index)
            similarity_by_index = {
                index: similarity for similarity, index in ranked_neighbors
            }
            clusters.append(
                {
                    "members": [kept[index] for index in chosen],
                    "within_seed_similarity": [
                        float(similarity_by_index.get(index, 1.0 if index == seed else 0.0))
                        for index in chosen
                    ],
                }
            )

    rendered_clusters: List[Dict[str, Any]] = []
    for cluster_index, cluster in enumerate(clusters, start=1):
        records = []
        for member, similarity in zip(
            cluster["members"], cluster["within_seed_similarity"]
        ):
            term = str(member["term"])
            records.append(
                {
                    key: value
                    for key, value in member.items()
                    if key not in {"tokens", "column_index"}
                }
                | {
                    "cluster_seed_similarity": float(similarity),
                    "nested_aliases": [
                        {
                            key: value
                            for key, value in alias.items()
                            if key not in {"tokens", "column_index"}
                        }
                        for alias in aliases.get(term, [])
                    ],
                }
            )
        rendered_clusters.append(
            {
                "cluster_id": f"effect_orphan_cluster_{cluster_index:03d}",
                "bank": "effect",
                "evidence_kind": "fit_side_orphan_raw_ngram_cluster",
                "terms": records,
            }
        )
    return {
        "candidate_count_before_topic_exclusion": int(len(frame) + len(represented_rows)),
        "represented_topic_term_exclusion_count": int(len(represented_rows)),
        "candidate_count_before_nested_deduplication": int(len(frame)),
        "deduplicated_alias_count": int(sum(map(len, aliases.values()))),
        "representative_count": int(len(kept)),
        "clusters": rendered_clusters,
    }


def _single_topic_score(
    *,
    topic: Mapping[str, Any],
    fit_topic_values: np.ndarray,
    heldout_topic_values: np.ndarray,
    fit_matrix: sparse.spmatrix,
    heldout_matrix: sparse.spmatrix,
    vocabulary: Mapping[str, int],
    contribution: np.ndarray,
    centering_weights: np.ndarray,
    terms_per_topic: int,
) -> Dict[str, Any]:
    records, columns = _topic_columns(
        topic,
        vocabulary,
        terms_per_topic=terms_per_topic,
    )
    fit_topic = np.asarray(fit_topic_values, dtype=float).reshape(-1)
    heldout_topic = np.asarray(heldout_topic_values, dtype=float).reshape(-1)
    if fit_topic.shape[0] != fit_matrix.shape[0]:
        raise ValueError(f"Topic {topic.get('topic_id')} fit scores are misaligned")
    if heldout_topic.shape[0] != heldout_matrix.shape[0]:
        raise ValueError(f"Topic {topic.get('topic_id')} held-out scores are misaligned")
    topic_fit_scale = float(np.std(fit_topic))
    topic_fit_testable = topic_fit_scale > 1e-12
    if not topic_fit_testable:
        topic_fit_scale = 1.0
    standardized_topic = (
        heldout_topic - float(np.mean(fit_topic))
    ) / topic_fit_scale
    weight_sum = float(np.sum(centering_weights))
    if weight_sum <= 1e-12:
        centered_topic = standardized_topic - float(np.mean(standardized_topic))
    else:
        centered_topic = standardized_topic - float(
            np.dot(centering_weights, standardized_topic) / weight_sum
        )
    topic_row_scores = centered_topic * contribution
    topic_moment = float(np.mean(topic_row_scores))
    topic_row_scale = float(np.std(topic_row_scores, ddof=1))
    topic_testable = bool(topic_fit_testable and topic_row_scale > 1e-12)
    topic_standardized_score = (
        float(np.sqrt(len(topic_row_scores)) * topic_moment / topic_row_scale)
        if topic_testable
        else 0.0
    )
    scalar_topic_result = {
        "topic_score_testable": topic_testable,
        "topic_score_moment": topic_moment,
        "topic_standardized_score": topic_standardized_score,
        "topic_unadjusted_two_sided_p": float(
            2.0 * norm.sf(abs(topic_standardized_score))
        ),
        "topic_fit_mean": float(np.mean(fit_topic)),
        "topic_fit_standard_deviation": (
            float(np.std(fit_topic)) if topic_fit_testable else 0.0
        ),
        "_topic_bootstrap_rows": (
            ((topic_row_scores - topic_moment) / topic_row_scale)[:, None]
            if topic_testable
            else np.zeros((len(topic_row_scores), 0), dtype=float)
        ),
    }
    fit_values = np.asarray(fit_matrix[:, columns].toarray(), dtype=float)
    heldout_values = np.asarray(heldout_matrix[:, columns].toarray(), dtype=float)
    fit_means = np.mean(fit_values, axis=0)
    fit_scales = np.std(fit_values, axis=0)
    fit_scales = np.where(fit_scales > 1e-12, fit_scales, 1.0)
    standardized = (heldout_values - fit_means) / fit_scales

    if weight_sum <= 1e-12:
        centered = standardized - np.mean(standardized, axis=0)
    else:
        weighted_means = np.sum(
            centering_weights[:, None] * standardized, axis=0
        ) / weight_sum
        centered = standardized - weighted_means
    row_scores = centered * contribution[:, None]
    column_means = np.mean(row_scores, axis=0)
    column_scales = np.std(row_scores, axis=0, ddof=1)
    retained = column_scales > 1e-12
    n_rows = int(row_scores.shape[0])

    if not np.any(retained):
        return {
            "topic_id": str(topic["topic_id"]),
            **scalar_topic_result,
            "quadratic_statistic": 0.0,
            "quadratic_covariance_rank": 0,
            "quadratic_statistic_per_rank": 0.0,
            "asymptotic_p": 1.0,
            "maximum_absolute_standardized_score": 0.0,
            "term_scores": [
                {
                    **record,
                    "heldout_standardized_score": 0.0,
                    "heldout_score_moment": 0.0,
                    "unadjusted_two_sided_p": 1.0,
                    "testable_in_heldout": False,
                }
                for record in records
            ],
            "_row_scores": np.zeros((n_rows, 0), dtype=float),
            "_inverse_covariance": np.zeros((0, 0), dtype=float),
            "_column_scales": np.zeros(0, dtype=float),
            "_retained_term_positions": [],
        }

    test_scores = row_scores[:, retained]
    means = column_means[retained]
    scales = column_scales[retained]
    score_vector = np.sqrt(n_rows) * means
    covariance = np.atleast_2d(np.cov(test_scores, rowvar=False, ddof=1))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    tolerance = max(float(np.max(eigenvalues)) * 1e-8, 1e-12)
    nonzero = eigenvalues > tolerance
    rank = int(np.sum(nonzero))
    if rank:
        inverse = (eigenvectors[:, nonzero] / eigenvalues[nonzero]) @ eigenvectors[
            :, nonzero
        ].T
        quadratic = float(score_vector @ inverse @ score_vector)
        asymptotic_p = float(chi2.sf(quadratic, rank))
        energy = float(quadratic / rank)
    else:
        inverse = np.zeros_like(covariance)
        quadratic = 0.0
        asymptotic_p = 1.0
        energy = 0.0
    standardized_scores = score_vector / scales

    term_scores: List[Dict[str, Any]] = []
    retained_position = 0
    for record, keep, moment in zip(records, retained, column_means):
        if keep:
            statistic = float(standardized_scores[retained_position])
            retained_position += 1
            term_scores.append(
                {
                    **record,
                    "heldout_standardized_score": statistic,
                    "heldout_score_moment": float(moment),
                    "unadjusted_two_sided_p": float(2.0 * norm.sf(abs(statistic))),
                    "testable_in_heldout": True,
                }
            )
        else:
            term_scores.append(
                {
                    **record,
                    "heldout_standardized_score": 0.0,
                    "heldout_score_moment": float(moment),
                    "unadjusted_two_sided_p": 1.0,
                    "testable_in_heldout": False,
                }
            )
    return {
        "topic_id": str(topic["topic_id"]),
        **scalar_topic_result,
        "quadratic_statistic": quadratic,
        "quadratic_covariance_rank": rank,
        "quadratic_statistic_per_rank": energy,
        "asymptotic_p": asymptotic_p,
        "maximum_absolute_standardized_score": float(
            np.max(np.abs(standardized_scores))
        ),
        "term_scores": term_scores,
        "_row_scores": test_scores,
        "_inverse_covariance": inverse,
        "_column_scales": scales,
        "_retained_term_positions": np.flatnonzero(retained).astype(int).tolist(),
    }


def _single_orphan_cluster_score(
    *,
    cluster: Mapping[str, Any],
    fit_matrix: sparse.spmatrix,
    heldout_matrix: sparse.spmatrix,
    vocabulary: Mapping[str, int],
    contribution: np.ndarray,
    centering_weights: np.ndarray,
) -> Dict[str, Any]:
    records = [dict(record) for record in cluster.get("terms", [])]
    if not 1 <= len(records) <= 15:
        raise ValueError(
            f"Orphan cluster {cluster.get('cluster_id')} must contain 1-15 terms"
        )
    missing = [
        str(record.get("term"))
        for record in records
        if str(record.get("term")) not in vocabulary
    ]
    if missing:
        raise ValueError(
            f"Orphan cluster {cluster.get('cluster_id')} terms are absent from "
            f"the common vocabulary: {missing[:3]}"
        )
    columns = [int(vocabulary[str(record["term"])]) for record in records]
    fit_values = np.asarray(fit_matrix[:, columns].toarray(), dtype=float)
    heldout_values = np.asarray(heldout_matrix[:, columns].toarray(), dtype=float)
    fit_means = np.mean(fit_values, axis=0)
    raw_fit_scales = np.std(fit_values, axis=0)
    fit_scales = np.where(raw_fit_scales > 1e-12, raw_fit_scales, 1.0)
    standardized = (heldout_values - fit_means) / fit_scales
    weight_sum = float(np.sum(centering_weights))
    if weight_sum <= 1e-12:
        centered = standardized - np.mean(standardized, axis=0)
    else:
        centered = standardized - (
            np.sum(centering_weights[:, None] * standardized, axis=0)
            / weight_sum
        )
    row_scores = centered * contribution[:, None]
    column_means = np.mean(row_scores, axis=0)
    column_scales = np.std(row_scores, axis=0, ddof=1)
    retained = (raw_fit_scales > 1e-12) & (column_scales > 1e-12)
    n_rows = int(row_scores.shape[0])
    base = {
        "topic_id": str(cluster["cluster_id"]),
        "cluster_id": str(cluster["cluster_id"]),
        "bank": "effect",
        "evidence_kind": "orphan_raw_ngram_cluster",
        "topic_score_testable": False,
        "topic_score_moment": 0.0,
        "topic_standardized_score": 0.0,
        "topic_unadjusted_two_sided_p": 1.0,
        "topic_fit_mean": 0.0,
        "topic_fit_standard_deviation": 0.0,
        "_topic_bootstrap_rows": np.zeros((n_rows, 0), dtype=float),
    }
    if not np.any(retained):
        return {
            **base,
            "quadratic_statistic": 0.0,
            "quadratic_covariance_rank": 0,
            "quadratic_statistic_per_rank": 0.0,
            "asymptotic_p": 1.0,
            "maximum_absolute_standardized_score": 0.0,
            "term_scores": [
                {
                    **record,
                    "heldout_standardized_score": 0.0,
                    "heldout_score_moment": 0.0,
                    "unadjusted_two_sided_p": 1.0,
                    "testable_in_heldout": False,
                }
                for record in records
            ],
            "_row_scores": np.zeros((n_rows, 0), dtype=float),
            "_inverse_covariance": np.zeros((0, 0), dtype=float),
            "_column_scales": np.zeros(0, dtype=float),
            "_retained_term_positions": [],
        }
    test_scores = row_scores[:, retained]
    means = column_means[retained]
    scales = column_scales[retained]
    score_vector = np.sqrt(n_rows) * means
    covariance = np.atleast_2d(np.cov(test_scores, rowvar=False, ddof=1))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    tolerance = max(float(np.max(eigenvalues)) * 1e-8, 1e-12)
    nonzero = eigenvalues > tolerance
    rank = int(np.sum(nonzero))
    if rank:
        inverse = (
            eigenvectors[:, nonzero] / eigenvalues[nonzero]
        ) @ eigenvectors[:, nonzero].T
        quadratic = float(score_vector @ inverse @ score_vector)
        asymptotic_p = float(chi2.sf(quadratic, rank))
        energy = float(quadratic / rank)
    else:
        inverse = np.zeros_like(covariance)
        quadratic = 0.0
        asymptotic_p = 1.0
        energy = 0.0
    standardized_scores = score_vector / scales
    term_scores: List[Dict[str, Any]] = []
    retained_position = 0
    for record, keep, moment in zip(records, retained, column_means):
        if keep:
            statistic = float(standardized_scores[retained_position])
            retained_position += 1
            term_scores.append(
                {
                    **record,
                    "heldout_standardized_score": statistic,
                    "heldout_score_moment": float(moment),
                    "unadjusted_two_sided_p": float(2.0 * norm.sf(abs(statistic))),
                    "testable_in_heldout": True,
                }
            )
        else:
            term_scores.append(
                {
                    **record,
                    "heldout_standardized_score": 0.0,
                    "heldout_score_moment": float(moment),
                    "unadjusted_two_sided_p": 1.0,
                    "testable_in_heldout": False,
                }
            )
    return {
        **base,
        "quadratic_statistic": quadratic,
        "quadratic_covariance_rank": rank,
        "quadratic_statistic_per_rank": energy,
        "asymptotic_p": asymptotic_p,
        "maximum_absolute_standardized_score": float(
            np.max(np.abs(standardized_scores))
        ),
        "term_scores": term_scores,
        "_row_scores": test_scores,
        "_inverse_covariance": inverse,
        "_column_scales": scales,
        "_retained_term_positions": np.flatnonzero(retained).astype(int).tolist(),
    }


def _bootstrap_candidates(
    results: List[Dict[str, Any]],
    *,
    repeats: int,
    top_topics: int,
    chunk_size: int,
    seed: int,
) -> Dict[str, Any]:
    """Add shared-multiplier p-values for topic, term-group, and n-gram families."""
    scalar_testable = [
        result for result in results if bool(result.get("topic_score_testable"))
    ]
    term_group_testable = [
        result
        for result in results
        if int(result.get("quadratic_covariance_rank", 0)) > 0
    ]
    testable_ngram_occurrences = int(
        sum(
            bool(term.get("testable_in_heldout"))
            for result in results
            for term in result.get("term_scores", [])
        )
    )
    if repeats <= 0 or not results:
        return {
            "enabled": False,
            "repeats": int(repeats),
            "testable_topic_count": len(scalar_testable),
            "bootstrapped_topic_count": 0,
            "complete_topic_family": False,
            "testable_term_group_count": len(term_group_testable),
            "bootstrapped_term_group_count": 0,
            "complete_term_group_family": False,
            "testable_ngram_occurrence_count": testable_ngram_occurrences,
            "bootstrapped_ngram_occurrence_count": 0,
            "complete_ngram_family": False,
        }
    testable = sorted(
        {
            id(result): result
            for result in [*scalar_testable, *term_group_testable]
        }.values(),
        key=lambda result: (
            float(
                result.get(
                    "topic_unadjusted_two_sided_p",
                    result.get("asymptotic_p", 1.0),
                )
            ),
            -abs(float(result.get("topic_standardized_score", 0.0))),
            -float(result["quadratic_statistic_per_rank"]),
            str(result["topic_id"]),
        ),
    )
    limit = len(testable) if int(top_topics) == 0 else min(int(top_topics), len(testable))
    candidates = testable[:limit]
    if not candidates:
        return {
            "enabled": True,
            "repeats": int(repeats),
            "testable_topic_count": 0,
            "bootstrapped_topic_count": 0,
            "complete_topic_family": True,
            "testable_term_group_count": 0,
            "bootstrapped_term_group_count": 0,
            "complete_term_group_family": True,
            "testable_ngram_occurrence_count": 0,
            "bootstrapped_ngram_occurrence_count": 0,
            "complete_ngram_family": True,
        }
    n_rows = int(candidates[0]["_topic_bootstrap_rows"].shape[0])
    if any(
        int(result["_topic_bootstrap_rows"].shape[0]) != n_rows
        or int(result["_row_scores"].shape[0]) != n_rows
        for result in candidates
    ):
        raise ValueError("Topic score rows are misaligned within a bank")

    joint_blocks: List[np.ndarray] = []
    slices: List[Dict[str, slice]] = []
    column_offset = 0
    for result in candidates:
        scalar_rows = np.asarray(result["_topic_bootstrap_rows"], dtype=float)
        row_scores = np.asarray(result["_row_scores"], dtype=float)
        centered = row_scores - np.mean(row_scores, axis=0)
        if int(result["quadratic_covariance_rank"]) > 0:
            inverse = np.asarray(result["_inverse_covariance"], dtype=float)
            eigenvalues, eigenvectors = np.linalg.eigh(inverse)
            tolerance = max(float(np.max(eigenvalues)) * 1e-10, 1e-14)
            retained = eigenvalues > tolerance
            whitener = eigenvectors[:, retained] * np.sqrt(eigenvalues[retained])
            whitened_rows = centered @ whitener
            standardized_rows = centered / np.asarray(
                result["_column_scales"], dtype=float
            )
        else:
            whitened_rows = np.zeros((n_rows, 0), dtype=float)
            standardized_rows = np.zeros((n_rows, 0), dtype=float)
        scalar_slice = slice(column_offset, column_offset + scalar_rows.shape[1])
        column_offset = scalar_slice.stop
        white_slice = slice(column_offset, column_offset + whitened_rows.shape[1])
        column_offset = white_slice.stop
        standardized_slice = slice(
            column_offset, column_offset + standardized_rows.shape[1]
        )
        column_offset = standardized_slice.stop
        joint_blocks.extend([scalar_rows, whitened_rows, standardized_rows])
        slices.append(
            {
                "scalar": scalar_slice,
                "whitened": white_slice,
                "standardized": standardized_slice,
            }
        )
    joint_rows = np.column_stack(joint_blocks)

    raw_topic_exceed = np.zeros(len(candidates), dtype=int)
    family_topic_exceed = np.zeros(len(candidates), dtype=int)
    raw_quadratic_exceed = np.zeros(len(candidates), dtype=int)
    raw_maximum_exceed = np.zeros(len(candidates), dtype=int)
    family_energy_exceed = np.zeros(len(candidates), dtype=int)
    family_maximum_exceed = np.zeros(len(candidates), dtype=int)
    raw_term_exceed = [
        np.zeros(len(result["_retained_term_positions"]), dtype=int)
        for result in candidates
    ]
    family_term_exceed = [np.zeros_like(values) for values in raw_term_exceed]
    observed_energy = np.asarray(
        [float(result["quadratic_statistic_per_rank"]) for result in candidates]
    )
    observed_topics = np.asarray(
        [abs(float(result.get("topic_standardized_score", 0.0))) for result in candidates]
    )
    observed_maximum = np.asarray(
        [float(result["maximum_absolute_standardized_score"]) for result in candidates]
    )
    observed_terms = [
        np.asarray(
            [
                abs(
                    float(
                        result["term_scores"][position][
                            "heldout_standardized_score"
                        ]
                    )
                )
                for position in result["_retained_term_positions"]
            ],
            dtype=float,
        )
        for result in candidates
    ]
    rng = np.random.default_rng(seed)
    root_n = np.sqrt(n_rows)
    for start in range(0, repeats, chunk_size):
        stop = min(repeats, start + chunk_size)
        multipliers = rng.choice(
            np.asarray([-1.0, 1.0]), size=(stop - start, n_rows)
        )
        bootstrap_joint = multipliers @ joint_rows / root_n
        topic_columns = []
        energy_columns = []
        maximum_columns = []
        for index, (result, topic_slices) in enumerate(zip(candidates, slices)):
            scalar = np.abs(bootstrap_joint[:, topic_slices["scalar"]])
            whitened = bootstrap_joint[:, topic_slices["whitened"]]
            standardized = bootstrap_joint[:, topic_slices["standardized"]]
            topic_statistic = (
                scalar[:, 0] if scalar.shape[1] else np.zeros(stop - start)
            )
            if bool(result.get("topic_score_testable")):
                raw_topic_exceed[index] += int(
                    np.sum(topic_statistic >= observed_topics[index])
                )
            if int(result["quadratic_covariance_rank"]) > 0:
                quadratic = np.sum(np.square(whitened), axis=1)
                energy = quadratic / int(result["quadratic_covariance_rank"])
                maximum = np.max(np.abs(standardized), axis=1)
                raw_quadratic_exceed[index] += int(
                    np.sum(quadratic >= float(result["quadratic_statistic"]))
                )
                raw_maximum_exceed[index] += int(
                    np.sum(
                        maximum
                        >= float(result["maximum_absolute_standardized_score"])
                    )
                )
                raw_term_exceed[index] += np.sum(
                    np.abs(standardized) >= observed_terms[index][None, :],
                    axis=0,
                ).astype(int)
            else:
                energy = np.zeros(stop - start)
                maximum = np.zeros(stop - start)
            topic_columns.append(topic_statistic)
            energy_columns.append(energy)
            maximum_columns.append(maximum)
        family_topic = np.max(np.column_stack(topic_columns), axis=1)
        family_energy = np.max(np.column_stack(energy_columns), axis=1)
        family_maximum = np.max(np.column_stack(maximum_columns), axis=1)
        for index in range(len(candidates)):
            if bool(candidates[index].get("topic_score_testable")):
                family_topic_exceed[index] += int(
                    np.sum(family_topic >= observed_topics[index])
                )
            family_energy_exceed[index] += int(
                np.sum(family_energy >= observed_energy[index])
            )
            family_maximum_exceed[index] += int(
                np.sum(family_maximum >= observed_maximum[index])
            )
            family_term_exceed[index] += np.sum(
                family_maximum[:, None] >= observed_terms[index][None, :],
                axis=0,
            ).astype(int)

    bootstrapped_topic_count = int(
        sum(bool(result.get("topic_score_testable")) for result in candidates)
    )
    bootstrapped_term_group_count = int(
        sum(int(result["quadratic_covariance_rank"]) > 0 for result in candidates)
    )
    bootstrapped_ngram_occurrences = int(
        sum(len(result["_retained_term_positions"]) for result in candidates)
    )
    complete_topic_family = bootstrapped_topic_count == len(scalar_testable)
    complete_term_group_family = (
        bootstrapped_term_group_count == len(term_group_testable)
    )
    complete_ngram_family = (
        complete_term_group_family
        and bootstrapped_ngram_occurrences == testable_ngram_occurrences
    )
    for index, result in enumerate(candidates):
        if bool(result.get("topic_score_testable")):
            result["topic_multiplier_p"] = float(
                (1 + raw_topic_exceed[index]) / (repeats + 1)
            )
            result["topic_familywise_p"] = float(
                (1 + family_topic_exceed[index]) / (repeats + 1)
            )
        if int(result["quadratic_covariance_rank"]) > 0:
            result["quadratic_multiplier_p"] = float(
                (1 + raw_quadratic_exceed[index]) / (repeats + 1)
            )
            result["maximum_multiplier_p"] = float(
                (1 + raw_maximum_exceed[index]) / (repeats + 1)
            )
            result["candidate_familywise_energy_p"] = float(
                (1 + family_energy_exceed[index]) / (repeats + 1)
            )
            result["candidate_familywise_maximum_p"] = float(
                (1 + family_maximum_exceed[index]) / (repeats + 1)
            )
        result["bootstrap_repeats"] = int(repeats)
        result["bootstrap_complete_topic_family"] = complete_topic_family
        for retained_index, term_position in enumerate(
            result["_retained_term_positions"]
        ):
            term = result["term_scores"][term_position]
            term["multiplier_p"] = float(
                (1 + raw_term_exceed[index][retained_index]) / (repeats + 1)
            )
            term["ngram_familywise_p"] = float(
                (1 + family_term_exceed[index][retained_index]) / (repeats + 1)
            )
            term["bootstrap_complete_ngram_family"] = complete_ngram_family
    return {
        "enabled": True,
        "repeats": int(repeats),
        "top_topic_limit": int(top_topics),
        "testable_topic_count": len(scalar_testable),
        "bootstrapped_topic_count": bootstrapped_topic_count,
        "complete_topic_family": complete_topic_family,
        "testable_term_group_count": len(term_group_testable),
        "bootstrapped_term_group_count": bootstrapped_term_group_count,
        "complete_term_group_family": complete_term_group_family,
        "testable_ngram_occurrence_count": testable_ngram_occurrences,
        "bootstrapped_ngram_occurrence_count": bootstrapped_ngram_occurrences,
        "complete_ngram_family": complete_ngram_family,
        "shared_row_multipliers_across_topics": True,
        "topic_familywise_statistic": (
            "maximum_absolute_standardized_nmf_topic_score"
        ),
        "term_group_familywise_statistic": (
            "maximum_quadratic_statistic_per_covariance_rank"
        ),
        "ngram_familywise_statistic": (
            "maximum_absolute_standardized_ngram_score"
        ),
    }


def _select_ngrams(
    results: List[Dict[str, Any]],
    config: TfidfTopicDiscoveryConfig,
    bootstrap: Mapping[str, Any],
) -> Dict[str, Any]:
    """Calibrate every unique supplied n-gram and annotate all occurrences.

    A term may load on more than one NMF topic.  It is one statistical
    hypothesis, so multiplicity adjustment is performed over unique strings,
    while every topic occurrence retains the resulting evidence and topic
    provenance.  There is deliberately no minimum-significance fallback for
    individual terms: the bounded topic fallback already guarantees that an
    agent receives evidence when a small held-out fold has low power.
    """
    instances_by_term: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    for result in results:
        topic_id = str(result["topic_id"])
        for term_score in result.get("term_scores", []):
            term = str(term_score.get("term") or "")
            if not term:
                continue
            instances_by_term.setdefault(term, []).append((topic_id, term_score))
    if not instances_by_term:
        return {
            "ngram_tests": [],
            "selected_ngrams": [],
            "selected_ngram_terms": [],
            "ngram_selection_count": 0,
            "ngram_selection_rule": "no_ngrams",
            "unique_ngram_count": 0,
            "testable_unique_ngram_count": 0,
        }

    complete_bootstrap = bool(bootstrap.get("complete_ngram_family"))
    rows: List[Dict[str, Any]] = []
    for term, instances in sorted(instances_by_term.items()):
        testable = [
            (topic_id, score)
            for topic_id, score in instances
            if bool(score.get("testable_in_heldout"))
        ]
        candidates = testable or instances
        representative_topic, representative = min(
            candidates,
            key=lambda item: (
                float(item[1].get("unadjusted_two_sided_p", 1.0)),
                -abs(float(item[1].get("heldout_standardized_score", 0.0))),
                item[0],
            ),
        )
        if complete_bootstrap and "multiplier_p" in representative:
            primary_p = float(representative["multiplier_p"])
            primary_source = "complete_family_multiplier_bootstrap"
        else:
            primary_p = float(
                representative.get("unadjusted_two_sided_p", 1.0)
            )
            primary_source = "asymptotic_complete_unique_ngram_family"
        familywise_values = [
            float(score["ngram_familywise_p"])
            for _topic_id, score in instances
            if "ngram_familywise_p" in score
        ]
        rows.append(
            {
                "term": term,
                "topic_ids": sorted({topic_id for topic_id, _score in instances}),
                "representative_topic_id": representative_topic,
                "topic_occurrence_count": len(instances),
                "testable_in_heldout": bool(testable),
                "heldout_standardized_score": float(
                    representative.get("heldout_standardized_score", 0.0)
                ),
                "heldout_score_moment": float(
                    representative.get("heldout_score_moment", 0.0)
                ),
                "unadjusted_two_sided_p": float(
                    representative.get("unadjusted_two_sided_p", 1.0)
                ),
                "primary_p": primary_p,
                "primary_p_source": primary_source,
                "familywise_p": (
                    min(familywise_values) if familywise_values else None
                ),
                "best_screen_rank": min(
                    int(score.get("screen_rank") or 10**9)
                    for _topic_id, score in instances
                ),
                "maximum_topic_loading": max(
                    float(score.get("loading") or 0.0)
                    for _topic_id, score in instances
                ),
                "fit_signed_score": float(
                    representative.get("signed_score", 0.0)
                ),
            }
        )

    adjusted = benjamini_hochberg([row["primary_p"] for row in rows])
    for row, q_value in zip(rows, adjusted):
        row["fdr_q"] = float(q_value)
    order = sorted(
        range(len(rows)),
        key=lambda index: (
            not bool(rows[index]["testable_in_heldout"]),
            float(rows[index]["primary_p"]),
            -abs(float(rows[index]["heldout_standardized_score"])),
            int(rows[index]["best_screen_rank"]),
            str(rows[index]["term"]),
        ),
    )
    selected: List[int] = []
    for index in order:
        row = rows[index]
        row["selection_reason"] = "not_selected"
        if not bool(row["testable_in_heldout"]):
            continue
        if float(row["fdr_q"]) <= float(config.score_test_fdr_level):
            row["selection_reason"] = "fdr"
            selected.append(index)
        elif (
            complete_bootstrap
            and row["familywise_p"] is not None
            and float(row["familywise_p"])
            <= float(config.score_test_p_threshold)
        ):
            row["selection_reason"] = "ngram_familywise_p"
            selected.append(index)
    selected_set = set(selected)
    rank_by_index = {index: rank for rank, index in enumerate(order, start=1)}
    row_by_term: Dict[str, Dict[str, Any]] = {}
    for index, row in enumerate(rows):
        row["evidence_rank"] = int(rank_by_index[index])
        row["selected_for_agent_evidence"] = index in selected_set
        row_by_term[str(row["term"])] = row
    for term, instances in instances_by_term.items():
        evidence = row_by_term[term]
        for _topic_id, score in instances:
            score["ngram_primary_p"] = float(evidence["primary_p"])
            score["ngram_primary_p_source"] = str(evidence["primary_p_source"])
            score["ngram_fdr_q"] = float(evidence["fdr_q"])
            score["ngram_global_familywise_p"] = evidence["familywise_p"]
            score["ngram_evidence_rank"] = int(evidence["evidence_rank"])
            score["selected_for_agent_evidence"] = bool(
                evidence["selected_for_agent_evidence"]
            )
            score["ngram_selection_reason"] = str(evidence["selection_reason"])

    selected_rows = [rows[index] for index in selected]
    selected_rows.sort(key=lambda row: int(row["evidence_rank"]))
    return {
        "ngram_tests": sorted(rows, key=lambda row: int(row["evidence_rank"])),
        "selected_ngrams": selected_rows,
        "selected_ngram_terms": [str(row["term"]) for row in selected_rows],
        "ngram_selection_count": len(selected_rows),
        "ngram_selection_rule": (
            "unique_ngram_fdr_or_complete_familywise_p"
            if complete_bootstrap
            else "unique_ngram_fdr"
        ),
        "unique_ngram_count": len(rows),
        "testable_unique_ngram_count": int(
            sum(bool(row["testable_in_heldout"]) for row in rows)
        ),
        "complete_family_multiplier_bootstrap": complete_bootstrap,
    }


def _select_topics(
    results: List[Dict[str, Any]],
    config: TfidfTopicDiscoveryConfig,
    bootstrap: Mapping[str, Any],
) -> Dict[str, Any]:
    if not results:
        return {
            "selected_topic_ids": [],
            "selection_count": 0,
            "selection_rule": "no_topics",
        }
    complete_bootstrap = bool(bootstrap.get("complete_topic_family"))
    complete_term_group_bootstrap = bool(
        bootstrap.get("complete_term_group_family")
    )
    for result in results:
        if complete_bootstrap and "topic_multiplier_p" in result:
            result["primary_p"] = float(result["topic_multiplier_p"])
            result["primary_p_source"] = "complete_family_multiplier_bootstrap"
        else:
            result["primary_p"] = float(
                result.get("topic_unadjusted_two_sided_p", 1.0)
            )
            result["primary_p_source"] = "asymptotic_nmf_topic_score"
        result["familywise_p"] = (
            float(result["topic_familywise_p"])
            if "topic_familywise_p" in result
            else None
        )
        if complete_term_group_bootstrap and "quadratic_multiplier_p" in result:
            result["term_group_primary_p"] = float(
                result["quadratic_multiplier_p"]
            )
            result["term_group_primary_p_source"] = (
                "complete_family_multiplier_bootstrap"
            )
        else:
            result["term_group_primary_p"] = float(result["asymptotic_p"])
            result["term_group_primary_p_source"] = (
                "asymptotic_joint_ngram_group_score"
            )
        result["term_group_familywise_p"] = (
            float(result["candidate_familywise_energy_p"])
            if "candidate_familywise_energy_p" in result
            else None
        )
    adjusted = benjamini_hochberg([result["primary_p"] for result in results])
    term_group_adjusted = benjamini_hochberg(
        [result["term_group_primary_p"] for result in results]
    )
    for result, q_value, term_group_q in zip(
        results, adjusted, term_group_adjusted
    ):
        result["fdr_q"] = float(q_value)
        result["term_group_fdr_q"] = float(term_group_q)
        selected_terms = [
            term
            for term in result.get("term_scores", [])
            if bool(term.get("selected_for_agent_evidence"))
        ]
        result["selected_ngram_count"] = len(selected_terms)
        result["selected_ngram_terms"] = [
            str(term.get("term")) for term in selected_terms
        ]
        result["strongest_selected_ngram_rank"] = (
            min(int(term["ngram_evidence_rank"]) for term in selected_terms)
            if selected_terms
            else None
        )
        result["filter_priority_p"] = min(
            float(result["primary_p"]),
            float(result["term_group_primary_p"]),
            min(
                (
                    float(term.get("ngram_primary_p", 1.0))
                    for term in selected_terms
                ),
                default=1.0,
            ),
        )
    order = sorted(
        range(len(results)),
        key=lambda index: (
            float(results[index]["filter_priority_p"]),
            -int(results[index]["selected_ngram_count"]),
            float(results[index]["primary_p"]),
            -abs(float(results[index].get("topic_standardized_score", 0.0))),
            -float(results[index]["quadratic_statistic_per_rank"]),
            str(results[index]["topic_id"]),
        ),
    )
    testable_order = [
        index
        for index in order
        if bool(results[index].get("topic_score_testable"))
        or int(results[index]["quadratic_covariance_rank"]) > 0
        or int(results[index]["selected_ngram_count"]) > 0
    ]
    maximum = min(
        int(config.score_test_max_topics_per_bank), len(testable_order)
    )
    minimum = min(int(config.score_test_min_topics_per_bank), maximum)
    selected = [
        index
        for index in testable_order
        if float(results[index]["fdr_q"]) <= float(config.score_test_fdr_level)
        or float(results[index]["term_group_fdr_q"])
        <= float(config.score_test_fdr_level)
        or int(results[index]["selected_ngram_count"]) > 0
        or (
            complete_bootstrap
            and results[index]["familywise_p"] is not None
            and float(results[index]["familywise_p"])
            <= float(config.score_test_p_threshold)
        )
        or (
            complete_term_group_bootstrap
            and results[index]["term_group_familywise_p"] is not None
            and float(results[index]["term_group_familywise_p"])
            <= float(config.score_test_p_threshold)
        )
        or (
            not complete_bootstrap
            and float(results[index]["primary_p"])
            <= float(config.score_test_p_threshold)
        )
        or (
            not complete_term_group_bootstrap
            and float(results[index]["term_group_primary_p"])
            <= float(config.score_test_p_threshold)
        )
    ][:maximum]
    selected_set = set(selected)
    for index in testable_order:
        if len(selected) >= minimum:
            break
        if index not in selected_set:
            selected.append(index)
            selected_set.add(index)
    selected = sorted(
        selected,
        key=lambda index: order.index(index),
    )[:maximum]
    selected_set = set(selected)
    for rank, index in enumerate(order, start=1):
        result = results[index]
        result["evidence_rank"] = int(rank)
        result["selected_for_agent"] = index in selected_set
        topic_fdr = float(result["fdr_q"]) <= float(config.score_test_fdr_level)
        topic_familywise = (
            complete_bootstrap
            and result["familywise_p"] is not None
            and float(result["familywise_p"])
            <= float(config.score_test_p_threshold)
        )
        topic_unadjusted = (
            not complete_bootstrap
            and float(result["primary_p"])
            <= float(config.score_test_p_threshold)
        )
        term_group_fdr = float(result["term_group_fdr_q"]) <= float(
            config.score_test_fdr_level
        )
        term_group_familywise = (
            complete_term_group_bootstrap
            and result["term_group_familywise_p"] is not None
            and float(result["term_group_familywise_p"])
            <= float(config.score_test_p_threshold)
        )
        term_group_unadjusted = (
            not complete_term_group_bootstrap
            and float(result["term_group_primary_p"])
            <= float(config.score_test_p_threshold)
        )
        ngram_selected = int(result["selected_ngram_count"]) > 0
        topic_selected = topic_fdr or topic_familywise or topic_unadjusted
        term_group_selected = (
            term_group_fdr or term_group_familywise or term_group_unadjusted
        )
        if index not in selected_set:
            result["selection_reason"] = "not_selected"
        elif ngram_selected and topic_selected and term_group_selected:
            result["selection_reason"] = (
                "nmf_topic_term_group_and_ngram_score_evidence"
            )
        elif ngram_selected and (topic_selected or term_group_selected):
            result["selection_reason"] = "topic_or_term_group_and_ngram_score_evidence"
        elif topic_selected and term_group_selected:
            result["selection_reason"] = "nmf_topic_and_term_group_score_evidence"
        elif ngram_selected:
            result["selection_reason"] = "ngram_score_evidence"
        elif topic_selected:
            result["selection_reason"] = "nmf_topic_score_evidence"
        elif term_group_selected:
            result["selection_reason"] = "joint_ngram_group_score_evidence"
        else:
            result["selection_reason"] = "minimum_evidence_rank_fallback"
    return {
        "selected_topic_ids": [results[index]["topic_id"] for index in selected],
        "selection_count": len(selected),
        "selection_rule": (
            "nmf_topic_or_joint_ngram_group_or_individual_ngram_evidence_"
            "then_minimum_rank_fallback_with_maximum"
            if complete_bootstrap and complete_term_group_bootstrap
            else "complete_family_asymptotic_union_then_minimum_rank_fallback_with_maximum"
        ),
        "fdr_level": float(config.score_test_fdr_level),
        "p_threshold": float(config.score_test_p_threshold),
        "minimum_topics": int(minimum),
        "maximum_topics": int(maximum),
        "complete_family_multiplier_bootstrap": complete_bootstrap,
        "complete_term_group_multiplier_bootstrap": (
            complete_term_group_bootstrap
        ),
    }


def _select_orphan_clusters(
    results: List[Dict[str, Any]],
    config: TfidfTopicDiscoveryConfig,
    bootstrap: Mapping[str, Any],
) -> Dict[str, Any]:
    """Select a bounded orphan-cluster set from persisted score statistics.

    Cluster construction and every score statistic are upstream of this
    helper.  Keeping the policy application separate allows a minimum/maximum
    shortlist change to be applied without rereading held-out labels or
    repeating the multiplier bootstrap.
    """
    complete_bootstrap = bool(bootstrap.get("complete_term_group_family"))
    order = sorted(
        range(len(results)),
        key=lambda index: (
            int(results[index].get("quadratic_covariance_rank", 0)) <= 0,
            float(results[index].get("primary_p", 1.0)),
            -float(results[index].get("quadratic_statistic_per_rank", 0.0)),
            -float(
                results[index].get(
                    "maximum_absolute_standardized_score", 0.0
                )
            ),
            str(results[index]["cluster_id"]),
        ),
    )
    selected: List[int] = []
    maximum = min(int(config.orphan_ngram_max_selected_clusters), len(order))
    minimum = min(int(config.orphan_ngram_min_selected_clusters), maximum)
    for result in results:
        result["selection_reason"] = "not_selected"
        result["selected_for_agent"] = False
    for index in ([] if maximum == 0 else order):
        result = results[index]
        if int(result.get("quadratic_covariance_rank", 0)) <= 0:
            continue
        reason = None
        if float(result.get("fdr_q", 1.0)) <= float(
            config.orphan_ngram_fdr_level
        ):
            reason = "orphan_cluster_fdr"
        elif (
            complete_bootstrap
            and result.get("familywise_p") is not None
            and float(result["familywise_p"])
            <= float(config.orphan_ngram_p_threshold)
        ):
            reason = "orphan_cluster_familywise_p"
        elif (
            not complete_bootstrap
            and float(result.get("primary_p", 1.0))
            <= float(config.orphan_ngram_p_threshold)
        ):
            reason = "orphan_cluster_asymptotic_p"
        if reason is None:
            continue
        result["selection_reason"] = reason
        selected.append(index)
        if len(selected) >= maximum:
            break
    selected_set = set(selected)
    for index in order:
        if len(selected) >= minimum:
            break
        if (
            index not in selected_set
            and int(results[index].get("quadratic_covariance_rank", 0)) > 0
        ):
            results[index]["selection_reason"] = (
                "minimum_heldout_evidence_rank_power_safeguard"
            )
            selected.append(index)
            selected_set.add(index)
    selected = sorted(selected, key=order.index)[:maximum]
    selected_set = set(selected)
    for evidence_rank, index in enumerate(order, start=1):
        result = results[index]
        result["evidence_rank"] = int(evidence_rank)
        result["selected_for_agent"] = index in selected_set
        for private_key in (
            "_row_scores",
            "_inverse_covariance",
            "_column_scales",
            "_retained_term_positions",
            "_topic_bootstrap_rows",
        ):
            result.pop(private_key, None)
    selected_results = [results[index] for index in selected]
    return {
        "selected_clusters": selected_results,
        "selected_cluster_ids": [
            str(result["cluster_id"]) for result in selected_results
        ],
        "selection_count": len(selected_results),
        "selection_rule": (
            "heldout_quadratic_group_fdr_or_complete_familywise_p_then_"
            "bounded_minimum_evidence_rank_power_safeguard"
        ),
        "fdr_level": float(config.orphan_ngram_fdr_level),
        "p_threshold": float(config.orphan_ngram_p_threshold),
        "maximum_selected_clusters": maximum,
        "minimum_selected_clusters": minimum,
    }


def score_effect_orphan_ngram_clusters(
    *,
    fit_matrix: sparse.spmatrix,
    heldout_matrix: sparse.spmatrix,
    feature_names: Sequence[str],
    effect_scores: Any,
    effect_topics: Sequence[Mapping[str, Any]],
    fit_treatment: np.ndarray,
    fit_outcome: np.ndarray,
    heldout_treatment: np.ndarray,
    heldout_outcome: np.ndarray,
    fit_propensity: np.ndarray,
    fit_outcome_prediction: np.ndarray,
    heldout_propensity: np.ndarray,
    heldout_outcome_prediction: np.ndarray,
    dina_nuisance=None,
    config: TfidfTopicDiscoveryConfig,
) -> Dict[str, Any]:
    """Build fit-side orphan groups and score their complete held-out family."""
    if not bool(config.orphan_ngram_enabled):
        return {
            "status": "disabled",
            "uses_heldout_treatment_and_outcome": False,
            "clusters": [],
            "selected_clusters": [],
            "selected_cluster_ids": [],
            "selection_count": 0,
        }
    represented_terms = sorted(
        {
            str(term.get("term"))
            for topic in effect_topics
            for term in topic.get("terms", [])
            if str(term.get("term") or "").strip()
        }
    )
    universe = build_fit_side_orphan_ngram_clusters(
        fit_matrix=fit_matrix,
        feature_names=feature_names,
        effect_scores=effect_scores,
        represented_topic_terms=represented_terms,
        config=config,
    )
    clusters = list(universe.pop("clusters", []))
    contribution, weights, definition = _bank_contribution(
        dina_nuisance=dina_nuisance,
        bank="effect",
        fit_treatment=np.asarray(fit_treatment, dtype=float),
        fit_outcome=np.asarray(fit_outcome, dtype=float),
        heldout_treatment=np.asarray(heldout_treatment, dtype=float),
        heldout_outcome=np.asarray(heldout_outcome, dtype=float),
        fit_propensity=np.asarray(fit_propensity, dtype=float),
        fit_outcome_prediction=np.asarray(fit_outcome_prediction, dtype=float),
        heldout_propensity=np.asarray(heldout_propensity, dtype=float),
        heldout_outcome_prediction=np.asarray(heldout_outcome_prediction, dtype=float),
    )
    vocabulary = {str(name): index for index, name in enumerate(feature_names)}
    results = [
        _single_orphan_cluster_score(
            cluster=cluster,
            fit_matrix=fit_matrix,
            heldout_matrix=heldout_matrix,
            vocabulary=vocabulary,
            contribution=contribution,
            centering_weights=weights,
        )
        for cluster in clusters
    ]
    bootstrap = _bootstrap_candidates(
        results,
        repeats=int(config.score_test_bootstrap_repeats),
        top_topics=int(config.score_test_bootstrap_top_topics),
        chunk_size=int(config.score_test_bootstrap_chunk_size),
        seed=int(config.random_state) + 1907,
    )
    complete_bootstrap = bool(bootstrap.get("complete_term_group_family"))
    for result in results:
        if complete_bootstrap and "quadratic_multiplier_p" in result:
            result["primary_p"] = float(result["quadratic_multiplier_p"])
            result["primary_p_source"] = (
                "complete_orphan_cluster_multiplier_bootstrap"
            )
        else:
            result["primary_p"] = float(result.get("asymptotic_p", 1.0))
            result["primary_p_source"] = (
                "asymptotic_complete_orphan_cluster_family"
            )
        result["familywise_p"] = (
            float(result["candidate_familywise_energy_p"])
            if "candidate_familywise_energy_p" in result
            else None
        )
    adjusted = benjamini_hochberg(
        [float(result.get("primary_p", 1.0)) for result in results]
    )
    for result, q_value in zip(results, adjusted):
        result["fdr_q"] = float(q_value)
    selection = _select_orphan_clusters(results, config, bootstrap)
    return {
        "status": "completed",
        "uses_heldout_treatment_and_outcome": True,
        "fits_patient_level_cate_model": False,
        "candidate_definition": (
            "eligible_stable_fit_effect_ngrams_with_abs_score_at_least_"
            f"{float(config.orphan_ngram_min_abs_fit_score):g}_excluding_all_"
            "fitted_effect_topic_top_terms"
        ),
        "topic_term_exclusion_is_fit_side": True,
        "cluster_construction_uses_heldout_rows_or_labels": False,
        "target_definition": definition,
        "represented_topic_terms": represented_terms,
        **universe,
        "cluster_count": len(results),
        "clusters": results,
        **selection,
        "bootstrap_calibration": bootstrap,
    }


def score_topic_banks(
    *,
    fit_matrix: sparse.spmatrix,
    heldout_matrix: sparse.spmatrix,
    feature_names: Sequence[str],
    topic_banks: Mapping[str, Mapping[str, Any]],
    fit_topic_values: Mapping[str, np.ndarray],
    heldout_topic_values: Mapping[str, np.ndarray],
    fit_treatment: np.ndarray,
    fit_outcome: np.ndarray,
    heldout_treatment: np.ndarray,
    heldout_outcome: np.ndarray,
    fit_propensity: np.ndarray,
    fit_outcome_prediction: np.ndarray,
    heldout_propensity: np.ndarray,
    heldout_outcome_prediction: np.ndarray,
    dina_nuisance=None,
    config: TfidfTopicDiscoveryConfig,
    scope_id: str,
    raw_ngram_scores: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Score and select every fitted topic using exact held-out rows."""
    fit_matrix = sparse.csr_matrix(fit_matrix)
    heldout_matrix = sparse.csr_matrix(heldout_matrix)
    if fit_matrix.shape[1] != heldout_matrix.shape[1]:
        raise ValueError("Fit and held-out TF-IDF matrices have different vocabularies")
    if heldout_matrix.shape[0] != len(heldout_treatment):
        raise ValueError("Held-out score rows do not align with held-out labels")
    vocabulary = {str(name): index for index, name in enumerate(feature_names)}
    output: Dict[str, Any] = {
        "schema_version": TOPIC_SCORE_TEST_SCHEMA_VERSION,
        "scope_id": str(scope_id),
        "terms_per_topic": int(config.terms_per_topic),
        "fit_n": int(fit_matrix.shape[0]),
        "heldout_n": int(heldout_matrix.shape[0]),
        "uses_heldout_treatment_and_outcome": True,
        "fits_patient_level_cate_model": False,
        "constructs_divided_pseudo_target": False,
        "banks": {},
    }
    for bank_index, bank in enumerate(_BANKS):
        topics = list((topic_banks.get(bank) or {}).get("topics") or [])
        bank_fit_topics = np.asarray(
            fit_topic_values.get(bank, np.zeros((fit_matrix.shape[0], 0))),
            dtype=float,
        )
        bank_heldout_topics = np.asarray(
            heldout_topic_values.get(
                bank, np.zeros((heldout_matrix.shape[0], 0))
            ),
            dtype=float,
        )
        if bank_fit_topics.ndim == 1:
            bank_fit_topics = bank_fit_topics[:, None]
        if bank_heldout_topics.ndim == 1:
            bank_heldout_topics = bank_heldout_topics[:, None]
        expected_fit_shape = (fit_matrix.shape[0], len(topics))
        expected_heldout_shape = (heldout_matrix.shape[0], len(topics))
        if bank_fit_topics.shape != expected_fit_shape:
            raise ValueError(
                f"{bank} fit topic scores have shape {bank_fit_topics.shape}; "
                f"expected {expected_fit_shape}"
            )
        if bank_heldout_topics.shape != expected_heldout_shape:
            raise ValueError(
                f"{bank} held-out topic scores have shape "
                f"{bank_heldout_topics.shape}; expected {expected_heldout_shape}"
            )
        contribution, weights, definition = _bank_contribution(
            dina_nuisance=dina_nuisance,
            bank=bank,
            fit_treatment=np.asarray(fit_treatment, dtype=float),
            fit_outcome=np.asarray(fit_outcome, dtype=float),
            heldout_treatment=np.asarray(heldout_treatment, dtype=float),
            heldout_outcome=np.asarray(heldout_outcome, dtype=float),
            fit_propensity=np.asarray(fit_propensity, dtype=float),
            fit_outcome_prediction=np.asarray(fit_outcome_prediction, dtype=float),
            heldout_propensity=np.asarray(heldout_propensity, dtype=float),
            heldout_outcome_prediction=np.asarray(heldout_outcome_prediction, dtype=float),
        )
        results = [
            _single_topic_score(
                topic=topic,
                fit_topic_values=bank_fit_topics[:, topic_index],
                heldout_topic_values=bank_heldout_topics[:, topic_index],
                fit_matrix=fit_matrix,
                heldout_matrix=heldout_matrix,
                vocabulary=vocabulary,
                contribution=contribution,
                centering_weights=weights,
                terms_per_topic=int(config.terms_per_topic),
            )
            for topic_index, topic in enumerate(topics)
        ]
        bootstrap = _bootstrap_candidates(
            results,
            repeats=int(config.score_test_bootstrap_repeats),
            top_topics=int(config.score_test_bootstrap_top_topics),
            chunk_size=int(config.score_test_bootstrap_chunk_size),
            seed=int(config.random_state) + 701 + 101 * bank_index,
        )
        ngram_selection = _select_ngrams(results, config, bootstrap)
        selection = _select_topics(results, config, bootstrap)
        for result in results:
            for private_key in [
                "_row_scores",
                "_inverse_covariance",
                "_column_scales",
                "_retained_term_positions",
                "_topic_bootstrap_rows",
            ]:
                result.pop(private_key, None)
        output["banks"][bank] = {
            "bank": bank,
            "target_definition": definition,
            "contribution_mean": float(np.mean(contribution)),
            "contribution_standard_deviation": float(
                np.std(contribution, ddof=1)
            ),
            "topic_tests": results,
            "bootstrap_calibration": bootstrap,
            **ngram_selection,
            **selection,
        }
    effect_score_frame = (raw_ngram_scores or {}).get("effect")
    if effect_score_frame is None:
        output["effect_orphan_ngram_branch"] = {
            "status": "not_available",
            "reason": "raw_effect_ngram_scores_not_supplied",
            "uses_heldout_treatment_and_outcome": False,
            "clusters": [],
            "selected_clusters": [],
            "selected_cluster_ids": [],
            "selection_count": 0,
        }
    else:
        output["effect_orphan_ngram_branch"] = score_effect_orphan_ngram_clusters(
            dina_nuisance=dina_nuisance,
            fit_matrix=fit_matrix,
            heldout_matrix=heldout_matrix,
            feature_names=feature_names,
            effect_scores=effect_score_frame,
            effect_topics=list((topic_banks.get("effect") or {}).get("topics") or []),
            fit_treatment=np.asarray(fit_treatment, dtype=float),
            fit_outcome=np.asarray(fit_outcome, dtype=float),
            heldout_treatment=np.asarray(heldout_treatment, dtype=float),
            heldout_outcome=np.asarray(heldout_outcome, dtype=float),
            fit_propensity=np.asarray(fit_propensity, dtype=float),
            fit_outcome_prediction=np.asarray(fit_outcome_prediction, dtype=float),
            heldout_propensity=np.asarray(heldout_propensity, dtype=float),
            heldout_outcome_prediction=np.asarray(heldout_outcome_prediction, dtype=float),
            config=config,
        )
    return output


def topic_score_test_by_id(
    score_tests: Mapping[str, Any],
    bank: str,
) -> Dict[str, Dict[str, Any]]:
    """Index one bank's persisted topic tests by topic id."""
    rows = (
        ((score_tests.get("banks") or {}).get(bank) or {}).get("topic_tests")
        or []
    )
    return {str(row["topic_id"]): dict(row) for row in rows}


def reselect_persisted_topic_scores(
    score_tests: Mapping[str, Any],
    config: TfidfTopicDiscoveryConfig,
) -> Dict[str, Any]:
    """Apply current selection policy to already-calibrated score statistics.

    This never recomputes a statistic or reads a label.  It is intended for a
    schema migration in which the persisted scalar-topic, joint-term-group,
    and individual-ngram families are unchanged but their selection union has
    been corrected.
    """
    output = copy.deepcopy(dict(score_tests))
    banks = output.get("banks") or {}
    for bank in _BANKS:
        bank_payload = banks.get(bank)
        if not isinstance(bank_payload, dict):
            raise ValueError(f"Persisted score payload is missing bank {bank!r}")
        results = list(bank_payload.get("topic_tests") or [])
        bootstrap = bank_payload.get("bootstrap_calibration") or {}
        bank_payload.update(_select_topics(results, config, bootstrap))
        bank_payload["topic_tests"] = results
    orphan = output.get("effect_orphan_ngram_branch")
    if isinstance(orphan, dict) and orphan.get("status") == "completed":
        clusters = list(orphan.get("clusters") or [])
        orphan.update(
            _select_orphan_clusters(
                clusters,
                config,
                orphan.get("bootstrap_calibration") or {},
            )
        )
        orphan["clusters"] = clusters
        orphan["selection_recomputed_without_labels"] = True
        orphan["score_statistics_recomputed"] = False
    output["schema_version"] = TOPIC_SCORE_TEST_SCHEMA_VERSION
    output["selection_recomputed_without_labels"] = True
    output["score_statistics_recomputed"] = False
    return output
