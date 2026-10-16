"""Neural-query Stage 1 configuration and training-only retrieved evidence.

Three ungated query banks are learned elsewhere: treatment and outcome queries
optimize direct patient-level target contrasts, and effect queries optimize the
constant-effect-orthogonalized cohort contrast.  This module holds their
configuration and builds the retrieved-chunk evidence handed to Stage 2.

No oracle column is accepted by any public discovery interface in this file.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .neural_cohort_witness import pad_chunk_embeddings
from .neural_query_evidence_contract import (
    NEURAL_QUERY_RANKING_RULE,
    NEURAL_QUERY_RANKING_VERSION,
    query_retrieval_policy,
    query_term_normalization,
)


@dataclass(frozen=True)
class NeuralQueryAgenticForestConfig:
    """Configuration for three small, independently sized semantic banks."""

    treatment_query_count: int = 5
    outcome_query_count: int = 5
    effect_query_count: int = 5
    query_inner_folds: int = 5
    initial_pool_size: int = 24
    query_epochs: int = 120
    final_refit_epochs: int = 80
    learning_rate: float = 0.025
    temperature: float = 0.05
    max_query_drift: float = 0.35
    final_refit_max_query_drift: float = 0.20
    kmeans_iterations: int = 20
    kmeans_sample_chunks: int = 8000
    query_diversity_weight: float = 0.08
    activation_diversity_weight: float = 0.04
    anchor_weight: float = 0.15
    min_activation_sd: float = 0.01
    activation_scale_weight: float = 1.0
    initialization_max_cosine: float = 0.96
    witness_epsilon: float = 1e-6
    optimizer_beta1: float = 0.9
    optimizer_beta2: float = 0.999
    optimizer_epsilon: float = 1e-8
    optimizer_weight_decay: float = 0.0
    optimizer_amsgrad: bool = False
    optimizer_maximize: bool = False
    optimizer_foreach: bool = False
    optimizer_capturable: bool = False
    optimizer_differentiable: bool = False
    optimizer_fused: bool = False
    gradient_clip_norm: float = 5.0
    consensus_kmeans_init: str = "k-means++"
    consensus_kmeans_n_init: int = 20
    consensus_kmeans_max_iter: int = 300
    consensus_kmeans_tolerance: float = 1e-4
    consensus_kmeans_copy_x: bool = True
    consensus_kmeans_algorithm: str = "lloyd"
    retrieval_patient_batch_size: int = 128
    evidence_patient_batch_size: int = 96
    evidence_top_patients: int = 10
    # Scientific retrieval selection, distinct from capacity/length ceilings.
    # Preserve the complete text of each selected chunk, not every chunk of a
    # patient whose best chunk matched. Both contrastive arms use this rule.
    evidence_retrieval_top_k: int = 1
    # ``None`` is an explicit lossless allocation: retain every remaining
    # patient/positive term. A finite value is an acceptance ceiling and is
    # never permission for silent selection.
    evidence_background_patients: Optional[int] = 40
    evidence_top_ngrams: Optional[int] = 20
    evidence_excerpt_chars: Optional[int] = None
    evidence_chunks_per_patient_per_query: Optional[int] = None
    evidence_ngram_analyzer: str = "word"
    evidence_ngram_range_min: int = 1
    evidence_ngram_range_max: int = 3
    evidence_ngram_min_df: int | float = 1
    evidence_ngram_max_df: int | float = 1.0
    evidence_ngram_vocabulary_max_features: Optional[int] = None
    evidence_ngram_strip_accents: Optional[str] = None
    evidence_ngram_lowercase: bool = True
    evidence_ngram_stop_words: Optional[str | Sequence[str]] = "english"
    evidence_ngram_token_pattern: Optional[str] = r"(?u)\b\w\w+\b"
    evidence_ngram_binary: bool = False
    evidence_ngram_norm: Optional[str] = "l2"
    evidence_ngram_use_idf: bool = True
    evidence_ngram_smooth_idf: bool = True
    evidence_ngram_sublinear_tf: bool = True
    evidence_ngram_dtype: str = "float64"

    def validate(self) -> None:
        counts = (
            self.treatment_query_count,
            self.outcome_query_count,
            self.effect_query_count,
        )
        if any(int(value) < 1 for value in counts):
            raise ValueError("every query-bank count must be positive")
        if int(self.query_inner_folds) < 2:
            raise ValueError("query_inner_folds must be at least 2")
        if int(self.initial_pool_size) < max(map(int, counts)):
            raise ValueError("initial_pool_size must cover the largest query bank")
        if int(self.query_epochs) < 1 or int(self.final_refit_epochs) < 1:
            raise ValueError("query epoch counts must be positive")
        if not 0.0 < float(self.temperature) <= 1.0:
            raise ValueError("temperature must be in (0, 1]")
        for field_name in (
            "learning_rate",
            "query_diversity_weight",
            "activation_diversity_weight",
            "anchor_weight",
            "min_activation_sd",
            "activation_scale_weight",
            "initialization_max_cosine",
            "witness_epsilon",
            "optimizer_beta1",
            "optimizer_beta2",
            "optimizer_epsilon",
            "optimizer_weight_decay",
            "gradient_clip_norm",
            "consensus_kmeans_tolerance",
        ):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(float(value))
            ):
                raise ValueError(f"{field_name} must be finite")
        if float(self.learning_rate) <= 0.0:
            raise ValueError("learning_rate must be positive")
        for field_name in (
            "query_diversity_weight",
            "activation_diversity_weight",
            "anchor_weight",
            "min_activation_sd",
            "activation_scale_weight",
            "optimizer_weight_decay",
            "consensus_kmeans_tolerance",
        ):
            if float(getattr(self, field_name)) < 0.0:
                raise ValueError(f"{field_name} must be nonnegative")
        if not 0.0 <= float(self.initialization_max_cosine) <= 1.0:
            raise ValueError("initialization_max_cosine must be in [0, 1]")
        if float(self.witness_epsilon) <= 0.0:
            raise ValueError("witness_epsilon must be positive")
        if not 0.0 <= float(self.optimizer_beta1) < 1.0:
            raise ValueError("optimizer_beta1 must be in [0, 1)")
        if not 0.0 <= float(self.optimizer_beta2) < 1.0:
            raise ValueError("optimizer_beta2 must be in [0, 1)")
        if float(self.optimizer_epsilon) <= 0.0:
            raise ValueError("optimizer_epsilon must be positive")
        if float(self.gradient_clip_norm) <= 0.0:
            raise ValueError("gradient_clip_norm must be positive")
        for field_name in (
            "optimizer_amsgrad",
            "optimizer_maximize",
            "optimizer_foreach",
            "optimizer_capturable",
            "optimizer_differentiable",
            "optimizer_fused",
            "consensus_kmeans_copy_x",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be boolean")
        if self.optimizer_foreach and self.optimizer_fused:
            raise ValueError("optimizer_foreach and optimizer_fused cannot both be true")
        for field_name in (
            "consensus_kmeans_n_init",
            "consensus_kmeans_max_iter",
            "retrieval_patient_batch_size",
            "evidence_patient_batch_size",
            "evidence_retrieval_top_k",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field_name} must be a positive integer")
        if self.consensus_kmeans_init not in {"k-means++", "random"}:
            raise ValueError("consensus_kmeans_init must be k-means++ or random")
        if self.consensus_kmeans_algorithm not in {"lloyd", "elkan"}:
            raise ValueError("consensus_kmeans_algorithm must be lloyd or elkan")
        if int(self.evidence_top_patients) < 1:
            raise ValueError("evidence_top_patients must be positive")
        if (
            self.evidence_background_patients is not None
            and int(self.evidence_background_patients) < 0
        ):
            raise ValueError(
                "evidence_background_patients must be null or nonnegative"
            )
        if (
            self.evidence_top_ngrams is not None
            and int(self.evidence_top_ngrams) < 1
        ):
            raise ValueError("evidence_top_ngrams must be null or positive")
        if (
            self.evidence_excerpt_chars is not None
            and int(self.evidence_excerpt_chars) < 1
        ):
            raise ValueError("evidence_excerpt_chars must be null or positive")
        if (
            self.evidence_chunks_per_patient_per_query is not None
            and int(self.evidence_chunks_per_patient_per_query) < 1
        ):
            raise ValueError(
                "evidence_chunks_per_patient_per_query must be null or positive"
            )
        if not 1 <= int(self.evidence_ngram_range_min) <= int(
            self.evidence_ngram_range_max
        ):
            raise ValueError("evidence n-gram range must be positive and ordered")
        if self.evidence_ngram_analyzer not in {"word", "char", "char_wb"}:
            raise ValueError(
                "evidence_ngram_analyzer must be word, char, or char_wb"
            )
        for field_name in ("evidence_ngram_min_df", "evidence_ngram_max_df"):
            value = getattr(self, field_name)
            if isinstance(value, bool):
                raise TypeError(f"{field_name} must be an integer count or fraction")
            if isinstance(value, int):
                if value < 1:
                    raise ValueError(f"{field_name} integer count must be positive")
            elif isinstance(value, float):
                if not 0.0 < value <= 1.0:
                    raise ValueError(f"{field_name} fraction must be in (0, 1]")
            else:
                raise TypeError(f"{field_name} must be an integer count or fraction")
        if (
            self.evidence_ngram_vocabulary_max_features is not None
            and int(self.evidence_ngram_vocabulary_max_features) < 1
        ):
            raise ValueError(
                "evidence_ngram_vocabulary_max_features must be null or positive"
            )
        if self.evidence_ngram_strip_accents not in {None, "ascii", "unicode"}:
            raise ValueError(
                "evidence_ngram_strip_accents must be null, ascii, or unicode"
            )
        stop_words = self.evidence_ngram_stop_words
        if isinstance(stop_words, str):
            if stop_words != "english":
                raise ValueError(
                    "string evidence_ngram_stop_words must be the closed value english"
                )
        elif stop_words is not None:
            if (
                not isinstance(stop_words, (list, tuple))
                or not stop_words
                or any(not str(value).strip() for value in stop_words)
                or len({str(value) for value in stop_words}) != len(stop_words)
            ):
                raise ValueError(
                    "evidence_ngram_stop_words must be null, english, or a "
                    "nonempty unique string list"
                )
        if self.evidence_ngram_analyzer == "word":
            if not str(self.evidence_ngram_token_pattern or ""):
                raise ValueError(
                    "word evidence_ngram_token_pattern must be non-empty"
                )
        elif (
            self.evidence_ngram_stop_words is not None
            or self.evidence_ngram_token_pattern is not None
        ):
            raise ValueError(
                "character analyzers require null stop_words and token_pattern"
            )
        if self.evidence_ngram_norm not in {None, "l1", "l2"}:
            raise ValueError("evidence_ngram_norm must be null, l1, or l2")
        if self.evidence_ngram_dtype not in {"float32", "float64"}:
            raise ValueError("evidence_ngram_dtype must be float32 or float64")
        for field_name in (
            "evidence_ngram_lowercase",
            "evidence_ngram_binary",
            "evidence_ngram_use_idf",
            "evidence_ngram_smooth_idf",
            "evidence_ngram_sublinear_tf",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be boolean")

    def query_count(self, bank: str) -> int:
        mapping = {
            "treatment": int(self.treatment_query_count),
            "outcome": int(self.outcome_query_count),
            "effect": int(self.effect_query_count),
        }
        if bank not in mapping:
            raise ValueError(f"unknown query bank: {bank!r}")
        return mapping[bank]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def mechanical_role_for_bank(bank: str) -> str:
    if bank in {"treatment", "outcome"}:
        return "confounder"
    if bank == "effect":
        return "effect_modifier"
    raise ValueError(f"unknown query bank: {bank!r}")


def query_patient_top_chunks(
    chunk_matrices: Sequence[np.ndarray],
    queries: np.ndarray,
    *,
    top_k: int,
    device: str,
    patient_batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return each query's highest-cosine chunk indices and similarities."""

    import torch

    if int(top_k) < 1:
        raise ValueError("top_k must be positive")
    if (
        isinstance(patient_batch_size, bool)
        or not isinstance(patient_batch_size, int)
        or patient_batch_size < 1
    ):
        raise ValueError("patient_batch_size must be a positive integer")
    padded, mask = pad_chunk_embeddings(chunk_matrices)
    query_array = np.asarray(queries, dtype=np.float32)
    if query_array.ndim != 2 or query_array.shape[1] != padded.shape[2]:
        raise ValueError("queries must have shape (queries, embedding_dim)")
    query_array /= np.maximum(
        np.linalg.norm(query_array, axis=1, keepdims=True), 1e-12
    )
    maximum_k = min(int(top_k), int(padded.shape[1]))
    score_blocks: List[np.ndarray] = []
    index_blocks: List[np.ndarray] = []
    query_tensor = torch.as_tensor(query_array, device=device)
    with torch.no_grad():
        for start in range(0, len(padded), int(patient_batch_size)):
            stop = min(len(padded), start + int(patient_batch_size))
            chunks = torch.as_tensor(padded[start:stop], device=device)
            valid = torch.as_tensor(mask[start:stop], device=device)
            scores = torch.einsum("bcd,qd->bcq", chunks, query_tensor)
            scores = torch.where(
                valid[:, :, None], scores, torch.full_like(scores, -torch.inf)
            )
            top_scores, top_indices = torch.topk(
                scores, k=maximum_k, dim=1, largest=True, sorted=True
            )
            # patient x query x rank is easier for downstream provenance.
            score_blocks.append(
                top_scores.permute(0, 2, 1).detach().cpu().numpy().astype(np.float32)
            )
            index_blocks.append(
                top_indices.permute(0, 2, 1).detach().cpu().numpy().astype(np.int32)
            )
    return np.vstack(score_blocks), np.vstack(index_blocks)


class NeuralQueryEvidenceVocabularyOverflowError(RuntimeError):
    """A configured vocabulary ceiling was exceeded before witness selection."""


class NeuralQueryEvidenceCapacityOverflowError(RuntimeError):
    """A configured evidence allocation cannot hold the complete result."""


def _contrastive_ngrams(
    foreground: Sequence[str],
    background: Sequence[str],
    *,
    limit: Optional[int],
    config: NeuralQueryAgenticForestConfig,
) -> List[Dict[str, Any]]:
    documents = [str(value) for value in foreground] + [str(value) for value in background]
    if not foreground or not documents:
        return []
    try:
        vectorizer = TfidfVectorizer(
            analyzer=str(config.evidence_ngram_analyzer),
            strip_accents=config.evidence_ngram_strip_accents,
            lowercase=bool(config.evidence_ngram_lowercase),
            stop_words=config.evidence_ngram_stop_words,
            token_pattern=config.evidence_ngram_token_pattern,
            ngram_range=(
                int(config.evidence_ngram_range_min),
                int(config.evidence_ngram_range_max),
            ),
            binary=bool(config.evidence_ngram_binary),
            min_df=config.evidence_ngram_min_df,
            max_df=config.evidence_ngram_max_df,
            # A finite value is an acceptance ceiling, not a lossy sklearn
            # selector. Fit the complete eligible vocabulary, then fail closed
            # below if it exceeds the configured ceiling.
            max_features=None,
            norm=config.evidence_ngram_norm,
            use_idf=bool(config.evidence_ngram_use_idf),
            smooth_idf=bool(config.evidence_ngram_smooth_idf),
            sublinear_tf=bool(config.evidence_ngram_sublinear_tf),
            dtype=(
                np.float32
                if config.evidence_ngram_dtype == "float32"
                else np.float64
            ),
        )
        matrix = vectorizer.fit_transform(documents)
    except ValueError as exc:
        message = str(exc).lower()
        if "empty vocabulary" in message or "after pruning, no terms remain" in message:
            return []
        raise
    foreground_mean = np.asarray(matrix[: len(foreground)].mean(axis=0)).ravel()
    background_mean = (
        np.asarray(matrix[len(foreground) :].mean(axis=0)).ravel()
        if background
        else np.zeros_like(foreground_mean)
    )
    contrast = foreground_mean - background_mean
    names = vectorizer.get_feature_names_out()
    vocabulary_cap = config.evidence_ngram_vocabulary_max_features
    if vocabulary_cap is not None and len(names) > int(vocabulary_cap):
        raise NeuralQueryEvidenceVocabularyOverflowError(
            "complete eligible neural-query evidence vocabulary contains "
            f"{len(names)} terms, exceeding configured acceptance ceiling "
            f"{int(vocabulary_cap)}; no terms were silently discarded"
        )
    ranked = [
        {"term": str(names[int(index)]), "tfidf_contrast": float(contrast[int(index)])}
        for index in np.argsort(contrast)[::-1]
        if float(contrast[int(index)]) > 0.0
    ]
    if limit is not None and len(ranked) > int(limit):
        raise NeuralQueryEvidenceCapacityOverflowError(
            "complete positive neural-query contrastive evidence contains "
            f"{len(ranked)} terms, exceeding configured evidence_top_ngrams "
            f"allocation {int(limit)}; no terms were silently discarded"
        )
    return ranked


def _query_ranking_payload(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a query's consensus rank, both ranking scores, and held-out p-values."""

    members = [
        member for member in record.get("members") or [] if isinstance(member, Mapping)
    ]
    return {
        "rank": record.get("rank"),
        "ranking": {
            "version": NEURAL_QUERY_RANKING_VERSION,
            "rule": NEURAL_QUERY_RANKING_RULE,
            "score_source": record.get("ranking_score_source"),
        },
        "median_abs_heldout_z": record.get("median_abs_heldout_z"),
        "median_abs_train_score": record.get("median_abs_train_score"),
        "median_heldout_p_value": record.get("median_heldout_p_value"),
        "heldout_member_count": record.get("heldout_member_count"),
        "member_heldout_statistics": [
            {
                "candidate_id": member.get("candidate_id"),
                "subfold": member.get("subfold"),
                "heldout_z": member.get("validation_audit_standardized_score"),
                "heldout_p_value": member.get("validation_audit_two_sided_p_value"),
                "train_score": member.get("train_standardized_score"),
            }
            for member in members
        ],
    }


def build_query_evidence(
    *,
    bank: str,
    queries: np.ndarray,
    query_records: Sequence[Mapping[str, Any]],
    row_ids: Sequence[int],
    chunk_matrices: Sequence[np.ndarray],
    all_chunk_texts: Sequence[Sequence[str]],
    config: NeuralQueryAgenticForestConfig,
    device: str,
    seed: int,
) -> List[Dict[str, Any]]:
    """Build training-only retrieved evidence for every final query."""

    config.validate()
    if len(queries) != len(query_records):
        raise ValueError("one query record is required per query")
    if len(row_ids) != len(chunk_matrices):
        raise ValueError("row_ids and chunk_matrices must have equal lengths")
    if any(not 0 <= int(row_id) < len(all_chunk_texts) for row_id in row_ids):
        raise IndexError("an evidence row id is outside the chunk-text corpus")
    chunk_counts = [int(np.asarray(matrix).shape[0]) for matrix in chunk_matrices]
    chunk_texts_by_position = [
        all_chunk_texts[int(row_id)] for row_id in row_ids
    ]
    if any(
        len(texts) != count
        for texts, count in zip(chunk_texts_by_position, chunk_counts)
    ):
        raise ValueError(
            "neural-query embedding/text cache chunk counts are inconsistent"
        )
    maximum_chunks = max(max(chunk_counts), 1)
    retrieval_chunks = min(int(config.evidence_retrieval_top_k), maximum_chunks)
    configured_chunks = config.evidence_chunks_per_patient_per_query
    if configured_chunks is not None and retrieval_chunks > int(configured_chunks):
        raise NeuralQueryEvidenceCapacityOverflowError(
            "selected neural-query evidence requires "
            f"{retrieval_chunks} chunks for at least one fit row, exceeding "
            "configured evidence_chunks_per_patient_per_query allocation "
            f"{int(configured_chunks)}; no chunks were silently discarded"
        )
    scores, indices = query_patient_top_chunks(
        chunk_matrices,
        queries,
        top_k=retrieval_chunks,
        device=device,
        patient_batch_size=int(config.evidence_patient_batch_size),
    )
    rng = np.random.default_rng(int(seed))
    output: List[Dict[str, Any]] = []
    for query_index, record in enumerate(query_records):
        order = np.argsort(scores[:, query_index, 0])[::-1]
        selected_positions = order[: int(config.evidence_top_patients)]
        remaining = order[int(config.evidence_top_patients) :]
        if config.evidence_background_patients is None:
            # Complete-background mode preserves all fit patients that are
            # not in the configured foreground.
            background_positions = remaining
        else:
            background_count = min(
                int(config.evidence_background_patients), len(remaining)
            )
            background_positions = (
                rng.choice(remaining, size=background_count, replace=False)
                if background_count
                else np.empty(0, dtype=int)
            )

        def item(position: int, rank: int) -> Dict[str, Any]:
            row_id = int(row_ids[int(position)])
            chunk_index = int(indices[int(position), query_index, int(rank)])
            texts = chunk_texts_by_position[int(position)]
            text = str(texts[chunk_index])
            excerpt_limit = config.evidence_excerpt_chars
            if excerpt_limit is not None and len(text) > int(excerpt_limit):
                raise ValueError(
                    "neural-query evidence chunk exceeds configured "
                    "evidence_excerpt_chars; refusing silent text truncation"
                )
            return {
                "evidence_id": (
                    f"{record['query_id']}__row_{row_id:05d}__chunk_{chunk_index:03d}"
                ),
                "_oci_row_id": row_id,
                "chunk_index": chunk_index,
                "similarity": float(scores[int(position), query_index, int(rank)]),
                "text": text,
            }

        def patient_items(position: int) -> List[Dict[str, Any]]:
            return [
                item(int(position), rank)
                for rank in range(
                    min(int(chunk_counts[int(position)]), int(scores.shape[2]))
                )
            ]

        foreground_items = [
            item
            for position in selected_positions
            for item in patient_items(int(position))
        ]
        background_items = [
            item
            for position in background_positions
            for item in patient_items(int(position))
        ]
        output.append(
            {
                "query_id": str(record["query_id"]),
                "bank": str(bank),
                "retrieval_policy": query_retrieval_policy(
                    int(config.evidence_retrieval_top_k)
                ),
                "mechanical_role": mechanical_role_for_bank(bank),
                "statistical_gate_applied": False,
                "member_count": int(record.get("member_count", 0)),
                "member_subfolds": list(record.get("member_subfolds") or []),
                "fit_standardized_score": record.get("fit_standardized_score"),
                **_query_ranking_payload(record),
                "top_chunks": foreground_items,
                "term_normalization": query_term_normalization(config),
                "top_contrastive_ngrams": _contrastive_ngrams(
                    [item["text"] for item in foreground_items],
                    [item["text"] for item in background_items],
                    limit=config.evidence_top_ngrams,
                    config=config,
                ),
            }
        )
    return output
