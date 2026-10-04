"""Ungated neural-query evidence for bounded structured-feature discovery.

The module deliberately separates three concerns:

* treatment and outcome queries optimize direct patient-level target contrasts;
* effect queries optimize the constant-effect-orthogonalized cohort contrast;
* every final query is shown to the feature agent, while feature-count limits
  are enforced only on executable contracts and extraction requests.

No oracle column is accepted by any public discovery interface in this file.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from ..config import ExplicitFeatureSpec
from .neural_cohort_witness import pad_chunk_embeddings
from .neural_query_evidence_contract import query_retrieval_policy, query_term_normalization


QUERY_FEATURE_PROMPT_VERSION = "neural_query_feature_v1"
QUERY_REGISTRY_PROMPT_VERSION = "neural_query_registry_v1"
QUERY_REVIEW_PROMPT_VERSION = "neural_query_review_v1"
QUERY_RAG_TEXT_VERSION = "neural_query_rag_v1"

_FIELD_CUE_CLAUSE_SPLIT = re.compile(r"[\n\r,;|]+")
_FIELD_CUE_NUMBER = re.compile(
    r"(?<![A-Za-z0-9_])(?:[<>]=?|[~≈])?\s*[-+]?(?:\d+(?:\.\d+)?|\.\d+)"
)
_FIELD_CUE_WORD = re.compile(r"[A-Za-z][A-Za-z0-9+/%().-]*")
_FIELD_CUE_TRAILING_CONNECTORS = frozenset(
    {"at", "are", "is", "of", "was", "were"}
)


class FrozenChunkEmbeddingCache:
    """Read-only, row-addressed access to a previously computed embedding cache.

    The cache is deliberately immutable in this workflow.  In particular, the
    query learner never invokes or loads the sentence encoder, which makes it
    straightforward to audit that outer-held-out text was transformed only by
    the already-frozen representation.
    """

    def __init__(self, path: Path | str, *, expected_rows: int) -> None:
        self.path = Path(path)
        metadata_path = self.path / "metadata.json"
        embeddings_path = self.path / "chunk_embeddings.npy"
        offsets_path = self.path / "offsets.npy"
        chunks_path = self.path / "chunk_texts.jsonl"
        for required in (metadata_path, embeddings_path, offsets_path, chunks_path):
            if not required.exists():
                raise FileNotFoundError(f"incomplete frozen embedding cache: {required}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.embeddings = np.load(embeddings_path, mmap_mode="r")
        self.offsets = np.load(offsets_path)
        if int(self.metadata.get("num_samples", -1)) != int(expected_rows):
            raise ValueError("embedding-cache row count does not match the dataset")
        if len(self.offsets) != int(expected_rows) + 1:
            raise ValueError("embedding-cache offsets do not match dataset rows")
        if int(self.offsets[-1]) != int(self.embeddings.shape[0]):
            raise ValueError("embedding-cache offsets do not span all embeddings")
        if int(self.metadata.get("hidden_size", -1)) != int(self.embeddings.shape[1]):
            raise ValueError("embedding-cache hidden size is inconsistent")
        self._chunks_path = chunks_path
        self._chunk_texts: Optional[List[List[str]]] = None

    def matrices(self, row_ids: Sequence[int]) -> List[np.ndarray]:
        output: List[np.ndarray] = []
        for raw_row_id in row_ids:
            row_id = int(raw_row_id)
            if not 0 <= row_id < len(self.offsets) - 1:
                raise IndexError(f"embedding-cache row id is out of range: {row_id}")
            start, stop = int(self.offsets[row_id]), int(self.offsets[row_id + 1])
            output.append(np.asarray(self.embeddings[start:stop], dtype=np.float32))
        return output

    def chunk_texts(self) -> List[List[str]]:
        if self._chunk_texts is None:
            rows: List[List[str]] = []
            with self._chunks_path.open(encoding="utf-8") as handle:
                for line in handle:
                    payload = json.loads(line)
                    rows.append([str(value) for value in payload.get("chunks", [])])
            if len(rows) != int(self.metadata["num_samples"]):
                raise ValueError("chunk-text row count does not match cache metadata")
            self._chunk_texts = rows
        return self._chunk_texts


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
    evidence_safe_term_max_tokens: int = 6
    evidence_safe_term_max_chars: int = 160
    rag_chunks_per_query: int = 1
    # These are allocation ceilings, not truncation controls. Portable
    # scientific profiles may configure ``None`` for no ceiling.
    rag_max_chunks_per_patient: Optional[int] = 18
    rag_excerpt_chars: Optional[int] = 1800
    max_features_per_query: int = 3
    max_raw_feature_candidates: int = 45
    max_canonical_features: int = 20
    max_review_rounds: int = 2
    max_review_additions_per_round: int = 4
    max_variables_per_extraction_request: int = 10

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
        if int(self.max_features_per_query) < 1:
            raise ValueError("max_features_per_query must be positive")
        maximum_possible = sum(map(int, counts)) * int(self.max_features_per_query)
        if int(self.max_raw_feature_candidates) > maximum_possible:
            raise ValueError(
                "max_raw_feature_candidates cannot exceed query_count * "
                "max_features_per_query"
            )
        if not 1 <= int(self.max_canonical_features) <= int(
            self.max_raw_feature_candidates
        ):
            raise ValueError("max_canonical_features must be in the raw-candidate range")
        if not 1 <= int(self.max_variables_per_extraction_request) <= 10:
            raise ValueError("max_variables_per_extraction_request must be in [1, 10]")
        if int(self.rag_chunks_per_query) < 1:
            raise ValueError("rag_chunks_per_query must be positive")
        if (
            self.rag_max_chunks_per_patient is not None
            and int(self.rag_max_chunks_per_patient) < 1
        ):
            raise ValueError(
                "rag_max_chunks_per_patient must be null or positive"
            )
        if self.rag_excerpt_chars is not None and int(self.rag_excerpt_chars) < 1:
            raise ValueError("rag_excerpt_chars must be null or positive")
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
        if int(self.evidence_safe_term_max_tokens) < 1:
            raise ValueError("evidence_safe_term_max_tokens must be positive")
        if int(self.evidence_safe_term_max_chars) < 1:
            raise ValueError("evidence_safe_term_max_chars must be positive")
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


def _literal_field_labels(text: str) -> List[str]:
    """Return short labels next to values without using a domain vocabulary."""

    labels: List[str] = []
    for clause in _FIELD_CUE_CLAUSE_SPLIT.split(str(text)):
        delimiter = re.search(r"[:=]", clause)
        if delimiter is not None:
            words = list(_FIELD_CUE_WORD.finditer(clause[: delimiter.start()]))
            if words:
                words = words[-4:]
                labels.append(clause[words[0].start() : words[-1].end()])

        cursor = 0
        for value_match in _FIELD_CUE_NUMBER.finditer(clause):
            prefix = clause[cursor : value_match.start()]
            words = list(_FIELD_CUE_WORD.finditer(prefix))
            while (
                words
                and words[-1].group(0).lower() in _FIELD_CUE_TRAILING_CONNECTORS
            ):
                words.pop()
            if words:
                words = words[-4:]
                label = prefix[words[0].start() : words[-1].end()].strip()
                if 1 < len(label) <= 80:
                    labels.append(label)
            cursor = value_match.end()
    return labels


def _literal_ngram_match(term: str, text: str) -> bool:
    tokens = re.findall(r"[A-Za-z0-9]+", str(term))
    if not tokens:
        return False
    pattern = r"(?<![A-Za-z0-9])" + r"\W+".join(
        re.escape(token) for token in tokens
    ) + r"(?![A-Za-z0-9])"
    return re.search(pattern, str(text), flags=re.IGNORECASE) is not None


def derive_evidence_field_cues(
    retrieved_training_chunks: Sequence[Mapping[str, Any]],
    contrastive_ngrams: Sequence[Mapping[str, Any]],
    *,
    limit: Optional[int],
) -> List[Dict[str, Any]]:
    """Derive bounded navigation cues only from supplied training evidence.

    Literal labels adjacent to values help expose multi-field records.  Ranked
    n-grams retain the broader semantic retrieval signal.  Both sources must be
    grounded in a supplied excerpt, and no clinical term inventory is used.
    """

    if limit is not None and int(limit) < 1:
        return []
    chunks = [dict(item) for item in retrieved_training_chunks]
    records: Dict[str, Dict[str, Any]] = {}
    first_seen = 0

    def add(cue: str, *, source: str, evidence_ids: Sequence[str]) -> None:
        nonlocal first_seen
        literal = re.sub(r"\s+", " ", str(cue)).strip(" \t:=-")
        if not literal or len(literal) > 80:
            return
        key = literal.casefold()
        if key not in records:
            records[key] = {
                "cue": literal,
                "source_kinds": [],
                "supporting_evidence_ids": [],
                "_first_seen": first_seen,
            }
            first_seen += 1
        record = records[key]
        if source not in record["source_kinds"]:
            record["source_kinds"].append(source)
        for evidence_id in evidence_ids:
            if evidence_id and evidence_id not in record["supporting_evidence_ids"]:
                record["supporting_evidence_ids"].append(evidence_id)

    for chunk in chunks:
        evidence_id = str(chunk.get("evidence_id") or "")
        for label in _literal_field_labels(str(chunk.get("text") or "")):
            add(
                label,
                source="literal_value_label",
                evidence_ids=[evidence_id],
            )

    for item in contrastive_ngrams:
        term = str(item.get("term") or "").strip()
        supporting_ids = [
            str(chunk.get("evidence_id") or "")
            for chunk in chunks
            if _literal_ngram_match(term, str(chunk.get("text") or ""))
        ]
        if supporting_ids:
            add(
                term,
                source="contrastive_ngram",
                evidence_ids=supporting_ids,
            )

    ranked = sorted(
        records.values(),
        key=lambda item: (
            "literal_value_label" not in item["source_kinds"],
            -len(item["supporting_evidence_ids"]),
            int(item["_first_seen"]),
        ),
    )
    if limit is not None and len(ranked) > int(limit):
        raise NeuralQueryEvidenceCapacityOverflowError(
            "complete neural-query field-cue evidence contains "
            f"{len(ranked)} cues, exceeding configured evidence_top_ngrams "
            f"allocation {int(limit)}; no cues were silently discarded"
        )
    return [
        {key: value for key, value in item.items() if not key.startswith("_")}
        for item in ranked
    ]


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


def build_query_rag_documents(
    *,
    row_ids: Sequence[int],
    chunk_matrices: Sequence[np.ndarray],
    all_chunk_texts: Sequence[Sequence[str]],
    queries: np.ndarray,
    query_ids: Sequence[str],
    query_banks: Sequence[str],
    config: NeuralQueryAgenticForestConfig,
    device: str,
) -> List[str]:
    """Create compact per-patient evidence documents for structured extraction."""

    if not (len(queries) == len(query_ids) == len(query_banks)):
        raise ValueError("query arrays, ids, and banks must have equal lengths")
    scores, indices = query_patient_top_chunks(
        chunk_matrices,
        queries,
        top_k=int(config.rag_chunks_per_query),
        device=device,
        patient_batch_size=int(config.evidence_patient_batch_size),
    )
    documents: List[str] = []
    for position, row_id_value in enumerate(row_ids):
        row_id = int(row_id_value)
        candidates: Dict[int, Dict[str, Any]] = {}
        for query_index, query_id in enumerate(query_ids):
            for rank in range(scores.shape[2]):
                chunk_index = int(indices[position, query_index, rank])
                candidate = {
                    "chunk_index": chunk_index,
                    "similarity": float(scores[position, query_index, rank]),
                    "query_id": str(query_id),
                    "bank": str(query_banks[query_index]),
                    "rank": int(rank + 1),
                }
                prior = candidates.get(chunk_index)
                if prior is None or candidate["similarity"] > prior["similarity"]:
                    candidates[chunk_index] = candidate
        ranked_candidates = sorted(
            candidates.values(), key=lambda item: item["similarity"], reverse=True
        )
        if (
            config.rag_max_chunks_per_patient is not None
            and len(ranked_candidates)
            > int(config.rag_max_chunks_per_patient)
        ):
            raise NeuralQueryEvidenceCapacityOverflowError(
                "complete query-RAG result contains "
                f"{len(ranked_candidates)} distinct retrieved chunks for row "
                f"{row_id}, exceeding configured rag_max_chunks_per_patient "
                f"allocation {int(config.rag_max_chunks_per_patient)}; no "
                "retrieved chunks were silently discarded"
            )
        selected = ranked_candidates
        lines = [
            f"[{QUERY_RAG_TEXT_VERSION}]",
            "These are query-retrieved excerpts from the information available "
            "before the current treatment decision. Read every excerpt. Extract "
            "only explicitly supported values; do not guess missing values.",
        ]
        texts = all_chunk_texts[row_id]
        for excerpt_index, item in enumerate(selected, start=1):
            chunk_index = int(item["chunk_index"])
            text = str(texts[chunk_index]) if chunk_index < len(texts) else ""
            if (
                config.rag_excerpt_chars is not None
                and len(text) > int(config.rag_excerpt_chars)
            ):
                raise NeuralQueryEvidenceCapacityOverflowError(
                    "complete query-RAG chunk contains "
                    f"{len(text)} characters for row {row_id}, exceeding "
                    "configured rag_excerpt_chars allocation "
                    f"{int(config.rag_excerpt_chars)}; text truncation is forbidden"
                )
            lines.extend(
                [
                    "",
                    (
                        f"<retrieved_excerpt id=\"E{excerpt_index:02d}\" "
                        f"query=\"{item['query_id']}\" bank=\"{item['bank']}\">"
                    ),
                    text,
                    "</retrieved_excerpt>",
                ]
            )
        documents.append("\n".join(lines))
    return documents


def build_query_feature_context(
    evidence: Mapping[str, Any],
    *,
    config: NeuralQueryAgenticForestConfig,
) -> Dict[str, Any]:
    role = mechanical_role_for_bank(str(evidence["bank"]))
    retrieved_training_chunks = [
        {
            key: value
            for key, value in {
                "evidence_id": str(item.get("evidence_id") or ""),
                "chunk_index": item.get("chunk_index"),
                "similarity": item.get("similarity"),
                "text": str(item.get("text") or ""),
            }.items()
            if value is not None
        }
        for item in evidence.get("top_chunks") or []
        if isinstance(item, Mapping)
    ]
    top_contrastive_ngrams = [
        {
            key: value
            for key, value in {
                "term": str(item.get("term") or ""),
                "tfidf_contrast": item.get("tfidf_contrast"),
            }.items()
            if value is not None
        }
        for item in evidence.get("top_contrastive_ngrams") or []
        if isinstance(item, Mapping)
    ]
    return {
        "prompt_version": QUERY_FEATURE_PROMPT_VERSION,
        "query_id": str(evidence["query_id"]),
        "bank": str(evidence["bank"]),
        "mechanical_role": role,
        "max_features": int(config.max_features_per_query),
        "query_diagnostics": {
            "fit_standardized_score": evidence.get("fit_standardized_score"),
            "member_count": evidence.get("member_count"),
            "member_subfolds": evidence.get("member_subfolds"),
            "statistical_gate_applied": False,
        },
        "evidence_field_cues": derive_evidence_field_cues(
            retrieved_training_chunks,
            top_contrastive_ngrams,
            limit=config.evidence_top_ngrams,
        ),
        "field_cue_policy": {
            "source_fields": [
                "retrieved_training_chunks.text",
                "top_contrastive_ngrams.term",
            ],
            "uses_fixed_clinical_vocabulary": False,
            "forwards_unlisted_evidence_metadata": False,
        },
        "top_contrastive_ngrams": top_contrastive_ngrams,
        "retrieved_training_chunks": retrieved_training_chunks,
    }


def render_query_feature_prompt(context: Mapping[str, Any]) -> str:
    """Render one bounded prompt per ungated semantic query."""

    payload = json.dumps(context, indent=2, default=str)
    return f"""You are interpreting one learned semantic retrieval direction over pre-treatment clinical text.

The direction was learned from the {context['bank']} prediction objective. It was not selected or rejected by a statistical gate. Review all supplied retrieved chunks and contrastive phrases, identify the general semantic theme, and propose zero or more distinct patient-level variables actually represented by this evidence.

Important rules:
- You may propose multiple variables when a retrieved object contains separable fields. To avoid overlooking them, inspect evidence_field_cues, which were mechanically derived only from literal labels and n-grams in the supplied training excerpts. Treat cues as navigation aids, not as a seed vocabulary, and require direct excerpt support for every proposal.
- Return no variables when the query is administrative, incoherent, post-decision, or not operationally extractable.
- Every feature must cite one or more supplied evidence_id values and short supporting phrases copied from those excerpts.
- Use only information documented before the current treatment decision. Prior treatments,
  responses, toxicities, and outcomes are valid baseline history when the excerpts clearly
  place them before that decision.
- Do not infer unavailable values, and exclude only events occurring after the current
  treatment decision.
- Do not choose analytic roles. Every proposal receives the fixed role {context['mechanical_role']} from its source bank.
- Continuous features must name the value and canonical unit to extract. Categorical features need 2-8 mutually exclusive clinical categories, including unknown/not_documented only when appropriate.
- Return at most {context['max_features']} proposals.

Return exactly one JSON object:
{{
  "general_topic": "short neutral semantic label",
  "query_quality": "coherent|mixed|weak|administrative_or_artifactual|post_decision_leakage",
  "proposals": [{{
    "action": "add",
    "name": "snake_case_variable_name",
    "type": "categorical|continuous",
    "categories": ["canonical categories for categorical variables"],
    "description": "precise pre-treatment extraction contract, including units/timing",
    "clinical_domain": "short domain",
    "parent_object": "reusable source object or report section",
    "supporting_evidence_ids": ["supplied evidence_id"],
    "supporting_phrases": ["short exact phrase from supplied excerpt"],
    "rationale": "why the evidence represents this variable"
  }}]
}}

Query evidence:
{payload}
"""


def query_feature_response_issues(
    response: Any,
    context: Mapping[str, Any],
) -> List[str]:
    if not isinstance(response, dict):
        return ["response must be one JSON object"]
    issues: List[str] = []
    if not str(response.get("general_topic") or "").strip():
        issues.append("general_topic is required")
    if response.get("query_quality") not in {
        "coherent",
        "mixed",
        "weak",
        "administrative_or_artifactual",
        "post_decision_leakage",
    }:
        issues.append("query_quality is invalid")
    proposals = response.get("proposals")
    if not isinstance(proposals, list):
        return [*issues, "proposals must be a list"]
    if len(proposals) > int(context["max_features"]):
        issues.append("too many proposals")
    evidence_ids = {
        str(item.get("evidence_id"))
        for item in context.get("retrieved_training_chunks", [])
    }
    names: set[str] = set()
    for index, proposal in enumerate(proposals, start=1):
        if not isinstance(proposal, dict):
            issues.append(f"proposal {index} must be an object")
            continue
        name = normalize_feature_name(proposal.get("name"))
        if not name or name in names:
            issues.append(f"proposal {index} has an invalid or duplicate name")
        names.add(name)
        if proposal.get("type") not in {"categorical", "continuous"}:
            issues.append(f"proposal {index} has invalid type")
        cited = proposal.get("supporting_evidence_ids")
        if not isinstance(cited, list) or not cited:
            issues.append(f"proposal {index} must cite evidence ids")
        elif any(str(value) not in evidence_ids for value in cited):
            issues.append(f"proposal {index} cites unknown evidence")
        phrases = proposal.get("supporting_phrases")
        if not isinstance(phrases, list) or not any(str(value).strip() for value in phrases):
            issues.append(f"proposal {index} must cite supporting phrases")
        if proposal.get("type") == "categorical":
            categories = proposal.get("categories")
            if not isinstance(categories, list) or not 2 <= len(categories) <= 8:
                issues.append(f"proposal {index} needs 2-8 categories")
    return issues


def normalize_feature_name(value: Any) -> str:
    name = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower())
    return name.strip("_")


def query_candidates_from_response(
    response: Mapping[str, Any],
    context: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    if query_feature_response_issues(response, context):
        raise ValueError("query feature response is not executable")
    role = str(context["mechanical_role"])
    output: List[Dict[str, Any]] = []
    for index, proposal in enumerate(response.get("proposals") or [], start=1):
        name = normalize_feature_name(proposal["name"])
        feature_type = str(proposal["type"])
        categories = (
            [str(value).strip() for value in proposal.get("categories") or []]
            if feature_type == "categorical"
            else None
        )
        output.append(
            {
                "candidate_id": f"{context['query_id']}__candidate_{index:02d}",
                "name": name,
                "type": feature_type,
                "categories": categories,
                "roles": [role],
                "description": str(proposal.get("description") or name).strip(),
                "clinical_domain": str(proposal.get("clinical_domain") or "clinical").strip(),
                "parent_object": str(proposal.get("parent_object") or name).strip(),
                "supporting_evidence_ids": [
                    str(value) for value in proposal.get("supporting_evidence_ids") or []
                ],
                "supporting_phrases": [
                    str(value) for value in proposal.get("supporting_phrases") or []
                ],
                "rationale": str(proposal.get("rationale") or "").strip(),
                "provenance": [
                    {
                        "query_id": str(context["query_id"]),
                        "bank": str(context["bank"]),
                        "mechanical_role": role,
                        "general_topic": str(response.get("general_topic")),
                        "query_quality": str(response.get("query_quality")),
                        "supporting_evidence_ids": [
                            str(value)
                            for value in proposal.get("supporting_evidence_ids") or []
                        ],
                        "supporting_phrases": [
                            str(value)
                            for value in proposal.get("supporting_phrases") or []
                        ],
                    }
                ],
            }
        )
    return output


def build_query_registry_context(
    candidates: Sequence[Mapping[str, Any]],
    *,
    config: NeuralQueryAgenticForestConfig,
) -> Dict[str, Any]:
    if len(candidates) > int(config.max_raw_feature_candidates):
        raise NeuralQueryEvidenceCapacityOverflowError(
            "complete neural-query candidate registry contains "
            f"{len(candidates)} candidates, exceeding configured "
            f"max_raw_feature_candidates allocation "
            f"{int(config.max_raw_feature_candidates)}; no candidates were "
            "silently discarded"
        )
    compact = []
    for candidate in candidates:
        provenance = [
            {
                key: row.get(key)
                for key in (
                    "query_id",
                    "bank",
                    "mechanical_role",
                    "general_topic",
                    "query_quality",
                )
            }
            for row in candidate.get("provenance") or []
        ]
        compact.append(
            {
                "candidate_id": candidate.get("candidate_id"),
                "name": candidate.get("name"),
                "type": candidate.get("type"),
                "categories": candidate.get("categories"),
                "roles": candidate.get("roles"),
                "description": candidate.get("description"),
                "clinical_domain": candidate.get("clinical_domain"),
                "parent_object": candidate.get("parent_object"),
                "supporting_phrases": [
                    str(value)
                    for value in candidate.get("supporting_phrases") or []
                ],
                "provenance": provenance,
            }
        )
    return {
        "prompt_version": QUERY_REGISTRY_PROMPT_VERSION,
        "max_canonical_features": int(config.max_canonical_features),
        "candidates": compact,
    }


def render_query_registry_prompt(context: Mapping[str, Any]) -> str:
    payload = json.dumps(context, indent=2, default=str)
    return f"""You are creating a small executable registry from variables proposed independently from treatment, outcome, and effect semantic queries.

Resolve spelling variants and true aliases globally. Keep clinically distinct subfields separate. A query may support more than one feature, and multiple queries may support one feature. Prefer precise baseline measurements over broad composite labels. Do not add unsupported variables.

Roles are mechanical provenance:
- treatment/outcome support contributes confounder
- effect support contributes effect_modifier
- a merged feature with both kinds of support receives both roles

Return at most {context['max_canonical_features']} canonical features. Every input candidate_id must appear exactly once, either inside one feature's source_candidate_ids or in dropped_candidates. Dropping is appropriate for duplication, post-decision leakage, non-extractability, or weak/administrative evidence; it is not a statistical query gate.

Return exactly one JSON object:
{{
  "features": [{{
    "name": "canonical_snake_case_name",
    "type": "categorical|continuous",
    "categories": ["2-8 canonical categories for categorical features"],
    "description": "complete extraction contract with timing, unit, and null policy",
    "clinical_domain": "clinical domain",
    "parent_object": "reusable parent object",
    "source_candidate_ids": ["one or more supplied candidate_id values"],
    "reason": "alias/value harmonization decision"
  }}],
  "dropped_candidates": [{{
    "candidate_id": "supplied candidate_id",
    "reason": "specific reason"
  }}]
}}

Candidate context:
{payload}
"""


def query_registry_response_issues(
    response: Any,
    context: Mapping[str, Any],
) -> List[str]:
    if not isinstance(response, dict):
        return ["registry response must be one object"]
    features = response.get("features")
    dropped = response.get("dropped_candidates")
    if not isinstance(features, list) or not isinstance(dropped, list):
        return ["features and dropped_candidates must both be lists"]
    issues: List[str] = []
    if len(features) > int(context["max_canonical_features"]):
        issues.append("registry exceeds max_canonical_features")
    supplied = {str(row["candidate_id"]) for row in context.get("candidates", [])}
    covered: List[str] = []
    names: set[str] = set()
    by_id = {str(row["candidate_id"]): row for row in context.get("candidates", [])}
    for index, feature in enumerate(features, start=1):
        if not isinstance(feature, dict):
            issues.append(f"feature {index} must be an object")
            continue
        name = normalize_feature_name(feature.get("name"))
        if not name or name in names:
            issues.append(f"feature {index} has invalid or duplicate name")
        names.add(name)
        sources = feature.get("source_candidate_ids")
        if not isinstance(sources, list) or not sources:
            issues.append(f"feature {index} needs source_candidate_ids")
            continue
        source_ids = [str(value) for value in sources]
        if any(value not in supplied for value in source_ids):
            issues.append(f"feature {index} cites an unknown candidate")
        covered.extend(source_ids)
        source_types = {str(by_id[value]["type"]) for value in source_ids if value in by_id}
        if feature.get("type") not in {"categorical", "continuous"}:
            issues.append(f"feature {index} has invalid type")
        elif len(source_types) == 1 and str(feature.get("type")) not in source_types:
            issues.append(f"feature {index} changes a unanimous source type")
        if feature.get("type") == "categorical":
            categories = feature.get("categories")
            if not isinstance(categories, list) or not 2 <= len(categories) <= 8:
                issues.append(f"feature {index} needs 2-8 categories")
    for index, row in enumerate(dropped, start=1):
        if not isinstance(row, dict) or str(row.get("candidate_id")) not in supplied:
            issues.append(f"dropped candidate {index} is invalid")
            continue
        if not str(row.get("reason") or "").strip():
            issues.append(f"dropped candidate {index} needs a reason")
        covered.append(str(row["candidate_id"]))
    if sorted(covered) != sorted(supplied):
        issues.append("every supplied candidate must be covered exactly once")
    return issues


def registry_from_response(
    response: Mapping[str, Any],
    context: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    issues = query_registry_response_issues(response, context)
    if issues:
        raise ValueError("; ".join(issues))
    by_id = {str(row["candidate_id"]): dict(row) for row in context["candidates"]}
    registry: List[Dict[str, Any]] = []
    for feature in response["features"]:
        source_ids = [str(value) for value in feature["source_candidate_ids"]]
        sources = [by_id[value] for value in source_ids]
        roles = list(
            dict.fromkeys(
                role for source in sources for role in (source.get("roles") or [])
            )
        )
        feature_type = str(feature["type"])
        categories = (
            [str(value).strip() for value in feature.get("categories") or []]
            if feature_type == "categorical"
            else None
        )
        description = str(feature.get("description") or feature["name"]).strip()
        structured_description = (
            f"clinical_domain={feature.get('clinical_domain') or 'clinical'}; "
            f"parent_object={feature.get('parent_object') or feature['name']}: "
            f"{description} Use only pre-treatment query-retrieved evidence; "
            "return null when not explicitly documented."
        )
        registry.append(
            {
                "name": normalize_feature_name(feature["name"]),
                "type": feature_type,
                "categories": categories,
                "roles": roles,
                "description": structured_description,
                "clinical_domain": str(feature.get("clinical_domain") or "clinical"),
                "parent_object": str(feature.get("parent_object") or feature["name"]),
                "source_candidate_ids": source_ids,
                "source_query_ids": list(
                    dict.fromkeys(
                        str(provenance.get("query_id"))
                        for source in sources
                        for provenance in source.get("provenance") or []
                    )
                ),
                "provenance": [
                    provenance
                    for source in sources
                    for provenance in source.get("provenance") or []
                ],
                "harmonization_reason": str(feature.get("reason") or ""),
            }
        )
    return registry, [dict(value) for value in response["dropped_candidates"]]


def registry_specs(registry: Sequence[Mapping[str, Any]]) -> List[ExplicitFeatureSpec]:
    return [
        ExplicitFeatureSpec(
            name=str(row["name"]),
            type=str(row["type"]),
            categories=(
                [str(value) for value in row.get("categories") or []]
                if row["type"] == "categorical"
                else None
            ),
            description=str(row.get("description") or row["name"]),
            roles=[str(value) for value in row.get("roles") or []],
        )
        for row in registry
    ]


def extraction_request_groups(
    registry: Sequence[Mapping[str, Any]],
    *,
    maximum: int,
) -> List[List[str]]:
    """Return domain-coherent extraction groups with a hard 1-10 cap."""

    if not 1 <= int(maximum) <= 10:
        raise ValueError("maximum must be in [1, 10]")
    domains: Dict[str, List[str]] = {}
    order: List[str] = []
    for row in registry:
        domain = normalize_feature_name(row.get("clinical_domain") or "clinical")
        if domain not in domains:
            domains[domain] = []
            order.append(domain)
        domains[domain].append(str(row["name"]))
    groups = [
        names[start : start + int(maximum)]
        for domain in order
        for names in [domains[domain]]
        for start in range(0, len(names), int(maximum))
    ]
    if any(not group or len(group) > 10 for group in groups):
        raise RuntimeError("invalid extraction request grouping")
    return groups


def render_query_review_prompt(context: Mapping[str, Any]) -> str:
    """Render a bounded additive review prompt for later agent iterations."""

    payload = json.dumps(context, indent=2, default=str)
    maximum = int(context.get("max_additions", 0))
    return f"""You are reviewing a bounded structured-feature registry derived from fifteen ungated semantic queries.

Use the oracle-free inner-fold diagnostics, extraction failures, and the original query evidence to decide whether a small number of additional or materially refined extraction contracts are warranted. Existing valid features are additive and cannot be removed here. Refer back to supplied query_id and evidence excerpts; do not invent variables absent from them.

Return at most {maximum} additions. Return JSON only:
{{
  "proposals": [{{
    "action": "add|refine",
    "name": "new_or_existing_snake_case_name",
    "type": "categorical|continuous",
    "categories": ["canonical categories"],
    "roles": ["confounder|effect_modifier"],
    "description": "complete revised extraction contract",
    "source_query_ids": ["supplied query_id"],
    "rationale": "specific diagnostic and source evidence"
  }}]
}}

Review context:
{payload}
"""


def query_review_response_issues(
    response: Any,
    context: Mapping[str, Any],
) -> List[str]:
    if not isinstance(response, dict):
        return ["review response must be one JSON object"]
    proposals = response.get("proposals")
    if not isinstance(proposals, list):
        return ["proposals must be a list"]
    issues: List[str] = []
    if len(proposals) > int(context.get("max_additions", 0)):
        issues.append("too many review proposals")
    supplied_queries = {
        str(item.get("query_id")) for item in context.get("query_evidence", [])
    }
    current_names = {
        normalize_feature_name(item.get("name"))
        for item in context.get("current_registry", [])
    }
    seen: set[str] = set()
    for index, proposal in enumerate(proposals, start=1):
        if not isinstance(proposal, dict):
            issues.append(f"proposal {index} must be an object")
            continue
        action = str(proposal.get("action") or "")
        if action not in {"add", "refine"}:
            issues.append(f"proposal {index} has invalid action")
        name = normalize_feature_name(proposal.get("name"))
        if not name or name in seen:
            issues.append(f"proposal {index} has invalid or duplicate name")
        seen.add(name)
        if action == "refine" and name not in current_names:
            issues.append(f"proposal {index} refines an unknown feature")
        if proposal.get("type") not in {"categorical", "continuous"}:
            issues.append(f"proposal {index} has invalid type")
        query_ids = proposal.get("source_query_ids")
        if not isinstance(query_ids, list) or not query_ids:
            issues.append(f"proposal {index} needs source_query_ids")
        elif any(str(value) not in supplied_queries for value in query_ids):
            issues.append(f"proposal {index} cites unknown query evidence")
        if proposal.get("type") == "categorical":
            categories = proposal.get("categories")
            if not isinstance(categories, list) or not 2 <= len(categories) <= 8:
                issues.append(f"proposal {index} needs 2-8 categories")
    return issues


def review_candidates_from_response(
    response: Mapping[str, Any],
    context: Mapping[str, Any],
    *,
    round_index: int,
) -> List[Dict[str, Any]]:
    issues = query_review_response_issues(response, context)
    if issues:
        raise ValueError("; ".join(issues))
    evidence_by_query = {
        str(item["query_id"]): item for item in context.get("query_evidence", [])
    }
    output: List[Dict[str, Any]] = []
    for index, proposal in enumerate(response.get("proposals") or [], start=1):
        query_ids = [str(value) for value in proposal["source_query_ids"]]
        roles = list(
            dict.fromkeys(
                mechanical_role_for_bank(str(evidence_by_query[query_id]["bank"]))
                for query_id in query_ids
            )
        )
        feature_type = str(proposal["type"])
        output.append(
            {
                "candidate_id": (
                    f"review_{int(round_index):02d}__candidate_{int(index):02d}"
                ),
                "action": str(proposal["action"]),
                "name": normalize_feature_name(proposal["name"]),
                "type": feature_type,
                "categories": (
                    [str(value) for value in proposal.get("categories") or []]
                    if feature_type == "categorical"
                    else None
                ),
                "roles": roles,
                "description": str(proposal.get("description") or proposal["name"]),
                "clinical_domain": str(proposal.get("clinical_domain") or "clinical"),
                "parent_object": str(proposal.get("parent_object") or proposal["name"]),
                "supporting_phrases": [],
                "rationale": str(proposal.get("rationale") or ""),
                "provenance": [
                    {
                        "query_id": query_id,
                        "bank": str(evidence_by_query[query_id]["bank"]),
                        "mechanical_role": mechanical_role_for_bank(
                            str(evidence_by_query[query_id]["bank"])
                        ),
                        "review_round": int(round_index),
                    }
                    for query_id in query_ids
                ],
            }
        )
    return output


def apply_review_candidates_to_registry(
    registry: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    *,
    maximum: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Apply bounded additive/refinement review without deleting valid features.

    Roles are unioned from mechanical query provenance.  Additions beyond the
    predeclared registry cap are rejected explicitly rather than silently
    displacing an existing feature.
    """

    if int(maximum) < 1:
        raise ValueError("maximum must be positive")
    output = [dict(row) for row in registry]
    by_name = {normalize_feature_name(row.get("name")): index for index, row in enumerate(output)}
    decisions: List[Dict[str, Any]] = []
    for candidate in candidates:
        name = normalize_feature_name(candidate.get("name"))
        action = str(candidate.get("action") or "add")
        existing_index = by_name.get(name)
        if action == "refine" and existing_index is None:
            decisions.append({"name": name, "action": action, "accepted": False,
                              "reason": "refinement target is absent"})
            continue
        if action == "add" and existing_index is not None:
            action = "refine"
        if action == "add" and len(output) >= int(maximum):
            decisions.append({"name": name, "action": action, "accepted": False,
                              "reason": "canonical feature cap reached"})
            continue

        candidate_roles = [str(value) for value in candidate.get("roles") or []]
        clinical_domain = str(candidate.get("clinical_domain") or "clinical")
        parent_object = str(candidate.get("parent_object") or name)
        description = str(candidate.get("description") or name).strip()
        structured_description = (
            f"clinical_domain={clinical_domain}; parent_object={parent_object}: "
            f"{description} Use only query-retrieved history documented before "
            "the current treatment decision; return null when not explicitly documented."
        )
        if action == "refine":
            assert existing_index is not None
            prior = output[existing_index]
            prior_roles = [str(value) for value in prior.get("roles") or []]
            updated = dict(prior)
            updated.update(
                {
                    "type": str(candidate["type"]),
                    "categories": (
                        [str(value) for value in candidate.get("categories") or []]
                        if candidate["type"] == "categorical"
                        else None
                    ),
                    "roles": list(dict.fromkeys([*prior_roles, *candidate_roles])),
                    "description": structured_description,
                    "clinical_domain": clinical_domain,
                    "parent_object": parent_object,
                    "provenance": [
                        *(prior.get("provenance") or []),
                        *(candidate.get("provenance") or []),
                    ],
                    "review_refined": True,
                }
            )
            output[existing_index] = updated
        else:
            output.append(
                {
                    "name": name,
                    "type": str(candidate["type"]),
                    "categories": (
                        [str(value) for value in candidate.get("categories") or []]
                        if candidate["type"] == "categorical"
                        else None
                    ),
                    "roles": list(dict.fromkeys(candidate_roles)),
                    "description": structured_description,
                    "clinical_domain": clinical_domain,
                    "parent_object": parent_object,
                    "source_candidate_ids": [str(candidate.get("candidate_id"))],
                    "source_query_ids": list(
                        dict.fromkeys(
                            str(row.get("query_id"))
                            for row in candidate.get("provenance") or []
                        )
                    ),
                    "provenance": list(candidate.get("provenance") or []),
                    "review_added": True,
                }
            )
            by_name[name] = len(output) - 1
        decisions.append({"name": name, "action": action, "accepted": True})
    if len(output) > int(maximum):
        raise RuntimeError("review exceeded the canonical feature cap")
    return output, decisions
