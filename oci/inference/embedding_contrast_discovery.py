"""Embedding-contrast evidence for agentic explicit-variable discovery."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from ..config import (
    AppliedInferenceConfig,
    ClusterLocalEmbeddingScientificConfig,
    EmbeddingContrastDiscoveryConfig,
)
from ..models.concept_embedding_cache import (
    ConceptEmbeddingCache,
    clear_sentence_transformer_cache,
    load_sentence_transformer,
)
from ..models.concept_embedding_utils import chunk_text_words

logger = logging.getLogger(__name__)


class ClusterLocalEmbeddingFeasibilityError(ValueError):
    """Fail-closed cluster geometry error carrying its lossless fit summary."""

    def __init__(self, message: str, *, summary: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.summary = copy.deepcopy(dict(summary))


def _cluster_local_scientific_config(
    embedding_config: EmbeddingContrastDiscoveryConfig | Mapping[str, Any],
) -> ClusterLocalEmbeddingScientificConfig:
    if isinstance(embedding_config, Mapping):
        raw = embedding_config.get("cluster_local_scientific")
    else:
        raw = embedding_config.cluster_local_scientific
    if isinstance(raw, ClusterLocalEmbeddingScientificConfig):
        return raw
    if isinstance(raw, Mapping):
        return ClusterLocalEmbeddingScientificConfig.from_mapping(raw)
    raise ValueError(
        "cluster-local embedding execution requires an explicit closed "
        "cluster_local_scientific configuration"
    )


def _embedding_cluster_kmeans_parameters(
    embedding_config: EmbeddingContrastDiscoveryConfig | Mapping[str, Any],
    *,
    n_usable: int,
    canonical_group_seed: int,
    n_clusters: int | None = None,
) -> Dict[str, Any]:
    """Return the one closed MiniBatchKMeans parameterization used natively."""

    usable = int(n_usable)
    if usable < 1:
        raise ValueError("clustered embedding KMeans requires usable rows")
    config = _cluster_local_scientific_config(embedding_config)
    cluster_count = int(config.requested_cluster_count)
    if n_clusters is not None and int(n_clusters) != cluster_count:
        raise ValueError(
            "cluster-local KMeans cannot adapt the configured cluster count"
        )
    seed = int(canonical_group_seed)
    if isinstance(canonical_group_seed, bool) or not 0 <= seed < 2**31:
        raise ValueError("canonical cluster group seed must be a 31-bit integer")
    actual_batch_size = max(
        int(config.kmeans_batch_size_lower_bound),
        min(int(config.kmeans_batch_size_upper_bound), usable),
    )
    return {
        "n_clusters": cluster_count,
        "init": config.kmeans_init,
        "max_iter": int(config.kmeans_max_iter),
        "batch_size": actual_batch_size,
        "verbose": int(config.kmeans_verbose),
        "compute_labels": bool(config.kmeans_compute_labels),
        "random_state": seed,
        "tol": float(config.kmeans_tol),
        "max_no_improvement": config.kmeans_max_no_improvement,
        "init_size": config.kmeans_init_size,
        "n_init": config.kmeans_n_init,
        "reassignment_ratio": float(config.kmeans_reassignment_ratio),
    }


def _canonicalize_svd_component_signs(
    components: np.ndarray,
    *,
    policy: str,
) -> np.ndarray:
    """Remove the mathematically arbitrary sign from every right singular vector."""

    if policy != "largest_absolute_coordinate_positive_v1":
        raise ValueError("unsupported cluster-local SVD sign policy")
    output = np.asarray(components).copy()
    if output.ndim != 2:
        raise ValueError("cluster-local SVD components must be a matrix")
    for index in range(output.shape[0]):
        row = output[index]
        pivot = int(np.argmax(np.abs(row)))
        if not np.isfinite(row[pivot]) or row[pivot] == 0:
            raise ValueError("cluster-local SVD component has no canonical sign pivot")
        if row[pivot] < 0:
            output[index] *= -1
    return output


def _retrieval_tfidf_terms(
    positive_chunks: Sequence[Mapping[str, Any]],
    negative_chunks: Sequence[Mapping[str, Any]],
    *,
    ngram_range: Tuple[int, int],
    max_features: int,
    top_terms_per_side: int,
) -> List[Dict[str, Any]]:
    """Return a lexical projection of opposing semantic-retrieval tails."""

    positive = [
        str(row.get("text") or row.get("chunk_text") or "").strip()
        for row in positive_chunks
        if isinstance(row, Mapping)
    ]
    negative = [
        str(row.get("text") or row.get("chunk_text") or "").strip()
        for row in negative_chunks
        if isinstance(row, Mapping)
    ]
    positive = [text for text in positive if text]
    negative = [text for text in negative if text]
    if not positive or not negative:
        return []
    documents = [*positive, *negative]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=tuple(map(int, ngram_range)),
        min_df=1,
        max_features=max(1, int(max_features)),
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
        token_pattern=r"(?u)\b[\w][\w+./-]*\b",
    )
    try:
        matrix = vectorizer.fit_transform(documents)
    except ValueError:
        return []
    terms = vectorizer.get_feature_names_out().astype(str)
    positive_mean = np.asarray(matrix[: len(positive)].mean(axis=0)).ravel()
    negative_mean = np.asarray(matrix[len(positive) :].mean(axis=0)).ravel()
    contrast = positive_mean - negative_mean
    document_frequency = np.asarray((matrix > 0).sum(axis=0)).ravel().astype(int)

    limit = max(1, int(top_terms_per_side))
    positive_order = sorted(
        np.flatnonzero(contrast > 0.0),
        key=lambda index: (-float(contrast[index]), str(terms[index])),
    )[:limit]
    negative_order = sorted(
        np.flatnonzero(contrast < 0.0),
        key=lambda index: (float(contrast[index]), str(terms[index])),
    )[:limit]
    output: List[Dict[str, Any]] = []
    for polarity, indices in (("positive", positive_order), ("negative", negative_order)):
        for index in indices:
            output.append(
                {
                    "term": str(terms[index]),
                    "polarity": polarity,
                    "tfidf_contrast": float(contrast[index]),
                    "positive_mean_tfidf": float(positive_mean[index]),
                    "negative_mean_tfidf": float(negative_mean[index]),
                    "document_frequency": int(document_frequency[index]),
                }
            )
    return output


class EmbeddingContrastEvidenceGenerator:
    """Build train-fold embedding contrasts and retrieve aligned text chunks."""

    def __init__(
        self,
        *,
        config: AppliedInferenceConfig,
        output_dir: Path,
        embedding_provider: Optional[Any] = None,
        precompute_devices: Optional[Sequence[Any]] = None,
    ) -> None:
        self.config = config
        if str(config.architecture.model_type) == "multi_model_forest":
            model_config = config.architecture.multi_model_forest
        else:
            model_config = config.architecture.multi_model_agentic_forest
        self.embedding_config: EmbeddingContrastDiscoveryConfig = (
            model_config.embedding_contrast
        )
        self.output_dir = Path(output_dir)
        self.embedding_provider = embedding_provider
        self.precompute_devices = list(precompute_devices or [])
        self._prepared = False
        self._row_ids: List[Any] = []
        self._row_id_to_position: Dict[Any, int] = {}
        self._chunks_by_position: List[List[str]] = []
        self._flat_embeddings = None
        self._offsets = None
        self._cache = None
        self._cache_dir: Optional[Path] = None
        self._chunk_cache_reused = False
        self._concept_probe_skip_reason: Optional[str] = None
        self._external_corpora: List[Dict[str, Any]] = []
        self._cluster_fit_row_ids: tuple[int, ...] | None = None
        self._cluster_group_seed: int | None = None

    @property
    def enabled(self) -> bool:
        return bool(getattr(self.embedding_config, "enabled", False))

    def prepare(self, dataset: pd.DataFrame) -> None:
        """Prepare chunk embeddings for the dataset order used by this runner."""
        if not self.enabled or self._prepared:
            return
        if self.config.text_column not in dataset.columns:
            raise ValueError(f"Embedding contrast requires text column {self.config.text_column!r}")

        texts = [str(text or "") for text in dataset[self.config.text_column].fillna("")]
        if "_oci_row_id" in dataset.columns:
            self._row_ids = dataset["_oci_row_id"].tolist()
        else:
            self._row_ids = list(range(len(dataset)))
        self._row_id_to_position = {
            _row_key(row_id): idx for idx, row_id in enumerate(self._row_ids)
        }
        if self.embedding_provider is not None:
            self._chunks_by_position = [
                chunk_text_words(
                    text,
                    int(self.embedding_config.chunk_size_words),
                    int(self.embedding_config.chunk_overlap_words),
                    int(self.embedding_config.max_chunks),
                    str(self.embedding_config.chunk_selection),
                )
                for text in texts
            ]
            self._prepare_from_provider()
        else:
            self._prepare_from_sentence_transformer_cache(texts)
        self._external_corpora = self._load_external_corpora()
        self._prepared = True

    def bind_cluster_fit_context(
        self,
        *,
        ordered_fit_row_ids: Sequence[int],
        canonical_group_seed: int,
    ) -> None:
        """Bind KMeans to the ordered rows and seed of one discovery context."""

        rows = tuple(map(int, ordered_fit_row_ids))
        seed = int(canonical_group_seed)
        if (
            not rows
            or len(rows) != len(set(rows))
            or any(row < 0 for row in rows)
            or isinstance(canonical_group_seed, bool)
            or not 0 <= seed < 2**31
        ):
            raise ValueError("cluster fit context is invalid")
        if self._cluster_fit_row_ids is not None and (
            self._cluster_fit_row_ids != rows or self._cluster_group_seed != seed
        ):
            raise RuntimeError("cluster fit context was rebound")
        self._cluster_fit_row_ids = rows
        self._cluster_group_seed = seed

    def _cluster_fit_context(
        self,
        positions: Sequence[int],
    ) -> tuple[tuple[int, ...], int]:
        rows = tuple(int(self._row_ids[int(position)]) for position in positions)
        if self._cluster_fit_row_ids != rows or self._cluster_group_seed is None:
            raise ValueError(
                "cluster-local embedding fit lacks its ordered rows and seed"
            )
        return rows, int(self._cluster_group_seed)

    def build_cluster_only_evidence(
        self,
        *,
        discovery_df: pd.DataFrame,
        y: np.ndarray,
        t: np.ndarray,
    ) -> Dict[str, Any]:
        """Fit only KMeans plus the two configured cluster-local SVD families.

        This is the readiness-preflight entry point.  It deliberately cannot
        construct whole-cohort contrasts, concept probes, or any supervised
        logistic model.
        """

        if not self.enabled or not bool(
            self.embedding_config.include_cluster_contrast_vectors
        ):
            raise ValueError("cluster-only embedding evidence is disabled")
        if not self._prepared:
            self.prepare(discovery_df)
        positions = self._positions_for_frame(discovery_df)
        if len(positions) != len(discovery_df):
            raise ValueError("cluster-only evidence lost an ordered discovery row")
        self._cluster_fit_context(positions)
        patient_embeddings = self._patient_embeddings(positions)
        patient_embeddings = _residualize_embeddings(
            patient_embeddings,
            discovery_df,
            self.embedding_config.residualize_columns,
        )
        config = _cluster_local_scientific_config(self.embedding_config)
        patient_embeddings = _normalize_rows_configured(
            patient_embeddings,
            normalize=bool(config.normalize_patient_embeddings),
            epsilon=float(config.normalization_epsilon),
            zero_vector_policy=config.zero_vector_policy,
            dtype=config.computation_dtype,
        )
        contrasts, summary = self._build_cluster_contrast_vectors(
            positions=positions,
            patient_embeddings=patient_embeddings,
            y=np.asarray(y, dtype=float),
            t=np.asarray(t, dtype=float),
            concept_phrases=(),
            concept_embeddings=None,
        )
        evidence: Dict[str, Any] = {
            "enabled": True,
            "execution_mode": "cluster_only_no_probe_or_whole_cohort_v1",
            "model_name": self.embedding_config.model_name,
            "unit": "patient_row",
            "n_patients": len(positions),
            "n_concept_phrases": 0,
            "external_corpora": [],
            "contrasts": contrasts,
            "cluster_contrast_vectors": summary,
        }
        return evidence

    def build_evidence(
        self,
        *,
        discovery_df: pd.DataFrame,
        y: np.ndarray,
        t: np.ndarray,
        pseudo_target: Any,
        t_resid: Any,
        pseudo_target_names: Optional[Sequence[str]] = None,
        importance: Optional[Dict[str, Any]] = None,
        dina_nuisance=None,
    ) -> Dict[str, Any]:
        """Return agent-facing embedding contrast evidence for one discovery fold."""
        if not self.enabled:
            return {}
        if not self._prepared:
            self.prepare(discovery_df)

        positions = self._positions_for_frame(discovery_df)
        if not positions:
            return {"enabled": True, "skipped": "no_rows"}
        cluster_config = self.embedding_config.cluster_local_scientific
        if bool(self.embedding_config.include_cluster_contrast_vectors):
            self._cluster_fit_context(positions)

        patient_embeddings = self._patient_embeddings(positions)
        patient_embeddings = _residualize_embeddings(
            patient_embeddings,
            discovery_df,
            self.embedding_config.residualize_columns,
        )
        if cluster_config is None:
            patient_embeddings = _normalize_rows(patient_embeddings)
        else:
            patient_embeddings = _normalize_rows_configured(
                patient_embeddings,
                normalize=bool(cluster_config.normalize_patient_embeddings),
                epsilon=float(cluster_config.normalization_epsilon),
                zero_vector_policy=cluster_config.zero_vector_policy,
                dtype=cluster_config.computation_dtype,
            )

        concept_phrases = self._concept_phrases(importance or {})
        self._concept_probe_skip_reason = None
        concept_embeddings = self._encode_concepts(concept_phrases) if concept_phrases else None
        y_array = np.asarray(y, dtype=float)
        t_array = np.asarray(t, dtype=float)
        pseudo_targets = _named_pseudo_targets(
            pseudo_target,
            t_resid,
            pseudo_target_names,
        )
        contrasts = []
        cluster_contrast_summary: Dict[str, Any] = {}
        for contrast in self._contrast_specs(
            y=y_array,
            t=t_array,
            pseudo_targets=pseudo_targets,
            dina_nuisance=dina_nuisance,
        ):
            contrasts.append(
                self._build_one_contrast(
                    positions=positions,
                    patient_embeddings=patient_embeddings,
                    concept_phrases=concept_phrases,
                    concept_embeddings=concept_embeddings,
                    **contrast,
                )
            )
        if bool(self.embedding_config.include_confounder_vector_contrast):
            contrasts.append(
                self._build_confounder_vector_contrast(
                    positions=positions,
                    patient_embeddings=patient_embeddings,
                    y=y_array,
                    t=t_array,
                    concept_phrases=concept_phrases,
                    concept_embeddings=concept_embeddings,
                )
            )
        if bool(self.embedding_config.include_residualized_interaction_contrast):
            contrasts.append(
                self._build_residualized_interaction_contrast(
                    positions=positions,
                    patient_embeddings=patient_embeddings,
                    y=y_array,
                    t=t_array,
                    concept_phrases=concept_phrases,
                    concept_embeddings=concept_embeddings,
                )
            )
        if bool(self.embedding_config.include_cluster_contrast_vectors):
            cluster_contrasts, cluster_contrast_summary = self._build_cluster_contrast_vectors(
                positions=positions,
                patient_embeddings=patient_embeddings,
                y=y_array,
                t=t_array,
                concept_phrases=concept_phrases,
                concept_embeddings=concept_embeddings,
            )
            contrasts.extend(cluster_contrasts)

        evidence = {
            "enabled": True,
            "model_name": self.embedding_config.model_name,
            "unit": "patient_row",
            "chunking": {
                "chunk_size_words": int(self.embedding_config.chunk_size_words),
                "chunk_overlap_words": int(self.embedding_config.chunk_overlap_words),
                "max_seq_length_tokens": (
                    None
                    if self.embedding_config.max_seq_length is None
                    else int(self.embedding_config.max_seq_length)
                ),
                "max_chunks": int(self.embedding_config.max_chunks),
                "chunk_selection": str(self.embedding_config.chunk_selection),
            },
            "residualized_columns": [
                col
                for col in self.embedding_config.residualize_columns
                if col in discovery_df.columns
            ],
            "n_patients": int(len(positions)),
            "n_concept_phrases": int(len(concept_phrases)),
            "external_corpora": [
                {
                    "name": str(corpus["name"]),
                    "cache_path": str(corpus["cache_path"]),
                    "n_chunks": int(corpus["embeddings"].shape[0]),
                }
                for corpus in self._external_corpora
            ],
            "contrasts": contrasts,
        }
        if cluster_contrast_summary:
            evidence["cluster_contrast_vectors"] = cluster_contrast_summary
        if self._concept_probe_skip_reason:
            evidence["concept_probe_skipped"] = self._concept_probe_skip_reason
        return evidence

    def _prepare_from_provider(self) -> None:
        flat_chunks = [chunk for chunks in self._chunks_by_position for chunk in chunks]
        offsets = np.zeros(len(self._chunks_by_position) + 1, dtype=np.int64)
        cursor = 0
        for idx, chunks in enumerate(self._chunks_by_position):
            cursor += len(chunks)
            offsets[idx + 1] = cursor
        embeddings = self._encode_with_provider(flat_chunks)
        embeddings = _coerce_embedding_matrix(embeddings, expected_rows=len(flat_chunks))
        if bool(self.embedding_config.normalize_embeddings):
            embeddings = _normalize_rows(embeddings)
        self._flat_embeddings = embeddings.astype(np.float32, copy=False)
        self._offsets = offsets

    def _prepare_from_sentence_transformer_cache(self, texts: Sequence[str]) -> None:
        dataset_path = self.config.dataset_path or str(self.output_dir / "in_memory_dataset")
        if self.embedding_config.cache_dir:
            raw_cache_dir = Path(str(self.embedding_config.cache_dir))
            cache_dir = (
                raw_cache_dir.parent
                if raw_cache_dir.name.startswith("cecnn_chunk_embeddings_")
                else raw_cache_dir
            )
        else:
            cache_dir = _default_embedding_cache_dir(dataset_path, self.output_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_dir = cache_dir
        cache = ConceptEmbeddingCache(
            cache_dir=str(cache_dir),
            sentence_model_name=str(self.embedding_config.model_name),
            dataset_path=str(dataset_path),
            chunk_size_words=int(self.embedding_config.chunk_size_words),
            chunk_overlap_words=int(self.embedding_config.chunk_overlap_words),
            max_chunks=int(self.embedding_config.max_chunks),
            normalize_embeddings=bool(self.embedding_config.normalize_embeddings),
            chunk_selection=str(self.embedding_config.chunk_selection),
            max_seq_length=self.embedding_config.max_seq_length,
        )
        logger.info("Embedding contrast chunk cache: %s", cache.cache_path)
        cache_valid = cache.is_valid(expected_num_samples=len(texts))
        self._chunk_cache_reused = bool(cache_valid)
        if cache_valid:
            logger.info("Reusing embedding contrast chunk cache")
        else:
            logger.info("Building embedding contrast chunk cache")
            try:
                precompute_devices = _coerce_torch_devices(self.precompute_devices)
                if len(precompute_devices) > 1:
                    cache.precompute_multi_gpu(
                        list(texts),
                        devices=precompute_devices,
                        batch_size=int(self.embedding_config.batch_size),
                    )
                elif len(precompute_devices) == 1:
                    cache.precompute(
                        list(texts),
                        device=precompute_devices[0],
                        batch_size=int(self.embedding_config.batch_size),
                    )
                else:
                    cache.precompute(
                        list(texts),
                        device=_torch_device_or_none(self.embedding_config.device),
                        batch_size=int(self.embedding_config.batch_size),
                    )
            finally:
                _release_sentence_transformer_model(str(self.embedding_config.model_name))
        cache.open()
        self._cache = cache
        self._flat_embeddings = cache.hidden_states_array.flat
        self._offsets = cache.hidden_states_array.offsets
        self._chunks_by_position = cache.load_chunks(expected_num_samples=len(texts))

    def _load_external_corpora(self) -> List[Dict[str, Any]]:
        corpora: List[Dict[str, Any]] = []
        for raw_path in self.embedding_config.external_corpus_cache_dirs:
            for cache_path in _resolve_external_cache_paths(raw_path):
                try:
                    corpus = _load_external_corpus_cache(cache_path)
                except Exception as exc:
                    logger.warning(
                        "Skipping external embedding corpus cache %s: %s",
                        cache_path,
                        exc,
                    )
                    continue
                corpora.append(corpus)
                logger.info(
                    "Loaded external embedding corpus %s: %d chunks",
                    corpus["name"],
                    int(corpus["embeddings"].shape[0]),
                )
        return corpora

    def _positions_for_frame(self, frame: pd.DataFrame) -> List[int]:
        if "_oci_row_id" in frame.columns:
            row_ids = frame["_oci_row_id"].tolist()
        else:
            row_ids = list(range(len(frame)))
        positions = []
        for row_id in row_ids:
            position = self._row_id_to_position.get(_row_key(row_id))
            if position is not None:
                positions.append(int(position))
        return positions

    def _chunk_matrix(self, position: int) -> np.ndarray:
        start = int(self._offsets[position])
        end = int(self._offsets[position + 1])
        return np.asarray(self._flat_embeddings[start:end], dtype=np.float32)

    def chunk_matrices(
        self,
        row_ids: Optional[Sequence[Any]] = None,
    ) -> List[np.ndarray]:
        """Return prepared chunk embeddings in the requested dataset-row order.

        This small public accessor lets the researcher Stage 1 runner reuse the
        ordinary embedding cache for neural-query discovery without depending
        on a sealed-cache adapter.
        """

        if not self._prepared:
            raise RuntimeError("prepare() must be called before reading chunk embeddings")
        requested = self._row_ids if row_ids is None else list(row_ids)
        positions: List[int] = []
        for row_id in requested:
            position = self._row_id_to_position.get(_row_key(row_id))
            if position is None:
                raise KeyError(f"embedding cache has no dataset row {row_id!r}")
            positions.append(int(position))
        return [self._chunk_matrix(position) for position in positions]

    def chunk_texts(
        self,
        row_ids: Optional[Sequence[Any]] = None,
    ) -> List[List[str]]:
        """Return prepared chunk text in the requested dataset-row order."""

        if not self._prepared:
            raise RuntimeError("prepare() must be called before reading chunk text")
        requested = self._row_ids if row_ids is None else list(row_ids)
        output: List[List[str]] = []
        for row_id in requested:
            position = self._row_id_to_position.get(_row_key(row_id))
            if position is None:
                raise KeyError(f"embedding cache has no dataset row {row_id!r}")
            output.append(list(self._chunks_by_position[int(position)]))
        return output

    def _patient_embeddings(self, positions: Sequence[int]) -> np.ndarray:
        config = self.embedding_config.cluster_local_scientific
        if config is not None:
            dtype = np.dtype(config.computation_dtype)
            pooled = []
            for position in positions:
                chunks = np.asarray(self._chunk_matrix(int(position)), dtype=dtype)
                if chunks.ndim != 2 or chunks.shape[0] < 1:
                    raise ValueError(
                        "cluster-local pooling requires at least one authenticated "
                        "chunk per patient"
                    )
                pooled.append(np.mean(chunks, axis=0, dtype=dtype))
            return _normalize_rows_configured(
                np.vstack(pooled).astype(dtype, copy=False),
                normalize=bool(config.normalize_patient_embeddings),
                epsilon=float(config.normalization_epsilon),
                zero_vector_policy=config.zero_vector_policy,
                dtype=config.computation_dtype,
            )
        pooled = []
        for position in positions:
            chunks = self._chunk_matrix(int(position))
            if chunks.size == 0:
                pooled.append(np.zeros(1, dtype=np.float32))
            else:
                pooled.append(np.mean(chunks, axis=0))
        matrix = np.vstack(pooled).astype(np.float32, copy=False)
        return _normalize_rows(matrix)

    def _contrast_specs(
        self,
        *,
        y: np.ndarray,
        t: np.ndarray,
        pseudo_targets: Sequence[Tuple[str, np.ndarray, np.ndarray]],
        dina_nuisance=None,
    ) -> List[Dict[str, Any]]:
        specs: List[Dict[str, Any]] = []
        treatment_labels, treatment_mask = _binary_labels(t)
        specs.append(
            {
                "name": "treatment",
                "positive_label": "treated",
                "negative_label": "untreated",
                "labels": treatment_labels,
                "mask": treatment_mask,
                "sample_weights": None,
                "role_hint": "confounder",
                "metadata": {
                    "contrast_family": "marginal",
                    "direction_formula": "mean_embedding(T=1) - mean_embedding(T=0)",
                },
            }
        )

        if str(self.config.outcome_type).lower() == "continuous":
            labels, mask = _tail_labels(y, float(self.embedding_config.pseudo_target_quantile))
            positive_label = "higher_outcome"
            negative_label = "lower_outcome"
        else:
            labels, mask = _binary_labels(y)
            positive_label = "outcome_present"
            negative_label = "outcome_absent"
        specs.append(
            {
                "name": "outcome",
                "positive_label": positive_label,
                "negative_label": negative_label,
                "labels": labels,
                "mask": mask,
                "sample_weights": None,
                "role_hint": "confounder",
                "metadata": {
                    "contrast_family": "marginal",
                    "direction_formula": (
                        f"mean_embedding({positive_label}) - " f"mean_embedding({negative_label})"
                    ),
                },
            }
        )

        if bool(getattr(self.embedding_config, "include_cell_contrasts", True)):
            specs.extend(
                self._cell_contrast_specs(
                    treatment_labels=treatment_labels,
                    treatment_mask=treatment_mask,
                    outcome_labels=labels,
                    outcome_mask=mask,
                    outcome_positive_label=positive_label,
                    outcome_negative_label=negative_label,
                )
            )

        if dina_nuisance is not None:
            from ..models import dina

            contribution, weights, constant = dina.score(
                y, t, dina_nuisance["a"], dina_nuisance["nu"]
            )
            score_labels, score_mask = _tail_labels(
                contribution, float(self.embedding_config.pseudo_target_quantile)
            )
            weighted_modes = [bool(self.embedding_config.pseudo_target_weighted)]
            if weighted_modes[0] and bool(
                getattr(self.embedding_config, "include_orthogonal_r_score_contrasts", True)
            ):
                weighted_modes.append(False)
            for weighted in weighted_modes:
                specs.append(
                    {
                        "name": "dina_fisher_score" if weighted else "dina_score",
                        "positive_label": "higher_DINA_score",
                        "negative_label": "lower_DINA_score",
                        "labels": score_labels,
                        "mask": score_mask,
                        "sample_weights": weights if weighted else None,
                        "role_hint": "effect_modifier",
                        "metadata": {
                            "contrast_family": "dina_score",
                            "score_formula": "(T-a)*(Y-expit(nu+(T-a)*delta0))",
                            "constant_log_odds_ratio": constant,
                            "effect_scale": "conditional_log_odds_ratio",
                        },
                    }
                )
            return specs
        if str(self.config.outcome_type).lower() == "binary":
            raise ValueError("binary whole-cohort effect contrasts require DINA nuisances")
        multiple_pseudo_targets = len(pseudo_targets) > 1
        for pseudo_name, pseudo_target, t_resid in pseudo_targets:
            pseudo_labels, pseudo_mask = _tail_labels(
                pseudo_target,
                float(self.embedding_config.pseudo_target_quantile),
            )
            pseudo_weights = None
            if bool(self.embedding_config.pseudo_target_weighted):
                pseudo_weights = np.square(np.asarray(t_resid, dtype=float))
            contrast_name = "r_pseudo_target"
            if multiple_pseudo_targets:
                contrast_name = f"r_pseudo_target__{_safe_contrast_suffix(pseudo_name)}"
            specs.append(
                {
                    "name": contrast_name,
                    "positive_label": f"higher_r_pseudo_target:{pseudo_name}",
                    "negative_label": f"lower_r_pseudo_target:{pseudo_name}",
                    "labels": pseudo_labels,
                    "mask": pseudo_mask,
                    "sample_weights": pseudo_weights,
                    "role_hint": "effect_modifier",
                    "metadata": {
                        "contrast_family": "r_pseudo_target",
                        "direction_formula": (
                            "weighted_mean_embedding(high R pseudo-target) - "
                            "weighted_mean_embedding(low R pseudo-target)"
                            if pseudo_weights is not None
                            else "mean_embedding(high R pseudo-target) - "
                            "mean_embedding(low R pseudo-target)"
                        ),
                        "score_formula": "(Y - m_hat) / (T - e_hat)",
                    },
                }
            )
            if bool(
                getattr(
                    self.embedding_config,
                    "include_orthogonal_r_score_contrasts",
                    True,
                )
            ):
                score = np.asarray(pseudo_target, dtype=float) * np.square(
                    np.asarray(t_resid, dtype=float)
                )
                score_labels, score_mask = _tail_labels(
                    score,
                    float(self.embedding_config.pseudo_target_quantile),
                )
                score_name = "orthogonal_r_score"
                if multiple_pseudo_targets:
                    score_name = f"orthogonal_r_score__{_safe_contrast_suffix(pseudo_name)}"
                specs.append(
                    {
                        "name": score_name,
                        "positive_label": f"higher_orthogonal_r_score:{pseudo_name}",
                        "negative_label": f"lower_orthogonal_r_score:{pseudo_name}",
                        "labels": score_labels,
                        "mask": score_mask,
                        "sample_weights": None,
                        "role_hint": "effect_modifier",
                        "metadata": {
                            "contrast_family": "orthogonal_r_score",
                            "direction_formula": (
                                "mean_embedding(high orthogonal R-score) - "
                                "mean_embedding(low orthogonal R-score)"
                            ),
                            "score_formula": (
                                "(Y - m_hat) * (T - e_hat), computed as "
                                "R pseudo-target * (T - e_hat)^2"
                            ),
                        },
                    }
                )
        return specs

    def _cell_contrast_specs(
        self,
        *,
        treatment_labels: np.ndarray,
        treatment_mask: np.ndarray,
        outcome_labels: np.ndarray,
        outcome_mask: np.ndarray,
        outcome_positive_label: str,
        outcome_negative_label: str,
    ) -> List[Dict[str, Any]]:
        """Return within-arm outcome and 2x2 interaction contrast specs."""
        treatment_labels = np.asarray(treatment_labels, dtype=int)
        outcome_labels = np.asarray(outcome_labels, dtype=int)
        base_mask = np.asarray(treatment_mask, dtype=bool) & np.asarray(
            outcome_mask,
            dtype=bool,
        )
        specs: List[Dict[str, Any]] = []
        for treatment_value, treatment_name in [(1, "treated"), (0, "untreated")]:
            arm_mask = base_mask & (treatment_labels == treatment_value)
            specs.append(
                {
                    "name": f"{treatment_name}_outcome",
                    "positive_label": f"{treatment_name}_{outcome_positive_label}",
                    "negative_label": f"{treatment_name}_{outcome_negative_label}",
                    "labels": outcome_labels,
                    "mask": arm_mask,
                    "sample_weights": None,
                    "role_hint": "effect_modifier",
                    "metadata": {
                        "contrast_family": "within_treatment_arm_outcome",
                        "direction_formula": (
                            f"mean_embedding(T={treatment_value}, "
                            f"{outcome_positive_label}) - "
                            f"mean_embedding(T={treatment_value}, "
                            f"{outcome_negative_label})"
                        ),
                    },
                }
            )

        treated_positive = base_mask & (treatment_labels == 1) & (outcome_labels == 1)
        treated_negative = base_mask & (treatment_labels == 1) & (outcome_labels == 0)
        untreated_positive = base_mask & (treatment_labels == 0) & (outcome_labels == 1)
        untreated_negative = base_mask & (treatment_labels == 0) & (outcome_labels == 0)
        interaction_labels = np.zeros(len(treatment_labels), dtype=int)
        interaction_labels[treated_positive | untreated_negative] = 1
        specs.append(
            {
                "name": "treatment_outcome_interaction",
                "positive_label": "higher_treatment_effect_cells",
                "negative_label": "lower_treatment_effect_cells",
                "labels": interaction_labels,
                "mask": base_mask,
                "sample_weights": None,
                "role_hint": "effect_modifier",
                "direction_components": [
                    {
                        "label": f"treated_{outcome_positive_label}",
                        "coefficient": 1.0,
                        "mask": treated_positive,
                    },
                    {
                        "label": f"treated_{outcome_negative_label}",
                        "coefficient": -1.0,
                        "mask": treated_negative,
                    },
                    {
                        "label": f"untreated_{outcome_positive_label}",
                        "coefficient": -1.0,
                        "mask": untreated_positive,
                    },
                    {
                        "label": f"untreated_{outcome_negative_label}",
                        "coefficient": 1.0,
                        "mask": untreated_negative,
                    },
                ],
                "direction_source": "cell_mean_difference_in_differences",
                "metadata": {
                    "contrast_family": "treatment_outcome_cell_interaction",
                    "direction_formula": (
                        "mean(T=1,Y=high/present) - mean(T=1,Y=low/absent) "
                        "- mean(T=0,Y=high/present) + mean(T=0,Y=low/absent)"
                    ),
                    "positive_cell_labels": [
                        f"treated_{outcome_positive_label}",
                        f"untreated_{outcome_negative_label}",
                    ],
                    "negative_cell_labels": [
                        f"treated_{outcome_negative_label}",
                        f"untreated_{outcome_positive_label}",
                    ],
                },
            }
        )
        return specs

    def _build_confounder_vector_contrast(
        self,
        *,
        positions: Sequence[int],
        patient_embeddings: np.ndarray,
        y: np.ndarray,
        t: np.ndarray,
        concept_phrases: Sequence[str],
        concept_embeddings: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        """Build averaged marginal T/Y contrast evidence for confounder discovery."""
        treatment_labels, treatment_mask = _binary_labels(t)
        outcome_labels, outcome_mask, outcome_positive_label, outcome_negative_label = (
            self._outcome_label_spec(y)
        )
        record: Dict[str, Any] = {
            "name": "confounder_vector",
            "positive_label": "treated_or_outcome_positive_side",
            "negative_label": "untreated_or_outcome_negative_side",
            "role_hint": "confounder",
            "contrast_family": "marginal_confounder_average",
            "direction_source": "average_normalized_treatment_and_outcome_mean_differences",
            "direction_formula": (
                "normalize((normalize(mean_embedding(T=1)-mean_embedding(T=0)) + "
                f"normalize(mean_embedding({outcome_positive_label}) - "
                f"mean_embedding({outcome_negative_label}))) / 2)"
            ),
        }
        usable = np.all(np.isfinite(patient_embeddings), axis=1)
        t_direction, t_counts = _binary_mean_difference_direction(
            patient_embeddings,
            treatment_labels,
            treatment_mask & usable,
        )
        y_direction, y_counts = _binary_mean_difference_direction(
            patient_embeddings,
            outcome_labels,
            outcome_mask & usable,
        )
        record["component_counts"] = [
            {"label": "treatment_positive", "n": t_counts[1], "coefficient": 0.5},
            {"label": "treatment_negative", "n": t_counts[0], "coefficient": -0.5},
            {"label": "outcome_positive", "n": y_counts[1], "coefficient": 0.5},
            {"label": "outcome_negative", "n": y_counts[0], "coefficient": -0.5},
        ]
        record["n_positive"] = min(t_counts[1], y_counts[1])
        record["n_negative"] = min(t_counts[0], y_counts[0])
        if t_direction is None or y_direction is None:
            record["retrieval_skipped"] = "too_few_examples_per_component"
            return record
        t_unit = _normalize_vector(t_direction)
        y_unit = _normalize_vector(y_direction)
        direction = 0.5 * t_unit + 0.5 * y_unit
        record["treatment_direction_norm"] = _finite_or_none(float(np.linalg.norm(t_direction)))
        record["outcome_direction_norm"] = _finite_or_none(float(np.linalg.norm(y_direction)))
        record["component_cosine"] = _finite_or_none(float(np.dot(t_unit, y_unit)))
        return self._finalize_direction_record(
            record=record,
            positions=positions,
            direction=direction,
            concept_phrases=concept_phrases,
            concept_embeddings=concept_embeddings,
        )

    def _build_residualized_interaction_contrast(
        self,
        *,
        positions: Sequence[int],
        patient_embeddings: np.ndarray,
        y: np.ndarray,
        t: np.ndarray,
        concept_phrases: Sequence[str],
        concept_embeddings: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        """Build no-nuisance residualized 2x2 interaction evidence."""
        treatment_labels, treatment_mask = _binary_labels(t)
        outcome_labels, outcome_mask, outcome_positive_label, outcome_negative_label = (
            self._outcome_label_spec(y)
        )
        base_mask = (
            np.asarray(treatment_mask, dtype=bool)
            & np.asarray(outcome_mask, dtype=bool)
            & np.all(np.isfinite(patient_embeddings), axis=1)
        )
        treated_positive = base_mask & (treatment_labels == 1) & (outcome_labels == 1)
        treated_negative = base_mask & (treatment_labels == 1) & (outcome_labels == 0)
        untreated_positive = base_mask & (treatment_labels == 0) & (outcome_labels == 1)
        untreated_negative = base_mask & (treatment_labels == 0) & (outcome_labels == 0)
        component_masks = [
            ("treated_outcome_positive", 1.0, treated_positive),
            ("treated_outcome_negative", -1.0, treated_negative),
            ("untreated_outcome_positive", -1.0, untreated_positive),
            ("untreated_outcome_negative", 1.0, untreated_negative),
        ]
        record: Dict[str, Any] = {
            "name": "residualized_treatment_outcome_interaction",
            "positive_label": "treated_outcome_association_exceeds_untreated",
            "negative_label": "untreated_outcome_association_exceeds_treated",
            "role_hint": "effect_modifier",
            "contrast_family": "residualized_treatment_outcome_cell_interaction",
            "direction_source": ("cell_mean_difference_in_differences_residualized_from_marginals"),
            "direction_formula": (
                "residualize(mean(T=1,Y=high/present) - "
                "mean(T=1,Y=low/absent) - mean(T=0,Y=high/present) + "
                "mean(T=0,Y=low/absent), basis=[treatment contrast, "
                "outcome contrast])"
            ),
            "projection_basis": ["treatment", "outcome"],
            "positive_cell_labels": [
                f"treated_{outcome_positive_label}",
                f"untreated_{outcome_negative_label}",
            ],
            "negative_cell_labels": [
                f"treated_{outcome_negative_label}",
                f"untreated_{outcome_positive_label}",
            ],
        }
        raw_direction = np.zeros(patient_embeddings.shape[1], dtype=np.float32)
        component_counts = []
        for label, coefficient, component_mask in component_masks:
            count = int(np.sum(component_mask))
            component_counts.append(
                {
                    "label": label,
                    "coefficient": _finite_or_none(coefficient),
                    "n": count,
                }
            )
            if count < 2:
                record["component_counts"] = component_counts
                record["retrieval_skipped"] = "too_few_examples_per_component"
                return record
            raw_direction += coefficient * np.mean(
                patient_embeddings[component_mask],
                axis=0,
            )
        record["component_counts"] = component_counts
        record["n_positive"] = int(np.sum(treated_positive | untreated_negative))
        record["n_negative"] = int(np.sum(treated_negative | untreated_positive))
        treatment_direction, _ = _binary_mean_difference_direction(
            patient_embeddings,
            treatment_labels,
            treatment_mask & np.all(np.isfinite(patient_embeddings), axis=1),
        )
        outcome_direction, _ = _binary_mean_difference_direction(
            patient_embeddings,
            outcome_labels,
            outcome_mask & np.all(np.isfinite(patient_embeddings), axis=1),
        )
        if treatment_direction is None or outcome_direction is None:
            record["retrieval_skipped"] = "too_few_examples_per_marginal_basis"
            return record
        raw_norm = float(np.linalg.norm(raw_direction))
        residual_direction = _residualize_vector_from_basis(
            raw_direction,
            [treatment_direction, outcome_direction],
        )
        record["raw_interaction_norm"] = _finite_or_none(raw_norm)
        record["residualized_direction_norm"] = _finite_or_none(
            float(np.linalg.norm(residual_direction))
        )
        record["treatment_direction_cosine_before_residualization"] = _finite_or_none(
            float(np.dot(_normalize_vector(raw_direction), _normalize_vector(treatment_direction)))
        )
        record["outcome_direction_cosine_before_residualization"] = _finite_or_none(
            float(np.dot(_normalize_vector(raw_direction), _normalize_vector(outcome_direction)))
        )
        return self._finalize_direction_record(
            record=record,
            positions=positions,
            direction=residual_direction,
            concept_phrases=concept_phrases,
            concept_embeddings=concept_embeddings,
        )

    def _build_cluster_contrast_vectors(
        self,
        *,
        positions: Sequence[int],
        patient_embeddings: np.ndarray,
        y: np.ndarray,
        t: np.ndarray,
        concept_phrases: Sequence[str],
        concept_embeddings: Optional[np.ndarray],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Build SVD components over cluster-local contrast vectors."""
        config = _cluster_local_scientific_config(self.embedding_config)
        usable = np.all(np.isfinite(patient_embeddings), axis=1)
        n_usable = int(np.sum(usable))
        requested_clusters = int(config.requested_cluster_count)
        min_cluster_size = int(config.minimum_cluster_size)
        n_clusters = requested_clusters
        summary: Dict[str, Any] = {
            "enabled": True,
            "execution_mode": "cluster_only_native_geometry_v1",
            "n_clusters_requested": requested_clusters,
            "n_clusters": n_clusters,
            "cluster_count_policy": config.cluster_count_policy,
            "min_cluster_size": min_cluster_size,
            "min_group_size": int(config.minimum_group_size),
            "min_cell_size": int(config.minimum_cell_size),
            "max_components": int(config.maximum_components_per_family),
            "n_usable_patients": n_usable,
            "scientific_configuration": config.as_dict(),
        }
        if n_usable < requested_clusters * min_cluster_size:
            raise ClusterLocalEmbeddingFeasibilityError(
                "cluster-local exact configured cluster count is infeasible for "
                "the minimum support requirement",
                summary=summary,
            )

        _ordered_rows, canonical_group_seed = (
            self._cluster_fit_context(positions)
        )
        kmeans_parameters = _embedding_cluster_kmeans_parameters(
            self.embedding_config,
            n_usable=n_usable,
            canonical_group_seed=canonical_group_seed,
            n_clusters=n_clusters,
        )
        kmeans = MiniBatchKMeans(**kmeans_parameters)
        cluster_labels = np.full(len(patient_embeddings), -1, dtype=int)
        cluster_labels[usable] = kmeans.fit_predict(patient_embeddings[usable])

        cluster_counts = np.bincount(cluster_labels[usable], minlength=n_clusters).astype(int)
        summary["n_clusters"] = int(n_clusters)
        summary["cluster_counts"] = [int(item) for item in cluster_counts]
        summary["actual_kmeans_parameters"] = copy.deepcopy(kmeans_parameters)
        summary["actual_canonical_group_seed"] = canonical_group_seed
        summary["actual_batch_size"] = int(kmeans_parameters["batch_size"])
        records: List[Dict[str, Any]] = []
        treatment_items = self._cluster_treatment_local_contrasts(
            patient_embeddings=patient_embeddings,
            t=t,
            cluster_labels=cluster_labels,
            n_clusters=n_clusters,
            cluster_counts=cluster_counts,
        )
        interaction_items = self._cluster_residualized_interaction_local_contrasts(
            patient_embeddings=patient_embeddings,
            y=y,
            t=t,
            cluster_labels=cluster_labels,
            n_clusters=n_clusters,
            cluster_counts=cluster_counts,
        )
        summary["usable_treatment_local_contrasts"] = len(treatment_items)
        summary["usable_residualized_interaction_local_contrasts"] = len(interaction_items)
        if (
            len(treatment_items)
            < int(config.minimum_distinct_local_clusters_per_family)
            or len(interaction_items)
            < int(config.minimum_distinct_local_clusters_per_family)
        ):
            raise ClusterLocalEmbeddingFeasibilityError(
                "cluster-local evidence lacks the configured number of "
                "independently supported local contrasts",
                summary=summary,
            )
        records.extend(
            self._cluster_component_records(
                family_key="treatment",
                items=treatment_items,
                positions=positions,
                role_hint="confounder",
                positive_label="cluster_local_treated_side",
                negative_label="cluster_local_untreated_side",
                contrast_family="cluster_local_treatment_contrast_basis",
                direction_source="svd_of_cluster_local_treatment_mean_differences",
                direction_formula=(
                    "top right singular vectors of rows "
                    "sqrt(n_cluster) * normalize(mean_embedding(T=1,S=s) - "
                    "mean_embedding(T=0,S=s))"
                ),
                concept_phrases=concept_phrases,
                concept_embeddings=concept_embeddings,
            )
        )
        records.extend(
            self._cluster_component_records(
                family_key="residualized_interaction",
                items=interaction_items,
                positions=positions,
                role_hint="effect_modifier",
                positive_label="cluster_local_treated_outcome_association_exceeds_untreated",
                negative_label="cluster_local_untreated_outcome_association_exceeds_treated",
                contrast_family="cluster_local_residualized_interaction_contrast_basis",
                direction_source=(
                    "svd_of_cluster_local_treatment_outcome_interactions_residualized_from_marginals"
                ),
                direction_formula=(
                    "top right singular vectors of rows sqrt(n_cluster) * "
                    "normalize(residualize(local 2x2 treatment-outcome interaction, "
                    "basis=[local treatment contrast, local outcome contrast]))"
                ),
                concept_phrases=concept_phrases,
                concept_embeddings=concept_embeddings,
            )
        )
        summary["n_cluster_contrast_components"] = len(records)
        expected_minimum = (
            2 * int(config.minimum_numerical_rank_per_family)
        )
        if len(records) < expected_minimum:
            raise ClusterLocalEmbeddingFeasibilityError(
                "cluster-local evidence emitted fewer components than its rank contract",
                summary=summary,
            )
        return records, summary

    def _cluster_treatment_local_contrasts(
        self,
        *,
        patient_embeddings: np.ndarray,
        t: np.ndarray,
        cluster_labels: np.ndarray,
        n_clusters: int,
        cluster_counts: np.ndarray,
    ) -> List[Dict[str, Any]]:
        labels, treatment_mask = _binary_labels(t)
        config = _cluster_local_scientific_config(self.embedding_config)
        min_cluster_size = int(config.minimum_cluster_size)
        min_group_size = int(config.minimum_group_size)
        items: List[Dict[str, Any]] = []
        for cluster_id in range(n_clusters):
            if int(cluster_counts[cluster_id]) < min_cluster_size:
                continue
            cluster_mask = cluster_labels == cluster_id
            local_mask = cluster_mask & treatment_mask
            positive_mask = local_mask & (labels == 1)
            negative_mask = local_mask & (labels == 0)
            n_positive = int(np.sum(positive_mask))
            n_negative = int(np.sum(negative_mask))
            if n_positive < min_group_size or n_negative < min_group_size:
                continue
            direction = (
                np.mean(patient_embeddings[positive_mask], axis=0)
                - np.mean(patient_embeddings[negative_mask], axis=0)
            ).astype(np.dtype(config.computation_dtype), copy=False)
            direction_norm = float(np.linalg.norm(direction))
            if not np.isfinite(direction_norm) or direction_norm <= 0.0:
                continue
            items.append(
                {
                    "cluster_id": int(cluster_id),
                    "n_cluster": int(cluster_counts[cluster_id]),
                    "n_positive": n_positive,
                    "n_negative": n_negative,
                    "local_direction_norm": direction_norm,
                    "direction": direction,
                }
            )
        return items

    def _cluster_residualized_interaction_local_contrasts(
        self,
        *,
        patient_embeddings: np.ndarray,
        y: np.ndarray,
        t: np.ndarray,
        cluster_labels: np.ndarray,
        n_clusters: int,
        cluster_counts: np.ndarray,
    ) -> List[Dict[str, Any]]:
        treatment_labels, treatment_mask = _binary_labels(t)
        outcome_labels, outcome_mask, _positive_label, _negative_label = self._outcome_label_spec(y)
        config = _cluster_local_scientific_config(self.embedding_config)
        min_cluster_size = int(config.minimum_cluster_size)
        min_cell_size = int(config.minimum_cell_size)
        items: List[Dict[str, Any]] = []
        for cluster_id in range(n_clusters):
            if int(cluster_counts[cluster_id]) < min_cluster_size:
                continue
            cluster_mask = cluster_labels == cluster_id
            base_mask = cluster_mask & treatment_mask & outcome_mask
            treated_positive = base_mask & (treatment_labels == 1) & (outcome_labels == 1)
            treated_negative = base_mask & (treatment_labels == 1) & (outcome_labels == 0)
            untreated_positive = base_mask & (treatment_labels == 0) & (outcome_labels == 1)
            untreated_negative = base_mask & (treatment_labels == 0) & (outcome_labels == 0)
            cell_counts = {
                "treated_outcome_positive": int(np.sum(treated_positive)),
                "treated_outcome_negative": int(np.sum(treated_negative)),
                "untreated_outcome_positive": int(np.sum(untreated_positive)),
                "untreated_outcome_negative": int(np.sum(untreated_negative)),
            }
            if min(cell_counts.values()) < min_cell_size:
                continue
            raw_direction = (
                np.mean(patient_embeddings[treated_positive], axis=0)
                - np.mean(patient_embeddings[treated_negative], axis=0)
                - np.mean(patient_embeddings[untreated_positive], axis=0)
                + np.mean(patient_embeddings[untreated_negative], axis=0)
            ).astype(np.dtype(config.computation_dtype), copy=False)
            local_treatment_direction, _ = _binary_mean_difference_direction(
                patient_embeddings,
                treatment_labels,
                cluster_mask & treatment_mask,
            )
            local_outcome_direction, _ = _binary_mean_difference_direction(
                patient_embeddings,
                outcome_labels,
                cluster_mask & outcome_mask,
            )
            if local_treatment_direction is None or local_outcome_direction is None:
                continue
            direction = _residualize_vector_from_basis(
                raw_direction,
                [local_treatment_direction, local_outcome_direction],
            )
            direction_norm = float(np.linalg.norm(direction))
            if not np.isfinite(direction_norm) or direction_norm <= 0.0:
                continue
            items.append(
                {
                    "cluster_id": int(cluster_id),
                    "n_cluster": int(cluster_counts[cluster_id]),
                    "n_positive": int(np.sum(treated_positive | untreated_negative)),
                    "n_negative": int(np.sum(treated_negative | untreated_positive)),
                    "cell_counts": cell_counts,
                    "local_direction_norm": direction_norm,
                    "raw_direction_norm": float(np.linalg.norm(raw_direction)),
                    "direction": direction,
                }
            )
        return items

    def _cluster_component_records(
        self,
        *,
        family_key: str,
        items: Sequence[Dict[str, Any]],
        positions: Sequence[int],
        role_hint: str,
        positive_label: str,
        negative_label: str,
        contrast_family: str,
        direction_source: str,
        direction_formula: str,
        concept_phrases: Sequence[str],
        concept_embeddings: Optional[np.ndarray],
    ) -> List[Dict[str, Any]]:
        config = _cluster_local_scientific_config(self.embedding_config)
        if len(items) < int(config.minimum_distinct_local_clusters_per_family):
            raise ValueError(
                f"cluster-local {family_key} family lacks configured local support"
            )
        dtype = np.dtype(config.computation_dtype)
        weighted_rows = []
        for item in items:
            direction = _normalize_rows_configured(
                np.asarray(item["direction"], dtype=dtype).reshape(1, -1),
                normalize=True,
                epsilon=float(config.normalization_epsilon),
                zero_vector_policy=config.zero_vector_policy,
                dtype=config.computation_dtype,
            )[0]
            weighted_rows.append(direction * np.sqrt(float(item["n_cluster"])))
        matrix = np.vstack(weighted_rows).astype(dtype, copy=False)
        _left, singular_values, components = np.linalg.svd(
            matrix,
            full_matrices=bool(config.svd_full_matrices),
            compute_uv=bool(config.svd_compute_uv),
            hermitian=bool(config.svd_hermitian),
        )
        components = _canonicalize_svd_component_signs(
            components,
            policy=config.svd_sign_canonicalization_policy,
        )
        total_energy = float(np.sum(np.square(singular_values)))
        rank_dtype = np.dtype(config.svd_rank_tolerance_dtype)
        rank_tolerance = (
            float(config.svd_rank_tolerance_multiplier)
            * float(np.finfo(rank_dtype).eps)
            * max(matrix.shape)
            * float(singular_values[0])
        )
        numerical_rank = int(np.sum(singular_values > rank_tolerance))
        if numerical_rank < int(config.minimum_numerical_rank_per_family):
            raise ValueError(
                f"cluster-local {family_key} SVD lacks configured numerical rank"
            )
        max_components = min(
            int(config.maximum_components_per_family),
            numerical_rank,
        )
        records: List[Dict[str, Any]] = []
        for component_index in range(max_components):
            singular_value = float(singular_values[component_index])
            if not np.isfinite(singular_value) or singular_value <= 0.0:
                continue
            direction = _normalize_rows_configured(
                np.asarray(components[component_index]).reshape(1, -1),
                normalize=True,
                epsilon=float(config.normalization_epsilon),
                zero_vector_policy=config.zero_vector_policy,
                dtype=config.computation_dtype,
            )[0]
            loadings = np.asarray(matrix @ direction, dtype=float)
            top_indices = np.argsort(np.abs(loadings))[::-1]
            capacity = config.loading_evidence_capacity
            if capacity is not None and len(top_indices) > int(capacity):
                raise ValueError(
                    "cluster-local loading evidence exceeds its configured "
                    "capacity; truncation is forbidden"
                )
            loading_rows = []
            for item_index in top_indices:
                item = items[int(item_index)]
                row = {
                    "cluster_id": int(item["cluster_id"]),
                    "loading": _finite_or_none(float(loadings[item_index])),
                    "abs_loading": _finite_or_none(abs(float(loadings[item_index]))),
                    "n_cluster": int(item["n_cluster"]),
                    "n_positive": int(item["n_positive"]),
                    "n_negative": int(item["n_negative"]),
                    "local_direction_norm": _finite_or_none(float(item["local_direction_norm"])),
                }
                if "cell_counts" in item:
                    row["cell_counts"] = copy.deepcopy(item["cell_counts"])
                if "raw_direction_norm" in item:
                    row["raw_direction_norm"] = _finite_or_none(float(item["raw_direction_norm"]))
                loading_rows.append(row)
            record: Dict[str, Any] = {
                "name": f"cluster_{family_key}_pc{component_index + 1}",
                "positive_label": f"{positive_label}_pc{component_index + 1}",
                "negative_label": f"{negative_label}_pc{component_index + 1}",
                "role_hint": role_hint,
                "contrast_family": contrast_family,
                "direction_source": direction_source,
                "direction_formula": direction_formula,
                "n_positive": int(sum(int(item["n_positive"]) for item in items)),
                "n_negative": int(sum(int(item["n_negative"]) for item in items)),
                "local_contrast_count": int(len(items)),
                "cluster_component_index": int(component_index + 1),
                "cluster_component_singular_value": _finite_or_none(singular_value),
                "cluster_component_explained_energy": _finite_or_none(
                    float(singular_value**2 / total_energy) if total_energy > 0.0 else np.nan
                ),
                "cluster_component_sign_policy": (
                    config.svd_sign_canonicalization_policy
                ),
                "cluster_component_rank_tolerance": _finite_or_none(
                    rank_tolerance
                ),
                "cluster_component_numerical_rank": numerical_rank,
                "cluster_component_loadings": loading_rows,
                "mean_difference_norm": None,
            }
            records.append(
                self._finalize_direction_record(
                    record=record,
                    positions=positions,
                    direction=direction,
                    concept_phrases=concept_phrases,
                    concept_embeddings=concept_embeddings,
                )
            )
        return records

    def _outcome_label_spec(
        self,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, str, str]:
        if str(self.config.outcome_type).lower() == "continuous":
            labels, mask = _tail_labels(
                y,
                float(self.embedding_config.pseudo_target_quantile),
            )
            return labels, mask, "higher_outcome", "lower_outcome"
        labels, mask = _binary_labels(y)
        return labels, mask, "outcome_present", "outcome_absent"

    def _build_one_contrast(
        self,
        *,
        positions: Sequence[int],
        patient_embeddings: np.ndarray,
        concept_phrases: Sequence[str],
        concept_embeddings: Optional[np.ndarray],
        name: str,
        positive_label: str,
        negative_label: str,
        labels: np.ndarray,
        mask: np.ndarray,
        sample_weights: Optional[np.ndarray],
        role_hint: str,
        direction_components: Optional[Sequence[Dict[str, Any]]] = None,
        direction_source: str = "mean_difference",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        labels = np.asarray(labels, dtype=int)
        mask = np.asarray(mask, dtype=bool)
        usable = mask & np.all(np.isfinite(patient_embeddings), axis=1)
        pos_mask = usable & (labels == 1)
        neg_mask = usable & (labels == 0)
        n_pos = int(np.sum(pos_mask))
        n_neg = int(np.sum(neg_mask))
        record: Dict[str, Any] = {
            "name": name,
            "positive_label": positive_label,
            "negative_label": negative_label,
            "role_hint": role_hint,
            "n_positive": n_pos,
            "n_negative": n_neg,
            "min_probe_auc": float(self.embedding_config.min_probe_auc),
        }
        if metadata:
            record.update(copy.deepcopy(metadata))
        if n_pos < 2 or n_neg < 2:
            record["retrieval_skipped"] = "too_few_examples_per_group"
            return record

        weights = None
        if sample_weights is not None:
            weights = np.asarray(sample_weights, dtype=float)
            weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)

        if direction_components:
            component_counts = []
            mean_direction = np.zeros(patient_embeddings.shape[1], dtype=np.float32)
            for component in direction_components:
                component_mask = usable & np.asarray(component["mask"], dtype=bool)
                component_count = int(np.sum(component_mask))
                coefficient = float(component.get("coefficient", 1.0))
                component_counts.append(
                    {
                        "label": str(component.get("label", "")),
                        "coefficient": _finite_or_none(coefficient),
                        "n": component_count,
                    }
                )
                if component_count < 2:
                    record["component_counts"] = component_counts
                    record["retrieval_skipped"] = "too_few_examples_per_component"
                    return record
                mean_direction += coefficient * _weighted_mean(
                    patient_embeddings[component_mask],
                    _subset(weights, component_mask),
                )
            record["component_counts"] = component_counts
        else:
            mean_direction = _weighted_mean(
                patient_embeddings[pos_mask],
                _subset(weights, pos_mask),
            )
            mean_direction -= _weighted_mean(
                patient_embeddings[neg_mask],
                _subset(weights, neg_mask),
            )
        mean_norm = float(np.linalg.norm(mean_direction))
        record["mean_difference_norm"] = _finite_or_none(mean_norm)

        probe_auc, _probe_direction = _linear_probe_direction(
            patient_embeddings[usable],
            labels[usable],
        )
        record["probe_auc"] = _finite_or_none(probe_auc)

        min_auc = float(self.embedding_config.min_probe_auc)
        if min_auc > 0.0 and (
            probe_auc is None or not np.isfinite(probe_auc) or probe_auc < min_auc
        ):
            record["retrieval_skipped"] = "probe_auc_below_threshold"
            return record
        if mean_norm <= 0.0:
            record["retrieval_skipped"] = "zero_mean_difference_direction"
            return record

        direction = _normalize_vector(mean_direction)
        record["direction_source"] = direction_source
        record["probe_auc_role"] = "diagnostic_gate_only" if min_auc > 0.0 else "diagnostic_only"
        return self._finalize_direction_record(
            record=record,
            positions=positions,
            direction=direction,
            concept_phrases=concept_phrases,
            concept_embeddings=concept_embeddings,
        )

    def _finalize_direction_record(
        self,
        *,
        record: Dict[str, Any],
        positions: Sequence[int],
        direction: np.ndarray,
        concept_phrases: Sequence[str],
        concept_embeddings: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        direction_norm = float(np.linalg.norm(direction))
        record["mean_difference_norm"] = record.get(
            "mean_difference_norm",
            _finite_or_none(direction_norm),
        )
        if not np.isfinite(direction_norm) or direction_norm <= 0.0:
            record["retrieval_skipped"] = "zero_direction"
            return record
        direction = _normalize_vector(direction)
        record["direction_norm"] = _finite_or_none(float(np.linalg.norm(direction)))
        record["positive_aligned_chunks"] = self._retrieve_chunks(
            positions,
            direction,
            descending=True,
        )
        record["negative_aligned_chunks"] = self._retrieve_chunks(
            positions,
            direction,
            descending=False,
        )
        if bool(self.embedding_config.retrieval_tfidf_enabled):
            record["tfidf_retrieval_terms"] = _retrieval_tfidf_terms(
                record["positive_aligned_chunks"],
                record["negative_aligned_chunks"],
                ngram_range=(
                    int(self.embedding_config.retrieval_tfidf_ngram_range_min),
                    int(self.embedding_config.retrieval_tfidf_ngram_range_max),
                ),
                max_features=int(self.embedding_config.retrieval_tfidf_max_features),
                top_terms_per_side=int(
                    self.embedding_config.retrieval_tfidf_top_terms_per_side
                ),
            )
            record["tfidf_retrieval"] = {
                "enabled": True,
                "fit_unit": "retrieved_fold_local_chunk",
                "positive_documents": len(record["positive_aligned_chunks"]),
                "negative_documents": len(record["negative_aligned_chunks"]),
                "ngram_range": [
                    int(self.embedding_config.retrieval_tfidf_ngram_range_min),
                    int(self.embedding_config.retrieval_tfidf_ngram_range_max),
                ],
                "max_features": int(self.embedding_config.retrieval_tfidf_max_features),
                "top_terms_per_side": int(
                    self.embedding_config.retrieval_tfidf_top_terms_per_side
                ),
                "selected_terms": len(record["tfidf_retrieval_terms"]),
            }
        if self._external_corpora:
            record["positive_external_chunks"] = self._retrieve_external_chunks(
                direction,
                descending=True,
            )
            record["negative_external_chunks"] = self._retrieve_external_chunks(
                direction,
                descending=False,
            )
        record["concept_probe_scores"] = self._score_concepts(
            concept_phrases,
            concept_embeddings,
            direction,
        )
        return record

    def _retrieve_chunks(
        self,
        positions: Sequence[int],
        direction: np.ndarray,
        *,
        descending: bool,
    ) -> List[Dict[str, Any]]:
        candidates: List[Tuple[float, int, int, str]] = []
        for position in positions:
            chunks = self._chunks_by_position[int(position)]
            chunk_embeddings = self._chunk_matrix(int(position))
            if chunk_embeddings.size == 0:
                continue
            scores = np.asarray(chunk_embeddings @ direction, dtype=float)
            for chunk_idx, score in enumerate(scores):
                if not np.isfinite(score):
                    continue
                text = chunks[chunk_idx] if chunk_idx < len(chunks) else ""
                if not _informative_chunk_text(text):
                    continue
                candidates.append((float(score), int(position), int(chunk_idx), text))

        candidates.sort(key=lambda item: item[0], reverse=descending)
        rows: List[Dict[str, Any]] = []
        per_patient_counts: Dict[Any, int] = {}
        max_per_patient = int(self.embedding_config.max_chunks_per_patient)
        for score, position, chunk_idx, text in candidates:
            row_id = self._row_ids[position]
            key = _row_key(row_id)
            if per_patient_counts.get(key, 0) >= max_per_patient:
                continue
            per_patient_counts[key] = per_patient_counts.get(key, 0) + 1
            rows.append(
                {
                    "row_id": _jsonable_scalar(row_id),
                    "chunk_index": int(chunk_idx),
                    "score": _finite_or_none(score),
                    "text": text,
                }
            )
            if len(rows) >= int(self.embedding_config.top_k_chunks_per_tail):
                break
        return rows

    def _retrieve_external_chunks(
        self,
        direction: np.ndarray,
        *,
        descending: bool,
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        limit = int(self.embedding_config.external_top_k_chunks_per_tail)
        for corpus in self._external_corpora:
            embeddings = corpus["embeddings"]
            if embeddings.ndim != 2 or embeddings.shape[1] != len(direction):
                logger.warning(
                    "Skipping external corpus %s due to embedding dimension %s; "
                    "contrast direction has dimension %s",
                    corpus["name"],
                    embeddings.shape[1] if embeddings.ndim == 2 else embeddings.shape,
                    len(direction),
                )
                continue
            candidates = _top_scored_flat_indices(
                embeddings,
                direction,
                limit=limit,
                descending=descending,
            )
            offsets = corpus["offsets"]
            chunks_by_sample = corpus["chunks_by_sample"]
            row_metadata = corpus["row_metadata"]
            for score, flat_index in candidates:
                sample_index = int(np.searchsorted(offsets, flat_index, side="right") - 1)
                if sample_index < 0 or sample_index >= len(chunks_by_sample):
                    continue
                chunk_index = int(flat_index - int(offsets[sample_index]))
                chunks = chunks_by_sample[sample_index]
                text = chunks[chunk_index] if chunk_index < len(chunks) else ""
                if not _informative_chunk_text(text):
                    continue
                metadata = (
                    row_metadata[sample_index]
                    if sample_index < len(row_metadata)
                    else {"row_index": sample_index}
                )
                rows.append(
                    {
                        "corpus": str(corpus["name"]),
                        "cache_path": str(corpus["cache_path"]),
                        "row_index": int(sample_index),
                        "chunk_index": int(chunk_index),
                        "score": _finite_or_none(float(score)),
                        "text": text,
                        "metadata": _jsonable_value(metadata),
                    }
                )
        rows.sort(key=lambda item: float(item.get("score") or 0.0), reverse=descending)
        return rows[:limit]

    def _concept_phrases(self, importance: Dict[str, Any]) -> List[str]:
        phrases: List[str] = list(self.embedding_config.concept_phrases)
        if bool(self.embedding_config.include_bow_phrases_as_concepts):
            for row in importance.get("phrase_features", []) or []:
                phrase = str(row.get("feature", "")).strip()
                if phrase:
                    phrases.append(phrase)
        deduped = list(dict.fromkeys(phrases))
        limit = int(self.embedding_config.max_concept_phrases)
        return deduped[:limit] if limit > 0 else []

    def _score_concepts(
        self,
        concept_phrases: Sequence[str],
        concept_embeddings: Optional[np.ndarray],
        direction: np.ndarray,
    ) -> List[Dict[str, Any]]:
        if not concept_phrases or concept_embeddings is None:
            return []
        embeddings = _normalize_rows(concept_embeddings)
        scores = np.asarray(embeddings @ direction, dtype=float)
        order = np.argsort(np.abs(scores))[::-1]
        rows = []
        for idx in order[: int(self.embedding_config.concept_probe_top_k)]:
            rows.append(
                {
                    "concept": str(concept_phrases[int(idx)]),
                    "score": _finite_or_none(float(scores[int(idx)])),
                }
            )
        return rows

    def _encode_concepts(self, phrases: Sequence[str]) -> Optional[np.ndarray]:
        phrase_list = [str(phrase) for phrase in phrases]
        if self.embedding_provider is not None:
            embeddings = self._encode_with_provider(phrase_list)
        else:
            cached = self._load_concept_embedding_cache(phrase_list)
            if cached is not None:
                return cached
            if self._chunk_cache_reused:
                self._concept_probe_skip_reason = "concept_phrase_cache_miss_on_warm_chunk_cache"
                logger.info(
                    "Skipping embedding concept probes because chunk embeddings "
                    "were reused from cache and concept phrase embeddings are not "
                    "cached; avoiding a sentence-transformer load."
                )
                return None
            try:
                encoder = load_sentence_transformer(
                    str(self.embedding_config.model_name),
                    device=_torch_device_or_none(self.embedding_config.device),
                    max_seq_length=self.embedding_config.max_seq_length,
                )
                embeddings = encoder.encode(
                    phrase_list,
                    batch_size=max(
                        1,
                        min(int(self.embedding_config.batch_size), len(phrase_list)),
                    ),
                    convert_to_numpy=True,
                    normalize_embeddings=bool(self.embedding_config.normalize_embeddings),
                    show_progress_bar=False,
                )
            finally:
                _release_sentence_transformer_model(str(self.embedding_config.model_name))
        matrix = _coerce_embedding_matrix(embeddings, expected_rows=len(phrase_list))
        if self.embedding_provider is None:
            self._write_concept_embedding_cache(phrase_list, matrix)
        return matrix

    def _concept_embedding_cache_path(self, phrases: Sequence[str]) -> Optional[Path]:
        if self._cache_dir is None:
            return None
        payload = {
            "model_name": str(self.embedding_config.model_name),
            "normalize_embeddings": bool(self.embedding_config.normalize_embeddings),
            "max_seq_length": self.embedding_config.max_seq_length,
            "phrases": [str(phrase) for phrase in phrases],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()[:16]
        return self._cache_dir / f"embedding_concept_phrases_{digest}.npz"

    def _load_concept_embedding_cache(
        self,
        phrases: Sequence[str],
    ) -> Optional[np.ndarray]:
        cache_path = self._concept_embedding_cache_path(phrases)
        if cache_path is None or not cache_path.exists():
            return None
        try:
            with np.load(str(cache_path), allow_pickle=False) as payload:
                cached_phrases = [str(item) for item in payload["phrases"].tolist()]
                if cached_phrases != [str(phrase) for phrase in phrases]:
                    return None
                embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
            logger.info("Reusing embedding concept phrase cache: %s", cache_path)
            return _coerce_embedding_matrix(embeddings, expected_rows=len(phrases))
        except Exception as exc:
            logger.warning(
                "Failed to read embedding concept phrase cache %s: %s",
                cache_path,
                exc,
            )
            return None

    def _write_concept_embedding_cache(
        self,
        phrases: Sequence[str],
        embeddings: np.ndarray,
    ) -> None:
        cache_path = self._concept_embedding_cache_path(phrases)
        if cache_path is None:
            return
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        try:
            with open(tmp_path, "wb") as f:
                np.savez_compressed(
                    f,
                    phrases=np.asarray([str(phrase) for phrase in phrases], dtype=str),
                    embeddings=np.asarray(embeddings, dtype=np.float32),
                )
            tmp_path.replace(cache_path)
            logger.info("Wrote embedding concept phrase cache: %s", cache_path)
        except Exception as exc:
            logger.warning(
                "Failed to write embedding concept phrase cache %s: %s",
                cache_path,
                exc,
            )
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass

    def _encode_with_provider(self, texts: Sequence[str]) -> np.ndarray:
        provider = self.embedding_provider
        if hasattr(provider, "encode_chunks"):
            return provider.encode_chunks(list(texts))
        if hasattr(provider, "encode_texts"):
            return provider.encode_texts(list(texts))
        if hasattr(provider, "encode"):
            return provider.encode(list(texts))
        raise TypeError("embedding_provider must implement encode_chunks, encode_texts, or encode")


def redact_embedding_contrast_evidence(evidence: Any) -> Any:
    """Return a copy of embedding evidence with retrieved raw text removed."""
    if not evidence:
        return evidence

    def redact(value: Any) -> Any:
        if isinstance(value, dict):
            redacted: Dict[str, Any] = {}
            for key, item in value.items():
                if key == "text":
                    redacted[key] = None
                    redacted["text_redacted"] = True
                else:
                    redacted[key] = redact(item)
            return redacted
        if isinstance(value, list):
            return [redact(item) for item in value]
        return value

    return redact(copy.deepcopy(evidence))


def _informative_chunk_text(text: str) -> bool:
    """Return True when a retrieved chunk has enough clinical text to show."""
    compact = "".join(ch for ch in str(text or "") if ch.isalnum())
    return len(compact) >= 12


def _named_pseudo_targets(
    pseudo_target: Any,
    t_resid: Any,
    names: Optional[Sequence[str]],
) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    if pseudo_target is None or t_resid is None:
        return []
    targets = _as_target_list(pseudo_target)
    residuals = _as_target_list(t_resid)
    if not targets or not residuals:
        return []
    if len(residuals) == 1 and len(targets) > 1:
        residuals = residuals * len(targets)
    if len(targets) != len(residuals):
        raise ValueError("pseudo_target and t_resid must have matching view counts")
    if names is None:
        target_names = [
            "r_pseudo_target" if len(targets) == 1 else f"view_{idx}"
            for idx in range(1, len(targets) + 1)
        ]
    else:
        target_names = [str(name) for name in names]
    if len(target_names) != len(targets):
        raise ValueError("pseudo_target_names must match pseudo_target view count")
    return [
        (
            target_names[idx],
            np.asarray(targets[idx], dtype=float),
            np.asarray(residuals[idx], dtype=float),
        )
        for idx in range(len(targets))
    ]


def _as_target_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return [value]
    if isinstance(value, (list, tuple)):
        if not value:
            return []
        first = np.asarray(value[0])
        if first.ndim > 0:
            return list(value)
    return [value]


def _safe_contrast_suffix(value: Any) -> str:
    suffix = []
    for ch in str(value).strip().lower():
        if ch.isalnum():
            suffix.append(ch)
        elif suffix and suffix[-1] != "_":
            suffix.append("_")
    text = "".join(suffix).strip("_")
    return text or "view"


def _linear_probe_direction(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> Tuple[Optional[float], Optional[np.ndarray]]:
    labels = np.asarray(labels, dtype=int)
    if len(np.unique(labels)) < 2:
        return None, None
    class_counts = np.bincount(labels, minlength=2)
    min_class = int(np.min(class_counts[:2]))
    if min_class < 2:
        return None, None

    auc = None
    folds = min(5, min_class)
    if folds >= 2:
        oof = np.full(len(labels), np.nan, dtype=float)
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=83_001)
        for train_idx, heldout_idx in splitter.split(embeddings, labels):
            model = LogisticRegression(
                C=1.0,
                solver="liblinear",
                max_iter=1000,
            )
            model.fit(embeddings[train_idx], labels[train_idx])
            oof[heldout_idx] = model.predict_proba(embeddings[heldout_idx])[:, 1]
        if np.all(np.isfinite(oof)):
            try:
                auc = float(roc_auc_score(labels, oof))
            except ValueError:
                auc = None

    model = LogisticRegression(C=1.0, solver="liblinear", max_iter=1000)
    model.fit(embeddings, labels)
    coef = np.asarray(model.coef_, dtype=np.float32).reshape(-1)
    if not np.all(np.isfinite(coef)) or np.linalg.norm(coef) <= 0.0:
        return auc, None
    return auc, coef


def _binary_labels(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    labels = (values >= 0.5).astype(int)
    mask = np.isfinite(values)
    if len(np.unique(labels[mask])) < 2:
        mask = np.zeros_like(mask, dtype=bool)
    return labels, mask


def _tail_labels(values: np.ndarray, quantile: float) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    labels = np.zeros(len(values), dtype=int)
    mask = np.zeros(len(values), dtype=bool)
    if np.sum(finite) < 4:
        return labels, mask
    low, high = np.nanquantile(values[finite], [quantile, 1.0 - quantile])
    low_mask = finite & (values <= low)
    high_mask = finite & (values >= high)
    overlap = low_mask & high_mask
    low_mask[overlap] = False
    high_mask[overlap] = False
    labels[high_mask] = 1
    mask = low_mask | high_mask
    if np.sum(low_mask) < 2 or np.sum(high_mask) < 2:
        mask[:] = False
    return labels, mask


def _residualize_embeddings(
    embeddings: np.ndarray,
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> np.ndarray:
    usable_columns = [col for col in columns if col in frame.columns]
    if not usable_columns:
        return embeddings
    covariates = pd.get_dummies(
        frame[usable_columns].reset_index(drop=True),
        dummy_na=True,
        drop_first=True,
    )
    if covariates.shape[1] == 0:
        return embeddings
    x = covariates.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if x.ndim != 2 or x.shape[0] != embeddings.shape[0]:
        return embeddings
    x = np.column_stack([np.ones(x.shape[0], dtype=float), x])
    try:
        beta, *_ = np.linalg.lstsq(x, embeddings, rcond=None)
    except np.linalg.LinAlgError:
        logger.warning("Embedding residualization failed; using raw embeddings")
        return embeddings
    return embeddings - x @ beta


def _weighted_mean(values: np.ndarray, weights: Optional[np.ndarray]) -> np.ndarray:
    if weights is None or len(weights) != len(values) or np.sum(weights) <= 0.0:
        return np.mean(values, axis=0)
    weights = np.asarray(weights, dtype=float)
    weights = weights / np.sum(weights)
    return np.sum(values * weights[:, None], axis=0)


def _binary_mean_difference_direction(
    embeddings: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
) -> Tuple[Optional[np.ndarray], Dict[int, int]]:
    labels = np.asarray(labels, dtype=int)
    mask = np.asarray(mask, dtype=bool)
    pos_mask = mask & (labels == 1)
    neg_mask = mask & (labels == 0)
    counts = {1: int(np.sum(pos_mask)), 0: int(np.sum(neg_mask))}
    if counts[1] < 2 or counts[0] < 2:
        return None, counts
    return (
        np.mean(embeddings[pos_mask], axis=0) - np.mean(embeddings[neg_mask], axis=0),
        counts,
    )


def _residualize_vector_from_basis(
    vector: np.ndarray,
    basis: Sequence[np.ndarray],
) -> np.ndarray:
    columns = []
    for item in basis:
        item = np.asarray(item, dtype=np.float32).reshape(-1)
        if np.all(np.isfinite(item)) and np.linalg.norm(item) > 0.0:
            columns.append(_normalize_vector(item))
    if not columns:
        return np.asarray(vector, dtype=np.float32)
    design = np.vstack(columns).T.astype(np.float64, copy=False)
    try:
        coef, *_ = np.linalg.lstsq(
            design,
            np.asarray(vector, dtype=np.float64).reshape(-1),
            rcond=None,
        )
    except np.linalg.LinAlgError:
        logger.warning("Vector residualization failed; using raw direction")
        return np.asarray(vector, dtype=np.float32)
    residual = np.asarray(vector, dtype=np.float64).reshape(-1) - design @ coef
    return residual.astype(np.float32, copy=False)


def _subset(values: Optional[np.ndarray], mask: np.ndarray) -> Optional[np.ndarray]:
    if values is None:
        return None
    return np.asarray(values, dtype=float)[mask]


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def _normalize_rows_configured(
    matrix: np.ndarray,
    *,
    normalize: bool,
    epsilon: float,
    zero_vector_policy: str,
    dtype: str,
) -> np.ndarray:
    resolved_dtype = np.dtype(dtype)
    values = np.asarray(matrix, dtype=resolved_dtype)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("configured embedding matrix must be finite and two-dimensional")
    if not normalize:
        return values
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if zero_vector_policy != "reject":
        raise ValueError("configured cluster-local zero-vector policy is unsupported")
    if np.any(~np.isfinite(norms)) or np.any(norms <= float(epsilon)):
        raise ValueError("cluster-local normalization encountered a zero vector")
    normalized = values / norms
    if not np.isfinite(normalized).all():
        raise ValueError("cluster-local normalized patient embeddings are non-finite")
    return normalized.astype(resolved_dtype, copy=False)


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0.0:
        return vector
    return vector / norm


def _coerce_embedding_matrix(embeddings: Any, expected_rows: int) -> np.ndarray:
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2:
        raise RuntimeError(f"Embedding provider returned shape {matrix.shape}")
    if matrix.shape[0] != expected_rows:
        raise RuntimeError(
            f"Embedding provider returned {matrix.shape[0]} embeddings for "
            f"{expected_rows} texts"
        )
    return matrix


def _torch_device_or_none(device: Optional[str]):
    if device is None or str(device).strip().lower() in {"", "auto"}:
        return None
    import torch

    return torch.device(str(device))


def _coerce_torch_devices(devices: Sequence[Any]):
    if not devices:
        return []
    import torch

    return [
        device if isinstance(device, torch.device) else torch.device(str(device))
        for device in devices
    ]


def _release_sentence_transformer_model(model_name: str) -> None:
    try:
        clear_sentence_transformer_cache(model_name=model_name)
        logger.info("Released sentence-transformer model cache for %s", model_name)
    except Exception:
        logger.warning(
            "Failed to release sentence-transformer model cache for %s",
            model_name,
            exc_info=True,
        )


def _default_embedding_cache_dir(dataset_path: str, output_dir: Path) -> Path:
    path = Path(str(dataset_path))
    if str(path).endswith("in_memory_dataset"):
        return Path(output_dir) / "embedding_cache"
    try:
        resolved = path.expanduser().resolve()
    except OSError:
        resolved = path.expanduser().absolute()
    dataset_dir = resolved if resolved.is_dir() else resolved.parent
    if not str(dataset_dir):
        return Path(output_dir) / "embedding_cache"
    return dataset_dir / ".oci_cache" / "embedding_contrast"


def _resolve_external_cache_paths(raw_path: str) -> List[Path]:
    root = Path(str(raw_path)).expanduser()
    if _is_embedding_cache_path(root):
        return [root]
    if not root.exists():
        raise FileNotFoundError(f"External embedding corpus path not found: {root}")
    paths = [
        child
        for child in sorted(root.iterdir())
        if child.is_dir() and _is_embedding_cache_path(child)
    ]
    if not paths:
        raise FileNotFoundError(
            f"No embedding chunk cache found at {root}; expected chunk_embeddings.npy, "
            "offsets.npy, and chunk_texts.jsonl in the path or an immediate child."
        )
    return paths


def _is_embedding_cache_path(path: Path) -> bool:
    return all(
        (path / filename).exists()
        for filename in ["chunk_embeddings.npy", "offsets.npy", "chunk_texts.jsonl"]
    )


def _load_external_corpus_cache(cache_path: Path) -> Dict[str, Any]:
    metadata_path = cache_path / "metadata.json"
    metadata: Dict[str, Any] = {}
    if metadata_path.exists():
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)
    embeddings = np.load(str(cache_path / "chunk_embeddings.npy"), mmap_mode="r")
    offsets = np.load(str(cache_path / "offsets.npy"))
    chunks_by_sample = []
    with open(cache_path / "chunk_texts.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks_by_sample.append(
                    [str(chunk) for chunk in json.loads(line).get("chunks", [])]
                )
    if len(offsets) != len(chunks_by_sample) + 1:
        raise RuntimeError(f"External cache offsets/chunk text length mismatch in {cache_path}")
    row_metadata = _load_external_row_metadata(cache_path, len(chunks_by_sample))
    name = str(metadata.get("corpus_name") or metadata.get("dataset_path") or cache_path.name)
    return {
        "name": name,
        "cache_path": cache_path,
        "metadata": metadata,
        "embeddings": embeddings,
        "offsets": offsets,
        "chunks_by_sample": chunks_by_sample,
        "row_metadata": row_metadata,
    }


def _load_external_row_metadata(
    cache_path: Path,
    expected_rows: int,
) -> List[Dict[str, Any]]:
    path = cache_path / "row_metadata.jsonl"
    if not path.exists():
        return [{"row_index": idx} for idx in range(expected_rows)]
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                payload = json.loads(line)
                rows.append(payload if isinstance(payload, dict) else {"value": payload})
    if len(rows) != expected_rows:
        logger.warning(
            "External row metadata in %s has %d rows; expected %d",
            path,
            len(rows),
            expected_rows,
        )
    return rows


def _top_scored_flat_indices(
    embeddings: np.ndarray,
    direction: np.ndarray,
    *,
    limit: int,
    descending: bool,
) -> List[Tuple[float, int]]:
    direction = np.asarray(direction, dtype=np.float32).reshape(-1)
    block_size = 50_000
    candidates: List[Tuple[float, int]] = []
    n_rows = int(embeddings.shape[0])
    for start in range(0, n_rows, block_size):
        end = min(start + block_size, n_rows)
        scores = np.asarray(embeddings[start:end], dtype=np.float32) @ direction
        finite = np.isfinite(scores)
        if not np.any(finite):
            continue
        finite_indices = np.flatnonzero(finite)
        finite_scores = scores[finite_indices]
        k = min(int(limit), len(finite_scores))
        if k <= 0:
            continue
        if descending:
            local = np.argpartition(finite_scores, -k)[-k:]
        else:
            local = np.argpartition(finite_scores, k - 1)[:k]
        candidates.extend(
            (float(finite_scores[idx]), int(start + finite_indices[idx])) for idx in local
        )
    candidates.sort(key=lambda item: item[0], reverse=descending)
    return candidates[: int(limit)]


def _finite_or_none(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    numeric = float(value)
    if not np.isfinite(numeric):
        return None
    return numeric


def _row_key(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _jsonable_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    return value


def _jsonable_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
