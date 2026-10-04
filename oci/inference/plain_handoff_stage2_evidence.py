"""Fold-scoped evidence compilation for the plain Stage 2 workflow.

The Stage 1 handoff intentionally preserves detailed scientific evidence.  It
is much too redundant, however, to send directly to an interpretation model:
the same phrases and patient chunks recur across model views, evidence axes,
and full/inner training contexts.  This module turns that handoff into compact,
auditable evidence cards before the first Stage 2 model request.

The compiler reuses the allowlisted scientific adapters from
``all_evidence_fusion``.  It then performs deterministic reductions:

* exact text/content deduplication within (never across) outer folds;
* provenance and stability aggregation across full/inner training contexts;
* conservative, stratified semantic clustering with an oversampled card cap;
* source-relative strength ranking and overlap control within each card;
* supporting sentence lookup in the supplied fold-local clinical evidence.

Full raw evidence remains in the Stage 1 handoff.  Separate member and lineage
manifests retain the path from every prompt card back to its raw occurrences.
Cached Stage 1 chunk embeddings are memory-mapped when available; the compiler
never loads another embedding model beside the Stage 2 serving process.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .all_evidence_fusion import (
    _compact_bow_rows,
    _compact_tfidf_evidence,
    _compact_topic_terms,
    _embedding_family,
    _normalize_evidence_text,
)
from .stage1_architectures import (
    BOW_NUISANCE,
    BOW_R_LOSS,
    HTR_NEURAL,
    MATCHED_PAIR_UPLIFT,
    NEURAL_QUERY_MOMENTS,
    STAGE1_ARCHITECTURES,
    TFIDF_ORPHAN_NGRAMS,
    TFIDF_SEMANTIC_RETRIEVAL,
)
from .neural_query_evidence_contract import validate_query_retrieval_policy
from .stage2_evidence_selection import (
    SELECTION_POLICY,
    SupportingSentenceIndex,
    annotate_strength,
    select_exemplars as _select_exemplars,
)

EVIDENCE_COMPILER_VERSION = "semantic_cluster_cards_v5"
SUPPORTED_STAGE2_ARCHITECTURES = STAGE1_ARCHITECTURES
ALLOWED_AXES = {
    "treatment",
    "outcome",
    "residual_effect",
    "matched_pair",
    "semantic",
    "unclear",
}
_OPERATIONAL_KEYS = {
    "artifacts",
    "artifact_inventory",
    "common_vocabulary",
    "config",
    "fit_row_ids",
    "heldout_row_ids",
    "metrics",
    "model_diagnostics",
    "predictions",
    "run_config",
    "schema_version",
    "train_activations",
}
_NUMERIC_SCORE_KEYS = {
    "attention",
    "best_abs_confounder_score",
    "best_abs_effect_score",
    "coefficient",
    "combined_score",
    "effect_loss",
    "fit_standardized_score",
    "importance",
    "loading",
    "mean_abs_confounder_score",
    "mean_abs_effect_score",
    "probe_auc",
    "r_loss",
    "r_pseudo_outcome",
    "score",
    "signed_score",
    "similarity",
    "standardized_score",
    "tau_hat_r_stage",
    "tfidf_contrast",
    "rank",
    "fit_rank",
    "fit_signed_score",
    "abs_pseudo_target_score",
    "pseudo_target_score",
    "confounder_overlap_score",
}


@dataclass(frozen=True)
class CompiledStage2Evidence:
    packets: tuple[dict[str, Any], ...]
    cards_by_outer_fold: Mapping[int, tuple[dict[str, Any], ...]]
    members_by_outer_fold: Mapping[int, tuple[dict[str, Any], ...]]
    lineage_by_outer_fold: Mapping[int, tuple[dict[str, Any], ...]]
    summary: Mapping[str, Any]


def _discover_chunk_embedding_cache_directory(handoff_path: Path) -> Path | None:
    output_root = Path(handoff_path).parent.parent
    candidates = sorted(
        path.parent
        for path in (output_root / "components" / "embedding_cache" / "cache").glob(
            "*/metadata.json"
        )
        if (path.parent / "chunk_embeddings.npy").is_file()
        and (path.parent / "offsets.npy").is_file()
    )
    if not candidates:
        return None
    if len(candidates) != 1:
        raise RuntimeError(
            "Stage 2 evidence compilation found multiple Stage 1 embedding caches: "
            f"{[str(path) for path in candidates]}"
        )
    return candidates[0]


def stage1_embedding_cache_dependency_identity(
    handoff_path: Path,
) -> Mapping[str, Any] | None:
    """Return a lightweight identity for the optional Stage 1 embedding cache.

    Cache arrays can be many gigabytes, so restart compatibility deliberately
    uses only the producer's small semantic metadata plus path/stat identities.
    It never reads or hashes array contents.
    """

    directory = _discover_chunk_embedding_cache_directory(Path(handoff_path))
    if directory is None:
        return None
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    semantic_metadata_keys = (
        "sentence_model_name",
        "hidden_size",
        "num_samples",
        "total_chunks",
        "chunk_size_words",
        "chunk_overlap_words",
        "max_chunks",
        "chunk_selection",
        "normalize_embeddings",
        "max_seq_length",
        "effective_max_seq_length",
        "chunking_mode",
        "storage_format",
        "dataset_path",
        "cache_hash",
        "chunking_policy_version",
        "semantic_truncation_allowed",
        "dtype",
    )

    def product_identity(path: Path) -> dict[str, Any]:
        stat = path.stat()
        return {
            "path": str(path.resolve()),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "ctime_ns": int(stat.st_ctime_ns),
            "device": int(stat.st_dev),
            "inode": int(stat.st_ino),
        }

    return {
        "schema_version": "stage1_embedding_cache_dependency_v1",
        "directory": str(directory.resolve()),
        "semantic_metadata": {
            key: metadata.get(key) for key in semantic_metadata_keys
        },
        "products": {
            filename: product_identity(directory / filename)
            for filename in (
                "metadata.json",
                "chunk_embeddings.npy",
                "offsets.npy",
            )
        },
    }


class _ChunkEmbeddingCache:
    """Read-only row/chunk lookup over the existing Stage 1 embedding cache."""

    def __init__(self, directory: Path) -> None:
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        self.directory = directory
        self.metadata = metadata
        self.embeddings = np.load(directory / "chunk_embeddings.npy", mmap_mode="r")
        self.offsets = np.load(directory / "offsets.npy", mmap_mode="r")
        if self.embeddings.ndim != 2 or self.offsets.ndim != 1:
            raise ValueError("Stage 1 embedding cache arrays have invalid dimensions")
        if len(self.offsets) < 2 or int(self.offsets[-1]) != len(self.embeddings):
            raise ValueError("Stage 1 embedding cache offsets are inconsistent")

    @classmethod
    def discover(cls, handoff_path: Path) -> _ChunkEmbeddingCache | None:
        directory = _discover_chunk_embedding_cache_directory(Path(handoff_path))
        if directory is None:
            return None
        return cls(directory)

    def flat_index(self, row_id: Any, chunk_index: Any) -> int | None:
        try:
            row = int(row_id)
            chunk = int(chunk_index)
        except (TypeError, ValueError):
            return None
        if row < 0 or row + 1 >= len(self.offsets):
            return None
        start = int(self.offsets[row])
        stop = int(self.offsets[row + 1])
        if chunk < 0 or start + chunk >= stop:
            return None
        return start + chunk

    def vectors(self, indices: Sequence[int]) -> np.ndarray:
        matrix = np.asarray(self.embeddings[np.asarray(indices, dtype=int)], dtype=np.float32)
        if not bool(self.metadata.get("normalize_embeddings", False)):
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            matrix = matrix / np.maximum(norms, 1e-12)
        return matrix


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _safe_token(value: Any, *, default: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")
    return token or default


def _clean_text(value: Any) -> str:
    return _normalize_evidence_text(value)


def _content_text(value: Any, *, limit: int = 12_000) -> str:
    """Collect readable leaves for a last-resort structured evidence unit."""

    parts: list[str] = []

    def visit(child: Any, key: str = "") -> None:
        if sum(len(part) for part in parts) >= limit:
            return
        if isinstance(child, Mapping):
            for raw_key, nested in child.items():
                name = str(raw_key)
                lowered = name.lower()
                if (
                    lowered in _OPERATIONAL_KEYS
                    or lowered.endswith(("_sha256", "_hash", "_fingerprint", "_path"))
                    or lowered.startswith(("authenticated_", "attestation_", "immutable_"))
                ):
                    continue
                visit(nested, name)
        elif isinstance(child, (list, tuple)):
            for nested in child:
                visit(nested, key)
        elif isinstance(child, str):
            text = _clean_text(child)
            if text and not re.fullmatch(r"[a-f0-9]{32,}", text.lower()):
                parts.append(f"{key}: {text}" if key else text)

    visit(value)
    return _clean_text("; ".join(dict.fromkeys(parts)))[:limit]


def _finite_scores(value: Mapping[str, Any]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for key, raw in value.items():
        if str(key) not in _NUMERIC_SCORE_KEYS or isinstance(raw, bool):
            continue
        try:
            number = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            scores[str(key)] = number
    return scores


def _axes_from_descriptor(value: Any) -> set[str]:
    text = _clean_text(value).lower()
    axes: set[str] = set()
    if "treatment" in text or "propensity" in text or "assignment" in text:
        axes.add("treatment")
    if "outcome" in text or "prognostic" in text or "nuisance" in text:
        axes.add("outcome")
    if any(
        token in text
        for token in (
            "effect",
            "heterogeneity",
            "interaction",
            "pseudo target",
            "pseudo outcome",
            "r loss",
            "modifier",
        )
    ):
        axes.add("residual_effect")
    if "matched pair" in text or "uplift" in text:
        axes.add("matched_pair")
    return axes or {"semantic"}


def _axes_for_bank(bank: Any) -> set[str]:
    normalized = _safe_token(bank, default="semantic")
    if normalized == "treatment":
        return {"treatment"}
    if normalized == "outcome":
        return {"outcome"}
    if normalized in {"effect", "effect_modifier", "residual_effect"}:
        return {"residual_effect"}
    return {"semantic"}


def _base_reference(
    row: Mapping[str, Any],
    *,
    handoff_row: int,
    json_path: str,
) -> dict[str, Any]:
    return {
        "handoff_row": int(handoff_row),
        "source": str(row.get("source") or "unknown"),
        "inner_fold": row.get("inner_fold"),
        "scope": str(row.get("scope") or "unspecified"),
        "json_path": str(json_path),
    }


def _rebind_compact_handoff_references(
    occurrence: dict[str, Any],
    *,
    handoff_row: int,
) -> None:
    """Point representative and summarized provenance at the row being read."""

    reference = dict(occurrence.get("reference") or {})
    reference["handoff_row"] = int(handoff_row)
    occurrence["reference"] = reference
    summaries = occurrence.get("reference_summaries")
    if isinstance(summaries, Sequence) and not isinstance(summaries, (str, bytes)):
        occurrence["reference_summaries"] = [
            {**dict(summary), "handoff_row": int(handoff_row)}
            for summary in summaries
            if isinstance(summary, Mapping)
        ]


def _occurrence(
    *,
    text: Any,
    evidence_kind: str,
    axes: Sequence[str] | set[str],
    polarity: str,
    source_families: Sequence[str],
    architecture: str,
    reference: Mapping[str, Any],
    details: Mapping[str, Any] | None = None,
    scores: Mapping[str, float] | None = None,
    patient_row_id: Any = None,
    cache_coordinate: tuple[Any, Any] | None = None,
) -> dict[str, Any] | None:
    cleaned = _clean_text(text)
    if not cleaned:
        return None
    clean_axes = sorted(set(map(str, axes)).intersection(ALLOWED_AXES) or {"semantic"})
    ref = dict(reference)
    if patient_row_id is not None:
        try:
            ref["row_id"] = int(patient_row_id)
        except (TypeError, ValueError):
            pass
    if cache_coordinate is not None:
        try:
            ref["chunk_index"] = int(cache_coordinate[1])
        except (TypeError, ValueError):
            pass
    return {
        "text": cleaned,
        "evidence_kind": _safe_token(evidence_kind, default="structured_evidence"),
        "axes": clean_axes,
        "polarity": _safe_token(polarity, default="unsigned"),
        "source_families": sorted(set(map(str, source_families))),
        "architecture": _safe_token(architecture, default="compiled_evidence"),
        "reference": ref,
        "details": dict(details or {}),
        "scores": dict(scores or {}),
        "patient_row_id": ref.get("row_id"),
        "cache_coordinate": cache_coordinate,
    }


def _exact_occurrence_key(occurrence: Mapping[str, Any]) -> str:
    """Return the scientific identity used for lossless exact aggregation."""

    return _canonical_json(
        {
            "evidence_kind": occurrence["evidence_kind"],
            "text": _clean_text(occurrence["text"]).casefold(),
            "source_families": sorted(occurrence["source_families"]),
            "architecture": str(occurrence["architecture"]),
        }
    )


def _occurrence_count(occurrence: Mapping[str, Any]) -> int:
    raw = occurrence.get("raw_occurrence_count", 1)
    if isinstance(raw, bool):
        raise ValueError("compact evidence occurrence count must be a positive integer")
    try:
        count = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("compact evidence occurrence count must be a positive integer") from exc
    if count < 1:
        raise ValueError("compact evidence occurrence count must be a positive integer")
    return count


def _occurrence_polarities(occurrence: Mapping[str, Any]) -> set[str]:
    values = occurrence.get("polarities")
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        rendered = {str(value) for value in values if str(value)}
        if rendered:
            return rendered
    return {str(occurrence.get("polarity") or "unsigned")}


def _occurrence_reference_summaries(
    occurrence: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return compact provenance rows whose counts sum to the raw multiplicity."""

    raw_summaries = occurrence.get("reference_summaries")
    if isinstance(raw_summaries, Sequence) and not isinstance(
        raw_summaries, (str, bytes)
    ):
        summaries = [dict(value) for value in raw_summaries if isinstance(value, Mapping)]
    else:
        reference = dict(occurrence.get("reference") or {})
        details = occurrence.get("details")
        if isinstance(details, Mapping):
            for key in ("query_id", "bank", "term_normalization"):
                value = details.get(key)
                if value not in (None, ""):
                    reference[key] = value
        scores = _finite_scores(occurrence.get("scores") or {})
        if scores:
            # Keep the score-to-context association, not just the marginal
            # score distribution encoded on the compact representative.
            reference["scores"] = scores
        reference["occurrence_count"] = _occurrence_count(occurrence)
        summaries = [reference]
    total = 0
    for summary in summaries:
        raw_count = summary.get("occurrence_count", 1)
        if isinstance(raw_count, bool):
            raise ValueError("compact evidence reference count must be positive")
        try:
            count = int(raw_count)
        except (TypeError, ValueError) as exc:
            raise ValueError("compact evidence reference count must be positive") from exc
        if count < 1:
            raise ValueError("compact evidence reference count must be positive")
        summary["occurrence_count"] = count
        total += count
    if total != _occurrence_count(occurrence):
        raise ValueError(
            "compact evidence reference counts do not equal raw_occurrence_count"
        )
    return summaries


def _merge_reference_summaries(
    occurrences: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    counts: dict[str, int] = defaultdict(int)
    values: dict[str, dict[str, Any]] = {}
    for occurrence in occurrences:
        for raw_summary in _occurrence_reference_summaries(occurrence):
            summary = dict(raw_summary)
            count = int(summary.pop("occurrence_count", 1))
            key = _canonical_json(summary)
            values[key] = summary
            counts[key] += count
    return [
        {**values[key], "occurrence_count": counts[key]}
        for key in sorted(values)
    ]


def _occurrence_score_value_counts(
    occurrence: Mapping[str, Any],
) -> dict[str, dict[float, int]]:
    """Return exact run-length encoded score values for one compact occurrence."""

    encoded = occurrence.get("score_value_counts")
    output: dict[str, dict[float, int]] = defaultdict(lambda: defaultdict(int))
    if isinstance(encoded, Mapping):
        for raw_name, rows in encoded.items():
            if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
                raise ValueError("compact evidence score counts must be arrays")
            for row in rows:
                if not isinstance(row, Mapping):
                    raise ValueError("compact evidence score count rows must be objects")
                value = float(row["value"])
                count = int(row["count"])
                if not math.isfinite(value) or count < 1:
                    raise ValueError("compact evidence score values must be finite and counted")
                output[str(raw_name)][value] += count
        return {name: dict(counts) for name, counts in output.items()}

    count = _occurrence_count(occurrence)
    for name, value in _finite_scores(occurrence.get("scores") or {}).items():
        output[name][float(value)] += count
    return {name: dict(counts) for name, counts in output.items()}


@dataclass(slots=True)
class _ExactOccurrenceGroup:
    representative: Mapping[str, Any]
    axes: set[str]
    polarities: set[str]
    raw_occurrence_count: int
    reference_values: dict[str, dict[str, Any]]
    reference_counts: dict[str, int]
    score_counts: dict[str, dict[float, int]]

    @classmethod
    def from_occurrence(
        cls,
        occurrence: Mapping[str, Any],
    ) -> "_ExactOccurrenceGroup":
        group = cls(
            representative=occurrence,
            axes=set(),
            polarities=set(),
            raw_occurrence_count=0,
            reference_values={},
            reference_counts=defaultdict(int),
            score_counts=defaultdict(lambda: defaultdict(int)),
        )
        group.add(occurrence)
        return group

    def add(self, occurrence: Mapping[str, Any]) -> None:
        representative_rank = (
            len(self.representative.get("scores") or {}),
            len(self.representative.get("details") or {}),
        )
        occurrence_rank = (
            len(occurrence.get("scores") or {}),
            len(occurrence.get("details") or {}),
        )
        if occurrence_rank > representative_rank:
            self.representative = occurrence
        self.axes.update(map(str, occurrence.get("axes") or []))
        self.polarities.update(_occurrence_polarities(occurrence))
        self.raw_occurrence_count += _occurrence_count(occurrence)
        for raw_summary in _occurrence_reference_summaries(occurrence):
            summary = dict(raw_summary)
            count = int(summary.pop("occurrence_count", 1))
            reference_key = _canonical_json(summary)
            self.reference_values[reference_key] = summary
            self.reference_counts[reference_key] += count
        for score_name, value_counts in _occurrence_score_value_counts(
            occurrence
        ).items():
            for score_value, count in value_counts.items():
                self.score_counts[score_name][score_value] += count

    def render(self) -> dict[str, Any]:
        occurrence = copy.deepcopy(dict(self.representative))
        polarities = sorted(self.polarities)
        occurrence["axes"] = sorted(self.axes)
        occurrence["polarity"] = polarities[0]
        occurrence["polarities"] = polarities
        occurrence["raw_occurrence_count"] = int(self.raw_occurrence_count)
        occurrence["reference_summaries"] = [
            {
                **self.reference_values[reference_key],
                "occurrence_count": int(self.reference_counts[reference_key]),
            }
            for reference_key in sorted(self.reference_values)
        ]
        occurrence["score_value_counts"] = {
            score_name: [
                {
                    "value": score_value,
                    "count": int(value_counts[score_value]),
                }
                for score_value in sorted(value_counts)
            ]
            for score_name, value_counts in sorted(self.score_counts.items())
        }
        return occurrence


class _ExactOccurrenceCompactor:
    """Compact repeated occurrences online without retaining duplicate objects."""

    def __init__(self) -> None:
        self._groups: dict[
            str,
            Mapping[str, Any] | _ExactOccurrenceGroup,
        ] = {}

    def add(self, occurrence: Mapping[str, Any]) -> None:
        key = _exact_occurrence_key(occurrence)
        current = self._groups.get(key)
        if current is None:
            self._groups[key] = occurrence
            return
        if isinstance(current, _ExactOccurrenceGroup):
            current.add(occurrence)
            return
        group = _ExactOccurrenceGroup.from_occurrence(current)
        group.add(occurrence)
        self._groups[key] = group

    def finish(self) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for key in sorted(self._groups):
            value = self._groups.pop(key)
            if isinstance(value, _ExactOccurrenceGroup):
                output.append(value.render())
            else:
                output.append(dict(value))
        return output


def _extract_sparse_occurrences(
    row: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    handoff_row: int,
) -> list[dict[str, Any]]:
    importance = payload.get("importance")
    if not isinstance(importance, Mapping):
        return []
    output: list[dict[str, Any]] = []
    containers = [("importance", importance)]
    ensemble = importance.get("ensemble_r")
    if isinstance(ensemble, Mapping):
        containers.append(("importance.ensemble_r", ensemble))
    matched_pair = importance.get("matched_pair_uplift")
    if isinstance(matched_pair, Mapping):
        containers.append(("importance.matched_pair_uplift", matched_pair))

    def add_term(
        compact: Mapping[str, Any],
        *,
        path: str,
        axes: set[str],
        polarity: str,
        family: str,
        view_name: str = "",
    ) -> None:
        term = compact.get("term")
        pair_side = None
        if family == MATCHED_PAIR_UPLIFT and isinstance(term, str) and "::" in term:
            prefix, candidate = term.split("::", 1)
            if prefix in {"control", "treated"} and candidate.strip():
                pair_side = prefix
                term = candidate.strip()
        item = _occurrence(
            text=term,
            evidence_kind="lexical_term",
            axes=axes,
            polarity=polarity,
            source_families=[family],
            architecture=family,
            reference=_base_reference(row, handoff_row=handoff_row, json_path=path),
            details={"term": term, "view_name": view_name, "pair_side": pair_side},
            scores=_finite_scores(compact),
        )
        if item is not None:
            output.append(item)

    ranked_fields = {
        "treatment_positive": ({"treatment"}, "positive", BOW_NUISANCE),
        "treatment_negative": ({"treatment"}, "negative", BOW_NUISANCE),
        "outcome_positive": ({"outcome"}, "positive", BOW_NUISANCE),
        "outcome_negative": ({"outcome"}, "negative", BOW_NUISANCE),
        "pseudo_target_positive": ({"residual_effect"}, "positive", BOW_R_LOSS),
        "pseudo_target_negative": ({"residual_effect"}, "negative", BOW_R_LOSS),
        "confounder_overlap": ({"treatment", "outcome"}, "unsigned", BOW_NUISANCE),
        "uplift_pair_features": ({"matched_pair"}, "unsigned", MATCHED_PAIR_UPLIFT),
        "uplift_delta_logit_positive": (
            {"matched_pair"},
            "positive",
            MATCHED_PAIR_UPLIFT,
        ),
        "uplift_delta_logit_negative": (
            {"matched_pair"},
            "negative",
            MATCHED_PAIR_UPLIFT,
        ),
        "ridge_delta_probability_positive": (
            {"matched_pair"},
            "positive",
            MATCHED_PAIR_UPLIFT,
        ),
        "ridge_delta_probability_negative": (
            {"matched_pair"},
            "negative",
            MATCHED_PAIR_UPLIFT,
        ),
        # Read old handoffs without making these obsolete aliases the producer contract.
        "matched_pair_positive": ({"matched_pair"}, "positive", MATCHED_PAIR_UPLIFT),
        "matched_pair_negative": ({"matched_pair"}, "negative", MATCHED_PAIR_UPLIFT),
    }
    for container_path, container in containers:
        views = container.get("views")
        if isinstance(views, Sequence) and not isinstance(views, (str, bytes)):
            for view_index, view in enumerate(views):
                if not isinstance(view, Mapping):
                    continue
                view_name = str(view.get("view_name") or f"view_{view_index + 1}")
                for field, (axes, polarity, family) in ranked_fields.items():
                    for compact in _compact_bow_rows(view.get(field)):
                        add_term(
                            compact,
                            path=f"{container_path}.views[{view_index}].{field}",
                            axes=axes,
                            polarity=polarity,
                            family=family,
                            view_name=view_name,
                        )
        for field in ("phrase_consensus", "phrase_features"):
            values = container.get(field)
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                continue
            for index, raw in enumerate(values):
                if not isinstance(raw, Mapping):
                    continue
                term = raw.get("feature") or raw.get("term") or raw.get("phrase")
                scored_families: list[tuple[set[str], str]] = []
                if container_path == "importance.ensemble_r":
                    scored_families.append(({"residual_effect"}, BOW_R_LOSS))
                else:
                    confounder_scores = (
                        raw.get("best_abs_confounder_score"),
                        raw.get("confounder_overlap_score"),
                    )
                    effect_scores = (
                        raw.get("best_abs_effect_score"),
                        raw.get("abs_pseudo_target_score"),
                        raw.get("pseudo_target_score"),
                    )
                    if any(value not in (None, 0, 0.0) for value in confounder_scores):
                        scored_families.append(({"treatment", "outcome"}, BOW_NUISANCE))
                    if any(value not in (None, 0, 0.0) for value in effect_scores):
                        scored_families.append(({"residual_effect"}, BOW_R_LOSS))
                for axes, family in scored_families:
                    item = _occurrence(
                        text=term,
                        evidence_kind="lexical_term",
                        axes=axes,
                        polarity="unsigned",
                        source_families=[family],
                        architecture=family,
                        reference=_base_reference(
                            row,
                            handoff_row=handoff_row,
                            json_path=f"{container_path}.{field}[{index}]",
                        ),
                        details={
                            "term": term,
                            "supporting_views": list(raw.get("supporting_views") or []),
                        },
                        scores=_finite_scores(raw),
                    )
                    if item is not None:
                        output.append(item)
    return output


def _extract_embedding_occurrences(
    row: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    handoff_row: int,
) -> list[dict[str, Any]]:
    embedding = payload.get("embedding_contrast_evidence")
    contrasts = embedding.get("contrasts") if isinstance(embedding, Mapping) else None
    if not isinstance(contrasts, Sequence) or isinstance(contrasts, (str, bytes)):
        return []
    output: list[dict[str, Any]] = []
    for contrast_index, contrast in enumerate(contrasts):
        if not isinstance(contrast, Mapping):
            continue
        descriptor = " ".join(
            str(contrast.get(key) or "")
            for key in ("name", "contrast_family", "direction_source", "role_hint")
        )
        axes = _axes_from_descriptor(descriptor)
        family = _embedding_family(contrast)
        contrast_name = str(
            contrast.get("name") or contrast.get("contrast_family") or "embedding_contrast"
        )
        for side in (
            "positive_aligned_chunks",
            "negative_aligned_chunks",
            "positive_external_chunks",
            "negative_external_chunks",
        ):
            values = contrast.get(side)
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                continue
            polarity = "negative" if side.startswith("negative") else "positive"
            for item_index, raw in enumerate(values):
                if not isinstance(raw, Mapping):
                    continue
                row_id = raw.get("row_id")
                chunk_index = raw.get("chunk_index")
                occurrence = _occurrence(
                    text=raw.get("text") or raw.get("chunk_text"),
                    evidence_kind="clinical_text",
                    axes=axes,
                    polarity=polarity,
                    source_families=[family],
                    architecture=family,
                    reference=_base_reference(
                        row,
                        handoff_row=handoff_row,
                        json_path=(
                            f"embedding_contrast_evidence.contrasts[{contrast_index}]."
                            f"{side}[{item_index}]"
                        ),
                    ),
                    details={"contrast": contrast_name, "side": side},
                    scores=_finite_scores(raw),
                    patient_row_id=row_id,
                    cache_coordinate=(row_id, chunk_index),
                )
                if occurrence is not None:
                    output.append(occurrence)
        scores = contrast.get("concept_probe_scores")
        if isinstance(scores, Sequence) and not isinstance(scores, (str, bytes)):
            for score_index, raw in enumerate(scores):
                if not isinstance(raw, Mapping):
                    continue
                concept = raw.get("concept") or raw.get("phrase") or raw.get("label")
                occurrence = _occurrence(
                    text=concept,
                    evidence_kind="lexical_term",
                    axes=axes,
                    polarity="unsigned",
                    source_families=[family],
                    architecture=family,
                    reference=_base_reference(
                        row,
                        handoff_row=handoff_row,
                        json_path=(
                            f"embedding_contrast_evidence.contrasts[{contrast_index}]."
                            f"concept_probe_scores[{score_index}]"
                        ),
                    ),
                    details={"term": concept, "contrast": contrast_name},
                    scores=_finite_scores(raw),
                )
                if occurrence is not None:
                    output.append(occurrence)
        terms = contrast.get("tfidf_retrieval_terms")
        if isinstance(terms, Sequence) and not isinstance(terms, (str, bytes)):
            for term_index, raw in enumerate(terms):
                if not isinstance(raw, Mapping):
                    continue
                polarity = str(raw.get("polarity") or "unsigned").lower()
                if polarity not in {"positive", "negative"}:
                    polarity = "unsigned"
                occurrence = _occurrence(
                    text=raw.get("term") or raw.get("feature") or raw.get("ngram"),
                    evidence_kind="lexical_term",
                    axes=axes,
                    polarity=polarity,
                    source_families=[TFIDF_SEMANTIC_RETRIEVAL],
                    architecture=TFIDF_SEMANTIC_RETRIEVAL,
                    reference=_base_reference(
                        row,
                        handoff_row=handoff_row,
                        json_path=(
                            f"embedding_contrast_evidence.contrasts[{contrast_index}]."
                            f"tfidf_retrieval_terms[{term_index}]"
                        ),
                    ),
                    details={
                        "term": raw.get("term"),
                        "parent_contrast": contrast_name,
                        "parent_embedding_family": family,
                    },
                    scores=_finite_scores(raw),
                )
                if occurrence is not None:
                    output.append(occurrence)
    return output


def _extract_htr_occurrences(
    row: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    handoff_row: int,
) -> list[dict[str, Any]]:
    htr = payload.get("htr_evidence")
    if not isinstance(htr, Mapping):
        return []
    output: list[dict[str, Any]] = []
    branches: list[tuple[str, Mapping[str, Any]]] = []
    effect_variants = htr.get("effect_variants")
    for branch, branch_payload in htr.items():
        if branch == "effect" and isinstance(effect_variants, Mapping):
            # The canonical effect entry aliases one member of effect_variants.
            continue
        if branch == "effect_variants" and isinstance(branch_payload, Mapping):
            for variant, variant_payload in branch_payload.items():
                if isinstance(variant_payload, Mapping):
                    branches.append((f"effect_variants.{variant}", variant_payload))
            continue
        if isinstance(branch_payload, Mapping):
            branches.append((str(branch), branch_payload))

    for branch_name, branch_payload in branches:
        if not isinstance(branch_payload, Mapping):
            continue
        values = branch_payload.get("attention")
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            continue
        normalized_branch = branch_name.lower()
        is_matched_pair = "pair_uplift" in normalized_branch
        axes = (
            {"matched_pair"}
            if is_matched_pair
            else ({"residual_effect"} if "effect" in normalized_branch else {"treatment", "outcome"})
        )
        family = MATCHED_PAIR_UPLIFT if is_matched_pair else HTR_NEURAL
        for item_index, raw in enumerate(values):
            if not isinstance(raw, Mapping):
                continue
            # Use the unmarked chunk as the semantic identity. Highlight markup
            # differs across HTR stages even when the underlying patient text is
            # identical, and would defeat exact deduplication.
            full_text = raw.get("chunk_text") or raw.get("highlighted_chunk_text")
            occurrence = _occurrence(
                text=full_text or raw.get("attended_token_summary"),
                evidence_kind="clinical_text",
                axes=axes,
                polarity="unsigned",
                source_families=[family],
                architecture=family,
                reference=_base_reference(
                    row,
                    handoff_row=handoff_row,
                    json_path=f"htr_evidence.{branch_name}.attention[{item_index}]",
                ),
                details={
                    "stage": raw.get("stage") or branch_name,
                    "attended_token_summary": raw.get("attended_token_summary"),
                },
                scores=_finite_scores(raw),
                patient_row_id=raw.get("row_id"),
                # HTR chunk indices may use a different chunker, so they are
                # deliberately not bound to the embedding cache by position.
                cache_coordinate=None,
            )
            if occurrence is not None:
                output.append(occurrence)
    return output


def _extract_tfidf_occurrences(
    row: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    handoff_row: int,
) -> list[dict[str, Any]]:
    try:
        blocks = _compact_tfidf_evidence(payload)
    except ValueError:
        return []
    output: list[dict[str, Any]] = []
    for block_index, (families, _role, content) in enumerate(blocks):
        terms = content.get("terms") if isinstance(content, Mapping) else None
        term_values = [
            str(item.get("term") or "") if isinstance(item, Mapping) else str(item)
            for item in (terms or [])
        ]
        text = "; ".join(term for term in term_values if term)
        bank = content.get("bank") if isinstance(content, Mapping) else None
        axes = _axes_for_bank(bank)
        if TFIDF_ORPHAN_NGRAMS in families:
            axes = {"residual_effect"}
        occurrence = _occurrence(
            text=text,
            evidence_kind=("orphan_ngram_cluster" if TFIDF_ORPHAN_NGRAMS in families else "topic"),
            axes=axes,
            polarity="unsigned",
            source_families=families,
            architecture=(
                "tfidf_orphan_ngrams" if TFIDF_ORPHAN_NGRAMS in families else "tfidf_topics"
            ),
            reference=_base_reference(
                row,
                handoff_row=handoff_row,
                json_path=f"fusion_compacted_tfidf[{block_index}]",
            ),
            details=content,
        )
        if occurrence is not None:
            output.append(occurrence)
    return output


def _extract_neural_query_occurrences(
    row: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    handoff_row: int,
) -> list[dict[str, Any]]:
    values = payload.get("evidence")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return []
    output: list[dict[str, Any]] = []
    ngram_compactor = _ExactOccurrenceCompactor()
    for query_index, query in enumerate(values):
        if not isinstance(query, Mapping):
            continue
        retrieval_policy = query.get("retrieval_policy")
        validate_query_retrieval_policy(retrieval_policy)
        bank = str(query.get("bank") or "semantic")
        axes = _axes_for_bank(bank)
        query_id = str(query.get("query_id") or f"query_{query_index + 1}")
        chunks = query.get("top_chunks")
        if isinstance(chunks, Sequence) and not isinstance(chunks, (str, bytes)):
            patient_chunk_counts: Counter[Any] = Counter()
            for chunk_index_in_query, raw in enumerate(chunks):
                if not isinstance(raw, Mapping):
                    continue
                row_id = raw.get("_oci_row_id", raw.get("row_id"))
                if row_id is None:
                    raise ValueError("ranked neural-query chunks require patient row provenance")
                patient_chunk_counts[row_id] += 1
                if patient_chunk_counts[row_id] > retrieval_policy["chunks_per_patient"]:
                    raise ValueError(
                        "neural-query evidence exceeds its declared retrieval top-k; "
                        "regenerate query evidence before compiling cards"
                    )
                chunk_index = raw.get("chunk_index")
                occurrence = _occurrence(
                    text=raw.get("text") or raw.get("chunk_text"),
                    evidence_kind="clinical_text",
                    axes=axes,
                    polarity="positive",
                    source_families=[NEURAL_QUERY_MOMENTS],
                    architecture="neural_query_moments",
                    reference=_base_reference(
                        row,
                        handoff_row=handoff_row,
                        json_path=(f"evidence[{query_index}].top_chunks[{chunk_index_in_query}]"),
                    ),
                    details={
                        "query_id": query_id, "bank": bank,
                        "retrieval_policy": dict(retrieval_policy),
                    },
                    scores=_finite_scores(raw),
                    patient_row_id=row_id,
                    cache_coordinate=(row_id, chunk_index),
                )
                if occurrence is not None:
                    output.append(occurrence)
        for term_index, compact in enumerate(
            _compact_topic_terms(query.get("top_contrastive_ngrams"))
        ):
            occurrence = _occurrence(
                text=compact.get("term"),
                evidence_kind="lexical_term",
                axes=axes,
                polarity="positive",
                source_families=[NEURAL_QUERY_MOMENTS],
                architecture="neural_query_moments",
                reference=_base_reference(
                    row,
                    handoff_row=handoff_row,
                    json_path=(f"evidence[{query_index}].top_contrastive_ngrams[{term_index}]"),
                ),
                details={
                    "term": compact.get("term"), "query_id": query_id, "bank": bank,
                    "retrieval_policy": dict(retrieval_policy),
                    **({"term_normalization": dict(query["term_normalization"])}
                       if isinstance(query.get("term_normalization"), Mapping) else {}),
                },
                scores={
                    # Preserve native term strength and list rank. A query's
                    # fit statistic stays available for audit only.
                    **_finite_scores({"fit_standardized_score": query.get("fit_standardized_score")}),
                    **_finite_scores(compact),
                    "rank": float(compact.get("rank") or term_index + 1),
                },
            )
            if occurrence is not None:
                ngram_compactor.add(occurrence)
    output.extend(ngram_compactor.finish())
    return output


def _extract_generic_occurrence(
    row: Mapping[str, Any],
    payload: Any,
    *,
    handoff_row: int,
) -> list[dict[str, Any]]:
    text = _content_text(payload)
    if not text:
        return []
    architecture = (
        str(payload.get("architecture") or row.get("source") or "structured_evidence")
        if isinstance(payload, Mapping)
        else str(row.get("source") or "structured_evidence")
    )
    occurrence = _occurrence(
        text=text,
        evidence_kind="structured_evidence",
        axes=_axes_from_descriptor(_canonical_json(payload)),
        polarity="unsigned",
        source_families=[_safe_token(row.get("source"), default="unknown_source")],
        architecture=architecture,
        reference=_base_reference(row, handoff_row=handoff_row, json_path="evidence"),
        details={"architecture": architecture},
    )
    return [occurrence] if occurrence is not None else []


def _extract_occurrences(rows: Iterable[Mapping[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    by_outer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    neural_by_outer: dict[int, _ExactOccurrenceCompactor] = {}
    for handoff_row, row in enumerate(rows, start=1):
        try:
            outer_fold = int(row["outer_fold"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"handoff row {handoff_row} has no integer outer_fold") from exc
        payload = row.get("evidence")
        source = str(row.get("source") or "unknown")
        occurrences: list[dict[str, Any]] = []
        if isinstance(payload, Mapping) and source == "stage1_architecture":
            architecture = str(payload.get("architecture") or "").strip()
            occurrence_value = payload.get("occurrence")
            if architecture not in SUPPORTED_STAGE2_ARCHITECTURES:
                raise ValueError(
                    f"handoff row {handoff_row} has unsupported architecture {architecture!r}"
                )
            if not isinstance(occurrence_value, Mapping):
                raise ValueError(
                    f"handoff row {handoff_row} has no canonical architecture occurrence"
                )
            occurrence = copy.deepcopy(dict(occurrence_value))
            if str(occurrence.get("architecture") or "") != architecture:
                raise ValueError(
                    f"handoff row {handoff_row} architecture envelope is inconsistent"
                )
            if architecture == NEURAL_QUERY_MOMENTS:
                validate_query_retrieval_policy(
                    (occurrence.get("details") or {}).get("retrieval_policy")
                )
            _rebind_compact_handoff_references(
                occurrence,
                handoff_row=handoff_row,
            )
            occurrences.append(occurrence)
        elif isinstance(payload, Mapping) and source == "text_models":
            occurrences.extend(_extract_sparse_occurrences(row, payload, handoff_row=handoff_row))
            occurrences.extend(
                _extract_embedding_occurrences(row, payload, handoff_row=handoff_row)
            )
            occurrences.extend(_extract_htr_occurrences(row, payload, handoff_row=handoff_row))
        elif isinstance(payload, Mapping) and source == "tfidf":
            occurrences.extend(_extract_tfidf_occurrences(row, payload, handoff_row=handoff_row))
        elif isinstance(payload, Mapping) and source == "neural_queries":
            occurrences.extend(
                _extract_neural_query_occurrences(row, payload, handoff_row=handoff_row)
            )
        if not occurrences:
            occurrences = _extract_generic_occurrence(
                row,
                payload,
                handoff_row=handoff_row,
            )
        for occurrence in occurrences:
            if str(occurrence["architecture"]) == NEURAL_QUERY_MOMENTS:
                compactor = neural_by_outer.get(outer_fold)
                if compactor is None:
                    compactor = _ExactOccurrenceCompactor()
                    neural_by_outer[outer_fold] = compactor
                compactor.add(occurrence)
            else:
                by_outer[outer_fold].append(occurrence)
    for outer_fold, compactor in neural_by_outer.items():
        by_outer[outer_fold].extend(compactor.finish())
    if not by_outer or not any(by_outer.values()):
        raise ValueError("the Stage 1 handoff contains no compilable scientific evidence")
    return by_outer


def extract_stage1_architecture_occurrences(
    rows: Iterable[Mapping[str, Any]],
    *,
    included_architectures: Sequence[str] | None = None,
) -> dict[int, tuple[dict[str, Any], ...]]:
    """Project raw handoff rows into the canonical per-architecture evidence form."""

    included = (
        set(SUPPORTED_STAGE2_ARCHITECTURES)
        if included_architectures is None
        else set(map(str, included_architectures))
    )
    unsupported = sorted(included - set(SUPPORTED_STAGE2_ARCHITECTURES))
    if unsupported:
        raise ValueError(f"unsupported included Stage 1 architectures: {unsupported}")
    by_outer = _extract_occurrences(rows)
    return {
        outer_fold: tuple(
            copy.deepcopy(occurrence)
            for occurrence in occurrences
            if str(occurrence["architecture"]) in included
        )
        for outer_fold, occurrences in sorted(by_outer.items())
    }


def _aggregate_exact_occurrences(
    occurrences: Sequence[Mapping[str, Any]],
    *,
    embedding_cache: _ChunkEmbeddingCache | None,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for occurrence in occurrences:
        grouped[_exact_occurrence_key(occurrence)].append(occurrence)
    members: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items()):
        member_id = f"member_{_sha256_text(key)[:20]}"
        axes = sorted({axis for value in values for axis in value["axes"]})
        polarities = sorted(
            {polarity for value in values for polarity in _occurrence_polarities(value)}
        )
        families = sorted({family for value in values for family in value["source_families"]})
        architectures = sorted({str(value["architecture"]) for value in values})
        references = _merge_reference_summaries(values)
        raw_occurrence_count = sum(_occurrence_count(value) for value in values)
        cache_index = None
        if embedding_cache is not None:
            for value in values:
                coordinate = value.get("cache_coordinate")
                if coordinate is None:
                    continue
                cache_index = embedding_cache.flat_index(*coordinate)
                if cache_index is not None:
                    break
        members.append(
            {
                "member_id": member_id,
                "text": str(values[0]["text"]),
                "text_sha256": _sha256_text(str(values[0]["text"])),
                "evidence_kind": str(values[0]["evidence_kind"]),
                "evidence_axes": axes,
                "polarities": polarities,
                "source_families": families,
                "source_architectures": architectures,
                "occurrences": [dict(value) for value in values],
                "raw_references": references,
                "raw_occurrence_count": raw_occurrence_count,
                "cache_index": cache_index,
            }
        )
    return members


def _allocate_card_counts(
    group_sizes: Sequence[int],
    *,
    total_cards: int,
    minimum_per_group: int = 4,
) -> list[int]:
    if not group_sizes or any(size < 1 for size in group_sizes):
        raise ValueError("semantic card allocation requires nonempty groups")
    total_cards = min(sum(group_sizes), max(len(group_sizes), int(total_cards)))
    base = [min(size, max(1, int(minimum_per_group))) for size in group_sizes]
    if sum(base) > total_cards:
        base = [1 for _size in group_sizes]
    remaining = total_cards - sum(base)
    capacities = [size - count for size, count in zip(group_sizes, base)]
    if remaining <= 0 or not sum(capacities):
        return base
    raw = [remaining * capacity / sum(capacities) for capacity in capacities]
    additions = [min(capacity, math.floor(value)) for capacity, value in zip(capacities, raw)]
    result = [count + addition for count, addition in zip(base, additions)]
    leftover = total_cards - sum(result)
    order = sorted(
        range(len(group_sizes)),
        key=lambda index: (raw[index] - additions[index], capacities[index], -index),
        reverse=True,
    )
    while leftover:
        for index in order:
            if result[index] >= group_sizes[index]:
                continue
            result[index] += 1
            leftover -= 1
            if not leftover:
                break
    return result


def _semantic_matrix(
    members: Sequence[Mapping[str, Any]],
    *,
    embedding_cache: _ChunkEmbeddingCache | None,
    mode: str,
    seed: int,
) -> np.ndarray:
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.preprocessing import normalize
    from sklearn.random_projection import SparseRandomProjection

    if mode == "cached_embedding":
        if embedding_cache is None:
            raise RuntimeError("cached semantic grouping requested without an embedding cache")
        raw = embedding_cache.vectors([int(member["cache_index"]) for member in members])
        n_components = min(128, raw.shape[1])
        if raw.shape[1] > n_components:
            raw = SparseRandomProjection(
                n_components=n_components,
                dense_output=True,
                random_state=seed,
            ).fit_transform(raw)
        return np.asarray(normalize(raw), dtype=np.float32)
    sparse = HashingVectorizer(
        n_features=2**15,
        alternate_sign=False,
        analyzer="char_wb",
        ngram_range=(3, 5),
        norm="l2",
    ).transform([str(member["text"]) for member in members])
    projected = SparseRandomProjection(
        n_components=128,
        dense_output=True,
        random_state=seed,
    ).fit_transform(sparse)
    return np.asarray(normalize(projected), dtype=np.float32)


def _cluster_members(
    members: Sequence[Mapping[str, Any]],
    *,
    cluster_count: int,
    embedding_cache: _ChunkEmbeddingCache | None,
    mode: str,
    seed: int,
) -> list[tuple[list[Mapping[str, Any]], np.ndarray | None, np.ndarray | None]]:
    ordered = sorted(members, key=lambda member: str(member["member_id"]))
    if cluster_count >= len(ordered):
        return [([member], None, None) for member in ordered]
    matrix = _semantic_matrix(
        ordered,
        embedding_cache=embedding_cache,
        mode=mode,
        seed=seed,
    )
    if cluster_count == 1:
        labels = np.zeros(len(ordered), dtype=int)
        centers = np.mean(matrix, axis=0, keepdims=True)
    else:
        from sklearn.cluster import MiniBatchKMeans

        model = MiniBatchKMeans(
            n_clusters=cluster_count,
            random_state=seed,
            n_init=3,
            batch_size=min(1024, max(32, len(ordered))),
            max_iter=100,
            reassignment_ratio=0.0,
        )
        labels = model.fit_predict(matrix)
        centers = model.cluster_centers_
    output: list[tuple[list[Mapping[str, Any]], np.ndarray | None, np.ndarray | None]] = []
    for label in sorted(set(map(int, labels))):
        indices = np.flatnonzero(labels == label)
        cluster = [ordered[int(index)] for index in indices]
        output.append((cluster, matrix[indices], np.asarray(centers[label])))
    output.sort(key=lambda item: min(str(member["member_id"]) for member in item[0]))
    return output


def _score_summary(members: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values: dict[str, dict[float, int]] = defaultdict(lambda: defaultdict(int))
    for member in members:
        for occurrence in member["occurrences"]:
            for key, value_counts in _occurrence_score_value_counts(occurrence).items():
                for value, count in value_counts.items():
                    values[str(key)][float(value)] += int(count)

    def weighted_value_at(value_counts: Mapping[float, int], position: int) -> float:
        seen = 0
        for value, count in sorted(value_counts.items()):
            seen += int(count)
            if position < seen:
                return float(value)
        raise RuntimeError("weighted score position exceeds its encoded count")

    summary: dict[str, Any] = {}
    for key, value_counts in sorted(values.items()):
        if not value_counts:
            continue
        total = sum(value_counts.values())
        upper = weighted_value_at(value_counts, total // 2)
        median = upper
        if total % 2 == 0:
            lower = weighted_value_at(value_counts, total // 2 - 1)
            median = (lower + upper) / 2.0
        ordered_values = sorted(value_counts)
        summary[key] = {
            "minimum": ordered_values[0],
            "median": median,
            "maximum": ordered_values[-1],
        }
    return summary


def _truncate_exemplar(text: str, *, max_chars: int) -> dict[str, Any]:
    if len(text) <= max_chars:
        return {"text": text, "text_truncated": False}
    head = max(1, int(max_chars * 0.7))
    tail = max(1, max_chars - head)
    return {
        "text": f"{text[:head]} ... [middle omitted; full text retained in Stage 1] ... {text[-tail:]}",
        "text_truncated": True,
        "full_text_sha256": _sha256_text(text),
        "full_text_chars": len(text),
    }


def _build_card(
    *,
    outer_fold: int,
    members: Sequence[Mapping[str, Any]],
    matrix: np.ndarray | None,
    center: np.ndarray | None,
    semantic_mode: str,
    max_exemplars: int,
    max_exemplar_chars: int,
    sentence_index: SupportingSentenceIndex | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    member_ids = sorted(str(member["member_id"]) for member in members)
    digest = _sha256_text(_canonical_json(member_ids))[:20]
    card_id = f"outer_{outer_fold:03d}_card_{digest}"
    exemplars = _select_exemplars(
        members,
        matrix,
        center,
        limit=max(1, int(max_exemplars)),
    )
    references = [reference for member in members for reference in member["raw_references"]]
    raw_occurrence_count = sum(int(member["raw_occurrence_count"]) for member in members)
    inner_folds = sorted(
        {
            int(reference["inner_fold"])
            for reference in references
            if reference.get("inner_fold") is not None
        }
    )
    patient_ids = {
        int(reference["row_id"]) for reference in references if reference.get("row_id") is not None
    }
    axes = sorted({axis for member in members for axis in member["evidence_axes"]})
    families = sorted({family for member in members for family in member["source_families"]})
    architectures = sorted(
        {architecture for member in members for architecture in member["source_architectures"]}
    )
    polarities = sorted({polarity for member in members for polarity in member["polarities"]})
    representative_evidence = []
    selected_audit = []
    shown_sentences: set[str] = set()
    for member in exemplars:
        supporting_sentences = sentence_index.find(member) if sentence_index is not None else []
        sentences = [sentence for sentence in supporting_sentences if sentence["text"] not in shown_sentences]
        shown_sentences.update(str(sentence["text"]) for sentence in sentences)
        text = str(member["text"])
        if sentences:
            text += "\nExamples from clinical records:\n" + "\n\n".join(
                str(sentence["text"]) for sentence in sentences
            )
        rendered = _truncate_exemplar(
            text,
            max_chars=max(256, int(max_exemplar_chars)),
        )
        occurrence = max(
            member["occurrences"],
            key=lambda value: (
                len(value.get("scores") or {}),
                len(value.get("details") or {}),
            ),
        )
        representative_evidence.append(
            {
                **rendered,
                "source_families": list(member["source_families"]),
                "source_architectures": list(member["source_architectures"]),
                "evidence_axes": list(member["evidence_axes"]),
                "polarities": list(member["polarities"]),
                "supporting_context_count": len(
                    {(ref.get("scope"), ref.get("inner_fold")) for ref in member["raw_references"]}
                ),
                "details": dict(occurrence.get("details") or {}),
            }
        )
        selected_audit.append({
            "member_id": member["member_id"],
            "selection_strength": dict(member.get("selection_strength") or {}),
            "supporting_sentences": supporting_sentences,
            "displayed_supporting_sentence_count": len(sentences),
        })
    card = {
        "schema_version": EVIDENCE_COMPILER_VERSION,
        "card_id": card_id,
        "evidence_kind": str(members[0]["evidence_kind"]),
        "evidence_axes": axes,
        "polarities": polarities,
        "source_families": families,
        "source_architectures": architectures,
        "support": {
            "exact_member_count": len(members),
            "raw_occurrence_count": raw_occurrence_count,
            "patient_count": len(patient_ids),
            "inner_folds": inner_folds,
            "full_outer_train_support": any(
                str(reference.get("scope")) == "full_outer_train" for reference in references
            ),
        },
        "semantic_grouping": semantic_mode,
        "score_summary": _score_summary(members),
        "representative_evidence": representative_evidence,
    }
    lineage = {
        "card_id": card_id,
        "member_ids": member_ids,
        "raw_occurrence_count": raw_occurrence_count,
        "selection_policy": SELECTION_POLICY,
        "selected_exemplars": selected_audit,
    }
    return card, lineage


def _prompt_architecture(card: Mapping[str, Any]) -> str:
    architectures = list(card.get("source_architectures") or [])
    if len(architectures) != 1:
        raise ValueError(
            "compiled Stage 2 cards must contain exactly one Stage 1 architecture; "
            f"received {architectures}"
        )
    return str(architectures[0])


def _fit_packet_to_budget(
    packet: Mapping[str, Any],
    *,
    max_packet_chars: int,
) -> dict[str, Any]:
    """Bound a card packet without discarding its audit lineage.

    The complete member-to-raw lineage is written outside the prompt.  This
    method may therefore shorten representative text and optional summaries,
    but it never changes the card ID, axes, source families, or support counts.
    """

    fitted = json.loads(_canonical_json(packet))

    def chars() -> int:
        return len(_canonical_json(fitted))

    representatives = fitted["content"]["representative_evidence"]
    while chars() > max_packet_chars and len(representatives) > 1:
        representatives.pop()
    for representative in representatives:
        if chars() <= max_packet_chars:
            break
        representative["details"] = {}
        text = str(representative.get("text") or "")
        while chars() > max_packet_chars and len(text) > 96:
            overflow = chars() - max_packet_chars
            keep = max(96, len(text) - overflow - 48)
            text = text[:keep].rstrip() + " ... [prompt card truncated]"
            representative["text"] = text
            representative["text_truncated"] = True
    if chars() > max_packet_chars:
        fitted["content"]["score_summary"] = {}
    if chars() > max_packet_chars:
        raise ValueError(
            "Stage 2 evidence card cannot fit max_packet_chars even after "
            f"representative-text compaction ({chars()} > {max_packet_chars})"
        )
    return fitted


def compile_stage2_handoff_evidence(
    rows: Iterable[Mapping[str, Any]],
    *,
    handoff_path: Path,
    max_cards_per_outer_fold: int = 400,
    max_exemplars_per_card: int = 4,
    max_exemplar_chars: int = 2_400,
    max_packet_chars: int = 25_000,
    seed: int = 42,
    required_architectures: Sequence[str] = (),
    included_architectures: Sequence[str] | None = None,
) -> CompiledStage2Evidence:
    """Compile raw Stage 1 rows into fold-local semantic evidence cards."""

    if max_cards_per_outer_fold < 16:
        raise ValueError("Stage 2 evidence compilation requires at least 16 cards per fold")
    if max_exemplars_per_card < 1:
        raise ValueError("Stage 2 evidence compilation requires at least one exemplar")
    if max_exemplar_chars < 256:
        raise ValueError("Stage 2 evidence exemplar limit must be at least 256 characters")
    if max_packet_chars < 1_200:
        raise ValueError("Stage 2 evidence packet limit must be at least 1200 characters")
    required = tuple(dict.fromkeys(str(value).strip() for value in required_architectures))
    if any(not value for value in required):
        raise ValueError("required Stage 2 architecture names must be nonempty")
    unsupported = sorted(set(required) - set(SUPPORTED_STAGE2_ARCHITECTURES))
    if unsupported:
        raise ValueError(f"unsupported required Stage 2 architectures: {unsupported}")
    included = (
        tuple(dict.fromkeys(str(value).strip() for value in included_architectures))
        if included_architectures is not None
        else None
    )
    if included is not None:
        if any(not value for value in included):
            raise ValueError("included Stage 2 architecture names must be nonempty")
        unsupported_included = sorted(
            set(included) - set(SUPPORTED_STAGE2_ARCHITECTURES)
        )
        if unsupported_included:
            raise ValueError(
                f"unsupported included Stage 2 architectures: {unsupported_included}"
            )
        if not set(required).issubset(included):
            raise ValueError(
                "required Stage 2 architectures must be included in the compiler selection"
            )
    embedding_cache = _ChunkEmbeddingCache.discover(Path(handoff_path))
    occurrences_by_outer = _extract_occurrences(rows)
    if included is not None:
        included_set = set(included)
        occurrences_by_outer = {
            outer_fold: [
                occurrence
                for occurrence in occurrences
                if str(occurrence["architecture"]) in included_set
            ]
            for outer_fold, occurrences in occurrences_by_outer.items()
        }
    packets: list[dict[str, Any]] = []
    cards_by_outer: dict[int, tuple[dict[str, Any], ...]] = {}
    members_by_outer: dict[int, tuple[dict[str, Any], ...]] = {}
    lineage_by_outer: dict[int, tuple[dict[str, Any], ...]] = {}
    fold_summaries: dict[str, Any] = {}
    for outer_fold in sorted(occurrences_by_outer):
        occurrences = occurrences_by_outer[outer_fold]
        raw_occurrence_count = sum(_occurrence_count(value) for value in occurrences)
        present_architectures = {
            str(occurrence["architecture"]) for occurrence in occurrences
        }
        missing_architectures = [
            architecture
            for architecture in required
            if architecture not in present_architectures
        ]
        if missing_architectures:
            raise ValueError(
                f"Stage 2 outer_fold={int(outer_fold)} is missing enabled Stage 1 "
                f"architectures {missing_architectures}; present architectures are "
                f"{sorted(present_architectures)}. Rerun the corresponding Stage 1 "
                "component(s) and rebuild the handoff before interpretation."
            )
        members = _aggregate_exact_occurrences(
            occurrences,
            embedding_cache=embedding_cache,
        )
        annotate_strength(members)
        sentence_index = SupportingSentenceIndex(members)
        grouped: dict[
            tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...], str],
            list[dict[str, Any]],
        ] = defaultdict(list)
        for member in members:
            mode = (
                "cached_embedding"
                if member.get("cache_index") is not None
                else "lexical_hash_projection"
            )
            key = (
                str(member["evidence_kind"]),
                tuple(member["evidence_axes"]),
                tuple(member["polarities"]),
                tuple(member["source_architectures"]),
                mode,
            )
            grouped[key].append(member)
        group_items = sorted(grouped.items(), key=lambda item: item[0])
        allocations = _allocate_card_counts(
            [len(values) for _key, values in group_items],
            total_cards=max_cards_per_outer_fold,
        )
        compiled: list[tuple[dict[str, Any], dict[str, Any]]] = []
        group_audit: list[dict[str, Any]] = []
        for group_index, ((key, group_members), cluster_count) in enumerate(
            zip(group_items, allocations),
            start=1,
        ):
            mode = key[-1]
            group_seed = (
                int(seed)
                + 10_000 * int(outer_fold)
                + int(_sha256_text(_canonical_json(key))[:8], 16)
            )
            clusters = _cluster_members(
                group_members,
                cluster_count=cluster_count,
                embedding_cache=embedding_cache,
                mode="cached_embedding" if mode == "cached_embedding" else "lexical",
                seed=group_seed,
            )
            for cluster_members, matrix, center in clusters:
                compiled.append(
                    _build_card(
                        outer_fold=outer_fold,
                        members=cluster_members,
                        matrix=matrix,
                        center=center,
                        semantic_mode=mode,
                        max_exemplars=max_exemplars_per_card,
                        max_exemplar_chars=max_exemplar_chars,
                        sentence_index=sentence_index,
                    )
                )
            group_audit.append(
                {
                    "group_index": group_index,
                    "evidence_kind": key[0],
                    "evidence_axes": list(key[1]),
                    "polarities": list(key[2]),
                    "source_architectures": list(key[3]),
                    "semantic_mode": mode,
                    "exact_member_count": len(group_members),
                    "allocated_card_count": cluster_count,
                    "actual_card_count": len(clusters),
                }
            )
        compiled.sort(key=lambda item: str(item[0]["card_id"]))
        cards = [card for card, _lineage in compiled]
        lineage = [lineage for _card, lineage in compiled]
        fitted_cards: list[dict[str, Any]] = []
        fold_packets: list[dict[str, Any]] = []
        for card, card_lineage in zip(cards, lineage):
            packet = _fit_packet_to_budget(
                {
                    "packet_id": str(card["card_id"]),
                    "source": "compiled_stage1_evidence",
                    "architecture": _prompt_architecture(card),
                    "outer_fold": int(outer_fold),
                    "inner_fold": None,
                    "scope": "outer_fold_compiled_training_evidence",
                    "json_path": str(card["card_id"]),
                    "observable_axes": list(card["evidence_axes"]),
                    "content": card,
                },
                max_packet_chars=int(max_packet_chars),
            )
            fitted_cards.append(dict(packet["content"]))
            card_lineage["prompt_exemplar_text_sha256"] = [
                _sha256_text(str(item["text"]))
                for item in packet["content"]["representative_evidence"]
            ]
            fold_packets.append(packet)
            packets.append(packet)
        public_members = tuple(
            {
                "member_id": str(member["member_id"]),
                "text_sha256": str(member["text_sha256"]),
                "evidence_kind": str(member["evidence_kind"]),
                "evidence_axes": list(member["evidence_axes"]),
                "polarities": list(member["polarities"]),
                "source_families": list(member["source_families"]),
                "source_architectures": list(member["source_architectures"]),
                "raw_references": list(member["raw_references"]),
                "raw_occurrence_count": int(member["raw_occurrence_count"]),
                "selection_strength": dict(member["selection_strength"]),
            }
            for member in members
        )
        cards_by_outer[outer_fold] = tuple(fitted_cards)
        members_by_outer[outer_fold] = public_members
        lineage_by_outer[outer_fold] = tuple(lineage)
        fold_summaries[str(outer_fold)] = {
            "raw_occurrences": raw_occurrence_count,
            "compact_occurrence_records": len(occurrences),
            "exact_members": len(members),
            "cards": len(fitted_cards),
            "exact_duplicate_occurrences_removed": raw_occurrence_count - len(members),
            "prompt_packet_chars": sum(len(_canonical_json(packet)) for packet in fold_packets),
            "groups": group_audit,
            "source_family_occurrences": dict(
                sorted(
                    {
                        family: sum(
                            _occurrence_count(occurrence)
                            for occurrence in occurrences
                            if family in occurrence["source_families"]
                        )
                        for family in {
                            family
                            for occurrence in occurrences
                            for family in occurrence["source_families"]
                        }
                    }.items()
                )
            ),
            "architecture_occurrences": dict(
                sorted(
                    {
                        architecture: sum(
                            _occurrence_count(occurrence)
                            for occurrence in occurrences
                            if str(occurrence["architecture"]) == architecture
                        )
                        for architecture in {
                            str(occurrence["architecture"])
                            for occurrence in occurrences
                        }
                    }.items()
                )
            ),
            "architecture_packets": dict(
                sorted(Counter(str(packet["architecture"]) for packet in fold_packets).items())
            ),
        }
    summary = {
        "schema_version": EVIDENCE_COMPILER_VERSION,
        "selection_policy": SELECTION_POLICY,
        "required_architectures": list(required),
        "included_architectures": None if included is None else list(included),
        "embedding_cache": str(embedding_cache.directory) if embedding_cache else None,
        "embedding_cache_model": (
            embedding_cache.metadata.get("sentence_model_name") if embedding_cache else None
        ),
        "max_cards_per_outer_fold": int(max_cards_per_outer_fold),
        "max_exemplars_per_card": int(max_exemplars_per_card),
        "max_exemplar_chars": int(max_exemplar_chars),
        "max_packet_chars": int(max_packet_chars),
        "outer_folds": fold_summaries,
        "packets": len(packets),
    }
    return CompiledStage2Evidence(
        packets=tuple(packets),
        cards_by_outer_fold=cards_by_outer,
        members_by_outer_fold=members_by_outer,
        lineage_by_outer_fold=lineage_by_outer,
        summary=summary,
    )


__all__ = [
    "CompiledStage2Evidence",
    "EVIDENCE_COMPILER_VERSION",
    "SUPPORTED_STAGE2_ARCHITECTURES",
    "compile_stage2_handoff_evidence",
    "extract_stage1_architecture_occurrences",
    "stage1_embedding_cache_dependency_identity",
]
