"""Post-hoc oracle evaluation for each research Stage 1 architecture."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, roc_auc_score

from ..inference.stage1_architecture_artifacts import (
    EMBEDDING_DIRECTION_SCORE_SOURCES,
    embedding_direction_from_score_column,
    embedding_direction_slug,
    materialize_stage1_architecture_artifacts,
)
from ..inference.stage1_architectures import (
    BOW_NUISANCE,
    BOW_R_LOSS,
    EMBEDDING_CLUSTERED,
    EMBEDDING_WHOLE_COHORT,
    HTR_NEURAL,
    MATCHED_PAIR_UPLIFT,
    NEURAL_QUERY_MOMENTS,
    STAGE1_ARCHITECTURES,
    STAGE1_ARCHITECTURE_REGISTRY,
    TFIDF_ORPHAN_NGRAMS,
    TFIDF_SEMANTIC_RETRIEVAL,
    TFIDF_TOPICS,
    canonicalize_stage1_architectures,
)
from .stage1_lexical_null import (
    TrainingTextCorpus,
    null_seed,
    random_chunk_null,
    summarize_null,
)
from .stage1_timing_report import (
    architecture_context_seconds,
    load_timing_records,
    timing_paths,
    timing_summary,
    timing_table,
    untimed_component_records,
)

# v2: embedding row scores are the handed-off directions; lexical-recovery
# null comparisons and per-architecture wall time are reported.
EVALUATION_METRIC_SCHEMA_VERSION = "stage1_architecture_evaluation_metric_v2"
EVALUATION_MANIFEST_SCHEMA_VERSION = "stage1_architecture_evaluation_manifest_v2"
LEXICAL_NULL_SCHEMA_VERSION = "stage1_lexical_recovery_random_chunk_null_v1"
DEFAULT_NULL_DRAWS = 200
_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "at",
        "baseline",
        "by",
        "clinical",
        "for",
        "in",
        "level",
        "measured",
        "of",
        "or",
        "patient",
        "presence",
        "status",
        "the",
        "to",
        "treatment",
        "value",
        "with",
    }
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, allow_nan=False, default=str) + "\n")
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected one JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_dataset(run_dir: Path, run_config: Mapping[str, Any], override: Path | None) -> Path:
    if override is not None:
        return override.expanduser().resolve(strict=True)
    value = run_config.get("dataset")
    if not value:
        raise ValueError("saved run_config.json has no dataset path; pass --dataset")
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = (run_dir / path).resolve()
    return path.resolve(strict=True)


def _resolve_metadata(dataset: Path, override: Path | None) -> Path | None:
    if override is not None:
        return override.expanduser().resolve(strict=True)
    candidate = dataset.parent / "metadata.json"
    return candidate if candidate.is_file() else None


def _load_dataset(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        raise ValueError("evaluation dataset must be Parquet or CSV")
    frame = frame.reset_index(drop=True)
    frame["_oci_row_id"] = np.arange(len(frame), dtype=int)
    return frame


def _metadata_features(metadata: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = metadata.get("features")
    if not isinstance(raw, list):
        raw = [
            *list(metadata.get("confounders") or []),
            *list(metadata.get("effect_modifiers") or []),
        ]
    output: dict[str, dict[str, Any]] = {}
    for value in raw:
        if not isinstance(value, Mapping) or not str(value.get("name") or "").strip():
            continue
        item = dict(value)
        name = str(item["name"])
        roles = list(item.get("roles") or [])
        if not roles:
            if value in list(metadata.get("confounders") or []):
                roles.append("confounder")
            if value in list(metadata.get("effect_modifiers") or []):
                roles.append("effect_modifier")
        item["roles"] = roles
        output[name] = item
    return list(output.values())


def _feature_tokens(feature: Mapping[str, Any]) -> set[str]:
    values = [feature.get("name"), feature.get("description")]
    values.extend(feature.get("categories") or [])
    return {
        token
        for value in values
        for token in _TOKEN.findall(str(value or "").lower())
        if token not in _STOPWORDS and len(token) > 1
    }


def _finite_pair(left: Any, right: Any) -> tuple[np.ndarray, np.ndarray]:
    left_values = pd.to_numeric(pd.Series(left), errors="coerce").to_numpy(dtype=float)
    right_values = pd.to_numeric(pd.Series(right), errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(left_values) & np.isfinite(right_values)
    return left_values[mask], right_values[mask]


def _safe_abs_pearson(left: Any, right: Any) -> tuple[float | None, int]:
    x, y = _finite_pair(left, right)
    if len(x) < 3 or np.std(x) <= 0 or np.std(y) <= 0:
        return None, len(x)
    return abs(float(np.corrcoef(x, y)[0, 1])), len(x)


def _safe_abs_spearman(left: Any, right: Any) -> tuple[float | None, int]:
    x, y = _finite_pair(left, right)
    if len(x) < 3 or np.std(x) <= 0 or np.std(y) <= 0:
        return None, len(x)
    value = float(spearmanr(x, y).statistic)
    return (abs(value) if np.isfinite(value) else None), len(x)


def _orientation_free_auc(score: Any, target: Any) -> tuple[float | None, int]:
    score_values = pd.to_numeric(pd.Series(score), errors="coerce").to_numpy(dtype=float)
    target_series = pd.Series(target)
    valid = np.isfinite(score_values) & target_series.notna().to_numpy()
    score_values = score_values[valid]
    target_values = target_series.iloc[np.flatnonzero(valid)].astype(str)
    best: float | None = None
    for level in sorted(target_values.unique()):
        binary = (target_values == level).to_numpy(dtype=int)
        if len(np.unique(binary)) < 2:
            continue
        auc = float(roc_auc_score(binary, score_values))
        value = max(auc, 1.0 - auc)
        best = value if best is None else max(best, value)
    return best, len(score_values)


@dataclass(frozen=True)
class _MetricFactory:
    architecture: str
    source_artifact: str | None = None
    source_sha256: str | None = None

    def row(
        self,
        *,
        metric_family: str,
        metric: str,
        value: float | None,
        target: str | None = None,
        score: str | None = None,
        n: int | None = None,
        outer_fold: int | None = None,
        inner_fold: int | None = None,
        scope: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        applicable = value is not None and np.isfinite(float(value))
        return {
            "schema_version": EVALUATION_METRIC_SCHEMA_VERSION,
            "architecture": self.architecture,
            "outer_fold": outer_fold,
            "inner_fold": inner_fold,
            "scope": scope,
            "metric_family": metric_family,
            "metric": metric,
            "target": target,
            "score": score,
            "value": float(value) if applicable else None,
            "n": n,
            "applicability": "applicable" if applicable else "not_applicable",
            "reason": None if applicable else (reason or "insufficient usable data"),
            "source_artifact": self.source_artifact,
            "source_sha256": self.source_sha256,
        }


def _materialize_or_load_manifest(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "stage1_architectures" / "manifest.json"
    if manifest_path.is_file() and all(
        (run_dir / "stage1_architectures" / name / "evidence.jsonl").is_file()
        for name in _read_json(manifest_path).get("selected_architectures", [])
    ):
        return _read_json(manifest_path)
    handoff_path = run_dir / "handoff" / "evidence.jsonl"
    if not handoff_path.is_file():
        raise FileNotFoundError(
            "Stage 1 evaluation requires a completed handoff or architecture sidecars: "
            f"{handoff_path}"
        )
    rows = _read_jsonl(handoff_path)
    from ..inference.plain_handoff_stage2_evidence import (
        extract_stage1_architecture_occurrences,
    )

    by_outer = extract_stage1_architecture_occurrences(rows)
    present = {
        str(occurrence["architecture"])
        for occurrences in by_outer.values()
        for occurrence in occurrences
    }
    selected = tuple(name for name in STAGE1_ARCHITECTURES if name in present)
    source_artifacts: dict[str, Path] = {}
    index_path = handoff_path.parent / "index.json"
    if index_path.is_file():
        try:
            source_references = _read_json(index_path).get("sources") or {}
        except (OSError, ValueError, json.JSONDecodeError):
            source_references = {}
        if isinstance(source_references, Mapping):
            for source, raw_path in source_references.items():
                if not isinstance(raw_path, (str, os.PathLike)):
                    continue
                path = Path(raw_path).expanduser()
                if not path.is_absolute():
                    path = handoff_path.parent / path
                try:
                    path = path.resolve(strict=True)
                except (OSError, RuntimeError):
                    continue
                if path.is_file():
                    source_artifacts[str(source)] = path
    _rows, manifest = materialize_stage1_architecture_artifacts(
        output_dir=run_dir,
        raw_handoff_rows=rows,
        selected_architectures=selected,
        source_artifacts=source_artifacts,
        selection_mode="legacy_backfill",
    )
    return dict(manifest)


def _load_evidence(run_dir: Path, architecture: str) -> list[dict[str, Any]]:
    path = run_dir / "stage1_architectures" / architecture / "evidence.jsonl"
    return _read_jsonl(path) if path.is_file() else []


def _load_score_frames(
    run_dir: Path,
    architecture: str,
    manifest: Mapping[str, Any],
    cache: dict[Path, pd.DataFrame] | None = None,
    legacy_paths: list[Path] | None = None,
) -> list[tuple[Path, pd.DataFrame, pd.DataFrame]]:
    """Load an architecture's row-score frames.

    Embedding architectures keep only the scores of their handed-off
    directions.  Rows from the retired duplicate direction pipeline (older
    runs) are dropped, and their files are listed in ``legacy_paths``.
    """

    architecture_entry = (manifest.get("architectures") or {}).get(architecture) or {}
    output: list[tuple[Path, pd.DataFrame, pd.DataFrame]] = []
    cache = cache if cache is not None else {}
    handed_off_source = EMBEDDING_DIRECTION_SCORE_SOURCES.get(architecture)
    for relative in architecture_entry.get("score_artifacts") or []:
        path = run_dir / str(relative)
        if not path.is_file():
            continue
        if path not in cache:
            try:
                cache[path] = pd.read_parquet(path)
            except (OSError, ValueError):
                continue
        raw_frame = cache[path]
        frame = raw_frame
        if "_oci_row_id" not in frame.columns:
            continue
        if "architecture" in frame.columns:
            frame = frame.loc[frame["architecture"] == architecture].copy()
        if handed_off_source is not None and not frame.empty:
            handed_off = (
                frame.loc[frame["source_name"] == handed_off_source]
                if "source_name" in frame.columns
                else frame.iloc[0:0]
            )
            if handed_off.empty and legacy_paths is not None:
                legacy_paths.append(path)
            # Other producers' score columns are empty on these rows.
            frame = handed_off.drop(
                columns=[
                    column
                    for column in _score_columns(handed_off)
                    if handed_off[column].isna().all()
                ]
            ).copy()
        if frame.empty:
            continue
        output.append((path, frame, raw_frame))
    return output


def _annotate_metric_source(
    rows: Iterable[dict[str, Any]],
    *,
    source_artifact: str,
    source_sha256: str,
    frame: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    context: dict[str, Any] = {}
    if frame is not None:
        for column in ("outer_fold", "inner_fold", "scope"):
            if column not in frame.columns:
                continue
            values = frame[column].drop_duplicates().tolist()
            if len(values) != 1:
                continue
            value = values[0]
            if pd.isna(value):
                value = None
            elif column in {"outer_fold", "inner_fold"}:
                value = int(value)
            else:
                value = str(value)
            context[column] = value
    output = []
    for raw in rows:
        row = dict(raw)
        row["source_artifact"] = source_artifact
        row["source_sha256"] = source_sha256
        for key, value in context.items():
            if row.get(key) is None:
                row[key] = value
        output.append(row)
    return output


def _outer_heldout(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame
    outer_context_columns = {
        "scope",
        "inner_fold",
        "split_role",
        "honest_outer_holdout",
    }
    if "prediction_scope" in selected.columns and not (
        outer_context_columns & set(selected.columns)
    ):
        # external_heldout means only "held out from this fit".  Without an
        # enclosing scope it also describes candidate-selection inner folds,
        # so it is not sufficient evidence of an outer holdout.
        return selected.iloc[0:0].copy()
    if "scope" in selected.columns:
        selected = selected.loc[selected["scope"] == "full_outer_train"]
    if "inner_fold" in selected.columns:
        selected = selected.loc[selected["inner_fold"].isna()]
    if "honest_outer_holdout" in selected.columns:
        selected = selected.loc[selected["honest_outer_holdout"] == True]  # noqa: E712
    if "split_role" in selected.columns:
        heldout_roles = {"test_outer_train_fit", "heldout_query_projection"}
        selected = selected.loc[selected["split_role"].isin(heldout_roles)]
    if "prediction_scope" in selected.columns:
        selected = selected.loc[selected["prediction_scope"] == "external_heldout"]
    return selected.copy()


def _score_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {
        "_oci_row_id",
        "architecture",
        "estimation_provenance",
        "fit_row_ids",
        "fold_key",
        "honest_outer_holdout",
        "inner_fold",
        "nuisance_fold",
        "outer_fold",
        "prediction_scope",
        "scope",
        "source_name",
        "split_role",
        "target_source",
    }
    return [
        column
        for column in frame.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
    ]


def _occurrence_count(occurrence: Mapping[str, Any]) -> int:
    raw = occurrence.get("raw_occurrence_count", 1)
    if isinstance(raw, bool):
        raise ValueError("evidence raw_occurrence_count must be a positive integer")
    try:
        count = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("evidence raw_occurrence_count must be a positive integer") from exc
    if count < 1:
        raise ValueError("evidence raw_occurrence_count must be a positive integer")
    return count


def _occurrence_reference_summaries(
    occurrence: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return occurrence-aware references for compact schema v2 or legacy rows."""

    raw_summaries = occurrence.get("reference_summaries")
    if isinstance(raw_summaries, Sequence) and not isinstance(
        raw_summaries, (str, bytes)
    ):
        summaries = [dict(value) for value in raw_summaries if isinstance(value, Mapping)]
    else:
        summaries = [dict(occurrence.get("reference") or {})]
        summaries[0]["occurrence_count"] = _occurrence_count(occurrence)
    details = occurrence.get("details")
    if isinstance(details, Mapping):
        for summary in summaries:
            for key in ("query_id", "bank"):
                if summary.get(key) in (None, "") and details.get(key) not in (None, ""):
                    summary[key] = details[key]
    total = 0
    for summary in summaries:
        raw_count = summary.get("occurrence_count", 1)
        if isinstance(raw_count, bool):
            raise ValueError("evidence reference occurrence_count must be positive")
        try:
            count = int(raw_count)
        except (TypeError, ValueError) as exc:
            raise ValueError("evidence reference occurrence_count must be positive") from exc
        if count < 1:
            raise ValueError("evidence reference occurrence_count must be positive")
        summary["occurrence_count"] = count
        total += count
    if total != _occurrence_count(occurrence):
        raise ValueError(
            "evidence reference occurrence counts do not equal raw_occurrence_count"
        )
    return summaries


def _evidence_occurrence_count(evidence: Sequence[Mapping[str, Any]]) -> int:
    return sum(_occurrence_count(row.get("occurrence") or {}) for row in evidence)


def _lexical_metrics(
    architecture: str,
    evidence: Sequence[Mapping[str, Any]],
    features: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    factory = _MetricFactory(architecture)
    texts = [str((row.get("occurrence") or {}).get("text") or "") for row in evidence]
    text_tokens = [set(_TOKEN.findall(text.lower())) - _STOPWORDS for text in texts]
    occurrence_count = _evidence_occurrence_count(evidence)
    rows = [
        factory.row(
            metric_family="evidence_inventory",
            metric="occurrence_count",
            value=float(occurrence_count),
            n=occurrence_count,
        )
    ]
    for feature in features:
        tokens = _feature_tokens(feature)
        best = (
            max((len(tokens & candidate) / len(tokens) for candidate in text_tokens), default=0.0)
            if tokens
            else None
        )
        rows.append(
            factory.row(
                metric_family="oracle_feature_recovery",
                metric="best_lexical_token_recall",
                value=best,
                target=str(feature["name"]),
                n=occurrence_count,
                reason="metadata feature has no evaluable lexical tokens",
            )
        )
    return rows


def _stability_metrics(
    architecture: str,
    evidence: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    factory = _MetricFactory(architecture)
    grouped: dict[int, dict[int, set[str]]] = {}
    for row in evidence:
        outer = int(row["outer_fold"])
        occurrence = row.get("occurrence") or {}
        text = str(occurrence.get("text") or "").strip().casefold()
        if not text:
            continue
        for reference in _occurrence_reference_summaries(occurrence):
            inner = reference.get("inner_fold", row.get("inner_fold"))
            if inner is not None:
                grouped.setdefault(outer, {}).setdefault(int(inner), set()).add(text)
    values: list[float] = []
    for folds in grouped.values():
        for left, right in combinations(folds.values(), 2):
            union = left | right
            if union:
                values.append(len(left & right) / len(union))
    return [
        factory.row(
            metric_family="discovery_stability",
            metric="mean_inner_fold_jaccard",
            value=float(np.mean(values)) if values else None,
            n=len(values),
            reason="fewer than two inner-fold evidence sets",
        )
    ]


def _native_evidence_metrics(
    architecture: str,
    evidence: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Report representation-native coverage without consulting oracle values."""

    factory = _MetricFactory(architecture)
    occurrences = [dict(row.get("occurrence") or {}) for row in evidence]
    occurrence_count = sum(_occurrence_count(occurrence) for occurrence in occurrences)
    axes = {str(axis) for occurrence in occurrences for axis in occurrence.get("axes") or []}
    patient_rows = {
        int(row_id)
        for occurrence in occurrences
        for row_id in [
            occurrence.get("patient_row_id"),
            *[
                reference.get("row_id")
                for reference in _occurrence_reference_summaries(occurrence)
            ],
        ]
        if row_id is not None
    }
    scored_count = 0
    for occurrence in occurrences:
        has_compact_summaries = isinstance(
            occurrence.get("reference_summaries"), Sequence
        ) and not isinstance(occurrence.get("reference_summaries"), (str, bytes))
        for reference in _occurrence_reference_summaries(occurrence):
            if reference.get("scores") or (
                not has_compact_summaries and occurrence.get("scores")
            ):
                scored_count += int(reference["occurrence_count"])
    primary_family = STAGE1_ARCHITECTURE_REGISTRY[architecture].native_metric_families[0]
    rows = [
        factory.row(
            metric_family=primary_family,
            metric="evidence_axis_coverage",
            value=float(len(axes)),
            n=occurrence_count,
        ),
        factory.row(
            metric_family=primary_family,
            metric="scored_evidence_fraction",
            value=(float(scored_count / occurrence_count) if occurrence_count else None),
            n=occurrence_count,
            reason="architecture emitted no canonical evidence occurrences",
        ),
    ]
    if architecture in {HTR_NEURAL, NEURAL_QUERY_MOMENTS}:
        rows.append(
            factory.row(
                metric_family=primary_family,
                metric="witness_patient_coverage",
                value=float(len(patient_rows)),
                n=occurrence_count,
            )
        )
    if architecture == EMBEDDING_CLUSTERED:
        clusters = {
            str(value)
            for occurrence in occurrences
            for key, value in (occurrence.get("details") or {}).items()
            if "cluster" in str(key).lower() and value is not None
        }
        rows.append(
            factory.row(
                metric_family="cluster_stability",
                metric="represented_cluster_count",
                value=float(len(clusters)),
                n=occurrence_count,
            )
        )
    details = [dict(occurrence.get("details") or {}) for occurrence in occurrences]
    axis_counts = {
        axis: sum(
            _occurrence_count(occurrence)
            for occurrence in occurrences
            if axis in (occurrence.get("axes") or [])
        )
        for axis in ("treatment", "outcome", "residual_effect", "matched_pair")
    }
    if architecture == BOW_NUISANCE:
        denominator = max(axis_counts["treatment"], axis_counts["outcome"])
        rows.append(
            factory.row(
                metric_family="nuisance",
                metric="treatment_outcome_axis_balance",
                value=(
                    min(axis_counts["treatment"], axis_counts["outcome"]) / denominator
                    if denominator
                    else None
                ),
                n=occurrence_count,
                reason="no treatment or outcome evidence was emitted",
            )
        )
    elif architecture == BOW_R_LOSS:
        rows.append(
            factory.row(
                metric_family="r_loss",
                metric="residual_effect_evidence_fraction",
                value=(
                    axis_counts["residual_effect"] / occurrence_count
                    if occurrence_count
                    else None
                ),
                n=occurrence_count,
                reason="no sparse residual-effect evidence was emitted",
            )
        )
    elif architecture == MATCHED_PAIR_UPLIFT:
        pair_sides = {
            str(detail.get("pair_side"))
            for detail in details
            if detail.get("pair_side") is not None
        }
        rows.extend(
            [
                factory.row(
                    metric_family="matching",
                    metric="matched_pair_evidence_fraction",
                    value=(
                        axis_counts["matched_pair"] / occurrence_count
                        if occurrence_count
                        else None
                    ),
                    n=occurrence_count,
                    reason="no matched-pair evidence was emitted",
                ),
                factory.row(
                    metric_family="matching",
                    metric="represented_pair_side_count",
                    value=float(len(pair_sides)),
                    n=occurrence_count,
                ),
            ]
        )
    elif architecture == HTR_NEURAL:
        stages = {str(detail.get("stage")) for detail in details if detail.get("stage")}
        rows.append(
            factory.row(
                metric_family="oracle_attribution",
                metric="represented_htr_stage_count",
                value=float(len(stages)),
                n=occurrence_count,
            )
        )
    elif architecture in {EMBEDDING_WHOLE_COHORT, EMBEDDING_CLUSTERED}:
        contrasts = {str(detail.get("contrast")) for detail in details if detail.get("contrast")}
        polarities = {
            str(occurrence.get("polarity"))
            for occurrence in occurrences
            if occurrence.get("polarity") in {"positive", "negative"}
        }
        rows.extend(
            [
                factory.row(
                    metric_family=(
                        "cluster_stability"
                        if architecture == EMBEDDING_CLUSTERED
                        else "semantic_recovery"
                    ),
                    metric="represented_contrast_count",
                    value=float(len(contrasts)),
                    n=occurrence_count,
                ),
                factory.row(
                    metric_family="semantic_recovery",
                    metric="contrast_polarity_coverage",
                    value=float(len(polarities) / 2.0),
                    n=occurrence_count,
                ),
            ]
        )
    elif architecture == TFIDF_SEMANTIC_RETRIEVAL:
        parents = {
            str(detail.get("parent_contrast"))
            for detail in details
            if detail.get("parent_contrast")
        }
        rows.append(
            factory.row(
                metric_family="retrieval",
                metric="represented_parent_contrast_count",
                value=float(len(parents)),
                n=occurrence_count,
            )
        )
    elif architecture == TFIDF_TOPICS:
        topics = {str(detail.get("topic_id")) for detail in details if detail.get("topic_id")}
        banks = {str(detail.get("bank")) for detail in details if detail.get("bank")}
        rows.extend(
            [
                factory.row(
                    metric_family="topic_stability",
                    metric="represented_topic_count",
                    value=float(len(topics)),
                    n=occurrence_count,
                ),
                factory.row(
                    metric_family="topic_stability",
                    metric="topic_bank_coverage",
                    value=float(len(banks) / 3.0),
                    n=occurrence_count,
                ),
            ]
        )
    elif architecture == TFIDF_ORPHAN_NGRAMS:
        clusters = {str(detail.get("cluster_id")) for detail in details if detail.get("cluster_id")}
        rows.append(
            factory.row(
                metric_family="ngram_recovery",
                metric="represented_orphan_cluster_count",
                value=float(len(clusters)),
                n=occurrence_count,
            )
        )
    elif architecture == NEURAL_QUERY_MOMENTS:
        references = [
            reference
            for occurrence in occurrences
            for reference in _occurrence_reference_summaries(occurrence)
        ]
        query_ids = {
            str(reference.get("query_id"))
            for reference in references
            if reference.get("query_id")
        }
        banks = {
            str(reference.get("bank"))
            for reference in references
            if reference.get("bank")
        }
        rows.extend(
            [
                factory.row(
                    metric_family="query_stability",
                    metric="represented_query_count",
                    value=float(len(query_ids)),
                    n=occurrence_count,
                ),
                factory.row(
                    metric_family="query_stability",
                    metric="query_bank_coverage",
                    value=float(len(banks) / 3.0),
                    n=occurrence_count,
                ),
            ]
        )
    return rows


def _r_loss_metrics(
    architecture: str,
    raw_frame: pd.DataFrame,
    dataset: pd.DataFrame,
    *,
    treatment_column: str,
    outcome_column: str,
) -> list[dict[str, Any]]:
    if architecture not in {BOW_R_LOSS, HTR_NEURAL}:
        return []
    factory = _MetricFactory(architecture)
    heldout = _outer_heldout(raw_frame)
    if "source_name" not in heldout or heldout.empty:
        return [
            factory.row(
                metric_family="r_loss",
                metric="best_normalized_r_loss_gain",
                value=None,
                reason="source-resolved outer-heldout scores are unavailable",
            )
        ]
    nuisance = heldout.loc[
        heldout["source_name"] == "ensemble_mean_nuisance",
        ["_oci_row_id", "outer_fold", "e_hat", "m_hat"],
    ].dropna(subset=["e_hat", "m_hat"])
    selected = (
        heldout.loc[heldout["architecture"] == architecture]
        if "architecture" in heldout.columns
        else heldout
    )
    tau_columns = [
        column
        for column in _score_columns(selected)
        if any(token in column.lower() for token in ("tau", "effect"))
    ]
    candidates: list[tuple[float, str, int]] = []
    for score in tau_columns:
        values = selected.loc[
            selected[score].notna(),
            ["_oci_row_id", "outer_fold", score],
        ]
        merged = values.merge(
            nuisance,
            on=["_oci_row_id", "outer_fold"],
            how="inner",
        ).merge(dataset, on="_oci_row_id", how="inner")
        if merged.empty:
            continue
        y = pd.to_numeric(merged[outcome_column], errors="coerce").to_numpy(dtype=float)
        t = pd.to_numeric(merged[treatment_column], errors="coerce").to_numpy(dtype=float)
        m = pd.to_numeric(merged["m_hat"], errors="coerce").to_numpy(dtype=float)
        e = pd.to_numeric(merged["e_hat"], errors="coerce").to_numpy(dtype=float)
        tau = pd.to_numeric(merged[score], errors="coerce").to_numpy(dtype=float)
        finite = (
            np.isfinite(y) & np.isfinite(t) & np.isfinite(m) & np.isfinite(e) & np.isfinite(tau)
        )
        if np.sum(finite) < 3:
            continue
        residual = y[finite] - m[finite]
        treatment_residual = t[finite] - e[finite]
        baseline = float(np.mean(np.square(residual)))
        if baseline <= 0:
            continue
        loss = float(np.mean(np.square(residual - treatment_residual * tau[finite])))
        candidates.append(((baseline - loss) / baseline, score, int(np.sum(finite))))
    best = max(candidates, default=None)
    return [
        factory.row(
            metric_family="r_loss",
            metric="best_normalized_r_loss_gain",
            value=None if best is None else best[0],
            score=None if best is None else best[1],
            n=None if best is None else best[2],
            reason="no tau score could be aligned with held-out nuisance predictions",
        )
    ]


def _oracle_ite_column(
    columns: Iterable[str],
    *,
    outcome_type: str,
) -> tuple[str | None, str | None]:
    """Resolve the oracle ITE column: ``(column, reason_when_unusable)``.

    ``true_ite`` is on the outcome scale for both outcome types. Older binary
    datasets carry only ``true_ite_prob``, which is the same quantity for a
    binary outcome but is not comparable for a continuous one.
    """

    available = set(columns)
    if "true_ite" in available:
        return "true_ite", None
    if "true_ite_prob" in available:
        if str(outcome_type).strip().lower() == "binary":
            return "true_ite_prob", None
        return None, (
            "continuous-outcome dataset lacks outcome-scale true_ite; "
            "probability-scale true_ite_prob is not comparable"
        )
    return None, None


def _association_metrics(
    architecture: str,
    frame: pd.DataFrame,
    dataset: pd.DataFrame,
    features: Sequence[Mapping[str, Any]],
    *,
    outcome_type: str = "binary",
) -> list[dict[str, Any]]:
    factory = _MetricFactory(architecture)
    heldout = _outer_heldout(frame)
    if heldout.empty:
        return [
            factory.row(
                metric_family="common_oracle_association",
                metric="best_score_association",
                value=None,
                reason="no outer-heldout row scores were persisted",
            )
        ]
    joined = heldout.merge(dataset, on="_oci_row_id", how="inner", validate="many_to_one")
    scores = _score_columns(heldout)
    output: list[dict[str, Any]] = []
    for feature in features:
        target = f"true_{feature['name']}"
        if target not in joined.columns:
            output.append(
                factory.row(
                    metric_family="common_oracle_association",
                    metric="best_score_association",
                    value=None,
                    target=target,
                    reason="oracle feature column is absent",
                )
            )
            continue
        feature_type = str(feature.get("type") or "categorical").lower()
        candidates: list[tuple[float, str, int, str]] = []
        for score in scores:
            if feature_type == "continuous":
                value, n = _safe_abs_spearman(joined[score], joined[target])
                metric = "best_score_abs_spearman"
            else:
                value, n = _orientation_free_auc(joined[score], joined[target])
                metric = "best_score_orientation_free_auc"
            if value is not None:
                candidates.append((value, score, n, metric))
        best = max(candidates, default=None)
        output.append(
            factory.row(
                metric_family="common_oracle_association",
                metric=(
                    best[3]
                    if best is not None
                    else (
                        "best_score_abs_spearman"
                        if feature_type == "continuous"
                        else "best_score_orientation_free_auc"
                    )
                ),
                value=None if best is None else best[0],
                target=target,
                score=None if best is None else best[1],
                n=None if best is None else best[2],
                reason="no usable row score for oracle feature",
            )
        )
    ite_column, ite_unusable_reason = _oracle_ite_column(
        joined.columns,
        outcome_type=outcome_type,
    )
    if ite_column is not None or ite_unusable_reason is not None:
        tau_scores = [
            score
            for score in scores
            if any(token in score.lower() for token in ("tau", "uplift", "effect"))
        ]
        for metric, function in (
            ("best_tau_abs_pearson", _safe_abs_pearson),
            ("best_tau_abs_spearman", _safe_abs_spearman),
        ):
            candidates = []
            if ite_column is not None:
                for score in tau_scores:
                    value, n = function(joined[score], joined[ite_column])
                    if value is not None:
                        candidates.append((value, score, n))
            best = max(candidates, default=None)
            output.append(
                factory.row(
                    metric_family="true_ite",
                    metric=metric,
                    value=None if best is None else best[0],
                    target=ite_column,
                    score=None if best is None else best[1],
                    n=None if best is None else best[2],
                    reason=ite_unusable_reason or "no held-out tau-like score is available",
                )
            )
    return output


def _nuisance_metrics(
    architecture: str,
    frame: pd.DataFrame,
    dataset: pd.DataFrame,
    *,
    treatment_column: str,
    outcome_column: str,
    outcome_type: str,
) -> list[dict[str, Any]]:
    if architecture not in {BOW_NUISANCE, HTR_NEURAL} and not architecture.startswith("tfidf_"):
        return []
    factory = _MetricFactory(architecture)
    heldout = _outer_heldout(frame)
    joined = heldout.merge(dataset, on="_oci_row_id", how="inner")
    output: list[dict[str, Any]] = []
    treatment_scores = [
        column
        for column in _score_columns(heldout)
        if column == "e_hat" or column == "treatment_stacked" or column.startswith("treatment_view")
    ]
    candidates = []
    if treatment_column in joined:
        for score in treatment_scores:
            value, n = _orientation_free_auc(joined[score], joined[treatment_column])
            if value is not None:
                candidates.append((value, score, n))
    best = max(candidates, default=None)
    output.append(
        factory.row(
            metric_family="nuisance",
            metric="best_treatment_orientation_free_auc",
            value=None if best is None else best[0],
            target=treatment_column,
            score=None if best is None else best[1],
            n=None if best is None else best[2],
            reason="no held-out treatment nuisance score",
        )
    )
    outcome_scores = [
        column
        for column in _score_columns(heldout)
        if column == "m_hat" or column == "outcome_stacked" or column.startswith("outcome_view")
    ]
    candidates = []
    if outcome_column in joined:
        for score in outcome_scores:
            if str(outcome_type).lower() == "continuous":
                truth, prediction = _finite_pair(joined[outcome_column], joined[score])
                value = (
                    float(math.sqrt(mean_squared_error(truth, prediction)))
                    if len(truth) >= 2
                    else None
                )
                candidate_value = -value if value is not None else None
                metric = "best_outcome_rmse"
            else:
                value, n = _orientation_free_auc(joined[score], joined[outcome_column])
                candidate_value = value
                metric = "best_outcome_orientation_free_auc"
            if candidate_value is not None:
                candidates.append((candidate_value, value, score, len(joined), metric))
    best_outcome = max(candidates, default=None)
    output.append(
        factory.row(
            metric_family="nuisance",
            metric=(
                best_outcome[4]
                if best_outcome is not None
                else (
                    "best_outcome_rmse"
                    if str(outcome_type).lower() == "continuous"
                    else "best_outcome_orientation_free_auc"
                )
            ),
            value=None if best_outcome is None else best_outcome[1],
            target=outcome_column,
            score=None if best_outcome is None else best_outcome[2],
            n=None if best_outcome is None else best_outcome[3],
            reason="no held-out outcome nuisance score",
        )
    )
    return output


def _native_score_metrics(
    architecture: str,
    frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    factory = _MetricFactory(architecture)
    heldout = _outer_heldout(frame)
    if architecture == MATCHED_PAIR_UPLIFT:
        columns = [column for column in heldout if "matched_control_count" in column]
        values = (
            pd.concat(
                [pd.to_numeric(heldout[column], errors="coerce") for column in columns],
                ignore_index=True,
            )
            if columns
            else pd.Series(dtype=float)
        )
        values = values[np.isfinite(values)]
        return [
            factory.row(
                metric_family="matching",
                metric="heldout_positive_match_coverage",
                value=float(np.mean(values > 0)) if len(values) else None,
                n=len(values),
                reason="matched-control counts were not persisted",
            )
        ]
    return []


def _split_training_rows(
    split_rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, int | None], list[int]]:
    """Map (outer_fold, inner_fold or None) to that context's fit rows."""

    output: dict[tuple[int, int | None], list[int]] = {}
    for outer in split_rows:
        outer_fold = int(outer["outer_fold"])
        output[(outer_fold, None)] = [int(value) for value in outer.get("fit_row_ids") or []]
        for inner in outer.get("inner_splits") or []:
            output[(outer_fold, int(inner["inner_fold"]))] = [
                int(value) for value in inner.get("fit_row_ids") or []
            ]
    return output


def _evidence_texts_by_context(
    evidence: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, int | None], dict[str, Any]]:
    """Group evidence texts by every discovery context each item references."""

    grouped: dict[tuple[int, int | None], dict[str, Any]] = {}
    for row in evidence:
        occurrence = row.get("occurrence") or {}
        text = str(occurrence.get("text") or "")
        if not text.strip():
            continue
        outer = int(row["outer_fold"])
        seen: set[tuple[int, int | None]] = set()
        for reference in _occurrence_reference_summaries(occurrence):
            raw_inner = reference.get("inner_fold", row.get("inner_fold"))
            key = (outer, None if raw_inner is None else int(raw_inner))
            if key in seen:
                continue
            seen.add(key)
            entry = grouped.setdefault(
                key,
                {"texts": [], "scope": reference.get("scope", row.get("scope"))},
            )
            entry["texts"].append(text)
    return dict(sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1] or 0)))


def _lexical_null_metrics(
    architecture: str,
    evidence: Sequence[Mapping[str, Any]],
    features: Sequence[Mapping[str, Any]],
    *,
    corpus: TrainingTextCorpus | None,
    training_rows: Mapping[tuple[int, int | None], Sequence[int]],
    draws: int,
    run_seed: int,
    unavailable_reason: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compare per-context lexical recovery with a random-chunk null."""

    factory = _MetricFactory(architecture)
    feature_tokens = {
        str(feature["name"]): frozenset(_feature_tokens(feature)) for feature in features
    }
    feature_tokens = {name: tokens for name, tokens in feature_tokens.items() if tokens}
    metrics: list[dict[str, Any]] = []
    table: list[dict[str, Any]] = []
    if not feature_tokens:
        return metrics, table
    for (outer, inner), entry in _evidence_texts_by_context(evidence).items():
        rows = training_rows.get((outer, inner))
        reason = unavailable_reason
        if reason is None and not rows:
            reason = "the split provenance has no training rows for this context"
        seed = null_seed(
            run_seed=run_seed,
            architecture=architecture,
            outer_fold=outer,
            inner_fold=inner,
            schema_version=LEXICAL_NULL_SCHEMA_VERSION,
        )
        result = random_chunk_null(
            corpus=corpus if reason is None else None,
            training_row_ids=rows or [],
            evidence_texts=entry["texts"],
            feature_tokens=feature_tokens,
            draws=draws if reason is None else 0,
            seed=seed,
        )
        for name in feature_tokens:
            observed = result.observed.get(name)
            summary = summarize_null(observed, result.null_values.get(name))
            context = {
                "outer_fold": outer,
                "inner_fold": inner,
                "scope": entry.get("scope"),
                "target": name,
                "n": result.items,
            }
            metrics.append(
                factory.row(
                    metric_family="oracle_feature_recovery_null",
                    metric="best_lexical_token_recall_observed",
                    value=observed,
                    **context,
                )
            )
            for key, metric in (
                ("null_mean", "best_lexical_token_recall_null_mean"),
                ("null_p95", "best_lexical_token_recall_null_p95"),
                ("empirical_p_value", "best_lexical_token_recall_empirical_p_value"),
            ):
                metrics.append(
                    factory.row(
                        metric_family="oracle_feature_recovery_null",
                        metric=metric,
                        value=summary[key],
                        reason=reason or "random-chunk null draws are disabled",
                        **context,
                    )
                )
            table.append(
                {
                    "schema_version": LEXICAL_NULL_SCHEMA_VERSION,
                    "architecture": architecture,
                    **{key: context[key] for key in ("outer_fold", "inner_fold", "scope")},
                    "target": name,
                    "observed": observed,
                    **summary,
                    "draws": result.draws,
                    "evidence_items": result.items,
                    "length_unmatched_items": result.length_unmatched_items,
                    "seed": result.seed,
                    "reason": reason,
                }
            )
    return metrics, table


def _embedding_direction_coverage_metrics(
    architecture: str,
    evidence: Sequence[Mapping[str, Any]],
    frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Share of an outer fold's handed-off directions that have held-out scores."""

    if architecture not in EMBEDDING_DIRECTION_SCORE_SOURCES:
        return []
    heldout = _outer_heldout(frame)
    if heldout.empty or "outer_fold" not in heldout.columns:
        return []
    folds = sorted({int(value) for value in heldout["outer_fold"].dropna()})
    scored = {
        slug
        for column in heldout.columns
        if (slug := embedding_direction_from_score_column(column)) is not None
        and heldout[column].notna().any()
    }
    output = []
    for outer in folds:
        handed_off = set()
        for row in evidence:
            details = (row.get("occurrence") or {}).get("details") or {}
            if (
                int(row["outer_fold"]) == outer
                and row.get("inner_fold") is None
                and str(row.get("scope") or "full_outer_train") == "full_outer_train"
                and details.get("contrast")
            ):
                handed_off.add(embedding_direction_slug(details["contrast"]))
        output.append(
            _MetricFactory(architecture).row(
                metric_family=STAGE1_ARCHITECTURE_REGISTRY[architecture].native_metric_families[0],
                metric="handed_off_direction_score_coverage",
                value=(len(handed_off & scored) / len(handed_off)) if handed_off else None,
                n=len(handed_off),
                outer_fold=outer,
                scope="full_outer_train",
                reason="no handed-off directions in this outer fold's evidence",
            )
        )
    return output


def _timing_metrics(
    architecture: str,
    records: Sequence[Mapping[str, Any]],
    sources: Mapping[str, tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Wall seconds attributed to the architecture alone, per timed context."""

    producer = STAGE1_ARCHITECTURE_REGISTRY[architecture].component
    output = []
    for record in records:
        if str(record.get("component")) != producer:
            continue
        value = architecture_context_seconds(record, architecture)
        unknown = str(record.get("status")) == "unknown"
        if value is None and not unknown:
            continue
        source = (sources or {}).get(str(record.get("component")))
        factory = _MetricFactory(
            architecture,
            source_artifact=None if source is None else source[0],
            source_sha256=None if source is None else source[1],
        )
        output.append(
            factory.row(
                metric_family="cost",
                metric="wall_seconds",
                value=value,
                outer_fold=record.get("outer_fold"),
                inner_fold=record.get("inner_fold"),
                scope=record.get("scope"),
                reason=str(record.get("reason") or "context timing is unknown"),
            )
        )
    return output


def _summarize(metrics: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    applicable = [row for row in metrics if row.get("applicability") == "applicable"]
    grouped: dict[tuple[str, str, str, str | None], list[float]] = {}
    for row in applicable:
        key = (
            str(row["architecture"]),
            str(row["metric_family"]),
            str(row["metric"]),
            None if row.get("target") is None else str(row["target"]),
        )
        grouped.setdefault(key, []).append(float(row["value"]))
    return [
        {
            "architecture": key[0],
            "metric_family": key[1],
            "metric": key[2],
            "target": key[3],
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "n_estimates": len(values),
        }
        for key, values in sorted(grouped.items())
    ]


def evaluate_stage1_architectures(
    *,
    run_dir: Path,
    dataset_path: Path | None = None,
    metadata_path: Path | None = None,
    architectures: str | Sequence[str] | None = None,
    output_dir: Path | None = None,
    null_draws: int = DEFAULT_NULL_DRAWS,
) -> Mapping[str, Any]:
    """Evaluate frozen Stage 1 artifacts; never fit or select an architecture."""

    if isinstance(null_draws, bool) or not isinstance(null_draws, int) or null_draws < 0:
        raise ValueError("null_draws must be a nonnegative integer")

    run_dir = Path(run_dir).expanduser().resolve(strict=True)
    run_config_path = run_dir / "run_config.json"
    if not run_config_path.is_file():
        raise FileNotFoundError(f"Stage 1 run has no run_config.json: {run_config_path}")
    run_config = _read_json(run_config_path)
    architecture_manifest = _materialize_or_load_manifest(run_dir)
    frozen_selection = canonicalize_stage1_architectures(
        architecture_manifest.get("selected_architectures"),
        allow_none=False,
    )
    assert frozen_selection is not None
    requested = (
        frozen_selection
        if architectures is None
        or isinstance(architectures, str)
        and architectures.strip().lower() == "all"
        else canonicalize_stage1_architectures(architectures, allow_none=False)
    )
    assert requested is not None
    outside_run = sorted(set(requested) - set(frozen_selection))
    if outside_run:
        raise ValueError(f"architectures were not present in this Stage 1 run: {outside_run}")

    # Freeze and hash every consumed Stage 1 artifact before loading an
    # oracle-bearing table. Score files are also retained in memory so the
    # evaluation cannot observe a later rewrite of the pipeline artifacts.
    evidence_hashes: dict[str, str] = {}
    evidence_by_architecture: dict[str, list[dict[str, Any]]] = {}
    score_cache: dict[Path, pd.DataFrame] = {}
    score_frames_by_architecture: dict[str, list[tuple[Path, pd.DataFrame, pd.DataFrame]]] = {}
    consumed_score_hashes: dict[str, str] = {}
    legacy_embedding_scores: dict[str, list[Path]] = {}
    for architecture in requested:
        path = run_dir / "stage1_architectures" / architecture / "evidence.jsonl"
        evidence_hashes[architecture] = _file_sha256(path)
        evidence_by_architecture[architecture] = _load_evidence(run_dir, architecture)
        score_frames = _load_score_frames(
            run_dir,
            architecture,
            architecture_manifest,
            score_cache,
            legacy_paths=legacy_embedding_scores.setdefault(architecture, []),
        )
        score_frames_by_architecture[architecture] = score_frames
        for score_path, _selected_frame, _raw_frame in score_frames:
            relative = os.path.relpath(score_path, start=run_dir)
            if relative not in consumed_score_hashes:
                consumed_score_hashes[relative] = _file_sha256(score_path)
    # The split plan (for the null draws' training rows) and the timing
    # records are Stage 1 artifacts too: hash and load them before oracle data.
    split_path = run_dir / "components" / "tfidf" / "split_provenance.jsonl"
    split_sha256 = _file_sha256(split_path) if split_path.is_file() else None
    split_rows = _read_jsonl(split_path) if split_path.is_file() else []
    timing_files = timing_paths(run_dir)
    timing_hashes = {
        os.path.relpath(path, start=run_dir): _file_sha256(path) for path in timing_files
    }
    timing_records = [
        *load_timing_records(timing_files),
        *untimed_component_records(run_dir),
    ]
    timing_sources = {
        Path(relative).parent.name: (relative, digest)
        for relative, digest in timing_hashes.items()
    }

    dataset = _resolve_dataset(run_dir, run_config, dataset_path)
    metadata = _resolve_metadata(dataset, metadata_path)
    frame = _load_dataset(dataset)
    metadata_value = _read_json(metadata) if metadata is not None else {}
    features = _metadata_features(metadata_value)
    truth_columns = sorted(
        column
        for column in frame.columns
        if str(column).lower().startswith(("true_", "oracle_", "ground_truth"))
    )
    treatment_column = str(run_config.get("treatment_column") or "treatment_indicator")
    outcome_column = str(run_config.get("outcome_column") or "outcome_indicator")
    outcome_type = str(run_config.get("outcome_type") or "binary")
    text_column = str(run_config.get("text_column") or "clinical_text")
    run_seed = int(run_config.get("seed") or 0)

    training_rows = _split_training_rows(split_rows)
    null_unavailable: str | None = None
    corpus: TrainingTextCorpus | None = None
    if null_draws == 0:
        null_unavailable = "random-chunk null draws are disabled (--null-draws 0)"
    elif not split_rows:
        null_unavailable = "the run has no components/tfidf/split_provenance.jsonl"
    elif text_column not in frame.columns:
        null_unavailable = f"the dataset has no text column {text_column!r}"
    else:
        corpus = TrainingTextCorpus(frame[text_column].fillna("").astype(str).tolist())

    all_metrics: list[dict[str, Any]] = []
    null_table: list[dict[str, Any]] = []
    for architecture in requested:
        evidence = evidence_by_architecture[architecture]
        evidence_path = run_dir / "stage1_architectures" / architecture / "evidence.jsonl"
        null_metrics, null_rows = _lexical_null_metrics(
            architecture,
            evidence,
            features,
            corpus=corpus,
            training_rows=training_rows,
            draws=null_draws,
            run_seed=run_seed,
            unavailable_reason=null_unavailable,
        )
        null_table.extend(null_rows)
        metrics = _annotate_metric_source(
            [
                *_lexical_metrics(architecture, evidence, features),
                *null_metrics,
                *_stability_metrics(architecture, evidence),
                *_native_evidence_metrics(architecture, evidence),
            ],
            source_artifact=os.path.relpath(evidence_path, start=run_dir),
            source_sha256=evidence_hashes[architecture],
        )
        metrics.extend(_timing_metrics(architecture, timing_records, timing_sources))
        score_frames = score_frames_by_architecture[architecture]
        if not score_frames:
            metrics.append(
                _MetricFactory(architecture).row(
                    metric_family="common_oracle_association",
                    metric="best_score_association",
                    value=None,
                    reason=(
                        "the run's embedding row scores come from a retired duplicate "
                        "direction pipeline, not the handed-off directions; rerun Stage 1"
                        if legacy_embedding_scores.get(architecture)
                        else "this run has no row-level score sidecar for the architecture"
                    ),
                )
            )
        for path, score_frame, raw_score_frame in score_frames:
            relative = os.path.relpath(path, start=run_dir)
            generated = [
                *_association_metrics(
                    architecture,
                    score_frame,
                    frame,
                    features,
                    outcome_type=outcome_type,
                ),
                *_nuisance_metrics(
                    architecture,
                    score_frame,
                    frame,
                    treatment_column=treatment_column,
                    outcome_column=outcome_column,
                    outcome_type=outcome_type,
                ),
                *_native_score_metrics(architecture, score_frame),
                *_embedding_direction_coverage_metrics(architecture, evidence, score_frame),
                *_r_loss_metrics(
                    architecture,
                    raw_score_frame,
                    frame,
                    treatment_column=treatment_column,
                    outcome_column=outcome_column,
                ),
            ]
            metrics.extend(
                _annotate_metric_source(
                    generated,
                    source_artifact=relative,
                    source_sha256=consumed_score_hashes[relative],
                    frame=_outer_heldout(score_frame),
                )
            )
        all_metrics.extend(metrics)

    output_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else run_dir / "evaluation" / "stage1"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    all_metrics.sort(
        key=lambda row: (
            STAGE1_ARCHITECTURES.index(str(row["architecture"])),
            str(row["metric_family"]),
            str(row["metric"]),
            str(row.get("target") or ""),
            str(row.get("score") or ""),
        )
    )
    _write_jsonl(output_dir / "metrics.jsonl", all_metrics)
    for architecture in requested:
        _write_jsonl(
            output_dir / "architectures" / architecture / "metrics.jsonl",
            [row for row in all_metrics if row["architecture"] == architecture],
        )
    comparison = _summarize(all_metrics)
    pd.DataFrame(comparison).to_csv(output_dir / "comparison.csv", index=False)
    null_columns = [
        "schema_version", "architecture", "outer_fold", "inner_fold", "scope", "target",
        "observed", "null_mean", "null_p95", "empirical_p_value", "draws",
        "evidence_items", "length_unmatched_items", "seed", "reason",
    ]
    pd.DataFrame(null_table, columns=null_columns).to_csv(
        output_dir / "lexical_recovery_null.csv", index=False
    )
    timing_columns = [
        "component", "scope_id", "outer_fold", "inner_fold", "scope", "status",
        "context_wall_seconds", "unit", "attribution", "consumers", "seconds",
        "restored_checkpoint_units",
    ]
    pd.DataFrame(timing_table(timing_records), columns=timing_columns).to_csv(
        output_dir / "timing.csv", index=False
    )
    summary = {
        "schema_version": EVALUATION_MANIFEST_SCHEMA_VERSION,
        "architectures": list(requested),
        "metrics": len(all_metrics),
        "applicable_metrics": sum(row["applicability"] == "applicable" for row in all_metrics),
        "comparison_rows": len(comparison),
        "comparison": comparison,
        "lexical_recovery_null": {
            "schema_version": LEXICAL_NULL_SCHEMA_VERSION,
            "draws": int(null_draws),
            "rows": len(null_table),
            "unavailable_reason": null_unavailable,
        },
        "timing": timing_summary(timing_records, requested),
        "outputs": [
            "metrics.jsonl",
            "architectures/<architecture>/metrics.jsonl",
            "comparison.csv",
            "lexical_recovery_null.csv",
            "timing.csv",
            "summary.json",
            "evaluation_manifest.json",
        ],
    }
    _write_json(output_dir / "summary.json", summary)
    manifest = {
        "schema_version": EVALUATION_MANIFEST_SCHEMA_VERSION,
        "created_at": _now(),
        "run_dir": str(run_dir),
        "dataset": str(dataset),
        "dataset_sha256": _file_sha256(dataset),
        "metadata": None if metadata is None else str(metadata),
        "metadata_sha256": None if metadata is None else _file_sha256(metadata),
        "selected_architectures": list(requested),
        "stage1_evidence_sha256": evidence_hashes,
        "consumed_score_sha256": consumed_score_hashes,
        "consumed_split_provenance_sha256": split_sha256,
        "consumed_timing_sha256": timing_hashes,
        "legacy_embedding_score_artifacts_ignored": {
            architecture: sorted(os.path.relpath(path, start=run_dir) for path in paths)
            for architecture, paths in legacy_embedding_scores.items()
            if paths
        },
        "lexical_null_draws": int(null_draws),
        "oracle_columns": truth_columns,
        "oracle_columns_available_to_stage1": False,
        "stage1_artifacts_frozen_before_oracle_load": True,
        "evaluation_does_not_modify_pipeline_completion_or_fingerprints": True,
        "native_metric_families": {
            architecture: list(STAGE1_ARCHITECTURE_REGISTRY[architecture].native_metric_families)
            for architecture in requested
        },
    }
    _write_json(output_dir / "evaluation_manifest.json", manifest)
    return {**summary, "output_dir": str(output_dir)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen Stage 1 architectures against post-hoc oracle truth."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--architectures", help="comma-separated names or 'all'")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--null-draws",
        type=int,
        default=DEFAULT_NULL_DRAWS,
        help="random-chunk null draws per architecture and context (0 disables)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = evaluate_stage1_architectures(
        run_dir=args.run_dir,
        dataset_path=args.dataset,
        metadata_path=args.metadata,
        architectures=args.architectures,
        output_dir=args.output_dir,
        null_draws=args.null_draws,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_NULL_DRAWS",
    "EVALUATION_MANIFEST_SCHEMA_VERSION",
    "EVALUATION_METRIC_SCHEMA_VERSION",
    "LEXICAL_NULL_SCHEMA_VERSION",
    "build_parser",
    "evaluate_stage1_architectures",
    "main",
]
