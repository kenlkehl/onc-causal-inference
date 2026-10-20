"""Additive, non-oracle artifacts for individual Stage 1 architectures."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .stage1_architectures import (
    EMBEDDING_CLUSTERED,
    EMBEDDING_WHOLE_COHORT,
    STAGE1_ARCHITECTURE_REGISTRY,
    TFIDF_TOPICS,
    canonicalize_stage1_architectures,
    resolve_support_services,
)

ARCHITECTURE_EVIDENCE_SCHEMA_VERSION = "stage1_architecture_evidence_v2_compact_occurrences"
ARCHITECTURE_MANIFEST_SCHEMA_VERSION = "stage1_architecture_manifest_v2_compact_occurrences"

# Row-level scores of the embedding directions handed to Stage 2: each
# context's held-out rows projected on the exact directions whose aligned
# chunks entered the handoff (``predictions.parquet`` under text_models).
EMBEDDING_DIRECTION_SCORE_SOURCES = {
    EMBEDDING_WHOLE_COHORT: "embedding__whole_cohort__handed_off_directions",
    EMBEDDING_CLUSTERED: "embedding__clustered__handed_off_directions",
}
EMBEDDING_DIRECTION_SCORE_STATISTICS = ("mean_chunk_cosine", "max_chunk_cosine")
_EMBEDDING_DIRECTION_COLUMN_PREFIX = "direction__"


def embedding_direction_slug(name: Any) -> str:
    """Return the column-safe form of an embedding contrast name."""

    output: list[str] = []
    for character in str(name).strip().lower():
        if character.isascii() and character.isalnum():
            output.append(character)
        elif output and output[-1] != "_":
            output.append("_")
    return "".join(output).strip("_") or "contrast"


def embedding_direction_score_column(name: Any, statistic: str) -> str:
    if statistic not in EMBEDDING_DIRECTION_SCORE_STATISTICS:
        raise ValueError(f"unknown embedding direction statistic: {statistic!r}")
    return f"{_EMBEDDING_DIRECTION_COLUMN_PREFIX}{embedding_direction_slug(name)}__{statistic}"


def embedding_direction_from_score_column(column: Any) -> str | None:
    """Return the contrast slug a handed-off direction score column belongs to."""

    text = str(column)
    if not text.startswith(_EMBEDDING_DIRECTION_COLUMN_PREFIX):
        return None
    for statistic in EMBEDDING_DIRECTION_SCORE_STATISTICS:
        suffix = f"__{statistic}"
        if text.endswith(suffix):
            slug = text[len(_EMBEDDING_DIRECTION_COLUMN_PREFIX) : -len(suffix)]
            return slug or None
    return None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evidence_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    occurrence = row["occurrence"]
    return (
        int(row["outer_fold"]),
        int(row.get("inner_fold") or 0),
        str(row.get("scope") or ""),
        str(occurrence.get("evidence_kind") or ""),
        str(occurrence.get("text") or ""),
        json.dumps(occurrence.get("reference") or {}, sort_keys=True),
    )


def _score_artifacts(output_dir: Path, architecture: str) -> list[str]:
    root = Path(output_dir)
    producer = STAGE1_ARCHITECTURE_REGISTRY[architecture].component
    if producer == "text_models":
        paths = sorted(
            (root / "components" / "text_models").glob("*/worker_artifacts/**/predictions.parquet")
        )
    elif producer == "tfidf":
        primary = root / "components" / "tfidf" / "predictions.parquet"
        # The context-local nuisance files mix candidate-selection inner
        # heldouts with the full-outer heldout and the latter is copied into
        # this canonical sidecar.  Registering both makes post-hoc metrics
        # count the outer rows twice and can admit inner rows whose parquet
        # schema does not identify its enclosing context.
        # This is the topic producer's shared nuisance projection, not an
        # orphan-ngram-specific row score.  The orphan lane is evaluated from
        # its canonical evidence instead of receiving duplicate topic metrics.
        paths = [primary] if architecture == TFIDF_TOPICS and primary.is_file() else []
    elif producer == "neural_queries":
        paths = sorted((root / "components" / "neural_queries").glob("*/scores.parquet"))
    else:  # pragma: no cover - registry validation prevents this
        paths = []
    return [os.path.relpath(path, start=root) for path in paths if path.is_file()]


def _bind_occurrence_to_handoff_row(
    occurrence: dict[str, Any],
    *,
    handoff_row: int,
) -> None:
    """Bind every compact provenance reference to its canonical JSONL row."""

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


def materialize_stage1_architecture_artifacts(
    *,
    output_dir: Path,
    raw_handoff_rows: Iterable[Mapping[str, Any]],
    selected_architectures: Sequence[str],
    source_artifacts: Mapping[str, Path],
    selection_mode: str,
) -> tuple[list[dict[str, Any]], Mapping[str, Any]]:
    """Write per-lane evidence and return canonical targeted-handoff rows."""

    selected = canonicalize_stage1_architectures(
        tuple(selected_architectures),
        allow_none=False,
    )
    assert selected is not None
    from .plain_handoff_stage2_evidence import extract_stage1_architecture_occurrences

    by_outer = extract_stage1_architecture_occurrences(
        raw_handoff_rows,
        included_architectures=selected,
    )
    source_metadata = {
        name: {
            "path": os.path.relpath(path, start=Path(output_dir)),
            "sha256": _file_sha256(path),
        }
        for name, path in sorted(source_artifacts.items())
        if path.is_file()
    }
    rows_by_architecture: dict[str, list[dict[str, Any]]] = {name: [] for name in selected}
    for outer_fold, occurrences in by_outer.items():
        for occurrence in occurrences:
            architecture = str(occurrence["architecture"])
            reference = dict(occurrence.get("reference") or {})
            source = str(reference.get("source") or "unknown")
            rows_by_architecture[architecture].append(
                {
                    "schema_version": ARCHITECTURE_EVIDENCE_SCHEMA_VERSION,
                    "architecture": architecture,
                    "outer_fold": int(outer_fold),
                    "inner_fold": reference.get("inner_fold"),
                    "scope": str(reference.get("scope") or "unspecified"),
                    "occurrence": occurrence,
                    "lineage": {
                        "producer_component": STAGE1_ARCHITECTURE_REGISTRY[architecture].component,
                        "source": source,
                        "source_artifact": (
                            source_metadata.get(source)
                            or source_metadata.get(
                                STAGE1_ARCHITECTURE_REGISTRY[architecture].component
                            )
                        ),
                        "source_json_path": reference.get("json_path"),
                        "private_support_services": list(
                            STAGE1_ARCHITECTURE_REGISTRY[architecture].support_services
                        ),
                    },
                }
            )

    canonical_handoff: list[dict[str, Any]] = []
    evidence_rows_by_architecture: dict[str, list[dict[str, Any]]] = {}
    architecture_root = Path(output_dir) / "stage1_architectures"
    for architecture in selected:
        evidence_rows = sorted(
            rows_by_architecture[architecture],
            key=_evidence_sort_key,
        )
        evidence_rows_by_architecture[architecture] = evidence_rows
        canonical_handoff.extend(
            {
                "source": "stage1_architecture",
                "outer_fold": row["outer_fold"],
                "inner_fold": row["inner_fold"],
                "scope": row["scope"],
                "evidence": {
                    "schema_version": ARCHITECTURE_EVIDENCE_SCHEMA_VERSION,
                    "architecture": architecture,
                    "occurrence": row["occurrence"],
                    "lineage": row["lineage"],
                },
            }
            for row in evidence_rows
        )

    canonical_handoff.sort(
        key=lambda row: (
            int(row["outer_fold"]),
            int(row.get("inner_fold") or 0),
            str(row["evidence"]["architecture"]),
            str(row["evidence"]["occurrence"].get("evidence_kind") or ""),
            str(row["evidence"]["occurrence"].get("text") or ""),
        )
    )
    if selection_mode == "explicit":
        for handoff_row, row in enumerate(canonical_handoff, start=1):
            _bind_occurrence_to_handoff_row(
                row["evidence"]["occurrence"],
                handoff_row=handoff_row,
            )

    # Canonical handoff rows and architecture sidecars share the occurrence
    # objects above.  For an explicit handoff, write sidecars only after
    # rebinding compact summaries so every persisted handoff_row resolves in
    # the final combined handoff.  A legacy evaluation backfill keeps the
    # source handoff in place, so its original row references remain valid.
    artifact_index: dict[str, Any] = {}
    for architecture in selected:
        evidence_rows = evidence_rows_by_architecture[architecture]
        evidence_path = architecture_root / architecture / "evidence.jsonl"
        _write_jsonl(evidence_path, evidence_rows)
        artifact_index[architecture] = {
            "evidence": os.path.relpath(evidence_path, start=Path(output_dir)),
            "evidence_sha256": _file_sha256(evidence_path),
            "occurrences": sum(
                int(row["occurrence"].get("raw_occurrence_count", 1))
                for row in evidence_rows
            ),
            "compact_records": len(evidence_rows),
            "producer_component": STAGE1_ARCHITECTURE_REGISTRY[architecture].component,
            "support_services": list(STAGE1_ARCHITECTURE_REGISTRY[architecture].support_services),
            "score_artifacts": _score_artifacts(Path(output_dir), architecture),
        }

    manifest_path = architecture_root / "manifest.json"
    existing: dict[str, Any] = {}
    if manifest_path.is_file():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = {}
    manifest = {
        "schema_version": ARCHITECTURE_MANIFEST_SCHEMA_VERSION,
        "created_at": existing.get("created_at", _now()),
        "updated_at": _now(),
        "selection_mode": str(selection_mode),
        "selected_architectures": list(selected),
        "support_services": list(resolve_support_services(selected)),
        "source_artifacts": source_metadata,
        "architectures": artifact_index,
    }
    _write_json(manifest_path, manifest)
    return canonical_handoff, manifest


def iter_stage1_architecture_evidence(
    output_dir: Path | str,
    architecture: str | None = None,
) -> Iterable[Mapping[str, Any]]:
    """Yield canonical evidence rows for one lane or the saved selection."""

    root = Path(output_dir) / "stage1_architectures"
    if architecture is None:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        architectures = canonicalize_stage1_architectures(
            manifest.get("selected_architectures"),
            allow_none=False,
        )
        assert architectures is not None
    else:
        architectures = canonicalize_stage1_architectures(
            (architecture,),
            allow_none=False,
        )
        assert architectures is not None
    for name in architectures:
        path = root / name / "evidence.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


__all__ = [
    "ARCHITECTURE_EVIDENCE_SCHEMA_VERSION",
    "ARCHITECTURE_MANIFEST_SCHEMA_VERSION",
    "EMBEDDING_DIRECTION_SCORE_SOURCES",
    "EMBEDDING_DIRECTION_SCORE_STATISTICS",
    "embedding_direction_from_score_column",
    "embedding_direction_score_column",
    "embedding_direction_slug",
    "iter_stage1_architecture_evidence",
    "materialize_stage1_architecture_artifacts",
]
