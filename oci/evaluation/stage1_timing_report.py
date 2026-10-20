"""Per-architecture Stage 1 wall time for the post-hoc evaluator.

Reads the reporting-only ``components/<component>/timing.jsonl`` records that
Stage 1 writes (see :mod:`oci.inference.stage1_timing`).  Nothing here fits a
model or reads oracle data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..inference.stage1_architectures import STAGE1_ARCHITECTURES
from ..inference.stage1_timing import (
    COMPONENT_TIMING_FILE,
    SHARED_ATTRIBUTION,
    STAGE1_TIMING_SCHEMA_VERSION,
)

TIMING_COMPONENTS = ("embedding_cache", "tfidf", "text_models", "neural_queries")


def timing_paths(run_dir: Path) -> list[Path]:
    return [
        Path(run_dir) / "components" / component / COMPONENT_TIMING_FILE
        for component in TIMING_COMPONENTS
        if (Path(run_dir) / "components" / component / COMPONENT_TIMING_FILE).is_file()
    ]


def untimed_component_records(run_dir: Path) -> list[dict[str, Any]]:
    """``unknown`` records for completed components that wrote no timing."""

    records = []
    for component in TIMING_COMPONENTS:
        directory = Path(run_dir) / "components" / component
        if (directory / "complete.json").is_file() and not (
            directory / COMPONENT_TIMING_FILE
        ).is_file():
            records.append(
                {
                    "schema_version": STAGE1_TIMING_SCHEMA_VERSION,
                    "component": component,
                    "scope_id": None,
                    "outer_fold": None,
                    "inner_fold": None,
                    "scope": None,
                    "status": "unknown",
                    "reason": "the component completed without writing timing records",
                    "wall_seconds": None,
                    "entries": [],
                }
            )
    return records


def load_timing_records(paths: Sequence[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    if isinstance(value, dict):
                        records.append(value)
    return records


def _status(record: Mapping[str, Any]) -> str:
    if record.get("schema_version") != STAGE1_TIMING_SCHEMA_VERSION:
        return "unknown"
    return str(record.get("status") or "unknown")


def timing_table(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One row per timed section (or per context without timing)."""

    rows: list[dict[str, Any]] = []
    for record in records:
        base = {
            "component": record.get("component"),
            "scope_id": record.get("scope_id"),
            "outer_fold": record.get("outer_fold"),
            "inner_fold": record.get("inner_fold"),
            "scope": record.get("scope"),
            "status": _status(record),
            "context_wall_seconds": record.get("wall_seconds"),
        }
        entries = list(record.get("entries") or []) if _status(record) != "unknown" else []
        if not entries:
            rows.append(
                {
                    **base,
                    "unit": None,
                    "attribution": None,
                    "consumers": None,
                    "seconds": None,
                    "restored_checkpoint_units": None,
                }
            )
            continue
        for entry in entries:
            rows.append(
                {
                    **base,
                    "unit": entry.get("unit"),
                    "attribution": entry.get("attribution"),
                    "consumers": ";".join(map(str, entry.get("consumers") or [])),
                    "seconds": entry.get("seconds"),
                    "restored_checkpoint_units": len(
                        entry.get("restored_checkpoint_units") or []
                    ),
                }
            )
    return rows


def architecture_context_seconds(
    record: Mapping[str, Any],
    architecture: str,
) -> float | None:
    """Seconds attributed to one architecture alone in one timed context."""

    if _status(record) == "unknown":
        return None
    values = [
        float(entry.get("seconds") or 0.0)
        for entry in record.get("entries") or []
        if entry.get("attribution") == architecture
    ]
    return float(sum(values)) if values else None


def timing_summary(
    records: Sequence[Mapping[str, Any]],
    architectures: Sequence[str],
) -> dict[str, Any]:
    """Totals per architecture and per shared unit, with context status counts."""

    status_counts: dict[str, int] = {}
    for record in records:
        status_counts[_status(record)] = status_counts.get(_status(record), 0) + 1
    per_architecture: dict[str, Any] = {}
    for architecture in architectures:
        contexts = [
            (record, architecture_context_seconds(record, architecture))
            for record in records
        ]
        timed = [(record, value) for record, value in contexts if value is not None]
        per_architecture[architecture] = {
            "exclusive_seconds": float(sum(value for _record, value in timed)),
            "outer_context_seconds": float(
                sum(value for record, value in timed if record.get("inner_fold") is None)
            ),
            "timed_contexts": len(timed),
            "partially_restored_contexts": sum(
                _status(record) == "partially_restored" for record, _value in timed
            ),
        }
    shared: dict[str, Any] = {}
    for record in records:
        if _status(record) == "unknown":
            continue
        for entry in record.get("entries") or []:
            if entry.get("attribution") != SHARED_ATTRIBUTION:
                continue
            key = f"{record.get('component')}:{entry.get('unit')}"
            item = shared.setdefault(
                key,
                {
                    "component": record.get("component"),
                    "unit": entry.get("unit"),
                    "consumers": list(entry.get("consumers") or []),
                    "seconds": 0.0,
                    "contexts": 0,
                },
            )
            item["seconds"] += float(entry.get("seconds") or 0.0)
            item["contexts"] += 1
    unknown_contexts = [
        {
            "component": record.get("component"),
            "scope_id": record.get("scope_id"),
            "reason": record.get("reason"),
        }
        for record in records
        if _status(record) == "unknown"
    ]
    return {
        "schema_version": STAGE1_TIMING_SCHEMA_VERSION,
        "available": bool(records),
        "context_status_counts": status_counts,
        "architectures": per_architecture,
        "shared": dict(sorted(shared.items())),
        "shared_seconds": float(sum(item["seconds"] for item in shared.values())),
        "unknown_contexts": unknown_contexts,
        "attribution_rule": (
            "seconds of sections serving one architecture; sections that several "
            "architectures need are listed under shared with their consumers"
        ),
    }


def ordered_architectures(values: Sequence[str]) -> list[str]:
    return [name for name in STAGE1_ARCHITECTURES if name in set(values)]


__all__ = [
    "TIMING_COMPONENTS",
    "architecture_context_seconds",
    "load_timing_records",
    "ordered_architectures",
    "timing_paths",
    "timing_summary",
    "timing_table",
    "untimed_component_records",
]
