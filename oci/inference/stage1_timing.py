"""Reporting-only wall-time records for Stage 1 architectures.

Each Stage 1 discovery context records how long its sections took and which
architecture each section serves.  A section that several architectures need
(for example the bag-of-words nuisance fits or the embedding cache) is
attributed to ``"shared"`` and lists the architectures that consume it.

Timing never enters a checkpoint fingerprint, a handoff row, or any fitted
value: the records are written beside the context artifacts and read only by
the post-hoc evaluator.  A context restored from its checkpoint keeps the
record written when it was computed; a context restored from a checkpoint
that predates timing is recorded with status ``"unknown"``.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from .stage1_architectures import (
    BOW_NUISANCE,
    BOW_R_LOSS,
    EMBEDDING_CLUSTERED,
    EMBEDDING_WHOLE_COHORT,
    HTR_NEURAL,
    MATCHED_PAIR_UPLIFT,
    NEURAL_QUERY_MOMENTS,
    TFIDF_ORPHAN_NGRAMS,
    TFIDF_SEMANTIC_RETRIEVAL,
    TFIDF_TOPICS,
)

STAGE1_TIMING_SCHEMA_VERSION = "stage1_architecture_timing_v1"
SHARED_ATTRIBUTION = "shared"
CONTEXT_TIMING_FILE = "timing.json"
COMPONENT_TIMING_FILE = "timing.jsonl"

TEXT_MODEL_ARCHITECTURES = (
    BOW_NUISANCE,
    BOW_R_LOSS,
    MATCHED_PAIR_UPLIFT,
    HTR_NEURAL,
    EMBEDDING_WHOLE_COHORT,
    EMBEDDING_CLUSTERED,
    TFIDF_SEMANTIC_RETRIEVAL,
)
EMBEDDING_ARCHITECTURES = (
    EMBEDDING_WHOLE_COHORT,
    EMBEDDING_CLUSTERED,
    TFIDF_SEMANTIC_RETRIEVAL,
)
EMBEDDING_CACHE_CONSUMERS = (*EMBEDDING_ARCHITECTURES, NEURAL_QUERY_MOMENTS)
TFIDF_ARCHITECTURES = (TFIDF_TOPICS, TFIDF_ORPHAN_NGRAMS)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


class SectionTimer:
    """Attribute consecutive wall-clock sections of one discovery context.

    ``lap`` charges the time since the previous lap (or since construction or
    ``skip``) to one named unit.  Laps with the same unit accumulate.  When a
    checkpoint store is supplied, the units it restored during each section are
    listed on that section, because their time reflects loading rather than
    fitting.
    """

    def __init__(
        self,
        *,
        checkpoint_store: Any = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._clock = clock
        self._start = clock()
        self._last = self._start
        self._started_at = _now()
        self._store = checkpoint_store
        self._reused_seen = len(self._reused_units())
        self._entries: dict[str, dict[str, Any]] = {}

    def _reused_units(self) -> list[str]:
        return list(getattr(self._store, "reused_units", None) or [])

    def lap(
        self,
        unit: str,
        *,
        attribution: str,
        consumers: Sequence[str] | None = None,
    ) -> None:
        now = self._clock()
        elapsed = max(0.0, float(now - self._last))
        self._last = now
        entry = self._entries.get(unit)
        if entry is None:
            entry = {
                "unit": str(unit),
                "attribution": str(attribution),
                "consumers": sorted(
                    {str(value) for value in (consumers or (attribution,))}
                ),
                "seconds": 0.0,
                "laps": 0,
                "restored_checkpoint_units": [],
            }
            self._entries[unit] = entry
        elif entry["attribution"] != str(attribution):
            raise ValueError(f"timing unit {unit!r} changed its attribution")
        entry["seconds"] += elapsed
        entry["laps"] += 1
        reused = self._reused_units()
        entry["restored_checkpoint_units"].extend(reused[self._reused_seen :])
        self._reused_seen = len(reused)

    def skip(self) -> None:
        """Exclude the time since the previous lap from every unit."""

        self._last = self._clock()

    def record(
        self,
        *,
        component: str,
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        total = max(0.0, float(self._clock() - self._start))
        entries = [dict(entry) for entry in self._entries.values()]
        attributed = sum(float(entry["seconds"]) for entry in entries)
        overhead = total - attributed
        if overhead > 0.0:
            entries.append(
                {
                    "unit": "context_overhead",
                    "attribution": SHARED_ATTRIBUTION,
                    "consumers": [],
                    "seconds": overhead,
                    "laps": 0,
                    "restored_checkpoint_units": [],
                }
            )
        restored = sorted(
            {
                unit
                for entry in entries
                for unit in entry.get("restored_checkpoint_units") or []
            }
        )
        return {
            "schema_version": STAGE1_TIMING_SCHEMA_VERSION,
            "component": str(component),
            **_context_fields(context),
            "status": "partially_restored" if restored else "measured",
            "restored_checkpoint_units": restored,
            "started_at": self._started_at,
            "completed_at": _now(),
            "wall_seconds": total,
            "entries": entries,
        }


class NullTimer:
    """A timer that records nothing, for callers outside the workflow."""

    def lap(self, unit: str, *, attribution: str, consumers: Sequence[str] | None = None) -> None:
        del unit, attribution, consumers

    def skip(self) -> None:
        return None


def _context_fields(context: Mapping[str, Any] | None) -> dict[str, Any]:
    context = dict(context or {})
    inner = context.get("inner_fold")
    outer = context.get("outer_fold")
    return {
        "scope_id": None if context.get("scope_id") is None else str(context["scope_id"]),
        "outer_fold": None if outer is None else int(outer),
        "inner_fold": None if inner is None else int(inner),
        "scope": None if context.get("scope") is None else str(context["scope"]),
    }


def unknown_timing_record(
    *,
    component: str,
    context: Mapping[str, Any] | None = None,
    reason: str = "restored from a checkpoint written before timing was recorded",
) -> dict[str, Any]:
    return {
        "schema_version": STAGE1_TIMING_SCHEMA_VERSION,
        "component": str(component),
        **_context_fields(context),
        "status": "unknown",
        "reason": str(reason),
        "wall_seconds": None,
        "entries": [],
    }


def write_timing_record(path: Path, record: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(record), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_timing_record(path: Path) -> Optional[dict[str, Any]]:
    """Return a saved record, or ``None`` when absent, unreadable, or foreign."""

    path = Path(path)
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(value, dict) or value.get("schema_version") != STAGE1_TIMING_SCHEMA_VERSION:
        return None
    return value


def restored_timing_record(
    path: Path,
    *,
    component: str,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a restored context's original record, or an ``unknown`` record."""

    saved = read_timing_record(path)
    if saved is None:
        return unknown_timing_record(component=component, context=context)
    return {**saved, "restored_from_checkpoint": True}


def write_component_timing(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


__all__ = [
    "COMPONENT_TIMING_FILE",
    "CONTEXT_TIMING_FILE",
    "EMBEDDING_ARCHITECTURES",
    "EMBEDDING_CACHE_CONSUMERS",
    "NullTimer",
    "SHARED_ATTRIBUTION",
    "STAGE1_TIMING_SCHEMA_VERSION",
    "SectionTimer",
    "TEXT_MODEL_ARCHITECTURES",
    "TFIDF_ARCHITECTURES",
    "read_timing_record",
    "restored_timing_record",
    "unknown_timing_record",
    "write_component_timing",
    "write_timing_record",
]
