"""Validate exact-context Stage 1 handoff rows against saved split provenance."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any, Dict, Mapping, Sequence


class SplitRegistryError(ValueError):
    """Raised when handoff rows disagree with the saved split provenance."""


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _row_set_fingerprint(row_ids: Sequence[int]) -> str:
    return _stable_hash(sorted(str(value) for value in row_ids))


def validate_handoff_rows_against_split_registry(
    rows: Sequence[Mapping[str, Any]],
    registry: Mapping[str, Any],
) -> None:
    """Require every exact-context handoff row to match the registry exactly."""
    grouped: Dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    seen_fold_keys: set[int] = set()
    for row in rows:
        outer_fold = int(row.get("outer_fold"))
        fold_key = int(row.get("fold_key"))
        if fold_key in seen_fold_keys:
            raise SplitRegistryError(f"Duplicate handoff fold_key={fold_key}")
        seen_fold_keys.add(fold_key)
        grouped[outer_fold].append(row)

    expected_outer_ids = {int(item["outer_fold"]) for item in registry["outer_folds"]}
    if set(grouped) != expected_outer_ids:
        raise SplitRegistryError("Handoff outer folds do not match the explicit split registry")
    for fold in registry["outer_folds"]:
        outer_fold = int(fold["outer_fold"])
        fold_rows = grouped[outer_fold]
        full_rows = [row for row in fold_rows if row.get("scope") == "full_outer_train"]
        inner_rows = {
            int(row.get("inner_fold")): row
            for row in fold_rows
            if row.get("scope") == "candidate_selection_inner_fit"
        }
        if (
            len(full_rows) != 1
            or set(inner_rows) != {int(item["inner_fold"]) for item in fold["inner_folds"]}
            or len(fold_rows) != 1 + len(inner_rows)
        ):
            raise SplitRegistryError(
                f"Handoff exact-context set is incomplete for outer_fold={outer_fold}"
            )
        expected = [(full_rows[0], fold)] + [
            (inner_rows[int(item["inner_fold"])], item) for item in fold["inner_folds"]
        ]
        for row, split in expected:
            label = f"handoff fold_key={row.get('fold_key')}"
            for side in ("fit", "heldout"):
                expected_ids = list(map(int, split[f"{side}_row_ids"]))
                actual_ids = list(map(int, row.get(f"{side}_row_ids") or []))
                if actual_ids != expected_ids:
                    raise SplitRegistryError(
                        f"{label} {side} row ids/order differ from the registry"
                    )
                expected_fingerprint = _row_set_fingerprint(expected_ids)
                if row.get(f"{side}_row_fingerprint") != expected_fingerprint:
                    raise SplitRegistryError(
                        f"{label} {side} row fingerprint differs from the registry"
                    )
                discovery = row.get("discovery") or {}
                discovery_ids = list(map(int, discovery.get(f"{side}_row_ids") or []))
                if discovery_ids != expected_ids:
                    raise SplitRegistryError(
                        f"{label} discovery {side} row ids/order differ from the registry"
                    )
                if discovery.get(f"{side}_row_fingerprint") != expected_fingerprint:
                    raise SplitRegistryError(
                        f"{label} discovery {side} fingerprint differs from the registry"
                    )
