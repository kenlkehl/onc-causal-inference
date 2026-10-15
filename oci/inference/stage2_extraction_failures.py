"""Extraction-failure mask for Stage 2 measurements (S2A-004).

An extraction "fails" for a (patient, feature) cell when the LLM response for
that cell could not be validated after every repair (unresolved field,
invalid value, unmappable category, structurally invalid response or page)
and the final measured value is null. Such a null is not evidence that the
note does not document the feature, so it must not be encoded as the
"not documented" level.

Representation:

* Value columns keep the failed cell as NaN (no sentinel values).
* A feature whose training rows contain failures gets the definition flag
  ``"extraction_failure_indicator": True`` and a 0/1 companion column
  ``<name>__oci_extraction_failed`` in every modeling frame. Encoders turn
  it into one extra indicator column that belongs to the parent feature's
  column group, so selection penalties and permutation tests treat it with
  its parent. Failed cells get no "not documented" indicator.
* Features without training failures get no companion column; a held-out
  failure for such a feature is encoded as missing (and reported).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

EXTRACTION_FAILURE_COLUMN_SUFFIX = "__oci_extraction_failed"
FAILURE_MASK_SCHEMA_VERSION = "stage2_extraction_failure_mask_v1"
FAILURE_REPORT_SCHEMA_VERSION = "stage2_extraction_failure_report_v1"
TERMINAL_FAILURE_POLICY = (
    "A cell is an extraction failure when an issue event recorded after all repairs "
    "names the feature (or a structural failure covers the feature's request) for "
    "that patient AND the final extracted value is null."
)


def extraction_failure_column(name: Any) -> str:
    return f"{name}{EXTRACTION_FAILURE_COLUMN_SUFFIX}"


def has_failure_indicator(definition: Mapping[str, Any]) -> bool:
    return definition.get("extraction_failure_indicator") is True


def failure_flags(frame: pd.DataFrame, definition: Mapping[str, Any]) -> pd.Series:
    """Boolean failure flags aligned with ``frame`` rows (all False if not tracked)."""

    column = extraction_failure_column(definition["name"])
    if not has_failure_indicator(definition) or column not in frame:
        return pd.Series(np.zeros(len(frame), dtype=bool), index=frame.index)
    values = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    return values.astype(float).gt(0.5)


def companion_columns(
    frame: pd.DataFrame, definitions: Sequence[Mapping[str, Any]]
) -> list[str]:
    return [
        extraction_failure_column(feature["name"])
        for feature in definitions
        if has_failure_indicator(feature)
        and extraction_failure_column(feature["name"]) in frame.columns
    ]


def terminal_failure_cells(summary: Mapping[str, Any] | None) -> dict[str, set[int]]:
    """Read ``terminal_failure_cells`` (feature name -> patient row IDs)."""

    raw = (summary or {}).get("terminal_failure_cells") or {}
    if not isinstance(raw, Mapping):
        raise ValueError("terminal_failure_cells must map feature names to row IDs")
    return {str(name): {int(row) for row in rows} for name, rows in raw.items()}


def failure_mask(
    raw_frame: pd.DataFrame,
    summary: Mapping[str, Any] | None,
    definitions: Sequence[Mapping[str, Any]],
    *,
    latent_dependencies: Mapping[str, Sequence[str]] | None = None,
) -> pd.DataFrame:
    """Return ``_oci_row_id`` plus one 0/1 failure column per definition name.

    ``raw_frame`` holds the final (pre-harmonization) measurements. A latent
    feature fails where any measurement it depends on failed and the latent
    value itself is null.
    """

    cells = terminal_failure_cells(summary)
    row_ids = [int(value) for value in raw_frame["_oci_row_id"]]
    result = pd.DataFrame({"_oci_row_id": row_ids})
    for feature in definitions:
        name = str(feature["name"])
        dependencies = list((latent_dependencies or {}).get(name) or [name])
        failed_rows: set[int] = set().union(*(cells.get(item, set()) for item in dependencies))
        flagged = np.asarray([row in failed_rows for row in row_ids], dtype=bool)
        if name in raw_frame.columns:
            flagged &= raw_frame[name].isna().to_numpy(dtype=bool)
        result[name] = flagged.astype(int)
    return result


def failure_rate_report(
    mask: pd.DataFrame,
    definitions: Sequence[Mapping[str, Any]],
    *,
    scope: str,
    treatment: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Per-feature failure counts and fractions, overall and by treatment arm."""

    rows = len(mask)
    arms = None if treatment is None else np.asarray(treatment, dtype=float)
    if arms is not None and len(arms) != rows:
        raise ValueError("treatment must align with the failure mask rows")
    features = []
    for feature in definitions:
        name = str(feature["name"])
        flags = (
            mask[name].to_numpy(dtype=bool) if name in mask else np.zeros(rows, dtype=bool)
        )
        record: dict[str, Any] = {
            "feature_id": str(feature.get("feature_id") or name),
            "name": name,
            "failed_rows": int(flags.sum()),
            "failure_fraction": float(flags.mean()) if rows else 0.0,
        }
        if arms is not None:
            record["by_treatment_arm"] = {
                str(arm): {
                    "rows": int(np.sum(arms == arm)),
                    "failed_rows": int(np.sum(flags & (arms == arm))),
                    "failure_fraction": (
                        float(np.mean(flags[arms == arm])) if np.any(arms == arm) else None
                    ),
                }
                for arm in (0, 1)
            }
            fractions = [record["by_treatment_arm"][arm]["failure_fraction"] for arm in ("0", "1")]
            record["arm_difference_treated_minus_control"] = (
                float(fractions[1] - fractions[0]) if None not in fractions else None
            )
        features.append(record)
    any_failure = (
        mask[[str(f["name"]) for f in definitions if str(f["name"]) in mask]]
        .to_numpy(dtype=bool)
        .any(axis=1)
        if definitions
        else np.zeros(rows, dtype=bool)
    )
    return {
        "schema_version": FAILURE_REPORT_SCHEMA_VERSION,
        "scope": scope,
        "rows": rows,
        "features": len(features),
        "features_with_failures": int(sum(r["failed_rows"] > 0 for r in features)),
        "failed_cells": int(sum(r["failed_rows"] for r in features)),
        "rows_with_any_failure": int(np.sum(any_failure)),
        "by_treatment_arm_reported": arms is not None,
        "definition": TERMINAL_FAILURE_POLICY,
        "per_feature": features,
    }


def attach_failure_columns(
    frame: pd.DataFrame,
    mask: pd.DataFrame | None,
    definitions: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Add companion columns for flagged definitions, aligned by ``_oci_row_id``."""

    result = frame.copy()
    flagged = [feature for feature in definitions if has_failure_indicator(feature)]
    if not flagged:
        return result
    by_id = (
        mask.set_index("_oci_row_id") if mask is not None else pd.DataFrame(index=[])
    )
    ids = [int(value) for value in result["_oci_row_id"]]
    for feature in flagged:
        name = str(feature["name"])
        if name in by_id.columns:
            values = by_id[name].reindex(ids).fillna(0).astype(int).to_numpy()
        else:
            values = np.zeros(len(ids), dtype=int)
        result[extraction_failure_column(name)] = values
    return result


__all__ = [
    "EXTRACTION_FAILURE_COLUMN_SUFFIX",
    "FAILURE_MASK_SCHEMA_VERSION",
    "TERMINAL_FAILURE_POLICY",
    "attach_failure_columns",
    "companion_columns",
    "extraction_failure_column",
    "failure_flags",
    "failure_mask",
    "failure_rate_report",
    "has_failure_indicator",
    "terminal_failure_cells",
]
