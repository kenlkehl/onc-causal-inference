"""The reserved "documented as unknown" measurement state for Stage 2.

Every Stage 2 variable, whatever its value type, has three non-value states:

* **not documented**: the value is null because the record does not mention
  the variable (or no usable value could be resolved);
* **documented unknown**: the value is the reserved string
  :data:`DOCUMENTED_UNKNOWN` because the record explicitly states that the
  value is unknown, indeterminate, pending, not assessed, or equivocal;
* **extraction failure**: the value is null and the cell is flagged in the
  ``<name>__oci_extraction_failed`` companion column
  (:mod:`oci.inference.stage2_extraction_failures`).

Representation: the reserved string stays in the value column of every raw,
harmonized and modeling frame, for every value type. It is never a declared
category. Encoders give it one 0/1 indicator column in the parent feature's
column group, separate from the not-documented (missing) indicator and the
extraction-failure indicator; a continuous feature's value column treats it
like a missing value (median-imputed) and a categorical feature never counts
it as a category level.

Unknown-like wording is mapped to the reserved value conservatively: only a
whole value equal to one of :data:`UNKNOWN_LIKE_PHRASES` (case, spacing and
punctuation ignored) is mapped, so a category such as "Unknown primary" is
never changed. A declared category with such a label is removed from the
definition only when the remaining categories still form a valid ontology
(two for binary, at least two otherwise); otherwise it is a genuine answer and
is kept.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

DOCUMENTED_UNKNOWN = "documented_unknown"
DOCUMENTED_UNKNOWN_REPORT_SCHEMA_VERSION = "stage2_documented_unknown_report_v1"
# Whole-value phrases (after normalization) that mean the record states the
# value is unknown. Deliberately short: "not tested", "not performed", "not
# done", "n/a" and "none" are excluded because they are often genuine answers
# or ambiguous.
UNKNOWN_LIKE_PHRASES = (
    "unknown",
    "not known",
    "status unknown",
    "indeterminate",
    "undetermined",
    "pending",
    "result pending",
    "results pending",
    "not assessed",
    "not evaluated",
    "equivocal",
    "inconclusive",
)
DOCUMENTED_UNKNOWN_RULE = (
    "Answer documented_unknown when the record explicitly states that the value is "
    "unknown, indeterminate, pending, not assessed, or equivocal and documents no usable "
    "value; answer null when the record does not mention the variable."
)
DOCUMENTED_UNKNOWN_POLICY = (
    "documented_unknown is a reserved value accepted for every variable type when the "
    "record explicitly states that the value is unknown, indeterminate, pending, not "
    "assessed, or equivocal; null means the record does not mention the variable or no "
    "usable value was resolved; an extraction failure is a null flagged after all repairs. "
    "Only a whole value equal to one of the listed unknown-like phrases (case, spacing and "
    "punctuation ignored) is mapped to documented_unknown, and a declared category with "
    "such a label is removed only when the remaining categories still form a valid "
    "ontology. Encoders give documented_unknown its own indicator column in the parent "
    "feature's group; the informativeness filter counts it as an observed state."
)


def _phrase_key(value: str) -> str:
    return re.sub(r"[\W_]+", " ", value, flags=re.UNICODE).strip().casefold()


_UNKNOWN_LIKE_KEYS = frozenset(_phrase_key(phrase) for phrase in UNKNOWN_LIKE_PHRASES)


def is_documented_unknown(value: Any) -> bool:
    """True for the reserved value itself (surrounding space and case ignored)."""

    return isinstance(value, str) and value.strip().casefold() == DOCUMENTED_UNKNOWN


def is_malformed_documented_unknown(value: Any) -> bool:
    """A string that starts like the reserved value but is not exactly it."""

    return (
        isinstance(value, str)
        and not is_documented_unknown(value)
        and _phrase_key(value).startswith("documented unknown")
    )


def is_unknown_like(value: Any) -> bool:
    """True for the reserved value or a whole unknown-like phrase."""

    if not isinstance(value, str):
        return False
    return is_documented_unknown(value) or _phrase_key(value) in _UNKNOWN_LIKE_KEYS


def documented_unknown_mask(series: pd.Series) -> pd.Series:
    """Boolean mask of cells holding the reserved value."""

    return pd.Series(
        [is_documented_unknown(value) for value in series.tolist()],
        index=series.index,
        dtype=bool,
    )


def without_documented_unknown(series: pd.Series) -> pd.Series:
    """The series with reserved cells set to missing (for value encodings)."""

    mask = documented_unknown_mask(series)
    if not bool(mask.any()):
        return series
    return series.astype(object).where(~mask, None)


def split_unknown_like_categories(
    value_type: Any, categories: Sequence[str]
) -> tuple[list[str], list[str]]:
    """Return (kept, removed): unknown-like labels removed if the rest stays valid."""

    kept = [category for category in categories if not is_unknown_like(category)]
    removed = [category for category in categories if is_unknown_like(category)]
    if not removed:
        return list(categories), []
    normalized_type = str(value_type or "").strip().lower()
    reserved = [category for category in removed if is_documented_unknown(category)]
    valid = (len(kept) == 2) if normalized_type == "binary" else (len(kept) >= 2)
    if valid:
        return kept, removed
    # The labels are genuine answers of this ontology; only the reserved value
    # itself can never be a declared category.
    genuine = [category for category in categories if category not in reserved]
    return genuine, reserved


def documented_unknown_report(
    frame: pd.DataFrame,
    definitions: Sequence[Mapping[str, Any]],
    *,
    scope: str,
    treatment: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Per-feature counts and fractions of documented_unknown, overall and by arm."""

    rows = len(frame)
    arms = None if treatment is None else np.asarray(treatment, dtype=float)
    if arms is not None and len(arms) != rows:
        raise ValueError("treatment must align with the measurement rows")
    features = []
    for feature in definitions:
        name = str(feature["name"])
        flags = (
            documented_unknown_mask(frame[name]).to_numpy(dtype=bool)
            if name in frame.columns
            else np.zeros(rows, dtype=bool)
        )
        record: dict[str, Any] = {
            "feature_id": str(feature.get("feature_id") or name),
            "name": name,
            "documented_unknown_rows": int(flags.sum()),
            "documented_unknown_fraction": float(flags.mean()) if rows else 0.0,
        }
        if arms is not None:
            record["by_treatment_arm"] = {
                str(arm): {
                    "rows": int(np.sum(arms == arm)),
                    "documented_unknown_rows": int(np.sum(flags & (arms == arm))),
                    "documented_unknown_fraction": (
                        float(np.mean(flags[arms == arm])) if np.any(arms == arm) else None
                    ),
                }
                for arm in (0, 1)
            }
            fractions = [
                record["by_treatment_arm"][arm]["documented_unknown_fraction"]
                for arm in ("0", "1")
            ]
            record["arm_difference_treated_minus_control"] = (
                float(fractions[1] - fractions[0]) if None not in fractions else None
            )
        features.append(record)
    return {
        "schema_version": DOCUMENTED_UNKNOWN_REPORT_SCHEMA_VERSION,
        "scope": scope,
        "rows": rows,
        "features": len(features),
        "features_with_documented_unknown": int(
            sum(record["documented_unknown_rows"] > 0 for record in features)
        ),
        "documented_unknown_cells": int(
            sum(record["documented_unknown_rows"] for record in features)
        ),
        "by_treatment_arm_reported": arms is not None,
        "definition": DOCUMENTED_UNKNOWN_POLICY,
        "per_feature": features,
    }


__all__ = [
    "DOCUMENTED_UNKNOWN",
    "DOCUMENTED_UNKNOWN_POLICY",
    "DOCUMENTED_UNKNOWN_REPORT_SCHEMA_VERSION",
    "DOCUMENTED_UNKNOWN_RULE",
    "UNKNOWN_LIKE_PHRASES",
    "documented_unknown_mask",
    "documented_unknown_report",
    "is_documented_unknown",
    "is_malformed_documented_unknown",
    "is_unknown_like",
    "split_unknown_like_categories",
    "without_documented_unknown",
]
