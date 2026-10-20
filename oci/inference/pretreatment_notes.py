"""Dated-notes input for the all-evidence workflow.

When a run configuration has a ``notes`` block, each patient's analysis text is
built from a long-format notes table (one row per note) instead of being read
from one cohort column. Every cohort patient has an index date (the treatment
decision or treatment start). Notes dated before the index day (or on it, when
``include_index_date`` is set) and within the optional look-back window are
kept, ordered by date, given a header that states their position and their
distance from the index date in days (never a calendar date), and joined into
the configured ``columns.text`` column.

The derived cohort is written once to ``OUTPUT_DIR/input/analysis_cohort.parquet``
with a manifest and a per-patient audit. Later starts of the same run verify
the sources, the configuration, and the derived file against the manifest and
reuse the file without rewriting it: Stage 1 checkpoints identify the analysis
file by path, size, and modification time, so a rewrite would invalidate them.
Any mismatch fails closed.

Dates are compared by calendar day. Timezone-aware values are converted to UTC
before the day is taken, and timezone-naive values are used as given.

Only the configured note columns are read from the notes table, so oracle or
other unrelated columns in it never reach the analysis cohort. The audit is for
investigators; no modeling or LLM step reads it.

This module imports only the standard library, NumPy, and pandas (PyArrow only
to read a Parquet schema), so configuration parsing stays lightweight.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import os
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

MANIFEST_SCHEMA_VERSION = "pretreatment_notes_manifest_v1"
INPUT_DIRECTORY = "input"
ANALYSIS_COHORT_FILE = "analysis_cohort.parquet"
MANIFEST_FILE = "pretreatment_notes_manifest.json"
AUDIT_FILE = "pretreatment_notes_audit.csv"

NOTE_HEADERS = ("relative_days", "none")

# Reporting order of the note categories (manifest and audit).
NOTE_CATEGORIES = (
    "retained",
    "on_index_date",
    "after_index_date",
    "before_lookback",
    "undated",
    "unmatched_patient",
    "empty_text",
)
# A note is counted in the first category that applies, in this order. The
# date categories come before ``empty_text`` so that the audit shows every note
# the date rule removed, whatever its content.
CATEGORY_PRECEDENCE = (
    "unmatched_patient",
    "undated",
    "after_index_date",
    "on_index_date",
    "before_lookback",
    "empty_text",
    "retained",
)
# Notes of an unmatched patient belong to no cohort patient, so the per-patient
# audit has no column for them; the manifest reports their total.
PATIENT_AUDIT_CATEGORIES = tuple(
    category for category in NOTE_CATEGORIES if category != "unmatched_patient"
)

_NOTES_KEYS = frozenset(
    {
        "path",
        "columns",
        "index_date_column",
        "include_index_date",
        "lookback_days",
        "note_header",
    }
)
_NOTES_COLUMN_KEYS = ("unit_id", "date", "text", "note_type")
_PIPELINE_ROOTS = ("components", "handoff", "stage2")
_HEADER_SEPARATOR = " · "
_NOTE_SEPARATOR = "\n\n"
_HASH_CHUNK_BYTES = 8 * 1024 * 1024
_EXAMPLE_LIMIT = 5


@dataclass(frozen=True)
class PretreatmentNotesConfig:
    """Resolved ``notes`` block of an all-evidence run configuration."""

    path: Path
    unit_id_column: str
    date_column: str
    text_column: str
    note_type_column: str | None
    index_date_column: str
    include_index_date: bool = False
    lookback_days: int | None = None
    note_header: str = "relative_days"

    def as_dict(self) -> dict[str, Any]:
        """Return the configuration-file form, which ``compile_config`` reads back."""

        return {
            "path": str(self.path),
            "columns": {
                "unit_id": self.unit_id_column,
                "date": self.date_column,
                "text": self.text_column,
                "note_type": self.note_type_column,
            },
            "index_date_column": self.index_date_column,
            "include_index_date": self.include_index_date,
            "lookback_days": self.lookback_days,
            "note_header": self.note_header,
        }

    @property
    def source_columns(self) -> list[str]:
        """The only notes-table columns that are read."""

        names = [self.unit_id_column, self.date_column, self.text_column]
        if self.note_type_column is not None:
            names.append(self.note_type_column)
        return names


@dataclass(frozen=True)
class PretreatmentText:
    """Derived text for every cohort row plus the audit of every note."""

    texts: list[str]
    audit: pd.DataFrame
    note_counts: dict[str, int]
    note_categories: list[str]


def input_directory(output_dir: Path | str) -> Path:
    return Path(output_dir) / INPUT_DIRECTORY


def analysis_cohort_path(output_dir: Path | str) -> Path:
    return input_directory(output_dir) / ANALYSIS_COHORT_FILE


def manifest_path(output_dir: Path | str) -> Path:
    return input_directory(output_dir) / MANIFEST_FILE


def audit_path(output_dir: Path | str) -> Path:
    return input_directory(output_dir) / AUDIT_FILE


def _column_name(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty column name")
    return value


def notes_config_from_mapping(
    raw: Any,
    *,
    config_dir: Path,
    default_unit_id_column: str,
) -> PretreatmentNotesConfig | None:
    """Validate a ``notes`` block; ``None`` means the run uses one text column."""

    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("notes must be a configuration object or null")
    unknown = sorted(set(map(str, raw)) - _NOTES_KEYS)
    if unknown:
        raise ValueError(
            f"notes contains unsupported fields: {unknown}. Fix the spelling or "
            "remove them; the workflow does not ignore unknown settings."
        )
    path_value = raw.get("path")
    if not isinstance(path_value, (str, os.PathLike)) or not str(path_value).strip():
        raise ValueError("notes.path is required: a CSV or Parquet table with one row per note")
    columns = raw.get("columns")
    if columns is None:
        columns = {}
    if not isinstance(columns, Mapping):
        raise ValueError("notes.columns must be a configuration object")
    unknown_columns = sorted(set(map(str, columns)) - set(_NOTES_COLUMN_KEYS))
    if unknown_columns:
        raise ValueError(
            f"notes.columns contains unsupported fields: {unknown_columns}. "
            f"Supported fields: {list(_NOTES_COLUMN_KEYS)}"
        )

    def column(key: str, default: str) -> str:
        value = columns.get(key)
        return _column_name(default if value is None else value, f"notes.columns.{key}")

    unit_id_column = column("unit_id", default_unit_id_column)
    date_column = column("date", "note_date")
    text_column = column("text", "note_text")
    note_type_value = columns.get("note_type")
    note_type_column = (
        None
        if note_type_value is None
        else _column_name(note_type_value, "notes.columns.note_type")
    )
    names = [unit_id_column, date_column, text_column]
    if note_type_column is not None:
        names.append(note_type_column)
    if len(set(names)) != len(names):
        raise ValueError(f"notes.columns must name distinct columns: {names}")
    index_date_column = raw.get("index_date_column")
    if not isinstance(index_date_column, str) or not index_date_column.strip():
        raise ValueError(
            "notes.index_date_column is required: the cohort column holding each "
            "patient's index date"
        )
    include_index_date = raw.get("include_index_date", False)
    if not isinstance(include_index_date, bool):
        raise ValueError("notes.include_index_date must be true or false")
    lookback_days = raw.get("lookback_days")
    if lookback_days is not None and (
        isinstance(lookback_days, bool)
        or not isinstance(lookback_days, int)
        or lookback_days < 1
    ):
        raise ValueError("notes.lookback_days must be a positive integer or null")
    note_header = raw.get("note_header", "relative_days")
    if note_header not in NOTE_HEADERS:
        raise ValueError(f"notes.note_header must be one of {list(NOTE_HEADERS)}")
    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        path = Path(config_dir) / path
    return PretreatmentNotesConfig(
        path=path.resolve(),
        unit_id_column=unit_id_column,
        date_column=date_column,
        text_column=text_column,
        note_type_column=note_type_column,
        index_date_column=index_date_column,
        include_index_date=include_index_date,
        lookback_days=lookback_days,
        note_header=note_header,
    )


def read_table(
    path: Path,
    *,
    columns: list[str] | None = None,
    description: str = "dataset",
) -> pd.DataFrame:
    """Read a Parquet or CSV table, optionally only the named columns."""

    suffix = Path(path).suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path, columns=columns)
    if suffix == ".csv":
        return pd.read_csv(path, usecols=columns)
    raise ValueError(f"{description} must be Parquet or CSV")


def _table_columns(path: Path, *, description: str) -> list[str]:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        import pyarrow.parquet as pq

        return [str(name) for name in pq.read_schema(path).names]
    if suffix == ".csv":
        return [str(name) for name in pd.read_csv(path, nrows=0).columns]
    raise ValueError(f"{description} must be Parquet or CSV")


def read_notes_table(config: PretreatmentNotesConfig) -> pd.DataFrame:
    """Read only the configured note columns, in configuration order."""

    if not config.path.is_file():
        raise FileNotFoundError(f"notes.path does not exist: {config.path}")
    available = _table_columns(config.path, description="notes.path")
    wanted = config.source_columns
    missing = [name for name in wanted if name not in available]
    if missing:
        raise ValueError(
            f"notes table is missing configured columns: {missing} "
            "(set notes.columns to the table's column names)"
        )
    frame = read_table(config.path, columns=wanted, description="notes.path")
    if any(name not in frame.columns for name in wanted):
        # A Parquet file may restore a stored pandas index from a requested column.
        frame = frame.reset_index()
    return frame[wanted].reset_index(drop=True)


def _examples(values: Any) -> list[Any]:
    output = []
    for value in list(values)[:_EXAMPLE_LIMIT]:
        output.append(value.item() if isinstance(value, np.generic) else value)
    return output


def calendar_days(values: pd.Series, *, label: str) -> pd.Series:
    """Parse dates and return their calendar day as timezone-naive midnight.

    Timezone-aware values are converted to UTC first; naive values are used as
    given. Missing or unparseable values become ``NaT``. Elements that the
    vectorized parser cannot read under the format it infers from the column are
    parsed again one at a time, so a column that mixes date-only and date-time
    text keeps every readable value.
    """

    values = values.reset_index(drop=True)
    if (
        pd.api.types.is_numeric_dtype(values.dtype)
        and not pd.api.types.is_datetime64_any_dtype(values.dtype)
        and values.notna().any()
    ):
        raise ValueError(
            f"{label} holds numbers, not dates; store dates as ISO 8601 text "
            "(YYYY-MM-DD) or as a date type"
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(values, errors="coerce", utc=True)
        present = values.notna()
        if not pd.api.types.is_datetime64_any_dtype(values.dtype):
            present &= values.astype(str).str.strip().ne("")
        retry = parsed.isna() & present
        if retry.any():
            parsed = parsed.copy()
            parsed[retry] = pd.to_datetime(
                values[retry],
                errors="coerce",
                utc=True,
                format="mixed",
            )
    return parsed.dt.tz_localize(None).dt.normalize()


def _day_numbers(days: pd.Series) -> np.ndarray:
    return days.to_numpy().astype("datetime64[D]")


def _normalized_note_type(value: Any) -> str:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    return " ".join(str(value).split())


def note_header(position: int, total: int, days_before: int, note_type: str = "") -> str:
    """Header line of one retained note; it never contains a calendar date."""

    if days_before == 0:
        when = "on index date"
    elif days_before == 1:
        when = "1 day before index date"
    else:
        when = f"{days_before} days before index date"
    parts = [f"Note {position} of {total}", when]
    if note_type:
        parts.append(note_type)
    return "[" + _HEADER_SEPARATOR.join(parts) + "]"


def _validate_cohort(
    cohort: pd.DataFrame,
    *,
    config: PretreatmentNotesConfig,
    unit_id_column: str,
    text_column: str,
) -> pd.Series:
    if cohort.empty:
        raise ValueError("the cohort file has no rows")
    missing = [
        name
        for name in (unit_id_column, config.index_date_column)
        if name not in cohort.columns
    ]
    if missing:
        raise ValueError(
            f"the cohort file is missing columns needed for dated notes: {missing} "
            "(columns.unit_id and notes.index_date_column)"
        )
    if text_column in cohort.columns:
        raise ValueError(
            f"the cohort file already has a column named {text_column!r}. With notes "
            "configured, columns.text names the text column that OCI derives from the "
            "notes table; drop or rename the existing column, or set columns.text to "
            "an unused name"
        )
    ids = cohort[unit_id_column]
    if ids.isna().any():
        raise ValueError(
            f"cohort unit IDs must be nonmissing: {int(ids.isna().sum())} rows of "
            f"{unit_id_column!r} are missing"
        )
    duplicated = ids[ids.duplicated(keep=False)]
    if len(duplicated):
        raise ValueError(
            f"cohort unit IDs must be unique: {int(duplicated.nunique())} values of "
            f"{unit_id_column!r} repeat, for example {_examples(duplicated.unique())}"
        )
    index_days = calendar_days(
        cohort[config.index_date_column],
        label=f"cohort column {config.index_date_column!r}",
    )
    invalid = index_days.isna().to_numpy()
    if invalid.any():
        raise ValueError(
            f"every cohort patient needs an index date: {int(invalid.sum())} of "
            f"{len(cohort)} values of {config.index_date_column!r} are missing or "
            f"unparseable, for example unit IDs {_examples(ids.to_numpy()[invalid])}"
        )
    return index_days


def build_pretreatment_text(
    cohort: pd.DataFrame,
    notes: pd.DataFrame,
    *,
    config: PretreatmentNotesConfig,
    unit_id_column: str,
    text_column: str,
) -> PretreatmentText:
    """Classify every note and build each cohort patient's pretreatment text.

    ``cohort`` is the cohort table in its analysis row order and ``notes`` the
    notes table with the configured columns. Validation fails closed on
    missing or duplicate cohort unit IDs, a missing or unparseable index date,
    a cohort that already holds ``text_column``, and missing note columns.
    """

    cohort = cohort.reset_index(drop=True)
    notes = notes.reset_index(drop=True)
    missing_note_columns = [name for name in config.source_columns if name not in notes.columns]
    if missing_note_columns:
        raise ValueError(f"notes table is missing configured columns: {missing_note_columns}")
    index_days = _validate_cohort(
        cohort,
        config=config,
        unit_id_column=unit_id_column,
        text_column=text_column,
    )
    patient_count = len(cohort)
    note_count = len(notes)
    positions = pd.Index(cohort[unit_id_column]).get_indexer(notes[config.unit_id_column])
    matched = positions >= 0
    if note_count and not matched.any():
        raise ValueError(
            f"no note's {config.unit_id_column!r} value matches a cohort "
            f"{unit_id_column!r} value; check notes.columns.unit_id and that both "
            "files store identifiers with the same type"
        )
    note_days = calendar_days(
        notes[config.date_column],
        label=f"notes column {config.date_column!r}",
    )
    note_day_numbers = _day_numbers(note_days)
    undated = np.isnat(note_day_numbers)
    # Unmatched and undated notes get a placeholder day; precedence classifies
    # them before any day comparison is consulted.
    patient_days = _day_numbers(index_days)[np.where(matched, positions, 0)]
    comparable_note_days = np.where(undated, patient_days, note_day_numbers)
    days_before = (patient_days - comparable_note_days).astype(np.int64)
    bodies = [
        "" if value is None or (not isinstance(value, str) and pd.isna(value)) else str(value).strip()
        for value in notes[config.text_column].tolist()
    ]
    empty = np.fromiter((not body for body in bodies), dtype=bool, count=note_count)
    lookback = config.lookback_days
    conditions = {
        "unmatched_patient": ~matched,
        "undated": undated,
        "after_index_date": days_before < 0,
        "on_index_date": (days_before == 0) & (not config.include_index_date),
        "before_lookback": (
            days_before > lookback if lookback is not None else np.zeros(note_count, dtype=bool)
        ),
        "empty_text": empty,
    }
    codes = np.select(
        [conditions[name] for name in CATEGORY_PRECEDENCE[:-1]],
        list(range(len(CATEGORY_PRECEDENCE) - 1)),
        default=len(CATEGORY_PRECEDENCE) - 1,
    ).astype(np.int64)
    category_codes = {name: code for code, name in enumerate(CATEGORY_PRECEDENCE)}
    note_counts = {
        name: int(np.count_nonzero(codes == category_codes[name])) for name in NOTE_CATEGORIES
    }
    note_counts["total"] = note_count

    per_patient = {
        name: np.bincount(
            positions[matched & (codes == category_codes[name])],
            minlength=patient_count,
        ).astype(np.int64)
        for name in PATIENT_AUDIT_CATEGORIES
    }
    if config.note_type_column is not None:
        note_types = [_normalized_note_type(value) for value in notes[config.note_type_column].tolist()]
    else:
        note_types = [""] * note_count

    texts = [""] * patient_count
    earliest: list[Any] = [pd.NA] * patient_count
    latest: list[Any] = [pd.NA] * patient_count
    retained = np.flatnonzero(codes == category_codes["retained"])
    # Date ascending within each patient; equal days keep the notes-table order.
    ordered = retained[
        np.lexsort(
            (
                retained,
                note_day_numbers[retained].astype(np.int64),
                positions[retained],
            )
        )
    ]
    for patient, group in itertools.groupby(ordered.tolist(), key=lambda row: int(positions[row])):
        rows = list(group)
        pieces = []
        for number, row in enumerate(rows, start=1):
            if config.note_header == "relative_days":
                header = note_header(number, len(rows), int(days_before[row]), note_types[row])
                pieces.append(f"{header}\n{bodies[row]}")
            else:
                pieces.append(bodies[row])
        texts[patient] = _NOTE_SEPARATOR.join(pieces)
        earliest[patient] = int(max(days_before[row] for row in rows))
        latest[patient] = int(min(days_before[row] for row in rows))

    audit = pd.DataFrame(
        {
            "unit_id": cohort[unit_id_column].to_numpy(),
            **per_patient,
            "retained_characters": np.fromiter(
                (len(text) for text in texts), dtype=np.int64, count=patient_count
            ),
            "earliest_retained_days_before_index": pd.array(earliest, dtype="Int64"),
            "latest_retained_days_before_index": pd.array(latest, dtype="Int64"),
        }
    )
    return PretreatmentText(
        texts=texts,
        audit=audit,
        note_counts=note_counts,
        note_categories=[CATEGORY_PRECEDENCE[code] for code in codes.tolist()],
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "size_bytes": int(path.stat().st_size),
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _temporary(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.tmp")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = _temporary(path)
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _distribution(values: np.ndarray) -> dict[str, Any]:
    return {
        "min": int(np.min(values)),
        "median": float(np.median(values)),
        "max": int(np.max(values)),
    }


def _has_pipeline_artifacts(output_dir: Path) -> bool:
    for name in _PIPELINE_ROOTS:
        root = Path(output_dir) / name
        if root.is_dir() and any(path.is_file() for path in root.rglob("*")):
            return True
    return False


def check_output_directory(
    output_dir: Path,
    notes: PretreatmentNotesConfig | None,
) -> None:
    """Refuse to mix dated-notes and single-text-column runs in one directory.

    This check reads only the directory listing. A directory that holds a
    dated-notes manifest requires a ``notes`` block. A ``notes`` block requires
    either that manifest or a directory without Stage 1 or Stage 2 output, so
    the analysis cohort is never built under results computed from other text.
    """

    manifest = manifest_path(output_dir)
    if notes is None:
        if manifest.is_file():
            raise ValueError(
                f"the output directory holds a dated-notes run ({manifest}) but the "
                "configuration has no notes block; configure the same notes block "
                "(for example with --config OUTPUT_DIR/run_config.json) or use a "
                "fresh output directory"
            )
        return
    if not manifest.is_file() and _has_pipeline_artifacts(output_dir):
        raise ValueError(
            "notes are configured, but the output directory already holds pipeline "
            f"output without a dated-notes manifest ({manifest}), for example a run "
            "that read one text column; use a fresh output directory"
        )


def _expected_identity(
    *,
    notes: PretreatmentNotesConfig,
    unit_id_column: str,
    text_column: str,
) -> dict[str, Any]:
    return {
        "notes_config": notes.as_dict(),
        "cohort_columns": {"unit_id": unit_id_column, "text": text_column},
    }


def _verify_manifest(
    manifest_file: Path,
    *,
    cohort_path: Path,
    notes: PretreatmentNotesConfig,
    output_dir: Path,
    unit_id_column: str,
    text_column: str,
) -> None:
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read the dated-notes manifest: {manifest_file}") from exc
    if not isinstance(manifest, Mapping):
        raise ValueError(f"the dated-notes manifest must hold one JSON object: {manifest_file}")
    problems: list[str] = []
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        problems.append(f"manifest schema_version is not {MANIFEST_SCHEMA_VERSION!r}")
    expected = _expected_identity(
        notes=notes,
        unit_id_column=unit_id_column,
        text_column=text_column,
    )
    if manifest.get("notes_config") != expected["notes_config"]:
        problems.append("the notes configuration differs")
    if manifest.get("cohort_columns") != expected["cohort_columns"]:
        problems.append("columns.unit_id or columns.text differs")
    sources = manifest.get("sources")
    sources = sources if isinstance(sources, Mapping) else {}
    for key, path in (("cohort", Path(cohort_path)), ("notes", notes.path)):
        recorded = sources.get(key)
        recorded = recorded if isinstance(recorded, Mapping) else {}
        if recorded.get("path") != str(path):
            problems.append(f"the {key} file path differs (recorded {recorded.get('path')!r})")
        elif not path.is_file():
            problems.append(f"the {key} file is missing: {path}")
        elif file_sha256(path) != recorded.get("sha256"):
            problems.append(f"the {key} file contents differ (SHA-256): {path}")
    analysis = analysis_cohort_path(output_dir)
    recorded_analysis = manifest.get("analysis_cohort")
    recorded_analysis = recorded_analysis if isinstance(recorded_analysis, Mapping) else {}
    if not analysis.is_file():
        problems.append(f"the derived analysis cohort is missing: {analysis}")
    elif file_sha256(analysis) != recorded_analysis.get("sha256"):
        problems.append(f"the derived analysis cohort differs from its recorded SHA-256: {analysis}")
    if problems:
        raise ValueError(
            "dated-notes input does not match this output directory's manifest "
            f"({manifest_file}): " + "; ".join(problems) + ". The derived analysis "
            "cohort cannot change within an output directory; restore the original "
            "inputs and settings, or use a fresh output directory"
        )


def _materialize(
    *,
    cohort_path: Path,
    notes: PretreatmentNotesConfig,
    output_dir: Path,
    unit_id_column: str,
    text_column: str,
) -> None:
    cohort_path = Path(cohort_path)
    cohort = read_table(cohort_path).reset_index(drop=True)
    notes_frame = read_notes_table(notes)
    result = build_pretreatment_text(
        cohort,
        notes_frame,
        config=notes,
        unit_id_column=unit_id_column,
        text_column=text_column,
    )
    frame = cohort.copy()
    frame[text_column] = result.texts
    directory = input_directory(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    analysis = analysis_cohort_path(output_dir)
    temporary = _temporary(analysis)
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, analysis)
    audit = audit_path(output_dir)
    temporary = _temporary(audit)
    result.audit.to_csv(temporary, index=False)
    os.replace(temporary, audit)

    retained_per_patient = result.audit["retained"].to_numpy()
    characters = result.audit["retained_characters"].to_numpy()
    without_notes = int(np.count_nonzero(retained_per_patient == 0))
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at": _now(),
        "sources": {
            "cohort": _file_record(cohort_path),
            "notes": _file_record(notes.path),
        },
        **_expected_identity(
            notes=notes,
            unit_id_column=unit_id_column,
            text_column=text_column,
        ),
        "date_comparison": (
            "calendar day; timezone-aware values are converted to UTC before the "
            "day is taken"
        ),
        "category_precedence": list(CATEGORY_PRECEDENCE),
        "note_counts": result.note_counts,
        "patients": len(frame),
        "patients_without_retained_notes": without_notes,
        "retained_notes_per_patient": _distribution(retained_per_patient),
        "characters_per_patient": _distribution(characters),
        "analysis_cohort": {
            **_file_record(analysis),
            "rows": len(frame),
            "columns": [str(name) for name in frame.columns],
        },
        "audit": {"path": str(audit)},
    }
    _write_json(manifest_path(output_dir), manifest)
    LOGGER.info(
        "built dated-notes analysis cohort %s: patients=%s notes=%s",
        analysis,
        len(frame),
        result.note_counts,
    )
    if without_notes:
        LOGGER.warning(
            "%s of %s cohort patients have no retained pretreatment notes; their "
            "analysis text is empty (see %s)",
            without_notes,
            len(frame),
            audit,
        )


def prepare_analysis_cohort(
    *,
    cohort_path: Path,
    notes: PretreatmentNotesConfig,
    output_dir: Path,
    unit_id_column: str,
    text_column: str,
    materialize: bool,
) -> pd.DataFrame:
    """Return the derived analysis cohort, building it at most once per directory.

    With an existing manifest, the cohort and notes files are hashed again and,
    together with the configuration and the derived file's own hash, compared
    with the manifest; the derived file is then read without being rewritten.
    Without a manifest, ``materialize`` builds the derived file (the first
    Stage 1 start); otherwise the call fails, so Stage 2-only, reselection, and
    preflight never write input files.
    """

    output_dir = Path(output_dir)
    check_output_directory(output_dir, notes)
    manifest_file = manifest_path(output_dir)
    if manifest_file.is_file():
        _verify_manifest(
            manifest_file,
            cohort_path=Path(cohort_path),
            notes=notes,
            output_dir=output_dir,
            unit_id_column=unit_id_column,
            text_column=text_column,
        )
    elif materialize:
        _materialize(
            cohort_path=Path(cohort_path),
            notes=notes,
            output_dir=output_dir,
            unit_id_column=unit_id_column,
            text_column=text_column,
        )
    else:
        raise FileNotFoundError(
            "notes are configured, but the dated-notes analysis cohort has not been "
            f"built: {manifest_file} is missing. Stage 1 builds it; run Stage 1 in "
            "this output directory first"
        )
    return pd.read_parquet(analysis_cohort_path(output_dir))


__all__ = [
    "ANALYSIS_COHORT_FILE",
    "AUDIT_FILE",
    "CATEGORY_PRECEDENCE",
    "MANIFEST_FILE",
    "MANIFEST_SCHEMA_VERSION",
    "NOTE_CATEGORIES",
    "NOTE_HEADERS",
    "PATIENT_AUDIT_CATEGORIES",
    "PretreatmentNotesConfig",
    "PretreatmentText",
    "analysis_cohort_path",
    "audit_path",
    "build_pretreatment_text",
    "calendar_days",
    "check_output_directory",
    "file_sha256",
    "manifest_path",
    "note_header",
    "notes_config_from_mapping",
    "prepare_analysis_cohort",
    "read_notes_table",
    "read_table",
]
