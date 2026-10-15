"""Tabular dataset loading and validation for OCI workflows."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a CSV or Parquet cohort."""

    source = Path(path)
    if source.suffix == ".parquet":
        frame = pd.read_parquet(source)
    elif source.suffix == ".csv":
        frame = pd.read_csv(source)
    else:
        raise ValueError(f"Unsupported dataset format: {source.suffix or '<none>'}")
    logger.info("Loaded %d rows from %s", len(frame), source)
    return frame


def validate_dataset(
    frame: pd.DataFrame,
    text_column: str,
    outcome_column: str,
    treatment_column: str,
    split_column: Optional[str] = None,
) -> None:
    """Validate the columns and non-null values required by causal workflows."""

    required = {text_column, outcome_column, treatment_column}
    if split_column:
        required.add(split_column)
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    null_columns = [column for column in required if frame[column].isna().any()]
    if null_columns:
        raise ValueError(f"Null values in required columns: {sorted(null_columns)}")


__all__ = ["load_dataset", "validate_dataset"]
