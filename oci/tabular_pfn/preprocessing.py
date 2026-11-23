"""Context-only tabular preprocessor for real Stage-2 / all-evidence candidate tables.

Converts mixed pandas/PyArrow tables (numeric, binary, categorical, and 3-state EHR
missingness strings like ``not_documented`` / ``documented_unknown``) into dense
float32 matrices ``(X, M)`` strictly fitted on context rows with zero query leakage.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow.csv as pcsv
import pyarrow.parquet as pq

_MISSING_STRINGS = frozenset(
    {
        "",
        "na",
        "n/a",
        "nan",
        "none",
        "null",
        "not_documented",
        "not documented",
        "missing",
    }
)
_UNKNOWN_STRINGS = frozenset(
    {
        "documented_unknown",
        "documented unknown",
        "unknown",
        "unk",
        "indeterminate",
        "equivocal",
    }
)
_TRUE_STRINGS = frozenset({"true", "yes", "present", "positive", "1", "y", "t"})
_FALSE_STRINGS = frozenset({"false", "no", "absent", "negative", "0", "n", "f"})


def read_tabular_file(path: str | Path):
    """Read a Parquet or CSV tabular cohort into a PyArrow table."""
    path = Path(path)
    if path.suffix.lower() in (".csv", ".tsv"):
        return pcsv.read_csv(path)
    return pq.read_table(path)


@dataclass
class _ColumnSpec:
    source_name: str
    kind: str  # "numeric" | "categorical"
    categories: tuple[str, ...] = ()
    add_unknown_flag: bool = False


@dataclass
class TabularPreprocessor:
    """Fit on context rows only; expand mixed/3-state columns into numeric ``(X, M)``."""

    max_categories_per_col: int = 12
    min_category_count: int = 3
    specs: list[_ColumnSpec] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)

    @staticmethod
    def _parse_numeric_value(v):
        if v is None:
            return np.nan, False
        if isinstance(v, bool):
            return (1.0 if v else 0.0), False
        if isinstance(v, (int, float, np.integer, np.floating)):
            f = float(v)
            return (f if np.isfinite(f) else np.nan), False
        s = str(v).strip().lower()
        if s in _MISSING_STRINGS:
            return np.nan, False
        if s in _UNKNOWN_STRINGS:
            return np.nan, True
        if s in _TRUE_STRINGS:
            return 1.0, False
        if s in _FALSE_STRINGS:
            return 0.0, False
        try:
            f = float(s)
            return (f if np.isfinite(f) else np.nan), False
        except ValueError:
            return None, False

    def fit(self, table, feature_cols: list[str]) -> "TabularPreprocessor":
        self.specs = []
        self.feature_names = []
        n_rows = table.num_rows
        for col in feature_cols:
            raw = table[col].to_pylist()
            parsed_num = []
            unknown_count = 0
            all_numeric = True
            for v in raw:
                val, is_unk = self._parse_numeric_value(v)
                if val is None:
                    all_numeric = False
                    break
                parsed_num.append(val)
                if is_unk:
                    unknown_count += 1
            if all_numeric:
                arr = np.asarray(parsed_num, dtype=np.float32)
                finite = arr[np.isfinite(arr)]
                if len(finite) == 0 or float(np.std(finite)) < 1e-8:
                    # Drop strictly constant or 100%-missing columns
                    if unknown_count == 0 or unknown_count == n_rows:
                        continue
                add_unk = 0 < unknown_count < n_rows
                self.specs.append(
                    _ColumnSpec(source_name=col, kind="numeric", add_unknown_flag=add_unk)
                )
                self.feature_names.append(col)
                if add_unk:
                    self.feature_names.append(f"{col}__documented_unknown")
            else:
                counts: dict[str, int] = {}
                unknown_count = 0
                for v in raw:
                    if v is None:
                        continue
                    s = str(v).strip()
                    low = s.lower()
                    if low in _MISSING_STRINGS:
                        continue
                    if low in _UNKNOWN_STRINGS:
                        unknown_count += 1
                        continue
                    counts[s] = counts.get(s, 0) + 1
                kept = [
                    cat
                    for cat, cnt in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
                    if cnt >= self.min_category_count
                ][: self.max_categories_per_col]
                if not kept and not (0 < unknown_count < n_rows):
                    continue
                add_unk = 0 < unknown_count < n_rows
                self.specs.append(
                    _ColumnSpec(
                        source_name=col,
                        kind="categorical",
                        categories=tuple(kept),
                        add_unknown_flag=add_unk,
                    )
                )
                for cat in kept:
                    self.feature_names.append(f"{col}=={cat}")
                if add_unk:
                    self.feature_names.append(f"{col}__documented_unknown")
        if not self.feature_names:
            raise ValueError("No non-constant candidate features survived preprocessing")
        return self

    def transform(self, table) -> tuple[np.ndarray, np.ndarray]:
        if not self.specs:
            raise RuntimeError("TabularPreprocessor must be fit() before transform()")
        n_rows = table.num_rows
        n_out = len(self.feature_names)
        X = np.zeros((n_rows, n_out), dtype=np.float32)
        M = np.zeros((n_rows, n_out), dtype=np.float32)
        col_pos = 0
        for spec in self.specs:
            raw = table[spec.source_name].to_pylist()
            if spec.kind == "numeric":
                unk = np.zeros(n_rows, dtype=np.float32)
                for i, v in enumerate(raw):
                    val, is_unk = self._parse_numeric_value(v)
                    if val is None or not np.isfinite(val):
                        X[i, col_pos] = np.nan
                        M[i, col_pos] = 1.0
                    else:
                        X[i, col_pos] = float(val)
                    if is_unk:
                        unk[i] = 1.0
                col_pos += 1
                if spec.add_unknown_flag:
                    X[:, col_pos] = unk
                    col_pos += 1
            else:
                cat_to_offset = {cat: idx for idx, cat in enumerate(spec.categories)}
                n_cats = len(spec.categories)
                unk = np.zeros(n_rows, dtype=np.float32)
                for i, v in enumerate(raw):
                    if v is None:
                        if n_cats:
                            X[i, col_pos : col_pos + n_cats] = np.nan
                            M[i, col_pos : col_pos + n_cats] = 1.0
                        continue
                    s = str(v).strip()
                    low = s.lower()
                    if low in _MISSING_STRINGS:
                        if n_cats:
                            X[i, col_pos : col_pos + n_cats] = np.nan
                            M[i, col_pos : col_pos + n_cats] = 1.0
                        continue
                    if low in _UNKNOWN_STRINGS:
                        unk[i] = 1.0
                        if n_cats:
                            X[i, col_pos : col_pos + n_cats] = np.nan
                            M[i, col_pos : col_pos + n_cats] = 1.0
                        continue
                    off = cat_to_offset.get(s)
                    if off is not None:
                        X[i, col_pos + off] = 1.0
                col_pos += n_cats
                if spec.add_unknown_flag:
                    X[:, col_pos] = unk
                    col_pos += 1
        return X, M

    def fit_transform(self, table, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
        self.fit(table, feature_cols)
        return self.transform(table)
