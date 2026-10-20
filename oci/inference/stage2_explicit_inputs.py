"""Persist the cohort identity and honest splits for Stage 2 without discovery."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

INPUT_FILE = "explicit_inputs.json"


def prepare_explicit_inputs(
    *,
    dataset: pd.DataFrame,
    output_dir: Path,
    config: Any,
    clinical_question: str,
    unit_id_column: str,
    text_column: str,
    treatment_column: str,
    outcome_column: str,
    outcome_type: str,
    outer_folds: int,
    inner_folds: int,
    seed: int,
    persist: bool = True,
) -> dict[str, Any]:
    """Create splits once, or validate and reuse them without resampling.

    One atomic file holds both the input identity and the complete split plan.
    Missing provenance after Stage 2 has started is an error, including when a
    caller removes the coarse completion marker to request a rerun.
    """
    from .plain_handoff_stage2_analysis import _frame_fingerprint

    columns = [unit_id_column, text_column, treatment_column, outcome_column]
    missing = sorted(set(columns) - set(dataset.columns))
    if missing:
        raise ValueError(f"explicit-only dataset is missing configured columns: {missing}")
    data = dataset.loc[:, columns].reset_index(drop=True)
    if data[unit_id_column].isna().any() or data[unit_id_column].duplicated().any():
        raise ValueError("explicit-only dataset unit IDs must be nonmissing and unique")
    treatment = pd.to_numeric(data[treatment_column], errors="coerce")
    outcome = pd.to_numeric(data[outcome_column], errors="coerce")
    if treatment.isna().any() or set(treatment.unique()) != {0, 1}:
        raise ValueError("explicit-only treatment must be complete 0/1 with both arms present")
    if outcome_type not in {"binary", "continuous"}:
        raise ValueError("explicit-only outcome_type must be binary or continuous")
    if not np.isfinite(outcome.to_numpy(dtype=float)).all():
        raise ValueError("explicit-only outcome must be complete, finite, and numeric")
    if outcome_type == "binary" and not set(outcome.unique()) <= {0, 1}:
        raise ValueError("explicit-only binary outcome must contain only 0 and 1")
    if outer_folds < 2 or inner_folds < 2:
        raise ValueError("explicit-only requires outer_folds and inner_folds >= 2")
    identity = {
        "schema_version": "stage2_explicit_inputs_v1",
        "dataset_fingerprint": _frame_fingerprint(data),
        "rows": len(data),
        "columns": columns,
        "outcome_type": outcome_type,
        "clinical_question": clinical_question,
        "outer_folds": outer_folds,
        "inner_folds": inner_folds,
        "seed": seed,
        "explicit_features": [feature.as_definition() for feature in config.explicit_features],
    }
    return freeze_nested_splits(
        dataset=data,
        output_dir=output_dir,
        identity=identity,
        treatment_column=treatment_column,
        outcome_column=outcome_column,
        filename=INPUT_FILE,
        mode="explicit-only",
        persist=persist,
    )


def freeze_nested_splits(
    *,
    dataset: pd.DataFrame,
    output_dir: Path,
    identity: dict[str, Any],
    treatment_column: str,
    outcome_column: str,
    filename: str,
    mode: str,
    persist: bool,
) -> dict[str, Any]:
    """Share the split lifecycle between extraction and supplied-table analyses."""
    from .plain_handoff_stage2 import _value_fingerprint

    outer_folds, inner_folds = identity["outer_folds"], identity["inner_folds"]
    outcome_type, seed = identity["outcome_type"], identity["seed"]
    path = Path(output_dir) / filename
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("inputs") != identity:
            raise RuntimeError(
                f"{mode} inputs changed (cohort, columns, question, folds, seed, "
                "or feature definitions); use a fresh output directory"
            )
        if manifest.get("fingerprint") != _value_fingerprint(
            {"inputs": identity, "splits": manifest.get("splits")}
        ):
            raise RuntimeError(f"{mode} split provenance fingerprint mismatch")
    else:
        if Path(output_dir).exists() and any(Path(output_dir).iterdir()):
            raise RuntimeError(
                f"{mode} split provenance is missing from an existing Stage 2 run; "
                f"restore {filename} or use a fresh output directory"
            )
        from .tfidf_topic_stage1 import make_joint_treatment_outcome_splits

        def split(frame: pd.DataFrame, count: int, split_seed: int):
            return make_joint_treatment_outcome_splits(
                frame,
                treatment_column=treatment_column,
                outcome_column=outcome_column,
                outcome_type=outcome_type,
                n_splits=count,
                seed=split_seed,
            )

        outer, outer_metadata = split(dataset, outer_folds, seed)
        splits = []
        for fold, (fit, heldout) in enumerate(outer, 1):
            inner, inner_metadata = split(dataset.iloc[fit], inner_folds, seed + 51_000 + fold)
            splits.append(
                {
                    "outer_fold": fold,
                    "fit_row_ids": fit.tolist(),
                    "heldout_row_ids": heldout.tolist(),
                    "split_metadata": outer_metadata,
                    "inner_splits": [
                        {
                            "inner_fold": index,
                            "fit_row_ids": fit[inner_fit].tolist(),
                            "heldout_row_ids": fit[inner_heldout].tolist(),
                        }
                        for index, (inner_fit, inner_heldout) in enumerate(inner, 1)
                    ],
                    "inner_split_metadata": inner_metadata,
                }
            )
        manifest = {"inputs": identity, "splits": splits}
        manifest["fingerprint"] = _value_fingerprint(manifest)

    _validate_splits(manifest["splits"], len(dataset), outer_folds, inner_folds)
    if persist and not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        os.replace(temporary, path)
    return manifest


def _validate_splits(splits: list[dict[str, Any]], rows: int, outer: int, inner: int) -> None:
    def partition(folds, universe, count, key):
        if len(folds) != count or [fold[key] for fold in folds] != list(range(1, count + 1)):
            raise RuntimeError("split provenance has incorrect fold IDs")
        all_heldout = []
        for fold in folds:
            fit, heldout = fold["fit_row_ids"], fold["heldout_row_ids"]
            if (
                not fit
                or not heldout
                or len(fit) != len(set(fit))
                or len(heldout) != len(set(heldout))
                or set(fit) & set(heldout)
                or set(fit) | set(heldout) != universe
            ):
                raise RuntimeError("split provenance does not partition parent rows")
            all_heldout.extend(heldout)
        if sorted(all_heldout) != sorted(universe):
            raise RuntimeError(
                "split provenance held-out rows must cover their parent exactly once"
            )

    partition(splits, set(range(rows)), outer, "outer_fold")
    for fold in splits:
        partition(fold["inner_splits"], set(fold["fit_row_ids"]), inner, "inner_fold")
