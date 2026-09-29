#!/usr/bin/env python3
"""Reproduce the complete Gao/Zhao/OCI table reported in research note 030.

The runner generates every stochastic input from fixed seeds.  Its only data
input is the tracked five-confounder/five-modifier NSCLC parquet plus metadata.
No clinical notes, LLM calls, or prior run directories are required.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.model_selection import StratifiedKFold
import sklearn

from oci_ridge import ridge_interaction_gain


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
NSCLC_DATA = (
    REPO_ROOT
    / "synthetic_data"
    / "example_synthetic_datasets"
    / "five_confounders_five_effect_modifiers_nsclc_with_structured"
)
SEEDS = range(1, 11)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def balanced_groups() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "coordinate": np.arange(110),
            "coordinate_name": [f"x{j:03d}" for j in range(110)],
            "candidate_index": np.arange(110),
            "candidate_name": (
                [f"C{j}" for j in range(1, 6)]
                + [f"M{j}" for j in range(1, 6)]
                + [f"N{j:03d}" for j in range(1, 101)]
            ),
            "display_name": (
                [f"C{j}" for j in range(1, 6)]
                + [f"M{j}" for j in range(1, 6)]
                + [f"N{j:03d}" for j in range(1, 101)]
            ),
            "role": ["confounder"] * 5 + ["modifier"] * 5 + ["noise"] * 100,
        }
    )


def encode_saved_features(frame: pd.DataFrame, metadata: dict) -> tuple[np.ndarray, pd.DataFrame]:
    """Use the same reference-category encoding as the original Note 029 run."""

    blocks: list[np.ndarray] = []
    records: list[dict] = []
    candidate_index = 0
    coordinate = 0
    role_number = {"confounder": 0, "modifier": 0}
    ordered = [("confounder", feature) for feature in metadata["confounders"]]
    ordered += [("modifier", feature) for feature in metadata["effect_modifiers"]]
    for role, feature in ordered:
        role_number[role] += 1
        name = feature["name"]
        values = frame[f"true_{name}"].to_numpy()
        if feature["type"] == "continuous":
            feature_blocks = [(name, np.asarray(values, dtype=float)[:, None])]
        else:
            observed = list(pd.unique(values))
            declared = feature.get("categories", [])
            reference = declared[0] if declared and declared[0] in observed else observed[0]
            categories = [item for item in declared if item in observed and item != reference]
            categories += [item for item in observed if item != reference and item not in categories]
            feature_blocks = [
                (f"{name}={category}", (np.asarray(values, dtype=object) == category).astype(float)[:, None])
                for category in categories
            ]
        for coordinate_name, block in feature_blocks:
            if np.std(block[:, 0]) <= 0:
                continue
            blocks.append(block)
            records.append(
                {
                    "coordinate": coordinate,
                    "coordinate_name": coordinate_name,
                    "candidate_index": candidate_index,
                    "candidate_name": name,
                    "display_name": ("C" if role == "confounder" else "M")
                    + f"{role_number[role]}:{name}",
                    "role": role,
                }
            )
            coordinate += 1
        candidate_index += 1
    return np.column_stack(blocks), pd.DataFrame(records)


def add_noise_groups(
    x: np.ndarray, group_map: pd.DataFrame, noise: np.ndarray
) -> tuple[np.ndarray, pd.DataFrame]:
    records = group_map.to_dict("records")
    first_coordinate = x.shape[1]
    first_candidate = int(group_map.candidate_index.max()) + 1
    for j in range(noise.shape[1]):
        records.append(
            {
                "coordinate": first_coordinate + j,
                "coordinate_name": f"noise_{j + 1:03d}",
                "candidate_index": first_candidate + j,
                "candidate_name": f"N{j + 1:03d}",
                "display_name": f"N{j + 1:03d}",
                "role": "noise",
            }
        )
    return np.column_stack([x, noise]), pd.DataFrame(records)


def add_outcome_columns(
    x: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    e: np.ndarray,
    mu0: np.ndarray,
    mu1: np.ndarray,
    folds: np.ndarray,
) -> pd.DataFrame:
    table = pd.DataFrame(x, columns=[f"x{j:03d}" for j in range(x.shape[1])])
    columns = {
        "row_id": np.arange(len(x)),
        "A": treatment,
        "Y": outcome,
        "e": e,
        "mu0": mu0,
        "mu1": mu1,
        "truth_rd": mu1 - mu0,
        "truth_logor": logit(mu1) - logit(mu0),
        "inner_fold": folds,
    }
    for name, values in columns.items():
        table[name] = values
    return table


def generate_balanced_base(output: Path) -> list[dict]:
    """Regenerate the exact Note 025 q=5/logOR training tables."""

    specs: list[dict] = []
    groups = balanced_groups()
    for replicate in SEEDS:
        seed = 310000 + replicate
        rng = np.random.default_rng(seed + 50_000)
        x = rng.choice([-1, 1], size=(4000, 110))
        confounders, modifiers = x[:, :5], x[:, 5:10]
        e = expit(0.8 * confounders.sum(axis=1) / np.sqrt(5))
        treatment = (rng.random(4000) < e).astype(int)
        uniform = rng.random(4000)
        mu0 = expit(-0.4 + 0.5 * confounders.sum(axis=1) / np.sqrt(5))
        truth_logor = 0.3 + modifiers @ (0.6 * np.array([1, -1, 1, -1, 1]))
        mu1 = expit(logit(mu0) + truth_logor)
        outcome = (uniform < np.where(treatment == 1, mu1, mu0)).astype(int)
        folds = np.full(4000, -1, dtype=int)
        splitter = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
        for fold, (_, validation) in enumerate(splitter.split(x[:3200], treatment[:3200])):
            folds[validation] = fold
        full = add_outcome_columns(x, treatment, outcome, e, mu0, mu1, folds)
        full["uniform_y"] = uniform
        gaussian = np.random.default_rng(seed + 400_000).standard_normal((3200, 500))
        for n in (800, 3200):
            stem = f"balanced__n{n}__r{replicate:02d}"
            full.iloc[:n].to_csv(output / "inputs" / f"{stem}.csv", index=False)
            groups.to_csv(output / "groups" / f"{stem}.csv", index=False)
            with gzip.open(output / "gaussian" / f"{stem}.csv.gz", "wt", newline="") as stream:
                pd.DataFrame(gaussian[:n]).to_csv(stream, index=False, header=False)
            specs.append(
                {
                    "stem": stem,
                    "mechanism": "balanced",
                    "n": n,
                    "replicate": replicate,
                    "source_seed": seed,
                }
            )
    return specs


def generate_nsclc_base(output: Path) -> list[dict]:
    """Regenerate the exact Note 029 five-plus-five n=800 tables."""

    frame = pd.read_parquet(NSCLC_DATA / "dataset.parquet")
    metadata = json.loads((NSCLC_DATA / "metadata.json").read_text())
    base_x, base_map = encode_saved_features(frame, metadata)
    e = frame.true_treatment_prob.to_numpy(float)
    mu0 = frame.true_y0_prob.to_numpy(float)
    mu1 = frame.true_y1_prob.to_numpy(float)
    specs: list[dict] = []
    for replicate in SEEDS:
        seed = 620000 + replicate
        rng = np.random.default_rng(seed + 100_000)
        treatment = (rng.random(len(frame)) < e).astype(int)
        outcome = (rng.random(len(frame)) < np.where(treatment == 1, mu1, mu0)).astype(int)
        permutation = rng.permutation(len(frame))
        noise = rng.normal(size=(len(frame), 100))
        x, groups = add_noise_groups(base_x, base_map, noise)
        x = x[permutation]
        treatment = treatment[permutation]
        folds = np.full(len(frame), -1, dtype=int)
        splitter = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
        for fold, (_, validation) in enumerate(splitter.split(np.arange(800), treatment[:800])):
            folds[validation] = fold
        full = add_outcome_columns(
            x,
            treatment,
            outcome[permutation],
            e[permutation],
            mu0[permutation],
            mu1[permutation],
            folds,
        )
        source_stem = f"nsclc_source__r{replicate:02d}"
        full.to_csv(output / "sources" / f"{source_stem}.csv", index=False)
        groups.to_csv(output / "sources" / f"{source_stem}__groups.csv", index=False)
        stem = f"nsclc__n800__r{replicate:02d}"
        full.iloc[:800].to_csv(output / "inputs" / f"{stem}.csv", index=False)
        groups.to_csv(output / "groups" / f"{stem}.csv", index=False)
        gaussian = rng.normal(size=(800, 500))
        with gzip.open(output / "gaussian" / f"{stem}.csv.gz", "wt", newline="") as stream:
            pd.DataFrame(gaussian).to_csv(stream, index=False, header=False)
        specs.append(
            {
                "stem": stem,
                "mechanism": "nsclc",
                "n": 800,
                "replicate": replicate,
                "source_seed": seed,
            }
        )
    return specs


def standardize_group(
    x: np.ndarray, train: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    center = x[train].mean(axis=0)
    scale = x[train].std(axis=0, ddof=0)
    keep = np.isfinite(scale) & (scale > 1e-12)
    if not np.any(keep):
        raise ValueError("Candidate group has no varying coordinate")
    return (
        (x[train][:, keep] - center[keep]) / scale[keep],
        (x[valid][:, keep] - center[keep]) / scale[keep],
    )


def count_selected(selected: set[int], groups: pd.DataFrame) -> tuple[int, int, str]:
    candidates = groups.drop_duplicates("candidate_index").set_index("candidate_index")
    chosen = candidates.loc[sorted(selected)]
    modifiers = sorted(chosen.loc[chosen.role == "modifier", "candidate_name"].tolist())
    return int((chosen.role == "modifier").sum()), int((chosen.role == "noise").sum()), ";".join(modifiers)


def score_oci(
    data: pd.DataFrame, groups: pd.DataFrame, nuisance: str, table_dir: Path
) -> tuple[set[int], list[dict]]:
    x = data.filter(regex=r"^x\d+$").to_numpy(float)
    selected: set[int] = set()
    score_rows: list[dict] = []
    for fold in (0, 1):
        if nuisance == "oracle":
            e = data.e.to_numpy(float)
            mu0 = data.mu0.to_numpy(float)
            mu1 = data.mu1.to_numpy(float)
        else:
            predictions = pd.read_csv(table_dir / f"estimated_core_fold{fold}" / "predictions.csv")
            e = predictions.e.to_numpy(float)
            mu0 = predictions.mu0.to_numpy(float)
            mu1 = predictions.mu1.to_numpy(float)
        m = (1 - e) * mu0 + e * mu1
        ra = data.A.to_numpy(float) - e
        ry = data.Y.to_numpy(float) - m
        train = np.flatnonzero(data.inner_fold.to_numpy(int) != fold)
        valid = np.flatnonzero(data.inner_fold.to_numpy(int) == fold)
        fold_rows: list[dict] = []
        for candidate_index, part in groups.groupby("candidate_index", sort=True):
            coordinates = part.coordinate.to_numpy(int)
            x_train, x_valid = standardize_group(x[:, coordinates], train, valid)
            score = ridge_interaction_gain(
                x_train,
                x_valid,
                ra[train],
                ra[valid],
                ry[train],
                ry[valid],
            )
            first = part.iloc[0]
            fold_rows.append(
                {
                    "fold": fold,
                    "candidate_index": int(candidate_index),
                    "candidate_name": first.candidate_name,
                    "role": first.role,
                    "score": score,
                }
            )
        order = sorted(
            range(len(fold_rows)),
            key=lambda j: (-fold_rows[j]["score"], fold_rows[j]["candidate_index"]),
        )
        for rank, j in enumerate(order, start=1):
            fold_rows[j]["rank"] = rank
            fold_rows[j]["selected"] = rank <= 10
        selected.update(fold_rows[j]["candidate_index"] for j in order[:10])
        score_rows.extend(fold_rows)
    return selected, score_rows


def run_r(output: Path) -> None:
    command = [
        "Rscript",
        str(HERE / "fit_methods.R"),
        str(output),
        str(HERE / "methods.R"),
    ]
    with (output / "R_execution.log").open("w") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
    if completed.returncode:
        raise RuntimeError(f"R fitting failed; inspect {output / 'R_execution.log'}")


def build_results(output: Path) -> pd.DataFrame:
    rows = pd.read_csv(output / "sparse_replications.csv").to_dict("records")
    scores: list[dict] = []
    prepared = pd.read_csv(output / "prepared_tables.csv")
    for spec in prepared.itertuples(index=False):
        data = pd.read_csv(output / "inputs" / f"{spec.stem}.csv")
        groups = pd.read_csv(output / "groups" / f"{spec.stem}.csv")
        for nuisance in ("oracle", "estimated"):
            selected, fold_scores = score_oci(
                data, groups, nuisance, output / "tables" / spec.stem
            )
            selected_m, selected_n, modifiers = count_selected(selected, groups)
            rows.append(
                {
                    "mechanism": spec.mechanism,
                    "n": int(spec.n),
                    "nuisance": nuisance,
                    "replicate": int(spec.replicate),
                    "source_seed": int(spec.source_seed),
                    "method": "oci",
                    "selected_m": selected_m,
                    "selected_n": selected_n,
                    "modifiers": modifiers,
                }
            )
            scores.extend(
                {
                    "mechanism": spec.mechanism,
                    "n": int(spec.n),
                    "nuisance": nuisance,
                    "replicate": int(spec.replicate),
                    **item,
                }
                for item in fold_scores
            )
    result = pd.DataFrame(rows).sort_values(
        ["mechanism", "n", "nuisance", "replicate", "method"]
    )
    expected_rows = 2 * 3 * 2 * 10 * 3
    if len(result) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows; found {len(result)}")
    result.to_csv(output / "replications.csv", index=False)
    with gzip.open(output / "oci_candidate_scores.csv.gz", "wt", newline="") as stream:
        pd.DataFrame(scores).to_csv(stream, index=False)
    return result


def summarize(result: pd.DataFrame, output: Path) -> pd.DataFrame:
    summary = result.groupby(["mechanism", "n", "nuisance", "method"])[
        ["selected_m", "selected_n"]
    ].agg(["mean", "sem"]).reset_index()
    summary.columns = ["_".join(column).rstrip("_") for column in summary.columns]
    summary.to_csv(output / "summary.csv", index=False)
    table_rows: list[dict] = []
    for (mechanism, n, nuisance), part in result.groupby(["mechanism", "n", "nuisance"]):
        row: dict[str, object] = {"mechanism": mechanism, "n": n, "nuisance": nuisance}
        for method in ("gao_sparse", "zhao", "oci"):
            selected = part[part.method == method]
            row[f"{method}_m"] = float(selected.selected_m.mean())
            row[f"{method}_n"] = float(selected.selected_n.mean())
        table_rows.append(row)
    table = pd.DataFrame(table_rows).sort_values(["mechanism", "n", "nuisance"])
    table.to_csv(output / "full_table.csv", index=False)
    return table


def verify_expected(table: pd.DataFrame, output: Path) -> None:
    expected = pd.read_csv(HERE / "expected_note030.csv").sort_values(
        ["mechanism", "n", "nuisance"]
    )
    columns = [column for column in expected if column not in {"mechanism", "n", "nuisance"}]
    keys_match = table[["mechanism", "n", "nuisance"]].reset_index(drop=True).equals(
        expected[["mechanism", "n", "nuisance"]].reset_index(drop=True)
    )
    max_error = float(
        np.max(
            np.abs(
                table[columns].to_numpy(float)
                - expected[columns].to_numpy(float)
            )
        )
    )
    verification = {
        "verdict": "PASS" if keys_match and max_error < 1e-12 else "FAIL",
        "keys_match": keys_match,
        "maximum_table_error": max_error,
    }
    write_json(output / "verification.json", verification)
    if verification["verdict"] != "PASS":
        raise RuntimeError(f"Result differs from expected Note 030 table: {verification}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for name in ("inputs", "groups", "gaussian", "sources", "tables"):
        (output / name).mkdir()
    started = datetime.now(timezone.utc).isoformat()

    specs = generate_balanced_base(output) + generate_nsclc_base(output)
    pd.DataFrame(specs).to_csv(output / "prepared_base_tables.csv", index=False)
    run_r(output)
    result = build_results(output)
    table = summarize(result, output)
    verify_expected(table, output)
    write_json(
        output / "runtime.json",
        {
            "started_utc": started,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
            "R": json.loads((output / "R_runtime.json").read_text()),
        },
    )
    print(table.to_string(index=False))
    print(f"\nPASS: reproduced Note 030 at {output}")


if __name__ == "__main__":
    main()
