#!/usr/bin/env python3
"""Run the frozen downstream effect-estimation experiment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
import pandas as pd
from scipy.special import logit
import scipy
import sklearn

from core import (
    counterfactuals_from_log_or,
    counterfactuals_from_rd,
    fit_dina,
    oracle_nuisances,
    prediction_metrics,
)


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DATA_ROOT = REPO_ROOT / "synthetic_data" / "example_synthetic_datasets"
DATASETS = {
    "nsclc_1plus1": "one_confounder_one_effect_modifier_nsclc_with_structured",
    "nsclc_5plus5": "five_confounders_five_effect_modifiers_nsclc_with_structured",
}
DEFAULT_SIZES = (800, 3_200, 12_800)
DEFAULT_REPLICATIONS = tuple(range(1, 11))


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def encode_features(frame: pd.DataFrame, metadata: dict) -> tuple[np.ndarray, pd.DataFrame]:
    """Reference-code every designated clinical variable in metadata order."""

    blocks: list[np.ndarray] = []
    records: list[dict] = []
    coordinate = 0
    ordered = [("confounder", feature) for feature in metadata["confounders"]]
    ordered += [("modifier", feature) for feature in metadata["effect_modifiers"]]
    for candidate_index, (role, feature) in enumerate(ordered):
        name = feature["name"]
        values = frame[f"true_{name}"].to_numpy()
        if feature["type"] == "continuous":
            feature_blocks = [(name, np.asarray(values, dtype=float)[:, None])]
        else:
            observed = list(pd.unique(values))
            declared = feature.get("categories", [])
            reference = declared[0] if declared and declared[0] in observed else observed[0]
            categories = [value for value in declared if value in observed and value != reference]
            categories += [
                value for value in observed if value != reference and value not in categories
            ]
            feature_blocks = [
                (
                    f"{name}={category}",
                    (np.asarray(values, dtype=object) == category).astype(float)[:, None],
                )
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
                    "role": role,
                }
            )
            coordinate += 1
    return np.column_stack(blocks), pd.DataFrame(records)


def add_noise(
    clinical: np.ndarray, groups: pd.DataFrame, noise: np.ndarray
) -> tuple[np.ndarray, pd.DataFrame]:
    records = groups.to_dict("records")
    first_coordinate = clinical.shape[1]
    first_candidate = int(groups.candidate_index.max()) + 1
    for column in range(noise.shape[1]):
        records.append(
            {
                "coordinate": first_coordinate + column,
                "coordinate_name": f"noise_{column + 1:03d}",
                "candidate_index": first_candidate + column,
                "candidate_name": f"N{column + 1:03d}",
                "role": "noise",
            }
        )
    return np.column_stack([clinical, noise]), pd.DataFrame(records)


def coordinate_indices(groups: pd.DataFrame, support: set[str]) -> np.ndarray:
    return np.flatnonzero(groups.candidate_name.isin(support).to_numpy())


def run_gao_selection(
    x: np.ndarray,
    groups: pd.DataFrame,
    treatment: np.ndarray,
    outcome: np.ndarray,
    e: np.ndarray,
    mu0: np.ndarray,
    mu1: np.ndarray,
    seed: int,
    temporary: Path,
) -> dict:
    input_path = temporary / "selection_input.csv"
    groups_path = temporary / "selection_groups.csv"
    result_path = temporary / "selection_result.json"
    table = pd.DataFrame(x, columns=groups.coordinate_name)
    table.insert(0, "row_id", np.arange(len(x)))
    table["A"], table["Y"] = treatment, outcome
    table["e"], table["mu0"], table["mu1"] = e, mu0, mu1
    table["selection_seed"] = seed
    table.to_csv(input_path, index=False)
    groups.to_csv(groups_path, index=False)
    completed = subprocess.run(
        [
            "Rscript",
            str(HERE / "select_gao.R"),
            str(input_path),
            str(groups_path),
            str(result_path),
            str(HERE.parent / "methods.R"),
        ],
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(
            "Gao selection failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    result = json.loads(result_path.read_text())
    result["r_stderr"] = completed.stderr.strip()
    return result


def fit_causal_forest(
    x_train: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    nuisance: dict[str, np.ndarray],
    x_test: np.ndarray,
    seed: int,
) -> np.ndarray:
    residual_treatment = treatment - nuisance["e"]
    residual_outcome = outcome - nuisance["m"]
    if x_train.shape[1] == 0:
        denominator = float(residual_treatment @ residual_treatment)
        effect = float(residual_treatment @ residual_outcome / denominator)
        return np.full(len(x_test), effect)

    from econml.grf import CausalForest

    model = CausalForest(
        n_estimators=200,
        min_samples_leaf=10,
        max_samples=0.45,
        max_features=1.0,
        honest=True,
        inference=False,
        n_jobs=1,
        random_state=seed,
    )
    model.fit(x_train, residual_treatment.reshape(-1, 1), residual_outcome)
    return model.predict(x_test).reshape(-1)


def evaluate_effects(
    truth_mu0: np.ndarray,
    truth_mu1: np.ndarray,
    fitted_rd: np.ndarray,
    fitted_log_or: np.ndarray,
) -> dict[str, float]:
    truth_rd = truth_mu1 - truth_mu0
    truth_log_or = logit(truth_mu1) - logit(truth_mu0)
    result = {}
    for scale, truth, prediction in (
        ("rd", truth_rd, fitted_rd),
        ("log_or", truth_log_or, fitted_log_or),
    ):
        result.update({f"{scale}_{key}": value for key, value in prediction_metrics(truth, prediction).items()})
    return result


def parse_numbers(value: str, *, sizes: bool = False) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item)
    if not result or any(number < 1 for number in result):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    if sizes and tuple(sorted(set(result))) != result:
        raise argparse.ArgumentTypeError("sample sizes must be sorted and unique")
    return result


def summarize(replications: pd.DataFrame) -> pd.DataFrame:
    keys = ["mechanism", "n", "modifier_input", "estimator"]
    metric_names = [column for column in replications if column.startswith(("rd_", "log_or_"))]
    rows = []
    for values, group in replications.groupby(keys, sort=True):
        record = dict(zip(keys, values))
        record["replications"] = len(group)
        for metric in metric_names:
            record[f"{metric}_mean"] = group[metric].mean()
            record[f"{metric}_mcse"] = group[metric].std(ddof=1) / np.sqrt(len(group))
        record["selected_true_mean"] = group.selected_true.mean()
        record["selected_false_mean"] = group.selected_false.mean()
        rows.append(record)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-sizes", default=",".join(map(str, DEFAULT_SIZES)))
    parser.add_argument("--replications", default=",".join(map(str, DEFAULT_REPLICATIONS)))
    args = parser.parse_args()
    sample_sizes = parse_numbers(args.sample_sizes, sizes=True)
    replications = parse_numbers(args.replications)
    if args.output.exists():
        parser.error(f"output directory already exists: {args.output}")
    args.output.mkdir(parents=True)
    selection_dir = args.output / "selections"
    selection_dir.mkdir()

    result_rows: list[dict] = []
    selection_rows: list[dict] = []
    prediction_rows: list[pd.DataFrame] = []
    for mechanism_index, (mechanism, directory) in enumerate(DATASETS.items(), 1):
        source = DATA_ROOT / directory
        frame = pd.read_parquet(source / "dataset.parquet")
        metadata = json.loads((source / "metadata.json").read_text())
        clinical, clinical_groups = encode_features(frame, metadata)
        true_support = {feature["name"] for feature in metadata["effect_modifiers"]}
        true_e = frame.true_treatment_prob.to_numpy(float)
        true_mu0 = frame.true_y0_prob.to_numpy(float)
        true_mu1 = frame.true_y1_prob.to_numpy(float)

        for replicate in replications:
            seed = 910_000 + 10_000 * mechanism_index + replicate
            rng = np.random.default_rng(seed)
            permutation = rng.permutation(len(frame))
            test_index, training_pool = permutation[:200], permutation[200:]
            sampled_index = rng.choice(training_pool, size=max(sample_sizes), replace=True)
            training_noise = rng.normal(size=(max(sample_sizes), 100))
            test_noise = rng.normal(size=(len(test_index), 100))
            treatment_uniform = rng.random(max(sample_sizes))
            outcome_uniform = rng.random(max(sample_sizes))
            test_x_all, groups = add_noise(clinical[test_index], clinical_groups, test_noise)
            test_nuisance = oracle_nuisances(
                true_e[test_index], true_mu0[test_index], true_mu1[test_index]
            )

            for n in sample_sizes:
                index = sampled_index[:n]
                train_x_all, _ = add_noise(clinical[index], clinical_groups, training_noise[:n])
                e, mu0, mu1 = true_e[index], true_mu0[index], true_mu1[index]
                treatment = (treatment_uniform[:n] < e).astype(int)
                observed_probability = np.where(treatment == 1, mu1, mu0)
                outcome = (outcome_uniform[:n] < observed_probability).astype(int)
                nuisance = oracle_nuisances(e, mu0, mu1)

                with tempfile.TemporaryDirectory(prefix="oci-downstream-selection-") as temp:
                    selection = run_gao_selection(
                        train_x_all,
                        groups,
                        treatment,
                        outcome,
                        e,
                        mu0,
                        mu1,
                        seed + n,
                        Path(temp),
                    )
                selected_support = set(selection["selected_groups"] or [])
                selection_record = {
                    "mechanism": mechanism,
                    "n": n,
                    "replicate": replicate,
                    "selected_groups": sorted(selected_support),
                    "selected_coordinates": selection["selected_coordinates"],
                    "lambda": selection["lambda"],
                    "kkt": selection["kkt"],
                    "r_warnings": selection["r_stderr"],
                }
                selection_rows.append(selection_record)
                write_json(
                    selection_dir / f"{mechanism}__n{n}__r{replicate:02d}.json",
                    selection_record,
                )

                for modifier_input, support in (
                    ("true", true_support),
                    ("gao_selected", selected_support),
                ):
                    columns = coordinate_indices(groups, support)
                    train_x, test_x = train_x_all[:, columns], test_x_all[:, columns]
                    support_roles = groups.loc[groups.candidate_name.isin(support)].drop_duplicates("candidate_name")
                    selected_true = int(support_roles.role.eq("modifier").sum())
                    selected_false = int((~support_roles.role.eq("modifier")).sum())

                    forest_rd = fit_causal_forest(
                        train_x, treatment, outcome, nuisance, test_x, seed + n + 100
                    )
                    forest_mu0, forest_mu1 = counterfactuals_from_rd(
                        forest_rd, test_nuisance["e"], test_nuisance["m"]
                    )
                    forest_log_or = logit(forest_mu1) - logit(forest_mu0)
                    dina = fit_dina(train_x, treatment, outcome, nuisance["a"], nuisance["nu"])
                    dina_log_or = dina.predict_log_or(test_x)
                    dina_mu0, dina_mu1 = counterfactuals_from_log_or(
                        dina_log_or, test_nuisance["a"], test_nuisance["nu"]
                    )
                    dina_rd = dina_mu1 - dina_mu0

                    for estimator, fitted_rd, fitted_log_or, fitted_mu0, fitted_mu1, diagnostic in (
                        ("causal_forest", forest_rd, forest_log_or, forest_mu0, forest_mu1, {}),
                        (
                            "gao_counterfactual",
                            dina_rd,
                            dina_log_or,
                            dina_mu0,
                            dina_mu1,
                            {
                                "optimizer_converged": dina.converged,
                                "optimizer_gradient_max": dina.gradient_max,
                            },
                        ),
                    ):
                        key = {
                            "mechanism": mechanism,
                            "n": n,
                            "replicate": replicate,
                            "modifier_input": modifier_input,
                            "estimator": estimator,
                            "support_groups": ";".join(sorted(support)),
                            "support_group_count": len(support),
                            "support_coordinate_count": len(columns),
                            "selected_true": selected_true,
                            "selected_false": selected_false,
                            **diagnostic,
                        }
                        key.update(
                            evaluate_effects(
                                true_mu0[test_index],
                                true_mu1[test_index],
                                fitted_rd,
                                fitted_log_or,
                            )
                        )
                        result_rows.append(key)
                        prediction_rows.append(
                            pd.DataFrame(
                                {
                                    **{name: value for name, value in key.items() if not name.startswith(("rd_", "log_or_"))},
                                    "test_source_row": test_index,
                                    "truth_mu0": true_mu0[test_index],
                                    "truth_mu1": true_mu1[test_index],
                                    "fitted_rd": fitted_rd,
                                    "fitted_log_or": fitted_log_or,
                                    "fitted_mu0": fitted_mu0,
                                    "fitted_mu1": fitted_mu1,
                                }
                            )
                        )
                print(f"completed {mechanism} n={n} replicate={replicate}", flush=True)

    replication_table = pd.DataFrame(result_rows)
    replication_table.to_csv(args.output / "replications.csv", index=False)
    summarize(replication_table).to_csv(args.output / "summary.csv", index=False)
    pd.concat(prediction_rows, ignore_index=True).to_parquet(
        args.output / "heldout_predictions.parquet", index=False
    )
    (args.output / "selection_manifest.jsonl").write_text(
        "".join(json.dumps(row, allow_nan=False) + "\n" for row in selection_rows)
    )
    shutil.copy2(HERE / "PROTOCOL.md", args.output / "PROTOCOL.md")
    write_json(
        args.output / "runtime.json",
        {
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "econml": __import__("econml").__version__,
            "r": subprocess.run(
                ["Rscript", "-e", "cat(R.version.string)"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout,
            "glmnet": subprocess.run(
                ["Rscript", "-e", "cat(as.character(packageVersion('glmnet')))"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout,
            "sample_sizes": sample_sizes,
            "replications": replications,
        },
    )
    print(f"PASS: completed downstream experiment at {args.output}")


if __name__ == "__main__":
    main()
