"""Generate a temporary structured cohort from the unchanged saved DGP.

Call the repository's patient-scaffold generator directly. Never instantiate
an LLM, rescale/recalibrate equations, normalize categories, or generate notes.
"""
from pathlib import Path
import hashlib
import json
import os
import random
import sys
import tempfile
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.model_selection import KFold
from synthetic_data.config import SyntheticDataConfig
from synthetic_data.generator import _build_patient_scaffold_record, _load_dgp_metadata

SOURCE = ROOT / "synthetic_data/example_synthetic_datasets/five_confounders_five_effect_modifiers_nsclc_with_structured"
N = 100_000
SEED = 20260921
SPLIT_SEED = 42
SUBSET_SEED = 120042


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def progress(phase, **details):
    value = {"updated_at": now(), "phase": phase, **details}
    write(HERE / "generation_status.json", value)
    print(json.dumps(value), flush=True)


def vectorized_probabilities(frame, features, stats, treatment_eq, outcome_eq):
    """Independent vectorized check of the generator's scalar logit routine."""
    values = {}
    for feature in features:
        name = feature["name"]
        column = frame["true_" + name]
        if feature["type"] == "continuous":
            values[name] = (column.to_numpy(float) - stats[name]["mean"]) / (stats[name]["std"] or 1.0)
        else:
            for category in feature["categories"][1:]:
                values[name + "_" + category] = (column == category).to_numpy(float)

    def logit(equation):
        result = np.full(len(frame), equation["intercept"], dtype=float)
        for name, coefficient in equation.get("coefficients", {}).items():
            result += coefficient * values[name]
        for interaction in equation.get("interactions", []):
            result += interaction["coefficient"] * np.prod([values[term] for term in interaction["terms"]], axis=0)
        return result

    e = expit(logit(treatment_eq))
    baseline = logit(outcome_eq)
    delta = np.full(len(frame), outcome_eq["treatment_coefficient"], dtype=float)
    for interaction in outcome_eq["treatment_interactions"]:
        delta += interaction["coefficient"] * values[interaction["term"]]
    mu0, mu1 = expit(baseline), expit(baseline + delta)
    return {"true_treatment_prob": e, "true_y0_prob": mu0, "true_y1_prob": mu1,
            "true_ite_prob": mu1 - mu0,
            "true_outcome_prob": np.where(frame.treatment_indicator.to_numpy() == 1, mu1, mu0)}


def main():
    output = HERE / "generation_manifest_2026-09-21.json"
    if output.exists():
        raise ValueError("Generated cohort already frozen; refusing to overwrite")
    features, treatment, outcome, stats, source_config = _load_dgp_metadata(str(SOURCE / "metadata.json"))
    safe_keys = ["outcome_type", "outcome_noise_std", "enforce_positivity",
                 "min_treatment_rate_per_stratum", "max_treatment_rate_per_stratum"]
    safe = {key: source_config[key] for key in safe_keys}
    assert safe["outcome_type"] == "binary" and safe["enforce_positivity"] is False
    assert all(read(SOURCE / "generation_config.json")[key] == value for key, value in safe.items())
    config = SyntheticDataConfig(dataset_size=N, seed=SEED, **safe)
    directory = Path(tempfile.mkdtemp(prefix="oci_oracle_power_100k_2026-09-21_", dir="/tmp"))
    random.seed(SEED)
    np.random.seed(SEED)
    progress("generating_structured_rows", completed=0, total=N, temporary_directory=str(directory))
    records = []
    for index in range(N):
        row = _build_patient_scaffold_record(index, config, features, stats, treatment, outcome)
        row.pop("patient_prompt")
        records.append(row)
        if (index + 1) % 10_000 == 0:
            progress("generating_structured_rows", completed=index + 1, total=N, temporary_directory=str(directory))
    data = pd.DataFrame(records)
    del records
    assert len(data) == N and data.patient_id.tolist() == list(range(N))
    assert not data.isna().any().any()
    assert set(data.treatment_indicator) == {0, 1} and set(data.outcome_indicator) == {0, 1}
    expected = vectorized_probabilities(data, features, stats, treatment, outcome)
    errors = {}
    for name, values in expected.items():
        errors[name] = float(np.max(np.abs(values - data[name].to_numpy())))
        np.testing.assert_allclose(values, data[name], rtol=0, atol=1e-14)
    old = pd.read_parquet(SOURCE / "dataset.parquet", columns=list(data.columns))
    old_errors = {}
    for name, values in vectorized_probabilities(old, features, stats, treatment, outcome).items():
        old_errors[name] = float(np.max(np.abs(values - old[name].to_numpy())))
        np.testing.assert_allclose(values, old[name], rtol=0, atol=1e-14)
    dataset = directory / "structured_oracle_100k.parquet"
    data.to_parquet(dataset, index=False)
    train, test = next(KFold(n_splits=5, shuffle=True, random_state=SPLIT_SEED).split(np.arange(N)))
    permutation = np.random.default_rng(SUBSET_SEED).permutation(train)
    sizes = [800, 8000, 80000]
    training = {str(n): sorted(permutation[:n].tolist()) for n in sizes}
    assert len(test) == 20000 and all(set(ids).isdisjoint(test) for ids in training.values())
    assert set(training['800']) < set(training['8000']) < set(training['80000'])
    splits = {"training_rows": training, "heldout_rows": test.tolist(),
              "inner_splits": {str(n): [{"fit_local": a.tolist(), "heldout_local": b.tolist()}
                  for a, b in KFold(n_splits=5, shuffle=True, random_state=51043).split(training[str(n)])]
                  for n in sizes}}
    split_path = HERE / "splits_2026-09-21.json"
    write(split_path, splits)
    dgp = {"features": features, "treatment_equation": treatment, "outcome_equation": outcome,
           "summary_statistics": stats, "sampling_options": safe}
    dgp_path = HERE / "dgp_specification_2026-09-21.json"
    write(dgp_path, dgp)
    source_paths = [Path(__file__), ROOT / "synthetic_data/generator.py", ROOT / "synthetic_data/config.py",
                    ROOT / "synthetic_data/prompts.py", SOURCE / "metadata.json", SOURCE / "generation_config.json",
                    SOURCE / "dataset.parquet"]
    fingerprint = pd.util.hash_pandas_object(data[["true_" + f["name"] for f in features]], index=False)
    previous_fingerprint = pd.util.hash_pandas_object(old[["true_" + f["name"] for f in features]], index=False)
    assert not set(fingerprint) & set(previous_fingerprint)
    categorical = {}
    for feature in features:
        if feature['type'] != 'categorical':
            continue
        name, categories = feature['name'], feature['categories']
        raw = [stats[name]['proportions'].get(category, 0.0) for category in categories]
        categorical[name] = {category: {"effective_generator_probability": float(prob / sum(raw)),
                                       "observed_fraction": float((data['true_' + name] == category).mean())}
                             for category, prob in zip(categories, raw)}
    validation = {"generated_rows": N, "unique_patient_ids": int(data.patient_id.nunique()),
                  "maximum_probability_errors_new": errors, "maximum_probability_errors_original": old_errors,
                  "no_identical_covariate_rows_with_original_cohort": True,
                  "categorical_sampling_audit": categorical, "no_missing_values": True,
                  "treatment_rate": float(data.treatment_indicator.mean()),
                  "outcome_rate": float(data.outcome_indicator.mean()),
                  "propensity_below_0_1_fraction": float((data.true_treatment_prob < .1).mean()),
                  "propensity_above_0_9_fraction": float((data.true_treatment_prob > .9).mean())}
    validation_path = HERE / "generation_validation_2026-09-21.json"
    write(validation_path, validation)
    manifest = {"created_at": now(), "rows": N, "temporary_dataset": str(dataset),
                "temporary_directory": str(directory), "generation_seed": SEED,
                "outer_split_seed": SPLIT_SEED, "nested_subset_seed": SUBSET_SEED,
                "inner_split_seed": 51043, "training_sizes": sizes, "heldout_rows": len(test),
                "new_equations_or_calibration": False, "clinical_text_or_llm_used": False,
                "source_configuration_logged": "Only explicitly allowlisted DGP sampling options; no service credentials.",
                "sources": {str(path.resolve()): sha(path) for path in source_paths},
                "files": {str(path.resolve()): sha(path) for path in [dataset, split_path, dgp_path, validation_path]}}
    write(output, manifest)
    progress("generation_complete", total=N, temporary_dataset=str(dataset),
             training_sizes=sizes, heldout_rows=20000, dataset_sha256=sha(dataset))


if __name__ == "__main__":
    main()
