"""Extend the unchanged scalar generator to one million structured patients.

Preserve the previous 100K cohort and its split exactly, then append 900K
patients. No LLM calls, equation recalibration, or category normalization.
"""
from pathlib import Path
import importlib.util
import random
import tempfile

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "oracle_power_100k_2026-09-21"
spec = importlib.util.spec_from_file_location("previous_generator", PREVIOUS / "generate_oracle_100k_2026-09-21.py")
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import KFold

DATE = "2026-09-21"
N = 1_000_000
SEED = 20260921


def progress(phase, **details):
    value = {"updated_at": g.now(), "phase": phase, **details}
    g.write(HERE / "generation_status.json", value)
    print(g.json.dumps(value), flush=True)


def main():
    manifest_path = HERE / f"generation_manifest_{DATE}.json"
    if manifest_path.exists():
        raise ValueError("Generation already frozen; refusing to overwrite")
    previous = g.read(PREVIOUS / f"generation_manifest_{DATE}.json")
    for path, value in {**previous["sources"], **previous["files"]}.items():
        assert g.sha(path) == value, path
    features, treatment, outcome, stats, source_config = g._load_dgp_metadata(str(g.SOURCE / "metadata.json"))
    safe_keys = ["outcome_type", "outcome_noise_std", "enforce_positivity",
                 "min_treatment_rate_per_stratum", "max_treatment_rate_per_stratum"]
    safe = {key: source_config[key] for key in safe_keys}
    assert safe["outcome_type"] == "binary" and safe["enforce_positivity"] is False
    config = g.SyntheticDataConfig(dataset_size=N, seed=SEED, **safe)
    directory = Path(tempfile.mkdtemp(prefix="oci_oracle_power_1m_2026-09-21_", dir="/tmp"))
    dataset = directory / "structured_oracle_1m.parquet"
    random.seed(SEED)
    np.random.seed(SEED)
    records, writer = [], None
    progress("generating_structured_rows", completed=0, total=N, temporary_directory=str(directory))
    try:
        for index in range(N):
            row = g._build_patient_scaffold_record(index, config, features, stats, treatment, outcome)
            row.pop("patient_prompt")
            records.append(row)
            if (index + 1) % 10_000 == 0:
                batch = pa.Table.from_pandas(pd.DataFrame(records), preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(dataset, batch.schema)
                writer.write_table(batch)
                records.clear()
                if (index + 1) % 100_000 == 0:
                    progress("generating_structured_rows", completed=index + 1, total=N,
                             temporary_directory=str(directory))
    finally:
        if writer is not None:
            writer.close()
    assert not records
    data = pd.read_parquet(dataset)
    assert len(data) == N and np.array_equal(data.patient_id, np.arange(N))
    assert not data.isna().any().any()
    previous_data = pd.read_parquet(previous["temporary_dataset"])
    pd.testing.assert_frame_equal(data.iloc[:100_000].reset_index(drop=True), previous_data, check_exact=True)
    expected = g.vectorized_probabilities(data, features, stats, treatment, outcome)
    errors = {}
    for name, values in expected.items():
        errors[name] = float(np.max(np.abs(values - data[name].to_numpy())))
        np.testing.assert_allclose(values, data[name], rtol=0, atol=1e-14)
    old_split = g.read(PREVIOUS / f"splits_{DATE}.json")
    added_train, added_test = next(KFold(n_splits=5, shuffle=True, random_state=42).split(np.arange(900_000)))
    train = np.sort(np.concatenate([old_split["training_rows"]["80000"], added_train + 100_000]))
    test = np.sort(np.concatenate([old_split["heldout_rows"], added_test + 100_000]))
    common_test = np.asarray(old_split["heldout_rows"])
    assert len(train) == 800_000 and len(test) == 200_000
    assert np.intersect1d(train, test).size == 0
    assert np.array_equal(np.sort(np.concatenate([train, test])), np.arange(N))
    assert np.array_equal(test[:20_000], common_test)
    arrays = {"train": train, "test": test, "common_test": common_test}
    for index, (_, heldout) in enumerate(KFold(n_splits=5, shuffle=True, random_state=51043).split(train)):
        arrays[f"inner_heldout_{index}"] = heldout
    split_path = HERE / f"splits_{DATE}.npz"
    np.savez_compressed(split_path, **arrays)
    split_summary = HERE / f"split_design_{DATE}.json"
    g.write(split_summary, {"train_rows": len(train), "test_rows": len(test), "common_previous_test_rows": 20_000,
            "previous_training_and_test_membership_preserved": True, "additional_rows": 900_000,
            "additional_rows_split": "First of five shuffled KFold splits, seed 42",
            "inner_split": "Five shuffled KFold splits, seed 51043", "split_path": str(split_path)})
    dgp_path = HERE / f"dgp_specification_{DATE}.json"
    g.write(dgp_path, {"features": features, "treatment_equation": treatment, "outcome_equation": outcome,
                      "summary_statistics": stats, "sampling_options": safe})
    fingerprint = pd.util.hash_pandas_object(data[["true_" + f["name"] for f in features]], index=False)
    original = pd.read_parquet(g.SOURCE / "dataset.parquet", columns=["true_" + f["name"] for f in features])
    old_fingerprint = pd.util.hash_pandas_object(original, index=False)
    assert not np.isin(fingerprint, old_fingerprint).any()
    validation_path = HERE / f"generation_validation_{DATE}.json"
    g.write(validation_path, {"generated_rows": N, "unique_patient_ids": int(data.patient_id.nunique()),
            "first_100000_rows_identical_to_previous_cohort": True,
            "all_probability_and_effect_columns_independently_recomputed": True,
            "maximum_probability_errors": errors, "no_missing_values": True,
            "no_identical_covariate_rows_with_original_1000_cohort": True,
            "treatment_rate": float(data.treatment_indicator.mean()),
            "outcome_rate": float(data.outcome_indicator.mean()),
            "propensity_below_0_1_fraction": float((data.true_treatment_prob < .1).mean()),
            "propensity_above_0_9_fraction": float((data.true_treatment_prob > .9).mean()),
            "previous_category_label_mismatch_preserved": True})
    sources = [Path(__file__), PREVIOUS / f"generation_manifest_{DATE}.json",
               PREVIOUS / f"generate_oracle_100k_{DATE}.py", PREVIOUS / f"splits_{DATE}.json"]
    g.write(manifest_path, {"created_at": g.now(), "rows": N, "temporary_dataset": str(dataset),
            "temporary_directory": str(directory), "generation_seed": SEED,
            "training_rows": len(train), "heldout_rows": len(test), "common_test_rows": 20_000,
            "new_equations_or_calibration": False, "clinical_text_or_llm_used": False,
            "sources": {**previous["sources"], **{str(p.resolve()): g.sha(p) for p in sources}},
            "previous_generation_files": previous["files"],
            "files": {str(p.resolve()): g.sha(p) for p in [dataset, split_path, split_summary, dgp_path, validation_path]}})
    progress("generation_complete", total=N, temporary_dataset=str(dataset),
             training_rows=len(train), heldout_rows=len(test), common_test_rows=20_000)


if __name__ == "__main__":
    main()
