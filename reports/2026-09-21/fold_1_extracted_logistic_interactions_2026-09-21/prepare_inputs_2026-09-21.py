"""Freeze all 352 extracted fold-1 candidates and observed T/Y, without oracle data."""
from pathlib import Path
import hashlib
import importlib.util
import inspect
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PRIOR = HERE.parent / "fold_1_all_candidates_all_split_features_2026-09-21"
spec = importlib.util.spec_from_file_location("candidate_reference", PRIOR / "run_exploratory_comparison.py")
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
c = reference.comparison
read, write, sha = c.read_json, c.write_json, c.sha256
FOLD, SOURCE = reference.FOLD, reference.SOURCE
import numpy as np
import pandas as pd
from oci.inference.plain_handoff_stage2_analysis import (
    _FeatureEncoder, _apply_harmonization_plans, _feature_extraction_fingerprint,
)

DATE = "2026-09-21"


def main():
    if (HERE / f"inputs_frozen_{DATE}.json").exists():
        raise ValueError("Inputs already frozen")
    previous = read(PRIOR / f"input_manifest_{DATE}.json")
    files = {}

    def remember(path):
        key = str(path.resolve())
        actual = sha(path)
        if key in previous["files"]:
            assert actual == previous["files"][key], f"Previous input changed: {path}"
        files[key] = actual
        return path

    inp = read(remember(FOLD / "selection/input.json"))
    definitions = inp["definitions"]
    wanted = read(remember(FOLD / "comparison_admissions.json"))["methods"]["all_candidates"]
    assert len(definitions) == len(wanted) == 352
    assert {d["feature_id"] for d in definitions} == set(wanted)
    split = next(s for s in read(remember(SOURCE / "inputs/splits.json")) if s["outer_fold"] == 1)
    names = [d["name"] for d in definitions]
    remember(FOLD / "comparison_measurements/fit.pkl")
    remember(FOLD / "comparison_measurements/fit.json")
    training = c.align(c.frozen_training_frame(FOLD, inp, definitions), split["fit_row_ids"], names)
    final = read(remember(FOLD / "final_definitions.json"))["features"]
    by_id = {d["feature_id"]: d for d in final}
    reused = [d for d in definitions if d["feature_id"] in by_id and
              _feature_extraction_fingerprint(d) == _feature_extraction_fingerprint(by_id[d["feature_id"]])]
    reused_names = [d["name"] for d in reused]
    additional_names = [name for name in names if name not in reused_names]
    standard = c.align(pd.read_csv(remember(FOLD / "extraction/heldout/extracted.csv")),
                       split["heldout_row_ids"])[["_oci_row_id", *reused_names]]
    records = []
    for position, row_id in enumerate(split["heldout_row_ids"], 1):
        directory = FOLD / "comparison_measurements/heldout_additional/batches" / f"batch_{position:05d}"
        assert read(remember(directory / "complete.json"))["status"] == "complete"
        rows = read(remember(directory / "result.json"))["rows"]
        assert len(rows) == 1 and rows[0]["row_id"] == row_id
        assert set(rows[0]["values"]) == set(additional_names)
        records.append({"_oci_row_id": row_id, **rows[0]["values"]})
    additional = pd.DataFrame(records, columns=["_oci_row_id", *additional_names])
    raw = c.align(standard.merge(additional, on="_oci_row_id", validate="one_to_one"), split["heldout_row_ids"], names)
    heldout, _ = _apply_harmonization_plans(raw, definitions, scope="outer_heldout")
    statistical = read(remember(FOLD / "selection/statistical_evidence.json"))
    train_labels = c.align(pd.DataFrame(statistical["cross_fitted_nuisance_models"]["predictions"]), split["fit_row_ids"])
    test_labels = pd.read_csv(remember(FOLD / "comparison_nuisances/heldout.csv"), float_precision="round_trip")
    assert test_labels._oci_row_id.tolist() == split["heldout_row_ids"]
    assert len(training) == 800 and len(heldout) == 200
    assert set(train_labels._oci_row_id).isdisjoint(test_labels._oci_row_id)
    for frame in (train_labels, test_labels):
        assert set(frame.treatment) == {0, 1} and set(frame.outcome) == {0, 1}
    tr = train_labels.effect_eligible.to_numpy(bool)
    te = test_labels.effect_eligible.to_numpy(bool)
    assert tr.sum() == 720 and te.sum() == 180
    encoder = _FeatureEncoder(definitions).fit(training.loc[tr].reset_index(drop=True))
    xtrain = encoder.transform(training.loc[tr].reset_index(drop=True))
    xtest = encoder.transform(heldout.loc[te].reset_index(drop=True))
    assert xtrain.shape == (720, 1096) and xtest.shape == (180, 1096)
    assert hashlib.sha256(xtrain.tobytes()).hexdigest() == previous["train_design"]
    assert hashlib.sha256(xtest.tobytes()).hexdigest() == previous["test_design"]
    output = HERE / "inputs"
    output.mkdir(parents=True, exist_ok=True)
    training.to_pickle(output / "training.pkl")
    heldout.to_pickle(output / "heldout.pkl")
    allowed = ["_oci_row_id", "treatment", "outcome", "propensity", "outcome_prediction", "effect_eligible"]
    train_labels[allowed].to_parquet(output / "training_labels.parquet", index=False)
    test_labels[allowed].to_parquet(output / "heldout_labels.parquet", index=False)
    write(output / "definitions.json", definitions)
    write(output / "split.json", split)
    for path in [Path(__file__), ROOT / "oci/inference/plain_handoff_stage2_analysis.py",
                 ROOT / "oci/inference/feature_harmonization.py", reference.EXPERIMENT / "compare.py",
                 PRIOR / "run_exploratory_comparison.py", PRIOR / f"input_manifest_{DATE}.json"]:
        if path.exists():
            remember(path)
    manifest = {"created_at": c.now(), "purpose": __doc__, "candidate_features": len(definitions),
                "training_rows": 800, "heldout_rows": 200, "eligible_training_rows": 720, "eligible_heldout_rows": 180,
                "previous_eligible_design_reproduced_exactly": True, "previous_encoded_columns": 1096,
                "true_covariates_probabilities_or_effects_read": False,
                "clinical_extraction_refreshed": False, "candidate_selection_applied": False,
                "sources": files, "files": {str(path.resolve()): sha(path) for path in output.iterdir()},
                "versions": previous["versions"]}
    write(HERE / f"inputs_frozen_{DATE}.json", manifest)
    print(json.dumps({key: manifest[key] for key in ["candidate_features", "training_rows", "heldout_rows",
                      "eligible_training_rows", "eligible_heldout_rows", "previous_eligible_design_reproduced_exactly"]}), flush=True)


if __name__ == "__main__":
    main()
