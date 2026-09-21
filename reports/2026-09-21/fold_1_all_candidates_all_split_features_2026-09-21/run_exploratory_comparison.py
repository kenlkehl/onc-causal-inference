"""User-requested fold 1 split-search sensitivity analysis; no extraction or selection.

Run `fit` before `evaluate` with /home/klkehl/thisenv/bin/python.
The fit phase cannot read the oracle dataset. Evaluation verifies frozen predictions.
All outputs stay in this separate dated folder; the ongoing experiment is read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/oci_selection_comparison_matplotlib")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EXPERIMENT = ROOT / "reports/2026-09-19/selection_comparison_2026-09-19"
SOURCE = EXPERIMENT / "results"
FOLD = SOURCE / "refresh/outer_001"
INTERIM = ROOT / "reports/2026-09-21/fold_1_interim_oracle_review_2026-09-21"
SEEDS = (120042, 1120042, 2120042)
METHOD = "all_candidates_all_split_features"

spec = importlib.util.spec_from_file_location("frozen_selection_comparison", EXPERIMENT / "compare.py")
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)
read = comparison.read_json
write = comparison.write_json
sha = comparison.sha256

import numpy as np
import pandas as pd


def verify(files):
    for path, expected in files.items():
        if sha(path) != expected:
            raise ValueError(f"Frozen file changed: {path}")


def existing_freeze():
    frozen = read(INTERIM / "frozen_fold_1_inputs_2026-09-21.json")
    files = {**frozen["files"], **frozen["ancillary_files"]}
    verify(files)
    return files


def fit():
    from econml.grf import CausalForest
    from oci.inference.plain_handoff_stage2_analysis import (
        _FeatureEncoder, _apply_harmonization_plans, _feature_extraction_fingerprint,
    )

    if (HERE / "predictions_frozen_2026-09-21.json").exists():
        raise ValueError("Already frozen; evaluate the saved fits instead of overwriting them")
    files = existing_freeze()
    comparison.validate_run_inputs(SOURCE, SOURCE / "revisions/adjudication_budget_v4/manifest.json")

    def remember(path):
        files[str(path.resolve())] = sha(path)
        return path

    inp = read(remember(FOLD / "selection/input.json"))
    definitions = inp["definitions"]
    split = next(s for s in read(remember(SOURCE / "inputs/splits.json")) if s["outer_fold"] == 1)
    wanted = read(remember(FOLD / "comparison_admissions.json"))["methods"]["all_candidates"]
    subset = [d for d in definitions if d["feature_id"] in set(wanted)]
    assert len(subset) == len(wanted) == 352
    names = [d["name"] for d in definitions]
    training = comparison.align(comparison.frozen_training_frame(FOLD, inp, definitions), split["fit_row_ids"], names)
    remember(FOLD / "comparison_measurements/fit.pkl")

    # Reproduce complete_measurements exactly, including categorical Python types.
    # Reading its final combined CSV does not preserve all of those types.
    final = read(remember(FOLD / "final_definitions.json"))["features"]
    by_id = {d["feature_id"]: d for d in final}
    reused = [d for d in definitions if d["feature_id"] in by_id and
              _feature_extraction_fingerprint(d) == _feature_extraction_fingerprint(by_id[d["feature_id"]])]
    reused_names = [d["name"] for d in reused]
    missing_names = [d["name"] for d in definitions if d["name"] not in reused_names]
    standard = comparison.align(pd.read_csv(remember(FOLD / "extraction/heldout/extracted.csv")),
                                split["heldout_row_ids"])[["_oci_row_id", *reused_names]]
    records = []
    for position, row_id in enumerate(split["heldout_row_ids"], 1):
        directory = FOLD / "comparison_measurements/heldout_additional/batches" / f"batch_{position:05d}"
        assert read(remember(directory / "complete.json"))["status"] == "complete"
        rows = read(remember(directory / "result.json"))["rows"]
        assert len(rows) == 1 and rows[0]["row_id"] == row_id
        assert set(rows[0]["values"]) == set(missing_names)
        records.append({"_oci_row_id": row_id, **{name: rows[0]["values"][name] for name in missing_names}})
    additional = pd.DataFrame(records, columns=["_oci_row_id", *missing_names])
    raw = comparison.align(standard.merge(additional, on="_oci_row_id", validate="one_to_one"),
                           split["heldout_row_ids"], names)
    heldout, _ = _apply_harmonization_plans(raw, definitions, scope="outer_heldout")

    # Training nuisances originally came directly from this exact JSON evidence.
    statistical = read(remember(FOLD / "selection/statistical_evidence.json"))
    train_nuisance = comparison.align(pd.DataFrame(statistical["cross_fitted_nuisance_models"]["predictions"]),
                                      split["fit_row_ids"])
    test_nuisance = pd.read_csv(remember(FOLD / "comparison_nuisances/heldout.csv"), float_precision="round_trip")
    assert test_nuisance._oci_row_id.tolist() == split["heldout_row_ids"]
    assert not set(split["fit_row_ids"]) & set(split["heldout_row_ids"])
    tr = train_nuisance.effect_eligible.to_numpy(dtype=bool)
    te = test_nuisance.effect_eligible.to_numpy(dtype=bool)
    encoder = _FeatureEncoder(subset).fit(training.loc[tr].reset_index(drop=True))
    x_train = encoder.transform(training.loc[tr].reset_index(drop=True))
    x_test = encoder.transform(heldout.loc[te].reset_index(drop=True))
    assert x_train.shape == (720, 1096) and x_test.shape == (180, 1096)
    assert np.isfinite(x_train).all() and np.isfinite(x_test).all()
    t_res = (train_nuisance.treatment - train_nuisance.propensity).to_numpy()[tr]
    y_res = (train_nuisance.outcome - train_nuisance.outcome_prediction).to_numpy()[tr]
    constant = float(np.dot(t_res, y_res) / np.dot(t_res, t_res))
    nuisance_hash = comparison.fingerprint({"train": train_nuisance.to_json(), "test": test_nuisance.to_json()})
    designs = {"train_design": hashlib.sha256(x_train.tobytes()).hexdigest(),
               "test_design": hashlib.sha256(x_test.tobytes()).hexdigest()}

    planned = []
    for seed in SEEDS:
        old = FOLD / "comparison_forests/all_candidates" / f"seed_{seed}"
        params = read(old / "metrics.json")["forest_parameters"]
        guard = comparison.fingerprint({"features": subset, "params": params,
                                        "nuisance_hash": nuisance_hash, **designs})
        assert guard == read(old / "complete.json")["fingerprint"], "Original design fingerprint mismatch"
        new_params = {**params, "max_features": 1.0}
        assert [k for k in params if params[k] != new_params[k]] == ["max_features"]
        planned.append({"seed": seed, "old_parameters": params, "new_parameters": new_params,
                        "all_effective_new_parameters": CausalForest(**new_params).get_params()})
    remember(Path(__file__))
    remember(EXPERIMENT / "compare.py")
    manifest = {"created_at": comparison.now(), "outer_fold": 1, "purpose":
                "User-requested exploratory comparison after inspecting fold 1 results: change only all-candidates forest max_features from sqrt to 1.0.",
                "prior_oracle_review_disclosed": True, "oracle_used_in_fitting": False,
                "ongoing_experiment_modified": False, "clinical_features": len(subset),
                "encoded_columns": x_train.shape[1], "fit_rows": int(tr.sum()), "heldout_rows": int(te.sum()),
                "nuisance_hash": nuisance_hash, **designs, "constant_effect": constant,
                "fits": planned, "files": files,
                "versions": {p: importlib.metadata.version(p) for p in ["econml", "scikit-learn", "numpy", "pandas"]}}
    write(HERE / "input_manifest_2026-09-21.json", manifest)

    verification, frozen = [], {}
    for replicate, plan in enumerate(planned):
        seed = plan["seed"]
        old = FOLD / "comparison_forests/all_candidates" / f"seed_{seed}"
        reference = pd.read_csv(old / "predictions.csv", float_precision="round_trip")
        assert reference._oci_row_id.tolist() == test_nuisance.loc[te, "_oci_row_id"].tolist()
        # A complete replay confirms the copied procedure reproduces saved predictions.
        started = time.monotonic()
        replay = CausalForest(**plan["old_parameters"]).fit(x_train, t_res, y_res)
        replay_values = np.column_stack([v.ravel() for v in replay.predict(x_test, interval=True)])
        reference_values = reference[["estimated_cate", "lower_95", "upper_95"]].to_numpy()
        max_error = float(np.abs(replay_values - reference_values).max())
        np.testing.assert_allclose(replay_values, reference_values, rtol=0, atol=1e-12)
        verification.append({"seed": seed, "original_fingerprint_exact_match": True,
                             "baseline_prediction_interval_max_abs_error": max_error,
                             "replay_seconds": time.monotonic() - started})
        print(json.dumps({"phase": "baseline_reproduced", **verification[-1]}), flush=True)

        started = time.monotonic()
        model = CausalForest(**plan["new_parameters"]).fit(x_train, t_res, y_res)
        cate, lower, upper = model.predict(x_test, interval=True)
        pred = test_nuisance.loc[te].reset_index(drop=True).copy()
        pred["outer_fold"], pred["method"], pred["replicate"] = 1, METHOD, replicate
        pred["estimated_cate"], pred["lower_95"], pred["upper_95"] = cate.ravel(), lower.ravel(), upper.ravel()
        assert np.isfinite(pred[["estimated_cate", "lower_95", "upper_95"]].to_numpy()).all()
        dest = HERE / "fits" / f"seed_{seed}"
        dest.mkdir(parents=True, exist_ok=True)
        pred.to_csv(dest / "predictions.csv", index=False)
        metrics = comparison.loss_metrics(pred.treatment, pred.outcome, pred.propensity, pred.outcome_prediction,
                                          pred.estimated_cate, constant)
        metrics.update(seed=seed, replicate=replicate, method=METHOD, constant_effect=constant,
                       elapsed_seconds=time.monotonic() - started, nuisance_hash=nuisance_hash,
                       parameters=plan["new_parameters"])
        write(dest / "metrics.json", metrics)
        for path in [dest / "predictions.csv", dest / "metrics.json"]:
            frozen[str(path.resolve())] = sha(path)
        print(json.dumps({"phase": "all_features_fit_complete", "seed": seed,
                          "seconds": metrics["elapsed_seconds"], "r_loss": metrics["r_loss"]}), flush=True)
    verify(files)
    write(HERE / "reproduction_validation_2026-09-21.json", {"completed_at": comparison.now(), "results": verification})
    write(HERE / "predictions_frozen_2026-09-21.json", {"frozen_at": comparison.now(),
          "input_manifest_sha256": sha(HERE / "input_manifest_2026-09-21.json"),
          "new_fits": 3, "files": frozen, "oracle_read_by_fit_phase": False})


def evaluate():
    frozen = read(HERE / "predictions_frozen_2026-09-21.json")
    assert frozen["new_fits"] == 3
    assert sha(HERE / "input_manifest_2026-09-21.json") == frozen["input_manifest_sha256"]
    verify(frozen["files"])
    manifest = read(HERE / "input_manifest_2026-09-21.json")
    verify(manifest["files"])
    oracle_path = ROOT / "artifacts/research_all_evidence/five_conf_five_mod_nsclc_full/stage2/posthoc_predictions_with_oracle_ite.csv"
    truth = pd.read_csv(oracle_path, usecols=["_oci_row_id", "true_ite_prob"], float_precision="round_trip")
    assert not truth._oci_row_id.duplicated().any()
    records, frames = [], []
    for method in [*comparison.METHODS, METHOD]:
        for replicate, seed in enumerate(SEEDS):
            directory = HERE / "fits" / f"seed_{seed}" if method == METHOD else FOLD / "comparison_forests" / method / f"seed_{seed}"
            pred = pd.read_csv(directory / "predictions.csv", float_precision="round_trip")
            frame = pred.merge(truth, on="_oci_row_id", how="left", validate="one_to_one")
            assert len(frame) == 180 and frame.true_ite_prob.notna().all()
            error = frame.estimated_cate - frame.true_ite_prob
            row = comparison.loss_metrics(frame.treatment, frame.outcome, frame.propensity, frame.outcome_prediction,
                                          frame.estimated_cate, manifest["constant_effect"])
            row.update(method=method, seed=seed, replicate=replicate,
                       rmse=float(np.sqrt(np.mean(error ** 2))), mae=float(np.abs(error).mean()),
                       bias=float(error.mean()), correlation=float(frame.estimated_cate.corr(frame.true_ite_prob)),
                       oracle_sd=float(frame.true_ite_prob.std(ddof=0)), oracle_mean=float(frame.true_ite_prob.mean()),
                       constant_rmse=float(np.sqrt(np.mean((manifest["constant_effect"] - frame.true_ite_prob) ** 2))),
                       coverage_95=float(((frame.lower_95 <= frame.true_ite_prob) &
                                          (frame.true_ite_prob <= frame.upper_95)).mean()))
            records.append(row)
            frames.append(frame)
    table = pd.DataFrame(records)
    table.to_csv(HERE / "metrics_by_seed_2026-09-21.csv", index=False)
    pd.concat(frames, ignore_index=True).to_csv(HERE / "predictions_with_oracle_2026-09-21.csv", index=False)
    summaries = {}
    metrics = ["rmse", "mae", "correlation", "r_loss", "r_score", "mean_cate", "sd_cate", "bias", "coverage_95"]
    for method, frame in table.groupby("method", sort=False):
        summaries[method] = {m: {"mean": float(frame[m].mean()), "min": float(frame[m].min()),
                                "max": float(frame[m].max())} for m in metrics}
    old = table[table.method == "all_candidates"].set_index("seed")
    new = table[table.method == METHOD].set_index("seed")
    paired = [{"seed": seed, **{m + "_change": float(new.loc[seed, m] - old.loc[seed, m]) for m in metrics}}
              for seed in SEEDS]
    write(HERE / "evaluation_2026-09-21.json", {"evaluated_at": comparison.now(),
          "prediction_freeze_sha256": sha(HERE / "predictions_frozen_2026-09-21.json"),
          "oracle_sha256": sha(oracle_path), "summaries": summaries, "paired_changes_new_minus_old": paired,
          "eligible_rows": 180, "oracle_mean": records[0]["oracle_mean"], "oracle_sd": records[0]["oracle_sd"],
          "constant_rmse": records[0]["constant_rmse"], "constant_r_loss": records[0]["constant_r_loss"]})
    verify(frozen["files"])
    verify(manifest["files"])
    print(table[["method", "seed", "rmse", "correlation", "r_loss", "sd_cate", "coverage_95"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["fit", "evaluate"])
    args = parser.parse_args()
    fit() if args.phase == "fit" else evaluate()
