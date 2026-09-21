"""User-requested 2x2 inference/honesty diagnostic with true covariates.

Fixed fold 1, modifiers in X, confounders in W, observed binary outcomes,
and the original estimated nuisance functions. Reuse both-on forests;
fit only the remaining three configurations, each with three fixed seeds.
Sampling stays at 45%, square-root feature search, and 200 trees throughout.
"""
from pathlib import Path
import argparse
import importlib.util
import inspect
import json
import os
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/oci_inference_honesty_matplotlib")
import joblib
import numpy as np
import pandas as pd
from econml.grf import CausalForest

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "fold_1_true_oracle_xw_comparison_2026-09-21"
ABLATION = HERE.parent / "fold_1_oracle_estimation_ablation_2026-09-21"
spec = importlib.util.spec_from_file_location("original_oracle_covariate_run", BASE / "run_true_oracle_xw_comparison.py")
u = importlib.util.module_from_spec(spec)
spec.loader.exec_module(u)
CELLS = [(True, True), (True, False), (False, True), (False, False)]


def cell_name(inference, honesty):
    return f"inference_{str(inference).lower()}_honesty_{str(honesty).lower()}"


def describe(values):
    a = np.asarray(values)
    return {"min": float(a.min()), "median": float(np.median(a)), "mean": float(a.mean()), "max": float(a.max())}


def tree_audit(forest):
    counts, leaves, depth, leaf_n = [], [], [], []
    structural_usage, effect_usage = np.zeros(800, dtype=int), np.zeros(800, dtype=int)
    for tree, subsample in zip(forest.estimators_, forest.get_subsample_inds()):
        splitting, estimation = tree.get_train_test_split_inds()
        assert len(subsample) == 360
        if forest.honest:
            assert len(splitting) == len(estimation) == 180
            assert set(splitting).isdisjoint(estimation)
        else:
            assert len(splitting) == len(estimation) == 360
            np.testing.assert_array_equal(splitting, estimation)
        counts.append((len(subsample), len(splitting), len(estimation)))
        structural_usage[subsample[splitting]] += 1
        effect_usage[subsample[estimation]] += 1
        leaves.append(tree.get_n_leaves())
        depth.append(tree.get_depth())
        leaf_n.extend(tree.tree_.n_node_samples[tree.tree_.children_left == -1].tolist())
    return {"trees": len(forest.estimators_), "sample_split_effect_counts": sorted(set(counts)),
            "leaves_per_tree": describe(leaves), "depth": describe(depth), "effect_patients_per_leaf": describe(leaf_n),
            "distinct_patients_used_for_splits": int(np.sum(structural_usage > 0)),
            "distinct_patients_used_for_effects": int(np.sum(effect_usage > 0)),
            "trees_per_patient_for_splits": describe(structural_usage), "trees_per_patient_for_effects": describe(effect_usage)}


def verify_prior(manifest):
    u.verify(manifest["sources"])
    u.verify(u.read(BASE / "predictions_frozen_2026-09-21.json")["files"])
    ablation_freeze = u.read(ABLATION / "predictions_frozen_2026-09-21.json")
    path = str((ABLATION / "fit_validation_2026-09-21.json").resolve())
    assert u.sha(path) == ablation_freeze["files"][path]


def fit():
    if (HERE / "predictions_frozen_2026-09-21.json").exists():
        raise ValueError("Already frozen; evaluate instead of overwriting")
    bm = u.read(BASE / "input_manifest_2026-09-21.json")
    bf = u.read(BASE / "predictions_frozen_2026-09-21.json")
    u.verify(bm["sources"])
    u.verify(bf["files"])
    assert u.sha(BASE / "input_manifest_2026-09-21.json") == bf["input_manifest_sha256"]
    train, test = bm["training_rows"], bm["heldout_rows"]
    names = ["true_" + d["name"] for d in bm["confounders"] + bm["modifiers"]]
    # No oracle treatment or outcome probability/effect columns are read here.
    frame = pd.read_parquet(u.DATA / "dataset.parquet", columns=names)
    C, c_names, _, _ = u.encode(frame, bm["confounders"], train)
    M, m_names, _, _ = u.encode(frame, bm["modifiers"], train)
    assert u.array_hash(M[train]) == bm["scenarios"]["c"]["train_X_sha256"]
    assert u.array_hash(M[test]) == bm["scenarios"]["c"]["test_X_sha256"]
    initial = joblib.load(BASE / "fits/c/initial_model.joblib")
    yres, tres, cached_X, cached_W = initial.residuals_
    np.testing.assert_array_equal(cached_X, M[train])
    np.testing.assert_array_equal(cached_W, C[train])
    residual_hash = u.array_hash(yres, tres)
    prior_replays = u.read(ABLATION / "fit_validation_2026-09-21.json")["baseline_replays"]
    replays = [r for r in prior_replays if r["scenario"] == "c" and r["search"] == "sqrt"]
    assert sorted(r["seed"] for r in replays) == sorted(u.SEEDS)
    assert all(r["max_prediction_error"] == 0.0 for r in replays)
    first = joblib.load(BASE / "fits/c/sqrt" / f"seed_{u.SEEDS[0]}" / "forest.joblib")
    source_paths = [Path(__file__), BASE / "input_manifest_2026-09-21.json", BASE / "predictions_frozen_2026-09-21.json",
                    ABLATION / "fit_validation_2026-09-21.json", ABLATION / "predictions_frozen_2026-09-21.json",
                    Path(inspect.getfile(CausalForest)), Path(inspect.getfile(CausalForest.get_subsample_inds)),
                    Path(inspect.getfile(first.estimators_[0].get_train_test_split_inds))]
    manifest = {"created_at": u.now(), "purpose": __doc__, "training_rows": train, "heldout_rows": test,
                "cells": [{"inference": i, "honesty": h, "reused_baseline": i and h} for i, h in CELLS],
                "seeds": u.SEEDS, "max_samples": 0.45, "max_features": "sqrt", "n_estimators": 200,
                "X": "Five true modifiers, 12 encoded columns", "W": "Five true confounders, 13 encoded columns",
                "x_columns": m_names, "w_columns": c_names,
                "nuisance_policy": "Reuse original scenario c fitted nuisances and cached residuals exactly for every cell and seed.",
                "outcomes": "Observed binary outcomes, not noiseless/oracle outcome probabilities.",
                "oracle_effects_or_probabilities_used_for_fitting": False,
                "residual_sha256": residual_hash, "baseline_replay_evidence": replays,
                "sources": {**bm["sources"], **{str(p.resolve()): u.sha(p) for p in source_paths}},
                "interpretation": "Inference changes grouped half-sample versus independent full-pool subsampling. Honesty changes disjoint half-samples versus reuse of the complete tree subsample. Sampling fraction stays fixed.",
                "exploratory": "Fold 1 oracle effects were already inspected in earlier experiments; no untouched validation claim."}
    verify_prior(manifest)
    u.write(HERE / "input_manifest_2026-09-21.json", manifest)
    artifacts, baseline_records, new_records = {}, [], []
    for seed in u.SEEDS:
        old_dir = BASE / "fits/c/sqrt" / f"seed_{seed}"
        original = joblib.load(old_dir / "forest.joblib")
        original_params = original.get_params()
        assert original_params["inference"] is True and original_params["honest"] is True
        assert original_params["max_samples"] == 0.45 and original_params["max_features"] == "sqrt"
        old_pred = pd.read_csv(old_dir / "predictions.csv", float_precision="round_trip")
        assert old_pred._oci_row_id.tolist() == test
        np.testing.assert_array_equal(original.predict(M[test]).ravel(), old_pred.estimated_cate.to_numpy())
        # Baseline predictions and forest are reused, never refitted here.
        baseline_records.append({"seed": seed, "predictions_path": str((old_dir / "predictions.csv").resolve()),
                                 "forest_path": str((old_dir / "forest.joblib").resolve()), "reloaded_predictions_identical": True,
                                 "tree_audit": tree_audit(original)})
        for inference, honesty in CELLS[1:]:
            parameters = {**original_params, "inference": inference, "honest": honesty}
            assert {k for k in parameters if parameters[k] != original_params[k]} == ({"inference"} if not inference else set()) | ({"honest"} if not honesty else set())
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = CausalForest(**parameters).fit(M[train], tres, yres)
            assert u.array_hash(yres, tres) == residual_hash
            assert model.inference_ == inference and model.honest == honesty
            pred = model.predict(M[test]).ravel()
            assert np.isfinite(pred).all() and pred.shape == (200,)
            dest = HERE / "fits" / cell_name(inference, honesty) / f"seed_{seed}"
            dest.mkdir(parents=True, exist_ok=True)
            saved = old_pred[["_oci_row_id", "treatment", "outcome", "propensity", "outcome_prediction"]].copy()
            saved["inference"], saved["honesty"], saved["seed"] = inference, honesty, seed
            saved["estimated_cate"] = pred
            saved.to_csv(dest / "predictions.csv", index=False)
            rloss = float(np.mean((saved.outcome - saved.outcome_prediction - pred * (saved.treatment - saved.propensity)) ** 2))
            record = {"inference": inference, "honesty": honesty, "seed": seed, "parameters": parameters,
                      "residual_sha256": residual_hash, "r_loss": rloss, "tree_audit": tree_audit(model),
                      "warnings": [str(w.message) for w in caught]}
            u.write(dest / "fit_audit.json", record)
            joblib.dump(model, dest / "forest.joblib", compress=3)
            for path in [dest / "predictions.csv", dest / "fit_audit.json", dest / "forest.joblib"]:
                artifacts[str(path.resolve())] = u.sha(path)
            new_records.append(record)
            print(json.dumps({"phase": "fit_complete", "inference": inference, "honesty": honesty, "seed": seed,
                              "r_loss": rloss, "tree_counts": record["tree_audit"]["sample_split_effect_counts"]}), flush=True)
    u.write(HERE / "baseline_reuse_validation_2026-09-21.json", {"baseline_fits_reused": 3, "fits": baseline_records})
    u.write(HERE / "fit_validation_2026-09-21.json", {"new_fits": len(new_records), "baseline_fits_reused": 3,
            "new_fits_without_warnings": sum(not r["warnings"] for r in new_records),
            "cached_residuals_unchanged": True, "only_inference_and_honesty_parameters_changed": True,
            "all_predictions_finite": True, "per_tree_and_whole_forest_patient_usage_verified": True})
    for path in [HERE / "baseline_reuse_validation_2026-09-21.json", HERE / "fit_validation_2026-09-21.json"]:
        artifacts[str(path.resolve())] = u.sha(path)
    assert len(new_records) == 9 and len(artifacts) == 29
    verify_prior(manifest)
    u.write(HERE / "predictions_frozen_2026-09-21.json", {"frozen_at": u.now(), "new_fits": 9, "baseline_fits_reused": 3,
            "input_manifest_sha256": u.sha(HERE / "input_manifest_2026-09-21.json"), "files": artifacts})
    print("Nine new forests frozen; three baseline forests reused.", flush=True)


def evaluate():
    manifest = u.read(HERE / "input_manifest_2026-09-21.json")
    frozen = u.read(HERE / "predictions_frozen_2026-09-21.json")
    assert frozen["input_manifest_sha256"] == u.sha(HERE / "input_manifest_2026-09-21.json")
    verify_prior(manifest)
    u.verify(frozen["files"])
    truth = pd.read_parquet(u.DATA / "dataset.parquet", columns=["true_ite_prob"]).iloc[manifest["heldout_rows"]].true_ite_prob.to_numpy()
    rows = []
    for inference, honesty in CELLS:
        for seed in u.SEEDS:
            directory = BASE / "fits/c/sqrt" / f"seed_{seed}" if inference and honesty else HERE / "fits" / cell_name(inference, honesty) / f"seed_{seed}"
            pred = pd.read_csv(directory / "predictions.csv", float_precision="round_trip")
            assert pred._oci_row_id.tolist() == manifest["heldout_rows"]
            est = pred.estimated_cate.to_numpy()
            rloss = float(np.mean((pred.outcome - pred.outcome_prediction - est * (pred.treatment - pred.propensity)) ** 2))
            rows.append({"inference": inference, "honesty": honesty, "seed": seed, "reused_baseline": inference and honesty,
                         "n": len(pred), "correlation": float(np.corrcoef(est, truth)[0, 1]),
                         "rmse": float(np.sqrt(np.mean((est - truth) ** 2))), "mae": float(np.mean(np.abs(est - truth))),
                         "bias": float(np.mean(est - truth)), "prediction_sd": float(np.std(est)), "r_loss": rloss})
    table = pd.DataFrame(rows)
    assert len(table) == 12
    table.to_csv(HERE / "metrics_by_seed_2026-09-21.csv", index=False)
    summaries = []
    for (inference, honesty), part in table.groupby(["inference", "honesty"]):
        summaries.append({"inference": bool(inference), "honesty": bool(honesty),
                          "metrics": {k: {"mean": float(part[k].mean()), "min": float(part[k].min()), "max": float(part[k].max())}
                                      for k in ["correlation", "rmse", "mae", "bias", "prediction_sd", "r_loss"]}})
    contrasts = []
    for factor in ["inference", "honesty"]:
        other = "honesty" if factor == "inference" else "inference"
        for value in [True, False]:
            for seed in u.SEEDS:
                subset = table[(table[other] == value) & (table.seed == seed)]
                on, off = subset[subset[factor]].iloc[0], subset[~subset[factor]].iloc[0]
                contrasts.append({"factor_switched_off": factor, "other_factor": other, "other_value": value, "seed": seed,
                                  "off_minus_on": {k: float(off[k] - on[k]) for k in ["correlation", "rmse", "r_loss"]}})
    u.write(HERE / "evaluation_2026-09-21.json", {"evaluated_at": u.now(), "heldout_rows_each": 200,
            "summaries": summaries, "paired_contrasts": contrasts, "oracle_mean": float(np.mean(truth)), "oracle_sd": float(np.std(truth))})
    verify_prior(manifest)
    u.verify(frozen["files"])
    u.write(HERE / "evaluation_validation_2026-09-21.json", {"validated_at": u.now(), "new_fits": 9,
            "baseline_fits_reused": 3, "all_source_and_artifact_hashes_verified": True, "same_200_heldout_rows_each": True})
    print(table.groupby(["inference", "honesty"])[["correlation", "rmse", "bias", "prediction_sd", "r_loss"]].mean().to_string(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["fit", "evaluate"])
    args = parser.parse_args()
    fit() if args.phase == "fit" else evaluate()
