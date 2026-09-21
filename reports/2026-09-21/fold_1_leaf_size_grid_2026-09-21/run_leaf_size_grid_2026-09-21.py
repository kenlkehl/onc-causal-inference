"""Fold 1 leaf-size diagnostic: 10, 20, 40, 80; 95% sampling, honest, no inference.

Reuse the three size-10 forests; change only min_samples_leaf for nine new fits.
Use true modifiers in X, true confounders in W, and original fitted nuisances.
Oracle effects are loaded only after all new predictions have been frozen.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/oci_leaf_size_matplotlib")
import joblib
import numpy as np
import pandas as pd
from econml.grf import CausalForest

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "fold_1_sampling_095_2026-09-21"
spec = importlib.util.spec_from_file_location("previous_sampling_diagnostic", PRIOR / "run_sampling_095_2026-09-21.py")
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
u, BASE = s.u, s.BASE
LEAF_SIZES = (10, 20, 40, 80)
DATE = "2026-09-21"


def path_for(size, seed):
    if size == 10:
        return PRIOR / "fits/honesty_true" / f"seed_{seed}"
    return HERE / "fits" / f"min_samples_leaf_{size}" / f"seed_{seed}"


def verify_prior(manifest):
    s.verify(manifest)
    freeze = u.read(PRIOR / f"predictions_frozen_{DATE}.json")
    assert u.sha(PRIOR / f"input_manifest_{DATE}.json") == freeze["input_manifest_sha256"]
    u.verify(freeze["files"])


def tree_audit(model):
    audit = s.inspect_trees(model)
    audit["depth"] = s.g.describe([tree.get_depth() for tree in model.estimators_])
    return audit


def fit():
    if (HERE / f"predictions_frozen_{DATE}.json").exists():
        raise ValueError("Already frozen; evaluate instead of overwriting")
    prior = u.read(PRIOR / f"input_manifest_{DATE}.json")
    verify_prior(prior)
    original = u.read(BASE / f"input_manifest_{DATE}.json")
    train, test = prior["training_rows"], prior["heldout_rows"]
    assert len(train) == 800 and len(test) == 200 and set(train).isdisjoint(test)
    definitions = original["confounders"] + original["modifiers"]
    frame = pd.read_parquet(u.DATA / "dataset.parquet", columns=["true_" + d["name"] for d in definitions])
    C, _, _, _ = u.encode(frame, original["confounders"], train)
    M, _, _, _ = u.encode(frame, original["modifiers"], train)
    initial = joblib.load(BASE / "fits/c/initial_model.joblib")
    yres, tres, old_X, old_W = initial.residuals_
    np.testing.assert_array_equal(M[train], old_X)
    np.testing.assert_array_equal(C[train], old_W)
    assert u.array_hash(M[test]) == original["scenarios"]["c"]["test_X_sha256"]
    assert u.array_hash(yres, tres) == prior["residual_sha256"]
    extra = [Path(__file__), PRIOR / f"input_manifest_{DATE}.json",
             PRIOR / f"predictions_frozen_{DATE}.json", PRIOR / "run_sampling_095_2026-09-21.py"]
    manifest = {"created_at": u.now(), "purpose": __doc__, "training_rows": train, "heldout_rows": test,
                "min_samples_leaf": LEAF_SIZES, "reused_baseline_leaf_size": 10,
                "inference": False, "honesty": True, "sample_fraction": 0.95,
                "max_depth": None, "max_features": "sqrt", "n_estimators": 200, "seeds": u.SEEDS,
                "X": prior["X"], "W": prior["W"], "residual_sha256": prior["residual_sha256"],
                "oracle_effects_or_probabilities_used_for_fitting": False,
                "outcomes": "Original observed binary outcomes.",
                "nuisances": "Original fitted elastic-net nuisances and residuals; no refitting.",
                "sources": {**prior["sources"], **{str(p.resolve()): u.sha(p) for p in extra}},
                "exploratory": "Previously inspected fold; these oracle scores do not constitute untouched validation."}
    u.write(HERE / f"input_manifest_{DATE}.json", manifest)
    artifacts, reused, new = {}, [], []
    for seed in u.SEEDS:
        old_dir = path_for(10, seed)
        baseline = joblib.load(old_dir / "forest.joblib")
        params = baseline.get_params()
        assert params["min_samples_leaf"] == 10 and params["max_depth"] is None
        assert params["inference"] is False and params["honest"] is True and params["max_samples"] == 0.95
        saved = pd.read_csv(old_dir / "predictions.csv", float_precision="round_trip")
        assert saved._oci_row_id.tolist() == test
        np.testing.assert_array_equal(baseline.predict(M[test]).ravel(), saved.estimated_cate.to_numpy())
        reused.append({"seed": seed, "directory": str(old_dir.resolve()),
                       "reloaded_predictions_identical": True, "tree_audit": tree_audit(baseline)})
        for size in LEAF_SIZES[1:]:
            new_params = {**params, "min_samples_leaf": size}
            assert {k for k in params if params[k] != new_params[k]} == {"min_samples_leaf"}
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = CausalForest(**new_params).fit(M[train], tres, yres)
            assert u.array_hash(yres, tres) == manifest["residual_sha256"]
            for old_sub, new_sub in zip(baseline.get_subsample_inds(), model.get_subsample_inds()):
                np.testing.assert_array_equal(old_sub, new_sub)
            for old_tree, new_tree in zip(baseline.estimators_, model.estimators_):
                for old_indices, new_indices in zip(old_tree.get_train_test_split_inds(), new_tree.get_train_test_split_inds()):
                    np.testing.assert_array_equal(old_indices, new_indices)
            estimate = model.predict(M[test]).ravel()
            assert estimate.shape == (200,) and np.isfinite(estimate).all()
            pred = saved.copy()
            pred["estimated_cate"], pred["min_samples_leaf"] = estimate, size
            dest = path_for(size, seed)
            dest.mkdir(parents=True, exist_ok=True)
            pred.to_csv(dest / "predictions.csv", index=False)
            audit = {"min_samples_leaf": size, "seed": seed, "parameters": new_params,
                     "residual_sha256": manifest["residual_sha256"], "tree_audit": tree_audit(model),
                     "same_subsamples_and_honesty_halves_as_baseline": True,
                     "warnings": [str(w.message) for w in caught],
                     "r_loss": float(np.mean((pred.outcome - pred.outcome_prediction - estimate * (pred.treatment - pred.propensity)) ** 2))}
            u.write(dest / "fit_audit.json", audit)
            joblib.dump(model, dest / "forest.joblib", compress=3)
            for path in [dest / "predictions.csv", dest / "fit_audit.json", dest / "forest.joblib"]:
                artifacts[str(path.resolve())] = u.sha(path)
            new.append(audit)
            print(json.dumps({"at": u.now(), "phase": "fit_complete", "min_samples_leaf": size, "seed": seed,
                              "median_leaves": audit["tree_audit"]["leaves_per_tree"]["median"],
                              "median_effect_patients_per_leaf": audit["tree_audit"]["effect_patients_per_leaf"]["median"]}), flush=True)
    assert len(new) == 9 and len(reused) == 3
    u.write(HERE / f"baseline_reuse_validation_{DATE}.json", {"fits": reused})
    u.write(HERE / f"fit_validation_{DATE}.json", {"new_fits": 9, "baseline_fits_reused": 3,
            "new_fits_without_warnings": sum(not a["warnings"] for a in new),
            "only_minimum_leaf_size_changed": True, "cached_residuals_unchanged": True,
            "same_subsamples_and_honesty_halves_as_baseline": True,
            "all_baseline_predictions_reproduced_from_saved_models": True,
            "per_tree_760_with_disjoint_380_380_halves_verified": True})
    for path in [HERE / f"baseline_reuse_validation_{DATE}.json", HERE / f"fit_validation_{DATE}.json"]:
        artifacts[str(path.resolve())] = u.sha(path)
    verify_prior(manifest)
    u.write(HERE / f"predictions_frozen_{DATE}.json", {"frozen_at": u.now(), "new_fits": 9,
            "baseline_fits_reused": 3, "input_manifest_sha256": u.sha(HERE / f"input_manifest_{DATE}.json"),
            "files": artifacts})
    print("Nine new forests frozen; three size-10 forests reused.", flush=True)


def evaluate():
    manifest = u.read(HERE / f"input_manifest_{DATE}.json")
    frozen = u.read(HERE / f"predictions_frozen_{DATE}.json")
    assert u.sha(HERE / f"input_manifest_{DATE}.json") == frozen["input_manifest_sha256"]
    verify_prior(manifest)
    u.verify(frozen["files"])
    truth = pd.read_parquet(u.DATA / "dataset.parquet", columns=["true_ite_prob"]).iloc[manifest["heldout_rows"]].true_ite_prob.to_numpy()
    records = []
    baseline_audits = {a["seed"]: a["tree_audit"] for a in u.read(HERE / f"baseline_reuse_validation_{DATE}.json")["fits"]}
    for size in LEAF_SIZES:
        for seed in u.SEEDS:
            dest = path_for(size, seed)
            pred = pd.read_csv(dest / "predictions.csv", float_precision="round_trip")
            assert pred._oci_row_id.tolist() == manifest["heldout_rows"]
            estimate = pred.estimated_cate.to_numpy()
            audit = baseline_audits[seed] if size == 10 else u.read(dest / "fit_audit.json")["tree_audit"]
            records.append({"min_samples_leaf": size, "seed": seed, "reused_baseline": size == 10,
                            "correlation": float(np.corrcoef(estimate, truth)[0, 1]),
                            "rmse": float(np.sqrt(np.mean((estimate - truth) ** 2))),
                            "mae": float(np.mean(np.abs(estimate - truth))), "bias": float(np.mean(estimate - truth)),
                            "prediction_sd": float(np.std(estimate)),
                            "r_loss": float(np.mean((pred.outcome - pred.outcome_prediction - estimate * (pred.treatment - pred.propensity)) ** 2)),
                            "median_leaves_per_tree": audit["leaves_per_tree"]["median"],
                            "median_effect_patients_per_leaf": audit["effect_patients_per_leaf"]["median"],
                            "median_depth": audit["depth"]["median"]})
    table = pd.DataFrame(records)
    assert len(table) == 12
    table.to_csv(HERE / f"metrics_by_seed_{DATE}.csv", index=False)
    metrics = ["correlation", "rmse", "mae", "bias", "prediction_sd", "r_loss", "median_leaves_per_tree",
               "median_effect_patients_per_leaf", "median_depth"]
    summaries = [{"min_samples_leaf": int(size), "metrics": {
        k: {"mean": float(part[k].mean()), "min": float(part[k].min()), "max": float(part[k].max())}
        for k in metrics}} for size, part in table.groupby("min_samples_leaf")]
    contrasts = []
    for seed in u.SEEDS:
        part = table[table.seed == seed].set_index("min_samples_leaf")
        for size in LEAF_SIZES[1:]:
            contrasts.append({"seed": seed, "min_samples_leaf": size,
                              "minus_size_10": {k: float(part.loc[size, k] - part.loc[10, k])
                                                for k in ["correlation", "rmse", "r_loss"]}})
    u.write(HERE / f"evaluation_{DATE}.json", {"evaluated_at": u.now(), "heldout_rows_each": 200,
            "summaries": summaries, "paired_contrasts": contrasts,
            "oracle_mean": float(np.mean(truth)), "oracle_sd": float(np.std(truth))})
    verify_prior(manifest)
    u.verify(frozen["files"])
    u.write(HERE / f"evaluation_validation_{DATE}.json", {"validated_at": u.now(),
            "all_source_and_artifact_hashes_verified": True, "same_200_heldout_rows_each": True,
            "new_fits": 9, "baseline_fits_reused": 3})
    print(table.groupby("min_samples_leaf")[metrics].mean().to_string(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["fit", "evaluate"])
    args = parser.parse_args()
    fit() if args.phase == "fit" else evaluate()
