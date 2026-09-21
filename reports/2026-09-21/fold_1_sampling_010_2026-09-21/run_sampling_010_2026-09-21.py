"""Fold 1: 10% sampling, inference off, both honesty settings, three seeds.

Reuse the twelve corresponding 45%/95% forests. Change only max_samples.
True modifiers in X, confounders in W, observed Y, and original fitted nuisances.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/oci_sampling_010_matplotlib")
import joblib
import numpy as np
import pandas as pd
from econml.grf import CausalForest

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "fold_1_sampling_095_2026-09-21"
spec = importlib.util.spec_from_file_location("prior_sampling_095", PRIOR / "run_sampling_095_2026-09-21.py")
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
u, BASE, GRID = s.u, s.BASE, s.GRID
DATE = "2026-09-21"
FRACTIONS = (0.10, 0.45, 0.95)


def path_for(fraction, honesty, seed):
    if fraction == 0.45:
        return GRID / "fits" / s.g.cell_name(False, honesty) / f"seed_{seed}"
    parent = PRIOR if fraction == 0.95 else HERE
    return parent / "fits" / f"honesty_{str(honesty).lower()}" / f"seed_{seed}"


def verify_prior(manifest):
    s.verify(manifest)
    frozen = u.read(PRIOR / f"predictions_frozen_{DATE}.json")
    assert u.sha(PRIOR / f"input_manifest_{DATE}.json") == frozen["input_manifest_sha256"]
    u.verify(frozen["files"])


def tree_audit(model):
    n = 800
    expected = int(n * model.max_samples)
    counts, leaves, sizes, depths = [], [], [], []
    usage_s, usage_e = np.zeros(n, dtype=int), np.zeros(n, dtype=int)
    for tree, sub in zip(model.estimators_, model.get_subsample_inds()):
        split, estimate = tree.get_train_test_split_inds()
        assert len(sub) == len(np.unique(sub)) == expected
        if model.honest:
            assert len(split) == len(estimate) == expected // 2
            assert set(split).isdisjoint(estimate)
        else:
            assert len(split) == len(estimate) == expected
            np.testing.assert_array_equal(split, estimate)
        usage_s[sub[split]] += 1
        usage_e[sub[estimate]] += 1
        counts.append((len(sub), len(split), len(estimate)))
        leaves.append(tree.get_n_leaves())
        depths.append(tree.get_depth())
        sizes.extend(tree.tree_.n_node_samples[tree.tree_.children_left == -1].tolist())
    assert len(counts) == 200
    return {"trees": len(counts), "sample_split_effect_counts": sorted(set(counts)),
            "leaves_per_tree": s.g.describe(leaves), "effect_patients_per_leaf": s.g.describe(sizes),
            "depth": s.g.describe(depths), "unsplit_trees": int(sum(v == 1 for v in leaves)),
            "distinct_patients_used_for_splits": int(np.count_nonzero(usage_s)),
            "distinct_patients_used_for_effects": int(np.count_nonzero(usage_e)),
            "split_trees_per_patient": s.g.describe(usage_s), "effect_trees_per_patient": s.g.describe(usage_e)}


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
                "new_sample_fraction": 0.10, "reused_sample_fractions": [0.45, 0.95],
                "inference": False, "honesty_settings": [True, False], "min_samples_leaf": 10,
                "max_depth": None, "max_features": "sqrt", "n_estimators": 200, "seeds": u.SEEDS,
                "X": prior["X"], "W": prior["W"], "residual_sha256": prior["residual_sha256"],
                "oracle_effects_or_probabilities_used_for_fitting": False,
                "outcomes": "Original observed binary outcomes.",
                "nuisances": "Original fitted elastic-net nuisance residuals; unchanged for all settings.",
                "sources": {**prior["sources"], **{str(p.resolve()): u.sha(p) for p in extra}},
                "exploratory": "Previously inspected fold; no untouched validation claim."}
    u.write(HERE / f"input_manifest_{DATE}.json", manifest)
    artifacts, reused, new = {}, [], []
    for honesty in [True, False]:
        for seed in u.SEEDS:
            baseline = joblib.load(path_for(0.45, honesty, seed) / "forest.joblib")
            params = baseline.get_params()
            assert params["inference"] is False and params["honest"] == honesty
            assert params["min_samples_leaf"] == 10 and params["max_depth"] is None
            assert params["max_samples"] == 0.45 and params["random_state"] == seed
            reference = None
            for fraction in [0.45, 0.95]:
                old_dir = path_for(fraction, honesty, seed)
                old = baseline if fraction == 0.45 else joblib.load(old_dir / "forest.joblib")
                assert old.get_params() == {**params, "max_samples": fraction}
                saved = pd.read_csv(old_dir / "predictions.csv", float_precision="round_trip")
                assert saved._oci_row_id.tolist() == test
                np.testing.assert_array_equal(old.predict(M[test]).ravel(), saved.estimated_cate.to_numpy())
                shared = saved[["_oci_row_id", "treatment", "outcome", "propensity", "outcome_prediction"]]
                if reference is None:
                    reference = shared.copy()
                else:
                    pd.testing.assert_frame_equal(reference, shared)
                reused.append({"sample_fraction": fraction, "honesty": honesty, "seed": seed,
                               "directory": str(old_dir.resolve()), "reloaded_predictions_identical": True,
                               "tree_audit": tree_audit(old)})
            new_params = {**params, "max_samples": 0.10}
            assert {k for k in params if params[k] != new_params[k]} == {"max_samples"}
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = CausalForest(**new_params).fit(M[train], tres, yres)
            assert u.array_hash(yres, tres) == manifest["residual_sha256"]
            estimate = model.predict(M[test]).ravel()
            assert estimate.shape == (200,) and np.isfinite(estimate).all()
            pred = reference.copy()
            pred["estimated_cate"], pred["sample_fraction"] = estimate, 0.10
            pred["inference"], pred["honesty"], pred["seed"] = False, honesty, seed
            dest = path_for(0.10, honesty, seed)
            dest.mkdir(parents=True, exist_ok=True)
            pred.to_csv(dest / "predictions.csv", index=False)
            audit = {"honesty": honesty, "seed": seed, "parameters": new_params,
                     "residual_sha256": manifest["residual_sha256"], "tree_audit": tree_audit(model),
                     "warnings": [str(w.message) for w in caught],
                     "r_loss": float(np.mean((pred.outcome - pred.outcome_prediction - estimate * (pred.treatment - pred.propensity)) ** 2))}
            u.write(dest / "fit_audit.json", audit)
            joblib.dump(model, dest / "forest.joblib", compress=3)
            for path in [dest / "predictions.csv", dest / "fit_audit.json", dest / "forest.joblib"]:
                artifacts[str(path.resolve())] = u.sha(path)
            new.append(audit)
            print(json.dumps({"at": u.now(), "phase": "fit_complete", "honesty": honesty, "seed": seed,
                              "median_leaves": audit["tree_audit"]["leaves_per_tree"]["median"],
                              "median_effect_patients_per_leaf": audit["tree_audit"]["effect_patients_per_leaf"]["median"]}), flush=True)
    assert len(new) == 6 and len(reused) == 12
    u.write(HERE / f"baseline_reuse_validation_{DATE}.json", {"fits": reused})
    u.write(HERE / f"fit_validation_{DATE}.json", {"new_fits": 6, "baseline_fits_reused": 12,
            "new_fits_without_warnings": sum(not a["warnings"] for a in new),
            "only_sample_fraction_changed": True, "cached_residuals_unchanged": True,
            "all_baseline_predictions_reproduced_from_saved_models": True,
            "per_tree_sample_counts_and_honesty_halves_verified": True})
    for path in [HERE / f"baseline_reuse_validation_{DATE}.json", HERE / f"fit_validation_{DATE}.json"]:
        artifacts[str(path.resolve())] = u.sha(path)
    verify_prior(manifest)
    u.write(HERE / f"predictions_frozen_{DATE}.json", {"frozen_at": u.now(), "new_fits": 6,
            "baseline_fits_reused": 12, "input_manifest_sha256": u.sha(HERE / f"input_manifest_{DATE}.json"),
            "files": artifacts})
    print("Six new 10% forests frozen; twelve 45%/95% forests reused.", flush=True)


def evaluate():
    manifest = u.read(HERE / f"input_manifest_{DATE}.json")
    frozen = u.read(HERE / f"predictions_frozen_{DATE}.json")
    assert u.sha(HERE / f"input_manifest_{DATE}.json") == frozen["input_manifest_sha256"]
    verify_prior(manifest)
    u.verify(frozen["files"])
    truth = pd.read_parquet(u.DATA / "dataset.parquet", columns=["true_ite_prob"]).iloc[manifest["heldout_rows"]].true_ite_prob.to_numpy()
    prior_audits = {(a["sample_fraction"], a["honesty"], a["seed"]): a["tree_audit"]
                    for a in u.read(HERE / f"baseline_reuse_validation_{DATE}.json")["fits"]}
    records = []
    for fraction in FRACTIONS:
        for honesty in [True, False]:
            for seed in u.SEEDS:
                dest = path_for(fraction, honesty, seed)
                pred = pd.read_csv(dest / "predictions.csv", float_precision="round_trip")
                assert pred._oci_row_id.tolist() == manifest["heldout_rows"]
                estimate = pred.estimated_cate.to_numpy()
                audit = (u.read(dest / "fit_audit.json")["tree_audit"] if fraction == 0.10
                         else prior_audits[(fraction, honesty, seed)])
                records.append({"sample_fraction": fraction, "honesty": honesty, "seed": seed,
                                "correlation": float(np.corrcoef(estimate, truth)[0, 1]),
                                "rmse": float(np.sqrt(np.mean((estimate - truth) ** 2))),
                                "mae": float(np.mean(np.abs(estimate - truth))), "bias": float(np.mean(estimate - truth)),
                                "prediction_sd": float(np.std(estimate)),
                                "r_loss": float(np.mean((pred.outcome - pred.outcome_prediction - estimate * (pred.treatment - pred.propensity)) ** 2)),
                                "median_leaves_per_tree": audit["leaves_per_tree"]["median"],
                                "median_effect_patients_per_leaf": audit["effect_patients_per_leaf"]["median"],
                                "median_depth": audit["depth"]["median"], "unsplit_trees": audit["unsplit_trees"]})
    table = pd.DataFrame(records)
    assert len(table) == 18
    table.to_csv(HERE / f"metrics_by_seed_{DATE}.csv", index=False)
    metrics = ["correlation", "rmse", "mae", "bias", "prediction_sd", "r_loss", "median_leaves_per_tree",
               "median_effect_patients_per_leaf", "median_depth", "unsplit_trees"]
    summaries = [{"sample_fraction": float(fraction), "honesty": bool(honesty), "metrics": {
        k: {"mean": float(part[k].mean()), "min": float(part[k].min()), "max": float(part[k].max())}
        for k in metrics}} for (fraction, honesty), part in table.groupby(["sample_fraction", "honesty"])]
    contrasts = []
    for honesty in [True, False]:
        for seed in u.SEEDS:
            part = table[(table.honesty == honesty) & (table.seed == seed)].set_index("sample_fraction")
            for baseline in [0.45, 0.95]:
                contrasts.append({"honesty": honesty, "seed": seed, "baseline_fraction": baseline,
                                  "ten_percent_minus_baseline": {k: float(part.loc[0.10, k] - part.loc[baseline, k])
                                                                 for k in ["correlation", "rmse", "r_loss"]}})
    u.write(HERE / f"evaluation_{DATE}.json", {"evaluated_at": u.now(), "heldout_rows_each": 200,
            "summaries": summaries, "paired_contrasts": contrasts,
            "oracle_mean": float(np.mean(truth)), "oracle_sd": float(np.std(truth))})
    verify_prior(manifest)
    u.verify(frozen["files"])
    u.write(HERE / f"evaluation_validation_{DATE}.json", {"validated_at": u.now(),
            "all_source_and_artifact_hashes_verified": True, "same_200_heldout_rows_each": True,
            "new_fits": 6, "baseline_fits_reused": 12})
    print(table.groupby(["honesty", "sample_fraction"])[metrics].mean().to_string(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["fit", "evaluate"])
    args = parser.parse_args()
    fit() if args.phase == "fit" else evaluate()
