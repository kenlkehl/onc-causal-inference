"""Fold 1: inference off, sample fraction .95, honesty on/off, three seeds.

Reuse corresponding inference-off .45 baselines. Change only max_samples.
True covariates, original fitted nuisances, observed binary outcomes.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/oci_sampling_095_matplotlib")
import joblib
import numpy as np
import pandas as pd
from econml.grf import CausalForest

HERE = Path(__file__).resolve().parent
GRID = HERE.parent / "fold_1_inference_honesty_grid_2026-09-21"
spec = importlib.util.spec_from_file_location("previous_inference_honesty_grid", GRID / "run_inference_honesty_grid_2026-09-21.py")
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)
u, BASE = g.u, g.BASE


def verify(manifest):
    g.verify_prior(manifest)
    u.verify(u.read(GRID / "predictions_frozen_2026-09-21.json")["files"])


def inspect_trees(model):
    counts, leaves, sizes = [], [], []
    usage_s, usage_e = np.zeros(800, dtype=int), np.zeros(800, dtype=int)
    for tree, sub in zip(model.estimators_, model.get_subsample_inds()):
        split, estimate = tree.get_train_test_split_inds()
        assert len(sub) == len(np.unique(sub)) == 760
        if model.honest:
            assert len(split) == len(estimate) == 380 and set(split).isdisjoint(estimate)
        else:
            assert len(split) == len(estimate) == 760
            np.testing.assert_array_equal(split, estimate)
        counts.append((len(sub), len(split), len(estimate)))
        usage_s[sub[split]] += 1
        usage_e[sub[estimate]] += 1
        leaves.append(tree.get_n_leaves())
        sizes.extend(tree.tree_.n_node_samples[tree.tree_.children_left == -1].tolist())
    assert np.all(usage_s > 0) and np.all(usage_e > 0)
    return {"sample_split_effect_counts": sorted(set(counts)), "leaves_per_tree": g.describe(leaves),
            "effect_patients_per_leaf": g.describe(sizes), "all_800_patients_used_in_both_roles": True,
            "split_trees_per_patient": g.describe(usage_s), "effect_trees_per_patient": g.describe(usage_e)}


def fit():
    if (HERE / "predictions_frozen_2026-09-21.json").exists():
        raise ValueError("Already frozen; evaluate instead")
    gm = u.read(GRID / "input_manifest_2026-09-21.json")
    gf = u.read(GRID / "predictions_frozen_2026-09-21.json")
    assert u.sha(GRID / "input_manifest_2026-09-21.json") == gf["input_manifest_sha256"]
    verify(gm)
    bm = u.read(BASE / "input_manifest_2026-09-21.json")
    train, test = gm["training_rows"], gm["heldout_rows"]
    definitions = bm["confounders"] + bm["modifiers"]
    data = pd.read_parquet(u.DATA / "dataset.parquet", columns=["true_" + d["name"] for d in definitions])
    C, _, _, _ = u.encode(data, bm["confounders"], train)
    M, _, _, _ = u.encode(data, bm["modifiers"], train)
    initial = joblib.load(BASE / "fits/c/initial_model.joblib")
    yres, tres, old_X, old_W = initial.residuals_
    np.testing.assert_array_equal(M[train], old_X)
    np.testing.assert_array_equal(C[train], old_W)
    assert u.array_hash(M[test]) == bm["scenarios"]["c"]["test_X_sha256"]
    assert u.array_hash(yres, tres) == gm["residual_sha256"]
    extra = [Path(__file__), GRID / "input_manifest_2026-09-21.json", GRID / "predictions_frozen_2026-09-21.json",
             GRID / "run_inference_honesty_grid_2026-09-21.py"]
    manifest = {"created_at": u.now(), "purpose": __doc__, "training_rows": train, "heldout_rows": test,
                "inference": False, "honesty_settings": [True, False], "new_sample_fraction": 0.95,
                "reused_sample_fraction": 0.45, "seeds": u.SEEDS, "n_estimators": 200, "max_features": "sqrt",
                "X": gm["X"], "W": gm["W"], "residual_sha256": gm["residual_sha256"],
                "oracle_effects_or_probabilities_used_for_fitting": False,
                "nuisances": "Original fitted elastic-net nuisance residuals; unchanged across all settings.",
                "outcomes": "Original observed binary outcomes.",
                "sources": {**gm["sources"], **{str(p.resolve()): u.sha(p) for p in extra}},
                "exploratory": "The same fold has been examined previously; no untouched validation claim."}
    u.write(HERE / "input_manifest_2026-09-21.json", manifest)
    artifacts, audits = {}, []
    for honesty in [True, False]:
        for seed in u.SEEDS:
            old_dir = GRID / "fits" / g.cell_name(False, honesty) / f"seed_{seed}"
            old = joblib.load(old_dir / "forest.joblib")
            params = old.get_params()
            assert params["inference"] is False and params["honest"] == honesty and params["max_samples"] == 0.45
            saved = pd.read_csv(old_dir / "predictions.csv", float_precision="round_trip")
            assert saved._oci_row_id.tolist() == test
            np.testing.assert_array_equal(old.predict(M[test]).ravel(), saved.estimated_cate.to_numpy())
            new_params = {**params, "max_samples": 0.95}
            assert {k for k in params if params[k] != new_params[k]} == {"max_samples"}
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                forest = CausalForest(**new_params).fit(M[train], tres, yres)
            assert u.array_hash(yres, tres) == manifest["residual_sha256"]
            pred = forest.predict(M[test]).ravel()
            assert pred.shape == (200,) and np.isfinite(pred).all()
            saved["estimated_cate"], saved["sample_fraction"] = pred, 0.95
            dest = HERE / "fits" / f"honesty_{str(honesty).lower()}" / f"seed_{seed}"
            dest.mkdir(parents=True, exist_ok=True)
            saved.to_csv(dest / "predictions.csv", index=False)
            audit = {"honesty": honesty, "seed": seed, "parameters": new_params,
                     "residual_sha256": manifest["residual_sha256"], "tree_audit": inspect_trees(forest),
                     "warnings": [str(w.message) for w in caught],
                     "r_loss": float(np.mean((saved.outcome - saved.outcome_prediction - pred * (saved.treatment - saved.propensity)) ** 2))}
            u.write(dest / "fit_audit.json", audit)
            joblib.dump(forest, dest / "forest.joblib", compress=3)
            for path in [dest / "predictions.csv", dest / "fit_audit.json", dest / "forest.joblib"]:
                artifacts[str(path.resolve())] = u.sha(path)
            audits.append(audit)
            print(json.dumps({"phase": "fit_complete", "honesty": honesty, "seed": seed,
                              "sample_counts": audit["tree_audit"]["sample_split_effect_counts"], "r_loss": audit["r_loss"]}), flush=True)
    assert len(audits) == 6
    u.write(HERE / "fit_validation_2026-09-21.json", {"new_fits": 6, "baseline_fits_reused": 6,
            "new_fits_without_warnings": sum(not a["warnings"] for a in audits),
            "only_sample_fraction_changed": True, "cached_residuals_unchanged": True,
            "all_baseline_predictions_reproduced_from_saved_models": True, "tree_samples_verified": True})
    path = HERE / "fit_validation_2026-09-21.json"
    artifacts[str(path.resolve())] = u.sha(path)
    verify(manifest)
    u.write(HERE / "predictions_frozen_2026-09-21.json", {"frozen_at": u.now(), "new_fits": 6, "baseline_fits_reused": 6,
            "input_manifest_sha256": u.sha(HERE / "input_manifest_2026-09-21.json"), "files": artifacts})
    print("Six new forests frozen; six .45 baselines reused.", flush=True)


def evaluate():
    manifest = u.read(HERE / "input_manifest_2026-09-21.json")
    frozen = u.read(HERE / "predictions_frozen_2026-09-21.json")
    assert u.sha(HERE / "input_manifest_2026-09-21.json") == frozen["input_manifest_sha256"]
    verify(manifest)
    u.verify(frozen["files"])
    truth = pd.read_parquet(u.DATA / "dataset.parquet", columns=["true_ite_prob"]).iloc[manifest["heldout_rows"]].true_ite_prob.to_numpy()
    records = []
    for fraction in [0.45, 0.95]:
        for honesty in [True, False]:
            for seed in u.SEEDS:
                dest = GRID / "fits" / g.cell_name(False, honesty) / f"seed_{seed}" if fraction == 0.45 else HERE / "fits" / f"honesty_{str(honesty).lower()}" / f"seed_{seed}"
                pred = pd.read_csv(dest / "predictions.csv", float_precision="round_trip")
                assert pred._oci_row_id.tolist() == manifest["heldout_rows"]
                est = pred.estimated_cate.to_numpy()
                records.append({"inference": False, "sample_fraction": fraction, "honesty": honesty, "seed": seed,
                                "correlation": float(np.corrcoef(est, truth)[0, 1]), "rmse": float(np.sqrt(np.mean((est - truth) ** 2))),
                                "bias": float(np.mean(est - truth)), "prediction_sd": float(np.std(est)),
                                "r_loss": float(np.mean((pred.outcome - pred.outcome_prediction - est * (pred.treatment - pred.propensity)) ** 2))})
    table = pd.DataFrame(records)
    assert len(table) == 12
    table.to_csv(HERE / "metrics_by_seed_2026-09-21.csv", index=False)
    summaries = []
    for (fraction, honesty), part in table.groupby(["sample_fraction", "honesty"]):
        summaries.append({"sample_fraction": float(fraction), "honesty": bool(honesty),
                          "metrics": {k: {"mean": float(part[k].mean()), "min": float(part[k].min()), "max": float(part[k].max())}
                                      for k in ["correlation", "rmse", "bias", "prediction_sd", "r_loss"]}})
    u.write(HERE / "evaluation_2026-09-21.json", {"evaluated_at": u.now(), "heldout_rows_each": 200,
            "summaries": summaries, "oracle_mean": float(np.mean(truth)), "oracle_sd": float(np.std(truth))})
    verify(manifest)
    u.verify(frozen["files"])
    u.write(HERE / "evaluation_validation_2026-09-21.json", {"validated_at": u.now(), "new_fits": 6, "baseline_fits_reused": 6,
            "all_source_and_artifact_hashes_verified": True, "same_200_heldout_rows_each": True})
    print(table.groupby(["sample_fraction", "honesty"])[["correlation", "rmse", "bias", "prediction_sd", "r_loss"]].mean().to_string(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["fit", "evaluate"])
    args = parser.parse_args()
    fit() if args.phase == "fit" else evaluate()
