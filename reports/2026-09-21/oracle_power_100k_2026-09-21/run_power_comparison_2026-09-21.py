"""Sample-size diagnostic with historical forest/nuisance settings.

Nested 800/8,000/80,000 training samples share 20,000 independent test rows.
Use modifiers-only X and all-oracle X, sharing estimated nuisance residuals.
Three fixed forest seeds; no oracle probability or effect labels enter fitting.
Also evaluate the existing original-cohort 800-patient forests on the new test.
"""
from pathlib import Path
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib.util
import inspect
import json
import multiprocessing
import os
import time
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/oci_oracle_power_matplotlib")
HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "fold_1_true_oracle_xw_comparison_2026-09-21"
spec = importlib.util.spec_from_file_location("original_oracle_experiment", BASE / "run_true_oracle_xw_comparison.py")
u = importlib.util.module_from_spec(spec)
spec.loader.exec_module(u)
import joblib
import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from econml.grf import CausalForest
from oci.models.elastic_net_nuisance import ElasticNetLogisticClassifier

DATE = "2026-09-21"
SIZES = (800, 8000, 80000)
SCENARIOS = ("modifiers", "all")


def progress(phase, n=None, **details):
    record = {"updated_at": u.now(), "phase": phase, **details}
    if n is not None:
        record["training_rows"] = n
    u.write(HERE / ("status.json" if n is None else f"status_n_{n}.json"), record)
    print(json.dumps(record), flush=True)


def verify_inputs(manifest):
    u.verify(manifest["sources"])
    generation = u.read(HERE / f"generation_manifest_{DATE}.json")
    u.verify(generation["sources"])
    u.verify(generation["files"])


def describe(values):
    values = np.asarray(values)
    return {"min": float(values.min()), "median": float(np.median(values)),
            "mean": float(values.mean()), "max": float(values.max())}


def tree_audit(forest, n):
    leaves, sizes, depths = [], [], []
    split_usage, effect_usage = np.zeros(n, dtype=int), np.zeros(n, dtype=int)
    expected = int(n * 0.45)
    for tree, subsample in zip(forest.estimators_, forest.get_subsample_inds()):
        splitting, estimation = tree.get_train_test_split_inds()
        assert len(subsample) == len(np.unique(subsample)) == expected
        assert len(splitting) == len(estimation) == expected // 2
        assert set(splitting).isdisjoint(estimation)
        split_usage[subsample[splitting]] += 1
        effect_usage[subsample[estimation]] += 1
        leaves.append(tree.get_n_leaves())
        depths.append(tree.get_depth())
        sizes.extend(tree.tree_.n_node_samples[tree.tree_.children_left == -1].tolist())
    assert len(leaves) == 200 and min(sizes) >= 10
    return {"trees": len(leaves), "sample_split_estimation_counts": [expected, expected // 2, expected // 2],
            "leaves_per_tree": describe(leaves), "depth": describe(depths),
            "effect_patients_per_leaf": describe(sizes),
            "patients_used_for_splits": int(np.count_nonzero(split_usage)),
            "patients_used_for_effects": int(np.count_nonzero(effect_usage))}


def fixed_encode(frame, definitions, encoding):
    specs = {item["name"]: item for item in encoding}
    arrays = []
    for definition in definitions:
        name = definition["name"]
        values = frame["true_" + name]
        if definition["type"] == "continuous":
            arrays.append((values.to_numpy(float) - specs[name]["training_mean"]) / specs[name]["training_sd"])
        else:
            arrays.extend((values == category).to_numpy(float) for category in specs[name]["categories"])
    matrix = np.column_stack(arrays)
    assert np.isfinite(matrix).all()
    return matrix


def save_forest(forest, Xtest, prediction_base, destination, model_path, details, n):
    destination.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    estimate, lower, upper = [np.asarray(v).ravel() for v in forest.predict(Xtest, interval=True, alpha=0.05)]
    assert np.isfinite(np.column_stack([estimate, lower, upper])).all()
    assert np.all(lower <= estimate) and np.all(estimate <= upper)
    prediction = prediction_base.copy()
    prediction["estimated_cate"], prediction["lower_95"], prediction["upper_95"] = estimate, lower, upper
    prediction.to_parquet(destination / "predictions.parquet", index=False)
    audit = {**details, "parameters": forest.get_params(), "tree_audit": tree_audit(forest, n),
             "model_path": str(model_path), "prediction_path": str(destination / "predictions.parquet"),
             "r_loss": float(np.mean((prediction.outcome - prediction.outcome_prediction -
                                       estimate * (prediction.treatment - prediction.propensity)) ** 2))}
    joblib.dump(forest, model_path, compress=3)
    u.write(destination / "fit_audit.json", audit)
    return {str(path.resolve()): u.sha(path)
            for path in [model_path, destination / "predictions.parquet", destination / "fit_audit.json"]}


def fit_size(n):
    manifest = u.read(HERE / f"input_manifest_{DATE}.json")
    split = u.read(HERE / f"splits_{DATE}.json")
    train, test = split["training_rows"][str(n)], split["heldout_rows"]
    definitions = manifest["confounders"] + manifest["modifiers"]
    data = pd.read_parquet(manifest["temporary_dataset"], columns=[
        *["true_" + d["name"] for d in definitions], "treatment_indicator", "outcome_indicator"])
    C, cnames, _, caudit = u.encode(data, manifest["confounders"], train)
    M, mnames, _, maudit = u.encode(data, manifest["modifiers"], train)
    T, Y = data.treatment_indicator.to_numpy(), data.outcome_indicator.to_numpy()
    cv = [(np.asarray(fold["fit_local"]), np.asarray(fold["heldout_local"])) for fold in split["inner_splits"][str(n)]]
    assert sorted(np.concatenate([b for _, b in cv]).tolist()) == list(range(n))
    assert all(set(a).isdisjoint(b) and len(a) + len(b) == n for a, b in cv)
    dest = HERE / "fits" / f"n_{n}"
    temporary = Path(manifest["temporary_directory"]) / "models" / f"n_{n}"
    dest.mkdir(parents=True, exist_ok=True)
    temporary.mkdir(parents=True, exist_ok=True)
    design = {"training_rows": n, "encoding": caudit + maudit, "confounder_columns": cnames, "modifier_columns": mnames,
              "train_modifiers_sha256": u.array_hash(M[train]), "test_modifiers_sha256": u.array_hash(M[test]),
              "train_all_sha256": u.array_hash(np.column_stack([C[train], M[train]])),
              "test_all_sha256": u.array_hash(np.column_stack([C[test], M[test]]))}
    u.write(dest / "design.json", design)
    progress("fitting_nuisances_and_initial_forest", n=n)
    started = time.monotonic()
    model = CausalForestDML(model_t=ElasticNetLogisticClassifier(**u.NUISANCE),
                           model_y=ElasticNetLogisticClassifier(**u.NUISANCE),
                           discrete_treatment=True, discrete_outcome=True, cv=cv,
                           random_state=u.SEEDS[0], max_features="sqrt", **u.FOREST)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(Y[train], T[train], X=M[train], W=C[train], cache_values=True)
    yres, tres, cached_X, cached_W = model.residuals_
    np.testing.assert_array_equal(cached_X, M[train])
    np.testing.assert_array_equal(cached_W, C[train])
    nuisance_e, nuisance_m = u.nuisance_predictions(model, M[test], C[test])
    residual_hash = u.array_hash(yres, tres, nuisance_e, nuisance_m)
    np.savez_compressed(temporary / "residuals.npz", yres=yres, tres=tres,
                        heldout_propensity=nuisance_e, heldout_outcome_prediction=nuisance_m)
    pd.DataFrame({"_oci_row_id": train, "treatment": T[train], "outcome": Y[train],
                  "propensity": T[train] - np.asarray(tres).ravel(),
                  "outcome_prediction": Y[train] - np.asarray(yres).ravel()}).to_parquet(dest / "training_nuisances.parquet", index=False)
    warning_counts = Counter((type(w.message).__name__, str(w.message)) for w in caught)
    nuisance_audit = {"seconds_including_initial_forest": time.monotonic() - started,
                      "residual_sha256": residual_hash, "initial_model_path": str(temporary / "initial_model.joblib"),
                      "residuals_path": str(temporary / "residuals.npz"),
                      "models": {role: [fitted.fit_audit() for fitted in clones[0]]
                                 for role, clones in [("treatment", model.models_t), ("outcome", model.models_y)]},
                      "warnings": [{"type": k[0], "message": k[1], "count": value} for k, value in warning_counts.items()]}
    joblib.dump(model, temporary / "initial_model.joblib", compress=3)
    u.write(dest / "nuisance_audit.json", nuisance_audit)
    paths = [dest / "design.json", dest / "training_nuisances.parquet", dest / "nuisance_audit.json",
             temporary / "initial_model.joblib", temporary / "residuals.npz"]
    artifacts = {str(path.resolve()): u.sha(path) for path in paths}
    progress("nuisances_fitted", n=n, seconds=nuisance_audit["seconds_including_initial_forest"],
             warnings=sum(warning_counts.values()))
    initial = model.model_cate.estimators_[0]
    params = initial.get_params()
    prediction_base = pd.DataFrame({"_oci_row_id": test, "treatment": T[test], "outcome": Y[test],
                                    "propensity": nuisance_e, "outcome_prediction": nuisance_m})
    designs = {"modifiers": M, "all": np.column_stack([C, M])}
    for scenario in SCENARIOS:
        X = designs[scenario]
        for seed in u.SEEDS:
            start = time.monotonic()
            use_initial = scenario == "modifiers" and seed == u.SEEDS[0]
            with warnings.catch_warnings(record=True) as forest_warnings:
                warnings.simplefilter("always")
                forest = initial if use_initial else CausalForest(**{**params, "random_state": seed}).fit(X[train], tres, yres)
            assert forest.get_params() == {**params, "random_state": seed}
            assert u.array_hash(yres, tres, nuisance_e, nuisance_m) == residual_hash
            target = dest / scenario / f"seed_{seed}"
            details = {"cohort": "new", "training_rows": n, "scenario": scenario, "seed": seed,
                       "residual_sha256": residual_hash, "nuisances_shared_across_X_configurations": True,
                       "warnings": [str(w.message) for w in forest_warnings],
                       "native_initial_forest_reused": use_initial}
            artifacts.update(save_forest(forest, X[test], prediction_base, target,
                                         temporary / scenario / f"seed_{seed}.joblib", details, n))
            if use_initial:
                np.testing.assert_array_equal(forest.predict(X[test]).ravel(), model.effect(M[test]).ravel())
                calculated = u.read(target / "fit_audit.json")["r_loss"]
                assert abs(calculated - model.score(Y[test], T[test], X=M[test], W=C[test])) < 1e-12
            progress("forest_complete", n=n, scenario=scenario, seed=seed, seconds=time.monotonic() - start)
    u.write(dest / "completed.json", {"completed_at": u.now(), "forests": 6, "files": artifacts})
    return artifacts


def reuse_original(manifest):
    original = u.read(BASE / f"input_manifest_{DATE}.json")
    frozen = u.read(BASE / f"predictions_frozen_{DATE}.json")
    assert u.sha(BASE / f"input_manifest_{DATE}.json") == frozen["input_manifest_sha256"]
    u.verify(original["sources"])
    u.verify(frozen["files"])
    split = u.read(HERE / f"splits_{DATE}.json")
    test = split["heldout_rows"]
    names = ["true_" + d["name"] for d in original["confounders"] + original["modifiers"]]
    data = pd.read_parquet(manifest["temporary_dataset"], columns=names + ["treatment_indicator", "outcome_indicator"]).iloc[test]
    C = fixed_encode(data, original["confounders"], original["encoding"])
    M = fixed_encode(data, original["modifiers"], original["encoding"])
    artifacts = {}
    for scenario, old_scenario in [("modifiers", "c"), ("all", "a")]:
        X, W = (M, C) if scenario == "modifiers" else (np.column_stack([C, M]), None)
        native = joblib.load(BASE / "fits" / old_scenario / "initial_model.joblib")
        e, m = u.nuisance_predictions(native, X, W)
        for seed in u.SEEDS:
            model_path = BASE / "fits" / old_scenario / "sqrt" / f"seed_{seed}" / "forest.joblib"
            forest = joblib.load(model_path)
            prediction, lower, upper = [np.asarray(v).ravel() for v in forest.predict(X, interval=True)]
            target = HERE / "original_cohort_on_new_test" / scenario / f"seed_{seed}"
            target.mkdir(parents=True, exist_ok=True)
            output = pd.DataFrame({"_oci_row_id": test, "treatment": data.treatment_indicator.to_numpy(),
                                   "outcome": data.outcome_indicator.to_numpy(), "propensity": e,
                                   "outcome_prediction": m, "estimated_cate": prediction,
                                   "lower_95": lower, "upper_95": upper})
            output.to_parquet(target / "predictions.parquet", index=False)
            u.write(target / "fit_audit.json", {"cohort": "original", "training_rows": 800, "scenario": scenario,
                    "seed": seed, "model_path": str(model_path), "parameters": forest.get_params(),
                    "reused_without_refitting": True, "tree_audit": tree_audit(forest, 800)})
            for path in [target / "predictions.parquet", target / "fit_audit.json"]:
                artifacts[str(path.resolve())] = u.sha(path)
    u.verify(frozen["files"])
    return artifacts


def fit():
    if (HERE / f"predictions_frozen_{DATE}.json").exists():
        raise ValueError("Already frozen; evaluate instead")
    generation = u.read(HERE / f"generation_manifest_{DATE}.json")
    original = u.read(BASE / f"input_manifest_{DATE}.json")
    sources = [Path(__file__), HERE / f"generation_manifest_{DATE}.json", BASE / f"input_manifest_{DATE}.json",
               BASE / f"predictions_frozen_{DATE}.json", BASE / "run_true_oracle_xw_comparison.py",
               u.ROOT / "oci/models/elastic_net_nuisance.py", Path(inspect.getfile(CausalForestDML)),
               Path(inspect.getfile(CausalForest)), Path(inspect.getfile(CausalForest.fit))]
    manifest = {"created_at": u.now(), "purpose": __doc__, "training_sizes": SIZES, "heldout_rows": 20000,
                "temporary_dataset": generation["temporary_dataset"], "temporary_directory": generation["temporary_directory"],
                "confounders": original["confounders"], "modifiers": original["modifiers"],
                "scenarios": {"modifiers": {"X": "M", "W": "C"}, "all": {"X": "C+M", "W": "none"}},
                "nuisance_parameters": u.NUISANCE, "forest_parameters": {**u.FOREST, "max_features": "sqrt"},
                "forest_seeds": u.SEEDS, "five_crossfit_folds": True, "propensity_filtering": False,
                "oracle_effects_or_probabilities_used_for_fitting": False,
                "nuisance_policy": "Fit once per sample size, sharing exact residuals across six forests; no duplicated controls.",
                "process_parallelism": "Three independent sample sizes; each estimator retains historical n_jobs=1.",
                "sources": {str(path.resolve()): u.sha(path) for path in sources}, "versions": original["versions"]}
    verify_inputs(manifest)
    u.write(HERE / f"input_manifest_{DATE}.json", manifest)
    progress("fitting_learning_curve", training_sizes=SIZES, forests=18, reused_original_forests=6)
    artifacts = {}
    with ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(fit_size, n): n for n in SIZES}
        for future in as_completed(futures):
            artifacts.update(future.result())
            progress("sample_size_complete", completed_training_rows=futures[future])
    artifacts.update(reuse_original(manifest))
    verify_inputs(manifest)
    u.verify(artifacts)
    u.write(HERE / f"predictions_frozen_{DATE}.json", {"frozen_at": u.now(), "new_forests": 18,
            "original_forests_reused": 6, "input_manifest_sha256": u.sha(HERE / f"input_manifest_{DATE}.json"),
            "oracle_effects_and_probabilities_read_by_fit": False, "files": artifacts})
    progress("predictions_frozen", new_forests=18, original_forests_reused=6)


def prediction_files():
    return sorted((HERE / "fits").glob("n_*/*/seed_*/predictions.parquet")) + sorted((HERE / "original_cohort_on_new_test").glob("*/seed_*/predictions.parquet"))


def evaluate():
    manifest = u.read(HERE / f"input_manifest_{DATE}.json")
    frozen = u.read(HERE / f"predictions_frozen_{DATE}.json")
    assert u.sha(HERE / f"input_manifest_{DATE}.json") == frozen["input_manifest_sha256"]
    verify_inputs(manifest)
    u.verify(frozen["files"])
    ids = u.read(HERE / f"splits_{DATE}.json")["heldout_rows"]
    truth = pd.read_parquet(manifest["temporary_dataset"], columns=["true_ite_prob", "true_treatment_prob", "true_y0_prob", "true_y1_prob"]).iloc[ids]
    tau, e = truth.true_ite_prob.to_numpy(), truth.true_treatment_prob.to_numpy()
    m = (1 - e) * truth.true_y0_prob.to_numpy() + e * truth.true_y1_prob.to_numpy()
    rows = []
    for path in prediction_files():
        pred, audit = pd.read_parquet(path), u.read(path.parent / "fit_audit.json")
        assert pred._oci_row_id.tolist() == ids
        estimate = pred.estimated_cate.to_numpy()
        error = estimate - tau
        rows.append({"cohort": audit["cohort"], "training_rows": audit["training_rows"], "scenario": audit["scenario"],
                     "seed": audit["seed"], "correlation": float(np.corrcoef(estimate, tau)[0, 1]),
                     "rmse": float(np.sqrt(np.mean(error ** 2))), "mae": float(np.mean(np.abs(error))),
                     "bias": float(np.mean(error)), "prediction_sd": float(np.std(estimate)),
                     "coverage_95_of_individual_truth": float(np.mean((pred.lower_95 <= tau) & (tau <= pred.upper_95))),
                     "r_loss": float(np.mean((pred.outcome - pred.outcome_prediction - estimate * (pred.treatment - pred.propensity)) ** 2)),
                     "oracle_r_loss": float(np.mean((pred.outcome - m - estimate * (pred.treatment - e)) ** 2)),
                     "propensity_rmse": float(np.sqrt(np.mean((pred.propensity - e) ** 2))),
                     "outcome_nuisance_rmse": float(np.sqrt(np.mean((pred.outcome_prediction - m) ** 2))),
                     "median_leaves_per_tree": audit["tree_audit"]["leaves_per_tree"]["median"],
                     "median_effect_patients_per_leaf": audit["tree_audit"]["effect_patients_per_leaf"]["median"]})
    table = pd.DataFrame(rows)
    assert len(table) == 24
    table.to_csv(HERE / f"metrics_by_seed_{DATE}.csv", index=False)
    columns = [c for c in table if c not in ["cohort", "training_rows", "scenario", "seed"]]
    summary = table.groupby(["cohort", "training_rows", "scenario"])[columns].mean().reset_index()
    summary.to_csv(HERE / f"summary_metrics_{DATE}.csv", index=False)
    u.write(HERE / f"evaluation_{DATE}.json", {"evaluated_at": u.now(), "oracle_mean": float(tau.mean()),
            "oracle_sd": float(tau.std()), "test_rows": len(tau), "summaries": summary.to_dict("records"),
            "seed_ranges": [{"cohort": cohort, "training_rows": int(n), "scenario": scenario,
                              "correlation_min": float(part.correlation.min()), "correlation_max": float(part.correlation.max()),
                              "rmse_min": float(part.rmse.min()), "rmse_max": float(part.rmse.max())}
                             for (cohort, n, scenario), part in table.groupby(["cohort", "training_rows", "scenario"])]})
    verify_inputs(manifest)
    u.verify(frozen["files"])
    u.write(HERE / f"evaluation_validation_{DATE}.json", {"validated_at": u.now(),
            "all_source_and_artifact_hashes_verified": True, "same_20000_test_rows_each": True,
            "new_forests": 18, "original_forests_reused": 6})
    print(summary[["cohort", "training_rows", "scenario", "correlation", "rmse", "prediction_sd", "bias", "propensity_rmse", "outcome_nuisance_rmse"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["fit", "evaluate"])
    args = parser.parse_args()
    fit() if args.phase == "fit" else evaluate()
