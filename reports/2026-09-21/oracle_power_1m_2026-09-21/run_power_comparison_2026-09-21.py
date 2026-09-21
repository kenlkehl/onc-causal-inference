"""Historical oracle forest at 800,000 training / 200,000 test patients.

Two X/W designs and three seeds share cross-fitted estimated nuisances.
The original 20K test set remains held out for the extended learning curve.
Fit reads only true covariates and observed treatment/outcome, never oracle
effect or probability labels. Evaluation follows prediction freezing.
"""
from pathlib import Path
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib.util
import inspect
import json
import multiprocessing
import time
import warnings

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "oracle_power_100k_2026-09-21"
spec = importlib.util.spec_from_file_location("previous_power_experiment", PREVIOUS / "run_power_comparison_2026-09-21.py")
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
u = p.u
import joblib
import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from econml.grf import CausalForest
from oci.models.elastic_net_nuisance import ElasticNetLogisticClassifier

DATE = "2026-09-21"
N = 800_000


def progress(phase, key=None, **details):
    value = {"updated_at": u.now(), "phase": phase, **details}
    u.write(HERE / ("status.json" if key is None else f"status_{key}.json"), value)
    print(json.dumps(value), flush=True)


def verify_inputs(manifest):
    u.verify(manifest["sources"])
    generation = u.read(HERE / f"generation_manifest_{DATE}.json")
    u.verify(generation["sources"])
    u.verify(generation["files"])
    u.verify(generation["previous_generation_files"])
    previous_frozen = u.read(PREVIOUS / f"predictions_frozen_{DATE}.json")
    u.verify(previous_frozen["files"])


def paths_hash(paths):
    return {str(path.resolve()): u.sha(path) for path in paths}


def prepare(manifest):
    dest = HERE / "fits" / f"n_{N}"
    temporary = Path(manifest["temporary_directory"]) / "models"
    dest.mkdir(parents=True, exist_ok=True)
    temporary.mkdir(parents=True, exist_ok=True)
    checkpoint = dest / "prepared.json"
    if checkpoint.exists():
        result = u.read(checkpoint)
        u.verify(result["files"])
        return result
    split = np.load(HERE / f"splits_{DATE}.npz")
    train, test = split["train"], split["test"]
    definitions = manifest["confounders"] + manifest["modifiers"]
    data = pd.read_parquet(manifest["temporary_dataset"], columns=[
        *["true_" + item["name"] for item in definitions], "treatment_indicator", "outcome_indicator"])
    C, cnames, _, caudit = u.encode(data, manifest["confounders"], train)
    M, mnames, _, maudit = u.encode(data, manifest["modifiers"], train)
    T, Y = data.treatment_indicator.to_numpy(), data.outcome_indicator.to_numpy()
    cv = []
    for index in range(5):
        heldout = split[f"inner_heldout_{index}"]
        mask = np.ones(N, dtype=bool)
        mask[heldout] = False
        cv.append((np.flatnonzero(mask), heldout))
    assert np.array_equal(np.sort(np.concatenate([b for _, b in cv])), np.arange(N))
    matrices = {"train_modifiers": M[train], "train_all": np.column_stack([C[train], M[train]]),
                "test_modifiers": M[test], "test_all": np.column_stack([C[test], M[test]])}
    design = {"training_rows": N, "encoding": caudit + maudit, "confounder_columns": cnames,
              "modifier_columns": mnames, **{key + "_sha256": u.array_hash(value) for key, value in matrices.items()}}
    u.write(dest / "design.json", design)
    arrays = []
    for name, matrix in matrices.items():
        path = temporary / f"{name}.npy"
        np.save(path, matrix)
        arrays.append(path)
    progress("fitting_nuisances_and_initial_forest", training_rows=N, heldout_rows=len(test))
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
    audit = {"seconds_including_initial_forest": time.monotonic() - started,
             "residual_sha256": residual_hash, "initial_model_path": str(temporary / "initial_model.joblib"),
             "residuals_path": str(temporary / "residuals.npz"),
             "models": {role: [fitted.fit_audit() for fitted in clones[0]]
                        for role, clones in [("treatment", model.models_t), ("outcome", model.models_y)]},
             "warnings": [{"type": key[0], "message": key[1], "count": value} for key, value in warning_counts.items()]}
    joblib.dump(model, temporary / "initial_model.joblib", compress=3)
    u.write(dest / "nuisance_audit.json", audit)
    prediction_base = pd.DataFrame({"_oci_row_id": test, "treatment": T[test], "outcome": Y[test],
                                    "propensity": nuisance_e, "outcome_prediction": nuisance_m})
    prediction_base.to_parquet(temporary / "prediction_base.parquet", index=False)
    initial = model.model_cate.estimators_[0]
    u.write(temporary / "forest_parameters.json", initial.get_params())
    files = paths_hash([dest / "design.json", dest / "training_nuisances.parquet", dest / "nuisance_audit.json",
                        temporary / "initial_model.joblib", temporary / "residuals.npz",
                        temporary / "prediction_base.parquet", temporary / "forest_parameters.json", *arrays])
    details = {"cohort": "extended", "training_rows": N, "scenario": "modifiers", "seed": u.SEEDS[0],
               "residual_sha256": residual_hash, "nuisances_shared_across_X_configurations": True,
               "native_initial_forest_reused": True, "warnings": audit["warnings"]}
    target = dest / "modifiers" / f"seed_{u.SEEDS[0]}"
    files.update(p.save_forest(initial, M[test], prediction_base, target,
                              temporary / "modifiers" / f"seed_{u.SEEDS[0]}.joblib", details, N))
    check = u.read(target / "fit_audit.json")
    assert abs(check["r_loss"] - model.score(Y[test], T[test], X=M[test], W=C[test])) < 1e-12
    prepared = {"prepared_at": u.now(), "files": files, "residual_sha256": residual_hash}
    u.write(checkpoint, prepared)
    progress("nuisances_and_first_forest_complete", seconds=audit["seconds_including_initial_forest"],
             warnings=sum(warning_counts.values()), completed_forests=1, total_forests=6)
    return prepared


def fit_one(scenario, seed):
    manifest = u.read(HERE / f"input_manifest_{DATE}.json")
    temporary = Path(manifest["temporary_directory"]) / "models"
    target = HERE / "fits" / f"n_{N}" / scenario / f"seed_{seed}"
    completion = target / "completed.json"
    if completion.exists():
        value = u.read(completion)
        u.verify(value["files"])
        return value["files"]
    key = f"{scenario}_{seed}"
    progress("fitting_forest", key=key, scenario=scenario, seed=seed)
    started = time.monotonic()
    Xtrain = np.load(temporary / f"train_{scenario}.npy", mmap_mode="r")
    Xtest = np.load(temporary / f"test_{scenario}.npy", mmap_mode="r")
    residuals = np.load(temporary / "residuals.npz")
    yres, tres = residuals["yres"], residuals["tres"]
    residual_hash = u.array_hash(yres, tres, residuals["heldout_propensity"], residuals["heldout_outcome_prediction"])
    assert residual_hash == u.read(HERE / "fits" / f"n_{N}" / "nuisance_audit.json")["residual_sha256"]
    parameters = {**u.read(temporary / "forest_parameters.json"), "random_state": seed}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        forest = CausalForest(**parameters).fit(Xtrain, tres, yres)
    assert forest.get_params() == parameters
    details = {"cohort": "extended", "training_rows": N, "scenario": scenario, "seed": seed,
               "residual_sha256": residual_hash, "nuisances_shared_across_X_configurations": True,
               "native_initial_forest_reused": False, "warnings": [str(w.message) for w in caught],
               "fit_seconds": time.monotonic() - started}
    progress("saving_forest", key=key, scenario=scenario, seed=seed, seconds=details["fit_seconds"])
    files = p.save_forest(forest, Xtest, pd.read_parquet(temporary / "prediction_base.parquet"), target,
                          temporary / scenario / f"seed_{seed}.joblib", details, N)
    u.write(completion, {"completed_at": u.now(), "files": files})
    progress("forest_complete", key=key, scenario=scenario, seed=seed, seconds=time.monotonic() - started)
    return files


def fit():
    if (HERE / f"predictions_frozen_{DATE}.json").exists():
        raise ValueError("Already frozen; evaluate instead")
    generation = u.read(HERE / f"generation_manifest_{DATE}.json")
    previous = u.read(PREVIOUS / f"input_manifest_{DATE}.json")
    input_path = HERE / f"input_manifest_{DATE}.json"
    if input_path.exists():
        manifest = u.read(input_path)
    else:
        sources = [Path(__file__), HERE / f"generation_manifest_{DATE}.json", PREVIOUS / f"input_manifest_{DATE}.json",
                   PREVIOUS / f"predictions_frozen_{DATE}.json", PREVIOUS / f"run_power_comparison_{DATE}.py",
                   p.BASE / "run_true_oracle_xw_comparison.py", u.ROOT / "oci/models/elastic_net_nuisance.py",
                   Path(inspect.getfile(CausalForestDML)), Path(inspect.getfile(CausalForest)),
                   Path(inspect.getfile(CausalForest.fit))]
        manifest = {"created_at": u.now(), "purpose": __doc__, "training_rows": N, "heldout_rows": 200_000,
                    "common_test_rows": 20_000, "temporary_dataset": generation["temporary_dataset"],
                    "temporary_directory": generation["temporary_directory"],
                    "confounders": previous["confounders"], "modifiers": previous["modifiers"],
                    "scenarios": previous["scenarios"], "nuisance_parameters": u.NUISANCE,
                    "forest_parameters": {**u.FOREST, "max_features": "sqrt"}, "forest_seeds": u.SEEDS,
                    "five_crossfit_folds": True, "propensity_filtering": False,
                    "oracle_effects_or_probabilities_used_for_fitting": False,
                    "nuisance_policy": "One native DML fit; identical estimated residuals shared across six forests.",
                    "parallelism": "Up to three independent final forests; every estimator retains n_jobs=1.",
                    "versions": previous["versions"], "sources": paths_hash(sources)}
        u.write(input_path, manifest)
    verify_inputs(manifest)
    prepared = prepare(manifest)
    files = prepared["files"].copy()
    jobs = [(scenario, seed) for scenario in ("modifiers", "all") for seed in u.SEEDS
            if not (scenario == "modifiers" and seed == u.SEEDS[0])]
    progress("fitting_remaining_forests", remaining_forests=len(jobs), completed_forests=1)
    with ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(fit_one, scenario, seed): (scenario, seed) for scenario, seed in jobs}
        completed = 1
        for future in as_completed(futures):
            files.update(future.result())
            completed += 1
            progress("forest_completed", scenario=futures[future][0], seed=futures[future][1],
                     completed_forests=completed, total_forests=6)
    verify_inputs(manifest)
    u.verify(files)
    u.write(HERE / f"predictions_frozen_{DATE}.json", {"frozen_at": u.now(), "new_forests": 6,
            "input_manifest_sha256": u.sha(input_path), "oracle_effects_and_probabilities_read_by_fit": False,
            "files": files})
    progress("predictions_frozen", new_forests=6)


def prediction_files():
    return sorted((HERE / "fits" / f"n_{N}").glob("*/seed_*/predictions.parquet"))


def calculate(pred, truth):
    tau, e = truth.true_ite_prob.to_numpy(), truth.true_treatment_prob.to_numpy()
    m = (1 - e) * truth.true_y0_prob.to_numpy() + e * truth.true_y1_prob.to_numpy()
    estimate = pred.estimated_cate.to_numpy()
    error = estimate - tau
    return {"correlation": float(np.corrcoef(estimate, tau)[0, 1]),
            "rmse": float(np.sqrt(np.mean(error ** 2))), "mae": float(np.mean(np.abs(error))),
            "bias": float(np.mean(error)), "prediction_sd": float(np.std(estimate)),
            "coverage_95_of_individual_truth": float(np.mean((pred.lower_95 <= tau) & (tau <= pred.upper_95))),
            "r_loss": float(np.mean((pred.outcome - pred.outcome_prediction - estimate * (pred.treatment - pred.propensity)) ** 2)),
            "oracle_r_loss": float(np.mean((pred.outcome - m - estimate * (pred.treatment - e)) ** 2)),
            "propensity_rmse": float(np.sqrt(np.mean((pred.propensity - e) ** 2))),
            "outcome_nuisance_rmse": float(np.sqrt(np.mean((pred.outcome_prediction - m) ** 2)))}


def evaluate():
    manifest = u.read(HERE / f"input_manifest_{DATE}.json")
    frozen = u.read(HERE / f"predictions_frozen_{DATE}.json")
    assert u.sha(HERE / f"input_manifest_{DATE}.json") == frozen["input_manifest_sha256"]
    verify_inputs(manifest)
    u.verify(frozen["files"])
    split = np.load(HERE / f"splits_{DATE}.npz")
    data = pd.read_parquet(manifest["temporary_dataset"], columns=[
        "true_ite_prob", "true_treatment_prob", "true_y0_prob", "true_y1_prob"])
    test_truth = data.iloc[split["test"]].reset_index(drop=True)
    common_truth = data.iloc[split["common_test"]].reset_index(drop=True)
    rows = []
    for path in prediction_files():
        pred, audit = pd.read_parquet(path), u.read(path.parent / "fit_audit.json")
        assert np.array_equal(pred._oci_row_id, split["test"])
        for label, count, truth in [("full_200k", 200_000, test_truth), ("common_20k", 20_000, common_truth)]:
            rows.append({"cohort": "extended", "training_rows": N, "test_set": label, "test_rows": count,
                         "scenario": audit["scenario"], "seed": audit["seed"],
                         **calculate(pred.iloc[:count], truth),
                         "median_leaves_per_tree": audit["tree_audit"]["leaves_per_tree"]["median"],
                         "median_effect_patients_per_leaf": audit["tree_audit"]["effect_patients_per_leaf"]["median"]})
    previous_scores = pd.read_csv(PREVIOUS / f"metrics_by_seed_{DATE}.csv", float_precision="round_trip")
    for path in sorted((PREVIOUS / "fits").glob("n_*/*/seed_*/predictions.parquet")):
        pred, audit = pd.read_parquet(path), u.read(path.parent / "fit_audit.json")
        assert np.array_equal(pred._oci_row_id, split["common_test"])
        scores = calculate(pred, common_truth)
        previous_row = previous_scores[(previous_scores.cohort == "new") &
            (previous_scores.training_rows == audit["training_rows"]) &
            (previous_scores.scenario == audit["scenario"]) & (previous_scores.seed == audit["seed"])]
        assert len(previous_row) == 1
        for key, value in scores.items():
            np.testing.assert_allclose(previous_row.iloc[0][key], value, rtol=0, atol=1e-12)
        rows.append({"cohort": "previous", "training_rows": audit["training_rows"], "test_set": "common_20k",
                     "test_rows": 20_000, "scenario": audit["scenario"], "seed": audit["seed"], **scores,
                     "median_leaves_per_tree": audit["tree_audit"]["leaves_per_tree"]["median"],
                     "median_effect_patients_per_leaf": audit["tree_audit"]["effect_patients_per_leaf"]["median"]})
    table = pd.DataFrame(rows)
    assert len(table) == 30
    table.to_csv(HERE / f"metrics_by_seed_{DATE}.csv", index=False)
    keys = ["cohort", "training_rows", "test_set", "test_rows", "scenario"]
    summary = table.groupby(keys)[[key for key in table if key not in keys + ["seed"]]].mean().reset_index()
    summary.to_csv(HERE / f"summary_metrics_{DATE}.csv", index=False)
    u.write(HERE / f"evaluation_{DATE}.json", {"evaluated_at": u.now(), "summaries": summary.to_dict("records"),
            "test_truth": {name: {"mean": float(frame.true_ite_prob.mean()),
                                  "sd": float(frame.true_ite_prob.to_numpy().std())}
                           for name, frame in [("full_200k", test_truth), ("common_20k", common_truth)]},
            "seed_ranges": [{**dict(zip(keys, group)), "correlation_min": float(part.correlation.min()),
                             "correlation_max": float(part.correlation.max()), "rmse_min": float(part.rmse.min()),
                             "rmse_max": float(part.rmse.max())} for group, part in table.groupby(keys)]})
    verify_inputs(manifest)
    u.verify(frozen["files"])
    progress("evaluated", forests=6, common_test_learning_curve_sizes=[800, 8000, 80000, N])
    print(summary[["training_rows", "test_set", "scenario", "correlation", "rmse", "bias", "prediction_sd"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["fit", "evaluate"])
    arguments = parser.parse_args()
    fit() if arguments.phase == "fit" else evaluate()
