"""Reload every forest and independently verify scores and nuisance predictions."""
from pathlib import Path
import importlib.util
import json

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("power_experiment", HERE / "run_power_comparison_2026-09-21.py")
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)
u = e.u
import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

manifest = u.read(HERE / f"input_manifest_{e.DATE}.json")
frozen = u.read(HERE / f"predictions_frozen_{e.DATE}.json")
assert u.sha(HERE / f"input_manifest_{e.DATE}.json") == frozen["input_manifest_sha256"]
e.verify_inputs(manifest)
u.verify(frozen["files"])
split = u.read(HERE / f"splits_{e.DATE}.json")
ids = split["heldout_rows"]
data = pd.read_parquet(manifest["temporary_dataset"])
test = data.iloc[ids]
truth = test.true_ite_prob.to_numpy()
etrue = test.true_treatment_prob.to_numpy()
mtrue = (1 - etrue) * test.true_y0_prob.to_numpy() + etrue * test.true_y1_prob.to_numpy()
table = pd.read_csv(HERE / f"metrics_by_seed_{e.DATE}.csv", float_precision="round_trip")
designs, nuisance_predictions, nuisance_checks = {}, {}, []
original = u.read(e.BASE / f"input_manifest_{e.DATE}.json")
for n in e.SIZES:
    design = u.read(HERE / "fits" / f"n_{n}" / "design.json")
    C = e.fixed_encode(test, manifest["confounders"], design["encoding"])
    M = e.fixed_encode(test, manifest["modifiers"], design["encoding"])
    assert u.array_hash(M) == design["test_modifiers_sha256"]
    assert u.array_hash(np.column_stack([C, M])) == design["test_all_sha256"]
    designs[("new", n)] = (C, M)
    audit = u.read(HERE / "fits" / f"n_{n}" / "nuisance_audit.json")
    native = joblib.load(audit["initial_model_path"])
    propensity, outcome = u.nuisance_predictions(native, M, C)
    yres, tres, Xcached, Wcached = native.residuals_
    assert u.array_hash(yres, tres, propensity, outcome) == audit["residual_sha256"]
    train = data.iloc[split["training_rows"][str(n)]]
    np.testing.assert_array_equal(Xcached, e.fixed_encode(train, manifest["modifiers"], design["encoding"]))
    np.testing.assert_array_equal(Wcached, e.fixed_encode(train, manifest["confounders"], design["encoding"]))
    for role, clones in [("treatment", native.models_t[0]), ("outcome", native.models_y[0])]:
        for index, fitted in enumerate(clones):
            assert fitted.fit_audit() == audit["models"][role][index]
            assert not fitted.fit_audit()["optimization"]["iteration_limit_reached"]
    nuisance_predictions[("new", n, "all")] = (propensity, outcome)
    nuisance_predictions[("new", n, "modifiers")] = (propensity, outcome)
    nuisance_checks.append({"training_rows": n, "fitted_clones_verified": 10,
                            "iteration_limits_reached": 0, "warnings": audit["warnings"]})
    del native
C = e.fixed_encode(test, original["confounders"], original["encoding"])
M = e.fixed_encode(test, original["modifiers"], original["encoding"])
designs[("original", 800)] = (C, M)
for scenario, old in [("modifiers", "c"), ("all", "a")]:
    X, W = (M, C) if scenario == "modifiers" else (np.column_stack([C, M]), None)
    native = joblib.load(e.BASE / "fits" / old / "initial_model.joblib")
    nuisance_predictions[("original", 800, scenario)] = u.nuisance_predictions(native, X, W)

checks = []
for path in e.prediction_files():
    pred = pd.read_parquet(path)
    audit = u.read(path.parent / "fit_audit.json")
    cohort, n, scenario, seed = [audit[k] for k in ["cohort", "training_rows", "scenario", "seed"]]
    assert pred._oci_row_id.tolist() == ids
    assert set(split["training_rows"][str(n)]).isdisjoint(ids)
    np.testing.assert_array_equal(pred.treatment, test.treatment_indicator)
    np.testing.assert_array_equal(pred.outcome, test.outcome_indicator)
    epred, mpred = nuisance_predictions[(cohort, n, scenario)]
    np.testing.assert_array_equal(pred.propensity, epred)
    np.testing.assert_array_equal(pred.outcome_prediction, mpred)
    C, M = designs[(cohort, n)]
    X = M if scenario == "modifiers" else np.column_stack([C, M])
    model = joblib.load(audit["model_path"])
    parameters = model.get_params()
    for key, value in {**manifest["forest_parameters"], "random_state": seed}.items():
        assert parameters[key] == value, (key, parameters[key], value)
    estimated, lower, upper = [np.asarray(v).ravel() for v in model.predict(X, interval=True)]
    for calculated, saved in [(estimated, pred.estimated_cate), (lower, pred.lower_95), (upper, pred.upper_95)]:
        np.testing.assert_array_equal(calculated, saved)
    scores = {"correlation": float(pearsonr(estimated, truth).statistic),
              "rmse": float(root_mean_squared_error(truth, estimated)),
              "mae": float(mean_absolute_error(truth, estimated)),
              "bias": float(np.mean(estimated - truth)), "prediction_sd": float(np.std(estimated)),
              "coverage_95_of_individual_truth": float(np.mean((lower <= truth) & (truth <= upper))),
              "r_loss": float(np.mean(np.square(pred.outcome - mpred - estimated * (pred.treatment - epred)))),
              "oracle_r_loss": float(np.mean(np.square(pred.outcome - mtrue - estimated * (pred.treatment - etrue)))),
              "propensity_rmse": float(root_mean_squared_error(etrue, epred)),
              "outcome_nuisance_rmse": float(root_mean_squared_error(mtrue, mpred))}
    row = table[(table.cohort == cohort) & (table.training_rows == n) & (table.scenario == scenario) & (table.seed == seed)]
    assert len(row) == 1
    for key, value in scores.items():
        np.testing.assert_allclose(row.iloc[0][key], value, rtol=0, atol=1e-12)
    assert e.tree_audit(model, n) == audit["tree_audit"]
    checks.append({"cohort": cohort, "training_rows": n, "scenario": scenario, "seed": seed,
                   "maximum_prediction_and_interval_error": 0.0, "metrics_recomputed": True})
    print(json.dumps({"validated_model": len(checks), "total": 24, "cohort": cohort,
                      "training_rows": n, "scenario": scenario, "seed": seed}), flush=True)
assert len(checks) == 24
e.verify_inputs(manifest)
u.verify(frozen["files"])
u.verify(u.read(e.BASE / f"predictions_frozen_{e.DATE}.json")["files"])
u.write(HERE / f"independent_validation_{e.DATE}.json", {
    "validated_at": u.now(), "saved_forests_reloaded": 24, "metrics_recomputed": 24,
    "maximum_prediction_and_interval_error": 0.0, "nuisance_clones_checked": 30,
    "identical_nuisance_predictions_across_X_configurations": True,
    "all_source_and_artifact_hashes_verified": True, "nuisance_checks": nuisance_checks,
    "fits": checks, "validation_script_sha256": u.sha(Path(__file__))})
print("All 24 forests, predictions, intervals, scores, sample counts, 30 nuisance clones, and hashes verified.", flush=True)
