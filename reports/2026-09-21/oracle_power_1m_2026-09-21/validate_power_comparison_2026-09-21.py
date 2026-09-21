"""Independently reload models, check nuisance cross-fitting, and score effects."""
from pathlib import Path
import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("million_experiment", HERE / "run_power_comparison_2026-09-21.py")
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
split = np.load(HERE / f"splits_{e.DATE}.npz")
train_ids, test_ids = split["train"], split["test"]
assert np.intersect1d(train_ids, test_ids).size == 0
assert np.array_equal(np.sort(np.concatenate([train_ids, test_ids])), np.arange(1_000_000))
old_splits = u.read(e.PREVIOUS / f"splits_{e.DATE}.json")
assert np.array_equal(test_ids[:20_000], old_splits["heldout_rows"])
assert np.isin(old_splits["training_rows"]["80000"], train_ids).all()
data = pd.read_parquet(manifest["temporary_dataset"])
test = data.iloc[test_ids]
dest = HERE / "fits" / f"n_{e.N}"
design = u.read(dest / "design.json")
C = e.p.fixed_encode(test, manifest["confounders"], design["encoding"])
M = e.p.fixed_encode(test, manifest["modifiers"], design["encoding"])
assert u.array_hash(M) == design["test_modifiers_sha256"]
assert u.array_hash(np.column_stack([C, M])) == design["test_all_sha256"]
audit = u.read(dest / "nuisance_audit.json")
native = joblib.load(audit["initial_model_path"])
epred, mpred = u.nuisance_predictions(native, M, C)
yres, tres, Xcached, Wcached = native.residuals_
assert u.array_hash(yres, tres, epred, mpred) == audit["residual_sha256"]
train = data.iloc[train_ids]
np.testing.assert_array_equal(Xcached, e.p.fixed_encode(train, manifest["modifiers"], design["encoding"]))
np.testing.assert_array_equal(Wcached, e.p.fixed_encode(train, manifest["confounders"], design["encoding"]))
Z = np.column_stack([Xcached, Wcached])
oof_e, oof_m = np.empty(e.N), np.empty(e.N)
for index in range(5):
    heldout = split[f"inner_heldout_{index}"]
    for role, clones in [("treatment", native.models_t[0]), ("outcome", native.models_y[0])]:
        fitted = clones[index]
        assert fitted.get_params() == manifest["nuisance_parameters"]
        assert fitted.fit_audit() == audit["models"][role][index]
        assert not fitted.fit_audit()["optimization"]["iteration_limit_reached"]
        assert fitted.n_features_in_ == 25
    oof_e[heldout] = native.models_t[0][index].predict_proba(Z[heldout])[:, 1]
    oof_m[heldout] = native.models_y[0][index].predict_proba(Z[heldout])[:, 1]
np.testing.assert_allclose(train.treatment_indicator.to_numpy() - oof_e, tres.ravel(), rtol=0, atol=1e-12)
np.testing.assert_allclose(train.outcome_indicator.to_numpy() - oof_m, yres.ravel(), rtol=0, atol=1e-12)
saved_nuisances = pd.read_parquet(dest / "training_nuisances.parquet")
np.testing.assert_array_equal(saved_nuisances._oci_row_id, train_ids)
np.testing.assert_allclose(saved_nuisances.propensity, oof_e, rtol=0, atol=1e-12)
np.testing.assert_allclose(saved_nuisances.outcome_prediction, oof_m, rtol=0, atol=1e-12)
del native, Z, Xcached, Wcached, data, train
print("Verified all 800,000 out-of-fold nuisance predictions and all 10 fitted nuisance clones.", flush=True)
table = pd.read_csv(HERE / f"metrics_by_seed_{e.DATE}.csv", float_precision="round_trip")
def validate_one(path):
    pred = pd.read_parquet(path)
    fit = u.read(path.parent / "fit_audit.json")
    scenario, seed = fit["scenario"], fit["seed"]
    np.testing.assert_array_equal(pred._oci_row_id, test_ids)
    np.testing.assert_array_equal(pred.treatment, test.treatment_indicator)
    np.testing.assert_array_equal(pred.outcome, test.outcome_indicator)
    np.testing.assert_array_equal(pred.propensity, epred)
    np.testing.assert_array_equal(pred.outcome_prediction, mpred)
    X = M if scenario == "modifiers" else np.column_stack([C, M])
    forest = joblib.load(fit["model_path"])
    for key, value in {**manifest["forest_parameters"], "random_state": seed}.items():
        assert forest.get_params()[key] == value
    estimated, lower, upper = [np.asarray(value).ravel() for value in forest.predict(X, interval=True)]
    for actual, saved in [(estimated, pred.estimated_cate), (lower, pred.lower_95), (upper, pred.upper_95)]:
        np.testing.assert_array_equal(actual, saved)
    for test_set, count in [("full_200k", 200_000), ("common_20k", 20_000)]:
        truth = test.iloc[:count]
        tau = truth.true_ite_prob.to_numpy()
        etrue = truth.true_treatment_prob.to_numpy()
        mtrue = (1 - etrue) * truth.true_y0_prob.to_numpy() + etrue * truth.true_y1_prob.to_numpy()
        estimate = estimated[:count]
        Y, T = truth.outcome_indicator.to_numpy(), truth.treatment_indicator.to_numpy()
        scores = {"correlation": float(pearsonr(estimate, tau).statistic),
                  "rmse": float(root_mean_squared_error(tau, estimate)),
                  "mae": float(mean_absolute_error(tau, estimate)),
                  "bias": float(np.mean(estimate - tau)), "prediction_sd": float(np.std(estimate)),
                  "coverage_95_of_individual_truth": float(np.mean((lower[:count] <= tau) & (tau <= upper[:count]))),
                  "r_loss": float(np.mean((Y - mpred[:count] - estimate * (T - epred[:count])) ** 2)),
                  "oracle_r_loss": float(np.mean((Y - mtrue - estimate * (T - etrue)) ** 2)),
                  "propensity_rmse": float(root_mean_squared_error(etrue, epred[:count])),
                  "outcome_nuisance_rmse": float(root_mean_squared_error(mtrue, mpred[:count]))}
        row = table[(table.training_rows == e.N) & (table.scenario == scenario) &
                    (table.seed == seed) & (table.test_set == test_set)]
        assert len(row) == 1
        for key, value in scores.items():
            np.testing.assert_allclose(row.iloc[0][key], value, rtol=0, atol=1e-12)
    assert e.p.tree_audit(forest, e.N) == fit["tree_audit"]
    return {"scenario": scenario, "seed": seed, "maximum_prediction_and_interval_error": 0.0,
            "metrics_independently_recomputed_on_both_test_sets": True}


checks = []
with ThreadPoolExecutor(max_workers=3) as pool:
    futures = [pool.submit(validate_one, path) for path in e.prediction_files()]
    for future in as_completed(futures):
        result = future.result()
        checks.append(result)
        print(json.dumps({"validated_forests": len(checks), "total": 6,
                          "scenario": result["scenario"], "seed": result["seed"]}), flush=True)
assert len(checks) == 6
e.verify_inputs(manifest)
u.verify(frozen["files"])
u.write(HERE / f"independent_validation_{e.DATE}.json", {"validated_at": u.now(),
        "saved_forests_reloaded": 6, "full_test_predictions_and_intervals_per_forest": 200_000,
        "maximum_prediction_and_interval_error": 0.0, "nuisance_clones_checked": 10,
        "out_of_fold_nuisance_predictions_independently_reproduced": 800_000,
        "all_source_and_artifact_hashes_verified": True, "previous_splits_preserved": True,
        "fits": checks, "validation_script_sha256": u.sha(Path(__file__))})
print("All six forests, full test predictions/intervals, both sets of scores, nuisance models, and hashes verified.", flush=True)
