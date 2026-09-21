"""Independently rescore and reload the 10%/45%/95% comparison; no refitting."""
from pathlib import Path
import importlib.util

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("sampling_experiment", HERE / "run_sampling_010_2026-09-21.py")
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)
u = e.u
manifest = u.read(HERE / f"input_manifest_{e.DATE}.json")
frozen = u.read(HERE / f"predictions_frozen_{e.DATE}.json")
assert u.sha(HERE / f"input_manifest_{e.DATE}.json") == frozen["input_manifest_sha256"]
e.verify_prior(manifest)
u.verify(frozen["files"])
original = u.read(e.BASE / f"input_manifest_{e.DATE}.json")
names = ["true_" + d["name"] for d in original["modifiers"] + original["confounders"]]
data = pd.read_parquet(u.DATA / "dataset.parquet", columns=names + ["true_ite_prob", "treatment_indicator", "outcome_indicator"])
train, test = manifest["training_rows"], manifest["heldout_rows"]
M, _, _, _ = u.encode(data, original["modifiers"], train)
C, _, _, _ = u.encode(data, original["confounders"], train)
assert u.array_hash(M[test]) == original["scenarios"]["c"]["test_X_sha256"]
truth = data.iloc[test].true_ite_prob.to_numpy()
initial = joblib.load(e.BASE / "fits/c/initial_model.joblib")
assert u.array_hash(*initial.residuals_[:2]) == manifest["residual_sha256"]
propensity, outcome = u.nuisance_predictions(initial, M[test], C[test])
metrics = pd.read_csv(HERE / f"metrics_by_seed_{e.DATE}.csv", float_precision="round_trip")
checks = []
for honesty in [True, False]:
    for seed in u.SEEDS:
        baseline = joblib.load(e.path_for(0.45, honesty, seed) / "forest.joblib")
        for fraction in e.FRACTIONS:
            dest = e.path_for(fraction, honesty, seed)
            model = baseline if fraction == 0.45 else joblib.load(dest / "forest.joblib")
            prediction = pd.read_csv(dest / "predictions.csv", float_precision="round_trip")
            assert prediction._oci_row_id.tolist() == test
            np.testing.assert_array_equal(prediction.treatment, data.iloc[test].treatment_indicator)
            np.testing.assert_array_equal(prediction.outcome, data.iloc[test].outcome_indicator)
            np.testing.assert_allclose(prediction.propensity, propensity, rtol=0, atol=1e-14)
            np.testing.assert_allclose(prediction.outcome_prediction, outcome, rtol=0, atol=1e-14)
            estimate = model.predict(M[test]).ravel()
            np.testing.assert_array_equal(estimate, prediction.estimated_cate.to_numpy())
            assert model.get_params() == {**baseline.get_params(), "max_samples": fraction}
            scored = {"correlation": float(pearsonr(estimate, truth).statistic),
                      "rmse": float(root_mean_squared_error(truth, estimate)),
                      "mae": float(mean_absolute_error(truth, estimate)),
                      "bias": float((estimate - truth).mean()), "prediction_sd": float(estimate.std()),
                      "r_loss": float(np.square(prediction.outcome - prediction.outcome_prediction -
                                                  estimate * (prediction.treatment - prediction.propensity)).mean())}
            row = metrics[(metrics.seed == seed) & (metrics.honesty == honesty) & (metrics.sample_fraction == fraction)]
            assert len(row) == 1
            for name, value in scored.items():
                np.testing.assert_allclose(row.iloc[0][name], value, rtol=0, atol=1e-12)
            tree = e.tree_audit(model)
            assert tree["effect_patients_per_leaf"]["min"] >= 10
            for name, value in [("median_leaves_per_tree", tree["leaves_per_tree"]["median"]),
                                ("median_effect_patients_per_leaf", tree["effect_patients_per_leaf"]["median"]),
                                ("median_depth", tree["depth"]["median"]), ("unsplit_trees", tree["unsplit_trees"])]:
                assert row.iloc[0][name] == value
            checks.append({"sample_fraction": fraction, "honesty": honesty, "seed": seed,
                           "max_reloaded_prediction_error": 0.0, "recomputed_metrics": scored,
                           "actual_tree_sample_counts": tree["sample_split_effect_counts"],
                           "distinct_patients_used_for_splits": tree["distinct_patients_used_for_splits"],
                           "distinct_patients_used_for_effects": tree["distinct_patients_used_for_effects"]})
assert len(checks) == 18
e.verify_prior(manifest)
u.verify(frozen["files"])
u.write(HERE / f"independent_validation_{e.DATE}.json", {
    "validated_at": u.now(), "models_reloaded_and_metrics_recomputed": 18,
    "max_reloaded_prediction_error": 0.0, "nuisance_predictions_match_original_fitted_models": True,
    "observed_treatment_and_outcome_match_dataset": True, "only_sampling_fraction_changed": True,
    "all_source_and_artifact_hashes_verified": True,
    "validation_script_sha256": u.sha(Path(__file__)), "fits": checks})
print("Validated all 18 saved models, metrics, original nuisances, tree samples, and artifact hashes.", flush=True)
