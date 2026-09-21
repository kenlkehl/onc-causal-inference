"""Independently reproduce CV predictions, selected penalties, and final effect scores."""
from pathlib import Path
import importlib.util
import json

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("extracted_logistic_experiment", HERE / "run_logistic_interactions_2026-09-21.py")
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import pearsonr
from sklearn.metrics import log_loss, mean_absolute_error, root_mean_squared_error, roc_auc_score

manifest = e.read(HERE / f"input_manifest_{e.DATE}.json")
frozen = e.read(HERE / f"predictions_frozen_{e.DATE}.json")
assert e.sha(HERE / f"input_manifest_{e.DATE}.json") == frozen["input_manifest_sha256"]
e.verify_inputs(manifest)
e.verify(frozen["files"])
definitions, training, heldout, labels, test_labels, split = e.load_inputs()
assert set(labels._oci_row_id).isdisjoint(test_labels._oci_row_id)
cv_checks = []
for cohort in e.COHORTS:
    mask = np.ones(len(training), bool) if cohort == "all_800" else labels.effect_eligible.to_numpy(bool)
    for penalty in e.PENALTIES:
        losses, convergence, validation_ids = [], [], []
        for fold_index in range(1, 6):
            destination = HERE / "cv" / cohort / penalty / f"fold_{fold_index}"
            audit = e.read(destination / "audit.json")
            path = np.load(destination / "path.npz")
            itr = np.flatnonzero(mask & labels._oci_row_id.isin(split["inner_splits"][fold_index - 1]["fit_row_ids"]).to_numpy())
            iva = np.flatnonzero(mask & labels._oci_row_id.isin(split["inner_splits"][fold_index - 1]["heldout_row_ids"]).to_numpy())
            assert labels.iloc[itr]._oci_row_id.tolist() == audit["training_row_ids"]
            assert labels.iloc[iva]._oci_row_id.tolist() == audit["validation_row_ids"]
            np.testing.assert_array_equal(path["valid_row_ids"], audit["validation_row_ids"])
            validation_ids.extend(audit["validation_row_ids"])
            encoder = joblib.load(destination / "encoder.joblib")
            independent = e._FeatureEncoder(definitions).fit(training.iloc[itr].reset_index(drop=True))
            assert encoder.encodings == independent.encodings
            X = independent.transform(training.iloc[iva].reset_index(drop=True))
            T = labels.iloc[iva].treatment.to_numpy(float)
            Z = np.column_stack([T, X, X * T[:, None]])
            reconstructed = expit(Z @ path["coefficients"].T + path["intercepts"])
            np.testing.assert_allclose(reconstructed.T, path["predictions"], rtol=0, atol=1e-12)
            fold_losses = [float(log_loss(labels.iloc[iva].outcome, values, labels=[0, 1])) for values in reconstructed.T]
            np.testing.assert_allclose(fold_losses, [row["validation_log_loss"] for row in audit["path"]], rtol=0, atol=1e-12)
            losses.append(fold_losses)
            convergence.append([row["converged"] for row in audit["path"]])
        assert sorted(validation_ids) == sorted(labels.loc[mask, "_oci_row_id"].tolist())
        losses = np.asarray(losses).mean(axis=0)
        allowed = np.asarray(convergence).all(axis=0)
        selected_index = min(np.flatnonzero(allowed), key=lambda index: (losses[index], e.CS[index]))
        selected = e.read(HERE / "fits" / cohort / penalty / "selection.json")
        assert float(e.CS[selected_index]) == selected["selected_C"]
        np.testing.assert_allclose(losses, [row["mean_validation_log_loss"] for row in selected["path"]], rtol=0, atol=1e-12)
        cv_checks.append({"cohort": cohort, "penalty": penalty, "selected_C": selected["selected_C"],
                          "all_five_inner_encoders_fit_on_inner_training_only": True,
                          "all_65_validation_predictions_and_losses_recomputed": True})
print("All 260 CV predictions/losses, 20 inner encoders, and four selected penalties verified.", flush=True)

evaluation = e.read(HERE / f"evaluation_{e.DATE}.json")
assert e.sha(evaluation["oracle_path"]) == evaluation["oracle_sha256"]
truth = pd.read_parquet(evaluation["oracle_path"], columns=["true_ite_prob", "treatment_indicator", "outcome_indicator"])
actual = truth.iloc[test_labels._oci_row_id.to_numpy()]
np.testing.assert_array_equal(actual.treatment_indicator, test_labels.treatment)
np.testing.assert_array_equal(actual.outcome_indicator, test_labels.outcome)
metrics = pd.read_csv(HERE / f"logistic_metrics_{e.DATE}.csv", float_precision="round_trip")
checks = []
for cohort in e.COHORTS:
    mask = np.ones(len(training), bool) if cohort == "all_800" else labels.effect_eligible.to_numpy(bool)
    for penalty in e.PENALTIES:
        destination = HERE / "fits" / cohort / penalty
        bundle = joblib.load(destination / "model.joblib")
        model, encoder = bundle["model"], bundle["encoder"]
        assert bundle["definitions"] == definitions
        independent = e._FeatureEncoder(definitions).fit(training.loc[mask].reset_index(drop=True))
        assert encoder.encodings == independent.encodings
        X = encoder.transform(heldout)
        p = X.shape[1]
        b = model.coef_.ravel()
        eta0 = model.intercept_[0] + X @ b[1:p + 1]
        eta1 = eta0 + b[0] + X @ b[p + 1:]
        p0, p1 = expit(eta0), expit(eta1)
        factual = np.where(test_labels.treatment == 1, p1, p0)
        effect = p1 - p0
        pred = pd.read_parquet(destination / "predictions.parquet")
        errors = []
        for values, column in [(p0, "predicted_y0"), (p1, "predicted_y1"), (factual, "factual_probability"), (effect, "estimated_cate")]:
            np.testing.assert_allclose(values, pred[column], rtol=0, atol=1e-12)
            errors.append(float(np.max(np.abs(values - pred[column]))))
        audit = e.read(destination / "audit.json")
        assert int(model.n_iter_.max()) == audit["iterations"] < audit["maximum_iterations"]
        assert model.C == audit["selected_C"]
        assert model.get_params() == audit["parameters"]
        np.testing.assert_array_equal(pred._oci_row_id, test_labels._oci_row_id)
        for name, test_mask in [("full_200", np.ones(len(pred), bool)), ("eligible_180", test_labels.effect_eligible.to_numpy(bool))]:
            estimate = effect[test_mask]
            tau = actual.true_ite_prob.to_numpy()[test_mask]
            subset = pred.loc[test_mask]
            scores = {"correlation": float(pearsonr(tau, estimate).statistic) if estimate.std() > 1e-12 else None,
                      "rmse": float(root_mean_squared_error(tau, estimate)), "mae": float(mean_absolute_error(tau, estimate)),
                      "bias": float(np.mean(estimate - tau)), "mean_effect": float(estimate.mean()), "prediction_sd": float(estimate.std()),
                      "r_loss": float(np.mean(np.square(subset.outcome - subset.outcome_prediction - estimate * (subset.treatment - subset.propensity)))),
                      "factual_log_loss": float(log_loss(subset.outcome, factual[test_mask], labels=[0, 1])),
                      "factual_auc": float(roc_auc_score(subset.outcome, factual[test_mask]))}
            row = metrics[(metrics.model == penalty) & (metrics.training_cohort == cohort) & (metrics.test_set == name)]
            assert len(row) == 1
            for key, value in scores.items():
                if value is None:
                    assert pd.isna(row.iloc[0][key])
                else:
                    np.testing.assert_allclose(row.iloc[0][key], value, rtol=0, atol=1e-12)
        checks.append({"cohort": cohort, "penalty": penalty, "maximum_prediction_error": max(errors),
                       "effects_independently_reconstructed_from_coefficients": True, "both_test_set_metrics_verified": True})
        print(json.dumps(checks[-1]), flush=True)
previous_manifest = e.read(e.i.PRIOR / f"input_manifest_{e.DATE}.json")
previous_frozen = e.read(e.i.PRIOR / f"predictions_frozen_{e.DATE}.json")
reference_hashes = {**previous_manifest["files"], **previous_frozen["files"]}
reference_checks = {}
for method, root in [("forest_sqrt", e.i.FOLD / "comparison_forests/all_candidates"),
                     ("forest_all_split_features", e.i.PRIOR / "fits")]:
    for seed in e.i.reference.SEEDS:
        path = root / f"seed_{seed}" / "predictions.csv"
        assert str(path.resolve()) in reference_hashes
        assert e.sha(path) == reference_hashes[str(path.resolve())]
        reference_checks[str(path.resolve())] = e.sha(path)
e.verify_inputs(manifest)
e.verify(frozen["files"])
e.write(HERE / f"independent_validation_{e.DATE}.json", {"validated_at": e.c.now(), "models_reloaded": 4,
        "inner_preprocessors_verified": 20, "CV_validation_scores_recomputed": 260,
        "oracle_not_used_for_fit_or_penalty_selection": True, "all_source_and_frozen_artifact_hashes_verified": True,
        "cv_checks": cv_checks, "final_checks": checks, "reference_forest_prediction_hashes": reference_checks,
        "validation_script_sha256": e.sha(Path(__file__))})
print("All CV decisions, four logistic models, scores, and frozen forest references verified.", flush=True)
