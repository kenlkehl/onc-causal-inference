"""All extracted candidates as outcome main effects and treatment interactions.

Five-fold training-only outcome log-loss selects C separately for elastic net
and ridge, on all 800 training patients and the previous 720 eligible patients.
No oracle values, feature-role identities, or known DGP interactions enter fit.
"""
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.util
import inspect
import json
import multiprocessing
import time
import warnings

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("extracted_inputs", HERE / "prepare_inputs_2026-09-21.py")
i = importlib.util.module_from_spec(spec)
spec.loader.exec_module(i)
c, read, write, sha = i.c, i.read, i.write, i.sha
from oci.inference.plain_handoff_stage2_analysis import _FeatureEncoder
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

DATE = "2026-09-21"
CS = np.logspace(-4, 2, 13)
PENALTIES = ("elastic_net", "ridge")
COHORTS = ("all_800", "eligible_720")
SEED = 120042


def progress(phase, **details):
    value = {"updated_at": c.now(), "phase": phase, **details}
    write(HERE / "status.json", value)
    print(json.dumps(value), flush=True)


def verify(files):
    for path, value in files.items():
        assert sha(path) == value, f"Frozen artifact changed: {path}"


def hashes(paths):
    return {str(path.resolve()): sha(path) for path in paths}


def verify_inputs(manifest):
    verify(manifest["sources"])
    original = read(HERE / f"inputs_frozen_{DATE}.json")
    verify(original["sources"])
    verify(original["files"])


def load_inputs():
    directory = HERE / "inputs"
    return (read(directory / "definitions.json"), pd.read_pickle(directory / "training.pkl"),
            pd.read_pickle(directory / "heldout.pkl"), pd.read_parquet(directory / "training_labels.parquet"),
            pd.read_parquet(directory / "heldout_labels.parquet"), read(directory / "split.json"))


def design(X, treatment):
    t = np.asarray(treatment, float)
    assert np.isin(t, [0, 1]).all()
    matrix = sparse.hstack([sparse.csr_matrix(t[:, None]), sparse.csr_matrix(X),
                            sparse.csr_matrix(X * t[:, None])], format="csr")
    assert np.isfinite(matrix.data).all()
    return matrix


def make_model(penalty, C, *, warm_start=False, max_iter=5000):
    return LogisticRegression(penalty="elasticnet" if penalty == "elastic_net" else "l2",
                              solver="saga" if penalty == "elastic_net" else "lbfgs",
                              l1_ratio=.8 if penalty == "elastic_net" else None,
                              C=float(C), fit_intercept=True, max_iter=max_iter, tol=1e-4,
                              random_state=SEED, n_jobs=1, warm_start=warm_start)


def coefficient_columns(encoder, definitions):
    columns = []
    for definition, (name, kind, parameters) in zip(definitions, encoder.encodings, strict=True):
        if kind == "continuous":
            suffixes = ["value", "missing"]
        elif kind == "continuous_with_categorical_fallback":
            suffixes = ["value", "numeric_observed", "missing", *["category=" + str(x) for x in parameters[2]], "other"]
        else:
            suffixes = ["category=" + str(x) for x in parameters]
            if kind == "categorical_with_other":
                suffixes.append("other")
        columns.extend({"feature_id": definition["feature_id"], "feature_name": name,
                        "encoded_term": name + ":" + suffix} for suffix in suffixes)
    return columns


def cv_job(cohort, penalty, fold_index):
    definitions, training, _, labels, _, split = load_inputs()
    mask = np.ones(len(training), dtype=bool) if cohort == "all_800" else labels.effect_eligible.to_numpy(bool)
    fold = split["inner_splits"][fold_index]
    train_mask = mask & labels._oci_row_id.isin(fold["fit_row_ids"]).to_numpy()
    valid_mask = mask & labels._oci_row_id.isin(fold["heldout_row_ids"]).to_numpy()
    assert not (train_mask & valid_mask).any() and np.array_equal(train_mask | valid_mask, mask)
    encoder = _FeatureEncoder(definitions).fit(training.loc[train_mask].reset_index(drop=True))
    Xtr = encoder.transform(training.loc[train_mask].reset_index(drop=True))
    Xva = encoder.transform(training.loc[valid_mask].reset_index(drop=True))
    Ztr = design(Xtr, labels.loc[train_mask, "treatment"])
    Zva = design(Xva, labels.loc[valid_mask, "treatment"])
    y = labels.loc[train_mask, "outcome"].to_numpy(int)
    yvalid = labels.loc[valid_mask, "outcome"].to_numpy(int)
    destination = HERE / "cv" / cohort / penalty / f"fold_{fold_index + 1}"
    destination.mkdir(parents=True, exist_ok=True)
    scores, predictions, coefficients, intercepts = [], [], [], []
    model = make_model(penalty, CS[0], warm_start=True)
    for C in CS:
        model.set_params(C=float(C))
        started = time.monotonic()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(Ztr, y)
        probability = model.predict_proba(Zva)[:, 1]
        assert np.isfinite(probability).all()
        iterations = int(model.n_iter_.max())
        row = {"C": float(C), "validation_log_loss": float(log_loss(yvalid, probability, labels=[0, 1])),
               "iterations": iterations, "converged": iterations < model.max_iter,
               "warnings": [str(w.message) for w in caught], "seconds": time.monotonic() - started}
        scores.append(row)
        predictions.append(probability)
        coefficients.append(model.coef_.ravel().copy())
        intercepts.append(float(model.intercept_[0]))
    np.savez_compressed(destination / "path.npz", C=CS, valid_row_ids=labels.loc[valid_mask, "_oci_row_id"].to_numpy(),
                        predictions=np.asarray(predictions), coefficients=np.asarray(coefficients), intercepts=np.asarray(intercepts))
    joblib.dump(encoder, destination / "encoder.joblib", compress=3)
    write(destination / "audit.json", {"cohort": cohort, "penalty": penalty, "fold": fold_index + 1,
            "training_row_ids": labels.loc[train_mask, "_oci_row_id"].tolist(),
            "validation_row_ids": labels.loc[valid_mask, "_oci_row_id"].tolist(),
            "encoded_covariate_columns": Xtr.shape[1], "design_columns": Ztr.shape[1],
            "encoder_fit_on_inner_training_only": True, "path": scores})
    print(json.dumps({"phase": "cv_fold_complete", "cohort": cohort, "penalty": penalty,
                      "fold": fold_index + 1, "nonconverged_candidates": sum(not row["converged"] for row in scores)}), flush=True)
    return hashes([destination / "path.npz", destination / "encoder.joblib", destination / "audit.json"])


def select_penalty(cohort, penalty):
    audits = [read(HERE / "cv" / cohort / penalty / f"fold_{index}" / "audit.json") for index in range(1, 6)]
    records = []
    for index, C in enumerate(CS):
        losses = [a["path"][index]["validation_log_loss"] for a in audits]
        records.append({"C": float(C), "mean_validation_log_loss": float(np.mean(losses)),
                        "se_validation_log_loss": float(np.std(losses, ddof=1) / np.sqrt(5)),
                        "all_folds_converged": all(a["path"][index]["converged"] for a in audits),
                        "fold_losses": losses})
    allowed = [row for row in records if row["all_folds_converged"]]
    assert allowed, "No penalty value converged in every fold"
    selected = min(allowed, key=lambda row: (row["mean_validation_log_loss"], row["C"]))
    return {"selected_C": selected["C"], "mean_validation_log_loss": selected["mean_validation_log_loss"],
            "selection_rule": "Minimum mean observed-outcome validation log loss among C values converged in all five folds; ties use smaller C.",
            "selected_on_grid_boundary": selected["C"] in [float(CS[0]), float(CS[-1])], "path": records}


def fit_final(cohort, penalty):
    definitions, training, heldout, labels, test_labels, _ = load_inputs()
    mask = np.ones(len(training), dtype=bool) if cohort == "all_800" else labels.effect_eligible.to_numpy(bool)
    selection = select_penalty(cohort, penalty)
    encoder = _FeatureEncoder(definitions).fit(training.loc[mask].reset_index(drop=True))
    Xtr = encoder.transform(training.loc[mask].reset_index(drop=True))
    Xte = encoder.transform(heldout)
    columns = coefficient_columns(encoder, definitions)
    assert len(columns) == Xtr.shape[1]
    Z = design(Xtr, labels.loc[mask, "treatment"])
    model = make_model(penalty, selection["selected_C"], max_iter=20_000)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(Z, labels.loc[mask, "outcome"].to_numpy(int))
    assert int(model.n_iter_.max()) < model.max_iter, "Selected final model did not converge"
    p0 = model.predict_proba(design(Xte, np.zeros(len(heldout))))[:, 1]
    p1 = model.predict_proba(design(Xte, np.ones(len(heldout))))[:, 1]
    factual = model.predict_proba(design(Xte, test_labels.treatment))[:, 1]
    np.testing.assert_allclose(factual, np.where(test_labels.treatment == 1, p1, p0), rtol=0, atol=1e-14)
    pred = test_labels.copy()
    pred["predicted_y0"], pred["predicted_y1"] = p0, p1
    pred["estimated_cate"], pred["factual_probability"] = p1 - p0, factual
    assert np.isfinite(pred[["predicted_y0", "predicted_y1", "estimated_cate", "factual_probability"]]).all().all()
    destination = HERE / "fits" / cohort / penalty
    destination.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(destination / "predictions.parquet", index=False)
    joblib.dump({"model": model, "encoder": encoder, "definitions": definitions}, destination / "model.joblib", compress=3)
    coef = model.coef_.ravel()
    rows = [{"block": "treatment", "feature_id": None, "feature_name": "treatment", "encoded_term": "treatment", "coefficient": float(coef[0])}]
    for block, values in [("main", coef[1:1 + len(columns)]), ("interaction", coef[1 + len(columns):])]:
        rows.extend({"block": block, **column, "coefficient": float(value)} for column, value in zip(columns, values, strict=True))
    coefficients = pd.DataFrame(rows)
    coefficients.to_csv(destination / "coefficients.csv", index=False)
    active = coefficients[np.abs(coefficients.coefficient) > 1e-8]
    active_interactions = active[active.block == "interaction"]
    write(destination / "selection.json", selection)
    audit = {"cohort": cohort, "penalty": penalty, "training_rows": int(mask.sum()), "heldout_predictions": len(pred),
             "training_row_ids": labels.loc[mask, "_oci_row_id"].tolist(), "clinical_features": len(definitions),
             "encoded_covariate_columns": Xtr.shape[1], "regression_columns": Z.shape[1],
             "coefficients_including_intercept": Z.shape[1] + 1,
             "nonconstant_regression_columns": int(np.count_nonzero(np.std(Z.toarray(), axis=0) > 0)),
             "nonzero_coefficients_excluding_intercept": len(active),
             "nonzero_main_coefficients": int((active.block == "main").sum()),
             "nonzero_interaction_coefficients": len(active_interactions),
             "candidates_with_nonzero_interactions": int(active_interactions.feature_id.nunique()),
             "treatment_main_coefficient": float(coef[0]), "intercept": float(model.intercept_[0]),
             "nonzero_threshold": 1e-8, "selected_C": selection["selected_C"],
             "iterations": int(model.n_iter_.max()), "maximum_iterations": model.max_iter,
             "warnings": [str(w.message) for w in caught], "parameters": model.get_params(),
             "training_log_loss": float(log_loss(labels.loc[mask, "outcome"], model.predict_proba(Z)[:, 1], labels=[0, 1])),
             "train_covariate_sha256": hashlib.sha256(Xtr.tobytes()).hexdigest(),
             "test_covariate_sha256": hashlib.sha256(Xte.tobytes()).hexdigest()}
    write(destination / "audit.json", audit)
    print(json.dumps({"phase": "final_model_fitted", **{key: audit[key] for key in ["cohort", "penalty", "training_rows", "selected_C", "iterations", "nonzero_interaction_coefficients"]}}), flush=True)
    return hashes([destination / name for name in ["predictions.parquet", "model.joblib", "coefficients.csv", "selection.json", "audit.json"]])


def fit():
    if (HERE / f"predictions_frozen_{DATE}.json").exists():
        raise ValueError("Models already frozen")
    paths = [Path(__file__), HERE / f"inputs_frozen_{DATE}.json", HERE / f"prepare_inputs_{DATE}.py",
             Path(inspect.getfile(_FeatureEncoder)), Path(inspect.getfile(LogisticRegression))]
    manifest = {"created_at": c.now(), "purpose": __doc__, "cohorts": COHORTS, "penalties": PENALTIES,
                "C_grid": CS.tolist(), "elastic_net_l1_ratio": .8, "cv_folds": 5,
                "cv_splits": "Original fold-1 inner splits, intersected with each cohort",
                "cv_preprocessing": "Encoder fitted separately using only each inner training partition",
                "model_formula": "logit P(Y=1|T,Z) = intercept + beta*Z + delta*T + gamma*(T*Z)",
                "penalty_scope": "All slopes including treatment main effect; intercept unpenalized",
                "cv_max_iter": 5000, "final_max_iter": 20_000, "tolerance": 1e-4,
                "constant_and_redundant_columns": "Retained exactly as represented by the existing feature encoder",
                "oracle_values_used_for_fitting_or_tuning": False, "known_DGP_structure_supplied": False,
                "prior_results_already_seen": True, "sources": hashes(paths)}
    write(HERE / f"input_manifest_{DATE}.json", manifest)
    verify_inputs(manifest)
    progress("cross_validating", jobs=20, candidate_penalties=len(CS), final_models=4)
    files = {}
    jobs = [(cohort, penalty, fold) for cohort in COHORTS for fold in range(5) for penalty in PENALTIES]
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(cv_job, *job) for job in jobs]
        completed = 0
        for future in as_completed(futures):
            files.update(future.result())
            completed += 1
            progress("cross_validation_progress", completed_jobs=completed, total_jobs=20)
    progress("fitting_final_models", final_models=4)
    for cohort in COHORTS:
        for penalty in PENALTIES:
            files.update(fit_final(cohort, penalty))
    verify_inputs(manifest)
    verify(files)
    write(HERE / f"predictions_frozen_{DATE}.json", {"frozen_at": c.now(), "new_models": 4,
          "input_manifest_sha256": sha(HERE / f"input_manifest_{DATE}.json"), "files": files,
          "oracle_values_read_by_fit": False})
    progress("predictions_frozen", new_models=4)


def effect_metrics(pred, truth):
    tau = np.asarray(truth)
    effect = pred.estimated_cate.to_numpy()
    error = effect - tau
    return {"test_rows": len(pred), "correlation": float(np.corrcoef(effect, tau)[0, 1]) if np.std(effect) > 1e-12 else None,
            "rmse": float(np.sqrt(np.mean(error ** 2))), "mae": float(np.abs(error).mean()), "bias": float(error.mean()),
            "mean_effect": float(effect.mean()), "prediction_sd": float(effect.std()),
            "r_loss": float(np.mean((pred.outcome - pred.outcome_prediction - effect * (pred.treatment - pred.propensity)) ** 2))}


def evaluate():
    manifest = read(HERE / f"input_manifest_{DATE}.json")
    frozen = read(HERE / f"predictions_frozen_{DATE}.json")
    assert sha(HERE / f"input_manifest_{DATE}.json") == frozen["input_manifest_sha256"]
    verify_inputs(manifest)
    verify(frozen["files"])
    oracle_path = i.ROOT / "synthetic_data/example_synthetic_datasets/five_confounders_five_effect_modifiers_nsclc_with_structured/dataset.parquet"
    truth = pd.read_parquet(oracle_path, columns=["true_ite_prob", "treatment_indicator", "outcome_indicator"])
    rows = []
    for cohort in COHORTS:
        for penalty in PENALTIES:
            directory = HERE / "fits" / cohort / penalty
            pred = pd.read_parquet(directory / "predictions.parquet")
            audit = read(directory / "audit.json")
            actual = truth.iloc[pred._oci_row_id.to_numpy()]
            np.testing.assert_array_equal(pred.treatment, actual.treatment_indicator)
            np.testing.assert_array_equal(pred.outcome, actual.outcome_indicator)
            for test_set, mask in [("full_200", np.ones(len(pred), bool)), ("eligible_180", pred.effect_eligible.to_numpy(bool))]:
                subset = pred.loc[mask].reset_index(drop=True)
                scores = effect_metrics(subset, actual.true_ite_prob.to_numpy()[mask])
                rows.append({"model": penalty, "training_cohort": cohort, "test_set": test_set,
                             "training_rows": audit["training_rows"], "selected_C": audit["selected_C"], **scores,
                             "factual_log_loss": float(log_loss(subset.outcome, subset.factual_probability, labels=[0, 1])),
                             "factual_auc": float(roc_auc_score(subset.outcome, subset.factual_probability)),
                             "nonzero_interaction_coefficients": audit["nonzero_interaction_coefficients"]})
    logistic = pd.DataFrame(rows)
    logistic.to_csv(HERE / f"logistic_metrics_{DATE}.csv", index=False)
    forest_rows = []
    for method, root in [("forest_sqrt", i.FOLD / "comparison_forests/all_candidates"),
                         ("forest_all_split_features", i.PRIOR / "fits")]:
        for seed in i.reference.SEEDS:
            path = root / f"seed_{seed}" / "predictions.csv"
            pred = pd.read_csv(path, float_precision="round_trip")
            forest_rows.append({"model": method, "seed": seed, "training_rows": 720,
                                **effect_metrics(pred, truth.iloc[pred._oci_row_id.to_numpy()].true_ite_prob)})
    forest = pd.DataFrame(forest_rows)
    forest.to_csv(HERE / f"forest_reference_metrics_{DATE}.csv", index=False)
    forest_summary = forest.groupby("model").mean(numeric_only=True).drop(columns="seed").reset_index()
    eligible_comparison = pd.concat([logistic[(logistic.training_cohort == "eligible_720") &
                                               (logistic.test_set == "eligible_180")], forest_summary], ignore_index=True)
    eligible_comparison.to_csv(HERE / f"matched_comparison_{DATE}.csv", index=False)
    test_labels = pd.read_parquet(HERE / "inputs/heldout_labels.parquet")
    test_truth = truth.iloc[test_labels._oci_row_id.to_numpy()].true_ite_prob.to_numpy()
    write(HERE / f"evaluation_{DATE}.json", {"evaluated_at": c.now(), "logistic": rows,
          "forest_reference": forest_rows, "oracle_path": str(oracle_path), "oracle_sha256": sha(oracle_path),
          "oracle_full_test_mean": float(test_truth.mean()), "oracle_full_test_sd": float(test_truth.std()),
          "oracle_eligible_test_mean": float(test_truth[test_labels.effect_eligible].mean()),
          "oracle_eligible_test_sd": float(test_truth[test_labels.effect_eligible].std()),
          "note": "Forest references use 720 training and 180 eligible test rows; their summary averages three seed scores."})
    verify_inputs(manifest)
    verify(frozen["files"])
    print(logistic[["model", "training_cohort", "test_set", "selected_C", "correlation", "rmse", "bias", "prediction_sd"]].to_string(index=False), flush=True)
    print(eligible_comparison[["model", "correlation", "rmse", "r_loss", "prediction_sd"]].to_string(index=False), flush=True)
    progress("evaluated", new_models=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["fit", "evaluate"])
    arguments = parser.parse_args()
    fit() if arguments.phase == "fit" else evaluate()
