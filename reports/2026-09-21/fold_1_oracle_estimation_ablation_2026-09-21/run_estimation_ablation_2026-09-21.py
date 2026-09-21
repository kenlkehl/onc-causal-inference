"""Isolated diagnostics of oracle-covariate forest estimation, September 21.

Intentionally uses training-row oracle probabilities in diagnostic controls.
No production fitting is changed. Held-out effects are for evaluation only.
The held-out fold was already inspected in earlier experiments, so these are
exploratory ablations, not a new blinded validation study.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/oci_oracle_ablation_matplotlib")
import joblib
import numpy as np
import pandas as pd
from econml.grf import CausalForest
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "fold_1_true_oracle_xw_comparison_2026-09-21"
spec = importlib.util.spec_from_file_location("oracle_covariate_diagnostic", BASE / "run_true_oracle_xw_comparison.py")
u = importlib.util.module_from_spec(spec)
spec.loader.exec_module(u)


def fit():
    freeze_path = HERE / "predictions_frozen_2026-09-21.json"
    if freeze_path.exists():
        raise ValueError("Already frozen; do not overwrite diagnostic fits")
    base_manifest = u.read(BASE / "input_manifest_2026-09-21.json")
    base_freeze = u.read(BASE / "predictions_frozen_2026-09-21.json")
    u.verify(base_manifest["sources"])
    u.verify(base_freeze["files"])
    assert u.sha(BASE / "input_manifest_2026-09-21.json") == base_freeze["input_manifest_sha256"]
    train, test = base_manifest["training_rows"], base_manifest["heldout_rows"]
    names = ["true_" + d["name"] for d in base_manifest["confounders"] + base_manifest["modifiers"]]
    data = pd.read_parquet(u.DATA / "dataset.parquet", columns=names + ["treatment_indicator", "outcome_indicator"])
    # Only these training-row probabilities enter diagnostic fitting.
    probabilities = pd.read_parquet(u.DATA / "dataset.parquet", columns=["true_treatment_prob", "true_y0_prob", "true_y1_prob"]).iloc[train]
    e = probabilities.true_treatment_prob.to_numpy()
    p0, p1 = probabilities.true_y0_prob.to_numpy(), probabilities.true_y1_prob.to_numpy()
    true_m = (1 - e) * p0 + e * p1
    T = data.treatment_indicator.to_numpy()[train]
    Y = data.outcome_indicator.to_numpy()[train]
    factual_probability = (1 - T) * p0 + T * p1
    C, c_names, _, _ = u.encode(data, base_manifest["confounders"], train)
    M, m_names, _, _ = u.encode(data, base_manifest["modifiers"], train)
    all_X = np.column_stack([C, M])
    residuals = {"oracle_nuisances_observed_outcome": (Y - true_m, T - e),
                 "oracle_nuisances_noiseless_outcome": (factual_probability - true_m, T - e)}
    np.testing.assert_allclose(residuals["oracle_nuisances_noiseless_outcome"][0], (p1 - p0) * (T - e), atol=1e-15, rtol=0)
    manifest = {"created_at": u.now(), "purpose": __doc__, "train_rows": train, "test_rows": test,
                "scenarios": ["a", "c"], "search": u.SEARCH, "seeds": u.SEEDS,
                "residual_conditions": list(residuals), "oracle_probabilities_used_for_training_rows": True,
                "heldout_probabilities_or_effects_used_for_fitting": False,
                "forest_parameters": "Exact parameters copied from each corresponding frozen baseline forest.",
                "parametric_design": {"baseline": "All 13 confounder columns plus age*creatinine_clearance.",
                                      "treatment": "Treatment main effect plus treatment times all 12 modifier columns.",
                                      "coefficients": "Estimated solely from the 800 observed training outcomes; no true coefficients supplied.",
                                      "favorable_oracle_information": "Known DGP interaction structure and logistic link.",
                                      "fits": ["unpenalized logistic", "repository elastic-net logistic with original nuisance settings"]},
                "sources": {**base_manifest["sources"], str(Path(__file__).resolve()): u.sha(__file__),
                            str((BASE / "input_manifest_2026-09-21.json").resolve()): u.sha(BASE / "input_manifest_2026-09-21.json"),
                            str((BASE / "predictions_frozen_2026-09-21.json").resolve()): u.sha(BASE / "predictions_frozen_2026-09-21.json")}}
    u.write(HERE / "input_manifest_2026-09-21.json", manifest)
    files, replay_checks, warning_records = {}, [], []
    for scenario in ["a", "c"]:
        X = all_X if scenario == "a" else M
        assert u.array_hash(X[train]) == base_manifest["scenarios"][scenario]["train_X_sha256"]
        initial = joblib.load(BASE / "fits" / scenario / "initial_model.joblib")
        original_yres, original_tres, cached_X, _ = initial.residuals_
        np.testing.assert_array_equal(cached_X, X[train])
        for search in u.SEARCH:
            for seed in u.SEEDS:
                old_dir = BASE / "fits" / scenario / search / f"seed_{seed}"
                parameters = joblib.load(old_dir / "forest.joblib").get_params()
                replay = CausalForest(**parameters).fit(X[train], original_tres, original_yres)
                old = pd.read_csv(old_dir / "predictions.csv", float_precision="round_trip")
                error = float(np.max(np.abs(replay.predict(X[test]).ravel() - old.estimated_cate.to_numpy())))
                assert error == 0.0, (scenario, search, seed, error)
                replay_checks.append({"scenario": scenario, "search": search, "seed": seed, "max_prediction_error": error})
                for condition, (yr, tr) in residuals.items():
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        forest = CausalForest(**parameters).fit(X[train], tr.reshape(-1, 1), yr.reshape(-1, 1))
                    pred = forest.predict(X[test]).ravel()
                    assert np.isfinite(pred).all()
                    dest = HERE / "fits" / condition / scenario / search / f"seed_{seed}"
                    dest.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame({"_oci_row_id": test, "condition": condition, "scenario": scenario,
                                  "search": search, "seed": seed, "estimated_cate": pred}).to_csv(dest / "predictions.csv", index=False)
                    joblib.dump(forest, dest / "forest.joblib", compress=3)
                    for path in [dest / "predictions.csv", dest / "forest.joblib"]:
                        files[str(path.resolve())] = u.sha(path)
                    warning_records.extend({"condition": condition, "scenario": scenario, "search": search, "seed": seed,
                                            "warning": str(w.message)} for w in caught)
                    print(json.dumps({"phase": "forest_fitted", "condition": condition, "scenario": scenario, "search": search, "seed": seed}), flush=True)
    # Correct model family, but coefficients learned from noisy observed outcomes.
    product = C[:, c_names.index("age")] * C[:, c_names.index("creatinine_clearance")]
    def design(ids, treatment):
        t = np.asarray(treatment, dtype=float)
        return np.column_stack([C[ids], product[ids], t, M[ids] * t[:, None]])
    fit_design = design(train, T)
    d0, d1 = design(test, np.zeros(len(test))), design(test, np.ones(len(test)))
    parametric = {"unpenalized_logistic": LogisticRegression(penalty=None, solver="lbfgs", max_iter=5000, tol=1e-8, random_state=u.SEEDS[0]),
                  "elastic_net_logistic": u.ElasticNetLogisticClassifier(**u.NUISANCE)}
    parametric_audits = {}
    for name, model in parametric.items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(fit_design, Y)
        mu0, mu1 = model.predict_proba(d0)[:, 1], model.predict_proba(d1)[:, 1]
        pred = mu1 - mu0
        assert np.isfinite(pred).all()
        dest = HERE / "fits" / name
        dest.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"_oci_row_id": test, "condition": name, "scenario": "known_logistic_structure",
                      "search": "not_applicable", "seed": u.SEEDS[0], "estimated_cate": pred,
                      "estimated_mu0": mu0, "estimated_mu1": mu1}).to_csv(dest / "predictions.csv", index=False)
        joblib.dump(model, dest / "model.joblib", compress=3)
        for path in [dest / "predictions.csv", dest / "model.joblib"]:
            files[str(path.resolve())] = u.sha(path)
        parametric_audits[name] = model.fit_audit() if hasattr(model, "fit_audit") else {"iterations": model.n_iter_.tolist(), "iteration_limit_reached": bool(np.any(model.n_iter_ >= model.max_iter))}
        warning_records.extend({"condition": name, "warning": str(w.message)} for w in caught)
    u.write(HERE / "fit_validation_2026-09-21.json", {"created_at": u.now(), "baseline_replays": replay_checks,
            "parametric_audits": parametric_audits, "warnings": warning_records,
            "all_baseline_replays_identical": True, "forests_fitted": 24, "parametric_models_fitted": 2,
            "noiseless_residual_identity_verified": True})
    files[str((HERE / "fit_validation_2026-09-21.json").resolve())] = u.sha(HERE / "fit_validation_2026-09-21.json")
    u.verify(manifest["sources"])
    u.verify(base_freeze["files"])
    assert len(replay_checks) == 12 and len(files) == 53
    u.write(freeze_path, {"frozen_at": u.now(), "input_manifest_sha256": u.sha(HERE / "input_manifest_2026-09-21.json"), "prediction_sets": 26, "files": files})
    print("All 26 diagnostic prediction sets frozen.", flush=True)


def evaluate():
    manifest = u.read(HERE / "input_manifest_2026-09-21.json")
    frozen = u.read(HERE / "predictions_frozen_2026-09-21.json")
    assert frozen["input_manifest_sha256"] == u.sha(HERE / "input_manifest_2026-09-21.json")
    u.verify(manifest["sources"])
    u.verify(frozen["files"])
    truth = pd.read_parquet(u.DATA / "dataset.parquet", columns=["true_ite_prob"]).iloc[manifest["test_rows"]].true_ite_prob.to_numpy()
    paths = sorted((HERE / "fits").glob("**/predictions.csv"))
    assert len(paths) == 26
    paths += [BASE / "fits" / scenario / search / f"seed_{seed}" / "predictions.csv"
              for scenario in ["a", "c"] for search in u.SEARCH for seed in u.SEEDS]
    records = []
    for path in paths:
        pred = pd.read_csv(path, float_precision="round_trip")
        assert pred._oci_row_id.tolist() == manifest["test_rows"]
        est = pred.estimated_cate.to_numpy()
        condition = pred.condition.iloc[0] if "condition" in pred else "estimated_nuisances_observed_outcome"
        records.append({"condition": condition, "scenario": pred.scenario.iloc[0], "search": pred.search.iloc[0],
                        "seed": int(pred.seed.iloc[0]), "n": len(est), "correlation": float(np.corrcoef(est, truth)[0, 1]),
                        "rmse": float(np.sqrt(np.mean((est - truth) ** 2))), "bias": float(np.mean(est - truth)),
                        "prediction_sd": float(np.std(est)), "oracle_sd": float(np.std(truth))})
    table = pd.DataFrame(records)
    assert len(table) == 38
    table.to_csv(HERE / "metrics_by_seed_2026-09-21.csv", index=False)
    agg = table.groupby(["condition", "scenario", "search"], dropna=False)[["correlation", "rmse", "bias", "prediction_sd"]].agg(["mean", "min", "max"])
    agg.to_csv(HERE / "summary_metrics_2026-09-21.csv")
    u.verify(manifest["sources"])
    u.verify(frozen["files"])
    u.write(HERE / "evaluation_validation_2026-09-21.json", {"evaluated_at": u.now(), "new_fits": 26,
            "baseline_fits": 12, "heldout_rows_each": 200, "all_source_and_fit_hashes_verified": True})
    print(table.groupby(["condition", "scenario", "search"])[["correlation", "rmse", "bias", "prediction_sd"]].mean().to_string(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["fit", "evaluate"])
    args = parser.parse_args()
    fit() if args.phase == "fit" else evaluate()
