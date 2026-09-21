"""User-authorized fold 1 CausalForestDML diagnostic with true covariates.

Three literal X/W placements, two split-search settings, three forest seeds.
Only true covariates plus observed T/Y enter fitting. Oracle outcome probabilities
and treatment effects are loaded in evaluate, after all predictions are frozen.
No candidate definitions, extraction values, or production outputs are changed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import sys
import time
import warnings

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/oci_oracle_xw_matplotlib")

import joblib
import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from econml.dml.dml import _combine
from oci.models.elastic_net_nuisance import ElasticNetLogisticClassifier

DATA = ROOT / "synthetic_data/example_synthetic_datasets/five_confounders_five_effect_modifiers_nsclc_with_structured"
SPLITS = ROOT / "reports/2026-09-19/selection_comparison_2026-09-19/results/inputs/splits.json"
PREVIOUS = HERE.parent / "fold_1_oracle_modifier_candidates_2026-09-21/input_manifest_2026-09-21.json"
SEEDS = (120042, 1120042, 2120042)
SEARCH = {"sqrt": "sqrt", "all": 1.0}
SCENARIOS = {"a": {"X": "C+M", "W": "none"}, "b": {"X": "C+M", "W": "C"}, "c": {"X": "M", "W": "C"}}
NUISANCE = dict(l1_ratio=0.8, cv_folds=3, regularization_grid_size=16,
                minimum_log10_c=-2.0, maximum_log10_c=4.0, max_iter=5000,
                tolerance=1e-4, random_state=120042, n_jobs=1)
FOREST = dict(n_estimators=200, max_depth=None, min_samples_leaf=10, max_samples=0.45,
              honest=True, inference=True, subforest_size=4, n_jobs=1, criterion="mse")


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def array_hash(*arrays):
    h = hashlib.sha256()
    for array in arrays:
        array = np.asarray(array)
        h.update(str(array.shape).encode())
        h.update(str(array.dtype).encode())
        h.update(array.tobytes())
    return h.hexdigest()


def verify(files):
    for path, expected in files.items():
        if sha(path) != expected:
            raise ValueError(f"Frozen input changed: {path}")


def progress(phase, **values):
    record = {"at": now(), "phase": phase, **values}
    write(HERE / "status.json", record)
    print(json.dumps(record), flush=True)


def encode(frame, definitions, train_ids):
    """Full declared one-hot categories and training-only standardized numerics."""
    arrays, columns, owners, audit = [], [], [], []
    for definition in definitions:
        name = definition["name"]
        values = frame["true_" + name]
        if values.isna().any():
            raise ValueError(f"Oracle variable unexpectedly missing: {name}")
        if definition["type"] == "continuous":
            numeric = values.to_numpy(dtype=float)
            assert np.isfinite(numeric).all()
            mean = float(np.mean(numeric[train_ids]))
            scale = float(np.std(numeric[train_ids], ddof=0)) or 1.0
            arrays.append((numeric - mean) / scale)
            columns.append(name)
            owners.append(name)
            audit.append({"name": name, "type": "continuous", "training_mean": mean, "training_sd": scale})
        else:
            categories = definition["categories"]
            assert set(values).issubset(set(categories)), name
            for category in categories:
                arrays.append((values == category).to_numpy(dtype=float))
                columns.append(name + "=" + str(category))
                owners.append(name)
            audit.append({"name": name, "type": "categorical", "categories": categories})
    matrix = np.column_stack(arrays)
    assert np.isfinite(matrix).all()
    return matrix, columns, owners, audit


def nuisance_predictions(model, X, W):
    combined = _combine(X, W, len(X))
    assert len(model.models_t) == len(model.models_y) == 1
    propensity = np.mean([m.predict_proba(combined)[:, 1] for m in model.models_t[0]], axis=0)
    outcome = np.mean([m.predict_proba(combined)[:, 1] for m in model.models_y[0]], axis=0)
    return propensity, outcome


def fit():
    if (HERE / "predictions_frozen_2026-09-21.json").exists():
        raise ValueError("Fits already frozen; evaluate instead of overwriting")
    metadata = read(DATA / "metadata.json")
    definitions = [{k: d[k] for k in ["name", "type", "roles", "categories"] if k in d}
                   for d in metadata["features"]]
    confounders = [d for d in definitions if "confounder" in d["roles"]]
    modifiers = [d for d in definitions if "effect_modifier" in d["roles"]]
    assert len(confounders) == len(modifiers) == 5
    assert not {d["name"] for d in confounders} & {d["name"] for d in modifiers}
    covariates = ["true_" + d["name"] for d in definitions]
    # The oracle effect and probability columns are deliberately not read here.
    frame = pd.read_parquet(DATA / "dataset.parquet", columns=[*covariates, "treatment_indicator", "outcome_indicator"])
    split = next(s for s in read(SPLITS) if s["outer_fold"] == 1)
    train_ids, test_ids = split["fit_row_ids"], split["heldout_row_ids"]
    assert len(train_ids) == 800 and len(test_ids) == 200
    assert set(train_ids).isdisjoint(test_ids)
    local = {row: index for index, row in enumerate(train_ids)}
    cv, validation_rows = [], []
    for inner in split["inner_splits"]:
        itr = np.array([local[row] for row in inner["fit_row_ids"]], dtype=int)
        ite = np.array([local[row] for row in inner["heldout_row_ids"]], dtype=int)
        assert set(itr).isdisjoint(ite) and set(itr) | set(ite) == set(range(800))
        cv.append((itr, ite))
        validation_rows.extend(ite.tolist())
    assert sorted(validation_rows) == list(range(800))
    C, c_names, c_owners, c_encoding = encode(frame, confounders, train_ids)
    M, m_names, m_owners, m_encoding = encode(frame, modifiers, train_ids)
    CM = np.column_stack([C, M])
    designs = {"a": (CM, None, c_names + m_names, [], c_owners + m_owners),
               "b": (CM, C, c_names + m_names, c_names, c_owners + m_owners),
               "c": (M, C, m_names, c_names, m_owners)}
    assert C.shape[1] == 13 and M.shape[1] == 12
    T = frame.treatment_indicator.to_numpy(dtype=int)
    Y = frame.outcome_indicator.to_numpy(dtype=int)
    assert set(T) == set(Y) == {0, 1}
    paths = [Path(__file__), DATA / "dataset.parquet", DATA / "metadata.json", SPLITS, PREVIOUS,
             ROOT / "oci/models/elastic_net_nuisance.py", Path(inspect.getfile(CausalForestDML)), Path(inspect.getfile(_combine))]
    sources = {str(p.resolve()): sha(p) for p in paths}
    design_manifest = {}
    for name, (X, W, x_names, w_names, owners) in designs.items():
        nuisance = _combine(X, W, len(X))
        design_manifest[name] = {**SCENARIOS[name], "x_columns": x_names, "w_columns": w_names,
                                "x_encoded_columns": X.shape[1], "w_encoded_columns": 0 if W is None else W.shape[1],
                                "nuisance_encoded_columns": nuisance.shape[1],
                                "duplicated_nuisance_columns": {k: v for k, v in Counter(x_names + w_names).items() if v > 1},
                                "train_X_sha256": array_hash(X[train_ids]), "test_X_sha256": array_hash(X[test_ids]),
                                "train_nuisance_XW_sha256": array_hash(nuisance[train_ids])}
    previous_ids = read(PREVIOUS)["heldout_row_ids"]
    assert set(previous_ids).issubset(test_ids) and len(previous_ids) == 180
    manifest = {"created_at": now(), "purpose": "User-authorized true-oracle covariate comparison of three literal X/W configurations.",
                "true_covariates_used_in_fitting": True, "true_effect_or_outcome_probabilities_used_in_fitting": False,
                "extracted_measurements_used": False, "production_modified": False,
                "confounders": confounders, "modifiers": modifiers, "encoding": c_encoding + m_encoding,
                "training_rows": train_ids, "heldout_rows": test_ids, "previous_180_heldout_rows": previous_ids,
                "crossfit_splits": [{"train": a.tolist(), "test": b.tolist()} for a, b in cv],
                "nuisance_parameters": NUISANCE, "forest_parameters": FOREST,
                "forest_seeds": SEEDS, "search_settings": SEARCH, "scenarios": design_manifest,
                "rows_policy": "All 800/200 outer-fold rows; no estimated-propensity filtering. Prior 180 heldout IDs are a secondary reporting subset only.",
                "nuisance_seed_policy": "One fixed nuisance seed and fixed five crossfit splits per scenario; cached nuisance estimates shared across six final forests.",
                "sources": sources, "versions": {p: importlib.metadata.version(p) for p in ["econml", "numpy", "pandas", "scikit-learn"]}}
    write(HERE / "input_manifest_2026-09-21.json", manifest)
    frozen = {}
    for name, (X, W, x_names, w_names, owners) in designs.items():
        Xtr, Xte = X[train_ids], X[test_ids]
        Wtr, Wte = (None, None) if W is None else (W[train_ids], W[test_ids])
        directory = HERE / "fits" / name
        directory.mkdir(parents=True, exist_ok=True)
        progress("fitting_native_causal_forest_dml", scenario=name, x_columns=Xtr.shape[1],
                 w_columns=0 if Wtr is None else Wtr.shape[1])
        model = CausalForestDML(model_t=ElasticNetLogisticClassifier(**NUISANCE),
                               model_y=ElasticNetLogisticClassifier(**NUISANCE), discrete_treatment=True,
                               discrete_outcome=True, cv=cv, random_state=SEEDS[0], max_features="sqrt", **FOREST)
        started = time.monotonic()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(Y[train_ids], T[train_ids], X=Xtr, W=Wtr, cache_values=True)
        yres, tres, cached_X, cached_W = model.residuals_
        np.testing.assert_array_equal(cached_X, Xtr)
        if Wtr is not None:
            np.testing.assert_array_equal(cached_W, Wtr)
        else:
            assert cached_W is None
        yres, tres = np.asarray(yres).ravel(), np.asarray(tres).ravel()
        e_test, m_test = nuisance_predictions(model, Xte, Wte)
        e_train, m_train = T[train_ids] - tres, Y[train_ids] - yres
        assert np.isfinite(np.column_stack([e_train, m_train])).all()
        assert np.all((e_train >= 0) & (e_train <= 1) & (m_train >= 0) & (m_train <= 1))
        nuisance_hash = array_hash(yres, tres, e_test, m_test)
        pd.DataFrame({"_oci_row_id": train_ids, "treatment": T[train_ids], "outcome": Y[train_ids],
                      "propensity": e_train, "outcome_prediction": m_train}).to_csv(directory / "training_nuisances.csv", index=False)
        nuisance_audit = {role: [n.fit_audit() for n in models[0]]
                          for role, models in [("treatment", model.models_t), ("outcome", model.models_y)]}
        warning_counts = Counter((type(w.message).__name__, str(w.message)) for w in caught)
        audit = {"fit_seconds": time.monotonic() - started, "nuisance_hash": nuisance_hash,
                 "models": nuisance_audit, "warnings": [{"type": k[0], "message": k[1], "count": v}
                                                          for k, v in warning_counts.items()]}
        write(directory / "nuisance_audit.json", audit)
        joblib.dump(model, directory / "initial_model.joblib", compress=3)
        for q in [directory / "training_nuisances.csv", directory / "nuisance_audit.json", directory / "initial_model.joblib"]:
            frozen[str(q.resolve())] = sha(q)
        progress("nuisances_fitted", scenario=name, seconds=audit["fit_seconds"], warnings=sum(warning_counts.values()))
        for search, value in SEARCH.items():
            for replicate, seed in enumerate(SEEDS):
                if (search, seed) != ("sqrt", SEEDS[0]):
                    model.max_features, model.random_state = value, seed
                    model.refit_final()
                forest = model.model_cate.estimators_[0]
                assert forest.max_features == value and forest.random_state == seed
                fresh_y, fresh_t, _, _ = model.residuals_
                fresh_e, fresh_m = nuisance_predictions(model, Xte, Wte)
                assert array_hash(np.asarray(fresh_y).ravel(), np.asarray(fresh_t).ravel(), fresh_e, fresh_m) == nuisance_hash
                cate = np.asarray(model.effect(Xte)).ravel()
                lower, upper = [np.asarray(v).ravel() for v in model.effect_interval(Xte)]
                assert np.isfinite(np.column_stack([cate, lower, upper])).all()
                assert np.all(lower <= cate) and np.all(cate <= upper)
                rloss = float(np.mean((Y[test_ids] - m_test - cate * (T[test_ids] - e_test)) ** 2))
                native_score = float(model.score(Y[test_ids], T[test_ids], X=Xte, W=Wte))
                assert abs(rloss - native_score) < 1e-12
                dest = directory / search / f"seed_{seed}"
                dest.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({"_oci_row_id": test_ids, "scenario": name, "search": search, "seed": seed,
                              "treatment": T[test_ids], "outcome": Y[test_ids], "propensity": e_test,
                              "outcome_prediction": m_test, "estimated_cate": cate,
                              "lower_95": lower, "upper_95": upper}).to_csv(dest / "predictions.csv", index=False)
                imports = np.asarray(model.feature_importances_).ravel()
                assert len(imports) == len(owners)
                grouped = {feature: float(sum(v for feature_owner, v in zip(owners, imports) if feature_owner == feature))
                           for feature in dict.fromkeys(owners)}
                write(dest / "fit_metrics.json", {"scenario": name, "search": search, "seed": seed,
                      "r_loss": rloss, "native_score": native_score, "nuisance_hash": nuisance_hash,
                      "parameters": forest.get_params(), "grouped_feature_importance": grouped,
                      "mean_cate": float(np.mean(cate)), "sd_cate": float(np.std(cate))})
                joblib.dump(forest, dest / "forest.joblib", compress=3)
                for q in [dest / "predictions.csv", dest / "fit_metrics.json", dest / "forest.joblib"]:
                    frozen[str(q.resolve())] = sha(q)
                progress("forest_complete", scenario=name, search=search, seed=seed, r_loss=rloss)
    assert len(list((HERE / "fits").glob("*/*/seed_*/predictions.csv"))) == 18
    verify(sources)
    write(HERE / "predictions_frozen_2026-09-21.json", {"frozen_at": now(), "forest_fits": 18,
          "input_manifest_sha256": sha(HERE / "input_manifest_2026-09-21.json"), "files": frozen,
          "oracle_effects_and_probabilities_read_by_fit": False})
    progress("predictions_frozen", fits=18)


def evaluate():
    manifest = read(HERE / "input_manifest_2026-09-21.json")
    frozen = read(HERE / "predictions_frozen_2026-09-21.json")
    assert frozen["forest_fits"] == 18 and frozen["input_manifest_sha256"] == sha(HERE / "input_manifest_2026-09-21.json")
    verify(frozen["files"])
    verify(manifest["sources"])
    truth = pd.read_parquet(DATA / "dataset.parquet", columns=["true_ite_prob", "true_y0_prob", "true_y1_prob", "true_treatment_prob"])
    truth.insert(0, "_oci_row_id", np.arange(len(truth)))
    truth["oracle_marginal_outcome"] = (1-truth.true_treatment_prob)*truth.true_y0_prob + truth.true_treatment_prob*truth.true_y1_prob
    records, frames = [], []
    previous = set(manifest["previous_180_heldout_rows"])
    for path in sorted((HERE / "fits").glob("*/*/seed_*/predictions.csv")):
        pred = pd.read_csv(path, float_precision="round_trip")
        assert pred._oci_row_id.tolist() == manifest["heldout_rows"]
        merged = pred.merge(truth, on="_oci_row_id", how="left", validate="one_to_one")
        assert len(merged) == 200 and merged.true_ite_prob.notna().all()
        frames.append(merged)
        for population, frame in [("full_200", merged), ("previous_180", merged[merged._oci_row_id.isin(previous)])]:
            assert len(frame) == (200 if population == "full_200" else 180)
            error = frame.estimated_cate - frame.true_ite_prob
            row = {"population": population, "rows": len(frame), "scenario": frame.scenario.iloc[0],
                   "search": frame.search.iloc[0], "seed": int(frame.seed.iloc[0]),
                   "rmse": float(np.sqrt(np.mean(error**2))), "mae": float(np.abs(error).mean()),
                   "correlation": float(frame.estimated_cate.corr(frame.true_ite_prob)), "bias": float(error.mean()),
                   "mean_cate": float(frame.estimated_cate.mean()), "sd_cate": float(frame.estimated_cate.std(ddof=0)),
                   "oracle_mean": float(frame.true_ite_prob.mean()), "oracle_sd": float(frame.true_ite_prob.std(ddof=0)),
                   "coverage_95": float(((frame.lower_95 <= frame.true_ite_prob) & (frame.true_ite_prob <= frame.upper_95)).mean()),
                   "r_loss": float(np.mean((frame.outcome-frame.outcome_prediction-frame.estimated_cate*(frame.treatment-frame.propensity))**2)),
                   "common_oracle_r_loss": float(np.mean((frame.outcome-frame.oracle_marginal_outcome-frame.estimated_cate*(frame.treatment-frame.true_treatment_prob))**2)),
                   "propensity_rmse": float(np.sqrt(np.mean((frame.propensity-frame.true_treatment_prob)**2))),
                   "marginal_outcome_rmse": float(np.sqrt(np.mean((frame.outcome_prediction-frame.oracle_marginal_outcome)**2)))}
            if population == "full_200":
                assert abs(row["r_loss"]-read(path.parent/"fit_metrics.json")["r_loss"]) < 1e-12
            records.append(row)
    table = pd.DataFrame(records)
    assert len(table) == 36
    table.to_csv(HERE / "metrics_by_seed_2026-09-21.csv", index=False)
    pd.concat(frames, ignore_index=True).to_csv(HERE / "predictions_with_oracle_2026-09-21.csv", index=False)
    numeric = [k for k in records[0] if k not in {"population", "rows", "scenario", "search", "seed"}]
    summaries = []
    for (population, scenario, search), frame in table.groupby(["population", "scenario", "search"]):
        summaries.append({"population": population, "scenario": scenario, "search": search,
                          "metrics": {k: {"mean": float(frame[k].mean()), "min": float(frame[k].min()), "max": float(frame[k].max())} for k in numeric}})
    pairs = []
    for search in SEARCH:
        for seed in SEEDS:
            preds = {s: pd.read_csv(HERE/"fits"/s/search/f"seed_{seed}"/"predictions.csv", float_precision="round_trip") for s in SCENARIOS}
            for first, second in [("a", "b"), ("a", "c")]:
                pairs.append({"search": search, "seed": seed, "first": first, "second": second,
                              "max_absolute_propensity_difference": float(np.max(np.abs(preds[first].propensity-preds[second].propensity))),
                              "max_absolute_outcome_nuisance_difference": float(np.max(np.abs(preds[first].outcome_prediction-preds[second].outcome_prediction))),
                              "max_absolute_cate_difference": float(np.max(np.abs(preds[first].estimated_cate-preds[second].estimated_cate)))})
    write(HERE / "evaluation_2026-09-21.json", {"evaluated_at": now(), "summaries": summaries, "paired_differences": pairs,
          "common_oracle_r_loss_definition": "mean((Y - true_marginal_outcome - estimated_cate*(T - true_propensity))**2); evaluation only",
          "forest_fits": 18, "oracle_dataset_sha256": sha(DATA/"dataset.parquet")})
    verify(frozen["files"])
    verify(manifest["sources"])
    write(HERE / "validation_2026-09-21.json", {"validated_at": now(), "fits": 18,
          "full_heldout_rows_each": 200, "secondary_rows_each": 180, "training_rows": 800,
          "fixed_five_fold_crossfit_coverage": True, "cached_nuisances_unchanged_across_final_forests": True,
          "manual_r_loss_matches_native_score": True, "all_source_and_prediction_hashes_verified": True,
          "finite_ordered_intervals": True, "oracle_effect_values_excluded_from_fitting": True})
    progress("evaluation_complete", fits=18)
    print(table.groupby(["population", "scenario", "search"])[["rmse", "correlation", "r_loss", "common_oracle_r_loss", "sd_cate", "coverage_95"]].mean().to_string(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["fit", "evaluate"])
    args = parser.parse_args()
    fit() if args.phase == "fit" else evaluate()
