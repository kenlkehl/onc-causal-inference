"""Separate, explicitly oracle-informed feature-identity diagnostic for fold 1.

The user authorized selecting candidates matching known modifier identities.
Predictors are the frozen LLM measurements; latent oracle values and effects do
not enter forest fitting. Nuisances and the running main experiment stay fixed.
Run fit, then evaluate with /home/klkehl/thisenv/bin/python.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PRIOR = HERE.parent / "fold_1_all_candidates_all_split_features_2026-09-21"
spec = importlib.util.spec_from_file_location("prior_exploratory_helpers", PRIOR / "run_exploratory_comparison.py")
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
c = prior.comparison
read, write, sha, verify = prior.read, prior.write, prior.sha, prior.verify
FOLD, SOURCE, SEEDS = prior.FOLD, prior.SOURCE, prior.SEEDS

import numpy as np
import pandas as pd

CONCEPTS = {
    "histology_type": ["outer_001_feature_192"],
    "egfr_mutation_status": ["outer_001_feature_032", "outer_001_feature_111"],
    "baseline_nlr": ["outer_001_feature_228"],
    "brain_metastases_status": ["outer_001_feature_052"],
    "baseline_hemoglobin": ["outer_001_feature_155"],
}
METHODS = {"oracle_modifier_candidates_sqrt": "sqrt", "oracle_modifier_candidates_all": 1.0}


def reconstruct_inputs(files):
    from oci.inference.plain_handoff_stage2_analysis import (
        _apply_harmonization_plans, _feature_extraction_fingerprint,
    )

    def remember(path):
        files[str(path.resolve())] = sha(path)
        return path

    inp = read(FOLD / "selection/input.json")
    definitions = inp["definitions"]
    split = next(s for s in read(SOURCE / "inputs/splits.json") if s["outer_fold"] == 1)
    names = [d["name"] for d in definitions]
    training = c.align(c.frozen_training_frame(FOLD, inp, definitions), split["fit_row_ids"], names)
    final = read(FOLD / "final_definitions.json")["features"]
    by_id = {d["feature_id"]: d for d in final}
    reused = [d for d in definitions if d["feature_id"] in by_id and
              _feature_extraction_fingerprint(d) == _feature_extraction_fingerprint(by_id[d["feature_id"]])]
    reused_names = [d["name"] for d in reused]
    additional_names = [d["name"] for d in definitions if d["name"] not in reused_names]
    standard = c.align(pd.read_csv(FOLD / "extraction/heldout/extracted.csv"),
                       split["heldout_row_ids"])[["_oci_row_id", *reused_names]]
    records = []
    for position, row_id in enumerate(split["heldout_row_ids"], 1):
        directory = FOLD / "comparison_measurements/heldout_additional/batches" / f"batch_{position:05d}"
        assert read(directory / "complete.json")["status"] == "complete"
        rows = read(directory / "result.json")["rows"]
        assert len(rows) == 1 and rows[0]["row_id"] == row_id
        assert set(rows[0]["values"]) == set(additional_names)
        records.append({"_oci_row_id": row_id, **{name: rows[0]["values"][name] for name in additional_names}})
    additional = pd.DataFrame(records, columns=["_oci_row_id", *additional_names])
    raw = c.align(standard.merge(additional, on="_oci_row_id", validate="one_to_one"),
                  split["heldout_row_ids"], names)
    heldout, _ = _apply_harmonization_plans(raw, definitions, scope="outer_heldout")
    statistical = read(FOLD / "selection/statistical_evidence.json")
    train_nuisance = c.align(pd.DataFrame(statistical["cross_fitted_nuisance_models"]["predictions"]),
                             split["fit_row_ids"])
    test_nuisance = pd.read_csv(FOLD / "comparison_nuisances/heldout.csv", float_precision="round_trip")
    assert train_nuisance._oci_row_id.tolist() == split["fit_row_ids"]
    assert test_nuisance._oci_row_id.tolist() == split["heldout_row_ids"]
    assert not set(split["fit_row_ids"]) & set(split["heldout_row_ids"])
    # Read only the identity mapping from the earlier lineage audit.
    lineage_path = prior.INTERIM / "fold_1_feature_recovery_2026-09-21.json"
    lineage = read(remember(lineage_path))
    mapped = {d["concept"]: set(d["direct_feature_ids"]) for d in lineage["concepts"]
              if "effect_modifier" in d["oracle_roles"]}
    assert mapped == {name: set(ids) for name, ids in CONCEPTS.items()}
    return definitions, training, heldout, train_nuisance, test_nuisance


def fit():
    from econml.grf import CausalForest
    from oci.inference.plain_handoff_stage2_analysis import _FeatureEncoder

    if (HERE / "predictions_frozen_2026-09-21.json").exists():
        raise ValueError("Diagnostic already frozen; do not overwrite it")
    prior_manifest = read(PRIOR / "input_manifest_2026-09-21.json")
    prior_freeze = read(PRIOR / "predictions_frozen_2026-09-21.json")
    assert sha(PRIOR / "input_manifest_2026-09-21.json") == prior_freeze["input_manifest_sha256"]
    files = {**prior_manifest["files"], **prior_freeze["files"]}
    files[str((PRIOR / "input_manifest_2026-09-21.json").resolve())] = sha(PRIOR / "input_manifest_2026-09-21.json")
    files[str((PRIOR / "predictions_frozen_2026-09-21.json").resolve())] = sha(PRIOR / "predictions_frozen_2026-09-21.json")
    verify(files)
    c.validate_run_inputs(SOURCE, SOURCE / "revisions/adjudication_budget_v4/manifest.json")
    definitions, training, heldout, train_nuisance, test_nuisance = reconstruct_inputs(files)
    wanted = {fid for ids in CONCEPTS.values() for fid in ids}
    subset = [d for d in definitions if d["feature_id"] in wanted]
    assert len(subset) == len(wanted) == 6
    tr = train_nuisance.effect_eligible.to_numpy(dtype=bool)
    te = test_nuisance.effect_eligible.to_numpy(dtype=bool)
    assert int(tr.sum()) == 720 and int(te.sum()) == 180
    full_encoder = _FeatureEncoder(definitions).fit(training.loc[tr].reset_index(drop=True))
    full_train = full_encoder.transform(training.loc[tr].reset_index(drop=True))
    full_test = full_encoder.transform(heldout.loc[te].reset_index(drop=True))
    assert hashlib.sha256(full_train.tobytes()).hexdigest() == prior_manifest["train_design"]
    assert hashlib.sha256(full_test.tobytes()).hexdigest() == prior_manifest["test_design"]
    encoder = _FeatureEncoder(subset).fit(training.loc[tr].reset_index(drop=True))
    x_train = encoder.transform(training.loc[tr].reset_index(drop=True))
    x_test = encoder.transform(heldout.loc[te].reset_index(drop=True))
    columns, offset, encoding = [], 0, []
    for definition, (name, kind, parameters) in zip(definitions, full_encoder.encodings, strict=True):
        width = (2 if kind == "continuous" else 4 + len(parameters[2]) if kind == "continuous_with_categorical_fallback"
                 else len(parameters) + int(kind == "categorical_with_other"))
        if definition["feature_id"] in wanted:
            columns.extend(range(offset, offset + width))
            encoding.append({"feature_id": definition["feature_id"], "name": name,
                             "encoding": kind, "parameters": parameters, "encoded_columns": width})
        offset += width
    assert offset == 1096 and len(columns) == x_train.shape[1]
    np.testing.assert_array_equal(x_train, full_train[:, columns])
    np.testing.assert_array_equal(x_test, full_test[:, columns])
    assert np.isfinite(x_train).all() and np.isfinite(x_test).all()
    nuisance_hash = c.fingerprint({"train": train_nuisance.to_json(), "test": test_nuisance.to_json()})
    assert nuisance_hash == prior_manifest["nuisance_hash"]
    t_res = (train_nuisance.treatment - train_nuisance.propensity).to_numpy()[tr]
    y_res = (train_nuisance.outcome - train_nuisance.outcome_prediction).to_numpy()[tr]
    constant = float(np.dot(t_res, y_res) / np.dot(t_res, t_res))
    assert constant == prior_manifest["constant_effect"]
    planned = []
    for method, max_features in METHODS.items():
        for replicate, seed in enumerate(SEEDS):
            old = read(FOLD / "comparison_forests/all_candidates" / f"seed_{seed}" / "metrics.json")
            params = {**old["forest_parameters"], "max_features": max_features}
            planned.append({"method": method, "replicate": replicate, "seed": seed,
                            "parameters": params, "effective_parameters": CausalForest(**params).get_params()})
    files[str(Path(__file__).resolve())] = sha(__file__)
    manifest = {"created_at": c.now(), "purpose": "User-requested oracle-identity candidate restriction on fold 1, with existing extracted measurements and shared nuisance estimates.",
                "oracle_identities_used_for_candidate_selection": True,
                "latent_oracle_values_used_as_predictors_or_responses": False,
                "prior_oracle_results_seen": True, "ongoing_experiment_modified": False,
                "concepts": CONCEPTS, "selected_definitions": subset, "encodings": encoding,
                "candidate_features": len(subset), "modifier_concepts": len(CONCEPTS),
                "encoded_columns": x_train.shape[1], "fit_rows": int(tr.sum()), "heldout_rows": int(te.sum()),
                "train_row_ids": train_nuisance.loc[tr, "_oci_row_id"].tolist(),
                "heldout_row_ids": test_nuisance.loc[te, "_oci_row_id"].tolist(),
                "nuisance_hash": nuisance_hash, "constant_effect": constant,
                "train_design": hashlib.sha256(x_train.tobytes()).hexdigest(),
                "test_design": hashlib.sha256(x_test.tobytes()).hexdigest(),
                "full_design_matches_original": True, "subset_columns_match_original_exactly": True,
                "files": files, "fits": planned, "versions": prior_manifest["versions"]}
    write(HERE / "input_manifest_2026-09-21.json", manifest)
    print({"phase": "inputs_frozen", "features": len(subset), "columns": x_train.shape[1]}, flush=True)
    frozen = {}
    for plan in planned:
        started = time.monotonic()
        model = CausalForest(**plan["parameters"]).fit(x_train, t_res, y_res)
        values = model.predict(x_test, interval=True)
        prediction = test_nuisance.loc[te].reset_index(drop=True).copy()
        prediction["outer_fold"], prediction["method"], prediction["replicate"] = 1, plan["method"], plan["replicate"]
        for name, array in zip(["estimated_cate", "lower_95", "upper_95"], values, strict=True):
            prediction[name] = array.ravel()
        assert np.isfinite(prediction[["estimated_cate", "lower_95", "upper_95"]].to_numpy()).all()
        assert (prediction.lower_95 <= prediction.estimated_cate).all()
        assert (prediction.estimated_cate <= prediction.upper_95).all()
        dest = HERE / "fits" / plan["method"] / f"seed_{plan['seed']}"
        dest.mkdir(parents=True, exist_ok=True)
        prediction.to_csv(dest / "predictions.csv", index=False)
        metrics = c.loss_metrics(prediction.treatment, prediction.outcome, prediction.propensity,
                                 prediction.outcome_prediction, prediction.estimated_cate, constant)
        metrics.update(method=plan["method"], seed=plan["seed"], replicate=plan["replicate"],
                       elapsed_seconds=time.monotonic() - started, nuisance_hash=nuisance_hash,
                       constant_effect=constant, parameters=plan["parameters"])
        write(dest / "metrics.json", metrics)
        for path in [dest / "predictions.csv", dest / "metrics.json"]:
            frozen[str(path.resolve())] = sha(path)
        print({"phase": "fit_complete", "method": plan["method"], "seed": plan["seed"], "r_loss": metrics["r_loss"]}, flush=True)
    verify(files)
    write(HERE / "predictions_frozen_2026-09-21.json", {"frozen_at": c.now(), "new_fits": 6,
          "input_manifest_sha256": sha(HERE / "input_manifest_2026-09-21.json"), "files": frozen,
          "oracle_effect_values_read_by_fit_phase": False,
          "oracle_identities_used_for_candidate_selection": True})


def evaluate():
    frozen = read(HERE / "predictions_frozen_2026-09-21.json")
    assert frozen["new_fits"] == 6
    assert sha(HERE / "input_manifest_2026-09-21.json") == frozen["input_manifest_sha256"]
    manifest = read(HERE / "input_manifest_2026-09-21.json")
    verify(frozen["files"])
    verify(manifest["files"])
    oracle_path = ROOT / "artifacts/research_all_evidence/five_conf_five_mod_nsclc_full/stage2/posthoc_predictions_with_oracle_ite.csv"
    truth = pd.read_csv(oracle_path, usecols=["_oci_row_id", "true_ite_prob"], float_precision="round_trip")
    assert not truth._oci_row_id.duplicated().any()
    records, frames = [], []
    for method in [*c.METHODS, prior.METHOD, *METHODS]:
        for replicate, seed in enumerate(SEEDS):
            directory = (HERE / "fits" / method / f"seed_{seed}" if method in METHODS else
                         PRIOR / "fits" / f"seed_{seed}" if method == prior.METHOD else
                         FOLD / "comparison_forests" / method / f"seed_{seed}")
            prediction = pd.read_csv(directory / "predictions.csv", float_precision="round_trip")
            assert prediction._oci_row_id.tolist() == manifest["heldout_row_ids"]
            reference = pd.read_csv(FOLD / "comparison_forests/all_candidates" / f"seed_{seed}" / "predictions.csv",
                                    float_precision="round_trip")
            fields = ["_oci_row_id", "treatment", "outcome", "propensity", "outcome_prediction", "effect_eligible", "outer_fold", "replicate"]
            pd.testing.assert_frame_equal(prediction[fields], reference[fields], check_exact=True)
            frame = prediction.merge(truth, on="_oci_row_id", how="left", validate="one_to_one")
            assert len(frame) == 180 and frame.true_ite_prob.notna().all()
            error = frame.estimated_cate - frame.true_ite_prob
            result = c.loss_metrics(frame.treatment, frame.outcome, frame.propensity, frame.outcome_prediction,
                                    frame.estimated_cate, manifest["constant_effect"])
            saved = read(directory / "metrics.json")
            for metric in ["r_loss", "constant_r_loss", "mean_cate", "sd_cate"]:
                assert abs(result[metric] - saved[metric]) < 1e-12
            result.update(method=method, seed=seed, replicate=replicate,
                          rmse=float(np.sqrt(np.mean(error ** 2))), mae=float(np.abs(error).mean()),
                          bias=float(error.mean()), correlation=float(frame.estimated_cate.corr(frame.true_ite_prob)),
                          oracle_mean=float(frame.true_ite_prob.mean()), oracle_sd=float(frame.true_ite_prob.std(ddof=0)),
                          constant_rmse=float(np.sqrt(np.mean((manifest["constant_effect"] - frame.true_ite_prob) ** 2))),
                          coverage_95=float(((frame.lower_95 <= frame.true_ite_prob) & (frame.true_ite_prob <= frame.upper_95)).mean()))
            records.append(result)
            frames.append(frame)
    table = pd.DataFrame(records)
    assert len(table) == 21
    table.to_csv(HERE / "metrics_by_seed_2026-09-21.csv", index=False)
    pd.concat(frames, ignore_index=True).to_csv(HERE / "predictions_with_oracle_2026-09-21.csv", index=False)
    metrics = ["rmse", "mae", "correlation", "r_loss", "r_score", "mean_cate", "sd_cate", "bias", "coverage_95"]
    summary = {method: {m: {"mean": float(frame[m].mean()), "min": float(frame[m].min()), "max": float(frame[m].max())}
                        for m in metrics} for method, frame in table.groupby("method", sort=False)}
    write(HERE / "evaluation_2026-09-21.json", {"evaluated_at": c.now(), "summaries": summary,
          "oracle_sha256": sha(oracle_path), "freeze_sha256": sha(HERE / "predictions_frozen_2026-09-21.json"),
          "oracle_mean": records[0]["oracle_mean"], "oracle_sd": records[0]["oracle_sd"],
          "constant_rmse": records[0]["constant_rmse"], "constant_r_loss": records[0]["constant_r_loss"]})
    verify(frozen["files"])
    verify(manifest["files"])
    write(HERE / "validation_2026-09-21.json", {"validated_at": c.now(), "evaluated_fits": 21,
          "new_fits": 6, "heldout_rows_each": 180, "input_files_verified": len(manifest["files"]),
          "frozen_new_artifacts_verified": len(frozen["files"]), "shared_rows_nuisances_and_observed_values_exact": True,
          "original_full_design_and_subset_columns_exact": True, "saved_fitting_metrics_recomputed": True,
          "oracle_identities_used_for_candidate_selection": True, "oracle_values_excluded_from_predictors_and_responses": True})
    print(table[["method", "seed", "rmse", "correlation", "r_loss", "sd_cate", "coverage_95"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["fit", "evaluate"])
    args = parser.parse_args()
    fit() if args.phase == "fit" else evaluate()
