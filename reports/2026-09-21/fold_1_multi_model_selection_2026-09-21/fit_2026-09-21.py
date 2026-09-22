"""Fit frozen multi-model selections; no oracle data are opened in this phase."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("numerical_run", HERE / "numerical_2026-09-21.py")
n = importlib.util.module_from_spec(spec)
spec.loader.exec_module(n)


def main():
    import numpy as np
    import pandas as pd
    from threadpoolctl import threadpool_limits
    from econml.grf import CausalForest
    from oci.inference.plain_handoff_stage2_analysis import _FeatureEncoder, estimate_outer_fold

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    manifest = n.read(HERE / f"input_manifest_{n.DATE}.json")
    selected_freeze = n.read(HERE / f"selection_frozen_{n.DATE}.json")
    assert selected_freeze["input_manifest_sha256"] == n.sha(HERE / f"input_manifest_{n.DATE}.json")
    n.verify(selected_freeze["files"])
    n.verify(manifest["sources"])
    n.verify(manifest["input_files"])
    final_freeze = HERE / f"predictions_frozen_{n.DATE}.json"
    if final_freeze.exists():
        n.verify(n.read(final_freeze)["files"])
        print("Predictions already frozen and verified", flush=True)
        return
    policy = manifest["evaluation_plan"]
    inputs = n.INPUTS / "inputs"
    definitions = n.read(HERE / "selected_definitions.json")["features"]
    split = n.read(inputs / "split.json")
    train = pd.read_pickle(inputs / "training.pkl")
    test = pd.read_pickle(inputs / "heldout.pkl")
    train_labels = pd.read_parquet(inputs / "training_labels.parquet")
    test_labels = pd.read_parquet(inputs / "heldout_labels.parquet")
    assert train._oci_row_id.tolist() == train_labels._oci_row_id.tolist() == split["fit_row_ids"]
    assert test._oci_row_id.tolist() == test_labels._oci_row_id.tolist() == split["heldout_row_ids"]
    dataset = pd.concat([train_labels, test_labels]).sort_values("_oci_row_id")[["_oci_row_id", "treatment", "outcome"]].reset_index(drop=True)
    assert dataset._oci_row_id.tolist() == list(range(1000))
    # A local row key is sufficient; real patient IDs never need to be loaded.
    dataset["analysis_unit"] = dataset._oci_row_id
    fit_manifest = {
        "source_sha256": n.sha(Path(__file__)),
        "selection_freeze_sha256": n.sha(HERE / f"selection_frozen_{n.DATE}.json"),
        "input_manifest_sha256": n.sha(HERE / f"input_manifest_{n.DATE}.json"),
        "production_seeds": policy["estimation_seeds"], "matched_forest_seeds": policy["matched_forest_seeds"],
        "production_policy": {k: policy[k] for k in ["trees", "min_propensity", "max_propensity", "propensity_clip"]},
        "oracle_read": False,
    }
    path = HERE / "fit_input.json"
    if path.exists():
        assert n.read(path) == fit_manifest, "Fitting procedure changed on resume"
    else:
        n.write(path, fit_manifest)
    frozen = {str(path.resolve()): n.sha(path), str(Path(__file__).resolve()): n.sha(__file__)}

    with threadpool_limits(limits=1):
        for seed in policy["estimation_seeds"]:
            dest = HERE / "production" / f"seed_{seed}"
            n.write(HERE / "status.json", {"phase": "production_forest", "seed": seed, "updated_at": n.now()})
            start = time.monotonic()
            diagnostics = estimate_outer_fold(
                dataset=dataset, extracted_fit=train, extracted_heldout=test, definitions=definitions,
                split=split, unit_id_column="analysis_unit", treatment_column="treatment", outcome_column="outcome",
                outcome_type="binary", inner_folds=5, seed=seed,
                propensity_clip=policy["propensity_clip"], min_propensity=policy["min_propensity"],
                max_propensity=policy["max_propensity"], estimation_trees=policy["trees"], output_dir=dest,
            )
            assert diagnostics["nuisance_model_family"] == "elastic_net"
            assert diagnostics["causal_forest_fit_audit"]
            for file in dest.glob("*"):
                if file.is_file():
                    frozen[str(file.resolve())] = n.sha(file)
            print(json.dumps({"phase": "production_fit_complete", "seed": seed,
                              "seconds": time.monotonic() - start,
                              "fit_eligible": diagnostics["effect_fit_rows"],
                              "test_eligible": diagnostics["effect_estimation_rows"]}), flush=True)

        # Selection-only comparison: keep the earlier all-candidate elastic-net
        # residuals, patients, seeds, and forest settings exactly fixed.
        tr = train_labels.effect_eligible.to_numpy(bool)
        te = test_labels.effect_eligible.to_numpy(bool)
        assert tr.sum() == 720 and te.sum() == 180
        modifiers = [d for d in definitions if "effect_modifier" in d["roles"]]
        encoder = _FeatureEncoder(modifiers).fit(train.loc[tr].reset_index(drop=True))
        x = encoder.transform(train.loc[tr].reset_index(drop=True))
        xv = encoder.transform(test.loc[te].reset_index(drop=True))
        if x.shape[1] == 0:
            x, xv = np.ones((tr.sum(), 1)), np.ones((te.sum(), 1))
        tres = (train_labels.treatment - train_labels.propensity).to_numpy()[tr]
        yres = (train_labels.outcome - train_labels.outcome_prediction).to_numpy()[tr]
        constant = float(tres @ yres / (tres @ tres))
        for seed in policy["matched_forest_seeds"]:
            dest = HERE / "matched_residuals" / f"seed_{seed}"
            dest.mkdir(parents=True, exist_ok=True)
            n.write(HERE / "status.json", {"phase": "matched_residual_forest", "seed": seed, "updated_at": n.now()})
            params = dict(n_estimators=200, max_depth=None, min_samples_leaf=10, max_features="sqrt",
                          honest=True, inference=True, subforest_size=4, n_jobs=1, random_state=seed)
            old = n.read(n.SOURCE / f"refresh/outer_001/comparison_forests/all_candidates/seed_{seed}/metrics.json")
            assert params == old["forest_parameters"]
            model = CausalForest(**params).fit(x, tres, yres)
            cate, lo, hi = model.predict(xv, interval=True)
            pred = test_labels.loc[te].reset_index(drop=True).copy()
            pred["estimated_cate"], pred["lower_95"], pred["upper_95"] = cate.ravel(), lo.ravel(), hi.ravel()
            assert np.isfinite(pred[["estimated_cate", "lower_95", "upper_95"]]).all().all()
            pred.to_csv(dest / "predictions.csv", index=False)
            test_tr = pred.treatment - pred.propensity
            test_yr = pred.outcome - pred.outcome_prediction
            loss = float(np.mean((test_yr - test_tr * pred.estimated_cate) ** 2))
            null_loss = float(np.mean((test_yr - test_tr * constant) ** 2))
            n.write(dest / "metrics.json", {
                "seed": seed, "parameters": params, "effective_parameters": model.get_params(),
                "modifiers": len(modifiers), "encoded_columns": x.shape[1], "fit_rows": 720, "test_rows": 180,
                "constant_effect": constant, "r_loss": loss, "r_score": 1 - loss / null_loss,
                "train_design_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
                "test_design_sha256": hashlib.sha256(xv.tobytes()).hexdigest(),
            })
            for file in dest.glob("*"):
                if file.is_file():
                    frozen[str(file.resolve())] = n.sha(file)
            print(json.dumps({"phase": "matched_residual_fit_complete", "seed": seed, "r_loss": loss}), flush=True)

    n.verify(selected_freeze["files"])
    n.verify(manifest["sources"])
    n.verify(manifest["input_files"])
    n.write(final_freeze, {"frozen_at": n.now(), "files": frozen,
                          "selection_freeze_sha256": n.sha(HERE / f"selection_frozen_{n.DATE}.json"),
                          "production_fits": 3, "matched_residual_fits": 3, "oracle_read_by_fit": False})
    n.write(HERE / "status.json", {"phase": "predictions_frozen", "updated_at": n.now()})


if __name__ == "__main__":
    main()
