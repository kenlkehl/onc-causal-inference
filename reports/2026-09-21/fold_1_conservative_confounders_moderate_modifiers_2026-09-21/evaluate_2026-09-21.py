"""Compare frozen predictions, acknowledging prior oracle exposure."""
from pathlib import Path
import json
from common import HERE, BASE, PRIOR, DATE, n, verify_inputs


def main():
    import numpy as np
    import pandas as pd

    manifest, selection = verify_inputs()
    prediction_path = HERE / f"predictions_frozen_{DATE}.json"
    freeze = n.read(prediction_path)
    assert freeze["selection_freeze_sha256"] == n.sha(HERE / f"selection_frozen_{DATE}.json")
    assert freeze["production_fits"] == freeze["matched_residual_fits"] == 3
    n.verify(freeze["files"])
    prior_freeze = n.read(PRIOR / f"predictions_frozen_{DATE}.json")
    n.verify(prior_freeze["files"])
    marker_path = HERE / "oracle_access_started.json"
    if not marker_path.exists():
        n.write(marker_path, {"at": n.now(), "predictions_freeze_sha256": n.sha(prediction_path),
            "selection_freeze_sha256": n.sha(HERE / f"selection_frozen_{DATE}.json"),
            "source_sha256": n.sha(__file__), "prior_oracle_exposure_acknowledged": True})
    assert n.read(marker_path)["source_sha256"] == n.sha(__file__)
    truth_path = n.ROOT / "synthetic_data/example_synthetic_datasets/five_confounders_five_effect_modifiers_nsclc_with_structured/dataset.parquet"
    assert n.sha(truth_path) == "abed84f263290f20f59a04e565833676fa4b656971117a4b2deaa7a666d4c396"
    truth = pd.read_parquet(truth_path, columns=["true_ite_prob", "treatment_indicator", "outcome_indicator"]).reset_index(drop=True)
    assert len(truth) == 1000
    truth.index.name = "_oci_row_id"
    labels = pd.read_parquet(n.INPUTS / "inputs/heldout_labels.parquet")
    training = pd.read_parquet(n.INPUTS / "inputs/training_labels.parquet")
    for observed in (labels, training):
        rows = truth.iloc[observed._oci_row_id.to_numpy(int)]
        assert np.array_equal(rows.treatment_indicator.to_numpy(), observed.treatment.to_numpy())
        assert np.array_equal(rows.outcome_indicator.to_numpy(), observed.outcome.to_numpy())
    matched_ids = set(labels.loc[labels.effect_eligible, "_oci_row_id"])
    test_ids = set(labels._oci_row_id)
    assert len(matched_ids) == 180 and len(test_ids) == 200

    decisions = n.read(HERE / "role_report.json")["decisions"]
    roles = {d["feature_id"]: d["roles"] for d in decisions}
    prior_recovery_path = PRIOR / f"oracle_recovery_{DATE}.json"
    prior_recovery = n.read(prior_recovery_path)
    recovery = []
    for previous in prior_recovery["direct"]:
        ids, role = previous["candidate_ids"], previous["true_role"]
        recovery.append({"oracle_variable": previous["oracle_variable"], "true_role": role,
            "candidate_ids": ids, "candidate_names": previous["candidate_names"],
            "assigned_roles": {key: roles[key] for key in ids},
            "retained_in_correct_role": any(role in roles[key] for key in ids),
            "retained_in_any_role": any(roles[key] for key in ids),
            "previous_eight_correct_role": previous["retained_in_correct_role"]})
    proxies = [{**p, "assigned_roles": roles[p["feature_id"]]} for p in prior_recovery["proxies"]]
    n.write(HERE / f"oracle_recovery_{DATE}.json", {"direct": recovery, "proxies": proxies,
        "prior_lineage_sha256": n.sha(prior_recovery_path)})
    pd.DataFrame([{**d, "roles": ";".join(d["roles"])} for d in decisions]).to_csv(HERE / f"all_role_decisions_{DATE}.csv", index=False)

    rows, frames = [], []

    def record(frame, method, seed, cohort, *, production=False, diag=None):
        frame = frame.copy()
        assert not frame._oci_row_id.duplicated().any()
        assert set(frame._oci_row_id) <= test_ids
        frame["true_ite_prob"] = frame._oci_row_id.map(truth.true_ite_prob)
        assert frame.true_ite_prob.notna().all()
        estimate = frame.estimated_cate.to_numpy()
        actual = frame.true_ite_prob.to_numpy()
        assert np.isfinite(estimate).all() and len(frame) > 1
        error = estimate - actual
        row = {"method": method, "seed": seed, "cohort": cohort, "n": len(frame),
            "correlation": float(np.corrcoef(estimate, actual)[0, 1]) if np.ptp(estimate) > 1e-12 else None,
            "rmse": float(np.sqrt(np.mean(error ** 2))), "mae": float(np.abs(error).mean()),
            "bias": float(error.mean()), "estimated_mean": float(estimate.mean()),
            "true_mean": float(actual.mean()), "estimated_sd": float(estimate.std()), "true_sd": float(actual.std())}
        lower = "estimated_cate_lower_95" if production else "lower_95"
        upper = "estimated_cate_upper_95" if production else "upper_95"
        row["coverage_95"] = float(((frame[lower] <= actual) & (actual <= frame[upper])).mean())
        if production:
            aipw = frame.aipw_score.to_numpy()
            assert np.isfinite(aipw).all()
            se = float(aipw.std(ddof=1) / np.sqrt(len(aipw)))
            ate = float(aipw.mean())
            row.update(ate_aipw=ate, ate_aipw_se=se, ate_aipw_lower_95=ate - 1.96 * se,
                       ate_aipw_upper_95=ate + 1.96 * se, ate_aipw_bias=ate - float(actual.mean()))
            if diag:
                assert np.isclose(ate, diag["ate_aipw"], rtol=0, atol=1e-12)
                row["fit_n"] = diag["effect_fit_rows"]
                row["confounders"] = diag["confounders"]
                row["modifiers"] = diag["effect_modifiers"]
        else:
            residual = frame.outcome - frame.outcome_prediction - (frame.treatment - frame.propensity) * frame.estimated_cate
            row["r_loss"] = float(np.mean(residual ** 2))
        rows.append(row)
        frame["method"], frame["seed"], frame["cohort"] = method, seed, cohort
        frames.append(frame)

    for seed in manifest["evaluation_plan"]["matched_forest_seeds"]:
        for method, root in [("matched_224", PRIOR / "broad_matched"),
                             ("matched_8", PRIOR / "matched_residuals"),
                             ("matched_16", HERE / "matched_residuals")]:
            frame = pd.read_csv(root / f"seed_{seed}/predictions.csv", float_precision="round_trip")
            assert set(frame._oci_row_id) == matched_ids
            record(frame, method, seed, "fixed_180")
    intersections = []
    for seed in manifest["evaluation_plan"]["estimation_seeds"]:
        pair = []
        for method, root in [("production_8C_8M", PRIOR), ("production_189C_16M", HERE)]:
            path = root / f"production/seed_{seed}/predictions.csv"
            frame = pd.read_csv(path, float_precision="round_trip")
            assert set(frame._oci_row_id) == test_ids and not frame._oci_row_id.duplicated().any()
            assert np.array_equal(frame.effect_eligible, frame.estimated_cate.notna())
            frame = frame.loc[frame.effect_eligible].copy()
            diag = n.read(path.parent / "diagnostics.json")
            assert len(frame) == diag["effect_estimation_rows"]
            record(frame, method, seed, "native_eligible", production=True, diag=diag)
            pair.append((method, frame))
        common_ids = set(pair[0][1]._oci_row_id) & set(pair[1][1]._oci_row_id)
        intersections.append({"seed": seed, "n": len(common_ids), "row_ids": sorted(common_ids)})
        for method, frame in pair:
            record(frame.loc[frame._oci_row_id.isin(common_ids)], method, seed, "common_eligible", production=True)
    table = pd.DataFrame(rows)
    table.to_csv(HERE / f"metrics_by_seed_{DATE}.csv", index=False)
    pd.concat(frames, ignore_index=True).to_csv(HERE / f"predictions_with_oracle_{DATE}.csv", index=False)
    summaries = {}
    columns = ["n", "fit_n", "correlation", "rmse", "bias", "estimated_sd", "true_sd", "true_mean",
               "estimated_mean", "coverage_95", "ate_aipw", "ate_aipw_bias", "r_loss"]
    for (cohort, method), frame in table.groupby(["cohort", "method"], sort=False):
        summaries.setdefault(cohort, {})[method] = {
            column: {"mean": float(frame[column].mean()), "min": float(frame[column].min()), "max": float(frame[column].max())}
            for column in columns if frame[column].notna().any()}
    contrasts = []
    for cohort, earlier, current in [("fixed_180", "matched_8", "matched_16"),
                                      ("fixed_180", "matched_224", "matched_16"),
                                      ("native_eligible", "production_8C_8M", "production_189C_16M"),
                                      ("common_eligible", "production_8C_8M", "production_189C_16M")]:
        a = table.loc[(table.cohort == cohort) & (table.method == earlier)].set_index("seed")
        b = table.loc[(table.cohort == cohort) & (table.method == current)].set_index("seed")
        assert set(a.index) == set(b.index) and len(a) == 3
        contrasts.append({"cohort": cohort, "earlier": earlier, "current": current,
            "current_minus_earlier": {key: float((b[key] - a[key]).mean()) for key in ["correlation", "rmse", "bias"]}})
    result = {"evaluated_at": n.now(), "summaries": summaries, "contrasts": contrasts,
        "direct_recovery": {role: {"correct_role": sum(r["retained_in_correct_role"] for r in recovery if r["true_role"] == role),
                                  "any_role": sum(r["retained_in_any_role"] for r in recovery if r["true_role"] == role), "total": 5}
                            for role in ["confounder", "effect_modifier"]},
        "selection": {k: selection[k] for k in ["selected_count", "roles", "modifier_order"]},
        "common_cohorts": intersections, "oracle_source_sha256": n.sha(truth_path),
        "previous_prediction_freeze_sha256": n.sha(PRIOR / f"predictions_frozen_{DATE}.json"),
        "prediction_freeze_sha256": n.sha(prediction_path), "source_sha256": n.sha(__file__),
        "exploratory_post_hoc_comparison": True, "oracle_values_used_in_fitting": False,
        "prior_oracle_results_informed_experiment_request": True}
    n.write(HERE / f"evaluation_{DATE}.json", result)
    n.write(HERE / "status.json", {"phase": "evaluation_complete", "updated_at": n.now()})
    print(json.dumps({"direct_recovery": result["direct_recovery"], "summaries": summaries}), flush=True)


if __name__ == "__main__":
    main()
