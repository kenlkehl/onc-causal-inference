"""Post-hoc oracle evaluation, permitted only after selection/prediction freezing."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
from common import n, BASE


def main():
    import numpy as np
    import pandas as pd

    pred_freeze = n.read(HERE / f"predictions_frozen_{n.DATE}.json")
    selection_path = HERE / f"selection_frozen_{n.DATE}.json"
    assert pred_freeze["selection_freeze_sha256"] == n.sha(selection_path)
    selection_freeze = n.read(selection_path)
    manifest_path = HERE / f"input_manifest_{n.DATE}.json"
    assert selection_freeze["input_manifest_sha256"] == n.sha(manifest_path)
    manifest = n.read(manifest_path)
    n.verify(pred_freeze["files"])
    n.verify(selection_freeze["files"])
    n.verify(manifest["sources"])
    n.verify(manifest["input_files"])
    n.verify(n.read(BASE / f"selection_frozen_{n.DATE}.json")["files"])
    assert pred_freeze["production_fits"] == pred_freeze["matched_residual_fits"] == pred_freeze["broad_matched_fits"] == 3
    n.write(HERE / "oracle_access_started.json", {
        "at": n.now(), "predictions_freeze_sha256": n.sha(HERE / f"predictions_frozen_{n.DATE}.json"),
        "selection_freeze_sha256": n.sha(selection_path), "evaluation_source_sha256": n.sha(__file__),
        "oracle_used_in_fitting": False,
    })
    truth_path = n.ROOT / "synthetic_data/example_synthetic_datasets/five_confounders_five_effect_modifiers_nsclc_with_structured/dataset.parquet"
    lineage_path = HERE.parent / "fold_1_interim_oracle_review_2026-09-21/fold_1_feature_recovery_2026-09-21.json"
    lineage_reference = n.read(lineage_path)
    assert n.sha(truth_path) == "abed84f263290f20f59a04e565833676fa4b656971117a4b2deaa7a666d4c396"
    truth = pd.read_parquet(truth_path, columns=["true_ite_prob"])
    assert len(truth) == 1000
    truth["_oci_row_id"] = np.arange(1000)
    truth = truth.set_index("_oci_row_id").true_ite_prob
    definitions = n.read(n.INPUTS / "inputs/definitions.json")
    by_id = {d["feature_id"]: d for d in definitions}
    role_report = n.read(HERE / "role_report.json")
    decisions = {r["feature_id"]: r for r in role_report["decisions"]}

    # Direct lineage is inherited from the earlier frozen fold-1 recovery audit,
    # and is introduced only in this post-hoc script, never in the selector.
    direct = [
        ("age", "confounder", [9]), ("sex", "confounder", [302]),
        ("ECOG", "confounder", [109]), ("creatinine clearance", "confounder", [95]),
        ("prior platinum exposure", "confounder", [267]),
        ("histology", "effect_modifier", [192]), ("EGFR status", "effect_modifier", [32, 111]),
        ("NLR", "effect_modifier", [228]), ("brain metastasis presence", "effect_modifier", [52]),
        ("hemoglobin", "effect_modifier", [155]),
    ]
    recovery = []
    for name, role, numbers in direct:
        candidates = [f"outer_001_feature_{i:03d}" for i in numbers]
        assert all(k in by_id for k in candidates)
        recovery.append({
            "oracle_variable": name, "true_role": role, "candidate_ids": candidates,
            "candidate_names": [by_id[k]["name"] for k in candidates],
            "assigned_roles": {k: decisions[k]["roles"] for k in candidates},
            "retained_in_correct_role": any(role in decisions[k]["roles"] for k in candidates),
            "retained_in_any_role": any(decisions[k]["roles"] for k in candidates),
            "decisions": [decisions[k] for k in candidates],
        })
    proxies = []
    for proxy, target, number in [("gender identity", "sex", 140),
                                   ("brain lesion count", "brain metastasis presence", 49),
                                   ("hematocrit", "hemoglobin", 153),
                                   ("eGFR", "creatinine clearance", 125)]:
        key = f"outer_001_feature_{number:03d}"
        proxies.append({"proxy": proxy, "related_oracle_variable": target, "feature_id": key,
                        "candidate_name": by_id[key]["name"], "assigned_roles": decisions[key]["roles"],
                        "counted_as_direct_recovery": False})
    n.write(HERE / f"oracle_recovery_{n.DATE}.json", {"direct": recovery, "proxies": proxies})
    broad = {d["feature_id"]: d["roles"] for d in n.read(BASE / "role_report.json")["decisions"]}
    global_review = n.read(HERE / "global_response.json")
    global_roles = {key: [] for key in by_id}
    for role, field in [("confounder", "confounders"), ("effect_modifier", "modifier_ranking")]:
        for entry in global_review[field]:
            global_roles["outer_001_feature_" + entry["id"]].append(role)
    funnel = []
    for r in recovery:
        role, candidates = r["true_role"], r["candidate_ids"]
        funnel.append({"oracle_variable": r["oracle_variable"], "true_role": role,
                       "broad_correct_role": any(role in broad[k] for k in candidates),
                       "global_review_correct_role": any(role in global_roles[k] for k in candidates),
                       "final_correct_role": r["retained_in_correct_role"],
                       "broad_roles": {k: broad[k] for k in candidates},
                       "global_roles": {k: global_roles[k] for k in candidates},
                       "final_roles": r["assigned_roles"]})
    n.write(HERE / f"recovery_funnel_{n.DATE}.json", {"variables": funnel, "source": "post-freeze oracle lineage audit"})
    selection_table = [{"feature_id": k, "name": by_id[k]["name"], "roles": ";".join(d["roles"]),
                        "stability": d["stability"], "rationale": d["rationale"],
                        "evidence_ids": ";".join(d["evidence_ids"])} for k, d in decisions.items()]
    pd.DataFrame(selection_table).to_csv(HERE / f"all_role_decisions_{n.DATE}.csv", index=False)

    reference_dir = HERE.parent / "fold_1_all_candidates_all_split_features_2026-09-21"
    old_hashes = n.read(reference_dir / f"input_manifest_{n.DATE}.json")["files"]
    old_hashes.update(n.read(reference_dir / f"predictions_frozen_{n.DATE}.json")["files"])
    baseline_files, evaluations = {}, []
    for seed in manifest["evaluation_plan"]["matched_forest_seeds"]:
        for method in ["current_joint", "llm_adjudication", "permissive_union", "all_candidates"]:
            path = n.SOURCE / f"refresh/outer_001/comparison_forests/{method}/seed_{seed}/predictions.csv"
            baseline_files[str(path.resolve())] = old_hashes[str(path.resolve())]
            evaluations.append((method, seed, path, "matched_180"))
        path = reference_dir / f"fits/seed_{seed}/predictions.csv"
        baseline_files[str(path.resolve())] = old_hashes[str(path.resolve())]
        evaluations.append(("all_candidates_all_split_features", seed, path, "matched_180"))
        evaluations.append(("multi_model_matched", seed, HERE / f"matched_residuals/seed_{seed}/predictions.csv", "matched_180"))
        evaluations.append(("multi_model_broad", seed, HERE / f"broad_matched/seed_{seed}/predictions.csv", "matched_180"))
    n.verify(baseline_files)
    for seed in manifest["evaluation_plan"]["estimation_seeds"]:
        evaluations.append(("multi_model_production", seed, HERE / f"production/seed_{seed}/predictions.csv", "production_eligible"))
    matched_ids = pd.read_parquet(n.INPUTS / "inputs/heldout_labels.parquet")
    matched_ids = set(matched_ids.loc[matched_ids.effect_eligible, "_oci_row_id"])
    assert len(matched_ids) == 180
    rows, prediction_frames = [], []
    for method, seed, path, cohort in evaluations:
        frame = pd.read_csv(path, float_precision="round_trip")
        assert not frame._oci_row_id.duplicated().any()
        frame = frame.loc[frame.estimated_cate.notna()].copy()
        if cohort == "matched_180":
            assert set(frame._oci_row_id) == matched_ids
        frame["true_ite_prob"] = frame._oci_row_id.map(truth)
        assert frame.true_ite_prob.notna().all()
        frame["method"], frame["seed"] = method, seed
        estimate, actual = frame.estimated_cate.to_numpy(), frame.true_ite_prob.to_numpy()
        error = estimate - actual
        correlation = float(np.corrcoef(estimate, actual)[0, 1]) if np.ptp(estimate) > 1e-12 else None
        row = {"method": method, "seed": seed, "cohort": cohort, "n": len(frame),
               "correlation": correlation, "rmse": float(np.sqrt(np.mean(error ** 2))),
               "mae": float(np.abs(error).mean()), "bias": float(error.mean()),
               "estimated_mean": float(estimate.mean()), "true_mean": float(actual.mean()),
               "estimated_sd": float(estimate.std()), "true_sd": float(actual.std())}
        lower = "estimated_cate_lower_95" if method == "multi_model_production" else "lower_95"
        upper = "estimated_cate_upper_95" if method == "multi_model_production" else "upper_95"
        row["coverage_95"] = float(((frame[lower] <= actual) & (actual <= frame[upper])).mean())
        if method == "multi_model_production":
            diag = n.read(path.parent / "diagnostics.json")
            row.update(fit_n=diag["effect_fit_rows"], ate_aipw=diag["ate_aipw"],
                       ate_aipw_bias=diag["ate_aipw"] - float(actual.mean()))
        elif "outcome_prediction" in frame:
            residual = frame.outcome - frame.outcome_prediction - (frame.treatment - frame.propensity) * frame.estimated_cate
            row["r_loss"] = float(np.mean(residual ** 2))
        rows.append(row)
        prediction_frames.append(frame)
    table = pd.DataFrame(rows)
    table.to_csv(HERE / f"metrics_by_seed_{n.DATE}.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(HERE / f"predictions_with_oracle_{n.DATE}.csv", index=False)
    summaries = {}
    for method, frame in table.groupby("method", sort=False):
        summaries[method] = {column: {"mean": float(frame[column].mean()), "min": float(frame[column].min()),
                                      "max": float(frame[column].max())}
                             for column in ["n", "correlation", "rmse", "bias", "estimated_sd", "coverage_95"]
                             if frame[column].notna().any()}
    result = {
        "evaluated_at": n.now(), "outer_fold": 1, "summaries": summaries,
        "direct_recovery": {role: {"correct_role": sum(r["retained_in_correct_role"] for r in recovery if r["true_role"] == role),
                                  "any_role": sum(r["retained_in_any_role"] for r in recovery if r["true_role"] == role), "total": 5}
                            for role in ["confounder", "effect_modifier"]},
        "selected_count": selection_freeze["selected_count"], "selected_roles": selection_freeze["roles"],
        "oracle_source_sha256": n.sha(truth_path), "baseline_prediction_hashes_verified": baseline_files,
        "lineage_reference_path": str(lineage_path.resolve()), "lineage_reference_sha256": n.sha(lineage_path),
        "frozen_prediction_hashes_verified": True, "oracle_used_in_fitting": False,
        "evaluation_source_sha256": n.sha(__file__),
    }
    n.write(HERE / f"evaluation_{n.DATE}.json", result)
    n.write(HERE / "status.json", {"phase": "evaluation_complete", "updated_at": n.now()})
    print(json.dumps(result), flush=True)
    import subprocess
    import sys
    subprocess.run([sys.executable, str(HERE / f"build_report_{n.DATE}.py")], check=True)


if __name__ == "__main__":
    main()
