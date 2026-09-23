"""Evaluate only frozen predictions; no oracle access before this stage."""

import json
import traceback

from common import HERE, ROOT, PRIOR, COMPACT, INPUTS, DATE, OLD_DATE, read, write, sha, now, verify, manifest


def main():
    import numpy as np
    import pandas as pd

    plan = manifest()
    selection_path = HERE / f"selection_frozen_{DATE}.json"
    prediction_path = HERE / f"predictions_frozen_{DATE}.json"
    selection, freeze = read(selection_path), read(prediction_path)
    verify(selection["files"])
    verify(freeze["files"])
    assert freeze["selection_freeze_sha256"] == sha(selection_path)
    assert freeze["production_fits"] == 3
    prior_freeze_path = PRIOR / f"predictions_frozen_{OLD_DATE}.json"
    verify(read(prior_freeze_path)["files"])
    marker = HERE / "oracle_access_started.json"
    if not marker.exists():
        write(marker, {"at": now(), "prediction_freeze_sha256": sha(prediction_path),
                       "selection_freeze_sha256": sha(selection_path), "evaluation_source_sha256": sha(__file__)})
    assert read(marker)["evaluation_source_sha256"] == sha(__file__)
    truth_path = ROOT / "synthetic_data/example_synthetic_datasets/five_confounders_five_effect_modifiers_nsclc_with_structured/dataset.parquet"
    assert sha(truth_path) == "abed84f263290f20f59a04e565833676fa4b656971117a4b2deaa7a666d4c396"
    truth = pd.read_parquet(truth_path, columns=["true_ite_prob", "treatment_indicator", "outcome_indicator"]).reset_index(drop=True)
    train_labels, labels = pd.read_parquet(INPUTS / "training_labels.parquet"), pd.read_parquet(INPUTS / "heldout_labels.parquet")
    for observed in (train_labels, labels):
        expected = truth.iloc[observed._oci_row_id.to_numpy(int)]
        assert np.array_equal(observed.treatment, expected.treatment_indicator)
        assert np.array_equal(observed.outcome, expected.outcome_indicator)
    test_ids = set(labels._oci_row_id)
    count = read(HERE / "count_result.json")
    roles = {d["feature_id"]: d["roles"] for d in read(HERE / "role_report.json")["decisions"]}
    ranking = {r["feature_id"]: i for i, r in enumerate(count["full_training_ranking"]["ranking"], 1)}
    lineage_path = COMPACT / f"oracle_recovery_{OLD_DATE}.json"
    recovery = []
    chosen = count["choice"]["chosen_additional_count"]
    for item in read(lineage_path)["direct"]:
        ids, role = item["candidate_ids"], item["true_role"]
        ranks = []
        for fold in count["folds"]:
            ordered = {r["feature_id"]: i for i, r in enumerate(fold["ranking"], 1)}
            positions = [ordered[key] for key in ids if key in ordered]
            ranks.append({"inner_fold": fold["inner_fold"], "best_rank": min(positions) if positions else None,
                          "within_chosen_prefix": any(p <= chosen for p in positions)})
        recovery.append({"oracle_variable": item["oracle_variable"], "true_role": role,
                         "candidate_ids": ids, "candidate_names": item["candidate_names"],
                         "assigned_roles": {key: roles[key] for key in ids},
                         "retained_in_correct_role": any(role in roles[key] for key in ids),
                         "retained_in_any_role": any(roles[key] for key in ids),
                         "global_modifier_ranks": {key: ranking.get(key) for key in ids},
                         "nested_modifier_ranks": ranks})
    write(HERE / f"oracle_recovery_{DATE}.json", {"direct": recovery, "lineage_source_sha256": sha(lineage_path)})
    rows, frames, intersections = [], [], []

    def record(frame, method, seed, cohort, diag=None):
        frame = frame.copy()
        assert not frame._oci_row_id.duplicated().any() and set(frame._oci_row_id) <= test_ids
        frame["true_ite_prob"] = frame._oci_row_id.map(truth.true_ite_prob)
        estimated, actual = frame.estimated_cate.to_numpy(), frame.true_ite_prob.to_numpy()
        assert np.isfinite(estimated).all() and np.isfinite(actual).all() and len(frame) > 1
        error = estimated - actual
        low = frame.estimated_cate_lower_95.to_numpy()
        high = frame.estimated_cate_upper_95.to_numpy()
        interval_rows = np.isfinite(low) & np.isfinite(high)
        values = frame.aipw_score.to_numpy()
        assert np.isfinite(values).all()
        ate, se = float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))
        row = {"method": method, "seed": seed, "cohort": cohort, "n": len(frame),
               "correlation": float(np.corrcoef(estimated, actual)[0, 1]) if np.ptp(estimated) > 1e-12 else None,
               "rmse": float(np.sqrt(np.mean(error ** 2))), "mae": float(np.abs(error).mean()),
               "bias": float(error.mean()), "estimated_mean": float(estimated.mean()), "true_mean": float(actual.mean()),
               "estimated_sd": float(estimated.std()), "true_sd": float(actual.std()),
               "coverage_95": float(((low[interval_rows] <= actual[interval_rows]) & (actual[interval_rows] <= high[interval_rows])).mean()) if interval_rows.any() else None,
               "interval_rows": int(interval_rows.sum()), "ate_aipw": ate, "ate_aipw_se": se,
               "ate_aipw_lower_95": ate - 1.96 * se, "ate_aipw_upper_95": ate + 1.96 * se,
               "ate_aipw_bias": ate - float(actual.mean())}
        if diag:
            assert np.isclose(ate, diag["ate_aipw"], rtol=0, atol=1e-12)
            row.update(fit_n=diag["effect_fit_rows"], confounders=diag["confounders"], modifiers=diag["effect_modifiers"],
                       model_family=diag["model_family"])
        rows.append(row)
        frame["method"], frame["seed"], frame["cohort"] = method, seed, cohort
        frames.append(frame)

    for seed in plan["evaluation_plan"]["estimation_seeds"]:
        pair = []
        for method, root in (("previous_16_forest", PRIOR), ("overlap_architecture_selected", HERE)):
            path = root / f"production/seed_{seed}/predictions.csv"
            frame = pd.read_csv(path, float_precision="round_trip")
            assert set(frame._oci_row_id) == test_ids
            assert np.array_equal(frame.effect_eligible, frame.estimated_cate.notna())
            frame = frame.loc[frame.effect_eligible].copy()
            diag = read(path.parent / "diagnostics.json")
            assert len(frame) == diag["effect_estimation_rows"]
            record(frame, method, seed, "native_eligible", diag)
            pair.append((method, frame))
        common = set(pair[0][1]._oci_row_id) & set(pair[1][1]._oci_row_id)
        intersections.append({"seed": seed, "n": len(common), "row_ids": sorted(common)})
        for method, frame in pair:
            record(frame.loc[frame._oci_row_id.isin(common)], method, seed, "common_eligible")
    table = pd.DataFrame(rows)
    table.to_csv(HERE / f"metrics_by_seed_{DATE}.csv", index=False)
    pd.concat(frames, ignore_index=True).to_csv(HERE / f"predictions_with_oracle_{DATE}.csv", index=False)
    summaries = {}
    columns = ("n", "fit_n", "correlation", "rmse", "mae", "bias", "estimated_sd", "true_sd", "true_mean",
               "estimated_mean", "coverage_95", "ate_aipw", "ate_aipw_bias")
    for (cohort, method), group in table.groupby(["cohort", "method"], sort=False):
        summaries.setdefault(cohort, {})[method] = {
            k: {"mean": float(group[k].mean()), "min": float(group[k].min()), "max": float(group[k].max())}
            for k in columns if k in group and group[k].notna().any()}
    recovery_counts = {role: {"correct_role": sum(r["retained_in_correct_role"] for r in recovery if r["true_role"] == role),
                             "any_role": sum(r["retained_in_any_role"] for r in recovery if r["true_role"] == role), "total": 5}
                       for role in ("confounder", "effect_modifier")}
    result = {"at": now(), "summaries": summaries, "direct_recovery": recovery_counts, "roles": selection["roles"],
              "choice": count["choice"], "chosen_estimator": count["chosen_estimator"], "common_cohorts": intersections,
              "oracle_source_sha256": sha(truth_path), "prediction_freeze_sha256": sha(prediction_path),
              "previous_prediction_freeze_sha256": sha(prior_freeze_path), "source_sha256": sha(__file__),
              "exploratory_post_hoc_comparison": True, "oracle_values_used_in_fitting": False}
    write(HERE / f"evaluation_{DATE}.json", result)
    build_report(result, recovery, table, count)
    write(HERE / "experiment_complete.json", {"at": now(), "evaluation_sha256": sha(HERE / f"evaluation_{DATE}.json"),
          "report_sha256": sha(HERE / f"REPORT_{DATE}.md"), "prediction_hashes_verified": True,
          "all_189_confounders_preserved": True, "original_splits_and_observed_label_alignment_verified": True})
    write(HERE / "status.json", {"phase": "complete", "at": now(), "roles": selection["roles"],
          "report": str(HERE / f"REPORT_{DATE}.md"), "recovery": recovery_counts,
          "chosen_estimator": count["chosen_estimator"], "production": summaries["native_eligible"]["overlap_architecture_selected"]})
    print(json.dumps({"phase": "complete", "recovery": recovery_counts, "summaries": summaries}), flush=True)


def build_report(result, recovery, table, count):
    def fmt(value):
        return "unavailable" if value is None else f"{value:.3f}"

    choice = count["choice"]
    chosen = choice["chosen_additional_count"]
    lines = [f"# Fold 1: overlap screening and architecture search — {DATE}", "",
             "1. **Selected model**",
             f"   1. **{count['chosen_estimator']}**, **{count['chosen_modifier_count']} modifiers**, preserving **189 confounders**.",
             "   2. Selected jointly by minimum mean nested validation R-loss on patients with estimated propensity 0.1–0.9.",
             f"   3. Direct correct-role oracle recovery: **{result['direct_recovery']['confounder']['correct_role']}/5 confounders**, **{result['direct_recovery']['effect_modifier']['correct_role']}/5 modifiers**.",
             "", "2. **Cross-validated model search**", "",
             "   | Architecture | Additional modifiers | Mean R-loss | Excess vs best | Paired SE |",
             "   | --- | ---: | ---: | ---: | ---: |"]
    for estimator, sizes in choice["options"].items():
        for size in sorted(sizes, key=int):
            cell = sizes[size]
            marker = " (selected)" if estimator == count["chosen_estimator"] and int(size) == chosen else ""
            lines.append(f"   | {estimator}{marker} | {size} | {cell['mean_r_loss']:.7f} | {cell['mean_excess_vs_best']:.7f} | {cell['paired_standard_error']:.7f} |")
    diagnostic = read(HERE / "one_standard_error_diagnostic.json")
    lines += ["", f"   1. The optional one-standard-error heuristic would choose {diagnostic['chosen_estimator']} with {diagnostic['chosen_additional_count']} additional modifiers; it did not determine fitting.",
              "   2. Five original validation folds; evidence and LLM rankings rebuilt within training. Both architectures share scoring residuals and eligible validation patients. Forest losses average three seeds; logistic penalties are tuned within training.",
              "", "3. **Held-out estimation, means across three seeds**", "",
              "   | Population | Model | N | ITE correlation | RMSE | Bias | Effect SD | Oracle SD | CATE interval coverage |",
              "   | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for cohort, methods in result["summaries"].items():
        for method, summary in methods.items():
            values = [summary.get(k, {}).get("mean") for k in ("n", "correlation", "rmse", "bias", "estimated_sd", "true_sd", "coverage_95")]
            lines.append(f"   | {cohort} | {method} | " + " | ".join(fmt(v) for v in values) + " |")
    lines += ["", "   1. Common eligibility means evaluation on patients eligible under both methods, without refitting. Three seeds describe algorithmic variation, not independent patient samples.",
              "   2. Penalized interaction CATE intervals are unavailable. Binary effects are predicted probability differences. AIPW ATE estimates use separate nuisance models.", "",
              "   | Model | Seed | Eligible N | AIPW ATE | 95% interval | Oracle mean ITE |",
              "   | --- | ---: | ---: | ---: | --- | ---: |"]
    for row in table.loc[table.cohort == "native_eligible"].itertuples():
        lines.append(f"   | {row.method} | {row.seed} | {row.n} | {row.ate_aipw:+.3f} | [{row.ate_aipw_lower_95:+.3f}, {row.ate_aipw_upper_95:+.3f}] | {row.true_mean:+.3f} |")
    lines += ["", "4. **Oracle recovery after prediction freeze**", "",
              "   | Variable | True role | Correct role retained | Final modifier ranks | Inner folds within chosen prefix |",
              "   | --- | --- | --- | --- | ---: |"]
    for item in recovery:
        ranks = ", ".join(str(v) for v in item["global_modifier_ranks"].values() if v is not None) or "outside ranked shortlist"
        n = sum(r["within_chosen_prefix"] for r in item["nested_modifier_ranks"])
        lines.append(f"   | {item['oracle_variable']} | {item['true_role']} | {'Yes' if item['retained_in_correct_role'] else 'No'} | {ranks} | {n}/5 |")
    definitions = {f["feature_id"]: f for f in read(HERE / "selected_definitions.json")["features"]}
    lines += ["", "5. **Final modifiers**", "", "   | Rank | Candidate | Also confounder |", "   | ---: | --- | --- |"]
    for rank, row in enumerate(count["full_training_ranking"]["ranking"][:chosen], 1):
        item = definitions[row["feature_id"]]
        lines.append(f"   | {rank} | {item['name']} | {'Yes' if 'confounder' in item['roles'] else 'No'} |")
    if chosen == 0:
        lines.append("   | — | No additional modifiers | — |")
    lines += ["", "6. **Limits and provenance**",
              "   1. Exploratory follow-up on a previously examined outer fold. No oracle values or outer-test outcomes entered selection or model fitting; observed outer outcomes enter AIPW evaluation only.",
              "   2. Conditional on the frozen catalog, extraction, consolidation, and 189-confounder policy. Validation uses all-candidate adjustment; the final refit uses retained roles. R-loss does not optimize oracle recall or ITE correlation directly.",
              f"   3. [Protocol](PROTOCOL_{DATE}.md), [manifest](input_manifest_{DATE}.json), [selection](count_result.json), [per-seed metrics](metrics_by_seed_{DATE}.csv), [recovery](oracle_recovery_{DATE}.json), [evaluation](evaluation_{DATE}.json).", ""]
    (HERE / f"REPORT_{DATE}.md").write_text("\n".join(lines))


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        write(HERE / "status.json", {"phase": "evaluation_failed", "at": now(), "error": str(exc), "traceback": traceback.format_exc()})
        raise
