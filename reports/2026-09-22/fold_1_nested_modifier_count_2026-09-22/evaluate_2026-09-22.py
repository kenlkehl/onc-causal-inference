"""Evaluate frozen predictions, then write a dated report and scientific figure."""

import json
import os
import traceback

os.environ.setdefault("MPLCONFIGDIR", "/tmp/stage2-mpl")

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
    assert freeze["production_fits"] == freeze["matched_residual_fits"] == 3
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
    matched_ids = set(labels.loc[labels.effect_eligible, "_oci_row_id"])
    assert len(test_ids) == 200 and len(matched_ids) == 180
    count = read(HERE / "count_result.json")
    decisions = read(HERE / "role_report.json")["decisions"]
    roles = {d["feature_id"]: d["roles"] for d in decisions}
    ranking = {r["feature_id"]: i for i, r in enumerate(count["full_training_ranking"]["ranking"], 1)}
    lineage_path = COMPACT / f"oracle_recovery_{OLD_DATE}.json"
    lineage = read(lineage_path)
    recovery = []
    chosen = count["choice"]["chosen_additional_count"]
    for item in lineage["direct"]:
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

    def record(frame, method, seed, cohort, production=False, diag=None):
        frame = frame.copy()
        assert not frame._oci_row_id.duplicated().any() and set(frame._oci_row_id) <= test_ids
        frame["true_ite_prob"] = frame._oci_row_id.map(truth.true_ite_prob)
        estimated, actual = frame.estimated_cate.to_numpy(), frame.true_ite_prob.to_numpy()
        assert np.isfinite(estimated).all() and np.isfinite(actual).all() and len(frame) > 1
        error = estimated - actual
        row = {"method": method, "seed": seed, "cohort": cohort, "n": len(frame),
               "correlation": float(np.corrcoef(estimated, actual)[0, 1]) if np.ptp(estimated) > 1e-12 else None,
               "rmse": float(np.sqrt(np.mean(error ** 2))), "mae": float(np.abs(error).mean()),
               "bias": float(error.mean()), "estimated_mean": float(estimated.mean()), "true_mean": float(actual.mean()),
               "estimated_sd": float(estimated.std()), "true_sd": float(actual.std())}
        lo, hi = ("estimated_cate_lower_95", "estimated_cate_upper_95") if production else ("lower_95", "upper_95")
        row["coverage_95"] = float(((frame[lo] <= actual) & (actual <= frame[hi])).mean())
        if production:
            values = frame.aipw_score.to_numpy()
            assert np.isfinite(values).all()
            ate, se = float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))
            row.update(ate_aipw=ate, ate_aipw_se=se, ate_aipw_lower_95=ate - 1.96 * se,
                       ate_aipw_upper_95=ate + 1.96 * se, ate_aipw_bias=ate - float(actual.mean()))
            if diag:
                assert np.isclose(ate, diag["ate_aipw"], rtol=0, atol=1e-12)
                row.update(fit_n=diag["effect_fit_rows"], confounders=diag["confounders"], modifiers=diag["effect_modifiers"])
        else:
            row["r_loss"] = float(np.mean((frame.outcome - frame.outcome_prediction - (frame.treatment - frame.propensity) * estimated) ** 2))
        rows.append(row)
        frame["method"], frame["seed"], frame["cohort"] = method, seed, cohort
        frames.append(frame)

    for seed in plan["evaluation_plan"]["matched_forest_seeds"]:
        for method, root in (("previous_16", PRIOR), ("nested_selected", HERE)):
            frame = pd.read_csv(root / f"matched_residuals/seed_{seed}/predictions.csv", float_precision="round_trip")
            assert set(frame._oci_row_id) == matched_ids
            record(frame, method, seed, "fixed_180")
    for seed in plan["evaluation_plan"]["estimation_seeds"]:
        pair = []
        for method, root in (("previous_16", PRIOR), ("nested_selected", HERE)):
            path = root / f"production/seed_{seed}/predictions.csv"
            frame = pd.read_csv(path, float_precision="round_trip")
            assert set(frame._oci_row_id) == test_ids
            assert np.array_equal(frame.effect_eligible, frame.estimated_cate.notna())
            frame = frame.loc[frame.effect_eligible].copy()
            diag = read(path.parent / "diagnostics.json")
            assert len(frame) == diag["effect_estimation_rows"]
            record(frame, method, seed, "native_eligible", production=True, diag=diag)
            pair.append((method, frame))
        common = set(pair[0][1]._oci_row_id) & set(pair[1][1]._oci_row_id)
        intersections.append({"seed": seed, "n": len(common), "row_ids": sorted(common)})
        for method, frame in pair:
            record(frame.loc[frame._oci_row_id.isin(common)], method, seed, "common_eligible", production=True)
    table = pd.DataFrame(rows)
    table.to_csv(HERE / f"metrics_by_seed_{DATE}.csv", index=False)
    pd.concat(frames, ignore_index=True).to_csv(HERE / f"predictions_with_oracle_{DATE}.csv", index=False)
    summaries = {}
    columns = ("n", "fit_n", "correlation", "rmse", "mae", "bias", "estimated_sd", "true_sd", "true_mean",
               "estimated_mean", "coverage_95", "ate_aipw", "ate_aipw_bias", "r_loss")
    for (cohort, method), group in table.groupby(["cohort", "method"], sort=False):
        summaries.setdefault(cohort, {})[method] = {
            k: {"mean": float(group[k].mean()), "min": float(group[k].min()), "max": float(group[k].max())}
            for k in columns if group[k].notna().any()}
    recovery_counts = {role: {"correct_role": sum(r["retained_in_correct_role"] for r in recovery if r["true_role"] == role),
                             "any_role": sum(r["retained_in_any_role"] for r in recovery if r["true_role"] == role), "total": 5}
                       for role in ("confounder", "effect_modifier")}
    result = {"at": now(), "summaries": summaries, "direct_recovery": recovery_counts, "roles": selection["roles"],
              "choice": count["choice"], "common_cohorts": intersections, "oracle_source_sha256": sha(truth_path),
              "prediction_freeze_sha256": sha(prediction_path), "previous_prediction_freeze_sha256": sha(prior_freeze_path),
              "source_sha256": sha(__file__), "exploratory_post_hoc_comparison": True,
              "oracle_values_used_in_fitting": False, "prior_oracle_results_informed_experiment_request": True}
    write(HERE / f"evaluation_{DATE}.json", result)
    build_report(result, recovery, table, frames, count)
    write(HERE / "experiment_complete.json", {"at": now(), "evaluation_sha256": sha(HERE / f"evaluation_{DATE}.json"),
          "report_sha256": sha(HERE / f"REPORT_{DATE}.md"), "prediction_hashes_verified": True,
          "native_count_selector_checkpoint_replay_verified": True, "all_189_confounders_preserved": True,
          "original_splits_and_observed_label_alignment_verified": True})
    write(HERE / "status.json", {"phase": "complete", "at": now(), "roles": selection["roles"],
          "report": str(HERE / f"REPORT_{DATE}.md"), "recovery": recovery_counts,
          "production": summaries["native_eligible"]["nested_selected"]})
    print(json.dumps({"phase": "complete", "recovery": recovery_counts, "summaries": summaries}), flush=True)


def build_report(result, recovery, table, frames, count):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    chosen = count["choice"]["chosen_additional_count"]
    one_se = read(HERE / "one_standard_error_diagnostic.json")["chosen_additional_count"]
    summaries = result["summaries"]
    curve = count["choice"]["options"]

    def fmt(value, places=3):
        return "undefined (constant predictions)" if value is None else f"{value:.{places}f}"

    def metric(cohort, method, key):
        return summaries[cohort][method].get(key, {}).get("mean")

    lines = ["# Fold 1: nested automatic modifier-count selection — September 22, 2026", "",
             "1. **Result**",
             f"   1. Selected **{chosen} modifiers** by minimum mean nested validation R-loss, retaining all **189 confounders**.",
             f"   2. Native production ITE correlation averaged **{fmt(metric('native_eligible', 'nested_selected', 'correlation'))}**, versus **{fmt(metric('native_eligible', 'previous_16', 'correlation'))}** for the prior 16-modifier model; RMSE was **{fmt(metric('native_eligible', 'nested_selected', 'rmse'))}** versus **{fmt(metric('native_eligible', 'previous_16', 'rmse'))}**.",
             f"   3. Correct-role oracle recovery: **{result['direct_recovery']['confounder']['correct_role']}/5 confounders** and **{result['direct_recovery']['effect_modifier']['correct_role']}/5 modifiers**.",
             "", "2. **Selection procedure and loss curve**",
             "   1. Reused the frozen 352 measurements, 800 training patients, and five original inner folds. Reproduced the full numerical evidence before reusing the broad confounder review.",
             "   2. Each count-validation fold used a new three-subfold, seven-family evidence run restricted to its training patients, followed by a fresh Gemma 4 31B ranking. Validation outcomes were masked from ranking evidence.",
             "   3. Compared the default modifier prefixes using fixed all-candidate elastic-net residuals, common propensity-eligible patients within each fold, and three 200-tree forest seeds. Seed losses were averaged within folds, then equally across folds.",
             "   4. Mean validation R-loss:", "",
             "      | Additional modifiers | Mean R-loss | Mean excess vs best | Paired SE vs best |", "      | ---: | ---: | ---: | ---: |"]
    for size in sorted(curve, key=int):
        cell = curve[size]
        lines.append(f"      | {size}{' (selected)' if int(size) == chosen else ''} | {cell['mean_r_loss']:.7f} | {cell['mean_excess_vs_best']:.7f} | {cell['paired_standard_error']:.7f} |")
    lines += ["", f"   5. The optional one-standard-error rule would choose **{one_se}**. This was recorded as a diagnostic; the fitted model follows the prespecified minimum-loss rule.",
              "   6. Rankings are built separately in each training split, so a budget evaluates the ranking-and-fitting procedure; candidate membership can vary by split. A full-training ranking supplies the final selected prefix.",
              "", "3. **Held-out effect estimation**",
              "   1. Three seeds per method; reported means describe algorithmic variation on the same patients.", "",
              "      | Population | Model | Held-out n | ITE correlation | RMSE | Bias | Effect SD | Oracle SD | 95% ITE coverage |",
              "      | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for cohort in ("native_eligible", "common_eligible", "fixed_180"):
        for method in ("previous_16", "nested_selected"):
            values = [metric(cohort, method, k) for k in ("n", "correlation", "rmse", "bias", "estimated_sd", "true_sd", "coverage_95")]
            lines.append(f"      | {cohort} | {'Prior 16 modifiers' if method == 'previous_16' else f'Nested {chosen} modifiers'} | {values[0]:.0f} | " + " | ".join(fmt(v) for v in values[1:]) + " |")
    lines += ["", "   2. Native estimates refit the production elastic-net nuisances and forest. Common-patient results restrict evaluation to patients eligible under both models without refitting. Fixed-180 results reuse the old all-candidate residuals and 720/180 fitting/evaluation population to isolate the modifier-input change.",
              "   3. Forest settings remain historical: 200 trees, leaf size 10, square-root feature sampling, 45% subsampling, honesty and inference enabled. Interval coverage is for full oracle ITE, not a formal validation of intervals for CATE conditional only on selected X.",
              "   4. Native AIPW ATE is distinct from mean predicted forest CATE:", "",
              "      | Model | Seed | Eligible n | AIPW ATE | Estimated 95% interval | Oracle mean ITE |",
              "      | --- | ---: | ---: | ---: | --- | ---: |"]
    for row in table.loc[table.cohort == "native_eligible"].itertuples():
        lines.append(f"      | {row.method} | {row.seed} | {row.n} | {row.ate_aipw:+.3f} | [{row.ate_aipw_lower_95:+.3f}, {row.ate_aipw_upper_95:+.3f}] | {row.true_mean:+.3f} |")
    lines += ["", "4. **Oracle feature recovery**", "",
              "   | Oracle variable | Role | Correct role retained? | Final modifier rank(s) | Nested folds within chosen prefix |",
              "   | --- | --- | --- | --- | ---: |"]
    for item in recovery:
        ranks = ", ".join(str(v) for v in item["global_modifier_ranks"].values() if v is not None) or "outside top 64"
        n = sum(r["within_chosen_prefix"] for r in item["nested_modifier_ranks"])
        lines.append(f"   | {item['oracle_variable']} | {item['true_role']} | {'Yes' if item['retained_in_correct_role'] else 'No'} | {ranks} | {n}/5 |")
    definitions = {f["feature_id"]: f for f in read(HERE / "selected_definitions.json")["features"]}
    lines += ["", "5. **Final modifier inputs**", "", "   | Rank | Candidate | Also retained as confounder? |", "   | ---: | --- | --- |"]
    for rank, row in enumerate(count["full_training_ranking"]["ranking"][:chosen], 1):
        item = definitions[row["feature_id"]]
        lines.append(f"   | {rank} | {item['name']} | {'Yes' if 'confounder' in item['roles'] else 'No'} |")
    if chosen == 0:
        lines.append("   | — | Constant-effect model | — |")
    lines += ["", "6. **Visual comparison**", "",
              f"   ![Nested R-loss curve and first-seed ITE comparison](comparison_{DATE}.png)",
              "", "7. **Interpretation and provenance**",
              "   1. This outer fold had already been examined. The experiment is exploratory, with settings fixed before these new predictions and metrics; it is not fresh confirmatory validation.",
              "   2. Nesting is conditional on the frozen upstream catalog, extraction, ontology, and consolidation. R-loss targets residual outcome prediction, not oracle-variable recall or Pearson ITE correlation.",
              "   3. The loss curve is noisy and the folds overlap in their training sets. The paired one-standard-error quantity is a simplification heuristic, not a significance test. Three forest seeds are not three independent patient samples.",
              "   4. All 189 confounders were preserved, and numerical/ranking checkpoints were replayed by the native count selector. Selected roles and all six prediction sets were hashed before oracle evaluation.",
              f"   5. [Protocol](PROTOCOL_{DATE}.md), [input manifest](input_manifest_{DATE}.json), [count result](count_result.json), [selection freeze](selection_frozen_{DATE}.json), [prediction freeze](predictions_frozen_{DATE}.json), [per-seed metrics](metrics_by_seed_{DATE}.csv), [oracle recovery](oracle_recovery_{DATE}.json), and [evaluation](evaluation_{DATE}.json).", ""]
    (HERE / f"REPORT_{DATE}.md").write_text("\n".join(lines))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    sizes = sorted(map(int, curve))
    axes[0].plot(sizes, [curve[str(k)]["mean_r_loss"] for k in sizes], "o-", color="#245c91")
    axes[0].axvline(chosen, color="#ac4436", ls="--", label=f"Selected: {chosen}")
    axes[0].set(xlabel="Modifier budget", ylabel="Mean validation R-loss", title="Nested count selection")
    axes[0].legend(frameon=False)
    axes[0].ticklabel_format(axis="y", style="plain", useOffset=False)
    first_seed = 100042
    for ax, method in zip(axes[1:], ("previous_16", "nested_selected")):
        frame = next(f for f in frames if f.method.iloc[0] == method and f.seed.iloc[0] == first_seed and f.cohort.iloc[0] == "common_eligible")
        ax.scatter(frame.true_ite_prob, frame.estimated_cate, s=18, alpha=.65, color="#245c91")
        limits = (min(-.5, float(frame.true_ite_prob.min()) - .03), max(.5, float(frame.true_ite_prob.max()) + .03))
        ax.plot(limits, limits, "--", color="gray", linewidth=1)
        ax.set(xlabel="Oracle probability ITE", ylabel="Estimated CATE", xlim=limits, ylim=limits,
               title=("Previous 16 modifiers" if method == "previous_16" else f"Nested {chosen} modifiers") + f"\nCommon patients, seed {first_seed}")
    fig.savefig(HERE / f"comparison_{DATE}.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        write(HERE / "status.json", {"phase": "evaluation_failed", "at": now(), "error": str(exc), "traceback": traceback.format_exc()})
        raise
