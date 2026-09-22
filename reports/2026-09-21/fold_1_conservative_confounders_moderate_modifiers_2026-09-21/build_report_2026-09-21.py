"""Create a dated outline report and scientific comparison figure."""
from common import HERE, PRIOR, DATE, n


def main():
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    evaluation = n.read(HERE / f"evaluation_{DATE}.json")
    audit = n.read(HERE / f"final_validation_{DATE}.json")
    assert audit["status"] == "passed"
    recovery = n.read(HERE / f"oracle_recovery_{DATE}.json")["direct"]
    metrics = pd.read_csv(HERE / f"metrics_by_seed_{DATE}.csv")
    predictions = pd.read_csv(HERE / f"predictions_with_oracle_{DATE}.csv")
    definitions = {d["feature_id"]: d for d in n.read(HERE / "selected_definitions.json")["features"]}
    s = evaluation["summaries"]
    old, new = s["native_eligible"]["production_8C_8M"], s["native_eligible"]["production_189C_16M"]
    old_m, new_m = s["fixed_180"]["matched_8"], s["fixed_180"]["matched_16"]

    def mean(summary, key, digits=3):
        return f"{summary[key]['mean']:.{digits}f}"

    def count_range(summary, key):
        lo, hi = int(summary[key]["min"]), int(summary[key]["max"])
        return str(lo) if lo == hi else f"{lo}–{hi}"

    labels = {"matched_224": "224 modifiers", "matched_8": "8 modifiers", "matched_16": "16 modifiers (new)",
              "production_8C_8M": "8 confounders / 8 modifiers", "production_189C_16M": "189 confounders / 16 modifiers (new)"}

    def table(cohort, methods, *, bias=True):
        lines = ["| Model | Held-out n | ITE correlation | ITE RMSE | CATE bias | Estimated effect SD | 95% interval coverage |",
                 "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for method in methods:
            item = s[cohort][method]
            lines.append(f"| {labels[method]} | {count_range(item, 'n')} | {mean(item, 'correlation')} | {mean(item, 'rmse')} | {mean(item, 'bias')} | {mean(item, 'estimated_sd')} | {100 * item['coverage_95']['mean']:.1f}% |")
        return "\n".join("      " + line for line in lines)

    recovery_lines = ["| Oracle variable | Role | Previous 8C/8M retained it? | New 189C/16M retained it? |",
                      "| --- | --- | --- | --- |"]
    for r in recovery:
        recovery_lines.append(f"| {r['oracle_variable']} | {'Confounder' if r['true_role']=='confounder' else 'Modifier'} | {'Yes' if r['previous_eight_correct_role'] else 'No'} | {'Yes' if r['retained_in_correct_role'] else 'No'} |")
    modifier_lines = ["| Existing global rank | Candidate | Also a confounder? |", "| ---: | --- | --- |"]
    for rank, key in enumerate(evaluation["selection"]["modifier_order"], 1):
        d = definitions[key]
        modifier_lines.append(f"| {rank} | {d['name']} ({key[-3:]}) | {'Yes' if 'confounder' in d['roles'] else 'No'} |")
    aipw_lines = ["| Model | Seed | Eligible n | AIPW ATE | Estimated 95% interval | Oracle eligible ATE |",
                  "| --- | ---: | ---: | ---: | --- | ---: |"]
    native = metrics.loc[metrics.cohort == "native_eligible"]
    for _, row in native.sort_values(["method", "seed"]).iterrows():
        aipw_lines.append(f"| {labels[row.method]} | {int(row.seed)} | {int(row.n)} | {row.ate_aipw:+.3f} | [{row.ate_aipw_lower_95:+.3f}, {row.ate_aipw_upper_95:+.3f}] | {row.true_mean:+.3f} |")
    contrasts = evaluation["contrasts"]
    fixed_delta = next(x["current_minus_earlier"] for x in contrasts if x["cohort"] == "fixed_180" and x["earlier"] == "matched_8")
    common_delta = next(x["current_minus_earlier"] for x in contrasts if x["cohort"] == "common_eligible")
    omitted_m = ", ".join(r["oracle_variable"] for r in recovery if r["true_role"] == "effect_modifier" and not r["retained_in_correct_role"])

    panels = [("fixed_180", "matched_8", 120042, "Fixed residuals: 8 modifiers"),
              ("fixed_180", "matched_16", 120042, "Fixed residuals: 16 modifiers"),
              ("common_eligible", "production_8C_8M", 100042, "Full estimator: 8C / 8M"),
              ("common_eligible", "production_189C_16M", 100042, "Full estimator: 189C / 16M")]
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 8.8), constrained_layout=True)
    low = float(min(predictions.true_ite_prob.min(), predictions.estimated_cate.min())) - .025
    high = float(max(predictions.true_ite_prob.max(), predictions.estimated_cate.max())) + .025
    for axis, (cohort, method, seed, title) in zip(axes.flat, panels):
        f = predictions.loc[(predictions.cohort == cohort) & (predictions.method == method) & (predictions.seed == seed)]
        m = metrics.loc[(metrics.cohort == cohort) & (metrics.method == method) & (metrics.seed == seed)].iloc[0]
        axis.scatter(f.true_ite_prob, f.estimated_cate, s=22, alpha=.65, color="#245a81", edgecolors="none")
        axis.plot([low, high], [low, high], color="#777777", linewidth=1, linestyle="--")
        axis.axhline(0, color="#cccccc", linewidth=.7)
        axis.set(xlim=(low, high), ylim=(low, high), xlabel="True probability-scale ITE", ylabel="Estimated CATE",
                 title=f"{title}\nn = {len(f)}, r = {m.correlation:.3f}, RMSE = {m.rmse:.3f}")
        axis.set_aspect("equal")
        axis.spines[["top", "right"]].set_visible(False)
    figure = HERE / f"ite_comparison_{DATE}.png"
    fig.savefig(figure, dpi=170)
    plt.close(fig)

    body = f"""# Fold 1: conservative confounders and moderate modifier pruning — September 21, 2026

1. **Result**
   1. Ran the requested middle ground with **189 confounders and 16 modifiers**, reusing the completed multi-model evidence, saved global ranking, existing extractions, and historical estimator settings.
   2. The full estimator's mean ITE correlation was **{mean(new, 'correlation')}**, compared with **{mean(old, 'correlation')}** for the previous 8-confounder/8-modifier version. RMSE was **{mean(new, 'rmse')} versus {mean(old, 'rmse')}**. These summaries use each estimator's eligible population; the common-patient comparison appears below.
   3. Direct correct-role recovery was **{evaluation['direct_recovery']['confounder']['correct_role']}/5 confounders and {evaluation['direct_recovery']['effect_modifier']['correct_role']}/5 modifiers**, versus 2/5 and 1/5 previously. Retention means availability to the fitted models; it does not guarantee nonzero elastic-net coefficients or accurate extracted measurements.
   4. With nuisance residuals and patients fixed, moving from eight to 16 modifiers changed correlation from **{mean(old_m, 'correlation')} to {mean(new_m, 'correlation')}**, and RMSE from **{mean(old_m, 'rmse', 4)} to {mean(new_m, 'rmse', 4)}**. The incremental gain from this modifier change was small.

2. **How the middle ground was constructed**
   1. Restored all 189 confounders in the original multi-model role adjudication. Used the top 16 modifiers from the already accepted global review. No new LLM review or extraction was needed.
   2. Sixteen had the lowest previous mean inner-fold R-loss: **0.1882194**, versus 0.1884813 for eight, 0.1889159 for 31, and 0.1902919 for the broad 224 reference. The earlier paired one-standard-error rule chose eight; this comparison uses the mean-loss winner.
   3. There are **197 distinct candidates**: 189 confounders, 16 modifiers, and eight with both roles. The forest receives the 16 modifiers in X and the 181 remaining confounders in W. Dual-role candidates appear once in X and also enter nuisance adjustment. External propensity models use the 189 confounders; outcome nuisance models use all 197 retained candidates.
   4. Forests retain 200 trees, minimum leaf size 10, square-root split search, 45% subsampling, honesty and inference enabled, and no forest tuning. Native estimation retains elastic-net logistic nuisance models, the original five inner splits, propensity eligibility 0.1–0.9, and clipping at 0.02.

3. **Modifier comparison with everything else fixed**
   1. Reused the original all-candidate elastic-net residuals, 720 training patients, 180 held-out patients, and three forest seeds. This evaluates the change in forest inputs without also changing the adjustment fit or eligibility.
   2. Mean results across the three seeds:

{table('fixed_180', ['matched_224', 'matched_8', 'matched_16'])}

   3. The 16-modifier design has **54 encoded columns**, versus 28 for eight and 705 for 224. Relative to eight, mean correlation changed by {fixed_delta['correlation']:+.3f} and RMSE by {fixed_delta['rmse']:+.4f}. The broad reference was reused from frozen predictions.
   4. The 16-modifier version improved correlation and RMSE over eight in only one of three matched seeds; that larger improvement outweighed two smaller degradations. Held-out R-loss was slightly worse for 16 in all three seeds (mean 0.1886691 versus 0.1882474). There is no decisive evidence here that 16 is intrinsically a better forest size than eight.

4. **Full estimator with conservative adjustment**
   1. Refitted nuisance models and the forest for three seeds. The new run retained {count_range(new, 'fit_n')} training and {count_range(new, 'n')} held-out patients after propensity screening; the previous run retained {count_range(old, 'fit_n')} and {count_range(old, 'n')} respectively.
   2. Results on each estimator's own eligible population:

{table('native_eligible', ['production_8C_8M', 'production_189C_16M'])}

   3. Results restricted to patients eligible under both estimators, without refitting either model:

{table('common_eligible', ['production_8C_8M', 'production_189C_16M'])}

   4. On the common patients, correlation changed by {common_delta['correlation']:+.3f} and RMSE by {common_delta['rmse']:+.3f}. This controls the evaluation population, but both the adjustment set and modifier set changed. The comparison does not isolate their separate contributions.
   5. AIPW estimates the average treatment effect; it is separate from the mean forest CATE. The following intervals are the native estimated intervals for each seed, not uncertainty across independent datasets:

{chr(10).join('      '+line for line in aipw_lines)}

   6. Across seeds, mean AIPW ATE was **{new['ate_aipw']['mean']:+.3f}**, versus mean oracle ATE **{new['true_mean']['mean']:+.3f}** in the corresponding eligible populations. Previous values were {old['ate_aipw']['mean']:+.3f} and {old['true_mean']['mean']:+.3f}. The new point estimates still have the wrong sign, although all three estimated 95% AIPW intervals include the oracle ATE. This single fold does not separate sampling error from estimation bias.
   7. New native-population ITE correlation ranged from {new['correlation']['min']:.3f} to {new['correlation']['max']:.3f} across seeds. Estimated effects remain compressed: mean estimated SD **{mean(new, 'estimated_sd')}**, versus oracle ITE SD **{mean(new, 'true_sd')}**. The reported interval coverage is coverage of the full oracle ITE, so omitted modifiers and approximation error also contribute; it is not a formal validation of intervals for CATE conditional only on the selected X.

5. **Which oracle variables survived**
   1. Correct-role recovery, counted using the existing direct lineage audit after prediction freezing:

{chr(10).join('      '+line for line in recovery_lines)}

   2. The missing modifier concepts are **{omitted_m}**. They were not manually restored from oracle knowledge. Related proxies are recorded separately in the [recovery audit](oracle_recovery_{DATE}.json).
   3. The 16 modifiers, in their previously frozen global order:

{chr(10).join('      '+line for line in modifier_lines)}

6. **Visual comparison**
   1. First seed only; tables above average all three seeds. Each row uses the same patients for its two panels. The bottom row uses the common production-eligible patients. Dashed lines indicate perfect agreement.

      ![ITE comparison](ite_comparison_{DATE}.png)

7. **What this experiment can establish**
   1. This is a post-hoc comparison on an already examined fold. Earlier oracle findings motivated the requested variant. Its role sets were assembled mechanically from prior frozen outputs, and all six new prediction sets were frozen before computing this run's oracle metrics. That sequence does not make the test set untouched again.
   2. The same inner folds informed the global ranking and the prefix scores. The tiny mean-loss advantage of 16 over eight is an adaptive tuning result, not evidence of a statistically established optimum.
   3. Three seeds measure algorithmic variation on the same patients. They do not provide three independent validation datasets. Recovery of an oracle candidate does not repair missing or inaccurate text extraction, guarantee its coefficient survives shrinkage, or ensure that a forest recovers its interaction.
   4. This experiment preserves broad confounder availability while limiting forest split candidates. Broader availability can still leave difficult high-dimensional nuisance fits. Assessment on other outer folds would be required before adopting this as a generally better policy.

8. **Verification and saved artifacts**
   1. Integrity checks **passed**: exact frozen inputs and prior predictions, all 189 inherited confounders, the unchanged top-16 ranking, original measurement definitions, treatment/outcome row alignment, nuisance role routing, and matched rows/residuals.
   2. Inspected all **{len(audit['production_nuisance_clones'])} actual internal fitted nuisance clones**. All use elastic-net logistic regression; **{audit['nuisance_clones_at_iteration_limit']}** reached the recorded iteration limit. The final audit records encoded X/W dimensions and effective forest parameters. This clone audit does not independently certify every external nuisance CV path.
   3. [Protocol](PROTOCOL_{DATE}.md), [input manifest](input_manifest_{DATE}.json), [selection freeze](selection_frozen_{DATE}.json), [prediction freeze](predictions_frozen_{DATE}.json), [per-seed metrics](metrics_by_seed_{DATE}.csv), [evaluation](evaluation_{DATE}.json), [all role decisions](all_role_decisions_{DATE}.csv), and [final validation](final_validation_{DATE}.json).
   4. Experiment-only scripts and outputs are in this dated folder. The core estimator and previous experiment files were preserved.
"""
    report = HERE / f"REPORT_{DATE}.md"
    report.write_text(body)
    n.write(HERE / f"report_complete_{DATE}.json", {"at": n.now(), "source_sha256": n.sha(__file__),
        "report": str(report), "report_sha256": n.sha(report), "figure_sha256": n.sha(figure),
        "evaluation_sha256": n.sha(HERE / f"evaluation_{DATE}.json"), "audit_sha256": n.sha(HERE / f"final_validation_{DATE}.json")})
    print(str(report), flush=True)


if __name__ == "__main__":
    main()
