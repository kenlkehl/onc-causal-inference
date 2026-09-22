"""Build a dated hierarchical report from the completed, frozen experiment."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import json

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("numerical_run", HERE / "numerical_2026-09-21.py")
n = importlib.util.module_from_spec(spec)
spec.loader.exec_module(n)


def main():
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    evaluation = n.read(HERE / f"evaluation_{n.DATE}.json")
    recovery = n.read(HERE / f"oracle_recovery_{n.DATE}.json")
    roles = n.read(HERE / "role_report.json")
    validation = n.read(HERE / f"numerical_validation_{n.DATE}.json")
    selected = n.read(HERE / "selected_definitions.json")["features"]
    metrics = pd.read_csv(HERE / f"metrics_by_seed_{n.DATE}.csv")
    pred = pd.read_csv(HERE / f"predictions_with_oracle_{n.DATE}.csv")
    summary = evaluation["summaries"]
    labels = {
        "current_joint": "Previous joint elastic-net selection",
        "llm_adjudication": "Previous LLM selection",
        "permissive_union": "Previous permissive union",
        "all_candidates": "All 352 candidates; sqrt split search",
        "all_candidates_all_split_features": "All 352 candidates; all-column split search",
        "multi_model_matched": "New multi-model selection",
    }
    rows = []
    for method, label in labels.items():
        s = summary[method]
        rows.append(f"| {label} | {s['correlation']['mean']:.3f} | {s['rmse']['mean']:.3f} | {s['bias']['mean']:+.3f} | {s['estimated_sd']['mean']:.3f} |")
    oracle_rows = []
    for r in recovery["direct"]:
        parts = []
        for key, assigned in r["assigned_roles"].items():
            abbrev = "/".join("C" if x == "confounder" else "M" for x in assigned) or "none"
            parts.append(f"{key.rsplit('_', 1)[-1]}: {abbrev}")
        oracle_rows.append(f"| {r['oracle_variable']} | {'C' if r['true_role']=='confounder' else 'M'} | {'; '.join(parts)} | {'Yes' if r['retained_in_correct_role'] else 'No'} |")
    stability = pd.Series([r["stability"] for r in roles["decisions"] if r["roles"]]).value_counts().to_dict()
    c = evaluation["selected_roles"]["confounder"]
    m = evaluation["selected_roles"]["effect_modifier"]
    both = sum(len(d["roles"]) == 2 for d in selected)
    direct_c = evaluation["direct_recovery"]["confounder"]["correct_role"]
    direct_m = evaluation["direct_recovery"]["effect_modifier"]["correct_role"]
    production = summary["multi_model_production"]
    pm = metrics.loc[metrics.method == "multi_model_production"]
    current = summary["current_joint"]
    new = summary["multi_model_matched"]
    proxies = "; ".join(f"{r['proxy']}: {', '.join(r['assigned_roles']) or 'not selected'}" for r in recovery["proxies"])

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), constrained_layout=True)
    plotted = [
        ("current_joint", 120042, "Previous joint selection"),
        ("multi_model_matched", 120042, "New selection; fixed residuals"),
        ("multi_model_production", 100042, "New selection; production estimator"),
    ]
    limits = []
    for method, seed, _ in plotted:
        f = pred.loc[(pred.method == method) & (pred.seed == seed)]
        limits.extend(f.true_ite_prob.tolist() + f.estimated_cate.tolist())
    lo, hi = float(min(limits)) - .025, float(max(limits)) + .025
    for axis, (method, seed, title) in zip(axes, plotted):
        f = pred.loc[(pred.method == method) & (pred.seed == seed)]
        row = metrics.loc[(metrics.method == method) & (metrics.seed == seed)].iloc[0]
        axis.scatter(f.true_ite_prob, f.estimated_cate, s=19, alpha=.65, color="#245a81", edgecolors="none")
        axis.plot([lo, hi], [lo, hi], color="#777777", linewidth=1, linestyle="--")
        axis.axhline(0, color="#cccccc", linewidth=.7)
        axis.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="True probability-scale ITE", ylabel="Estimated CATE",
                 title=f"{title}\nn = {len(f)}, r = {row.correlation:.3f}, RMSE = {row.rmse:.3f}")
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_aspect("equal")
    figure = HERE / f"ite_scatter_{n.DATE}.png"
    fig.savefig(figure, dpi=180)
    plt.close(fig)

    text = f"""# Outer fold 1: multi-model Stage 2 selection — September 21, 2026

1. **Run and validation**
   1. Reused all 352 candidate definitions and the frozen Gemma 4 26B A4B extractions: 800 training patients, 200 outer-held-out patients, five original inner folds. Stage 1 and extraction were not rerun.
   2. Completed all 305 planned family/subset cells: univariable models, penalized main effects, penalized outcome interactions, orthogonal linear effects, candidate R-learners, predictive forests, and causal forests. All 40 nuisance fits converged; no candidate was omitted from any planned family/repetition.
   3. Verified 310 checkpoint hashes, input/source hashes, fold boundaries, nested nuisance cross-fitting, and full candidate coverage. An exact replay reproduced the frozen numerical report with all numerical fitters disabled. Four nuisance fits chose a penalty at a grid boundary; none of their CV grid fits failed to converge.
   4. Gemma 4 31B reviewed aggregate evidence, reconciled cross-candidate themes, and assigned final roles. The native adjudicator replayed and validated every cached response. Selection and all six new forest predictions were frozen before oracle evaluation.

2. **Selected variables**
   1. Selected **{len(selected)} distinct candidates**: **{c} confounders**, **{m} modifiers**, including **{both} with both roles**. The remaining {352-len(selected)} candidates received neither role. Thus {m/352:.1%} of all candidates still enter the forest as modifiers; this broad selection leaves a large modifier search space.
   2. Selected-candidate stability labels: {json.dumps(stability, sort_keys=True)}. These are LLM interpretations of overlapping samples, not independent replication probabilities.
   3. Direct oracle recovery in the correct role: **{direct_c}/5 confounders and {direct_m}/5 modifiers**. C = confounder, M = modifier; numeric IDs are the suffix of `outer_001_feature_`.

      | Oracle variable | True role | Assigned candidate roles | Correct role recovered? |
      | --- | --- | --- | --- |
{chr(10).join('      ' + row for row in oracle_rows)}

   4. Proxy audit, excluded from direct recovery counts: {proxies}.
   5. Full decisions, citations, and rationales: [all candidate decisions](all_role_decisions_{n.DATE}.csv), [modeling evidence summary](modeling_evidence_summary_{n.DATE}.csv), and [oracle lineage audit](oracle_recovery_{n.DATE}.json).

3. **Effect estimation: matched comparison**
   1. This comparison preserves the earlier 720 training / 180 held-out overlap population, all-candidate elastic-net residuals, forest seeds, and historical forest settings. Only the new method's selected modifier inputs change. Values below are means over three seeds; each seed uses the same patients.

      | Selection / forest inputs | ITE correlation | ITE RMSE | Mean bias | Estimated effect SD |
      | --- | ---: | ---: | ---: | ---: |
{chr(10).join('      ' + row for row in rows)}

   2. Relative to joint elastic-net selection, the new method changes mean correlation by **{new['correlation']['mean']-current['correlation']['mean']:+.3f}** and RMSE by **{new['rmse']['mean']-current['rmse']['mean']:+.3f}**. Lower RMSE is better.
   3. Probability differences are the effect scale. Correlation concerns ranking/shape; RMSE and bias also penalize effect magnitude and calibration errors.

4. **Effect estimation: production routing**
   1. Used the existing Stage 2 estimator with selected confounders in nuisance adjustment, modifiers in forest X, and pure confounders in W. Nuisances remained elastic nets. Forest settings: 200 trees, honesty and inference enabled, minimum leaf size 10, 45% subsampling, sqrt split search.
   2. Across three seeds: **correlation {production['correlation']['mean']:.3f}** (range {production['correlation']['min']:.3f}–{production['correlation']['max']:.3f}); **RMSE {production['rmse']['mean']:.3f}**; **bias {production['bias']['mean']:+.3f}**; effect SD {production['estimated_sd']['mean']:.3f}.
   3. Propensity eligibility produced {int(pm.fit_n.min())}–{int(pm.fit_n.max())} training and {int(pm.n.min())}–{int(pm.n.max())} held-out patients. Because nuisance fitting and eligibility also change, this production result is distinct from the matched comparison above.
   4. Mean AIPW ATE across seeds: {pm.ate_aipw.mean():+.3f}; mean oracle ATE in each seed's eligible population: {pm.true_mean.mean():+.3f}. These are separate from the forest's mean CATE.

      ![First-seed effect estimates](ite_scatter_{n.DATE}.png)

   5. The plot shows the first seed of each method; the tables summarize all three. The dashed line is perfect agreement. Production eligibility may differ from the two matched panels.

5. **Interpretation and limits**
   1. Broader selection evidence does not remove the small-sample difficulty of heterogeneous-effect estimation. During selection, 58/115 causal-forest subset fits improved validation R-loss over a constant effect; the mean gain was −0.000098. Penalized outcome interactions improved effect R-loss in 12/15 fits, with mean gain +0.000756. Outcome/treatment prediction was more consistently useful than heterogeneity prediction.
   2. Sparse/constant measurements and unestimable individual tests remain explicitly marked; no complete model cell failed. More evidence families do not repair measurement error or missing extracted values.
   3. This is one exploratory outer fold, after prior fold-1 oracle diagnostics were seen. The new algorithm/settings were frozen before new evaluation; no oracle values or roles entered its numerical models or LLM prompts. Three forest seeds are not three independent clinical datasets.
   4. The permissive LLM retention policy is a concrete target for improvement. Forest and candidate R-learner support can mean only a positive validation-loss difference; it is not a test against a calibrated null. The LLM was warned about this distinction, but still retained {m} modifiers. A separate experiment could add independent noise features and compare their selection rates and effect scores with real candidates. That would diagnose permissiveness; it would not itself establish formal error control. This run did not apply such controls or change selection after observing truth.
   5. Assess recovery, effect ranking, RMSE, calibration, and population eligibility together. This run alone does not establish that the method generalizes across outer folds.

6. **Reproducibility files**
   1. [Protocol](PROTOCOL_{n.DATE}.md), [input manifest](input_manifest_{n.DATE}.json), [numerical validation](numerical_validation_{n.DATE}.json).
   2. [Selection freeze](selection_frozen_{n.DATE}.json), [prediction freeze](predictions_frozen_{n.DATE}.json), [per-seed metrics](metrics_by_seed_{n.DATE}.csv), [evaluation](evaluation_{n.DATE}.json).
   3. Existing upstream experiments were read-only inputs. Request concurrency increased from 4 to 16; its runtime revision preserves completed reviews and leaves prompts, model identity, and scientific settings unchanged.
"""
    path = HERE / f"REPORT_{n.DATE}.md"
    path.write_text(text)
    n.write(HERE / f"report_complete_{n.DATE}.json", {
        "completed_at": n.now(), "report": str(path.resolve()), "report_sha256": n.sha(path),
        "figure_sha256": n.sha(figure), "source_sha256": n.sha(__file__),
        "evaluation_sha256": n.sha(HERE / f"evaluation_{n.DATE}.json"),
    })
    print(str(path), flush=True)


if __name__ == "__main__":
    main()
