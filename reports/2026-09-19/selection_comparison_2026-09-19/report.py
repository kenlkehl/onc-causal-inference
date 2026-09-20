"""Render the completed comparison; refuse incomplete or changed predictions."""
from pathlib import Path
import json
import hashlib

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
LABELS = {"current_joint": "Current joint gate", "llm_adjudication": "LLM adjudication",
          "permissive_union": "Permissive combination", "all_candidates": "No hard effect screen"}


def load(path):
    return json.loads(Path(path).read_text())


def main():
    frozen = load(OUT / "predictions_frozen.json")
    if len(frozen["files"]) != 60:
        raise ValueError("All 60 fits must finish before reporting")
    for path, expected in frozen["files"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError("Frozen predictions changed")
    oracle = pd.DataFrame(load(OUT / "oracle_evaluation.json")["results"])
    loss = pd.DataFrame(load(OUT / "heldout_r_loss_results.json"))
    predictions = pd.read_csv(OUT / "predictions_with_oracle.csv")
    assert len(loss) == 60 and len(oracle) == 12
    for fold, frame in loss.groupby("outer_fold"):
        assert frame.nuisance_hash.nunique() == 1
        assert frame.rows.nunique() == 1 and frame.fit_rows.nunique() == 1
    coverage = predictions.groupby(["method", "replicate"])._oci_row_id.apply(lambda x: sorted(x)).tolist()
    assert all(ids == coverage[0] for ids in coverage)
    identity = load(OUT / "model_identity.json")
    trace = load(HERE.parent / "five_conf_five_mod_selection_trace_2026-09-19.json")
    retention = []
    for fold in trace["folds"]:
        directory = OUT / "refresh" / f"outer_{fold['outer_fold']:03d}"
        admission = load(directory / "comparison_admissions.json")
        for method, ids in admission["methods"].items():
            hits = []
            for name, concept in fold["concepts"].items():
                if "effect_modifier" in concept["oracle_roles"]:
                    candidates = {f["feature_id"] for f in concept["mapped_features"]}
                    if candidates.intersection(ids):
                        hits.append(name)
            retention.append({"outer_fold": fold["outer_fold"], "method": method,
                              "features": len(ids), "oracle_modifiers_admitted": len(hits), "concepts": hits})
    retention_frame = pd.DataFrame(retention)
    summaries = []
    for method in LABELS:
        metrics = oracle[oracle.method == method]
        rloss = loss[loss.method == method]
        pooled_losses = []
        pooled_base = []
        for _, seed in rloss.groupby("replicate"):
            pooled_losses.append(float(np.average(seed.r_loss, weights=seed.rows)))
            pooled_base.append(float(np.average(seed.constant_r_loss, weights=seed.rows)))
        retained = retention_frame[retention_frame.method == method]
        summaries.append({"method": method, "label": LABELS[method], "rows": int(metrics.rows.iloc[0]),
            "rmse_mean": float(metrics.rmse.mean()), "rmse_min": float(metrics.rmse.min()),
            "rmse_max": float(metrics.rmse.max()), "correlation_mean": float(metrics.correlation.mean()),
            "bias_mean": float(metrics.bias.mean()), "cate_sd_mean": float(metrics.predicted_sd.mean()),
            "oracle_sd": float(metrics.oracle_sd.iloc[0]), "r_loss_mean": float(np.mean(pooled_losses)),
            "constant_r_loss": float(np.mean(pooled_base)), "coverage_mean": float(metrics.coverage_95.mean()),
            "oracle_modifier_admissions": int(retained.oracle_modifiers_admitted.sum()),
            "feature_counts": retained.sort_values("outer_fold").features.tolist()})
    summary = {"methods": summaries, "retention": retention,
               "prediction_manifest": str(OUT / "predictions_frozen.json")}
    (OUT / "comparison_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    rows = []
    for s in summaries:
        rows.append(f"      | {s['label']} | {s['rmse_mean']:.4f} ({s['rmse_min']:.4f}–{s['rmse_max']:.4f}) | "
                    f"{s['correlation_mean']:.3f} | {s['r_loss_mean']:.5f} | {s['oracle_modifier_admissions']}/25 |")
    counts = []
    per_fold = []
    for fold in range(1, 6):
        data = loss[loss.outer_fold == fold]
        counts.append(f"      | {fold} | " + " | ".join(str(next(s for s in summaries if s['method'] == m)['feature_counts'][fold-1])
                                                       for m in LABELS) + " |")
        values = []
        for method in LABELS:
            estimates = []
            for seed in range(3):
                p = predictions[(predictions.outer_fold == fold) & (predictions.method == method) & (predictions.replicate == seed)]
                estimates.append(float(np.sqrt(np.mean((p.estimated_cate-p.oracle_effect)**2))))
            values.append(f"{np.mean(estimates):.4f}")
        per_fold.append(f"      | {fold} | {int(data.rows.iloc[0])} | " + " | ".join(values) + " |")
    best = min(summaries, key=lambda s: s["rmse_mean"])
    current = next(s for s in summaries if s["method"] == "current_joint")
    gain = 100 * (current["rmse_mean"] - best["rmse_mean"]) / current["rmse_mean"]
    if best["method"] == "current_joint":
        headline = (f"The current joint gate had the lowest mean held-out CATE RMSE "
                    f"({best['rmse_mean']:.4f}); none of the three alternatives improved this metric.")
    else:
        headline = (f"The lowest mean held-out CATE RMSE was achieved by **{best['label']}**: "
                    f"{best['rmse_mean']:.4f}, compared with {current['rmse_mean']:.4f} "
                    f"for the current joint gate ({gain:.1f}% lower).")
    calibration_rows = [
        f"      | {s['label']} | {s['bias_mean']:.4f} | {s['cate_sd_mean']:.4f} | "
        f"{s['oracle_sd']:.4f} | {s['coverage_mean']:.1%} |" for s in summaries
    ]
    extraction = identity["extraction"]["selected_model"]
    primary = identity["primary"]["selected_model"]
    text = f"""# Four-way Stage 2 selection comparison — September 19, 2026

1. **Result**
   1. {headline}
   2. All methods were evaluated on the same {best['rows']} eligible held-out patients across five outer folds, with three forest seeds per method per fold.
   3. This is an exploratory comparison on an already-inspected cohort. It identifies performance within this experiment, not general superiority across datasets.

2. **Pooled performance**
   1. Lower RMSE and R-loss are better; higher correlation is better. RMSE is on the outcome-probability difference scale. Parentheses show the range across forest seeds, not a confidence interval. Modifier admissions count five direct reference concepts across five folds.

      | Method | CATE RMSE: mean (seed range) | Correlation | R-loss | Direct modifier admissions |
      |---|---:|---:|---:|---:|
{chr(10).join(rows)}

   2. The shared training-fitted constant-effect comparator had pooled held-out R-loss {best['constant_r_loss']:.5f}.
   3. Direct modifier admission does not establish faithful extraction or complete recovery of the original category distinctions. Related proxies may carry useful information without counting as direct admissions.
   4. Bias, effect spread, and conditional interval coverage provide additional context. A small predicted spread can indicate flattened effect estimates; matching the oracle spread alone does not establish accurate individual estimates.

      | Method | Mean bias | Predicted effect SD | Oracle effect SD | Conditional 95% coverage |
      |---|---:|---:|---:|---:|
{chr(10).join(calibration_rows)}

3. **Fold-specific CATE RMSE, averaged across seeds**

      | Fold | Eligible test rows | Current joint gate | LLM adjudication | Permissive combination | No hard effect screen |
      |---|---:|---:|---:|---:|---:|
{chr(10).join(per_fold)}

4. **Number of measurements admitted to the effect inputs**

      | Fold | Current joint gate | LLM adjudication | Permissive combination | No hard effect screen |
      |---|---:|---:|---:|---:|
{chr(10).join(counts)}

5. **What was held fixed**
   1. All training and held-out measurements were refreshed using `{extraction}`; adjudication used `{primary}`. Discovery and initial candidates came from the frozen completed run. Training-only ontology supervision and harmonization used the existing workflow.
   2. Every method used identical cross-fitted grouped-elastic-net nuisance predictions, propensity bounds, eligible rows, and forest settings. The same three forest seeds were used for each policy.
   3. The LLM arm used the existing binding adjudication prompt with the shared aggregate statistical evidence. It changed effect-input admission only; it did not change nuisance fitting.
   4. The permissive method admitted the union of joint-model selections, LLM modifier assignments, and candidates with positive mean individual R-loss improvement and positive gains in at least three of five inner folds.
   5. The no-screen method admitted all candidate measurements under the existing upstream eligibility contract. It did not require a statistical vote or a modifier role annotation.
   6. The final forest was fitted directly on shared residuals. A numerical test verified equivalence with the final stage inside EconML's `CausalForestDML`.

6. **Limits**
   1. Changing the extractor and refreshing measurements means these results are not an isolated comparison of the new extractor against E4B. They isolate the selection policy within the refreshed measurements.
   2. Three forest seeds measure algorithmic variability, not sampling uncertainty. The five training partitions overlap, and this cohort's oracle results informed the preceding method discussion.
   3. Oracle effects were opened only after all 60 prediction files were frozen. They were not used to choose individual features or fit any models. Oracle modifier admissions are a post-hoc diagnostic.
   4. A new independent-cohort and null-effect validation is still needed before claiming general improvement or false-positive control.
   5. Reported interval coverage concerns the conditional forest intervals; it does not account for the full upstream discovery, extraction, and policy-development process.

7. **Artifacts**
   1. [Frozen protocol]({HERE / 'PROTOCOL_2026-09-19.md'}).
   2. [Summary and feature admissions]({OUT / 'comparison_summary.json'}).
   3. [All fold/seed R-loss results]({OUT / 'heldout_r_loss_results.json'}).
   4. [Oracle evaluation]({OUT / 'oracle_evaluation.json'}).
   5. [Prediction hashes, frozen before oracle evaluation]({OUT / 'predictions_frozen.json'}).
   6. [Source hashes]({OUT / 'inputs/source_manifest.json'}).
"""
    target = HERE / "selection_comparison_report_2026-09-19.md"
    target.write_text(text)
    print(target)


if __name__ == "__main__":
    main()
