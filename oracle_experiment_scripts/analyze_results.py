#!/data1/ken/pcori/bin/python
"""Analyze causal forest + explicit confounder experiment results.

This script reads all result JSONs from the experiment directory and produces
a comprehensive analysis. It is designed to be re-run as experiments progress.

Usage:
    python analyze_results.py [--results-dir DIR] [--output FILE]

    # Default: looks in ./results/, writes to ./results/results_analysis.txt
    ./analyze_results.py               # uses shebang
    python analyze_results.py           # or explicit python

    # Custom paths
    python analyze_results.py --results-dir /path/to/results --output report.txt

============================================================================
INTERPRETATION NOTES (for generating narrative analysis from output)
============================================================================

CONTINUOUS OUTCOMES:
- When outcome_type="continuous", the same column names are used (true_y0_prob,
  true_y1_prob, true_ite_prob, pred_y0_prob, pred_y1_prob, pred_ite_prob) but
  values represent raw outcome values instead of probabilities. All metrics
  (ite_corr, ate_bias, ite_mse, ci_coverage, etc.) are valid for both types.
  propensity_auroc is still meaningful since treatment is always binary.

KEY METRICS:
- ite_corr (Pearson correlation of predicted vs true ITE): PRIMARY metric.
  Higher is better. This measures how well the model ranks individuals by
  treatment effect. Values above 0.5 are good; above 0.6 is strong.
- ite_spearman_corr (Spearman rank correlation of predicted vs true ITE):
  Non-parametric rank correlation. More robust to outliers and non-linear
  monotonic relationships. Complements Pearson; large discrepancies between
  the two suggest non-linear effects or outlier influence.
- ate_bias (|mean(pred_ITE) - mean(true_ITE)|): Measures aggregate accuracy.
  Lower is better. Values < 0.05 are acceptable; > 0.1 is concerning.
- propensity_auroc: How well the model separates treated/untreated. Should be
  well above 0.5 (random). High values (>0.8) mean the model learns confounders.
  If this is low, the text extractor is failing to capture treatment assignment.
- ci_coverage: Fraction of true ITEs within predicted 95% CIs. Ideal = 0.95.
  Values < 0.7 mean CIs are too narrow (overconfident). Values > 0.95 mean
  CIs are too wide (uninformative).
- mean_ci_width: Width of CIs. Narrower is better IF coverage is adequate.
- y0_mse, y1_mse: Outcome prediction accuracy for control/treated. Lower = better.
  If y1_mse >> y0_mse, the model struggles more with treated outcomes.

EXPERIMENTAL FACTORS:
- rlearner_mode: "none" = basic propensity+outcome only (no tau head in Stage 1).
  "shared" = adds R-learner tau head with shared extractor.
  "dual" = separate extractors for nuisance (e,m) and effect (tau).
  INTERPRETATION: "shared" should outperform "none" if R-loss helps learn
  better representations. "dual" isolates effect learning from nuisance
  gradients, but risks insufficient training signal for the effect extractor.
  If dual is much worse than shared, it means the effect extractor can't learn
  useful features from R-loss alone — it needs the propensity/outcome gradients
  to bootstrap the representation.

- use_explicit_confounders: Whether LLM-extracted confounder features are
  concatenated to text embeddings. Should help because oracle confounders
  provide ground-truth signal. If it HURTS, possible reasons:
  (1) Random subset sampling picks irrelevant/noisy confounders
  (2) Featurizer MLP underfits at 30 epochs
  (3) Concatenation dilutes text representation signal
  (4) Causal forest sees more features but same n_estimators/min_samples_leaf
  IMPORTANT: The experiment randomly samples 1-to-N confounders per run, so
  compare effect of num_sampled_confounders, not just on/off.

- clam_enabled: CLAM instance-level loss supervises top-attended chunks with
  document labels. Should help hierarchical extractors focus attention.
  In causal forest pipeline, the benefit is indirect: better chunk attention
  -> better text representation -> better features for forest.

- dataset_name: "one_confounder" has 1 confounder (simpler, higher signal).
  "ten_confounders" has 10 confounders (more complex, harder to learn).
  "ten_confounders_50K" has 10 confounders with 50K rows (more data).
  INTERPRETATION: If ten_confounders is much harder, the GRU-pool extractor
  may lack capacity or need more epochs to capture 10 confounders from text.

- Hyperparameters (embedding_dim, gru_hidden_dim, gru_num_layers,
  transformer_layers, transformer_heads): These vary across experiments.
  When comparing conditions, hyperparameter variation adds noise. Look at
  mean AND std to assess consistency. If std is high, the condition's
  performance depends heavily on hyperparameters.

WHAT TO LOOK FOR IN THE OUTPUT:
1. "Overall" section: How many experiments done, overall performance range.
2. "By extractor type": Which feature extractor performs best? Are LLM-based
   extractors (frozen_llm_pooler, hierarchical_llm) better than learned-from-
   scratch extractors (hierarchical_cnn, hierarchical_gru, simple_cnn)?
3. "By dataset": Is ten_confounders much harder? If so, need more capacity.
4. "By explicit_confounders": Do oracle confounders help? If not, why?
5. "Confounders within best model type": Controls for model type to isolate
   the effect of explicit confounders.
6. "By CLAM": Does CLAM instance-level loss help?
7. "By num_sampled_confounders": Does adding more confounders help linearly,
   or is there a sweet spot?
8. "Specific confounders": Are certain confounders consistently helpful?
9. "Hyperparameter effects": Do larger models help? Is there a sweet spot?
10. "Cross-tabulated": Interaction effects between factors.
11. "Top/bottom experiments": What distinguishes the best from the worst?
============================================================================
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def load_results(results_dir: Path) -> pd.DataFrame:
    """Load all result JSONs into a DataFrame."""
    records = []
    for f in sorted(results_dir.glob("*.json")):
        with open(f) as fh:
            data = json.load(fh)

        if data.get("skipped", False):
            continue

        config = data.get("config", {})
        metrics = data.get("metrics", {})

        row = {
            "file": f.stem,
            "n_samples": data.get("n_samples"),
            **config,
            **metrics,
        }

        # Derived columns
        names = config.get("sampled_confounder_names", [])
        row["num_confounders"] = len(names) if isinstance(names, list) else 0
        row["confounder_names_str"] = (
            ",".join(sorted(names)) if isinstance(names, list) and names else ""
        )

        records.append(row)

    if not records:
        print("ERROR: No non-skipped results found.")
        sys.exit(1)

    df = pd.DataFrame(records)
    return df


def section(title: str, lines: list[str]) -> list[str]:
    """Format a section with a header."""
    sep = "=" * 80
    return ["", sep, title, sep, ""] + lines + [""]


def fmt(val, decimals=4):
    """Format a number, handling NaN."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:.{decimals}f}"


def has_present_values(df: pd.DataFrame, col: str) -> bool:
    """Return True if a column exists and has at least one non-missing value."""
    return col in df.columns and df[col].notna().any()


def sorted_unique_present(series: pd.Series) -> list:
    """Return sorted non-missing unique values, robust to mixed scalar types."""
    values = series.dropna().unique()
    return sorted(values, key=lambda v: (type(v).__name__, str(v)))


def choose_primary_group(df: pd.DataFrame) -> str | None:
    """Choose the best available model grouping column."""
    for col in ["rlearner_mode", "model_type"]:
        if has_present_values(df, col):
            return col
    return None


def group_summary(
    df: pd.DataFrame,
    group_cols: list[str],
    metrics: list[str] | None = None,
) -> str:
    """Produce a grouped summary table as a string."""
    if metrics is None:
        metrics = ["ite_corr", "ite_spearman_corr", "ate_bias", "propensity_auroc", "ci_coverage"]

    available = [m for m in metrics if m in df.columns]

    agg_funcs = {m: ["count", "mean", "std", "min", "max"] for m in available}
    # count only needed once
    for i, m in enumerate(available):
        if i > 0:
            agg_funcs[m] = ["mean", "std", "min", "max"]

    grouped = df.groupby(group_cols, dropna=False, sort=False).agg(agg_funcs).round(4)
    return grouped.to_string()


def pairwise_ttest(df: pd.DataFrame, group_col: str, metric: str) -> list[str]:
    """Run pairwise t-tests between groups for a metric."""
    lines = []
    groups = sorted_unique_present(df[group_col])
    if len(groups) < 2:
        return [f"  Only {len(groups)} group(s), skipping t-tests."]

    for i, g1 in enumerate(groups):
        for g2 in groups[i + 1 :]:
            v1 = df.loc[df[group_col] == g1, metric].dropna()
            v2 = df.loc[df[group_col] == g2, metric].dropna()
            if len(v1) < 2 or len(v2) < 2:
                lines.append(
                    f"  {g1} vs {g2}: insufficient data "
                    f"(n={len(v1)} vs n={len(v2)})"
                )
                continue
            t_stat, p_val = scipy_stats.ttest_ind(v1, v2, equal_var=False)
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            lines.append(
                f"  {g1} (n={len(v1)}, mean={v1.mean():.4f}) vs "
                f"{g2} (n={len(v2)}, mean={v2.mean():.4f}): "
                f"t={t_stat:.3f}, p={p_val:.4f} {sig}"
            )
    return lines


def analyze(df: pd.DataFrame) -> list[str]:
    """Run full analysis, returning lines of text."""
    output = []

    # ---------------------------------------------------------------
    # 0. Overall summary
    # ---------------------------------------------------------------
    lines = [
        f"Total experiments completed: {len(df)}",
        f"Datasets represented: {sorted_unique_present(df['dataset_name'])}",
    ]
    if has_present_values(df, 'model_type'):
        lines.append(f"Model types: {sorted_unique_present(df['model_type'])}")
    if has_present_values(df, 'feature_extractor_type'):
        lines.append(f"Extractor types: {sorted_unique_present(df['feature_extractor_type'])}")
    if has_present_values(df, 'rlearner_mode'):
        lines.append(f"R-learner modes: {sorted_unique_present(df['rlearner_mode'])}")
    if has_present_values(df, 'clam_enabled'):
        lines.append(f"CLAM enabled: {sorted_unique_present(df['clam_enabled'])}")
    lines.append(f"Explicit confounders: {sorted_unique_present(df['use_explicit_confounders'])}")
    if has_present_values(df, 'outcome_type'):
        lines.append(f"Outcome types: {sorted_unique_present(df['outcome_type'])}")
    lines += [
        "",
        "ITE correlation:    "
        f"mean={fmt(df['ite_corr'].mean())}  "
        f"std={fmt(df['ite_corr'].std())}  "
        f"min={fmt(df['ite_corr'].min())}  "
        f"max={fmt(df['ite_corr'].max())}",
        "ITE Spearman corr:  "
        f"mean={fmt(df['ite_spearman_corr'].mean())}  "
        f"std={fmt(df['ite_spearman_corr'].std())}  "
        f"min={fmt(df['ite_spearman_corr'].min())}  "
        f"max={fmt(df['ite_spearman_corr'].max())}"
        if 'ite_spearman_corr' in df.columns else "",
        "ATE bias:           "
        f"mean={fmt(df['ate_bias'].mean())}  "
        f"std={fmt(df['ate_bias'].std())}  "
        f"min={fmt(df['ate_bias'].min())}  "
        f"max={fmt(df['ate_bias'].max())}",
        "Propensity AUROC:   "
        f"mean={fmt(df['propensity_auroc'].mean())}  "
        f"std={fmt(df['propensity_auroc'].std())}  "
        f"min={fmt(df['propensity_auroc'].min())}  "
        f"max={fmt(df['propensity_auroc'].max())}",
        "CI coverage:        "
        f"mean={fmt(df['ci_coverage'].mean())}  "
        f"std={fmt(df['ci_coverage'].std())}  "
        f"min={fmt(df['ci_coverage'].min())}  "
        f"max={fmt(df['ci_coverage'].max())}",
    ]
    output.extend(section("0. OVERALL SUMMARY", lines))

    # ---------------------------------------------------------------
    # 1. By model type / rlearner_mode
    # ---------------------------------------------------------------
    primary_group = choose_primary_group(df)
    if primary_group and df[primary_group].nunique() > 1:
        lines = [
            group_summary(df, [primary_group]),
            "",
            "Pairwise t-tests on ite_corr:",
        ] + pairwise_ttest(df, primary_group, "ite_corr")
        if "ite_spearman_corr" in df.columns:
            lines += [
                "",
                "Pairwise t-tests on ite_spearman_corr:",
            ] + pairwise_ttest(df, primary_group, "ite_spearman_corr")
        lines += [
            "",
            "Pairwise t-tests on ate_bias:",
        ] + pairwise_ttest(df, primary_group, "ate_bias")
    else:
        lines = ["Only one model type — skipping comparison."]
    output.extend(section(f"1. BY {(primary_group or 'MODEL').upper()}", lines))

    # ---------------------------------------------------------------
    # 2. By feature extractor type
    # ---------------------------------------------------------------
    if "feature_extractor_type" in df.columns and df["feature_extractor_type"].nunique() > 1:
        lines = [
            group_summary(df, ["feature_extractor_type"]),
            "",
            "Pairwise t-tests on ite_corr:",
        ] + pairwise_ttest(df, "feature_extractor_type", "ite_corr")
        if "ite_spearman_corr" in df.columns:
            lines += [
                "",
                "Pairwise t-tests on ite_spearman_corr:",
            ] + pairwise_ttest(df, "feature_extractor_type", "ite_spearman_corr")
        lines += [
            "",
            "Pairwise t-tests on ate_bias:",
        ] + pairwise_ttest(df, "feature_extractor_type", "ate_bias")
    else:
        lines = ["Only one extractor type — skipping comparison."]
    output.extend(section("2. BY FEATURE EXTRACTOR TYPE", lines))

    # ---------------------------------------------------------------
    # 3. By dataset
    # NOTE: ten_confounders is much harder. Low propensity AUROC (~0.68)
    # means the extractor can't even separate T=0/T=1, let alone estimate
    # heterogeneous effects. Could need more capacity, more epochs, or
    # the 50K-row version to have enough signal.
    # ---------------------------------------------------------------
    lines = [group_summary(df, ["dataset_name"])]
    output.extend(section("3. BY DATASET", lines))

    # ---------------------------------------------------------------
    # 4. By explicit confounders (overall)
    # NOTE: This comparison is confounded by rlearner_mode distribution.
    # Look at section 5 (within shared mode) for a cleaner comparison.
    # If explicit confounders hurt even within shared mode, the issue is
    # not just mode selection but something about how the confounders are
    # integrated (featurizer capacity, concatenation approach, etc.).
    # ---------------------------------------------------------------
    lines = [
        group_summary(df, ["use_explicit_confounders"]),
        "",
        "Pairwise t-tests on ite_corr:",
    ] + pairwise_ttest(df, "use_explicit_confounders", "ite_corr")
    if "ite_spearman_corr" in df.columns:
        lines += [
            "",
            "Pairwise t-tests on ite_spearman_corr:",
        ] + pairwise_ttest(df, "use_explicit_confounders", "ite_spearman_corr")
    output.extend(section("4. BY EXPLICIT CONFOUNDERS (overall)", lines))

    # ---------------------------------------------------------------
    # 5. Explicit confounders within best-performing model type
    # ---------------------------------------------------------------
    if primary_group and primary_group in df.columns:
        # Pick the model type with highest mean ite_corr
        best_type = df.groupby(primary_group)["ite_corr"].mean().idxmax()
        sub_df = df[df[primary_group] == best_type]
        if len(sub_df) > 0 and sub_df["use_explicit_confounders"].nunique() > 1:
            lines = [
                f"(Restricted to {primary_group}='{best_type}', n={len(sub_df)})",
                "",
                group_summary(sub_df, ["use_explicit_confounders"]),
                "",
                "Pairwise t-tests on ite_corr:",
            ] + pairwise_ttest(sub_df, "use_explicit_confounders", "ite_corr")
            if "ite_spearman_corr" in sub_df.columns:
                lines += [
                    "",
                    "Pairwise t-tests on ite_spearman_corr:",
                ] + pairwise_ttest(sub_df, "use_explicit_confounders", "ite_spearman_corr")
        else:
            lines = [f"No variation in use_explicit_confounders within {best_type}."]
    else:
        lines = ["No primary grouping column available."]
    output.extend(
        section("5. EXPLICIT CONFOUNDERS WITHIN BEST MODEL TYPE", lines)
    )

    # ---------------------------------------------------------------
    # 6. By CLAM (if present)
    # ---------------------------------------------------------------
    if "clam_enabled" in df.columns and df["clam_enabled"].nunique() > 1:
        lines = [
            group_summary(df, ["clam_enabled"]),
            "",
            "Pairwise t-tests on ite_corr:",
        ] + pairwise_ttest(df, "clam_enabled", "ite_corr")
        if "ite_spearman_corr" in df.columns:
            lines += [
                "",
                "Pairwise t-tests on ite_spearman_corr:",
            ] + pairwise_ttest(df, "clam_enabled", "ite_spearman_corr")
    else:
        lines = ["CLAM not varied in these experiments — skipping."]
    output.extend(section("6. BY CLAM", lines))

    # ---------------------------------------------------------------
    # 7. By number of sampled confounders
    # NOTE: Only relevant for use_explicit_confounders=True experiments.
    # Look for a dose-response: does adding more confounders improve ITE
    # correlation? Or is there a sweet spot (e.g., 1-3 is good, >5 hurts)?
    # If more confounders = worse, the model may be overfitting to noisy
    # features or the forest can't handle the extra dimensionality.
    # ---------------------------------------------------------------
    conf_df = df[df["use_explicit_confounders"] == True].copy()
    if len(conf_df) > 0:
        lines = [
            f"(Restricted to use_explicit_confounders=True, n={len(conf_df)})",
            "",
            group_summary(conf_df, ["num_confounders"]),
        ]

        # Correlation between num_confounders and ite_corr
        if len(conf_df) >= 3 and conf_df["num_confounders"].nunique() > 1:
            r, p = scipy_stats.pearsonr(
                conf_df["num_confounders"], conf_df["ite_corr"]
            )
            lines.extend([
                "",
                f"Pearson corr(num_confounders, ite_corr): r={r:.4f}, p={p:.4f}",
                "  (Negative r = more confounders -> worse performance)",
            ])
            if "ite_spearman_corr" in conf_df.columns:
                r_sp, p_sp = scipy_stats.pearsonr(
                    conf_df["num_confounders"], conf_df["ite_spearman_corr"]
                )
                lines.extend([
                    f"Pearson corr(num_confounders, ite_spearman_corr): r={r_sp:.4f}, p={p_sp:.4f}",
                ])
    else:
        lines = ["No explicit-confounder experiments yet."]
    output.extend(
        section("7. BY NUMBER OF SAMPLED CONFOUNDERS", lines)
    )

    # ---------------------------------------------------------------
    # 8. Which specific confounders appear in best/worst experiments?
    # NOTE: Since confounders are randomly sampled, we can check whether
    # certain confounders consistently appear in high-performing runs.
    # This is exploratory — small sample sizes mean low power.
    # ---------------------------------------------------------------
    if len(conf_df) > 0:
        all_names = set()
        for names_str in conf_df["confounder_names_str"]:
            if names_str:
                all_names.update(names_str.split(","))

        if all_names:
            confounder_stats = []
            for name in sorted(all_names):
                mask = conf_df["confounder_names_str"].str.contains(
                    name, na=False
                )
                present = conf_df.loc[mask, "ite_corr"]
                absent = conf_df.loc[~mask, "ite_corr"]
                confounder_stats.append({
                    "confounder": name,
                    "n_present": len(present),
                    "mean_ite_corr_present": present.mean() if len(present) > 0 else np.nan,
                    "n_absent": len(absent),
                    "mean_ite_corr_absent": absent.mean() if len(absent) > 0 else np.nan,
                    "diff": (
                        (present.mean() - absent.mean())
                        if len(present) > 0 and len(absent) > 0
                        else np.nan
                    ),
                })
            cs_df = pd.DataFrame(confounder_stats).sort_values(
                "diff", ascending=False
            )
            lines = [
                "Mean ITE correlation when each confounder is present vs absent:",
                "(Positive diff = confounder helps; negative = hurts)",
                "(WARNING: confounded by other factors — these are correlational only)",
                "",
                cs_df.to_string(index=False),
            ]
        else:
            lines = ["No confounder names found."]
    else:
        lines = ["No explicit-confounder experiments yet."]
    output.extend(
        section("8. INDIVIDUAL CONFOUNDER EFFECTS (exploratory)", lines)
    )

    # ---------------------------------------------------------------
    # 9. Hyperparameter effects
    # NOTE: The grid varies embedding_dim, gru_hidden_dim, gru_num_layers,
    # transformer_layers, transformer_heads. Since the grid is shuffled and
    # experiments are still running, these correlations are noisy. Look for
    # strong monotonic effects (e.g., bigger always better or always worse).
    # If no clear pattern, hyperparameters matter less than the structural
    # choices (rlearner_mode, confounders, etc.).
    # ---------------------------------------------------------------
    hp_cols = [
        # GRU-Pool hyperparameters
        "embedding_dim", "gru_hidden_dim", "gru_num_layers",
        "transformer_layers", "transformer_heads",
        # Transformer Pool hyperparameters
        "token_transformer_layers", "token_transformer_heads",
        "token_transformer_dim", "chunk_transformer_layers",
        # Frozen LLM Pooler hyperparameters
        "flp_max_length", "flp_downprojection_dim",
        "flp_projection_dim", "flp_gated_attention_dim",
    ]
    available_hp = [c for c in hp_cols if c in df.columns and df[c].nunique() > 1]

    lines = []
    for hp in available_hp:
        unique_vals = sorted(df[hp].dropna().unique())
        if len(unique_vals) < 2:
            continue

        lines.append(f"\n--- {hp} ---")
        lines.append(group_summary(df, [hp], ["ite_corr", "ite_spearman_corr", "ate_bias"]))

        # Correlation with ite_corr
        valid = df[[hp, "ite_corr"]].dropna()
        if len(valid) >= 3:
            r, p = scipy_stats.pearsonr(valid[hp], valid["ite_corr"])
            lines.append(
                f"  Pearson corr({hp}, ite_corr): r={r:.4f}, p={p:.4f}"
            )
        if "ite_spearman_corr" in df.columns:
            valid_sp = df[[hp, "ite_spearman_corr"]].dropna()
            if len(valid_sp) >= 3:
                r_sp, p_sp = scipy_stats.pearsonr(valid_sp[hp], valid_sp["ite_spearman_corr"])
                lines.append(
                    f"  Pearson corr({hp}, ite_spearman_corr): r={r_sp:.4f}, p={p_sp:.4f}"
                )

    if not lines:
        lines = ["Insufficient hyperparameter variation in results so far."]
    output.extend(section("9. HYPERPARAMETER EFFECTS", lines))

    # ---------------------------------------------------------------
    # 10. Cross-tabulated: model_type x use_explicit_confounders
    # ---------------------------------------------------------------
    if primary_group and primary_group in df.columns:
        lines = [
            "Mean ITE Pearson correlation:",
            df.pivot_table(
                values="ite_corr",
                index=primary_group,
                columns="use_explicit_confounders",
                aggfunc=["mean", "count"],
            )
            .round(4)
            .to_string(),
        ]
        if "ite_spearman_corr" in df.columns:
            lines += [
                "",
                "Mean ITE Spearman correlation:",
                df.pivot_table(
                    values="ite_spearman_corr",
                    index=primary_group,
                    columns="use_explicit_confounders",
                    aggfunc=["mean", "count"],
                )
                .round(4)
                .to_string(),
            ]
        lines += [
            "",
            "Mean ATE bias:",
            df.pivot_table(
                values="ate_bias",
                index=primary_group,
                columns="use_explicit_confounders",
                aggfunc=["mean", "count"],
            )
            .round(4)
            .to_string(),
        ]
    else:
        lines = ["No primary grouping column — skipping cross-tab."]
    output.extend(
        section(f"10. CROSS-TAB: {(primary_group or 'model').upper()} x EXPLICIT_CONFOUNDERS", lines)
    )

    # ---------------------------------------------------------------
    # 10b. Cross-tabulated: model_type x clam_enabled (if present)
    # ---------------------------------------------------------------
    if primary_group and "clam_enabled" in df.columns and df["clam_enabled"].nunique() > 1:
        lines = [
            "Mean ITE Pearson correlation:",
            df.pivot_table(
                values="ite_corr",
                index=primary_group,
                columns="clam_enabled",
                aggfunc=["mean", "count"],
            )
            .round(4)
            .to_string(),
        ]
        if "ite_spearman_corr" in df.columns:
            lines += [
                "",
                "Mean ITE Spearman correlation:",
                df.pivot_table(
                    values="ite_spearman_corr",
                    index=primary_group,
                    columns="clam_enabled",
                    aggfunc=["mean", "count"],
                )
                .round(4)
                .to_string(),
            ]
    else:
        lines = ["CLAM not varied — skipping cross-tab."]
    output.extend(
        section("10b. CROSS-TAB: MODEL_TYPE x CLAM_ENABLED", lines)
    )

    # ---------------------------------------------------------------
    # 10c. Cross-tabulated: dataset_name x extractor_type
    # ---------------------------------------------------------------
    if "feature_extractor_type" in df.columns and df["feature_extractor_type"].nunique() > 1:
        lines = [
            "Mean ITE Pearson correlation:",
            df.pivot_table(
                values="ite_corr",
                index="dataset_name",
                columns="feature_extractor_type",
                aggfunc=["mean", "count"],
            )
            .round(4)
            .to_string(),
        ]
        if "ite_spearman_corr" in df.columns:
            lines += [
                "",
                "Mean ITE Spearman correlation:",
                df.pivot_table(
                    values="ite_spearman_corr",
                    index="dataset_name",
                    columns="feature_extractor_type",
                    aggfunc=["mean", "count"],
                )
                .round(4)
                .to_string(),
            ]
    else:
        lines = ["Only one extractor type — skipping cross-tab."]
    output.extend(
        section("10c. CROSS-TAB: DATASET_NAME x FEATURE_EXTRACTOR_TYPE", lines)
    )

    # ---------------------------------------------------------------
    # 11. Top and bottom experiments
    # NOTE: Look at what distinguishes the best from worst runs.
    # Consistent patterns across top-5 (e.g., all shared mode, all
    # one_confounder) confirm the factor-level findings. If top-5 has
    # a mix, hyperparameters or specific confounders matter more.
    # ---------------------------------------------------------------
    display_cols = [
        "file", "dataset_name", "feature_extractor_type", "outcome_type",
        "rlearner_mode", "clam_enabled",
        "use_explicit_confounders", "num_confounders",
        "confounder_names_str",
        # GRU-Pool hyperparameters
        "embedding_dim", "gru_hidden_dim", "gru_num_layers",
        "transformer_layers", "transformer_heads",
        # Transformer Pool hyperparameters
        "token_transformer_layers", "token_transformer_heads",
        "token_transformer_dim", "chunk_transformer_layers",
        # Metrics
        "ite_corr", "ite_spearman_corr", "ate_bias", "propensity_auroc", "ci_coverage",
    ]
    display_cols = [c for c in display_cols if c in df.columns]

    n_show = min(10, len(df))

    top = df.nlargest(n_show, "ite_corr")[display_cols]
    bottom = df.nsmallest(n_show, "ite_corr")[display_cols]

    lines = [
        f"Top {n_show} by ITE correlation:",
        top.to_string(index=False),
        "",
        f"Bottom {n_show} by ITE correlation:",
        bottom.to_string(index=False),
    ]
    output.extend(section("11. TOP AND BOTTOM EXPERIMENTS", lines))

    # ---------------------------------------------------------------
    # 12. Outcome model quality
    # NOTE: y0_mse and y1_mse measure how well the model predicts
    # potential outcomes. If the model can't predict outcomes well, it
    # can't estimate treatment effects well either. Compare across modes:
    # dual mode might have better nuisance models (dedicated extractor)
    # but worse tau, or vice versa.
    # ---------------------------------------------------------------
    if "y0_mse" in df.columns and "y1_mse" in df.columns:
        lines = []
        if primary_group and primary_group in df.columns:
            lines += [
                f"By {primary_group}:",
                group_summary(
                    df, [primary_group],
                    ["y0_mse", "y1_mse", "propensity_auroc"],
                ),
                "",
            ]
        if "feature_extractor_type" in df.columns and df["feature_extractor_type"].nunique() > 1:
            lines += [
                "By feature_extractor_type:",
                group_summary(
                    df, ["feature_extractor_type"],
                    ["y0_mse", "y1_mse", "propensity_auroc"],
                ),
                "",
            ]
        lines += [
            "By dataset:",
            group_summary(
                df, ["dataset_name"],
                ["y0_mse", "y1_mse", "propensity_auroc"],
            ),
        ]
    else:
        lines = ["y0_mse/y1_mse not available."]
    output.extend(section("12. OUTCOME MODEL QUALITY", lines))

    # ---------------------------------------------------------------
    # 13. CI calibration
    # NOTE: Ideal ci_coverage is 0.95 for 95% CIs. Systematically low
    # coverage means the causal forest is overconfident. Check if
    # coverage varies by condition — e.g., dual mode may have tighter
    # but poorly calibrated CIs.
    # ---------------------------------------------------------------
    if "ci_coverage" in df.columns:
        lines = [
            f"Overall CI coverage: {fmt(df['ci_coverage'].mean())} "
            f"(target: 0.95)",
            f"Overall CI width:    {fmt(df['mean_ci_width'].mean())}",
            "",
        ]
        if primary_group and primary_group in df.columns:
            lines += [
                f"By {primary_group}:",
                group_summary(
                    df, [primary_group], ["ci_coverage", "mean_ci_width"]
                ),
                "",
            ]
        if "feature_extractor_type" in df.columns and df["feature_extractor_type"].nunique() > 1:
            lines += [
                "By feature_extractor_type:",
                group_summary(
                    df, ["feature_extractor_type"], ["ci_coverage", "mean_ci_width"]
                ),
                "",
            ]
        lines += [
            "By dataset:",
            group_summary(
                df, ["dataset_name"], ["ci_coverage", "mean_ci_width"]
            ),
        ]
    else:
        lines = ["CI metrics not available."]
    output.extend(section("13. CONFIDENCE INTERVAL CALIBRATION", lines))

    # ---------------------------------------------------------------
    # 14. Progress tracking
    # NOTE: The full grid is very large (3 datasets x 3 modes x 2 CLAM
    # x 2 explicit x 3 emb x 3 gru x 2 layers x 3 trans_layers x 3 heads
    # = ~1944+ configs, minus some filtered). Track coverage to know
    # how representative the current results are.
    # ---------------------------------------------------------------
    lines = [
        "Experiments per condition:",
        "",
    ]
    if primary_group and primary_group in df.columns:
        lines += [
            f"By {primary_group}:",
            df[primary_group].value_counts().sort_index().to_string(),
            "",
        ]
    lines += [
        "By dataset_name:",
        df["dataset_name"].value_counts().sort_index().to_string(),
        "",
    ]
    if "clam_enabled" in df.columns:
        lines += [
            "By clam_enabled:",
            df["clam_enabled"].value_counts().sort_index().to_string(),
            "",
        ]
    if "feature_extractor_type" in df.columns:
        lines += [
            "By feature_extractor_type:",
            df["feature_extractor_type"].value_counts().sort_index().to_string(),
            "",
        ]
    lines += [
        "By use_explicit_confounders:",
        df["use_explicit_confounders"].value_counts().sort_index().to_string(),
        "",
    ]
    if primary_group and primary_group in df.columns:
        lines += [
            f"Full cross-tab ({primary_group} x dataset x explicit_confounders):",
            pd.crosstab(
                [df[primary_group], df["dataset_name"]],
                df["use_explicit_confounders"],
                margins=True,
            ).to_string(),
        ]
    if "feature_extractor_type" in df.columns and df["feature_extractor_type"].nunique() > 1:
        lines += [
            "",
            "Full cross-tab (feature_extractor_type x dataset):",
            pd.crosstab(
                df["feature_extractor_type"],
                df["dataset_name"],
                margins=True,
            ).to_string(),
        ]
    if 'outcome_type' in df.columns:
        lines += [
            "",
            "By outcome_type:",
            df["outcome_type"].value_counts().sort_index().to_string(),
        ]
    output.extend(section("14. EXPERIMENT COVERAGE / PROGRESS", lines))

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Analyze causal forest + explicit confounder experiments"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory containing result JSONs (e.g., /path/to/experiment/results/)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file (default: results_analysis.txt inside results-dir)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    output_file = (
        Path(args.output).resolve()
        if args.output
        else results_dir / "results_analysis.txt"
    )

    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}")
        sys.exit(1)

    n_files = len(list(results_dir.glob("*.json")))
    print(f"Loading {n_files} result files from {results_dir} ...")

    df = load_results(results_dir)
    print(f"Loaded {len(df)} successful experiments.")

    output_lines = analyze(df)
    report = "\n".join(output_lines)

    # Write to file
    with open(output_file, "w") as f:
        f.write(report)
    print(f"Analysis written to: {output_file}")

    # Also print to stdout
    print(report)


if __name__ == "__main__":
    main()
