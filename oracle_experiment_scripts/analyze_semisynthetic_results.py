#!/usr/bin/env python
"""Analyze semi-synthetic experiment results.

Loads results from run_semisynthetic_experiments.py and produces summary
statistics and plots.

Usage:
    python oracle_experiment_scripts/analyze_semisynthetic_results.py \
        --results-dir ../pcori_experiments/semisynthetic
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_results(results_dir: Path) -> pd.DataFrame:
    """Load all result JSON files into a DataFrame."""
    results = []
    for path in sorted((results_dir / "results").glob("*.json")):
        with open(path) as f:
            data = json.load(f)

        row = {
            'dgp_index': data['dgp_index'],
            'repeat_index': data['repeat_index'],
            'arm': data['arm'],
            'uses_text': data['uses_text'],
            'n_confounders_used': data['n_confounders_used'],
            'n_confounders_total': data['n_confounders_total'],
            'confounder_fraction': data['confounder_fraction'],
            'equation_mode': data['equation_mode'],
        }
        row.update(data.get('metrics', {}))
        row.update({f"dgp_{k}": v for k, v in data.get('dgp_stats', {}).items()})
        results.append(row)

    df = pd.DataFrame(results)
    logger.info(f"Loaded {len(df)} results from {results_dir}")
    return df


def print_summary(df: pd.DataFrame):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("SEMI-SYNTHETIC EXPERIMENT RESULTS")
    print("=" * 80)

    # Config info
    equation_mode = df['equation_mode'].iloc[0] if len(df) > 0 else "unknown"
    n_dgps = df['dgp_index'].nunique()
    n_repeats = df.groupby('dgp_index')['repeat_index'].nunique().max()
    n_confounders = df['n_confounders_total'].iloc[0] if len(df) > 0 else 0
    print(f"\nMode: {equation_mode} | DGPs: {n_dgps} | Repeats/DGP: {n_repeats} | "
          f"Confounders: {n_confounders}")

    # Overall summary by arm type
    print("\n--- ITE Correlation by Arm ---")
    print(f"{'Arm':<45} {'Mean':>8} {'Std':>8} {'N':>5}")
    print("-" * 70)

    # Sort: confounder_forest < text_forest < best_attainable, then by fraction
    arm_order = []
    for arm in sorted(df['arm'].unique()):
        if arm.startswith('confounder_forest'):
            arm_order.append((0, arm))
        elif arm.startswith('text_forest'):
            arm_order.append((1, arm))
        elif arm == 'best_attainable':
            arm_order.append((2, arm))
        else:
            arm_order.append((3, arm))
    arm_order.sort()

    for _, arm in arm_order:
        arm_data = df[df['arm'] == arm]
        mean_corr = arm_data['ite_corr'].mean()
        std_corr = arm_data['ite_corr'].std()
        n = len(arm_data)
        print(f"  {arm:<43} {mean_corr:>8.3f} {std_corr:>8.3f} {n:>5}")

    # Marginal value of text
    print("\n--- Marginal Value of Text (ITE corr improvement) ---")
    print(f"{'Fraction':<15} {'Conf Only':>12} {'Text+Conf':>12} {'Delta':>12}")
    print("-" * 55)

    fractions = sorted(df['confounder_fraction'].unique())
    for frac in fractions:
        conf_arm = f"confounder_forest_{frac:.2f}"
        text_arm = f"text_forest_{frac:.2f}"

        conf_data = df[df['arm'] == conf_arm]
        text_data = df[df['arm'] == text_arm]

        if len(conf_data) > 0 and len(text_data) > 0:
            conf_corr = conf_data['ite_corr'].mean()
            text_corr = text_data['ite_corr'].mean()
            delta = text_corr - conf_corr
            print(f"  {frac:<13.2f} {conf_corr:>12.3f} {text_corr:>12.3f} {delta:>+12.3f}")
        elif len(text_data) > 0:
            text_corr = text_data['ite_corr'].mean()
            print(f"  {frac:<13.2f} {'N/A':>12} {text_corr:>12.3f} {'N/A':>12}")

    # Cross-DGP variability
    if n_dgps > 1:
        print("\n--- Cross-DGP Variability (std across DGPs) ---")
        for _, arm in arm_order:
            arm_data = df[df['arm'] == arm]
            dgp_means = arm_data.groupby('dgp_index')['ite_corr'].mean()
            if len(dgp_means) > 1:
                cross_dgp_std = dgp_means.std()
                print(f"  {arm:<43} std={cross_dgp_std:.3f}")

    # Within-DGP variability (ML initialization)
    if n_repeats and n_repeats > 1:
        print("\n--- Within-DGP Variability (std across repeats) ---")
        for _, arm in arm_order:
            arm_data = df[df['arm'] == arm]
            within_stds = arm_data.groupby('dgp_index')['ite_corr'].std()
            if len(within_stds) > 0:
                mean_within_std = within_stds.mean()
                print(f"  {arm:<43} mean_std={mean_within_std:.3f}")

    # ATE bias
    print("\n--- ATE Bias ---")
    print(f"{'Arm':<45} {'Mean':>8} {'Std':>8}")
    print("-" * 65)
    for _, arm in arm_order:
        arm_data = df[df['arm'] == arm]
        if 'ate_bias' in arm_data.columns:
            mean_bias = arm_data['ate_bias'].mean()
            std_bias = arm_data['ate_bias'].std()
            print(f"  {arm:<43} {mean_bias:>8.4f} {std_bias:>8.4f}")

    # DGP statistics
    print("\n--- Simulated Data Statistics (across DGPs) ---")
    for stat_col in ['dgp_true_ate', 'dgp_true_ite_std', 'dgp_simulated_treatment_rate',
                     'dgp_simulated_outcome_rate', 'dgp_extraction_missingness_rate']:
        if stat_col in df.columns:
            mean_val = df.drop_duplicates(['dgp_index', 'repeat_index'])[stat_col].mean()
            std_val = df.drop_duplicates(['dgp_index', 'repeat_index'])[stat_col].std()
            label = stat_col.replace('dgp_', '')
            print(f"  {label:<40} {mean_val:>8.4f} +/- {std_val:.4f}")


def try_plot(df: pd.DataFrame, output_dir: Path):
    """Generate plots if matplotlib is available."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        logger.info("matplotlib not available, skipping plots")
        return

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    fractions = sorted(df['confounder_fraction'].unique())

    # ITE correlation vs confounder fraction
    fig, ax = plt.subplots(figsize=(10, 6))

    for arm_prefix, label, color, marker in [
        ('confounder_forest_', 'Confounders Only', '#1f77b4', 'o'),
        ('text_forest_', 'Text + Confounders', '#ff7f0e', 's'),
    ]:
        means = []
        stds = []
        valid_fracs = []
        for frac in fractions:
            arm_name = f"{arm_prefix}{frac:.2f}"
            arm_data = df[df['arm'] == arm_name]
            if len(arm_data) > 0:
                means.append(arm_data['ite_corr'].mean())
                stds.append(arm_data['ite_corr'].std())
                valid_fracs.append(frac)

        if means:
            ax.errorbar(valid_fracs, means, yerr=stds, label=label,
                        color=color, marker=marker, capsize=4, linewidth=2)

    # Best attainable line
    best = df[df['arm'] == 'best_attainable']
    if len(best) > 0:
        best_mean = best['ite_corr'].mean()
        ax.axhline(y=best_mean, color='green', linestyle='--', linewidth=1.5,
                   label=f'Best Attainable ({best_mean:.3f})')

    ax.set_xlabel('Fraction of Confounders Specified', fontsize=12)
    ax.set_ylabel('ITE Correlation (Pearson)', fontsize=12)
    ax.set_title('Semi-Synthetic Sensitivity Analysis', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(plot_dir / 'ite_correlation_vs_confounders.png', dpi=150)
    plt.close()
    logger.info(f"Saved plot: {plot_dir / 'ite_correlation_vs_confounders.png'}")

    # Marginal value of text
    fig, ax = plt.subplots(figsize=(10, 6))

    deltas = []
    delta_stds = []
    valid_fracs = []
    for frac in fractions:
        conf_data = df[df['arm'] == f'confounder_forest_{frac:.2f}']
        text_data = df[df['arm'] == f'text_forest_{frac:.2f}']
        if len(conf_data) > 0 and len(text_data) > 0:
            # Per-repeat delta
            merged = pd.merge(
                conf_data[['dgp_index', 'repeat_index', 'ite_corr']],
                text_data[['dgp_index', 'repeat_index', 'ite_corr']],
                on=['dgp_index', 'repeat_index'],
                suffixes=('_conf', '_text'),
            )
            if len(merged) > 0:
                delta = merged['ite_corr_text'] - merged['ite_corr_conf']
                deltas.append(delta.mean())
                delta_stds.append(delta.std())
                valid_fracs.append(frac)
        elif len(text_data) > 0 and frac == 0.0:
            deltas.append(text_data['ite_corr'].mean())
            delta_stds.append(text_data['ite_corr'].std())
            valid_fracs.append(frac)

    if deltas:
        ax.bar(valid_fracs, deltas, width=0.08, color='#2ca02c', alpha=0.7)
        ax.errorbar(valid_fracs, deltas, yerr=delta_stds, fmt='none',
                    color='black', capsize=5)

    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Fraction of Confounders Specified', fontsize=12)
    ax.set_ylabel('ITE Correlation Improvement (Text - Confounders)', fontsize=12)
    ax.set_title('Marginal Value of Text Features', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(plot_dir / 'marginal_value_of_text.png', dpi=150)
    plt.close()
    logger.info(f"Saved plot: {plot_dir / 'marginal_value_of_text.png'}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze semi-synthetic experiment results"
    )
    parser.add_argument("--results-dir", required=True,
                        help="Directory containing experiment results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not (results_dir / "results").exists():
        logger.error(f"No results directory found at {results_dir / 'results'}")
        return

    df = load_results(results_dir)
    if len(df) == 0:
        logger.error("No results found")
        return

    print_summary(df)
    try_plot(df, results_dir)


if __name__ == "__main__":
    main()
