# oci/analysis/psm_analysis.py
"""
Propensity Score Matching analysis using DragonNet's propensity scores.

This module provides traditional PSM analysis as a complement to DragonNet's
ITE estimation, allowing for:
- Validation of DragonNet's average effect estimates
- Traditional statistical inference (ATT, ATE)
- Balance diagnostics
- Sensitivity analysis for unmeasured confounding
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd

from ..config import MatchingAnalysisConfig
from ..matching import (
    PropensityMatcher,
    MatchResult,
    compute_balance_statistics,
    assess_overlap
)
from .statistical_analysis import (
    estimate_att_matched,
    estimate_ate_ipw,
    estimate_ate_stratified,
    mcnemars_test,
    paired_t_test,
    sensitivity_analysis_rosenbaum,
    TreatmentEffectEstimate
)


logger = logging.getLogger(__name__)


def run_psm_analysis(
    predictions_df: pd.DataFrame,
    config: MatchingAnalysisConfig,
    output_dir: Optional[Path] = None,
    propensity_column: str = 'propensity_pred',
    treatment_column: str = 'treatment',
    outcome_column: str = 'outcome'
) -> Dict[str, Any]:
    """
    Run propensity score matching analysis on DragonNet predictions.

    This function takes the predictions from DragonNet (which include propensity
    scores) and performs traditional PSM analysis for comparison and validation.

    Args:
        predictions_df: DataFrame with DragonNet predictions, must contain:
            - propensity_pred: Predicted propensity scores
            - treatment: Actual treatment assignment
            - outcome: Actual outcome
            - y0_pred, y1_pred, ite_pred: DragonNet's outcome predictions (optional)
        config: Matching analysis configuration
        output_dir: Directory to save results (optional)
        propensity_column: Name of propensity score column
        treatment_column: Name of treatment column
        outcome_column: Name of outcome column

    Returns:
        Dictionary containing:
            - match_result: MatchResult object
            - overlap: Overlap assessment metrics
            - balance_stats: Balance statistics DataFrame
            - att_matched: ATT estimate from matched pairs
            - ate_ipw: ATE estimate from IPW
            - ate_stratified: ATE estimate from stratification
            - dragonnet_ate: Mean ITE from DragonNet (for comparison)
            - sensitivity: Rosenbaum sensitivity analysis
            - comparison: Comparison of PSM vs DragonNet estimates
    """
    logger.info("=" * 60)
    logger.info("PROPENSITY SCORE MATCHING ANALYSIS")
    logger.info("=" * 60)

    # Extract arrays
    propensity_scores = predictions_df[propensity_column].values
    treatment = predictions_df[treatment_column].values
    outcomes = predictions_df[outcome_column].values

    # Get DragonNet's ITE predictions if available
    has_ite = 'ite_pred' in predictions_df.columns
    if has_ite:
        dragonnet_ite = predictions_df['ite_pred'].values
        dragonnet_ate = np.mean(dragonnet_ite)
        dragonnet_att = np.mean(dragonnet_ite[treatment == 1])
        logger.info(f"DragonNet ATE: {dragonnet_ate:.4f}")
        logger.info(f"DragonNet ATT: {dragonnet_att:.4f}")
    else:
        dragonnet_ate = None
        dragonnet_att = None

    results = {}

    # =========================================================================
    # 1. Assess Overlap
    # =========================================================================
    logger.info("\n--- Overlap Assessment ---")
    overlap = assess_overlap(propensity_scores, treatment)
    results['overlap'] = overlap

    logger.info(f"Overlap coefficient: {overlap['overlap_coefficient']:.3f}")
    logger.info(f"PS treated:  mean={overlap['ps_treated_mean']:.3f}, std={overlap['ps_treated_std']:.3f}")
    logger.info(f"PS control:  mean={overlap['ps_control_mean']:.3f}, std={overlap['ps_control_std']:.3f}")
    logger.info(f"Common support: [{overlap['common_support_min']:.3f}, {overlap['common_support_max']:.3f}]")

    # =========================================================================
    # 2. Perform Matching
    # =========================================================================
    logger.info("\n--- Propensity Score Matching ---")
    matcher = PropensityMatcher(
        method=config.method,
        caliper=config.caliper,
        caliper_scale=config.caliper_scale,
        ratio=config.ratio,
        replacement=config.replacement,
        random_state=42
    )

    match_result = matcher.match(propensity_scores, treatment)
    results['match_result'] = match_result

    logger.info(f"Matching: {match_result}")
    logger.info(f"  Matched pairs: {len(match_result.matched_pairs)}")
    logger.info(f"  Unmatched treated: {match_result.n_unmatched_treated}")
    logger.info(f"  Mean distance: {match_result.distances.mean():.4f}" if len(match_result.distances) > 0 else "  No matches")

    # =========================================================================
    # 3. Balance Statistics
    # =========================================================================
    logger.info("\n--- Balance Diagnostics ---")

    # Use propensity score as the covariate for balance check
    # In practice, you'd include other covariates too
    covariates = pd.DataFrame({'propensity_score': propensity_scores})

    balance_stats = compute_balance_statistics(covariates, treatment, match_result)
    results['balance_stats'] = balance_stats

    for _, row in balance_stats.iterrows():
        logger.info(f"  {row['covariate']}: SMD before={row['smd_before']:.3f}", end="")
        if 'smd_after' in row:
            logger.info(f", after={row['smd_after']:.3f} (reduction: {row['smd_reduction']*100:.1f}%)")

    # =========================================================================
    # 4. Treatment Effect Estimation
    # =========================================================================
    logger.info("\n--- Treatment Effect Estimates ---")

    # Crude difference
    crude_diff = np.mean(outcomes[treatment == 1]) - np.mean(outcomes[treatment == 0])
    results['crude_difference'] = crude_diff
    logger.info(f"Crude difference: {crude_diff:.4f}")

    # ATT from matched pairs
    att_matched = estimate_att_matched(
        outcomes, treatment, match_result,
        ci_level=config.ci_level,
        n_bootstrap=config.n_bootstrap
    )
    results['att_matched'] = att_matched
    logger.info(f"Matched ATT: {att_matched}")

    # ATE via IPW
    ate_ipw = estimate_ate_ipw(
        outcomes, treatment, propensity_scores,
        ci_level=config.ci_level,
        n_bootstrap=config.n_bootstrap
    )
    results['ate_ipw'] = ate_ipw
    logger.info(f"IPW ATE: {ate_ipw}")

    # ATE via stratification
    ate_strat = estimate_ate_stratified(
        outcomes, treatment, propensity_scores,
        ci_level=config.ci_level,
        n_bootstrap=config.n_bootstrap
    )
    results['ate_stratified'] = ate_strat
    logger.info(f"Stratified ATE: {ate_strat}")

    # =========================================================================
    # 5. Statistical Tests
    # =========================================================================
    logger.info("\n--- Statistical Tests ---")

    if len(np.unique(outcomes)) == 2:  # Binary outcome
        mcnemar = mcnemars_test(outcomes, match_result)
        results['mcnemar_test'] = mcnemar
        logger.info(f"McNemar's test: statistic={mcnemar['statistic']:.3f}, p={mcnemar['p_value']:.4f}")
    else:
        paired_t = paired_t_test(outcomes, match_result)
        results['paired_t_test'] = paired_t
        logger.info(f"Paired t-test: t={paired_t['t_statistic']:.3f}, p={paired_t['p_value']:.4f}")

    # =========================================================================
    # 6. Sensitivity Analysis
    # =========================================================================
    logger.info("\n--- Sensitivity Analysis (Rosenbaum Bounds) ---")

    sensitivity = sensitivity_analysis_rosenbaum(outcomes, match_result)
    results['sensitivity_analysis'] = sensitivity

    for _, row in sensitivity.iterrows():
        logger.info(f"  Gamma={row['gamma']:.1f}: p_upper={row['p_upper']:.4f}")

    # =========================================================================
    # 7. Comparison with DragonNet
    # =========================================================================
    if has_ite:
        logger.info("\n--- Comparison: PSM vs DragonNet ---")

        comparison = {
            'dragonnet_ate': dragonnet_ate,
            'dragonnet_att': dragonnet_att,
            'psm_att': att_matched.estimate,
            'psm_ate_ipw': ate_ipw.estimate,
            'psm_ate_stratified': ate_strat.estimate,
            'att_difference': abs(dragonnet_att - att_matched.estimate) if dragonnet_att else None,
            'ate_difference': abs(dragonnet_ate - ate_ipw.estimate) if dragonnet_ate else None,
        }
        results['comparison'] = comparison

        logger.info(f"DragonNet ATE: {dragonnet_ate:.4f}")
        logger.info(f"PSM IPW ATE:   {ate_ipw.estimate:.4f}")
        logger.info(f"Difference:    {comparison['ate_difference']:.4f}")
        logger.info("")
        logger.info(f"DragonNet ATT: {dragonnet_att:.4f}")
        logger.info(f"PSM ATT:       {att_matched.estimate:.4f}")
        logger.info(f"Difference:    {comparison['att_difference']:.4f}")

    # =========================================================================
    # 8. Save Results
    # =========================================================================
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save matched pairs
        if len(match_result.matched_pairs) > 0:
            matched_df = pd.DataFrame({
                'treated_idx': match_result.matched_pairs[:, 0],
                'control_idx': match_result.matched_pairs[:, 1],
                'distance': match_result.distances
            })
            matched_df.to_csv(output_dir / "matched_pairs.csv", index=False)

        # Save balance statistics
        balance_stats.to_csv(output_dir / "balance_statistics.csv", index=False)

        # Save sensitivity analysis
        sensitivity.to_csv(output_dir / "sensitivity_analysis.csv", index=False)

        # Save summary
        summary = {
            'n_samples': len(predictions_df),
            'n_treated': int(np.sum(treatment)),
            'n_control': int(np.sum(1 - treatment)),
            'n_matched_pairs': len(match_result.matched_pairs),
            'overlap_coefficient': overlap['overlap_coefficient'],
            'crude_difference': crude_diff,
            'att_matched': att_matched.estimate,
            'att_matched_ci': [att_matched.ci_lower, att_matched.ci_upper],
            'att_matched_pvalue': att_matched.p_value,
            'ate_ipw': ate_ipw.estimate,
            'ate_ipw_ci': [ate_ipw.ci_lower, ate_ipw.ci_upper],
            'ate_ipw_pvalue': ate_ipw.p_value,
            'ate_stratified': ate_strat.estimate,
        }

        if has_ite:
            summary['dragonnet_ate'] = dragonnet_ate
            summary['dragonnet_att'] = dragonnet_att

        import json
        with open(output_dir / "psm_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"\nPSM analysis results saved to: {output_dir}")

    return results


def compare_estimates(
    dragonnet_predictions: pd.DataFrame,
    psm_results: Dict[str, Any]
) -> pd.DataFrame:
    """
    Create a comparison table of DragonNet vs PSM estimates.

    Args:
        dragonnet_predictions: DataFrame with DragonNet predictions
        psm_results: Results from run_psm_analysis

    Returns:
        DataFrame comparing estimates
    """
    rows = []

    # DragonNet estimates
    if 'ite_pred' in dragonnet_predictions.columns:
        ite = dragonnet_predictions['ite_pred'].values
        treatment = dragonnet_predictions['treatment'].values

        rows.append({
            'Method': 'DragonNet',
            'Estimand': 'ATE',
            'Estimate': np.mean(ite),
            'CI_Lower': np.percentile(ite, 2.5),
            'CI_Upper': np.percentile(ite, 97.5),
            'P_Value': None
        })

        rows.append({
            'Method': 'DragonNet',
            'Estimand': 'ATT',
            'Estimate': np.mean(ite[treatment == 1]),
            'CI_Lower': None,
            'CI_Upper': None,
            'P_Value': None
        })

    # PSM estimates
    if 'att_matched' in psm_results:
        att = psm_results['att_matched']
        rows.append({
            'Method': 'PSM (Matched)',
            'Estimand': 'ATT',
            'Estimate': att.estimate,
            'CI_Lower': att.ci_lower,
            'CI_Upper': att.ci_upper,
            'P_Value': att.p_value
        })

    if 'ate_ipw' in psm_results:
        ate = psm_results['ate_ipw']
        rows.append({
            'Method': 'PSM (IPW)',
            'Estimand': 'ATE',
            'Estimate': ate.estimate,
            'CI_Lower': ate.ci_lower,
            'CI_Upper': ate.ci_upper,
            'P_Value': ate.p_value
        })

    if 'ate_stratified' in psm_results:
        ate = psm_results['ate_stratified']
        rows.append({
            'Method': 'PSM (Stratified)',
            'Estimand': 'ATE',
            'Estimate': ate.estimate,
            'CI_Lower': ate.ci_lower,
            'CI_Upper': ate.ci_upper,
            'P_Value': ate.p_value
        })

    return pd.DataFrame(rows)
