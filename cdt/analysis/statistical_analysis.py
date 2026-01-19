# cdt/analysis/statistical_analysis.py
"""Statistical analysis for causal inference from matched samples."""

import logging
from typing import Optional, Dict, Tuple, List, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats

from ..matching import MatchResult


logger = logging.getLogger(__name__)


@dataclass
class TreatmentEffectEstimate:
    """Container for treatment effect estimates."""

    # Point estimate
    estimate: float

    # Standard error
    std_error: float

    # Confidence interval
    ci_lower: float
    ci_upper: float

    # Confidence level (e.g., 0.95)
    ci_level: float

    # P-value (two-sided test of H0: effect = 0)
    p_value: float

    # Sample sizes
    n_treated: int
    n_control: int

    # Type of estimate ('ATT', 'ATE', 'ATC')
    estimand: str

    # Method used ('matched_diff', 'ipw', 'aipw')
    method: str

    def __repr__(self) -> str:
        return (
            f"{self.estimand} = {self.estimate:.4f} "
            f"(SE: {self.std_error:.4f}, "
            f"{int(self.ci_level*100)}% CI: [{self.ci_lower:.4f}, {self.ci_upper:.4f}], "
            f"p = {self.p_value:.4f})"
        )


def estimate_att_matched(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    match_result: MatchResult,
    ci_level: float = 0.95,
    n_bootstrap: int = 1000,
    random_state: Optional[int] = None
) -> TreatmentEffectEstimate:
    """
    Estimate Average Treatment Effect on the Treated (ATT) from matched sample.

    Uses difference in means between treated and matched controls.

    Args:
        outcomes: Array of outcome values
        treatment: Binary treatment indicator
        match_result: Matching result with paired indices
        ci_level: Confidence level for CI
        n_bootstrap: Number of bootstrap iterations for SE and CI
        random_state: Random seed for reproducibility

    Returns:
        TreatmentEffectEstimate with ATT and inference statistics
    """
    if random_state is not None:
        np.random.seed(random_state)

    outcomes = np.asarray(outcomes).flatten()
    treatment = np.asarray(treatment).flatten()

    if len(match_result.matched_pairs) == 0:
        logger.warning("No matched pairs for ATT estimation")
        return TreatmentEffectEstimate(
            estimate=np.nan,
            std_error=np.nan,
            ci_lower=np.nan,
            ci_upper=np.nan,
            ci_level=ci_level,
            p_value=np.nan,
            n_treated=0,
            n_control=0,
            estimand='ATT',
            method='matched_diff'
        )

    # Extract matched outcomes
    treated_idx = match_result.matched_pairs[:, 0]
    control_idx = match_result.matched_pairs[:, 1]

    y_treated = outcomes[treated_idx]
    y_control = outcomes[control_idx]

    # Point estimate (mean difference)
    pair_diffs = y_treated - y_control
    att_estimate = np.mean(pair_diffs)

    # Bootstrap for SE and CI
    n_pairs = len(pair_diffs)
    bootstrap_estimates = []

    for _ in range(n_bootstrap):
        # Resample pairs with replacement
        boot_idx = np.random.choice(n_pairs, size=n_pairs, replace=True)
        boot_diffs = pair_diffs[boot_idx]
        bootstrap_estimates.append(np.mean(boot_diffs))

    bootstrap_estimates = np.array(bootstrap_estimates)
    std_error = np.std(bootstrap_estimates, ddof=1)

    # Percentile CI
    alpha = 1 - ci_level
    ci_lower = np.percentile(bootstrap_estimates, alpha / 2 * 100)
    ci_upper = np.percentile(bootstrap_estimates, (1 - alpha / 2) * 100)

    # P-value using t-test on pair differences
    t_stat = att_estimate / (std_error + 1e-10)
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n_pairs - 1))

    return TreatmentEffectEstimate(
        estimate=att_estimate,
        std_error=std_error,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_level=ci_level,
        p_value=p_value,
        n_treated=len(np.unique(treated_idx)),
        n_control=len(np.unique(control_idx)),
        estimand='ATT',
        method='matched_diff'
    )


def estimate_ate_ipw(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    propensity_scores: np.ndarray,
    ci_level: float = 0.95,
    n_bootstrap: int = 1000,
    trim_quantile: float = 0.01,
    random_state: Optional[int] = None
) -> TreatmentEffectEstimate:
    """
    Estimate Average Treatment Effect (ATE) using Inverse Probability Weighting.

    ATE = E[Y(1) - Y(0)] estimated by:
        mean(T*Y/e(X)) - mean((1-T)*Y/(1-e(X)))

    Args:
        outcomes: Array of outcome values
        treatment: Binary treatment indicator
        propensity_scores: Estimated propensity scores
        ci_level: Confidence level for CI
        n_bootstrap: Number of bootstrap iterations
        trim_quantile: Quantile for trimming extreme propensity scores
        random_state: Random seed for reproducibility

    Returns:
        TreatmentEffectEstimate with ATE and inference statistics
    """
    if random_state is not None:
        np.random.seed(random_state)

    outcomes = np.asarray(outcomes).flatten()
    treatment = np.asarray(treatment).flatten()
    propensity_scores = np.asarray(propensity_scores).flatten()

    n = len(outcomes)

    # Trim extreme propensity scores
    lower = np.quantile(propensity_scores, trim_quantile)
    upper = np.quantile(propensity_scores, 1 - trim_quantile)
    ps_trimmed = np.clip(propensity_scores, lower, upper)

    def compute_ate(y, t, ps):
        """Compute IPW ATE estimate."""
        # Treated component: T*Y/e(X)
        treated_term = np.sum(t * y / ps) / np.sum(t / ps)

        # Control component: (1-T)*Y/(1-e(X))
        control_term = np.sum((1 - t) * y / (1 - ps)) / np.sum((1 - t) / (1 - ps))

        return treated_term - control_term

    # Point estimate
    ate_estimate = compute_ate(outcomes, treatment, ps_trimmed)

    # Bootstrap for SE and CI
    bootstrap_estimates = []
    for _ in range(n_bootstrap):
        boot_idx = np.random.choice(n, size=n, replace=True)
        boot_ate = compute_ate(
            outcomes[boot_idx],
            treatment[boot_idx],
            ps_trimmed[boot_idx]
        )
        bootstrap_estimates.append(boot_ate)

    bootstrap_estimates = np.array(bootstrap_estimates)
    std_error = np.std(bootstrap_estimates, ddof=1)

    # Percentile CI
    alpha = 1 - ci_level
    ci_lower = np.percentile(bootstrap_estimates, alpha / 2 * 100)
    ci_upper = np.percentile(bootstrap_estimates, (1 - alpha / 2) * 100)

    # P-value
    z_stat = ate_estimate / (std_error + 1e-10)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    return TreatmentEffectEstimate(
        estimate=ate_estimate,
        std_error=std_error,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_level=ci_level,
        p_value=p_value,
        n_treated=int(np.sum(treatment)),
        n_control=int(np.sum(1 - treatment)),
        estimand='ATE',
        method='ipw'
    )


def estimate_ate_stratified(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    propensity_scores: np.ndarray,
    n_strata: int = 5,
    ci_level: float = 0.95,
    n_bootstrap: int = 1000,
    random_state: Optional[int] = None
) -> TreatmentEffectEstimate:
    """
    Estimate ATE using propensity score stratification (subclassification).

    Divides sample into strata based on propensity score quantiles and
    computes weighted average of within-strata treatment effects.

    Args:
        outcomes: Array of outcome values
        treatment: Binary treatment indicator
        propensity_scores: Estimated propensity scores
        n_strata: Number of propensity score strata (default 5)
        ci_level: Confidence level for CI
        n_bootstrap: Number of bootstrap iterations
        random_state: Random seed for reproducibility

    Returns:
        TreatmentEffectEstimate with stratified ATE
    """
    if random_state is not None:
        np.random.seed(random_state)

    outcomes = np.asarray(outcomes).flatten()
    treatment = np.asarray(treatment).flatten()
    propensity_scores = np.asarray(propensity_scores).flatten()

    n = len(outcomes)

    # Define strata boundaries
    quantiles = np.linspace(0, 1, n_strata + 1)
    boundaries = np.quantile(propensity_scores, quantiles)

    def compute_stratified_ate(y, t, ps):
        """Compute stratified ATE estimate."""
        stratum_effects = []
        stratum_weights = []

        for i in range(n_strata):
            lower = boundaries[i]
            upper = boundaries[i + 1] if i < n_strata - 1 else boundaries[i + 1] + 1

            if i == n_strata - 1:
                mask = (ps >= lower) & (ps <= upper)
            else:
                mask = (ps >= lower) & (ps < upper)

            y_s = y[mask]
            t_s = t[mask]

            n_treated_s = np.sum(t_s)
            n_control_s = np.sum(1 - t_s)

            if n_treated_s > 0 and n_control_s > 0:
                effect_s = np.mean(y_s[t_s == 1]) - np.mean(y_s[t_s == 0])
                stratum_effects.append(effect_s)
                stratum_weights.append(np.sum(mask))
            else:
                stratum_effects.append(0)
                stratum_weights.append(0)

        stratum_effects = np.array(stratum_effects)
        stratum_weights = np.array(stratum_weights)

        if stratum_weights.sum() == 0:
            return np.nan

        return np.average(stratum_effects, weights=stratum_weights)

    # Point estimate
    ate_estimate = compute_stratified_ate(outcomes, treatment, propensity_scores)

    # Bootstrap for SE and CI
    bootstrap_estimates = []
    for _ in range(n_bootstrap):
        boot_idx = np.random.choice(n, size=n, replace=True)
        boot_ate = compute_stratified_ate(
            outcomes[boot_idx],
            treatment[boot_idx],
            propensity_scores[boot_idx]
        )
        if not np.isnan(boot_ate):
            bootstrap_estimates.append(boot_ate)

    bootstrap_estimates = np.array(bootstrap_estimates)

    if len(bootstrap_estimates) == 0:
        std_error = np.nan
        ci_lower = np.nan
        ci_upper = np.nan
        p_value = np.nan
    else:
        std_error = np.std(bootstrap_estimates, ddof=1)
        alpha = 1 - ci_level
        ci_lower = np.percentile(bootstrap_estimates, alpha / 2 * 100)
        ci_upper = np.percentile(bootstrap_estimates, (1 - alpha / 2) * 100)
        z_stat = ate_estimate / (std_error + 1e-10)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    return TreatmentEffectEstimate(
        estimate=ate_estimate,
        std_error=std_error,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_level=ci_level,
        p_value=p_value,
        n_treated=int(np.sum(treatment)),
        n_control=int(np.sum(1 - treatment)),
        estimand='ATE',
        method='stratified'
    )


def mcnemars_test(
    outcomes: np.ndarray,
    match_result: MatchResult
) -> Dict[str, float]:
    """
    McNemar's test for binary outcomes in matched pairs.

    Tests whether the marginal probability of outcome differs between
    treated and control groups.

    Args:
        outcomes: Binary outcome values
        match_result: Matching result with paired indices

    Returns:
        Dictionary with test statistic, p-value, and discordant pair counts
    """
    outcomes = np.asarray(outcomes).flatten()

    if len(match_result.matched_pairs) == 0:
        return {
            'statistic': np.nan,
            'p_value': np.nan,
            'b': 0,  # treated=1, control=0
            'c': 0   # treated=0, control=1
        }

    treated_idx = match_result.matched_pairs[:, 0]
    control_idx = match_result.matched_pairs[:, 1]

    y_treated = outcomes[treated_idx]
    y_control = outcomes[control_idx]

    # Count discordant pairs
    b = np.sum((y_treated == 1) & (y_control == 0))  # Treated=1, Control=0
    c = np.sum((y_treated == 0) & (y_control == 1))  # Treated=0, Control=1

    # McNemar's test statistic (with continuity correction)
    if b + c == 0:
        return {
            'statistic': 0.0,
            'p_value': 1.0,
            'b': int(b),
            'c': int(c)
        }

    # Use exact binomial test if n < 25
    if b + c < 25:
        # Exact binomial test: P(X >= b) under H0: p = 0.5
        p_value = 2 * min(
            stats.binom.cdf(min(b, c), b + c, 0.5),
            1 - stats.binom.cdf(max(b, c) - 1, b + c, 0.5)
        )
        statistic = (b - c) ** 2 / (b + c)
    else:
        # Chi-squared approximation with continuity correction
        statistic = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = 1 - stats.chi2.cdf(statistic, df=1)

    return {
        'statistic': float(statistic),
        'p_value': float(p_value),
        'b': int(b),
        'c': int(c)
    }


def paired_t_test(
    outcomes: np.ndarray,
    match_result: MatchResult
) -> Dict[str, float]:
    """
    Paired t-test for continuous outcomes in matched pairs.

    Args:
        outcomes: Continuous outcome values
        match_result: Matching result with paired indices

    Returns:
        Dictionary with t-statistic, p-value, and mean difference
    """
    outcomes = np.asarray(outcomes).flatten()

    if len(match_result.matched_pairs) == 0:
        return {
            't_statistic': np.nan,
            'p_value': np.nan,
            'mean_diff': np.nan,
            'std_diff': np.nan,
            'n_pairs': 0
        }

    treated_idx = match_result.matched_pairs[:, 0]
    control_idx = match_result.matched_pairs[:, 1]

    y_treated = outcomes[treated_idx]
    y_control = outcomes[control_idx]

    # Paired differences
    diffs = y_treated - y_control
    n_pairs = len(diffs)

    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, ddof=1)

    # t-test
    t_stat = mean_diff / (std_diff / np.sqrt(n_pairs) + 1e-10)
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n_pairs - 1))

    return {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'mean_diff': float(mean_diff),
        'std_diff': float(std_diff),
        'n_pairs': n_pairs
    }


def sensitivity_analysis_rosenbaum(
    outcomes: np.ndarray,
    match_result: MatchResult,
    gamma_values: Optional[List[float]] = None
) -> pd.DataFrame:
    """
    Rosenbaum sensitivity analysis for matched pairs.

    Tests how sensitive the treatment effect is to unmeasured confounding
    by computing bounds on p-values under different levels of hidden bias (gamma).

    Args:
        outcomes: Outcome values
        match_result: Matching result with paired indices
        gamma_values: Values of gamma to test (odds ratio of hidden confounding)

    Returns:
        DataFrame with p-value bounds for each gamma
    """
    if gamma_values is None:
        gamma_values = [1.0, 1.1, 1.2, 1.3, 1.5, 2.0, 2.5, 3.0]

    outcomes = np.asarray(outcomes).flatten()

    if len(match_result.matched_pairs) == 0:
        return pd.DataFrame({
            'gamma': gamma_values,
            'p_upper': [np.nan] * len(gamma_values),
            'p_lower': [np.nan] * len(gamma_values)
        })

    treated_idx = match_result.matched_pairs[:, 0]
    control_idx = match_result.matched_pairs[:, 1]

    y_treated = outcomes[treated_idx]
    y_control = outcomes[control_idx]

    # Compute pair differences
    diffs = y_treated - y_control

    # Sign of differences (for Wilcoxon signed-rank)
    positive = np.sum(diffs > 0)
    negative = np.sum(diffs < 0)
    n_nonzero = positive + negative

    if n_nonzero == 0:
        return pd.DataFrame({
            'gamma': gamma_values,
            'p_upper': [1.0] * len(gamma_values),
            'p_lower': [1.0] * len(gamma_values)
        })

    results = []
    for gamma in gamma_values:
        if gamma == 1.0:
            # Standard sign test
            p = 2 * min(
                stats.binom.cdf(min(positive, negative), n_nonzero, 0.5),
                1 - stats.binom.cdf(max(positive, negative) - 1, n_nonzero, 0.5)
            )
            results.append({
                'gamma': gamma,
                'p_upper': p,
                'p_lower': p
            })
        else:
            # Bounds under gamma confounding
            p_plus = gamma / (1 + gamma)  # Upper bound on probability of positive diff
            p_minus = 1 / (1 + gamma)     # Lower bound

            # P-value under worst case (treated more likely to have higher outcome)
            p_upper = 2 * min(
                stats.binom.cdf(negative, n_nonzero, p_plus),
                1 - stats.binom.cdf(positive - 1, n_nonzero, p_plus)
            )

            # P-value under best case
            p_lower = 2 * min(
                stats.binom.cdf(negative, n_nonzero, p_minus),
                1 - stats.binom.cdf(positive - 1, n_nonzero, p_minus)
            )

            results.append({
                'gamma': gamma,
                'p_upper': min(1.0, p_upper),
                'p_lower': min(1.0, p_lower)
            })

    return pd.DataFrame(results)


def summarize_analysis(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    propensity_scores: np.ndarray,
    match_result: Optional[MatchResult] = None,
    ci_level: float = 0.95
) -> Dict[str, any]:
    """
    Comprehensive summary of causal analysis.

    Args:
        outcomes: Outcome values
        treatment: Binary treatment indicator
        propensity_scores: Estimated propensity scores
        match_result: Optional matching result
        ci_level: Confidence level for CIs

    Returns:
        Dictionary with all analysis results
    """
    outcomes = np.asarray(outcomes).flatten()
    treatment = np.asarray(treatment).flatten()

    results = {
        'sample_size': len(outcomes),
        'n_treated': int(np.sum(treatment)),
        'n_control': int(np.sum(1 - treatment)),
        'outcome_rate_treated': float(np.mean(outcomes[treatment == 1])),
        'outcome_rate_control': float(np.mean(outcomes[treatment == 0])),
        'crude_difference': float(np.mean(outcomes[treatment == 1]) - np.mean(outcomes[treatment == 0]))
    }

    # IPW estimate
    ate_ipw = estimate_ate_ipw(outcomes, treatment, propensity_scores, ci_level=ci_level)
    results['ate_ipw'] = ate_ipw

    # Stratified estimate
    ate_strat = estimate_ate_stratified(outcomes, treatment, propensity_scores, ci_level=ci_level)
    results['ate_stratified'] = ate_strat

    # Matched estimates (if matching done)
    if match_result is not None:
        results['n_matched_pairs'] = len(match_result.matched_pairs)

        # ATT from matched sample
        att_matched = estimate_att_matched(outcomes, treatment, match_result, ci_level=ci_level)
        results['att_matched'] = att_matched

        # Statistical tests
        if len(np.unique(outcomes)) == 2:  # Binary outcome
            results['mcnemar_test'] = mcnemars_test(outcomes, match_result)
        else:
            results['paired_t_test'] = paired_t_test(outcomes, match_result)

        # Sensitivity analysis
        results['sensitivity_analysis'] = sensitivity_analysis_rosenbaum(outcomes, match_result)

    return results
