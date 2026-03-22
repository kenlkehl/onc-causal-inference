# oci/matching/propensity_matcher.py
"""Propensity score matching algorithms for causal inference."""

import logging
from typing import Optional, List, Dict, Tuple, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Container for matching results."""

    # Indices of matched pairs (treated_idx, control_idx)
    matched_pairs: np.ndarray  # Shape: (n_matches, 2)

    # Propensity scores of matched units
    treated_propensities: np.ndarray
    control_propensities: np.ndarray

    # Distance between matched units
    distances: np.ndarray

    # Number of treated units that couldn't be matched
    n_unmatched_treated: int

    # Number of control units that couldn't be matched (for 1:1 matching)
    n_unmatched_control: int

    # Original sample sizes
    n_treated: int
    n_control: int

    def __repr__(self) -> str:
        return (
            f"MatchResult(n_matches={len(self.matched_pairs)}, "
            f"n_treated={self.n_treated}, n_control={self.n_control}, "
            f"unmatched_treated={self.n_unmatched_treated}, "
            f"mean_distance={self.distances.mean():.4f})"
        )


class PropensityMatcher:
    """
    Traditional propensity score matching.

    Supports:
    - Nearest neighbor matching (1:1 or 1:k with replacement)
    - Caliper matching (with or without replacement)
    - Optimal matching (minimizes total distance)
    """

    def __init__(
        self,
        method: str = 'nearest',
        caliper: Optional[float] = None,
        caliper_scale: str = 'propensity',  # 'propensity' or 'logit' or 'std'
        ratio: int = 1,  # 1:k matching ratio
        replacement: bool = False,
        random_state: Optional[int] = None
    ):
        """
        Initialize matcher.

        Args:
            method: Matching method ('nearest', 'optimal', 'caliper')
            caliper: Maximum distance for a valid match. If None, no caliper used.
                     For 'propensity' scale, this is absolute propensity difference.
                     For 'logit' scale, this is logit difference.
                     For 'std' scale, this is in standard deviations of logit propensity.
            caliper_scale: Scale for caliper ('propensity', 'logit', 'std')
            ratio: Number of controls to match per treated unit (1:k matching)
            replacement: Whether to match with replacement
            random_state: Random seed for reproducibility
        """
        valid_methods = {'nearest', 'optimal', 'caliper'}
        if method not in valid_methods:
            raise ValueError(f"Invalid method '{method}'. Must be one of {valid_methods}")

        self.method = method
        self.caliper = caliper
        self.caliper_scale = caliper_scale
        self.ratio = ratio
        self.replacement = replacement
        self.random_state = random_state

        if random_state is not None:
            np.random.seed(random_state)

    def match(
        self,
        propensity_scores: np.ndarray,
        treatment: np.ndarray
    ) -> MatchResult:
        """
        Perform propensity score matching.

        Args:
            propensity_scores: Array of propensity scores (n_samples,)
            treatment: Binary treatment indicator (n_samples,)

        Returns:
            MatchResult with matched pairs and diagnostics
        """
        propensity_scores = np.asarray(propensity_scores).flatten()
        treatment = np.asarray(treatment).flatten()

        if len(propensity_scores) != len(treatment):
            raise ValueError("Propensity scores and treatment must have same length")

        # Get treated and control indices
        treated_idx = np.where(treatment == 1)[0]
        control_idx = np.where(treatment == 0)[0]

        n_treated = len(treated_idx)
        n_control = len(control_idx)

        logger.info(f"Matching {n_treated} treated to {n_control} controls")
        logger.info(f"  Method: {self.method}, Ratio: 1:{self.ratio}, Replacement: {self.replacement}")

        if n_treated == 0 or n_control == 0:
            logger.warning("No treated or control units to match")
            return MatchResult(
                matched_pairs=np.array([]).reshape(0, 2),
                treated_propensities=np.array([]),
                control_propensities=np.array([]),
                distances=np.array([]),
                n_unmatched_treated=n_treated,
                n_unmatched_control=n_control,
                n_treated=n_treated,
                n_control=n_control
            )

        # Get propensity scores for each group
        ps_treated = propensity_scores[treated_idx]
        ps_control = propensity_scores[control_idx]

        # Convert to appropriate scale for matching
        if self.caliper_scale == 'logit':
            # Convert to logit scale
            eps = 1e-6
            ps_treated_match = np.log(np.clip(ps_treated, eps, 1 - eps) /
                                       np.clip(1 - ps_treated, eps, 1 - eps))
            ps_control_match = np.log(np.clip(ps_control, eps, 1 - eps) /
                                       np.clip(1 - ps_control, eps, 1 - eps))
        else:
            ps_treated_match = ps_treated
            ps_control_match = ps_control

        # Compute caliper in appropriate units
        if self.caliper is not None and self.caliper_scale == 'std':
            # Convert std to absolute scale
            all_logit = np.log(np.clip(propensity_scores, 1e-6, 1 - 1e-6) /
                               np.clip(1 - propensity_scores, 1e-6, 1 - 1e-6))
            caliper_abs = self.caliper * np.std(all_logit)
        else:
            caliper_abs = self.caliper

        # Perform matching based on method
        if self.method == 'optimal':
            result = self._optimal_matching(
                treated_idx, control_idx, ps_treated_match, ps_control_match,
                ps_treated, ps_control, caliper_abs
            )
        else:
            result = self._nearest_neighbor_matching(
                treated_idx, control_idx, ps_treated_match, ps_control_match,
                ps_treated, ps_control, caliper_abs
            )

        return result

    def _nearest_neighbor_matching(
        self,
        treated_idx: np.ndarray,
        control_idx: np.ndarray,
        ps_treated: np.ndarray,
        ps_control: np.ndarray,
        ps_treated_orig: np.ndarray,
        ps_control_orig: np.ndarray,
        caliper: Optional[float]
    ) -> MatchResult:
        """Nearest neighbor matching (greedy)."""

        n_treated = len(treated_idx)
        n_control = len(control_idx)

        matched_pairs = []
        matched_ps_treated = []
        matched_ps_control = []
        distances = []

        # Track which controls have been used (if no replacement)
        available_controls = set(range(n_control))

        # Shuffle treated order for randomization
        order = np.random.permutation(n_treated)

        for i in order:
            if not available_controls:
                break

            ps_t = ps_treated[i]

            # Find k nearest neighbors among available controls
            available_list = list(available_controls)
            ps_c_available = ps_control[available_list]

            # Compute distances
            dists = np.abs(ps_c_available - ps_t)

            # Apply caliper if specified
            if caliper is not None:
                valid_mask = dists <= caliper
                if not valid_mask.any():
                    continue  # No valid match for this treated unit
                valid_indices = np.where(valid_mask)[0]
                dists = dists[valid_indices]
                available_list = [available_list[j] for j in valid_indices]

            # Get k nearest
            k = min(self.ratio, len(available_list))
            nearest_indices = np.argsort(dists)[:k]

            for j in nearest_indices:
                control_local_idx = available_list[j]

                matched_pairs.append([treated_idx[i], control_idx[control_local_idx]])
                matched_ps_treated.append(ps_treated_orig[i])
                matched_ps_control.append(ps_control_orig[control_local_idx])
                distances.append(dists[j])

                if not self.replacement:
                    available_controls.discard(control_local_idx)

        matched_pairs = np.array(matched_pairs) if matched_pairs else np.array([]).reshape(0, 2)

        # Count unique treated that got matched
        if len(matched_pairs) > 0:
            unique_treated = len(np.unique(matched_pairs[:, 0]))
            n_unmatched_treated = n_treated - unique_treated
        else:
            n_unmatched_treated = n_treated

        n_unmatched_control = n_control - len(available_controls) if not self.replacement else 0

        return MatchResult(
            matched_pairs=matched_pairs,
            treated_propensities=np.array(matched_ps_treated),
            control_propensities=np.array(matched_ps_control),
            distances=np.array(distances),
            n_unmatched_treated=n_unmatched_treated,
            n_unmatched_control=len(available_controls),
            n_treated=n_treated,
            n_control=n_control
        )

    def _optimal_matching(
        self,
        treated_idx: np.ndarray,
        control_idx: np.ndarray,
        ps_treated: np.ndarray,
        ps_control: np.ndarray,
        ps_treated_orig: np.ndarray,
        ps_control_orig: np.ndarray,
        caliper: Optional[float]
    ) -> MatchResult:
        """Optimal matching using Hungarian algorithm."""

        n_treated = len(treated_idx)
        n_control = len(control_idx)

        # Compute full distance matrix
        dist_matrix = cdist(
            ps_treated.reshape(-1, 1),
            ps_control.reshape(-1, 1),
            metric='cityblock'
        )

        # Apply caliper by setting invalid matches to infinity
        if caliper is not None:
            dist_matrix[dist_matrix > caliper] = 1e10

        # Use Hungarian algorithm for 1:1 matching
        if self.ratio == 1:
            row_ind, col_ind = linear_sum_assignment(dist_matrix)

            # Filter out invalid matches (distance = inf)
            valid_mask = dist_matrix[row_ind, col_ind] < 1e9
            row_ind = row_ind[valid_mask]
            col_ind = col_ind[valid_mask]

            matched_pairs = np.column_stack([treated_idx[row_ind], control_idx[col_ind]])
            distances = dist_matrix[row_ind, col_ind]

            return MatchResult(
                matched_pairs=matched_pairs,
                treated_propensities=ps_treated_orig[row_ind],
                control_propensities=ps_control_orig[col_ind],
                distances=distances,
                n_unmatched_treated=n_treated - len(row_ind),
                n_unmatched_control=n_control - len(col_ind),
                n_treated=n_treated,
                n_control=n_control
            )
        else:
            # For k:1 matching, fall back to greedy
            logger.warning("Optimal matching with ratio > 1 not implemented, using nearest neighbor")
            return self._nearest_neighbor_matching(
                treated_idx, control_idx, ps_treated, ps_control,
                ps_treated_orig, ps_control_orig, caliper
            )


def compute_standardized_mean_difference(
    treated_values: np.ndarray,
    control_values: np.ndarray
) -> float:
    """
    Compute standardized mean difference (SMD) for covariate balance.

    SMD = (mean_treated - mean_control) / sqrt((var_treated + var_control) / 2)

    Returns:
        Absolute SMD value
    """
    mean_t = np.mean(treated_values)
    mean_c = np.mean(control_values)
    var_t = np.var(treated_values, ddof=1) if len(treated_values) > 1 else 0
    var_c = np.var(control_values, ddof=1) if len(control_values) > 1 else 0

    pooled_std = np.sqrt((var_t + var_c) / 2)

    if pooled_std < 1e-10:
        return 0.0

    return abs(mean_t - mean_c) / pooled_std


def compute_balance_statistics(
    covariates: pd.DataFrame,
    treatment: np.ndarray,
    match_result: Optional[MatchResult] = None
) -> pd.DataFrame:
    """
    Compute covariate balance statistics before and after matching.

    Args:
        covariates: DataFrame with covariate columns
        treatment: Binary treatment indicator
        match_result: Optional matching result. If provided, computes post-match balance.

    Returns:
        DataFrame with balance statistics for each covariate
    """
    treatment = np.asarray(treatment).flatten()

    results = []

    for col in covariates.columns:
        values = covariates[col].values

        # Pre-match balance
        treated_vals = values[treatment == 1]
        control_vals = values[treatment == 0]

        smd_before = compute_standardized_mean_difference(treated_vals, control_vals)

        result = {
            'covariate': col,
            'mean_treated_before': np.mean(treated_vals),
            'mean_control_before': np.mean(control_vals),
            'smd_before': smd_before
        }

        # Post-match balance (if matching result provided)
        if match_result is not None and len(match_result.matched_pairs) > 0:
            treated_matched_idx = match_result.matched_pairs[:, 0]
            control_matched_idx = match_result.matched_pairs[:, 1]

            treated_vals_post = values[treated_matched_idx]
            control_vals_post = values[control_matched_idx]

            smd_after = compute_standardized_mean_difference(treated_vals_post, control_vals_post)

            result.update({
                'mean_treated_after': np.mean(treated_vals_post),
                'mean_control_after': np.mean(control_vals_post),
                'smd_after': smd_after,
                'smd_reduction': (smd_before - smd_after) / smd_before if smd_before > 0 else 0
            })

        results.append(result)

    return pd.DataFrame(results)


def assess_overlap(
    propensity_scores: np.ndarray,
    treatment: np.ndarray,
    bins: int = 50
) -> Dict[str, float]:
    """
    Assess overlap (positivity) assumption using propensity score distributions.

    Returns:
        Dictionary with overlap metrics
    """
    propensity_scores = np.asarray(propensity_scores).flatten()
    treatment = np.asarray(treatment).flatten()

    ps_treated = propensity_scores[treatment == 1]
    ps_control = propensity_scores[treatment == 0]

    # Common support region
    min_treated, max_treated = ps_treated.min(), ps_treated.max()
    min_control, max_control = ps_control.min(), ps_control.max()

    common_min = max(min_treated, min_control)
    common_max = min(max_treated, max_control)

    # Fraction in common support
    treated_in_support = np.mean((ps_treated >= common_min) & (ps_treated <= common_max))
    control_in_support = np.mean((ps_control >= common_min) & (ps_control <= common_max))

    # Overlap coefficient (histogram intersection)
    hist_t, bin_edges = np.histogram(ps_treated, bins=bins, range=(0, 1), density=True)
    hist_c, _ = np.histogram(ps_control, bins=bins, range=(0, 1), density=True)

    # Normalize to make them proper densities
    bin_width = bin_edges[1] - bin_edges[0]
    hist_t = hist_t * bin_width
    hist_c = hist_c * bin_width

    overlap_coef = np.sum(np.minimum(hist_t, hist_c))

    return {
        'common_support_min': common_min,
        'common_support_max': common_max,
        'treated_in_support': treated_in_support,
        'control_in_support': control_in_support,
        'overlap_coefficient': overlap_coef,
        'ps_treated_mean': ps_treated.mean(),
        'ps_control_mean': ps_control.mean(),
        'ps_treated_std': ps_treated.std(),
        'ps_control_std': ps_control.std()
    }
