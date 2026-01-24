# cdt/matching/__init__.py
"""Propensity score matching module."""

from .propensity_matcher import (
    PropensityMatcher,
    MatchResult,
    compute_standardized_mean_difference,
    compute_balance_statistics,
    assess_overlap,
    match_by_cosine_similarity
)

__all__ = [
    'PropensityMatcher',
    'MatchResult',
    'compute_standardized_mean_difference',
    'compute_balance_statistics',
    'assess_overlap',
    'match_by_cosine_similarity'
]
