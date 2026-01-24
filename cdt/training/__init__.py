# cdt/training/__init__.py

"""Training modules for CDT - CNN-based approach."""

from .plasmode import run_plasmode_experiments
from .matched_pair_training import (
    MatchedPairDataset,
    matched_pair_loss,
    train_propensity_model,
    train_matched_pair_outcome_model,
    extract_all_representations,
    extract_propensity_scores
)

__all__ = [
    'run_plasmode_experiments',
    'MatchedPairDataset',
    'matched_pair_loss',
    'train_propensity_model',
    'train_matched_pair_outcome_model',
    'extract_all_representations',
    'extract_propensity_scores',
]
