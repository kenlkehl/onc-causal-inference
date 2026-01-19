# cdt/training/__init__.py
"""Training modules for propensity score matching and causal inference."""

# New propensity matching training
from .propensity_training import (
    train_propensity_model,
    predict_propensity_scores,
    run_propensity_matching_pipeline
)

# Legacy training modules (deprecated)
from .pretraining import run_pretraining
from .plasmode import run_plasmode_experiments

__all__ = [
    # New propensity matching
    'train_propensity_model',
    'predict_propensity_scores',
    'run_propensity_matching_pipeline',

    # Legacy (deprecated)
    'run_pretraining',
    'run_plasmode_experiments',
]
