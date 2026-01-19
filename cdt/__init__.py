"""
Propensity Score Matching for Clinical Text (PSM-CT)

A package for causal inference from clinical text using propensity score matching
with deep learning text encoders (CNN, Transformer, GRU with attention).

Main components:
- PropensityModel: Text encoder with propensity score prediction
- PropensityMatcher: Traditional propensity score matching algorithms
- Statistical analysis: ATE/ATT estimation, balance diagnostics, sensitivity analysis
"""

__version__ = "0.2.0"

# New propensity matching configuration (recommended)
from .config import (
    PropensityExperimentConfig,
    PropensityModelConfig,
    PropensityTrainingConfig,
    MatchingConfig,
    create_propensity_config
)

# New propensity matching models
from .models import (
    PropensityModel,
    CNNEncoder,
    TransformerEncoder,
    GRUAttentionEncoder
)

# Matching algorithms
from .matching import (
    PropensityMatcher,
    MatchResult,
    compute_balance_statistics,
    assess_overlap
)

# Statistical analysis
from .analysis import (
    TreatmentEffectEstimate,
    estimate_att_matched,
    estimate_ate_ipw,
    estimate_ate_stratified,
    summarize_analysis
)

# Training pipeline
from .training import (
    train_propensity_model,
    run_propensity_matching_pipeline
)

# Legacy imports for backward compatibility (deprecated)
from .config import (
    ExperimentConfig,
    AppliedInferenceConfig,
    PretrainingConfig,
    PlasmodeExperimentConfig,
    create_default_config
)

from .experiments import ExperimentRunner

__all__ = [
    # New propensity matching (recommended)
    'PropensityExperimentConfig',
    'PropensityModelConfig',
    'PropensityTrainingConfig',
    'MatchingConfig',
    'create_propensity_config',

    'PropensityModel',
    'CNNEncoder',
    'TransformerEncoder',
    'GRUAttentionEncoder',

    'PropensityMatcher',
    'MatchResult',
    'compute_balance_statistics',
    'assess_overlap',

    'TreatmentEffectEstimate',
    'estimate_att_matched',
    'estimate_ate_ipw',
    'estimate_ate_stratified',
    'summarize_analysis',

    'train_propensity_model',
    'run_propensity_matching_pipeline',

    # Legacy (deprecated)
    'ExperimentConfig',
    'AppliedInferenceConfig',
    'PretrainingConfig',
    'PlasmodeExperimentConfig',
    'ExperimentRunner',
    'create_default_config',
]
