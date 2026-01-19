"""
Causal DragonNet Text (CDT)

A package for causal inference from clinical text using DragonNet models.

Key Features:
- DragonNet/UpliftNet for Individual Treatment Effect (ITE) estimation
- Traditional PSM analysis using DragonNet's propensity scores for validation
- Balance diagnostics and sensitivity analysis
- Plasmode simulation for method validation
"""

__version__ = "0.2.0"

# Main configuration
from .config import (
    ExperimentConfig,
    AppliedInferenceConfig,
    PretrainingConfig,
    PlasmodeExperimentConfig,
    MatchingAnalysisConfig,
    create_default_config
)

# Models
from .models import (
    CausalDragonnetText,
    DragonNet,
    UpliftNet,
    FeatureExtractor
)

# Matching algorithms (for PSM analysis)
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
    run_psm_analysis,
    compare_estimates
)

# Experiment runner
from .experiments import ExperimentRunner

__all__ = [
    # Config
    'ExperimentConfig',
    'AppliedInferenceConfig',
    'PretrainingConfig',
    'PlasmodeExperimentConfig',
    'MatchingAnalysisConfig',
    'create_default_config',

    # Models
    'CausalDragonnetText',
    'DragonNet',
    'UpliftNet',
    'FeatureExtractor',

    # Matching
    'PropensityMatcher',
    'MatchResult',
    'compute_balance_statistics',
    'assess_overlap',

    # Analysis
    'TreatmentEffectEstimate',
    'estimate_att_matched',
    'estimate_ate_ipw',
    'estimate_ate_stratified',
    'run_psm_analysis',
    'compare_estimates',

    # Runner
    'ExperimentRunner',
]
