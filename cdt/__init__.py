"""
Causal Dragonnet Text (CDT)

A package for causal inference from clinical text using CNN-based DragonNet models.
"""

__version__ = "0.1.0"

from .config import (
    ExperimentConfig,
    AppliedInferenceConfig,
    PlasmodeExperimentConfig,
    ModelArchitectureConfig,
    TrainingConfig,
    MatchingAnalysisConfig,
    MatchedPairConfig,
    create_default_config
)

from .experiments import ExperimentRunner, MatchedPairExperimentRunner

__all__ = [
    'ExperimentConfig',
    'AppliedInferenceConfig',
    'PlasmodeExperimentConfig',
    'ModelArchitectureConfig',
    'TrainingConfig',
    'MatchingAnalysisConfig',
    'MatchedPairConfig',
    'ExperimentRunner',
    'MatchedPairExperimentRunner',
    'create_default_config',
]
