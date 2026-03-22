"""
Oncology Causal Inference (OCI)

A package for causal inference from clinical text using frozen LLM feature
extraction and causal inference heads (DragonNet, R-Learner, Causal Forest).
"""

__version__ = "0.1.0"

from .config import (
    ExperimentConfig,
    AppliedInferenceConfig,
    ModelArchitectureConfig,
    TrainingConfig,
    MatchingAnalysisConfig,
    create_default_config
)

from .experiments import ExperimentRunner

__all__ = [
    'ExperimentConfig',
    'AppliedInferenceConfig',
    'ModelArchitectureConfig',
    'TrainingConfig',
    'MatchingAnalysisConfig',
    'ExperimentRunner',
    'create_default_config',
]
