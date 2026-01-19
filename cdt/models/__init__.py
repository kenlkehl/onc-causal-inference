# cdt/models/__init__.py
"""Model components for causal inference from clinical text."""

# Main model
from .causal_dragonnet import CausalDragonnetText

# DragonNet architecture components
from .dragonnet import DragonNet
from .uplift import UpliftNet
from .feature_extractor import FeatureExtractor
from .components import ConfounderAggregator

# Outcome heads for oracle/plasmode mode
from .outcome_heads import OutcomeHeadsOnly, UpliftHeadsOnly

# Multi-treatment pretraining
from .multitreatment import MultiTreatmentDragonNetInternal, MultiTreatmentDragonnetText

__all__ = [
    # Main model
    'CausalDragonnetText',

    # Core components
    'DragonNet',
    'UpliftNet',
    'FeatureExtractor',
    'ConfounderAggregator',

    # Oracle mode
    'OutcomeHeadsOnly',
    'UpliftHeadsOnly',

    # Multi-treatment
    'MultiTreatmentDragonNetInternal',
    'MultiTreatmentDragonnetText',
]
