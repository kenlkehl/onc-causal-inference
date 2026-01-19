# cdt/models/__init__.py
"""Model components for propensity score matching and causal inference."""

# New propensity matching model
from .propensity_model import (
    PropensityModel,
    CNNEncoder,
    TransformerEncoder,
    GRUAttentionEncoder,
    PropensityHead,
    OutcomeHead
)

# Shared components
from .components import ConfounderAggregator
from .feature_extractor import FeatureExtractor

# Legacy DragonNet components (deprecated)
from .dragonnet import DragonNet
from .uplift import UpliftNet
from .outcome_heads import OutcomeHeadsOnly, UpliftHeadsOnly
from .causal_dragonnet import CausalDragonnetText
from .multitreatment import MultiTreatmentDragonNetInternal, MultiTreatmentDragonnetText

__all__ = [
    # New propensity matching
    'PropensityModel',
    'CNNEncoder',
    'TransformerEncoder',
    'GRUAttentionEncoder',
    'PropensityHead',
    'OutcomeHead',

    # Shared components
    'ConfounderAggregator',
    'FeatureExtractor',

    # Legacy (deprecated)
    'DragonNet',
    'UpliftNet',
    'OutcomeHeadsOnly',
    'UpliftHeadsOnly',
    'CausalDragonnetText',
    'MultiTreatmentDragonNetInternal',
    'MultiTreatmentDragonnetText',
]
