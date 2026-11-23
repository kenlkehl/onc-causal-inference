"""High-dimensional sparse tabular causal foundation model (SparseTabularCausalPFN).

Designed for Stage-2 / all-evidence extracted candidate tables with up to P = 2,000+
candidate features where most columns are noise, collinear duplicates, or pure
instruments, and only a sparse subset are confounders, prognostics, or effect modifiers.
"""

from .model import (
    CHECKPOINT_FORMAT_VERSION,
    ROLE_NAMES,
    SparseTabularCausalPFN,
    TabularModelConfig,
)
from .preprocessing import TabularPreprocessor
from .prior import ROLE_TO_ID, PriorConfig, sample_tabular_study

__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "ROLE_NAMES",
    "ROLE_TO_ID",
    "PriorConfig",
    "SparseTabularCausalPFN",
    "TabularModelConfig",
    "TabularPreprocessor",
    "sample_tabular_study",
]
