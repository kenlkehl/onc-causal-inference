# cdt/models/causal_cnn.py
"""Backward compatibility module - redirects to causal_text.py.

This module is deprecated. Use cdt.models.causal_text or cdt.models.CausalText instead.
"""

import warnings

warnings.warn(
    "cdt.models.causal_cnn is deprecated. Use cdt.models.causal_text instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from causal_text for backward compatibility
from .causal_text import CausalText, CausalCNNText

__all__ = ['CausalText', 'CausalCNNText']
