"""Small causal-modeling primitives shared by active Stage 1 lanes."""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.metrics import roc_auc_score


def safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    if len(np.unique(y_true)) < 2:
        return None
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return None


__all__ = ["safe_roc_auc"]
