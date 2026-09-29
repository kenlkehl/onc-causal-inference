"""Small, dependency-light reproduction of OCI's candidate ridge score.

The score matches the nested design used in
``oci/inference/stage2_elastic_net_selection.py``:

    reduced = [1, V, A - e(X)]
    full    = [1, V, A - e(X), (A - e(X)) * V]

Both models use ``Ridge(alpha=10, fit_intercept=False)``.  The candidate score
is held-out reduced MSE minus held-out full MSE.  The rank checks below are
copied into this research module so the experiment does not depend on private
functions or a historical Git object.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge


def rank_safe_columns(base: np.ndarray, additions: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Append columns only when they increase matrix rank, preserving order."""

    current = np.asarray(base, dtype=float)
    additions = np.asarray(additions, dtype=float)
    rank = int(np.linalg.matrix_rank(current))
    kept: list[int] = []
    for index in range(additions.shape[1]):
        candidate = np.column_stack([current, additions[:, index]])
        next_rank = int(np.linalg.matrix_rank(candidate))
        if next_rank > rank:
            current = candidate
            rank = next_rank
            kept.append(index)
    return current, kept


def rank_safe_pairs(
    train_base: np.ndarray,
    valid_base: np.ndarray,
    train_additions: np.ndarray,
    valid_additions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Choose columns using training rank and mirror them in validation."""

    train_result, kept = rank_safe_columns(train_base, train_additions)
    valid_result = np.column_stack([valid_base, valid_additions[:, kept]])
    return train_result, valid_result, kept


def ridge_interaction_gain(
    train_features: np.ndarray,
    valid_features: np.ndarray,
    train_treatment_residual: np.ndarray,
    valid_treatment_residual: np.ndarray,
    train_outcome_residual: np.ndarray,
    valid_outcome_residual: np.ndarray,
    *,
    alpha: float = 10.0,
) -> float:
    """Return the OCI held-out interaction gain for one candidate group."""

    x_train = np.asarray(train_features, dtype=float)
    x_valid = np.asarray(valid_features, dtype=float)
    ra_train = np.asarray(train_treatment_residual, dtype=float)
    ra_valid = np.asarray(valid_treatment_residual, dtype=float)
    ry_train = np.asarray(train_outcome_residual, dtype=float)
    ry_valid = np.asarray(valid_outcome_residual, dtype=float)

    reduced_train, reduced_valid, _ = rank_safe_pairs(
        np.ones((len(x_train), 1)),
        np.ones((len(x_valid), 1)),
        x_train,
        x_valid,
    )
    reduced_train, reduced_valid, kept_treatment = rank_safe_pairs(
        reduced_train,
        reduced_valid,
        ra_train[:, None],
        ra_valid[:, None],
    )
    if not kept_treatment:
        raise ValueError("Residualized treatment has no independent variation")

    full_train, full_valid, kept_interactions = rank_safe_pairs(
        reduced_train,
        reduced_valid,
        ra_train[:, None] * x_train,
        ra_valid[:, None] * x_valid,
    )
    if not kept_interactions:
        raise ValueError("Candidate has no independent treatment interaction")

    reduced = Ridge(alpha=alpha, fit_intercept=False).fit(reduced_train, ry_train)
    full = Ridge(alpha=alpha, fit_intercept=False).fit(full_train, ry_train)
    reduced_mse = float(np.mean((ry_valid - reduced.predict(reduced_valid)) ** 2))
    full_mse = float(np.mean((ry_valid - full.predict(full_valid)) ** 2))
    return reduced_mse - full_mse
