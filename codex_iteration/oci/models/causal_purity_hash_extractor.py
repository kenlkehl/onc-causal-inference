"""Outcome-guided token-hash feature discovery without clinical strings.

This module builds generic token-ID hash features from text and scores them by
whether they act like effect modifiers: a large treatment-effect contrast
between feature-present and feature-absent groups, with little control-arm
baseline shift. It never decodes tokens or names clinical concepts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
from scipy import sparse

from .token_windowing import select_token_window, validate_document_window


@dataclass
class CausalPurityHashSelection:
    """Selected hash buckets and their train-fold scores."""

    columns: np.ndarray
    original_hashes: np.ndarray
    scores: np.ndarray
    te_diff: np.ndarray
    control_diff: np.ndarray
    counts: np.ndarray


def hash_token_ngrams(
    ids: np.ndarray,
    num_hash_buckets: int = 32768,
    ngram_orders: Sequence[int] = (1, 2, 3),
) -> np.ndarray:
    """Hash generic token-ID n-grams to bucket ids."""
    parts = []
    for order in ngram_orders:
        order = int(order)
        if len(ids) < order:
            continue
        if order == 1:
            hashed = ids * 1_000_003 + 9_176
        else:
            hashed = np.zeros(len(ids) - order + 1, dtype=np.int64)
            for offset in range(order):
                hashed += ids[offset: len(ids) - order + 1 + offset] * (
                    1_000_003 + 9_176 * (offset + 1)
                )
            hashed += 19_260_817 * order
        parts.append(np.remainder(hashed, num_hash_buckets))
    if not parts:
        return np.zeros(1, dtype=np.int64)
    return np.unique(np.concatenate(parts)).astype(np.int64, copy=False)


def build_token_hash_matrix(
    texts: Sequence[str],
    model_name: str,
    max_length: int = 50000,
    num_hash_buckets: int = 32768,
    ngram_orders: Sequence[int] = (1, 2, 3),
    document_window: str = "tail",
    min_count: int = 5,
    max_count: int | None = None,
) -> Tuple[sparse.csr_matrix, np.ndarray]:
    """Build a binary document x token-hash matrix from raw text."""
    from transformers import AutoTokenizer

    document_window = validate_document_window(document_window)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
        truncation_side="left",
    )

    rows = []
    cols = []
    for row, text in enumerate(texts):
        encoding = tokenizer(str(text), truncation=False, padding=False)
        ids = np.asarray(
            select_token_window(
                encoding["input_ids"],
                max_length=max_length,
                document_window=document_window,
            ),
            dtype=np.int64,
        )
        if ids.size == 0:
            ids = np.zeros(1, dtype=np.int64)
        buckets = hash_token_ngrams(ids, num_hash_buckets, ngram_orders)
        rows.extend([row] * len(buckets))
        cols.extend(buckets.tolist())

    data = np.ones(len(rows), dtype=np.float32)
    matrix = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(len(texts), num_hash_buckets),
        dtype=np.float32,
    )
    matrix.sum_duplicates()
    matrix.data[:] = 1.0

    counts = np.asarray(matrix.sum(axis=0)).ravel()
    if max_count is None:
        max_count = len(texts) - min_count
    keep = np.flatnonzero((counts >= min_count) & (counts <= max_count))
    return matrix[:, keep].tocsr(), keep.astype(np.int64)


def score_causal_purity_hashes(
    X: sparse.csr_matrix,
    treatment: np.ndarray,
    outcome: np.ndarray,
    alpha: float = 1.0,
    beta: float = 2.0,
    gamma_control: float = 1.0,
    min_count: int = 20,
    min_arm_count: int = 5,
) -> Tuple[np.ndarray, dict]:
    """Score hash columns by pure treatment-effect interaction strength."""
    X = X.tocsr().astype(np.float32)
    treatment = np.asarray(treatment, dtype=int)
    outcome = np.asarray(outcome, dtype=float)
    n = X.shape[0]

    present_count = np.asarray(X.sum(axis=0)).ravel()
    freq = present_count / max(n, 1)
    treated_mask = treatment == 1
    control_mask = ~treated_mask

    present_treated = np.asarray(X[treated_mask].sum(axis=0)).ravel()
    present_control = np.asarray(X[control_mask].sum(axis=0)).ravel()
    absent_treated = int(treated_mask.sum()) - present_treated
    absent_control = int(control_mask.sum()) - present_control

    y_present_treated = np.asarray(X[treated_mask].T @ outcome[treated_mask]).ravel()
    y_present_control = np.asarray(X[control_mask].T @ outcome[control_mask]).ravel()
    y_absent_treated = outcome[treated_mask].sum() - y_present_treated
    y_absent_control = outcome[control_mask].sum() - y_present_control

    mean_treated_present = y_present_treated / np.maximum(present_treated, 1)
    mean_control_present = y_present_control / np.maximum(present_control, 1)
    mean_treated_absent = y_absent_treated / np.maximum(absent_treated, 1)
    mean_control_absent = y_absent_control / np.maximum(absent_control, 1)

    te_diff = (
        mean_treated_present
        - mean_control_present
        - (mean_treated_absent - mean_control_absent)
    )
    control_diff = mean_control_present - mean_control_absent
    pure_interaction = np.maximum(0.0, te_diff - gamma_control * np.abs(control_diff))
    min_arm = np.minimum.reduce([
        present_treated,
        present_control,
        absent_treated,
        absent_control,
    ])
    ok = (
        (present_count >= min_count)
        & (present_count <= n - min_count)
        & (min_arm >= min_arm_count)
        & (te_diff > 0)
    )
    score = pure_interaction * (np.maximum(min_arm, 1) ** alpha) * ((1.0 - freq) ** beta)
    score[~ok] = -np.inf
    details = {
        "te_diff": te_diff,
        "control_diff": control_diff,
        "counts": present_count,
        "min_arm": min_arm,
    }
    return score, details


def select_causal_purity_hashes(
    X: sparse.csr_matrix,
    original_hashes: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    top_k: int = 1,
    alpha: float = 1.0,
    beta: float = 2.0,
    gamma_control: float = 1.0,
    min_count: int = 20,
    min_arm_count: int = 5,
) -> CausalPurityHashSelection:
    """Select top causal-purity token-hash columns from a train fold."""
    score, details = score_causal_purity_hashes(
        X,
        treatment=treatment,
        outcome=outcome,
        alpha=alpha,
        beta=beta,
        gamma_control=gamma_control,
        min_count=min_count,
        min_arm_count=min_arm_count,
    )
    finite = np.flatnonzero(np.isfinite(score))
    if finite.size == 0:
        selected = np.asarray([], dtype=np.int64)
    else:
        order = finite[np.argsort(-score[finite])]
        selected = order[:top_k]
    return CausalPurityHashSelection(
        columns=selected.astype(np.int64),
        original_hashes=np.asarray(original_hashes)[selected].astype(np.int64),
        scores=score[selected].astype(float),
        te_diff=details["te_diff"][selected].astype(float),
        control_diff=details["control_diff"][selected].astype(float),
        counts=details["counts"][selected].astype(float),
    )
