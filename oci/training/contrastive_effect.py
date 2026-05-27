"""Utilities for matched contrastive effect-representation training."""

from __future__ import annotations

from typing import Iterator, List, Optional

import numpy as np
from torch.utils.data import Sampler


def make_propensity_bins(
    propensity: np.ndarray,
    treatment: np.ndarray,
    n_bins: int = 10,
    overlap_min: float = 0.05,
    overlap_max: float = 0.95,
    min_arm_per_bin: int = 2,
) -> np.ndarray:
    """Assign samples to propensity-neighborhood bins with both treatment arms.

    Returns an integer array of length n. Samples outside overlap support or in
    under-populated bins receive -1, unless no bin survives, in which case the
    function falls back to one global balanced bin when possible.
    """
    propensity = np.asarray(propensity, dtype=float)
    treatment = np.asarray(treatment, dtype=int)
    if propensity.shape[0] != treatment.shape[0]:
        raise ValueError("propensity and treatment must have the same length")
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    if min_arm_per_bin < 1:
        raise ValueError("min_arm_per_bin must be >= 1")

    finite = np.isfinite(propensity)
    overlap = finite & (propensity >= overlap_min) & (propensity <= overlap_max)
    if not np.any(overlap):
        overlap = finite.copy()

    bin_ids = np.full(propensity.shape[0], -1, dtype=int)
    valid_idx = np.flatnonzero(overlap)
    if valid_idx.size == 0:
        raise ValueError("No finite propensity scores available for contrastive bins")

    if n_bins == 1 or valid_idx.size < n_bins:
        bin_ids[valid_idx] = 0
    else:
        order = valid_idx[np.argsort(propensity[valid_idx], kind="mergesort")]
        ranks = np.arange(order.size)
        assigned = np.minimum((ranks * n_bins) // order.size, n_bins - 1)
        bin_ids[order] = assigned.astype(int)

    for b in np.unique(bin_ids[bin_ids >= 0]):
        idx = np.flatnonzero(bin_ids == b)
        n_treated = int(np.sum(treatment[idx] == 1))
        n_control = int(np.sum(treatment[idx] == 0))
        if n_treated < min_arm_per_bin or n_control < min_arm_per_bin:
            bin_ids[idx] = -1

    if np.any(bin_ids >= 0):
        return bin_ids

    fallback_idx = np.flatnonzero(finite)
    n_treated = int(np.sum(treatment[fallback_idx] == 1))
    n_control = int(np.sum(treatment[fallback_idx] == 0))
    if n_treated >= min_arm_per_bin and n_control >= min_arm_per_bin:
        bin_ids[fallback_idx] = 0
        return bin_ids

    raise ValueError(
        "No propensity bin has enough treated and control samples for "
        f"min_arm_per_bin={min_arm_per_bin}"
    )


class PropensityBinBalancedBatchSampler(Sampler[List[int]]):
    """Sample batches with treated/control balance inside propensity bins."""

    def __init__(
        self,
        treatment: np.ndarray,
        bin_ids: np.ndarray,
        batch_size: int,
        min_arm_per_bin: int = 1,
        seed: int = 42,
        batches_per_epoch: Optional[int] = None,
    ):
        treatment = np.asarray(treatment, dtype=int)
        bin_ids = np.asarray(bin_ids, dtype=int)
        if treatment.shape[0] != bin_ids.shape[0]:
            raise ValueError("treatment and bin_ids must have the same length")
        if batch_size < 2:
            raise ValueError("batch_size must be >= 2")
        if min_arm_per_bin < 1:
            raise ValueError("min_arm_per_bin must be >= 1")

        self.treatment = treatment
        self.bin_ids = bin_ids
        self.batch_size = max(int(batch_size), 2 * int(min_arm_per_bin))
        self.min_arm_per_bin = int(min_arm_per_bin)
        self.seed = int(seed)

        n_treated = max(self.min_arm_per_bin, self.batch_size // 2)
        self.n_treated = min(n_treated, self.batch_size - self.min_arm_per_bin)
        self.n_control = self.batch_size - self.n_treated

        self._bins = []
        for b in np.unique(bin_ids[bin_ids >= 0]):
            idx = np.flatnonzero(bin_ids == b)
            treated_idx = idx[treatment[idx] == 1]
            control_idx = idx[treatment[idx] == 0]
            if treated_idx.size >= self.min_arm_per_bin and control_idx.size >= self.min_arm_per_bin:
                self._bins.append((int(b), treated_idx, control_idx))

        if not self._bins:
            raise ValueError("No valid propensity bins for balanced sampling")

        n_valid = int(np.sum(bin_ids >= 0))
        self.batches_per_epoch = (
            int(batches_per_epoch)
            if batches_per_epoch is not None
            else max(1, int(np.ceil(n_valid / self.batch_size)))
        )
        self._iteration = 0

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.default_rng(self.seed + self._iteration)
        self._iteration += 1
        bin_order = np.arange(len(self._bins))
        for batch_num in range(self.batches_per_epoch):
            if batch_num % len(bin_order) == 0:
                rng.shuffle(bin_order)
            _, treated_idx, control_idx = self._bins[bin_order[batch_num % len(bin_order)]]

            treated = rng.choice(
                treated_idx,
                size=self.n_treated,
                replace=treated_idx.size < self.n_treated,
            )
            control = rng.choice(
                control_idx,
                size=self.n_control,
                replace=control_idx.size < self.n_control,
            )
            batch = np.concatenate([treated, control])
            rng.shuffle(batch)
            yield batch.astype(int).tolist()

    def __len__(self) -> int:
        return self.batches_per_epoch
