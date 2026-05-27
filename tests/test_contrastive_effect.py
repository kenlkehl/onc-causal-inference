import numpy as np
import torch

from oci.models.contrastive_causal_text_forest import (
    MatchedContrastiveEffectHead,
    grad_reverse,
)
from oci.training.contrastive_effect import (
    PropensityBinBalancedBatchSampler,
    make_propensity_bins,
)


def test_gradient_reversal_scales_and_flips_gradient():
    x = torch.tensor([1.0], requires_grad=True)
    y = (grad_reverse(x, 0.5) ** 2).sum()
    y.backward()

    assert torch.allclose(x.grad, torch.tensor([-1.0]))


def test_propensity_bins_keep_bins_with_both_arms():
    propensity = np.array([0.1, 0.12, 0.2, 0.21, 0.7, 0.72, 0.8, 0.82])
    treatment = np.array([0, 1, 0, 1, 0, 1, 0, 1])

    bin_ids = make_propensity_bins(
        propensity,
        treatment,
        n_bins=2,
        overlap_min=0.05,
        overlap_max=0.95,
        min_arm_per_bin=1,
    )

    assert set(np.unique(bin_ids)) == {0, 1}
    for b in (0, 1):
        idx = np.flatnonzero(bin_ids == b)
        assert np.any(treatment[idx] == 0)
        assert np.any(treatment[idx] == 1)


def test_balanced_batch_sampler_samples_within_one_propensity_bin():
    treatment = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    bin_ids = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    sampler = PropensityBinBalancedBatchSampler(
        treatment=treatment,
        bin_ids=bin_ids,
        batch_size=4,
        min_arm_per_bin=1,
        seed=1,
        batches_per_epoch=6,
    )

    for batch in sampler:
        batch = np.asarray(batch)
        assert len(batch) == 4
        assert len(np.unique(bin_ids[batch])) == 1
        assert np.sum(treatment[batch] == 1) == 2
        assert np.sum(treatment[batch] == 0) == 2


def test_matched_contrastive_effect_head_shapes():
    head = MatchedContrastiveEffectHead(
        input_dim=12,
        bottleneck_dim=3,
        hidden_dim=8,
        dropout=0.0,
    )

    outputs = head(torch.randn(5, 12))

    assert outputs['z'].shape == (5, 3)
    assert outputs['mu0_logit'].shape == (5, 1)
    assert outputs['mu1_logit'].shape == (5, 1)
    assert outputs['adv_logit'].shape == (5, 1)
