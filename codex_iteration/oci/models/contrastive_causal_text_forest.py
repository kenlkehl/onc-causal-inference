"""CausalTextForest variant with matched contrastive effect representation."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from itertools import chain
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ContrastiveEffectConfig
from ..data.cached_hidden_state_dataset import prepare_cached_batch
from .causal_text_forest import CausalTextForest


class GradientReversalFunction(torch.autograd.Function):
    """Identity in the forward pass, sign-reversed gradient in backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    """Apply gradient reversal to x."""
    return GradientReversalFunction.apply(x, lambd)


class MatchedContrastiveEffectHead(nn.Module):
    """Bottlenecked X head with potential-outcome and treatment-adversary heads."""

    def __init__(
        self,
        input_dim: int,
        bottleneck_dim: int,
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
        )
        self.mu0_head = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.mu1_head = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.treatment_adversary = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, effect_input: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.bottleneck(effect_input)
        return {
            'z': z,
            'mu0_logit': self.mu0_head(z),
            'mu1_logit': self.mu1_head(z),
            'adv_logit': self.treatment_adversary(z),
        }


class ContrastiveCausalTextForest(CausalTextForest):
    """Causal forest model with matched contrastive training for X features."""

    def __init__(
        self,
        *args,
        contrastive_effect_config: Optional[Any] = None,
        **kwargs,
    ):
        kwargs['cf_use_rlearner_representation'] = True
        super().__init__(*args, **kwargs)

        if contrastive_effect_config is None:
            contrastive_effect_config = ContrastiveEffectConfig(enabled=True)
        elif isinstance(contrastive_effect_config, dict):
            contrastive_effect_config = ContrastiveEffectConfig(**contrastive_effect_config)
        self.contrastive_effect_config = contrastive_effect_config
        self.use_contrastive_effect_representation = True

        cfg = self.contrastive_effect_config
        self.contrastive_effect_head = MatchedContrastiveEffectHead(
            input_dim=self._effect_input_dim,
            bottleneck_dim=cfg.bottleneck_dim,
            hidden_dim=cfg.hidden_dim,
            dropout=self.config.get('dropout', 0.2),
        )

        cfg_dict = asdict(cfg) if is_dataclass(cfg) else dict(cfg)
        self.config['contrastive_effect_config'] = cfg_dict
        self.config['cf_use_rlearner_representation'] = True
        self.to(self._device)

    def effect_parameters(self):
        """Return parameters used by the matched contrastive effect stage."""
        modules = [self.effect_feature_extractor or self.feature_extractor, self.contrastive_effect_head]
        if self.explicit_effect_featurizer is not None:
            modules.append(self.explicit_effect_featurizer)
        return chain.from_iterable(module.parameters() for module in modules)

    def _contrastive_effect_forward(
        self,
        extractor_input,
        explicit_feature_values: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Return bottleneck, potential-outcome heads, tau, and adversary logit."""
        extractor = self.effect_feature_extractor or self.feature_extractor
        effect_text_features = extractor(extractor_input)
        effect_input = self._append_role_features(
            effect_text_features,
            explicit_feature_values,
            "effect_modifier",
        )
        outputs = self.contrastive_effect_head(effect_input)
        if self.outcome_type == "continuous":
            tau = outputs['mu1_logit'] - outputs['mu0_logit']
        else:
            tau = torch.sigmoid(outputs['mu1_logit']) - torch.sigmoid(outputs['mu0_logit'])
        outputs['tau'] = tau
        return outputs

    def train_effect_contrastive_step(
        self,
        batch: Dict[str, Any],
        e_hat: torch.Tensor,
        m_hat: torch.Tensor,
        bin_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Train X from factual potential-outcome and bin-level residual contrasts."""
        cfg = self.contrastive_effect_config
        texts = batch['texts']
        treatments = batch['treatment'].to(self._device).float()
        outcomes = batch['outcome'].to(self._device).float()
        explicit_feature_values = batch.get('explicit_feature_values', None)
        extractor_input = self._get_extractor_input(batch, texts)

        outputs = self._contrastive_effect_forward(extractor_input, explicit_feature_values)
        mu0 = outputs['mu0_logit'].squeeze(-1)
        mu1 = outputs['mu1_logit'].squeeze(-1)
        tau = outputs['tau'].squeeze(-1)
        z = outputs['z']

        e_hat = e_hat.to(self._device).float().clamp(0.01, 0.99)
        m_hat = m_hat.to(self._device).float()
        bin_ids = bin_ids.to(self._device).long()

        factual_logit = torch.where(treatments > 0.5, mu1, mu0)
        overlap_weight = (e_hat * (1.0 - e_hat)).detach().clamp_min(1e-4)
        overlap_weight = overlap_weight / overlap_weight.mean().clamp_min(1e-6)
        if self.outcome_type == "continuous":
            factual_loss = (overlap_weight * (factual_logit - outcomes) ** 2).mean()
        else:
            factual_loss = F.binary_cross_entropy_with_logits(
                factual_logit,
                outcomes,
                weight=overlap_weight,
            )

        y_residual = outcomes - m_hat
        t_residual = treatments - e_hat
        contrast_terms = []
        for bin_id in torch.unique(bin_ids):
            if bin_id.item() < 0:
                continue
            mask = bin_ids == bin_id
            if torch.sum(mask & (treatments > 0.5)) < cfg.min_arm_per_bin:
                continue
            if torch.sum(mask & (treatments <= 0.5)) < cfg.min_arm_per_bin:
                continue
            denom = torch.sum(t_residual[mask] ** 2).clamp_min(1e-6)
            target = torch.sum(t_residual[mask] * y_residual[mask]) / denom
            target = target.clamp(-cfg.target_clip, cfg.target_clip).detach()
            pred = tau[mask].mean()
            contrast_terms.append((pred - target) ** 2)

        if contrast_terms:
            contrast_loss = torch.stack(contrast_terms).mean()
        else:
            contrast_loss = torch.zeros((), device=self._device)

        if cfg.lambda_adversary > 0:
            adv_z = grad_reverse(z, cfg.lambda_adversary)
            adv_logit = self.contrastive_effect_head.treatment_adversary(adv_z).squeeze(-1)
            adversary_loss = F.binary_cross_entropy_with_logits(adv_logit, treatments)
        else:
            adversary_loss = torch.zeros((), device=self._device)

        z_l2_loss = torch.mean(z ** 2)
        total_loss = (
            cfg.lambda_factual * factual_loss
            + cfg.lambda_contrast * contrast_loss
            + adversary_loss
            + cfg.lambda_z_l2 * z_l2_loss
        )

        return {
            'loss': total_loss,
            'factual_loss': factual_loss.detach(),
            'contrast_loss': contrast_loss.detach(),
            'adversary_loss': adversary_loss.detach(),
            'z_l2_loss': z_l2_loss.detach(),
            'r_loss': contrast_loss.detach(),
            'tau': tau.detach(),
            'x_hidden': z.detach(),
        }

    def train_effect_pairwise_contrastive_step(
        self,
        batch: Dict[str, Any],
        e_hat: torch.Tensor,
        m_hat: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Train X from true treated/control propensity-matched pairs.

        The dataloader sampler orders rows as ``[treated, control]`` pairs.
        Pair targets use residualized observed outcomes, relying only on
        cross-fitted nuisance predictions and observed treatment/outcome.
        """
        cfg = self.contrastive_effect_config
        texts = batch['texts']
        treatments = batch['treatment'].to(self._device).float()
        outcomes = batch['outcome'].to(self._device).float()
        explicit_feature_values = batch.get('explicit_feature_values', None)
        extractor_input = self._get_extractor_input(batch, texts)

        outputs = self._contrastive_effect_forward(extractor_input, explicit_feature_values)
        mu0 = outputs['mu0_logit'].squeeze(-1)
        mu1 = outputs['mu1_logit'].squeeze(-1)
        tau = outputs['tau'].squeeze(-1)
        z = outputs['z']

        e_hat = e_hat.to(self._device).float().clamp(0.01, 0.99)
        m_hat = m_hat.to(self._device).float()

        factual_logit = torch.where(treatments > 0.5, mu1, mu0)
        overlap_weight = (e_hat * (1.0 - e_hat)).detach().clamp_min(1e-4)
        overlap_weight = overlap_weight / overlap_weight.mean().clamp_min(1e-6)
        if self.outcome_type == "continuous":
            factual_loss = (overlap_weight * (factual_logit - outcomes) ** 2).mean()
        else:
            factual_loss = F.binary_cross_entropy_with_logits(
                factual_logit,
                outcomes,
                weight=overlap_weight,
            )

        n_pairs = outcomes.numel() // 2
        if n_pairs < 1:
            pair_loss = torch.zeros((), device=self._device)
            pair_pull_loss = torch.zeros((), device=self._device)
        else:
            pair_slice = slice(0, n_pairs * 2)
            pair_outcomes = outcomes[pair_slice].view(n_pairs, 2)
            pair_m = m_hat[pair_slice].view(n_pairs, 2)
            pair_e = e_hat[pair_slice].view(n_pairs, 2)
            pair_tau = tau[pair_slice].view(n_pairs, 2)
            pair_z = z[pair_slice].view(n_pairs, 2, -1)

            treated_first = treatments[pair_slice].view(n_pairs, 2)[:, 0] > 0.5
            control_second = treatments[pair_slice].view(n_pairs, 2)[:, 1] <= 0.5
            valid = treated_first & control_second
            if torch.any(valid):
                residual_diff = (
                    (pair_outcomes[:, 0] - pair_m[:, 0])
                    - (pair_outcomes[:, 1] - pair_m[:, 1])
                )
                target = residual_diff.clamp(-cfg.target_clip, cfg.target_clip).detach()
                pred = (1.0 - pair_e[:, 0]) * pair_tau[:, 0] + pair_e[:, 1] * pair_tau[:, 1]
                pair_loss = ((pred[valid] - target[valid]) ** 2).mean()
                pair_pull_loss = F.mse_loss(pair_z[valid, 0], pair_z[valid, 1])
            else:
                pair_loss = torch.zeros((), device=self._device)
                pair_pull_loss = torch.zeros((), device=self._device)

        if cfg.lambda_adversary > 0:
            adv_z = grad_reverse(z, cfg.lambda_adversary)
            adv_logit = self.contrastive_effect_head.treatment_adversary(adv_z).squeeze(-1)
            adversary_loss = F.binary_cross_entropy_with_logits(adv_logit, treatments)
        else:
            adversary_loss = torch.zeros((), device=self._device)

        z_l2_loss = torch.mean(z ** 2)
        total_loss = (
            cfg.lambda_factual * factual_loss
            + cfg.lambda_contrast * pair_loss
            + adversary_loss
            + cfg.lambda_pair_pull * pair_pull_loss
            + cfg.lambda_z_l2 * z_l2_loss
        )

        return {
            'loss': total_loss,
            'factual_loss': factual_loss.detach(),
            'contrast_loss': pair_loss.detach(),
            'adversary_loss': adversary_loss.detach(),
            'pair_pull_loss': pair_pull_loss.detach(),
            'z_l2_loss': z_l2_loss.detach(),
            'r_loss': pair_loss.detach(),
            'tau': tau.detach(),
            'x_hidden': z.detach(),
        }

    def _forest_x_from_contrastive_outputs(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        mode = self.contrastive_effect_config.forest_x_mode
        if mode == "bottleneck":
            return outputs['z']
        if mode == "tau":
            return outputs['tau']
        return torch.cat([outputs['z'], outputs['tau']], dim=1)

    def extract_forest_features(
        self,
        texts_or_loader,
        batch_size: int = 32,
        explicit_feature_values: Optional[List[Dict[str, Any]]] = None,
        explicit_confounder_values: Optional[List[Dict[str, Any]]] = None,
        gpu_store=None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]:
        """Extract forest X from contrastive z/tau and W from nuisance activations."""
        from torch.utils.data import DataLoader

        if explicit_feature_values is None:
            explicit_feature_values = explicit_confounder_values

        self.eval()
        all_x = []
        all_w = []
        all_propensity = []
        all_outcome = []
        all_feature_values = []

        is_batch_iterable = isinstance(texts_or_loader, DataLoader) or (
            hasattr(texts_or_loader, '__iter__') and not isinstance(texts_or_loader, (list, str))
        )

        def process_batch(extractor_input, batch_feature_values):
            text_features = self.feature_extractor(extractor_input)
            w_hidden, prop_logit, outcome_logit = self._nuisance_forward(
                text_features,
                batch_feature_values,
            )
            effect_outputs = self._contrastive_effect_forward(
                extractor_input,
                batch_feature_values,
            )
            x_matrix = self._forest_x_from_contrastive_outputs(effect_outputs)
            return x_matrix, w_hidden, prop_logit, outcome_logit

        with torch.no_grad():
            if is_batch_iterable:
                for batch in texts_or_loader:
                    prepare_cached_batch(batch, self._device, gpu_store=gpu_store)
                    texts = batch['texts']
                    extractor_input = self._get_extractor_input(batch, texts)
                    batch_feature_values = batch.get('explicit_feature_values', None)
                    x_matrix, w_matrix, prop_logit, outcome_logit = process_batch(
                        extractor_input,
                        batch_feature_values,
                    )
                    all_x.append(x_matrix.cpu().numpy())
                    all_w.append(w_matrix.cpu().numpy())
                    all_propensity.append(torch.sigmoid(prop_logit).cpu().numpy())
                    all_outcome.append(self._outcome_activation(outcome_logit).cpu().numpy())
                    if batch_feature_values is not None:
                        all_feature_values.extend(batch_feature_values)
            else:
                texts = texts_or_loader
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    batch_feature_values = None
                    if explicit_feature_values is not None:
                        batch_feature_values = explicit_feature_values[i:i + batch_size]
                    x_matrix, w_matrix, prop_logit, outcome_logit = process_batch(
                        batch_texts,
                        batch_feature_values,
                    )
                    all_x.append(x_matrix.cpu().numpy())
                    all_w.append(w_matrix.cpu().numpy())
                    all_propensity.append(torch.sigmoid(prop_logit).cpu().numpy())
                    all_outcome.append(self._outcome_activation(outcome_logit).cpu().numpy())

        x_features = np.vstack(all_x)
        w_features = np.vstack(all_w) if all_w else None

        feature_values_for_raw = all_feature_values if all_feature_values else explicit_feature_values
        if feature_values_for_raw is not None and self.explicit_feature_specs:
            raw_w = self._get_raw_explicit_features(feature_values_for_raw, role="confounder")
            raw_x = self._get_raw_explicit_features(feature_values_for_raw, role="effect_modifier")
            x_features = self._hstack_optional(x_features, raw_x)
            w_features = self._hstack_optional(w_features, raw_w)

        return (
            x_features,
            w_features,
            np.vstack(all_propensity).flatten(),
            np.vstack(all_outcome).flatten(),
        )
