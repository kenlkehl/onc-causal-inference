# cdt/models/uplift.py
"""Uplift Modeling architecture (Base + ITE) mirroring DragonNet."""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


class UpliftNet(nn.Module):
    """
    Uplift Modeling Network (Base + ITE parametrization).

    Architecture:
    - Shared Representation Phi(X)
    - Propensity Head: Phi(X) -> pi(X)
    - Baseline Head: Phi(X) -> mu_0(X)  (Prognostic score)
    - Effect Head: Phi(X) -> tau(X)     (Treatment effect)

    This parametrization forces the model to explicitly learn the treatment heterogeneity
    separate from the baseline risk, preventing the prognostic signal from drowning out
    the causal signal.

    Theoretical Justification:
    Like DragonNet, this architecture uses a shared representation regularized by the
    propensity score (confounding control). By explicitly parametrizing Tau(X),
    we reduce regularization bias where the model collapses ITE to the ATE
    to satisfy the dominant prognostic loss.
    """

    def __init__(self, input_dim, representation_dim=200, hidden_outcome_dim=100, dropout=0.2):
        super().__init__()
        self.dropout_rate = dropout

        # Shared representation layers (2 layers, matching DragonNet)
        self.representation_fc1 = nn.Linear(input_dim, representation_dim)
        self.representation_fc2 = nn.Linear(representation_dim, representation_dim)

        # Dropout for representation layers
        self.rep_dropout = nn.Dropout(dropout)

        # Propensity head (single linear layer, matching DragonNet)
        self.propensity_fc = nn.Linear(representation_dim, 1)

        # Baseline Outcome Head (Estimates Y|X, T=0) - 3 layers matching DragonNet outcome heads
        self.baseline_fc1 = nn.Linear(representation_dim, hidden_outcome_dim)
        self.baseline_fc2 = nn.Linear(hidden_outcome_dim, hidden_outcome_dim)
        self.baseline_fc3 = nn.Linear(hidden_outcome_dim, 1)

        # Treatment Effect Head (Estimates Tau(X)) - 3 layers matching DragonNet outcome heads
        # Note: We use Tanh or Linear final activation depending on needs,
        # but here we keep it linear (logits) to match DragonNet's flexibility.
        self.effect_fc1 = nn.Linear(representation_dim, hidden_outcome_dim)
        self.effect_fc2 = nn.Linear(hidden_outcome_dim, hidden_outcome_dim)
        self.effect_fc3 = nn.Linear(hidden_outcome_dim, 1)

        # Dropout for outcome/effect heads
        self.outcome_dropout = nn.Dropout(dropout)


    def forward(self, confounder_features):
        """
        Args:
            confounder_features: Output from FeatureExtractor
                Shape: (batch, num_confounders * features_per_confounder)

        Returns:
            y0_logit, tau_logit, t_logit, final_common_layer
        """
        # Shared Representation (2 layers)
        h = F.relu(self.representation_fc1(confounder_features))
        h = self.rep_dropout(h)
        final_common_layer = F.elu(self.representation_fc2(h))
        final_common_layer = self.rep_dropout(final_common_layer)

        # Propensity (single layer)
        t_logit = self.propensity_fc(final_common_layer)

        # Baseline Outcome (Y0) - 3 layers
        y0 = F.relu(self.baseline_fc1(final_common_layer))
        y0 = self.outcome_dropout(y0)
        y0 = F.elu(self.baseline_fc2(y0))
        y0 = self.outcome_dropout(y0)
        y0_logit = self.baseline_fc3(y0)

        # Treatment Effect (Tau) - 3 layers
        tau = F.relu(self.effect_fc1(final_common_layer))
        tau = self.outcome_dropout(tau)
        tau = F.elu(self.effect_fc2(tau))
        tau = self.outcome_dropout(tau)
        tau_logit = self.effect_fc3(tau)

        return y0_logit, tau_logit, t_logit, final_common_layer

    def get_representation(self, features):
        """Compute shared representation from input features."""
        h = F.relu(self.representation_fc1(features))
        h = self.rep_dropout(h)
        final_common_layer = F.elu(self.representation_fc2(h))
        final_common_layer = self.rep_dropout(final_common_layer)
        return final_common_layer

    def propensity_from_representation(self, phi):
        """Compute propensity logit from shared representation."""
        return self.propensity_fc(phi)

    def load_pretrained_representation(self, pretrained_state_dict):
        """
        Load pretrained representation layers (fc1-fc2) if dimensions match.
        """
        # (Logic identical to DragonNet's implementation)
        if 'dragonnet' in pretrained_state_dict:
            state_dict = pretrained_state_dict['dragonnet']
        elif 'dragonnet_representation' in pretrained_state_dict:
            state_dict = pretrained_state_dict['dragonnet_representation']
        elif 'representation_fc1.weight' in pretrained_state_dict:
            state_dict = pretrained_state_dict
        else:
            return False

        # Check dimensions
        if state_dict['representation_fc1.weight'].shape != self.representation_fc1.weight.shape:
            return False

        try:
            rep_state_dict = {}
            for key in ['representation_fc1', 'representation_fc2']:
                for param_name in ['weight', 'bias']:
                    full_key = f'{key}.{param_name}'
                    if full_key in state_dict:
                        rep_state_dict[full_key] = state_dict[full_key]

            self.load_state_dict(rep_state_dict, strict=False)
            return True
        except Exception as e:
            logger.warning(f"Failed to load pretrained weights: {e}")
            return False
