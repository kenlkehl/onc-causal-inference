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

        # Shared representation layers (identical to DragonNet)
        self.representation_fc1 = nn.Linear(input_dim, representation_dim)
        self.representation_fc2 = nn.Linear(representation_dim, representation_dim)
        self.representation_fc3 = nn.Linear(representation_dim, representation_dim)
        self.representation_fc4 = nn.Linear(representation_dim, representation_dim)
        self.representation_fc5 = nn.Linear(representation_dim, representation_dim)
        self.representation_fc6 = nn.Linear(representation_dim, representation_dim)

        # Dropout for representation layers
        self.rep_dropout = nn.Dropout(dropout)

        # Propensity head (identical to DragonNet)
        self.propensity_fc1 = nn.Linear(representation_dim, representation_dim)
        self.propensity_fc2 = nn.Linear(representation_dim, representation_dim)
        self.propensity_fc3 = nn.Linear(representation_dim, representation_dim)
        self.propensity_fc4 = nn.Linear(representation_dim, 1)

        # Baseline Outcome Head (Estimates Y|X, T=0)
        self.baseline_fc1 = nn.Linear(representation_dim, hidden_outcome_dim)
        self.baseline_fc2 = nn.Linear(hidden_outcome_dim, hidden_outcome_dim)
        self.baseline_fc2a = nn.Linear(hidden_outcome_dim, hidden_outcome_dim)
        self.baseline_fc2b = nn.Linear(hidden_outcome_dim, hidden_outcome_dim)
        self.baseline_fc2c = nn.Linear(hidden_outcome_dim, hidden_outcome_dim)
        self.baseline_fc3 = nn.Linear(hidden_outcome_dim, 1)

        # Treatment Effect Head (Estimates Tau(X))
        # Note: We use Tanh or Linear final activation depending on needs,
        # but here we keep it linear (logits) to match DragonNet's flexibility.
        self.effect_fc1 = nn.Linear(representation_dim, hidden_outcome_dim)
        self.effect_fc2 = nn.Linear(hidden_outcome_dim, hidden_outcome_dim)
        self.effect_fc2a = nn.Linear(hidden_outcome_dim, hidden_outcome_dim)
        self.effect_fc2b = nn.Linear(hidden_outcome_dim, hidden_outcome_dim)
        self.effect_fc2c = nn.Linear(hidden_outcome_dim, hidden_outcome_dim)
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
        # Shared Representation
        h = F.relu(self.representation_fc1(confounder_features))
        h = self.rep_dropout(h)
        h = F.relu(self.representation_fc2(h))
        h = self.rep_dropout(h)
        h = F.relu(self.representation_fc3(h))
        h = self.rep_dropout(h)
        h = F.relu(self.representation_fc4(h))
        h = self.rep_dropout(h)
        h = F.relu(self.representation_fc5(h))
        h = self.rep_dropout(h)
        final_common_layer = F.elu(self.representation_fc6(h))
        final_common_layer = self.rep_dropout(final_common_layer)

        # Propensity
        t = F.relu(self.propensity_fc1(final_common_layer))
        t = F.relu(self.propensity_fc2(t))
        t = F.relu(self.propensity_fc3(t))
        t_logit = self.propensity_fc4(t)

        # Baseline Outcome (Y0)
        y0 = F.relu(self.baseline_fc1(final_common_layer))
        y0 = self.outcome_dropout(y0)
        y0 = F.relu(self.baseline_fc2(y0))
        y0 = self.outcome_dropout(y0)
        y0 = F.relu(self.baseline_fc2a(y0))
        y0 = self.outcome_dropout(y0)
        y0 = F.relu(self.baseline_fc2b(y0))
        y0 = self.outcome_dropout(y0)
        y0 = F.elu(self.baseline_fc2c(y0))
        y0_logit = self.baseline_fc3(y0)

        # Treatment Effect (Tau)
        tau = F.relu(self.effect_fc1(final_common_layer))
        tau = self.outcome_dropout(tau)
        tau = F.relu(self.effect_fc2(tau))
        tau = self.outcome_dropout(tau)
        tau = F.relu(self.effect_fc2a(tau))
        tau = self.outcome_dropout(tau)
        tau = F.relu(self.effect_fc2b(tau))
        tau = self.outcome_dropout(tau)
        tau = F.elu(self.effect_fc2c(tau))
        tau_logit = self.effect_fc3(tau)

        return y0_logit, tau_logit, t_logit, final_common_layer

    def get_representation(self, features):
        """Compute shared representation from input features."""
        h = F.relu(self.representation_fc1(features))
        h = self.rep_dropout(h)
        h = F.relu(self.representation_fc2(h))
        h = self.rep_dropout(h)
        h = F.relu(self.representation_fc3(h))
        h = self.rep_dropout(h)
        h = F.relu(self.representation_fc4(h))
        h = self.rep_dropout(h)
        h = F.relu(self.representation_fc5(h))
        h = self.rep_dropout(h)
        final_common_layer = F.elu(self.representation_fc6(h))
        final_common_layer = self.rep_dropout(final_common_layer)
        return final_common_layer

    def propensity_from_representation(self, phi):
        """Compute propensity logit from shared representation."""
        t = F.relu(self.propensity_fc1(phi))
        t = F.relu(self.propensity_fc2(t))
        t = F.relu(self.propensity_fc3(t))
        return self.propensity_fc4(t)

    def load_pretrained_representation(self, pretrained_state_dict):
        """
        Load pretrained representation layers (fc1-fc6) if dimensions match.
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
            for key in ['representation_fc1', 'representation_fc2', 'representation_fc3',
                       'representation_fc4', 'representation_fc5', 'representation_fc6']:
                for param_name in ['weight', 'bias']:
                    full_key = f'{key}.{param_name}'
                    if full_key in state_dict:
                        rep_state_dict[full_key] = state_dict[full_key]
            
            self.load_state_dict(rep_state_dict, strict=False)
            return True
        except Exception as e:
            logger.warning(f"Failed to load pretrained weights: {e}")
            return False
