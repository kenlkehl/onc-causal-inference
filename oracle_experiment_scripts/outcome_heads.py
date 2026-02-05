# oracle_experiment_scripts/outcome_heads.py
"""Lightweight outcome heads for oracle mode - takes confounder_features directly, no representation layers.

Note: This file was moved from cdt/models/outcome_heads.py since it is
only used by oracle experiment scripts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OutcomeHeadsOnly(nn.Module):
    """
    Outcome heads that take confounder_features directly.

    In oracle mode, the generator's FeatureExtractor output is passed directly
    to these heads, bypassing text processing and DragonNet's representation layers.
    
    Args:
        confounder_dim: Dimension of input = num_confounders × features_per_confounder
        hidden_outcome_dim: Hidden layer dimension for outcome heads
    """

    def __init__(self, confounder_dim, hidden_outcome_dim=100):
        super().__init__()

        # Propensity head (single layer like old version)
        self.propensity_fc1 = nn.Linear(confounder_dim, 1)

        # Y0 outcome head (2 hidden layers like old version)
        self.outcome0_fc1 = nn.Linear(confounder_dim, hidden_outcome_dim)
        self.outcome0_fc2 = nn.Linear(hidden_outcome_dim, hidden_outcome_dim)
        self.outcome0_fc3 = nn.Linear(hidden_outcome_dim, 1)

        # Y1 outcome head (2 hidden layers like old version)
        self.outcome1_fc1 = nn.Linear(confounder_dim, hidden_outcome_dim)
        self.outcome1_fc2 = nn.Linear(hidden_outcome_dim, hidden_outcome_dim)
        self.outcome1_fc3 = nn.Linear(hidden_outcome_dim, 1)

    def forward(self, confounder_features):
        """
        Args:
            confounder_features: Pre-extracted features from generator's FeatureExtractor
                Shape: (batch, num_confounders * features_per_confounder)

        Returns:
            y0_logit, y1_logit, t_logit, confounder_features (pass-through for compatibility)
        """
        # Propensity (single layer)
        t_logit = self.propensity_fc1(confounder_features)

        # Y0 outcome (2 hidden layers: ReLU -> ELU -> Linear)
        y0 = F.relu(self.outcome0_fc1(confounder_features))
        y0 = F.elu(self.outcome0_fc2(y0))
        y0_logit = self.outcome0_fc3(y0)

        # Y1 outcome (2 hidden layers: ReLU -> ELU -> Linear)
        y1 = F.relu(self.outcome1_fc1(confounder_features))
        y1 = F.elu(self.outcome1_fc2(y1))
        y1_logit = self.outcome1_fc3(y1)

        return y0_logit, y1_logit, t_logit, confounder_features


class UpliftHeadsOnly(nn.Module):
    """
    Uplift parametrization (y0, tau) that takes confounder_features directly.

    In oracle mode with uplift modeling, the generator's FeatureExtractor output
    is passed directly to these heads.
    
    Args:
        confounder_dim: Dimension of input = num_confounders × features_per_confounder
        hidden_outcome_dim: Hidden layer dimension for outcome heads
    """

    def __init__(self, confounder_dim, hidden_outcome_dim=100):
        super().__init__()

        # Propensity head (single layer)
        self.propensity_fc1 = nn.Linear(confounder_dim, 1)

        # Baseline Y0 head (2 hidden layers)
        self.baseline_fc1 = nn.Linear(confounder_dim, hidden_outcome_dim)
        self.baseline_fc2 = nn.Linear(hidden_outcome_dim, hidden_outcome_dim)
        self.baseline_fc3 = nn.Linear(hidden_outcome_dim, 1)

        # Treatment effect Tau head (2 hidden layers)
        self.effect_fc1 = nn.Linear(confounder_dim, hidden_outcome_dim)
        self.effect_fc2 = nn.Linear(hidden_outcome_dim, hidden_outcome_dim)
        self.effect_fc3 = nn.Linear(hidden_outcome_dim, 1)

    def forward(self, confounder_features):
        """
        Args:
            confounder_features: Pre-extracted features from generator's FeatureExtractor
                Shape: (batch, num_confounders * features_per_confounder)

        Returns:
            y0_logit, tau_logit, t_logit, confounder_features (pass-through for compatibility)
        """
        # Propensity
        t_logit = self.propensity_fc1(confounder_features)

        # Baseline Y0
        y0 = F.relu(self.baseline_fc1(confounder_features))
        y0 = F.elu(self.baseline_fc2(y0))
        y0_logit = self.baseline_fc3(y0)

        # Treatment effect Tau
        tau = F.relu(self.effect_fc1(confounder_features))
        tau = F.elu(self.effect_fc2(tau))
        tau_logit = self.effect_fc3(tau)

        return y0_logit, tau_logit, t_logit, confounder_features
