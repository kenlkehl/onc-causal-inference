# oci/models/rlearner.py
"""R-Learner network for direct treatment effect optimization."""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


class RLearnerNet(nn.Module):
    """
    R-Learner Network with three heads for direct treatment effect optimization.

    Architecture:
    - Shared representation layers (same as DragonNet)
    - Propensity head: e(X) = P(T=1|X)
    - Marginal outcome head: m(X) = E[Y|X]
    - Treatment effect head: τ(X) = E[Y(1)-Y(0)|X]

    The τ head is trained with R-learner loss that directly optimizes
    treatment effect estimation by minimizing:
        L_R = E[(Y - m(X) - τ(X)(T - e(X)))^2]

    Key advantages over DragonNet:
    - Direct gradient signal to τ(X) from treatment effect loss
    - Nuisance functions (e, m) are detached in R-loss, preventing
      interference with effect estimation
    - τ(X) is unbounded (can be negative) - represents true effect

    References:
        Nie & Wager (2021). Quasi-oracle estimation of heterogeneous
        treatment effects. Biometrika.
    """

    def __init__(
        self,
        input_dim: int,
        representation_dim: int = 200,
        hidden_outcome_dim: int = 100,
        dropout: float = 0.2
    ):
        """
        Initialize R-Learner network.

        Args:
            input_dim: Dimension of input features from feature extractor
            representation_dim: Dimension of shared representation
            hidden_outcome_dim: Hidden dimension for outcome/effect heads
            dropout: Dropout rate
        """
        super().__init__()
        self.dropout_rate = dropout

        # Shared representation layers (2 layers like simple DragonNet)
        self.representation_fc1 = nn.Linear(input_dim, representation_dim)
        self.representation_fc2 = nn.Linear(representation_dim, representation_dim)
        self.rep_dropout = nn.Dropout(dropout)

        # Propensity head: P(T=1|X) - single linear layer
        self.propensity_fc = nn.Linear(representation_dim, 1)

        # Marginal outcome head: E[Y|X] - 2 hidden layers
        self.outcome_fc1 = nn.Linear(representation_dim, hidden_outcome_dim)
        self.outcome_fc2 = nn.Linear(hidden_outcome_dim, hidden_outcome_dim)
        self.outcome_fc3 = nn.Linear(hidden_outcome_dim, 1)

        # Treatment effect head: τ(X) = E[Y(1)-Y(0)|X] - 2 hidden layers
        # Note: τ is unbounded (no final activation), can be positive or negative
        self.effect_fc1 = nn.Linear(representation_dim, hidden_outcome_dim)
        self.effect_fc2 = nn.Linear(hidden_outcome_dim, hidden_outcome_dim)
        self.effect_fc3 = nn.Linear(hidden_outcome_dim, 1)

        # Dropout for outcome/effect heads
        self.outcome_dropout = nn.Dropout(dropout)

    def forward(self, features: torch.Tensor):
        """
        Forward pass through R-Learner network.

        Args:
            features: Output from feature extractor, shape (batch, input_dim)

        Returns:
            m_logit: Marginal outcome logit E[Y|X], shape (batch, 1)
            tau: Treatment effect τ(X), shape (batch, 1) - unbounded
            t_logit: Propensity logit, shape (batch, 1)
            phi: Shared representation, shape (batch, representation_dim)
        """
        # Shared representation
        h = F.relu(self.representation_fc1(features))
        h = self.rep_dropout(h)
        phi = F.elu(self.representation_fc2(h))
        phi = self.rep_dropout(phi)

        # Propensity head
        t_logit = self.propensity_fc(phi)

        # Marginal outcome head
        m = F.relu(self.outcome_fc1(phi))
        m = self.outcome_dropout(m)
        m = F.elu(self.outcome_fc2(m))
        m = self.outcome_dropout(m)
        m_logit = self.outcome_fc3(m)

        # Treatment effect head (no final activation - τ can be negative)
        tau = F.relu(self.effect_fc1(phi))
        tau = self.outcome_dropout(tau)
        tau = F.elu(self.effect_fc2(tau))
        tau = self.outcome_dropout(tau)
        tau_out = self.effect_fc3(tau)

        return m_logit, tau_out, t_logit, phi

    def get_representation(self, features):
        """Compute shared representation from input features."""
        h = F.relu(self.representation_fc1(features))
        h = self.rep_dropout(h)
        phi = F.elu(self.representation_fc2(h))
        phi = self.rep_dropout(phi)
        return phi

    def propensity_from_representation(self, phi):
        """Compute propensity logit from shared representation."""
        return self.propensity_fc(phi)
