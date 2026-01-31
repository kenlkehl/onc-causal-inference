# cdt/models/traditional_logreg.py
"""Traditional logistic regression causal head with treatment as feature."""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


class TraditionalLogRegNet(nn.Module):
    """
    Traditional Logistic Regression approach for causal inference.

    Architecture:
    - Shared representation layers: Phi(X)
    - Propensity head: Phi(X) -> P(T=1|X)
    - Outcome head: [Phi(X), T] -> P(Y|X, T)

    Key difference from DragonNet/UpliftNet:
    Instead of predicting potential outcomes Y0 and Y1 from separate heads,
    this approach predicts Y directly conditioned on the actual treatment T
    by concatenating T as a feature to the outcome head.

    ITE Computation at inference:
    - y1_prob = sigmoid(outcome_head([Phi(X), 1]))
    - y0_prob = sigmoid(outcome_head([Phi(X), 0]))
    - ITE = y1_prob - y0_prob

    This is the classical parametric approach to estimating treatment effects
    from observational data using logistic regression with adjustment for
    confounders (represented by Phi(X)).
    """

    def __init__(
        self,
        input_dim: int,
        representation_dim: int = 200,
        hidden_outcome_dim: int = 100,
        dropout: float = 0.2
    ):
        """
        Initialize TraditionalLogRegNet.

        Args:
            input_dim: Dimension of input features from feature extractor
            representation_dim: Dimension of shared representation
            hidden_outcome_dim: Hidden dimension for outcome head
            dropout: Dropout rate
        """
        super().__init__()
        self.dropout_rate = dropout
        self.representation_dim = representation_dim

        # Shared representation layers (matching DragonNet's simple 2-layer version)
        self.representation_fc1 = nn.Linear(input_dim, representation_dim)
        self.representation_fc2 = nn.Linear(representation_dim, representation_dim)
        self.rep_dropout = nn.Dropout(dropout)

        # Propensity head: P(T=1|X) - single linear layer
        self.propensity_fc = nn.Linear(representation_dim, 1)

        # Outcome head: P(Y|X, T) - takes [Phi(X), T] as input
        # Input dimension is representation_dim + 1 (for treatment indicator)
        self.outcome_fc1 = nn.Linear(representation_dim + 1, hidden_outcome_dim)
        self.outcome_fc2 = nn.Linear(hidden_outcome_dim, hidden_outcome_dim)
        self.outcome_fc3 = nn.Linear(hidden_outcome_dim, 1)
        self.outcome_dropout = nn.Dropout(dropout)

    def forward(self, features: torch.Tensor, treatment: torch.Tensor = None):
        """
        Forward pass through TraditionalLogRegNet.

        Args:
            features: Output from feature extractor, shape (batch, input_dim)
            treatment: Treatment indicator, shape (batch,) or (batch, 1)
                      If None, returns potential outcomes for both T=0 and T=1

        Returns:
            If treatment is provided:
                y_logit: Outcome logit P(Y|X, T), shape (batch, 1)
                t_logit: Propensity logit, shape (batch, 1)
                phi: Shared representation, shape (batch, representation_dim)

            If treatment is None (counterfactual mode):
                y0_logit: Outcome logit with T=0, shape (batch, 1)
                y1_logit: Outcome logit with T=1, shape (batch, 1)
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

        if treatment is not None:
            # Training mode: predict Y given observed treatment
            if treatment.dim() == 1:
                treatment = treatment.unsqueeze(1)
            treatment = treatment.float()

            # Concatenate phi and treatment
            outcome_input = torch.cat([phi, treatment], dim=1)
            y_logit = self._outcome_head(outcome_input)

            return y_logit, t_logit, phi
        else:
            # Counterfactual mode: predict Y for both T=0 and T=1
            batch_size = features.size(0)
            device = features.device

            # T=0
            t0 = torch.zeros(batch_size, 1, device=device)
            outcome_input_0 = torch.cat([phi, t0], dim=1)
            y0_logit = self._outcome_head(outcome_input_0)

            # T=1
            t1 = torch.ones(batch_size, 1, device=device)
            outcome_input_1 = torch.cat([phi, t1], dim=1)
            y1_logit = self._outcome_head(outcome_input_1)

            return y0_logit, y1_logit, t_logit, phi

    def _outcome_head(self, outcome_input: torch.Tensor) -> torch.Tensor:
        """
        Compute outcome logit from [phi, treatment] input.

        Args:
            outcome_input: Concatenated [phi, treatment], shape (batch, representation_dim + 1)

        Returns:
            y_logit: Outcome logit, shape (batch, 1)
        """
        y = F.relu(self.outcome_fc1(outcome_input))
        y = self.outcome_dropout(y)
        y = F.elu(self.outcome_fc2(y))
        y = self.outcome_dropout(y)
        y_logit = self.outcome_fc3(y)
        return y_logit

    def get_representation(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute shared representation from input features.

        Args:
            features: Input features, shape (batch, input_dim)

        Returns:
            phi: Shared representation, shape (batch, representation_dim)
        """
        h = F.relu(self.representation_fc1(features))
        h = self.rep_dropout(h)
        phi = F.elu(self.representation_fc2(h))
        phi = self.rep_dropout(phi)
        return phi

    def propensity_from_representation(self, phi: torch.Tensor) -> torch.Tensor:
        """
        Compute propensity logit from shared representation.

        Args:
            phi: Shared representation, shape (batch, representation_dim)

        Returns:
            t_logit: Propensity logit, shape (batch, 1)
        """
        return self.propensity_fc(phi)

    def outcome_from_representation(
        self,
        phi: torch.Tensor,
        treatment: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute outcome logit from representation and treatment.

        Args:
            phi: Shared representation, shape (batch, representation_dim)
            treatment: Treatment indicator, shape (batch,) or (batch, 1)

        Returns:
            y_logit: Outcome logit, shape (batch, 1)
        """
        if treatment.dim() == 1:
            treatment = treatment.unsqueeze(1)
        treatment = treatment.float()
        outcome_input = torch.cat([phi, treatment], dim=1)
        return self._outcome_head(outcome_input)
