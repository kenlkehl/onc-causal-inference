# cdt/models/dr_moce.py
"""DR-MoCE: Doubly-Robust Mixture of Causal Experts.

Combines four properties of causal forests in a fully differentiable architecture:
1. Doubly-robust estimation via AIPW pseudo-outcomes
2. Local/adaptive estimation via mixture of experts
3. Honest estimation via mini-batch cross-fitting (prediction buffer)
4. Confidence intervals via heteroscedastic mixture variance

References:
    Kennedy (2023). Towards optimal doubly robust estimation of heterogeneous
    causal effects. Electronic Journal of Statistics.
"""

import logging
from collections import deque
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


def compute_dr_pseudo_outcome(
    Y: torch.Tensor,
    T: torch.Tensor,
    e: torch.Tensor,
    mu0: torch.Tensor,
    mu1: torch.Tensor,
    clip: float = 0.01
) -> torch.Tensor:
    """Compute the AIPW doubly-robust pseudo-outcome.

    Gamma = mu1(X) - mu0(X) + T*(Y - mu1(X))/e(X) - (1-T)*(Y - mu0(X))/(1-e(X))

    This is consistent for tau(X) if EITHER the propensity model e(X) OR
    the outcome models (mu0, mu1) are correctly specified.

    Args:
        Y: Observed outcomes, shape (batch,)
        T: Treatment indicators, shape (batch,)
        e: Propensity scores P(T=1|X), shape (batch,)
        mu0: Predicted E[Y|X, T=0], shape (batch,)
        mu1: Predicted E[Y|X, T=1], shape (batch,)
        clip: Clipping threshold for propensity scores to avoid extreme weights

    Returns:
        Gamma: DR pseudo-outcomes, shape (batch,)
    """
    e_clipped = e.clamp(clip, 1.0 - clip)

    Gamma = (
        (mu1 - mu0)
        + T * (Y - mu1) / e_clipped
        - (1.0 - T) * (Y - mu0) / (1.0 - e_clipped)
    )

    return Gamma


class DRMoCENet(nn.Module):
    """Doubly-Robust Mixture of Causal Experts network.

    Architecture:
        Shared Representation -> Nuisance Heads (e, mu0, mu1)
                              -> Router (soft assignment over K experts)
                              -> K Expert Heads (mean_k, log_var_k each)

    The mixture aggregation:
        tau(X) = Sum_k g_k(X) * mean_k(X)
        sigma^2(X) = Sum_k g_k(X) * (var_k + mean_k^2) - tau^2  (law of total variance)

    Args:
        input_dim: Dimension of input features from feature extractor
        representation_dim: Dimension of shared representation
        hidden_outcome_dim: Hidden dimension for nuisance/expert heads
        num_experts: Number of effect expert heads (K)
        router_temperature: Softmax temperature for routing (lower = sharper)
        dropout: Dropout rate
    """

    def __init__(
        self,
        input_dim: int,
        representation_dim: int = 200,
        hidden_outcome_dim: int = 100,
        num_experts: int = 8,
        router_temperature: float = 1.0,
        dropout: float = 0.2
    ):
        super().__init__()
        self.num_experts = num_experts
        self.temperature = router_temperature
        self.dropout_rate = dropout

        # Shared representation layers (2 layers, same as DragonNet)
        self.representation_fc1 = nn.Linear(input_dim, representation_dim)
        self.representation_fc2 = nn.Linear(representation_dim, representation_dim)
        self.rep_dropout = nn.Dropout(dropout)

        # Nuisance heads
        # Propensity head: P(T=1|X) - single linear layer (matches DragonNet)
        self.propensity_fc = nn.Linear(representation_dim, 1)

        # Potential outcome heads: E[Y|X, T=0] and E[Y|X, T=1]
        # 3 layers each (matching DragonNet outcome heads)
        self.mu0_fc1 = nn.Linear(representation_dim, hidden_outcome_dim)
        self.mu0_fc2 = nn.Linear(hidden_outcome_dim, hidden_outcome_dim)
        self.mu0_fc3 = nn.Linear(hidden_outcome_dim, 1)

        self.mu1_fc1 = nn.Linear(representation_dim, hidden_outcome_dim)
        self.mu1_fc2 = nn.Linear(hidden_outcome_dim, hidden_outcome_dim)
        self.mu1_fc3 = nn.Linear(hidden_outcome_dim, 1)

        self.outcome_dropout = nn.Dropout(dropout)

        # Router: maps representation to soft assignment over K experts
        self.router = nn.Linear(representation_dim, num_experts)

        # K effect experts: each outputs (mean_k, log_var_k) = 2 values
        # 3 layers each (matching other heads)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(representation_dim, hidden_outcome_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_outcome_dim, hidden_outcome_dim),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_outcome_dim, 2)  # (mean_k, log_var_k)
            )
            for _ in range(num_experts)
        ])

    def forward(
        self, features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through DR-MoCE network.

        Args:
            features: Output from feature extractor, shape (batch, input_dim)

        Returns:
            mu0_logit: E[Y|X, T=0] logit, shape (batch, 1)
            mu1_logit: E[Y|X, T=1] logit, shape (batch, 1)
            tau: Mixture treatment effect estimate, shape (batch,)
            sigma2: Mixture variance estimate, shape (batch,)
            t_logit: Propensity logit, shape (batch, 1)
            routing_weights: Expert assignment weights, shape (batch, K)
            expert_means: Per-expert mean predictions, shape (batch, K)
            phi: Shared representation, shape (batch, representation_dim)
        """
        # Shared representation
        h = F.relu(self.representation_fc1(features))
        h = self.rep_dropout(h)
        phi = F.elu(self.representation_fc2(h))
        phi = self.rep_dropout(phi)

        # Nuisance heads
        t_logit = self.propensity_fc(phi)

        mu0 = F.relu(self.mu0_fc1(phi))
        mu0 = self.outcome_dropout(mu0)
        mu0 = F.elu(self.mu0_fc2(mu0))
        mu0 = self.outcome_dropout(mu0)
        mu0_logit = self.mu0_fc3(mu0)

        mu1 = F.relu(self.mu1_fc1(phi))
        mu1 = self.outcome_dropout(mu1)
        mu1 = F.elu(self.mu1_fc2(mu1))
        mu1 = self.outcome_dropout(mu1)
        mu1_logit = self.mu1_fc3(mu1)

        # Router
        routing_logits = self.router(phi) / self.temperature
        g = F.softmax(routing_logits, dim=-1)  # (batch, K)

        # Expert predictions
        expert_outputs = [expert(phi) for expert in self.experts]
        # Stack: (batch, K, 2)
        stacked = torch.stack(expert_outputs, dim=1)
        expert_means = stacked[:, :, 0]     # (batch, K)
        expert_log_vars = stacked[:, :, 1]  # (batch, K)

        # Mixture aggregation
        tau = (g * expert_means).sum(dim=1)  # (batch,)

        # Law of total variance:
        # Var = E[Var_k] + Var[E_k]
        # = Sum_k g_k * var_k + Sum_k g_k * mean_k^2 - tau^2
        expert_vars = torch.exp(expert_log_vars)  # (batch, K)
        sigma2 = (g * (expert_vars + expert_means ** 2)).sum(dim=1) - tau ** 2

        return mu0_logit, mu1_logit, tau, sigma2, t_logit, g, expert_means, phi

    def get_representation(self, features: torch.Tensor) -> torch.Tensor:
        """Compute shared representation from input features."""
        h = F.relu(self.representation_fc1(features))
        h = self.rep_dropout(h)
        phi = F.elu(self.representation_fc2(h))
        phi = self.rep_dropout(phi)
        return phi

    def propensity_from_representation(self, phi: torch.Tensor) -> torch.Tensor:
        """Compute propensity logit from shared representation."""
        return self.propensity_fc(phi)


class NuisancePredictionBuffer:
    """FIFO buffer for nuisance predictions to enable cross-fitting.

    Maintains a rolling buffer of (e, mu0, mu1) predictions from previous
    mini-batches. When computing DR pseudo-outcomes, the buffered (stale)
    predictions can be used instead of current-batch predictions, providing
    implicit sample splitting analogous to causal forest honesty.

    The staleness means the nuisance parameters that generated the buffered
    predictions differ from the current parameters being optimized.

    Args:
        buffer_size: Maximum number of predictions to store
    """

    def __init__(self, buffer_size: int = 1024):
        self.buffer_size = buffer_size
        self.e_buffer = deque(maxlen=buffer_size)
        self.mu0_buffer = deque(maxlen=buffer_size)
        self.mu1_buffer = deque(maxlen=buffer_size)
        self.Y_buffer = deque(maxlen=buffer_size)
        self.T_buffer = deque(maxlen=buffer_size)

    def push(
        self,
        e: torch.Tensor,
        mu0: torch.Tensor,
        mu1: torch.Tensor,
        Y: torch.Tensor,
        T: torch.Tensor
    ) -> None:
        """Add a batch of predictions to the buffer.

        All tensors should be detached and on CPU to avoid memory leaks.
        """
        batch_size = e.shape[0]
        for i in range(batch_size):
            self.e_buffer.append(e[i].item())
            self.mu0_buffer.append(mu0[i].item())
            self.mu1_buffer.append(mu1[i].item())
            self.Y_buffer.append(Y[i].item())
            self.T_buffer.append(T[i].item())

    def is_ready(self, min_samples: int = 64) -> bool:
        """Check if buffer has enough samples for cross-fitting."""
        return len(self.e_buffer) >= min_samples

    def sample(
        self, n: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample n predictions from the buffer.

        Returns:
            e, mu0, mu1, Y, T tensors on the specified device
        """
        buf_size = len(self.e_buffer)
        indices = torch.randint(0, buf_size, (n,))

        e_list = [self.e_buffer[i] for i in indices]
        mu0_list = [self.mu0_buffer[i] for i in indices]
        mu1_list = [self.mu1_buffer[i] for i in indices]
        Y_list = [self.Y_buffer[i] for i in indices]
        T_list = [self.T_buffer[i] for i in indices]

        return (
            torch.tensor(e_list, device=device),
            torch.tensor(mu0_list, device=device),
            torch.tensor(mu1_list, device=device),
            torch.tensor(Y_list, device=device),
            torch.tensor(T_list, device=device),
        )

    def clear(self) -> None:
        """Clear the buffer."""
        self.e_buffer.clear()
        self.mu0_buffer.clear()
        self.mu1_buffer.clear()
        self.Y_buffer.clear()
        self.T_buffer.clear()
