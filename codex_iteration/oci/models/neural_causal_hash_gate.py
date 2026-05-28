"""Neural causal-purity gates over anonymous token-hash features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class NeuralCausalHashGateConfig:
    """Training/configuration knobs for neural hash gates."""

    n_gates: int = 8
    w_dim: int = 16
    selector_temperature: float = 1.0
    gamma_control: float = 1.0
    beta_frequency: float = 2.0
    lambda_purity: float = 20.0
    lambda_selector_entropy: float = 5e-4
    lambda_selector_diversity: float = 5e-2
    lambda_gate_frequency: float = 1e-3


class NeuralCausalHashGate(nn.Module):
    """Differentiable sparse subgroup gates plus a nuisance W branch.

    The input is a binary patient x anonymous-token-hash matrix. Each gate is a
    soft selector over hash columns, so the gate output is a differentiable
    "feature present" score. The W branch is a dense neural nuisance summary.
    """

    def __init__(
        self,
        input_dim: int,
        config: Optional[NeuralCausalHashGateConfig] = None,
    ):
        super().__init__()
        self.config = config or NeuralCausalHashGateConfig()
        if input_dim < 1:
            raise ValueError("input_dim must be positive")
        if self.config.n_gates < 1:
            raise ValueError("n_gates must be positive")
        if self.config.w_dim < 1:
            raise ValueError("w_dim must be positive")

        self.gate_logits = nn.Parameter(
            torch.randn(self.config.n_gates, input_dim) * 0.01
        )
        self.w_encoder = nn.Sequential(
            nn.Linear(input_dim, self.config.w_dim),
            nn.ReLU(),
            nn.LayerNorm(self.config.w_dim),
        )
        self.propensity_head = nn.Linear(self.config.w_dim, 1)
        self.mu0_head = nn.Linear(self.config.w_dim, 1)
        self.mu1_head = nn.Linear(self.config.w_dim, 1)

    def selector_probabilities(self) -> torch.Tensor:
        temperature = max(float(self.config.selector_temperature), 1e-3)
        return torch.softmax(self.gate_logits / temperature, dim=1)

    def forward(self, hash_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        selectors = self.selector_probabilities()
        gates = hash_features @ selectors.t()
        w = self.w_encoder(hash_features)
        return {
            "x_gates": gates,
            "w": w,
            "prop_logit": self.propensity_head(w).squeeze(-1),
            "mu0_logit": self.mu0_head(w).squeeze(-1),
            "mu1_logit": self.mu1_head(w).squeeze(-1),
            "selectors": selectors,
        }

    def nuisance_loss(
        self,
        out: Dict[str, torch.Tensor],
        treatment: torch.Tensor,
        outcome: torch.Tensor,
    ) -> torch.Tensor:
        treatment = treatment.float()
        outcome = outcome.float()
        prop_loss = F.binary_cross_entropy_with_logits(
            out["prop_logit"],
            treatment,
        )
        factual_logit = torch.where(
            treatment > 0.5,
            out["mu1_logit"],
            out["mu0_logit"],
        )
        outcome_loss = F.binary_cross_entropy_with_logits(factual_logit, outcome)
        return prop_loss + outcome_loss

    def causal_purity_score(
        self,
        gates: torch.Tensor,
        treatment: torch.Tensor,
        outcome: torch.Tensor,
        eps: float = 1e-4,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        z = gates.clamp(eps, 1.0 - eps)
        t = treatment.float().unsqueeze(1)
        c = 1.0 - t
        y = outcome.float().unsqueeze(1)
        present = z
        absent = 1.0 - z

        def weighted_mean(mask: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            numerator = (mask * weight * y).sum(dim=0)
            denominator = (mask * weight).sum(dim=0).clamp_min(eps)
            return numerator / denominator

        y_t_present = weighted_mean(t, present)
        y_c_present = weighted_mean(c, present)
        y_t_absent = weighted_mean(t, absent)
        y_c_absent = weighted_mean(c, absent)

        te_diff = (y_t_present - y_c_present) - (y_t_absent - y_c_absent)
        control_shift = y_c_present - y_c_absent
        frequency = present.mean(dim=0)

        arm_masses = torch.stack([
            (t * present).sum(dim=0),
            (c * present).sum(dim=0),
            (t * absent).sum(dim=0),
            (c * absent).sum(dim=0),
        ])
        overlap = arm_masses.min(dim=0).values / max(float(gates.shape[0]), 1.0)
        raw = te_diff - self.config.gamma_control * control_shift.abs()
        score = (
            F.relu(raw)
            * overlap.clamp_min(0.0)
            * (1.0 - frequency).clamp_min(0.0).pow(self.config.beta_frequency)
        )
        return score, {
            "te_diff": te_diff.detach(),
            "control_shift": control_shift.detach(),
            "frequency": frequency.detach(),
            "overlap": overlap.detach(),
        }

    def regularization(self, out: Dict[str, torch.Tensor]) -> torch.Tensor:
        selectors = out["selectors"]
        entropy = -(selectors * torch.log(selectors + 1e-12)).sum(dim=1).mean()
        gram = selectors @ selectors.t()
        eye = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
        diversity = (gram - eye).pow(2).mean()
        gate_frequency = out["x_gates"].mean()
        return (
            self.config.lambda_selector_entropy * entropy
            + self.config.lambda_selector_diversity * diversity
            + self.config.lambda_gate_frequency * gate_frequency
        )


def train_neural_causal_hash_gate(
    model: NeuralCausalHashGate,
    hash_features: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    steps: int = 500,
    learning_rate: float = 3e-2,
    device: str | torch.device = "cuda",
    seed: int = 0,
) -> Dict[str, float]:
    """Full-batch training for the group-level causal-purity objective."""
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    torch.manual_seed(seed)
    model.to(device)
    x = torch.as_tensor(hash_features, dtype=torch.float32, device=device)
    t = torch.as_tensor(treatment.astype(np.float32), dtype=torch.float32, device=device)
    y = torch.as_tensor(outcome.astype(np.float32), dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    best_state = None
    best_loss = float("inf")
    best_purity = 0.0
    for _step in range(int(steps)):
        out = model(x)
        purity, _stats = model.causal_purity_score(out["x_gates"], t, y)
        nuisance = model.nuisance_loss(out, t, y)
        loss = (
            nuisance
            - model.config.lambda_purity * purity.mean()
            + model.regularization(out)
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        loss_value = float(loss.detach().cpu())
        if loss_value < best_loss:
            best_loss = loss_value
            best_purity = float(purity.mean().detach().cpu())
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    return {"best_loss": best_loss, "best_purity": best_purity, "device": str(device)}


@torch.no_grad()
def extract_neural_causal_hash_features(
    model: NeuralCausalHashGate,
    hash_features: np.ndarray,
    threshold: Optional[float] = 0.5,
    device: str | torch.device = "cuda",
) -> Dict[str, np.ndarray]:
    """Return gate X features and neural W features for downstream forests."""
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    model.to(device)
    model.eval()
    x = torch.as_tensor(hash_features, dtype=torch.float32, device=device)
    out = model(x)
    gates = out["x_gates"]
    if threshold is not None:
        gates = (gates >= threshold).float()
    return {
        "X": gates.cpu().numpy().astype(np.float32),
        "W": out["w"].cpu().numpy().astype(np.float32),
        "selectors": out["selectors"].cpu().numpy().astype(np.float32),
    }
