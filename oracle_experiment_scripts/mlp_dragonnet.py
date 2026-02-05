# oracle_experiment_scripts/mlp_dragonnet.py
"""Simple MLP-based DragonNet for categorical/tabular features.

This model is used for condition 6 in the concept-aware experiment:
training DragonNet on LLM-extracted categorical features only.

Note: This file was moved from cdt/models/mlp_dragonnet.py since it is
only used by oracle experiment scripts.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from cdt.models import DragonNet


logger = logging.getLogger(__name__)


class MLPDragonNet(nn.Module):
    """
    MLP-based DragonNet for categorical/tabular features.

    Takes one-hot encoded categorical features as input and produces
    treatment effect estimates using the DragonNet architecture.

    This is simpler than CausalText - no text processing, just
    a direct MLP on tabular features.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [64, 64],
        causal_head_representation_dim: int = 64,
        causal_head_hidden_outcome_dim: int = 32,
        causal_head_dropout: float = 0.2,
        input_dropout: float = 0.1,
        device: str = "cuda:0"
    ):
        """
        Initialize MLP DragonNet.

        Args:
            input_dim: Dimension of input features (e.g., number of categories)
            hidden_dims: List of hidden layer dimensions for the MLP encoder
            causal_head_representation_dim: Causal head representation dimension
            causal_head_hidden_outcome_dim: Causal head outcome hidden dimension
            causal_head_dropout: Dropout rate for causal head layers
            input_dropout: Dropout rate for input layer
            device: Device string
        """
        super().__init__()

        self._device = torch.device(device)
        self.input_dim = input_dim

        # Store config for checkpointing
        self.config = {
            'input_dim': input_dim,
            'hidden_dims': hidden_dims,
            'causal_head_representation_dim': causal_head_representation_dim,
            'causal_head_hidden_outcome_dim': causal_head_hidden_outcome_dim,
            'causal_head_dropout': causal_head_dropout,
            'input_dropout': input_dropout
        }

        # Build MLP encoder
        layers = []
        prev_dim = input_dim

        # Input dropout
        layers.append(nn.Dropout(input_dropout))

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(causal_head_dropout))
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

        # Final projection to match DragonNet input
        encoder_output_dim = hidden_dims[-1] if hidden_dims else input_dim
        self.projection = nn.Linear(encoder_output_dim, causal_head_representation_dim)

        # DragonNet head
        self.dragonnet = DragonNet(
            input_dim=causal_head_representation_dim,
            representation_dim=causal_head_representation_dim,
            hidden_outcome_dim=causal_head_hidden_outcome_dim,
            dropout=causal_head_dropout
        )

        # Move to device
        self.to(self._device)

        logger.info(f"MLPDragonNet initialized:")
        logger.info(f"  Input dim: {input_dim}")
        logger.info(f"  Hidden dims: {hidden_dims}")
        logger.info(f"  Causal head representation dim: {causal_head_representation_dim}")
        logger.info(f"  Device: {self._device}")

    @property
    def output_dim(self) -> int:
        """Output dimension of the encoder."""
        return self.config['causal_head_representation_dim']

    def forward(
        self,
        features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the complete model.

        Args:
            features: Input features tensor (batch, input_dim)

        Returns:
            y0_logit: (batch, 1) - outcome prediction under control
            y1_logit: (batch, 1) - outcome prediction under treatment
            t_logit: (batch, 1) - treatment propensity logit
            final_common_layer: (batch, representation_dim) - shared representation
        """
        # Encode features
        h = self.encoder(features)
        h = self.projection(h)
        h = F.relu(h)

        # DragonNet head
        y0_logit, y1_logit, t_logit, final_common_layer = self.dragonnet(h)

        return y0_logit, y1_logit, t_logit, final_common_layer

    def train_step(
        self,
        batch: Dict[str, Any],
        alpha_propensity: float = 1.0,
        beta_targreg: float = 0.1,
        label_smoothing: float = 0.0
    ) -> Dict[str, torch.Tensor]:
        """
        Perform single training step.

        Args:
            batch: Dictionary with 'features', 'treatment', 'outcome' keys
            alpha_propensity: Weight for propensity loss
            beta_targreg: Weight for targeted regularization
            label_smoothing: Label smoothing factor

        Returns:
            Dictionary with loss components and detached predictions
        """
        features = batch['features']  # (batch, input_dim)
        treatments = batch['treatment']  # (batch,)
        outcomes = batch['outcome']  # (batch,)

        # Move to device if needed
        if features.device != self._device:
            features = features.to(self._device)
        if treatments.device != self._device:
            treatments = treatments.to(self._device)
        if outcomes.device != self._device:
            outcomes = outcomes.to(self._device)

        # Apply label smoothing if enabled
        if label_smoothing > 0:
            treatments_smooth = treatments * (1 - label_smoothing) + 0.5 * label_smoothing
            outcomes_smooth = outcomes * (1 - label_smoothing) + 0.5 * label_smoothing
        else:
            treatments_smooth = treatments
            outcomes_smooth = outcomes

        # Forward pass
        y0_logit, y1_logit, t_logit, phi = self.forward(features)

        # Propensity loss
        propensity_loss = F.binary_cross_entropy_with_logits(
            t_logit.squeeze(-1),
            treatments_smooth
        )

        # Outcome loss - factual outcome only
        factual_logit = torch.where(
            treatments.unsqueeze(1) > 0.5,
            y1_logit,
            y0_logit
        )

        outcome_loss = F.binary_cross_entropy_with_logits(
            factual_logit.squeeze(-1),
            outcomes_smooth
        )

        # Targeted regularization (R-loss)
        if beta_targreg > 0:
            with torch.no_grad():
                propensity = torch.sigmoid(t_logit).clamp(1e-3, 1 - 1e-3)
                H = (treatments.unsqueeze(1) / propensity) - \
                    ((1 - treatments.unsqueeze(1)) / (1 - propensity))

            factual_prob = torch.sigmoid(factual_logit)
            moment = torch.mean((outcomes.unsqueeze(1) - factual_prob) * H)
            targreg_loss = moment ** 2
        else:
            targreg_loss = torch.tensor(0.0, device=self._device)

        # Total loss
        total_loss = (
            outcome_loss +
            alpha_propensity * propensity_loss +
            beta_targreg * targreg_loss
        )

        return {
            'loss': total_loss,
            'outcome_loss': outcome_loss.detach(),
            'propensity_loss': propensity_loss.detach(),
            'targreg_loss': targreg_loss.detach() if isinstance(targreg_loss, torch.Tensor) else targreg_loss,
            'y0_logit': y0_logit.detach(),
            'y1_logit': y1_logit.detach(),
            't_logit': t_logit.detach()
        }

    def predict(
        self,
        features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions for inference.

        Args:
            features: Input features tensor (batch, input_dim)

        Returns:
            Dictionary with prediction outputs
        """
        # Move to device if needed
        if features.device != self._device:
            features = features.to(self._device)

        with torch.no_grad():
            y0_logit, y1_logit, t_logit, final_common_layer = self.forward(features)

            # Convert to probabilities
            y0_prob = torch.sigmoid(y0_logit).squeeze(-1)
            y1_prob = torch.sigmoid(y1_logit).squeeze(-1)
            propensity = torch.sigmoid(t_logit).squeeze(-1)

            return {
                'y0_prob': y0_prob,
                'y1_prob': y1_prob,
                'propensity': propensity,
                'y0_logit': y0_logit.squeeze(-1),
                'y1_logit': y1_logit.squeeze(-1),
                't_logit': t_logit.squeeze(-1),
                'final_common_layer': final_common_layer
            }

    def to(self, device):
        """Override to track device properly."""
        self._device = device if isinstance(device, torch.device) else torch.device(device)
        return super().to(device)


class CategoricalEncoder:
    """Encode categorical variables to one-hot tensors."""

    def __init__(self, categories: Optional[List[str]] = None):
        """
        Initialize encoder.

        Args:
            categories: List of category names in order. If None, will be inferred from data.
        """
        self.categories = categories
        self.category_to_idx = {}
        self._fitted = False

    def fit(self, values: List[str]) -> 'CategoricalEncoder':
        """
        Fit encoder on categorical values.

        Args:
            values: List of category strings

        Returns:
            self for method chaining
        """
        if self.categories is None:
            # Infer categories from data
            unique_values = sorted(set(values))
            self.categories = unique_values

        self.category_to_idx = {cat: i for i, cat in enumerate(self.categories)}
        self._fitted = True

        logger.info(f"CategoricalEncoder fitted with {len(self.categories)} categories: {self.categories}")
        return self

    def transform(self, values: List[str], device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Transform categorical values to one-hot tensor.

        Args:
            values: List of category strings
            device: Optional device to place tensor on

        Returns:
            One-hot tensor of shape (len(values), num_categories)
        """
        if not self._fitted:
            raise RuntimeError("Encoder not fitted. Call fit() first.")

        num_categories = len(self.categories)
        one_hot = torch.zeros(len(values), num_categories)

        for i, val in enumerate(values):
            idx = self.category_to_idx.get(str(val), -1)
            if idx >= 0:
                one_hot[i, idx] = 1.0
            else:
                # Unknown category - use uniform distribution
                one_hot[i, :] = 1.0 / num_categories
                logger.warning(f"Unknown category: {val}")

        if device is not None:
            one_hot = one_hot.to(device)

        return one_hot

    @property
    def num_categories(self) -> int:
        """Number of categories."""
        return len(self.categories) if self.categories else 0


def create_llm_extract_only_model(
    num_categories: int = 4,
    device: str = "cuda:0"
) -> MLPDragonNet:
    """
    Create MLPDragonNet for LLM-extract-only condition (condition 6).

    Args:
        num_categories: Number of categories for the confounder
        device: Device string

    Returns:
        Configured MLPDragonNet model
    """
    return MLPDragonNet(
        input_dim=num_categories,
        hidden_dims=[32, 32],  # Small network for simple categorical input
        causal_head_representation_dim=32,
        causal_head_hidden_outcome_dim=16,
        causal_head_dropout=0.2,
        input_dropout=0.0,
        device=device
    )
