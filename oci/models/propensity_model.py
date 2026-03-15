# oci/models/propensity_model.py
"""Propensity-only model for dataset trimming before causal inference."""

import gc
import logging
from typing import Optional, List, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .extractor_factory import create_feature_extractor
from ..config import normalize_feature_extractor_type


logger = logging.getLogger(__name__)


class PropensityNet(nn.Module):
    """
    Propensity prediction network with same representation as DragonNet.

    Uses 6-layer representation followed by a single propensity head.
    """

    def __init__(self, input_dim: int, representation_dim: int = 200):
        super().__init__()

        # Shared representation layers (same as DragonNet)
        self.representation_fc1 = nn.Linear(input_dim, representation_dim)
        self.representation_fc2 = nn.Linear(representation_dim, representation_dim)

        # Single propensity head (same as DragonNet)
        self.propensity_fc1 = nn.Linear(representation_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the propensity network.

        Args:
            features: Feature tensor from feature extractor (batch, input_dim)

        Returns:
            t_logit: Propensity logits (batch, 1)
        """
        h = F.relu(self.representation_fc1(features))
        h = F.elu(self.representation_fc2(h))

        t_logit = self.propensity_fc1(h)

        return t_logit


class PropensityOnlyModel(nn.Module):
    """
    Propensity-score-only model for dataset trimming.

    Uses Frozen LLM Pooler feature extractor + propensity head.
    This model is trained to predict P(T=1|X) using binary cross-entropy loss.
    Used for generating propensity scores for trimming before causal model training.
    """

    def __init__(
        self,
        # Feature extractor type
        feature_extractor_type: str = "frozen_llm_pooler",
        # Frozen LLM Pooler args
        flp_model_name: str = "Qwen/Qwen3-0.6B-Base",
        flp_max_length: int = 8192,
        flp_freeze_llm: bool = True,
        flp_gated_attention_dim: int = 128,
        flp_projection_dim: int = 128,
        flp_dropout: float = 0.1,
        flp_gradient_checkpointing: bool = True,
        flp_downprojection_dim: Optional[int] = None,
        flp_skip_llm: bool = False,
        flp_cached_hidden_size: int = 0,
        # Numeric feature args
        numeric_features_enabled: bool = False,
        numeric_embedding_dim: int = 32,
        numeric_magnitude_bins: int = 8,
        numeric_type_categories: int = 10,
        # Propensity network args
        representation_dim: int = 128,
        device: str = "cuda:0"
    ):
        """
        Initialize propensity-only model.

        Args:
            feature_extractor_type: Feature extractor type (only "frozen_llm_pooler" supported)
            flp_model_name: HuggingFace model name for frozen LLM
            flp_max_length: Maximum sequence length
            flp_freeze_llm: Whether to freeze LLM weights
            flp_gated_attention_dim: Gated attention dimension
            flp_projection_dim: Projection dimension
            flp_dropout: Dropout rate
            flp_gradient_checkpointing: Enable gradient checkpointing
            flp_downprojection_dim: Optional downprojection dimension
            flp_skip_llm: Skip LLM (use cached hidden states)
            flp_cached_hidden_size: Size of cached hidden states
            numeric_features_enabled: Enable numeric feature extraction
            numeric_embedding_dim: Numeric embedding dimension
            numeric_magnitude_bins: Number of magnitude bins
            numeric_type_categories: Number of type categories
            representation_dim: Dimension of representation layers
            device: Device string
        """
        super().__init__()

        self._device = torch.device(device)
        # Normalize feature extractor type
        self.feature_extractor_type = normalize_feature_extractor_type(feature_extractor_type)

        # Store config for checkpointing
        self.config = {
            'feature_extractor_type': feature_extractor_type,
            'flp_model_name': flp_model_name,
            'flp_max_length': flp_max_length,
            'flp_freeze_llm': flp_freeze_llm,
            'flp_gated_attention_dim': flp_gated_attention_dim,
            'flp_projection_dim': flp_projection_dim,
            'flp_dropout': flp_dropout,
            'flp_gradient_checkpointing': flp_gradient_checkpointing,
            'flp_downprojection_dim': flp_downprojection_dim,
            'flp_skip_llm': flp_skip_llm,
            'flp_cached_hidden_size': flp_cached_hidden_size,
            'numeric_features_enabled': numeric_features_enabled,
            'numeric_embedding_dim': numeric_embedding_dim,
            'numeric_magnitude_bins': numeric_magnitude_bins,
            'numeric_type_categories': numeric_type_categories,
            'representation_dim': representation_dim
        }

        # Initialize feature extractor using factory
        self.feature_extractor = create_feature_extractor(
            extractor_type=self.feature_extractor_type,
            device=self._device,
            model_type="rlearner",
            flp_model_name=flp_model_name,
            flp_max_length=flp_max_length,
            flp_freeze_llm=flp_freeze_llm,
            flp_gated_attention_dim=flp_gated_attention_dim,
            flp_projection_dim=flp_projection_dim,
            flp_dropout=flp_dropout,
            flp_gradient_checkpointing=flp_gradient_checkpointing,
            flp_downprojection_dim=flp_downprojection_dim,
            flp_skip_llm=flp_skip_llm,
            flp_cached_hidden_size=flp_cached_hidden_size,
            numeric_features_enabled=numeric_features_enabled,
            numeric_embedding_dim=numeric_embedding_dim,
            numeric_magnitude_bins=numeric_magnitude_bins,
            numeric_type_categories=numeric_type_categories,
        )
        logger.info(f"Propensity model using {self.feature_extractor_type.upper()} feature extractor")

        # Propensity network
        input_dim = self.feature_extractor.output_dim
        self.propensity_net = PropensityNet(
            input_dim=input_dim,
            representation_dim=representation_dim
        )

        # Move to device
        self.to(self._device)

        logger.info(f"PropensityOnlyModel initialized:")
        logger.info(f"  Feature extractor: {self.feature_extractor_type}")
        logger.info(f"  Feature extractor output: {input_dim}")
        logger.info(f"  Representation dim: {representation_dim}")
        logger.info(f"  Device: {self._device}")

    @staticmethod
    def _get_extractor_input(batch, texts):
        """Return preprocessed batch if available, otherwise raw texts."""
        if 'cached_hidden_states' in batch:
            return {
                'cached_hidden_states': batch['cached_hidden_states'],
                'cached_attention_mask': batch['cached_attention_mask'],
                'texts': texts,
            }
        if 'chunk_input_ids' in batch or 'chunk_token_ids' in batch:
            return batch
        return texts

    def forward(self, texts_or_batch) -> torch.Tensor:
        """
        Forward pass through the complete model.

        Args:
            texts_or_batch: List of text strings or preprocessed batch dict

        Returns:
            t_logit: Propensity logits (batch, 1)
        """
        features = self.feature_extractor(texts_or_batch)
        t_logit = self.propensity_net(features)
        return t_logit

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Perform single training step.

        Args:
            batch: Dictionary with 'texts' and 'treatment' keys

        Returns:
            Dictionary with loss and predictions
        """
        texts = batch['texts']
        treatments = batch['treatment']  # (batch,)
        extractor_input = self._get_extractor_input(batch, texts)

        # Forward pass
        t_logit = self.forward(extractor_input)

        # Binary cross-entropy loss for treatment prediction
        loss = F.binary_cross_entropy_with_logits(
            t_logit.squeeze(-1),
            treatments
        )

        return {
            'loss': loss,
            't_logit': t_logit.detach()
        }

    def predict(self, texts_or_batch) -> torch.Tensor:
        """
        Predict propensity scores.

        Args:
            texts_or_batch: List of text strings or preprocessed batch dict from DataLoader

        Returns:
            Propensity probabilities (batch,)
        """
        with torch.no_grad():
            if isinstance(texts_or_batch, dict):
                texts = texts_or_batch['texts']
                extractor_input = self._get_extractor_input(texts_or_batch, texts)
            else:
                extractor_input = texts_or_batch
            t_logit = self.forward(extractor_input)
            propensity = torch.sigmoid(t_logit).squeeze(-1)
            return propensity

    def to(self, device):
        """Override to track device properly."""
        self._device = device if isinstance(device, torch.device) else torch.device(device)
        return super().to(device)


def create_propensity_model_from_config(
    arch_config,
    representation_dim: int,
    device: torch.device,
    flp_skip_llm: bool = False,
    flp_cached_hidden_size: int = 0
) -> PropensityOnlyModel:
    """
    Create a PropensityOnlyModel from architecture config.

    Args:
        arch_config: ModelArchitectureConfig instance
        representation_dim: Dimension for representation layers
        device: PyTorch device

    Returns:
        PropensityOnlyModel instance
    """
    feature_extractor_type = getattr(arch_config, 'feature_extractor_type', 'frozen_llm_pooler')

    model = PropensityOnlyModel(
        feature_extractor_type=feature_extractor_type,
        # Frozen LLM Pooler args
        flp_model_name=getattr(arch_config, 'flp_model_name', 'Qwen/Qwen3-0.6B-Base'),
        flp_max_length=getattr(arch_config, 'flp_max_length', 8192),
        flp_freeze_llm=getattr(arch_config, 'flp_freeze_llm', True),
        flp_gated_attention_dim=getattr(arch_config, 'flp_gated_attention_dim', 128),
        flp_projection_dim=getattr(arch_config, 'flp_projection_dim', 128),
        flp_dropout=getattr(arch_config, 'flp_dropout', 0.1),
        flp_gradient_checkpointing=getattr(arch_config, 'flp_gradient_checkpointing', True),
        flp_downprojection_dim=getattr(arch_config, 'flp_downprojection_dim', None),
        flp_skip_llm=flp_skip_llm,
        flp_cached_hidden_size=flp_cached_hidden_size,
        # Numeric feature args
        numeric_features_enabled=getattr(arch_config, 'numeric_features_enabled', False),
        numeric_embedding_dim=getattr(arch_config, 'numeric_embedding_dim', 32),
        numeric_magnitude_bins=getattr(arch_config, 'numeric_magnitude_bins', 8),
        numeric_type_categories=getattr(arch_config, 'numeric_type_categories', 10),
        # Propensity network args
        representation_dim=representation_dim,
        device=str(device)
    )

    return model
