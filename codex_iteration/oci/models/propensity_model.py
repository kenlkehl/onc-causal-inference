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
        flp_chat_template_prompt: Optional[str] = None,
        flp_document_window: str = "tail",
        # Hierarchical LLM args
        hlm_model_name: str = "Qwen/Qwen3-0.6B-Base",
        hlm_chunk_size: int = 2048,
        hlm_chunk_overlap: int = 256,
        hlm_max_chunks: int = 16,
        hlm_freeze_llm: bool = True,
        hlm_gated_attention_dim: int = 128,
        hlm_projection_dim: int = 128,
        hlm_dropout: float = 0.1,
        hlm_gradient_checkpointing: bool = True,
        hlm_downprojection_dim: Optional[int] = None,
        hlm_skip_llm: bool = False,
        hlm_cached_hidden_size: int = 0,
        hlm_chat_template_prompt: Optional[str] = None,
        # Hierarchical CNN args
        hcnn_embedding_dim: int = 256,
        hcnn_conv_dim: int = 256,
        hcnn_kernel_size: int = 5,
        hcnn_num_conv_blocks: int = 4,
        hcnn_chunk_size: int = 512,
        hcnn_chunk_overlap: int = 64,
        hcnn_max_chunks: int = 32,
        hcnn_vocab_size: int = 50000,
        hcnn_gated_attention_dim: int = 128,
        hcnn_projection_dim: int = 128,
        hcnn_dropout: float = 0.1,
        # Hierarchical GRU args
        hgru_embedding_dim: int = 256,
        hgru_gru_hidden_dim: int = 256,
        hgru_num_gru_layers: int = 2,
        hgru_chunk_size: int = 512,
        hgru_chunk_overlap: int = 64,
        hgru_max_chunks: int = 32,
        hgru_vocab_size: int = 50000,
        hgru_gated_attention_dim: int = 128,
        hgru_projection_dim: int = 128,
        hgru_dropout: float = 0.1,
        # Simple CNN args
        scnn_embedding_dim: int = 256,
        scnn_conv_dim: int = 256,
        scnn_kernel_size: int = 5,
        scnn_num_conv_blocks: int = 4,
        scnn_max_length: int = 10000,
        scnn_vocab_size: int = 50000,
        scnn_gated_attention_dim: int = 128,
        scnn_projection_dim: int = 128,
        scnn_dropout: float = 0.1,
        # Propensity network args
        representation_dim: int = 128,
        device: str = "cuda:0"
    ):
        """
        Initialize propensity-only model.

        Args:
            feature_extractor_type: Feature extractor type
            flp_*: Frozen LLM Pooler args (see extractor_factory.py)
            hlm_*: Hierarchical LLM args (see extractor_factory.py)
            hcnn_*: Hierarchical CNN args (see extractor_factory.py)
            hgru_*: Hierarchical GRU args (see extractor_factory.py)
            scnn_*: Simple CNN args (see extractor_factory.py)
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
            'flp_chat_template_prompt': flp_chat_template_prompt,
            'flp_document_window': flp_document_window,
            'hlm_model_name': hlm_model_name,
            'hlm_chunk_size': hlm_chunk_size,
            'hlm_chunk_overlap': hlm_chunk_overlap,
            'hlm_max_chunks': hlm_max_chunks,
            'hlm_freeze_llm': hlm_freeze_llm,
            'hlm_gated_attention_dim': hlm_gated_attention_dim,
            'hlm_projection_dim': hlm_projection_dim,
            'hlm_dropout': hlm_dropout,
            'hlm_gradient_checkpointing': hlm_gradient_checkpointing,
            'hlm_downprojection_dim': hlm_downprojection_dim,
            'hlm_skip_llm': hlm_skip_llm,
            'hlm_cached_hidden_size': hlm_cached_hidden_size,
            'hlm_chat_template_prompt': hlm_chat_template_prompt,
            'hcnn_embedding_dim': hcnn_embedding_dim,
            'hcnn_conv_dim': hcnn_conv_dim,
            'hcnn_kernel_size': hcnn_kernel_size,
            'hcnn_num_conv_blocks': hcnn_num_conv_blocks,
            'hcnn_chunk_size': hcnn_chunk_size,
            'hcnn_chunk_overlap': hcnn_chunk_overlap,
            'hcnn_max_chunks': hcnn_max_chunks,
            'hcnn_vocab_size': hcnn_vocab_size,
            'hcnn_gated_attention_dim': hcnn_gated_attention_dim,
            'hcnn_projection_dim': hcnn_projection_dim,
            'hcnn_dropout': hcnn_dropout,
            'hgru_embedding_dim': hgru_embedding_dim,
            'hgru_gru_hidden_dim': hgru_gru_hidden_dim,
            'hgru_num_gru_layers': hgru_num_gru_layers,
            'hgru_chunk_size': hgru_chunk_size,
            'hgru_chunk_overlap': hgru_chunk_overlap,
            'hgru_max_chunks': hgru_max_chunks,
            'hgru_vocab_size': hgru_vocab_size,
            'hgru_gated_attention_dim': hgru_gated_attention_dim,
            'hgru_projection_dim': hgru_projection_dim,
            'hgru_dropout': hgru_dropout,
            'scnn_embedding_dim': scnn_embedding_dim,
            'scnn_conv_dim': scnn_conv_dim,
            'scnn_kernel_size': scnn_kernel_size,
            'scnn_num_conv_blocks': scnn_num_conv_blocks,
            'scnn_max_length': scnn_max_length,
            'scnn_vocab_size': scnn_vocab_size,
            'scnn_gated_attention_dim': scnn_gated_attention_dim,
            'scnn_projection_dim': scnn_projection_dim,
            'scnn_dropout': scnn_dropout,
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
            flp_chat_template_prompt=flp_chat_template_prompt,
            flp_document_window=flp_document_window,
            hlm_model_name=hlm_model_name,
            hlm_chunk_size=hlm_chunk_size,
            hlm_chunk_overlap=hlm_chunk_overlap,
            hlm_max_chunks=hlm_max_chunks,
            hlm_freeze_llm=hlm_freeze_llm,
            hlm_gated_attention_dim=hlm_gated_attention_dim,
            hlm_projection_dim=hlm_projection_dim,
            hlm_dropout=hlm_dropout,
            hlm_gradient_checkpointing=hlm_gradient_checkpointing,
            hlm_downprojection_dim=hlm_downprojection_dim,
            hlm_skip_llm=hlm_skip_llm,
            hlm_cached_hidden_size=hlm_cached_hidden_size,
            hlm_chat_template_prompt=hlm_chat_template_prompt,
            hcnn_embedding_dim=hcnn_embedding_dim,
            hcnn_conv_dim=hcnn_conv_dim,
            hcnn_kernel_size=hcnn_kernel_size,
            hcnn_num_conv_blocks=hcnn_num_conv_blocks,
            hcnn_chunk_size=hcnn_chunk_size,
            hcnn_chunk_overlap=hcnn_chunk_overlap,
            hcnn_max_chunks=hcnn_max_chunks,
            hcnn_vocab_size=hcnn_vocab_size,
            hcnn_gated_attention_dim=hcnn_gated_attention_dim,
            hcnn_projection_dim=hcnn_projection_dim,
            hcnn_dropout=hcnn_dropout,
            hgru_embedding_dim=hgru_embedding_dim,
            hgru_gru_hidden_dim=hgru_gru_hidden_dim,
            hgru_num_gru_layers=hgru_num_gru_layers,
            hgru_chunk_size=hgru_chunk_size,
            hgru_chunk_overlap=hgru_chunk_overlap,
            hgru_max_chunks=hgru_max_chunks,
            hgru_vocab_size=hgru_vocab_size,
            hgru_gated_attention_dim=hgru_gated_attention_dim,
            hgru_projection_dim=hgru_projection_dim,
            hgru_dropout=hgru_dropout,
            scnn_embedding_dim=scnn_embedding_dim,
            scnn_conv_dim=scnn_conv_dim,
            scnn_kernel_size=scnn_kernel_size,
            scnn_num_conv_blocks=scnn_num_conv_blocks,
            scnn_max_length=scnn_max_length,
            scnn_vocab_size=scnn_vocab_size,
            scnn_gated_attention_dim=scnn_gated_attention_dim,
            scnn_projection_dim=scnn_projection_dim,
            scnn_dropout=scnn_dropout,
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

    def fit_tokenizer(self, texts):
        """Fit tokenizer for trainable-from-scratch extractors. No-op for LLM-based."""
        if hasattr(self.feature_extractor, 'fit_tokenizer'):
            self.feature_extractor.fit_tokenizer(texts)

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
        flp_chat_template_prompt=getattr(arch_config, 'flp_chat_template_prompt', None),
        flp_document_window=getattr(arch_config, 'flp_document_window', 'tail'),
        # Hierarchical LLM args
        hlm_model_name=getattr(arch_config, 'hlm_model_name', 'Qwen/Qwen3-0.6B-Base'),
        hlm_chunk_size=getattr(arch_config, 'hlm_chunk_size', 2048),
        hlm_chunk_overlap=getattr(arch_config, 'hlm_chunk_overlap', 256),
        hlm_max_chunks=getattr(arch_config, 'hlm_max_chunks', 16),
        hlm_freeze_llm=getattr(arch_config, 'hlm_freeze_llm', True),
        hlm_gated_attention_dim=getattr(arch_config, 'hlm_gated_attention_dim', 128),
        hlm_projection_dim=getattr(arch_config, 'hlm_projection_dim', 128),
        hlm_dropout=getattr(arch_config, 'hlm_dropout', 0.1),
        hlm_gradient_checkpointing=getattr(arch_config, 'hlm_gradient_checkpointing', True),
        hlm_downprojection_dim=getattr(arch_config, 'hlm_downprojection_dim', None),
        hlm_skip_llm=getattr(arch_config, 'hlm_skip_llm', False),
        hlm_cached_hidden_size=getattr(arch_config, 'hlm_cached_hidden_size', 0),
        hlm_chat_template_prompt=getattr(arch_config, 'hlm_chat_template_prompt', None),
        # Hierarchical CNN args
        hcnn_embedding_dim=getattr(arch_config, 'hcnn_embedding_dim', 256),
        hcnn_conv_dim=getattr(arch_config, 'hcnn_conv_dim', 256),
        hcnn_kernel_size=getattr(arch_config, 'hcnn_kernel_size', 5),
        hcnn_num_conv_blocks=getattr(arch_config, 'hcnn_num_conv_blocks', 4),
        hcnn_chunk_size=getattr(arch_config, 'hcnn_chunk_size', 512),
        hcnn_chunk_overlap=getattr(arch_config, 'hcnn_chunk_overlap', 64),
        hcnn_max_chunks=getattr(arch_config, 'hcnn_max_chunks', 32),
        hcnn_vocab_size=getattr(arch_config, 'hcnn_vocab_size', 50000),
        hcnn_gated_attention_dim=getattr(arch_config, 'hcnn_gated_attention_dim', 128),
        hcnn_projection_dim=getattr(arch_config, 'hcnn_projection_dim', 128),
        hcnn_dropout=getattr(arch_config, 'hcnn_dropout', 0.1),
        # Hierarchical GRU args
        hgru_embedding_dim=getattr(arch_config, 'hgru_embedding_dim', 256),
        hgru_gru_hidden_dim=getattr(arch_config, 'hgru_gru_hidden_dim', 256),
        hgru_num_gru_layers=getattr(arch_config, 'hgru_num_gru_layers', 2),
        hgru_chunk_size=getattr(arch_config, 'hgru_chunk_size', 512),
        hgru_chunk_overlap=getattr(arch_config, 'hgru_chunk_overlap', 64),
        hgru_max_chunks=getattr(arch_config, 'hgru_max_chunks', 32),
        hgru_vocab_size=getattr(arch_config, 'hgru_vocab_size', 50000),
        hgru_gated_attention_dim=getattr(arch_config, 'hgru_gated_attention_dim', 128),
        hgru_projection_dim=getattr(arch_config, 'hgru_projection_dim', 128),
        hgru_dropout=getattr(arch_config, 'hgru_dropout', 0.1),
        # Simple CNN args
        scnn_embedding_dim=getattr(arch_config, 'scnn_embedding_dim', 256),
        scnn_conv_dim=getattr(arch_config, 'scnn_conv_dim', 256),
        scnn_kernel_size=getattr(arch_config, 'scnn_kernel_size', 5),
        scnn_num_conv_blocks=getattr(arch_config, 'scnn_num_conv_blocks', 4),
        scnn_max_length=getattr(arch_config, 'scnn_max_length', 10000),
        scnn_vocab_size=getattr(arch_config, 'scnn_vocab_size', 50000),
        scnn_gated_attention_dim=getattr(arch_config, 'scnn_gated_attention_dim', 128),
        scnn_projection_dim=getattr(arch_config, 'scnn_projection_dim', 128),
        scnn_dropout=getattr(arch_config, 'scnn_dropout', 0.1),
        # Propensity network args
        representation_dim=representation_dim,
        device=str(device)
    )

    return model
