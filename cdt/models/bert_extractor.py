# cdt/models/bert_extractor.py
"""Feature extractor using a HuggingFace transformer model's CLS token."""

import logging
from typing import Optional, List, Dict, Any
import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModel


logger = logging.getLogger(__name__)


class BertFeatureExtractor(nn.Module):
    """
    Extract text representations using a HuggingFace transformer model.

    Architecture:
    1. Tokenize text using the model's tokenizer
    2. Forward pass through transformer encoder
    3. Extract CLS token embedding (first token of last_hidden_state)
    4. Optional projection layer to match downstream dimension

    This provides a drop-in replacement for CNNFeatureExtractor with
    the same interface (forward takes texts, returns feature tensor).

    Supports any HuggingFace model with a CLS-style token output:
    - BERT, RoBERTa, DistilBERT
    - Bio_ClinicalBERT, PubMedBERT
    - ModernBERT, DeBERTa
    - Any other AutoModel-compatible encoder
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        projection_dim: Optional[int] = 128,
        max_length: int = 512,
        dropout: float = 0.1,
        freeze_encoder: bool = False,
        device: Optional[torch.device] = None
    ):
        """
        Initialize BERT feature extractor.

        Args:
            model_name: HuggingFace model name or path (e.g., "bert-base-uncased",
                "emilyalsentzer/Bio_ClinicalBERT", "answerdotai/ModernBERT-base")
            projection_dim: Final output dimension. If None, use model's hidden size.
            max_length: Maximum sequence length in tokens
            dropout: Dropout rate for projection layer
            freeze_encoder: If True, freeze transformer weights (only train projection)
            device: Device to place model on
        """
        super().__init__()

        self._model_name = model_name
        self._max_length = max_length
        self._projection_dim = projection_dim
        self._dropout_rate = dropout
        self._freeze_encoder = freeze_encoder

        if device is None:
            device = torch.device('cpu')
        self._device = device

        logger.info(f"Loading transformer model: {model_name}")

        # Load tokenizer and model from HuggingFace
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)

        # Get hidden size from model config
        self._hidden_size = self.encoder.config.hidden_size
        logger.info(f"  Hidden size: {self._hidden_size}")

        # Freeze encoder if requested
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            logger.info("  Encoder frozen (requires_grad=False)")

        # Optional projection layer
        if projection_dim is not None:
            self.projection = nn.Sequential(
                nn.Linear(self._hidden_size, projection_dim),
                nn.LayerNorm(projection_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(projection_dim, projection_dim),
                nn.LayerNorm(projection_dim),
            )
            logger.info(f"  Projection layer: {self._hidden_size} -> {projection_dim}")
        else:
            self.projection = None
            logger.info("  No projection layer (using raw CLS embedding)")

        # Dropout for CLS embedding (before projection)
        self.dropout = nn.Dropout(dropout)

        logger.info(f"BertFeatureExtractor initialized:")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Max length: {max_length}")
        logger.info(f"  Output dim: {self.output_dim}")
        logger.info(f"  Freeze encoder: {freeze_encoder}")

    @property
    def output_dim(self) -> int:
        """Total output dimension after optional projection."""
        if self._projection_dim is not None:
            return self._projection_dim
        return self._hidden_size

    @property
    def hidden_size(self) -> int:
        """The transformer model's hidden size."""
        return self._hidden_size

    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        Extract features from texts using transformer CLS token.

        Args:
            texts: List of text strings to encode

        Returns:
            Feature tensor: (batch, output_dim)
        """
        # Tokenize texts
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors='pt'
        )

        # Move to device
        input_ids = encoded['input_ids'].to(self._device)
        attention_mask = encoded['attention_mask'].to(self._device)

        # Build kwargs for model forward pass
        model_kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }

        # Some models (like RoBERTa, ModernBERT) don't use token_type_ids
        if 'token_type_ids' in encoded and hasattr(self.encoder.config, 'type_vocab_size'):
            if self.encoder.config.type_vocab_size > 0:
                model_kwargs['token_type_ids'] = encoded['token_type_ids'].to(self._device)

        # Forward through transformer
        outputs = self.encoder(**model_kwargs)

        # Extract CLS token embedding (first token)
        # outputs.last_hidden_state: (batch, seq_len, hidden_size)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # (batch, hidden_size)

        # Apply dropout
        cls_embedding = self.dropout(cls_embedding)

        # Apply projection if configured
        if self.projection is not None:
            features = self.projection(cls_embedding)
        else:
            features = cls_embedding

        return features

    def to(self, device):
        """Override to track device properly."""
        self._device = device if isinstance(device, torch.device) else torch.device(device)
        return super().to(device)

    def get_state(self) -> Dict[str, Any]:
        """Get extractor configuration for serialization."""
        return {
            'model_name': self._model_name,
            'projection_dim': self._projection_dim,
            'max_length': self._max_length,
            'dropout': self._dropout_rate,
            'freeze_encoder': self._freeze_encoder,
            'hidden_size': self._hidden_size,
        }

    def train(self, mode: bool = True):
        """
        Set training mode.

        If encoder is frozen, keeps it in eval mode even during training.
        """
        super().train(mode)
        if self._freeze_encoder:
            self.encoder.eval()
        return self

    def gradient_checkpointing_enable(self):
        """
        Enable gradient checkpointing for memory efficiency during fine-tuning.

        This trades compute for memory by recomputing activations during backward pass.
        Useful for fine-tuning large models with limited GPU memory.
        """
        if hasattr(self.encoder, 'gradient_checkpointing_enable'):
            self.encoder.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        else:
            logger.warning("Model does not support gradient checkpointing")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        if hasattr(self.encoder, 'gradient_checkpointing_disable'):
            self.encoder.gradient_checkpointing_disable()
            logger.info("Gradient checkpointing disabled")
