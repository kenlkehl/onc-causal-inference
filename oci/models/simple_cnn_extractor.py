# oci/models/simple_cnn_extractor.py
"""Simple 1D CNN feature extractor applied to whole-document text.

Architecture:
    Raw text
      -> LearnedTokenizer (truncated to max_length)
      -> nn.Embedding
      -> Dilated residual CNN stack
      -> GatedAttentionPooling -> document vector
      -> Projection MLP -> (batch, output_dim)

Trains from scratch. Requires fit_tokenizer() before training.
"""

import logging
from typing import Dict, List, Any, Optional

import torch
import torch.nn as nn

from .gated_attention_pooling import GatedAttentionPooling
from .learned_tokenizer import LearnedTokenizer

logger = logging.getLogger(__name__)


class DilatedConvStack(nn.Module):
    """Stack of dilated 1D convolutions with residual connections.

    Each block: Conv1d(dilation=2^i) -> BatchNorm -> GELU -> residual add.
    Dilation pattern [1, 2, 4, 8, ...] gives exponentially growing receptive field.
    """

    def __init__(
        self,
        input_dim: int,
        conv_dim: int,
        kernel_size: int,
        num_blocks: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Input projection from embedding dim to conv dim
        self.input_proj = nn.Conv1d(input_dim, conv_dim, kernel_size=1)

        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation // 2
            self.blocks.append(nn.Sequential(
                nn.Conv1d(conv_dim, conv_dim, kernel_size, dilation=dilation, padding=padding),
                nn.BatchNorm1d(conv_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len, input_dim) or (batch*chunks, seq_len, input_dim)

        Returns:
            (batch, seq_len, conv_dim)
        """
        # Conv1d expects (batch, channels, seq_len)
        x = x.transpose(1, 2)
        x = self.input_proj(x)

        for block in self.blocks:
            residual = x
            out = block(x)
            # Trim if padding caused length mismatch
            if out.shape[-1] != residual.shape[-1]:
                out = out[..., :residual.shape[-1]]
            x = residual + out

        return x.transpose(1, 2)  # back to (batch, seq_len, conv_dim)


class SimpleCNNExtractor(nn.Module):
    """Simple 1D CNN on whole text with gated attention pooling.

    Args:
        embedding_dim: Word embedding dimension.
        conv_dim: CNN hidden dimension.
        kernel_size: Convolution kernel size.
        num_conv_blocks: Number of dilated residual conv blocks.
        max_length: Maximum token sequence length.
        vocab_size: Vocabulary size (set after fit_tokenizer).
        gated_attention_dim: Hidden dim for gated attention pooling.
        projection_dim: Final output dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        conv_dim: int = 256,
        kernel_size: int = 5,
        num_conv_blocks: int = 4,
        max_length: int = 10000,
        vocab_size: int = 50000,
        gated_attention_dim: int = 128,
        projection_dim: int = 128,
        dropout: float = 0.1,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self._device = device or torch.device('cpu')
        self._embedding_dim = embedding_dim
        self._conv_dim = conv_dim
        self._kernel_size = kernel_size
        self._num_conv_blocks = num_conv_blocks
        self._max_length = max_length
        self._vocab_size = vocab_size
        self._gated_attention_dim = gated_attention_dim
        self._projection_dim = projection_dim
        self._dropout = dropout

        self._tokenizer = LearnedTokenizer()

        # Embedding (initialized with placeholder vocab_size, resized after fit_tokenizer)
        self._embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # CNN stack
        self._cnn = DilatedConvStack(
            input_dim=embedding_dim,
            conv_dim=conv_dim,
            kernel_size=kernel_size,
            num_blocks=num_conv_blocks,
            dropout=dropout,
        )

        # Gated attention pooling
        self._pooling = GatedAttentionPooling(
            hidden_dim=conv_dim,
            attention_dim=gated_attention_dim,
        )

        # Projection MLP
        self._output_dim = projection_dim
        self._projection = nn.Sequential(
            nn.Linear(conv_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, projection_dim),
            nn.LayerNorm(projection_dim),
        )

        logger.info(
            f"SimpleCNNExtractor initialized: "
            f"embedding_dim={embedding_dim}, conv_dim={conv_dim}, "
            f"kernel_size={kernel_size}, num_blocks={num_conv_blocks}, "
            f"max_length={max_length}, projection_dim={projection_dim}"
        )

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def fit_tokenizer(self, texts: List[str], min_freq: int = 2) -> None:
        """Build vocabulary from training texts.

        Args:
            texts: Training text corpus.
            min_freq: Minimum word frequency to include.
        """
        self._tokenizer.fit(texts, vocab_size=self._vocab_size, min_freq=min_freq)
        actual_vocab = self._tokenizer.vocab_size
        # Resize embedding if actual vocab differs from initial
        if actual_vocab != self._embedding.num_embeddings:
            old_weight = self._embedding.weight.data
            self._embedding = nn.Embedding(actual_vocab, self._embedding_dim, padding_idx=0)
            # Copy existing weights for overlap
            copy_size = min(old_weight.shape[0], actual_vocab)
            self._embedding.weight.data[:copy_size] = old_weight[:copy_size]
            self._embedding = self._embedding.to(self._device)
        logger.info(f"SimpleCNNExtractor: tokenizer fitted, vocab_size={actual_vocab}")

    def forward(self, texts_or_batch) -> torch.Tensor:
        """Extract features from raw texts.

        Args:
            texts_or_batch: List[str] of document texts, or dict (ignored, uses texts).

        Returns:
            Feature tensor of shape (batch_size, output_dim)
        """
        if isinstance(texts_or_batch, dict):
            texts = texts_or_batch.get('texts', [])
        else:
            texts = texts_or_batch

        if not self._tokenizer.is_fitted:
            raise RuntimeError(
                "Tokenizer not fitted. Call fit_tokenizer() before forward()."
            )

        input_ids, attention_mask = self._tokenizer.encode_batch(
            texts, max_length=self._max_length
        )
        input_ids = input_ids.to(self._device)
        attention_mask = attention_mask.to(self._device)

        # Embed -> CNN -> pool -> project
        embedded = self._embedding(input_ids)  # (B, seq_len, embedding_dim)
        encoded = self._cnn(embedded)  # (B, seq_len, conv_dim)
        pooled, self._last_attention_weights = self._pooling(
            encoded, attention_mask=attention_mask
        )  # (B, conv_dim)
        features = self._projection(pooled)  # (B, projection_dim)

        return features

    def get_state(self) -> Dict[str, Any]:
        return {
            'extractor_type': 'simple_cnn',
            'embedding_dim': self._embedding_dim,
            'conv_dim': self._conv_dim,
            'kernel_size': self._kernel_size,
            'num_conv_blocks': self._num_conv_blocks,
            'max_length': self._max_length,
            'vocab_size': self._vocab_size,
            'gated_attention_dim': self._gated_attention_dim,
            'projection_dim': self._projection_dim,
            'dropout': self._dropout,
            'output_dim': self._output_dim,
            'tokenizer_state': self._tokenizer.get_state() if self._tokenizer.is_fitted else None,
        }

    def get_num_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable, 'frozen': total - trainable}

    def to(self, device):
        self._device = device if isinstance(device, torch.device) else torch.device(device)
        return super().to(device)
