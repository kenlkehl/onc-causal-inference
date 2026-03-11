# cdt/models/conv1d_transformer_hybrid_extractor.py
"""Conv1D + Stride Downsample + Transformer Hybrid feature extractor.

This module processes full documents (up to 8192 tokens) without chunking by
combining dilated 1D convolutions with learned stride-based downsampling that
reduces sequence length by 2x per block. After 4 blocks, 8192 tokens become
512 positions, making transformer self-attention practical over the whole document.

Architecture:
    Raw Text (up to max_length tokens)
            |
    Word Embeddings -> LayerNorm -> Dropout
            |
    Linear projection (embedding_dim -> conv_dim)
            |
    [DilatedResidualBlock(dilation=2^i) + StrideDownsample(stride=2)] x num_blocks
        8192 -> 4096 -> 2048 -> 1024 -> 512 positions
            |
    Sinusoidal Positional Encoding
            |
    Transformer layers over downsampled positions
            |
    GatedAttentionPooling -> single vector (transformer_dim)
            |
    Optional NumericFeature merge
            |
    Output Projection -> Final Representation (projection_dim)

Key differences from conv_pool:
    - No chunking: processes full document as single sequence
    - StrideDownsample between each dilated block reduces length by stride
    - Convolutions see context across what would be chunk boundaries
    - Batch processing: all docs padded to same length
    - Attention mask propagated through stride downsampling

REQUIRES: fit_tokenizer(texts) before use
"""

import logging
import math
import re
from typing import Optional, List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cnn_extractor import WordTokenizer
from .conv_pool_extractor import DilatedResidualBlock
from .gated_attention_pooling import GatedAttentionPooling
from .hierarchical_transformer_extractor import InterpretableTransformerLayer
from .numeric_features import NumericFeatureVector


logger = logging.getLogger(__name__)


class StrideDownsample(nn.Module):
    """
    Learned stride-based downsampling for sequence length reduction.

    Uses a Conv1d with stride to downsample both the feature tensor and the
    attention mask. The mask is downsampled via max-pool so a position is valid
    if ANY input position in the stride window was valid.

    Args:
        channels: Number of input/output channels
        stride: Downsampling factor (default 2)
    """

    def __init__(self, channels: int, stride: int = 2):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv1d(
            channels, channels,
            kernel_size=stride,
            stride=stride
        )
        self.norm = nn.LayerNorm(channels)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Downsample tensor and mask.

        Args:
            x: (B, D, L) - feature tensor in channels-first format
            mask: (B, L) - attention mask (1=valid, 0=pad)

        Returns:
            x_down: (B, D, L//stride) - downsampled features
            mask_down: (B, L//stride) - downsampled mask
        """
        # Downsample features via learned Conv1d with stride
        x_down = self.conv(x)  # (B, D, L//stride)

        # LayerNorm (needs channels-last)
        x_down = x_down.transpose(1, 2)  # (B, L//stride, D)
        x_down = self.norm(x_down)
        x_down = x_down.transpose(1, 2)  # (B, D, L//stride)

        # Downsample mask via max-pool: position valid if any input was valid
        # Reshape mask for max_pool1d: (B, 1, L)
        mask_for_pool = mask.unsqueeze(1)
        mask_down = F.max_pool1d(
            mask_for_pool,
            kernel_size=self.stride,
            stride=self.stride
        ).squeeze(1)  # (B, L//stride)

        return x_down, mask_down


class Conv1dTransformerHybridExtractor(nn.Module):
    """
    Conv1D + Stride Downsample + Transformer Hybrid feature extractor.

    Processes full documents without chunking by combining dilated convolutions
    with stride-based downsampling to reduce sequence length before transformer
    self-attention.

    REQUIRES: fit_tokenizer(texts) before use

    Args:
        embedding_dim: Word embedding dimension
        conv_dim: Conv channel dimension
        kernel_size: Kernel size for dilated convolutions
        num_blocks: Number of dilated residual blocks (dilations: 1, 2, ..., 2^(N-1))
        conv_dropout: Dropout rate for conv blocks
        pool_stride: Downsampling factor per block (total: stride^num_blocks)
        max_length: Maximum document length in tokens
        transformer_layers: Number of transformer layers
        transformer_heads: Number of attention heads
        transformer_dim: Hidden dimension for transformer layers
        transformer_dropout: Dropout rate for transformer layers
        gated_attention_dim: Hidden dimension for gated attention pooling
        projection_dim: Final output dimension
        max_vocab_size: Maximum vocabulary size
        min_word_freq: Minimum word frequency for vocabulary inclusion
        device: PyTorch device
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        conv_dim: int = 256,
        kernel_size: int = 3,
        num_blocks: int = 4,
        conv_dropout: float = 0.1,
        pool_stride: int = 2,
        max_length: int = 8192,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        transformer_dim: int = 256,
        transformer_dropout: float = 0.1,
        gated_attention_dim: int = 128,
        projection_dim: int = 128,
        max_vocab_size: int = 50000,
        min_word_freq: int = 2,
        device: Optional[torch.device] = None,
        numeric_features_enabled: bool = False,
        numeric_embedding_dim: int = 32,
        numeric_magnitude_bins: int = 8,
        numeric_type_categories: int = 10
    ):
        super().__init__()

        self._device = device or torch.device('cpu')
        self._numeric_features_enabled = numeric_features_enabled
        self._numeric_embedding_dim = numeric_embedding_dim
        self._numeric_magnitude_bins = numeric_magnitude_bins
        self._numeric_type_categories = numeric_type_categories
        self._embedding_dim = embedding_dim
        self._conv_dim = conv_dim
        self._kernel_size = kernel_size
        self._num_blocks = num_blocks
        self._conv_dropout = conv_dropout
        self._pool_stride = pool_stride
        self._max_length = max_length
        self._transformer_layers = transformer_layers
        self._transformer_heads = transformer_heads
        self._transformer_dim = transformer_dim
        self._transformer_dropout = transformer_dropout
        self._gated_attention_dim = gated_attention_dim
        self._projection_dim = projection_dim
        self._max_vocab_size = max_vocab_size
        self._min_word_freq = min_word_freq

        # Compute transformer sequence length
        self._transformer_seq_len = max_length // (pool_stride ** num_blocks)
        if self._transformer_seq_len < 8:
            logger.warning(
                f"Transformer sequence length is very short ({self._transformer_seq_len}). "
                f"Consider reducing num_blocks ({num_blocks}) or pool_stride ({pool_stride}), "
                f"or increasing max_length ({max_length})."
            )
        if self._transformer_seq_len > 2048:
            logger.warning(
                f"Transformer sequence length is very long ({self._transformer_seq_len}). "
                f"Consider increasing num_blocks ({num_blocks}) or pool_stride ({pool_stride}), "
                f"or decreasing max_length ({max_length})."
            )

        # Minimum sequence length to survive all downsampling
        self._min_length = pool_stride ** num_blocks

        # Tokenizer (fit during fit_tokenizer)
        self._tokenizer = WordTokenizer(
            max_length=max_length,
            min_freq=min_word_freq,
            max_vocab_size=max_vocab_size
        )

        # Embedding layer (initialized after tokenizer is fitted)
        self._embedding = None

        # Embedding pre-processing
        self._embed_layer_norm = nn.LayerNorm(embedding_dim)
        self._embed_dropout = nn.Dropout(conv_dropout)

        # Project embedding dim to conv_dim
        self._embed_projection = nn.Linear(embedding_dim, conv_dim)

        # Dilated residual blocks with exponentially increasing dilation
        self._conv_blocks = nn.ModuleList([
            DilatedResidualBlock(
                channels=conv_dim,
                kernel_size=kernel_size,
                dilation=2 ** i,
                dropout=conv_dropout
            )
            for i in range(num_blocks)
        ])

        # Stride downsample modules (one per block)
        self._downsample_modules = nn.ModuleList([
            StrideDownsample(channels=conv_dim, stride=pool_stride)
            for _ in range(num_blocks)
        ])

        # Project conv output to transformer dim (if different)
        self._input_projection = nn.Linear(conv_dim, transformer_dim)

        # Positional encoding for downsampled positions
        self._register_positional_encoding()

        # Interpretable transformer layers
        self._transformer_layer_modules = nn.ModuleList([
            InterpretableTransformerLayer(
                d_model=transformer_dim,
                nhead=transformer_heads,
                dim_feedforward=transformer_dim * 4,
                dropout=transformer_dropout
            )
            for _ in range(transformer_layers)
        ])

        # Gated attention pooling for final aggregation
        self._gated_pooling = GatedAttentionPooling(
            hidden_dim=transformer_dim,
            attention_dim=gated_attention_dim
        )

        # Output projection
        self._output_projection = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim),
            nn.LayerNorm(transformer_dim),
            nn.GELU(),
            nn.Dropout(transformer_dropout),
            nn.Linear(transformer_dim, projection_dim),
            nn.LayerNorm(projection_dim)
        )

        # Numeric feature vector (concatenated to document embedding before output projection)
        self._numeric_feature_vector = None
        if numeric_features_enabled:
            self._numeric_feature_vector = NumericFeatureVector(
                num_magnitude_bins=numeric_magnitude_bins,
                num_type_categories=numeric_type_categories,
                output_dim=numeric_embedding_dim
            )
            # Merge layer: transformer_dim + numeric_dim -> transformer_dim
            self._numeric_merge = nn.Sequential(
                nn.Linear(transformer_dim + numeric_embedding_dim, transformer_dim),
                nn.LayerNorm(transformer_dim),
                nn.ReLU(),
            )

        self._initialized = False

        # Log info
        receptive_field = sum(2 * (kernel_size - 1) * (2 ** i) for i in range(num_blocks))
        dilations = [2 ** i for i in range(num_blocks)]
        total_downsample = pool_stride ** num_blocks

        logger.info(f"Conv1dTransformerHybridExtractor initialized:")
        logger.info(f"  Embedding dim: {embedding_dim}")
        logger.info(f"  Conv dim: {conv_dim}, kernel_size: {kernel_size}")
        logger.info(f"  Num blocks: {num_blocks}, dilations: {dilations}")
        logger.info(f"  Receptive field: {receptive_field} tokens")
        logger.info(f"  Pool stride: {pool_stride}, total downsample: {total_downsample}x")
        logger.info(f"  Max length: {max_length} -> {self._transformer_seq_len} transformer positions")
        logger.info(f"  Transformer layers: {transformer_layers}, dim: {transformer_dim}")
        logger.info(f"  Transformer heads: {transformer_heads}")
        logger.info(f"  Gated attention dim: {gated_attention_dim}")
        logger.info(f"  Projection dim: {projection_dim}")

    def _register_positional_encoding(self):
        """Create sinusoidal positional encoding for downsampled positions."""
        # Use generous max length to handle various configs
        max_len = max(self._transformer_seq_len, 2048)
        d_model = self._transformer_dim

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('_positional_encoding', pe)

    @property
    def output_dim(self) -> int:
        """Return the output dimension of this feature extractor."""
        return self._projection_dim

    @property
    def transformer_dim(self) -> int:
        """Return the transformer hidden dimension."""
        return self._transformer_dim

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return self._tokenizer.vocab_size

    def fit_tokenizer(self, texts: List[str]) -> 'Conv1dTransformerHybridExtractor':
        """
        Fit tokenizer on training texts and initialize embedding layer.

        MUST be called before using the model for training or inference.

        Args:
            texts: List of training text strings

        Returns:
            self for method chaining
        """
        self._tokenizer.fit(texts)

        # Initialize embedding layer with vocabulary size
        self._embedding = nn.Embedding(
            num_embeddings=self._tokenizer.vocab_size,
            embedding_dim=self._embedding_dim,
            padding_idx=self._tokenizer.pad_token
        )
        self._embedding.to(self._device)

        self._initialized = True
        logger.info(f"Tokenizer fitted: vocab size = {self._tokenizer.vocab_size}")

        return self

    def _tokenize_fn(self, text: str) -> List[str]:
        """Tokenize text into words using the tokenizer's method."""
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def _tokenize_and_pad(
        self,
        texts: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize texts and pad to max_length for batch processing.

        Short documents are padded. Long documents are truncated to max_length.
        Documents shorter than min_length are padded to min_length.

        Args:
            texts: List of document texts

        Returns:
            input_ids: (B, L) - padded token IDs
            attention_mask: (B, L) - mask (1=valid, 0=pad)
        """
        word_to_idx = self._tokenizer.word_to_id
        pad_token = self._tokenizer.pad_token

        all_ids = []
        for text in texts:
            tokens = self._tokenize_fn(text)
            ids = [word_to_idx.get(t, word_to_idx.get('<unk>', 1)) for t in tokens]

            # Truncate to max_length
            if len(ids) > self._max_length:
                ids = ids[:self._max_length]

            all_ids.append(ids)

        # Determine pad length: must be at least min_length and divisible by total downsample
        max_len_in_batch = max(len(ids) for ids in all_ids) if all_ids else self._min_length
        # Ensure at least min_length so stride convolutions don't fail
        pad_len = max(max_len_in_batch, self._min_length)
        # Round up to be divisible by total downsample factor
        total_ds = self._pool_stride ** self._num_blocks
        pad_len = ((pad_len + total_ds - 1) // total_ds) * total_ds

        padded = []
        masks = []
        for ids in all_ids:
            n = len(ids)
            pad_amount = pad_len - n
            padded.append(ids + [pad_token] * pad_amount)
            masks.append([1.0] * n + [0.0] * pad_amount)

        input_ids = torch.tensor(padded, dtype=torch.long, device=self._device)
        attention_mask = torch.tensor(masks, dtype=torch.float, device=self._device)

        return input_ids, attention_mask

    def _encode_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode a batch through conv blocks + stride downsample + transformer.

        Args:
            input_ids: (B, L) - token IDs
            attention_mask: (B, L) - attention mask

        Returns:
            pooled: (B, transformer_dim) - pooled document vectors
            transformer_output: (B, S, transformer_dim) - transformer positions
            attn_weights: (B, S) - gated attention weights
        """
        # Embed tokens
        embeddings = self._embedding(input_ids)  # (B, L, embedding_dim)
        embeddings = self._embed_layer_norm(embeddings)
        embeddings = self._embed_dropout(embeddings)

        # Project to conv_dim
        embeddings = self._embed_projection(embeddings)  # (B, L, conv_dim)

        # Transpose to channels-first for Conv1d: (B, conv_dim, L)
        x = embeddings.transpose(1, 2)
        mask = attention_mask

        # Zero out padded positions
        x = x * mask.unsqueeze(1)

        # Apply dilated residual blocks + stride downsample
        for conv_block, downsample in zip(self._conv_blocks, self._downsample_modules):
            x = conv_block(x)
            # Re-zero padded positions after each block
            x = x * mask.unsqueeze(1)
            # Stride downsample
            x, mask = downsample(x, mask)

        # Transpose back to (B, S, conv_dim)
        x = x.transpose(1, 2)

        # Project to transformer dim
        x = self._input_projection(x)  # (B, S, transformer_dim)

        # Add positional encoding
        seq_len = x.size(1)
        x = x + self._positional_encoding[:seq_len].to(self._device)

        # Run through transformer layers
        for layer in self._transformer_layer_modules:
            x, _ = layer(x, return_attention=False)

        # Gated attention pooling with mask
        pooled, attn_weights = self._gated_pooling(x, attention_mask=mask)  # (B, transformer_dim), (B, S)

        return pooled, x, attn_weights

    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        Extract features from texts.

        Args:
            texts: List of document texts

        Returns:
            Feature tensor of shape (batch_size, projection_dim)
        """
        if not self._initialized or self._embedding is None:
            raise RuntimeError("Must call fit_tokenizer() before forward()")

        # Tokenize and pad entire batch
        input_ids, attention_mask = self._tokenize_and_pad(texts)

        # Encode through conv + transformer
        pooled, _, _ = self._encode_batch(input_ids, attention_mask)

        # Numeric features (if enabled)
        if self._numeric_features_enabled and self._numeric_feature_vector is not None:
            numeric_feats = self._numeric_feature_vector(texts)  # (B, numeric_dim)
            pooled = self._numeric_merge(
                torch.cat([pooled, numeric_feats], dim=1)
            )

        # Output projection
        output = self._output_projection(pooled)  # (B, projection_dim)

        return output

    def forward_with_instances(
        self,
        texts: List[str]
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass returning document features AND position-level info for CLAM-style instance loss.

        Returns transformer-processed position embeddings (before gated pooling)
        and their gated attention weights, enabling instance-level supervision.

        Args:
            texts: List of document texts

        Returns:
            doc_features: (B, projection_dim) - document-level features
            position_embeddings_list: List of (S_i, transformer_dim) tensors per doc
            attention_weights_list: List of (S_i,) tensors - gated attention weights per doc
        """
        if not self._initialized or self._embedding is None:
            raise RuntimeError("Must call fit_tokenizer() before forward_with_instances()")

        # Tokenize and pad entire batch
        input_ids, attention_mask = self._tokenize_and_pad(texts)

        # Encode through conv + transformer
        pooled, transformer_output, attn_weights = self._encode_batch(input_ids, attention_mask)

        # Numeric features (if enabled)
        if self._numeric_features_enabled and self._numeric_feature_vector is not None:
            numeric_feats = self._numeric_feature_vector(texts)  # (B, numeric_dim)
            pooled = self._numeric_merge(
                torch.cat([pooled, numeric_feats], dim=1)
            )

        # Output projection
        doc_features = self._output_projection(pooled)  # (B, projection_dim)

        # Compute downsampled mask to determine valid positions per doc
        _, ds_mask = self._tokenize_and_pad(texts)
        # Re-derive the downsampled mask
        mask = attention_mask
        for downsample in self._downsample_modules:
            mask_for_pool = mask.unsqueeze(1)
            mask = F.max_pool1d(
                mask_for_pool,
                kernel_size=downsample.stride,
                stride=downsample.stride
            ).squeeze(1)

        # Split per-document
        position_embeddings_list = []
        attention_weights_list = []
        for i in range(len(texts)):
            # Get valid positions for this document
            valid_mask = mask[i] > 0
            valid_embs = transformer_output[i][valid_mask]  # (S_valid, transformer_dim)
            valid_weights = attn_weights[i][valid_mask]  # (S_valid,)

            position_embeddings_list.append(valid_embs)
            attention_weights_list.append(valid_weights)

        return doc_features, position_embeddings_list, attention_weights_list

    def get_state(self) -> Dict[str, Any]:
        """
        Get extractor state for checkpoint saving.

        Returns:
            Dictionary containing configuration for reconstruction
        """
        return {
            'embedding_dim': self._embedding_dim,
            'conv_dim': self._conv_dim,
            'kernel_size': self._kernel_size,
            'num_blocks': self._num_blocks,
            'conv_dropout': self._conv_dropout,
            'pool_stride': self._pool_stride,
            'max_length': self._max_length,
            'transformer_layers': self._transformer_layers,
            'transformer_heads': self._transformer_heads,
            'transformer_dim': self._transformer_dim,
            'transformer_dropout': self._transformer_dropout,
            'gated_attention_dim': self._gated_attention_dim,
            'projection_dim': self._projection_dim,
            'max_vocab_size': self._max_vocab_size,
            'min_word_freq': self._min_word_freq,
        }

    def interpret_attention(
        self,
        texts: List[str],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get human-readable interpretation of attention weights.

        Since there are no chunks, this shows which downsampled positions
        (each representing pool_stride^num_blocks original tokens) are
        most attended.

        Args:
            texts: List of document texts
            top_k: Number of top-attended positions to show

        Returns:
            List of dicts per document with attention interpretations
        """
        if not self._initialized or self._embedding is None:
            raise RuntimeError("Must call fit_tokenizer() before interpret_attention()")

        interpretations = []

        with torch.no_grad():
            input_ids, attention_mask = self._tokenize_and_pad(texts)
            _, transformer_output, attn_weights = self._encode_batch(input_ids, attention_mask)

            # Compute downsampled mask
            mask = attention_mask
            for downsample in self._downsample_modules:
                mask_for_pool = mask.unsqueeze(1)
                mask = F.max_pool1d(
                    mask_for_pool,
                    kernel_size=downsample.stride,
                    stride=downsample.stride
                ).squeeze(1)

            total_ds = self._pool_stride ** self._num_blocks
            id_to_word = self._tokenizer.id_to_word

            for i, text in enumerate(texts):
                tokens = self._tokenize_fn(text)
                valid_mask = mask[i] > 0
                num_positions = valid_mask.sum().item()
                weights = attn_weights[i][valid_mask].cpu()

                k_actual = min(top_k, num_positions)
                if k_actual == 0:
                    interpretations.append({
                        'num_positions': 0,
                        'position_attention': [],
                        'top_positions': [],
                    })
                    continue

                top_vals, top_indices = torch.topk(weights, k_actual)

                # Map positions back to approximate original token spans
                top_positions = []
                for val, idx in zip(top_vals.tolist(), top_indices.tolist()):
                    start_tok = idx * total_ds
                    end_tok = min(start_tok + total_ds, len(tokens))
                    span_tokens = tokens[start_tok:end_tok]
                    text_preview = ' '.join(span_tokens)
                    if len(text_preview) > 200:
                        text_preview = text_preview[:200] + '...'
                    top_positions.append({
                        'position_idx': int(idx),
                        'attention': float(val),
                        'token_range': [int(start_tok), int(end_tok)],
                        'text_preview': text_preview
                    })

                interpretations.append({
                    'num_positions': int(num_positions),
                    'position_attention': weights.tolist(),
                    'top_positions': top_positions,
                })

        return interpretations

    def get_attention_weights(self, texts: List[str]) -> Dict[str, Any]:
        """
        Get raw attention weights for visualization.

        Args:
            texts: List of document texts

        Returns:
            Dictionary with interpretations and model metadata
        """
        interpretations = self.interpret_attention(texts, top_k=100)
        return {
            'interpretations': interpretations,
            'transformer_layers': self._transformer_layers,
            'transformer_heads': self._transformer_heads,
            'conv_dim': self._conv_dim,
            'num_blocks': self._num_blocks,
            'kernel_size': self._kernel_size,
            'pool_stride': self._pool_stride,
            'max_length': self._max_length,
        }

    def to(self, device):
        """Override to track device properly."""
        self._device = device if isinstance(device, torch.device) else torch.device(device)

        if self._embedding is not None:
            self._embedding = self._embedding.to(self._device)

        return super().to(device)
