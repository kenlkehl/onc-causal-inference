"""Hierarchical Transformer feature extractor using chunk-level BERT + transformer pooling.

This module implements a simple hierarchical approach for extracting features from long
clinical text:

1. Split text into overlapping token chunks
2. Encode each chunk with a tiny BERT (e.g., prajjwal1/bert-tiny), taking the [CLS] token
3. Apply transformer layer(s) on top to pool chunk embeddings into a final representation

This bypasses the latent confounder mechanism entirely - just straightforward chunk
encoding with transformer pooling.

Architecture:
    Long Clinical Text
            |
    Split into Overlapping Token Chunks (C chunks)
            |
    Tiny BERT per Chunk -> [CLS] token (C x hidden_dim)
            |
    Transformer Layer(s) with learnable [POOL] token
            |
    Final Representation (D,) -> DragonNet/RLearner
"""

import logging
import math
from typing import Optional, List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .chunking import split_into_chunks_hf
from .numeric_features import NumericFeatureVector


logger = logging.getLogger(__name__)


class InterpretableTransformerLayer(nn.Module):
    """Transformer layer that can return attention weights for interpretability."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with optional attention weight extraction.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            return_attention: Whether to return attention weights

        Returns:
            output: Transformed tensor of shape (batch, seq_len, d_model)
            attn_weights: Optional attention weights of shape (batch, seq_len, seq_len)
        """
        # Self-attention with optional weights
        attn_output, attn_weights = self.self_attn(x, x, x, need_weights=return_attention)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.norm2(x + self.dropout(ff_output))

        return x, attn_weights


class HierarchicalTransformerExtractor(nn.Module):
    """
    Hierarchical transformer feature extractor.

    Architecture:
    1. Split text into overlapping token chunks
    2. Encode each chunk with tiny BERT -> [CLS] token
    3. Apply transformer layer(s) with learnable [POOL] token
    4. Output [POOL] representation for causal head

    This is simpler than ConfounderExtractor - no latent confounders,
    no sparse attention, just straightforward hierarchical encoding.

    Args:
        sentence_encoder_model: HuggingFace model name for chunk encoding (default: prajjwal1/bert-tiny)
        freeze_sentence_encoder: Whether to freeze the encoder weights
        max_chunks: Maximum number of chunks to process per document
        chunk_size: Number of tokens per chunk
        chunk_overlap: Number of overlapping tokens between chunks
        num_transformer_layers: Number of transformer layers for pooling
        num_attention_heads: Number of attention heads in transformer layers
        transformer_dim: Hidden dimension for transformer layers
        transformer_dropout: Dropout rate for transformer layers
        projection_dim: Final output dimension
        device: PyTorch device
    """

    def __init__(
        self,
        sentence_encoder_model: str = "prajjwal1/bert-tiny",
        freeze_sentence_encoder: bool = True,
        max_chunks: int = 100,
        chunk_size: int = 128,
        chunk_overlap: int = 32,
        num_transformer_layers: int = 2,
        num_attention_heads: int = 4,
        transformer_dim: int = 256,
        transformer_dropout: float = 0.1,
        projection_dim: int = 128,
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
        self._sentence_encoder_model = sentence_encoder_model
        self._freeze = freeze_sentence_encoder
        self._max_chunks = max_chunks
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._num_layers = num_transformer_layers
        self._num_heads = num_attention_heads
        self._transformer_dim = transformer_dim
        self._dropout = transformer_dropout
        self._projection_dim = projection_dim

        # Lazy initialization
        self._sentence_encoder = None
        self._tokenizer = None
        self._sentence_dim = None
        self._input_projection = None
        self._pool_token = None
        self._transformer_layers = None
        self._output_projection = None
        self._initialized = False

        logger.info(f"HierarchicalTransformerExtractor initialized:")
        logger.info(f"  Chunk encoder: {sentence_encoder_model}")
        logger.info(f"  Freeze encoder: {freeze_sentence_encoder}")
        logger.info(f"  Max chunks: {max_chunks}, chunk_size: {chunk_size}, overlap: {chunk_overlap}")
        logger.info(f"  Transformer layers: {num_transformer_layers}")
        logger.info(f"  Transformer dim: {transformer_dim}")
        logger.info(f"  Attention heads: {num_attention_heads}")
        logger.info(f"  Projection dim: {projection_dim}")

    def _ensure_initialized(self):
        """Lazily initialize components."""
        if self._initialized:
            return

        from transformers import AutoModel, AutoTokenizer

        logger.info(f"Loading sentence encoder: {self._sentence_encoder_model}")
        self._tokenizer = AutoTokenizer.from_pretrained(self._sentence_encoder_model)
        self._sentence_encoder = AutoModel.from_pretrained(self._sentence_encoder_model)
        self._sentence_encoder = self._sentence_encoder.to(self._device)
        self._sentence_dim = self._sentence_encoder.config.hidden_size
        logger.info(f"  Sentence encoder dim: {self._sentence_dim}")

        if self._freeze:
            for param in self._sentence_encoder.parameters():
                param.requires_grad = False
            logger.info("  Sentence encoder frozen")

        # Input projection: sentence_dim -> transformer_dim
        self._input_projection = nn.Linear(self._sentence_dim, self._transformer_dim).to(self._device)

        # Learnable [POOL] token
        self._pool_token = nn.Parameter(
            torch.randn(1, self._transformer_dim, device=self._device) * 0.02
        )

        # Positional encoding (sinusoidal)
        self._register_positional_encoding()

        # Interpretable transformer layers (custom to allow attention extraction)
        self._transformer_layers = nn.ModuleList([
            InterpretableTransformerLayer(
                d_model=self._transformer_dim,
                nhead=self._num_heads,
                dim_feedforward=self._transformer_dim * 4,
                dropout=self._dropout
            )
            for _ in range(self._num_layers)
        ]).to(self._device)

        # Output projection
        self._output_projection = nn.Sequential(
            nn.Linear(self._transformer_dim, self._transformer_dim),
            nn.LayerNorm(self._transformer_dim),
            nn.GELU(),
            nn.Dropout(self._dropout),
            nn.Linear(self._transformer_dim, self._projection_dim),
            nn.LayerNorm(self._projection_dim)
        ).to(self._device)

        # Numeric feature vector (merged after [POOL] extraction, before output projection)
        self._numeric_feature_vector = None
        self._numeric_merge = None
        if self._numeric_features_enabled:
            self._numeric_feature_vector = NumericFeatureVector(
                num_magnitude_bins=self._numeric_magnitude_bins,
                num_type_categories=self._numeric_type_categories,
                output_dim=self._numeric_embedding_dim
            ).to(self._device)
            self._numeric_merge = nn.Sequential(
                nn.Linear(self._transformer_dim + self._numeric_embedding_dim, self._transformer_dim),
                nn.LayerNorm(self._transformer_dim),
                nn.ReLU(),
            ).to(self._device)

        self._initialized = True
        logger.info("HierarchicalTransformerExtractor initialization complete")

    def _register_positional_encoding(self):
        """Create sinusoidal positional encoding."""
        max_len = self._max_chunks + 1  # +1 for pool token
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

    def _encode_chunks_batch(self, chunks: List[str]) -> torch.Tensor:
        """
        Encode chunks with BERT, returning [CLS] tokens.

        Args:
            chunks: List of chunk strings

        Returns:
            Tensor of shape (num_chunks, sentence_dim) containing [CLS] embeddings
        """
        if not chunks:
            self._ensure_initialized()
            return torch.zeros(0, self._sentence_dim, device=self._device)

        encoded = self._tokenizer(
            chunks,
            padding=True,
            truncation=True,
            max_length=self._chunk_size,
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].to(self._device)
        attention_mask = encoded['attention_mask'].to(self._device)

        with torch.set_grad_enabled(not self._freeze):
            outputs = self._sentence_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        # [CLS] token at position 0
        return outputs.last_hidden_state[:, 0, :]

    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        Extract features from texts.

        Args:
            texts: List of document texts

        Returns:
            Feature tensor of shape (batch_size, projection_dim)
        """
        self._ensure_initialized()
        batch_size = len(texts)
        batch_outputs = []

        for text in texts:
            # 1. Split into overlapping token chunks
            chunks = split_into_chunks_hf(
                text,
                self._tokenizer,
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
                max_chunks=self._max_chunks
            )
            if not chunks:
                chunks = [text[:500]]  # Fallback for short/malformed text

            # 2. Encode chunks with BERT
            chunk_embeddings = self._encode_chunks_batch(chunks)  # (C, sentence_dim)

            # 3. Project to transformer dim
            chunk_embeddings = self._input_projection(chunk_embeddings)  # (C, transformer_dim)

            # 4. Prepend [POOL] token
            sequence = torch.cat([self._pool_token, chunk_embeddings], dim=0)  # (C+1, transformer_dim)

            # 5. Add positional encoding
            seq_len = sequence.size(0)
            sequence = sequence + self._positional_encoding[:seq_len].to(self._device)

            # 6. Run through transformer layers
            sequence = sequence.unsqueeze(0)  # (1, C+1, transformer_dim)
            for layer in self._transformer_layers:
                sequence, _ = layer(sequence, return_attention=False)

            # 7. Extract [POOL] output (position 0)
            pool_output = sequence[0, 0, :]  # (transformer_dim,)

            # 7.5. Add numeric features if enabled
            if self._numeric_features_enabled and self._numeric_feature_vector is not None:
                numeric_feats = self._numeric_feature_vector([text])  # (1, numeric_dim)
                pool_output = self._numeric_merge(
                    torch.cat([pool_output.unsqueeze(0), numeric_feats], dim=1)
                ).squeeze(0)

            batch_outputs.append(pool_output)

        # Stack batch
        batch_outputs = torch.stack(batch_outputs)  # (B, transformer_dim)

        # 8. Output projection
        features = self._output_projection(batch_outputs)  # (B, projection_dim)

        return features

    def forward_with_instances(
        self,
        texts: List[str]
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass returning document features AND chunk-level info for CLAM-style instance loss.

        This method returns transformer-processed chunk embeddings (before output projection)
        and the [POOL] token attention weights to chunks, enabling instance-level supervision
        on top-attended chunks.

        Args:
            texts: List of document texts

        Returns:
            doc_features: (B, projection_dim) - document-level features
            chunk_embeddings_list: List of (C_i, transformer_dim) tensors per doc
            attention_weights_list: List of (C_i,) tensors - [POOL] attention weights per doc
        """
        self._ensure_initialized()
        batch_outputs = []
        chunk_embeddings_list = []
        attention_weights_list = []

        for text in texts:
            # 1. Split into overlapping token chunks
            chunks = split_into_chunks_hf(
                text,
                self._tokenizer,
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
                max_chunks=self._max_chunks
            )
            if not chunks:
                chunks = [text[:500]]  # Fallback for short/malformed text

            # 2. Encode chunks with BERT
            chunk_embeddings = self._encode_chunks_batch(chunks)  # (C, sentence_dim)

            if chunk_embeddings.size(0) == 0:
                # Fallback for empty text
                batch_outputs.append(
                    torch.zeros(self._projection_dim, device=self._device)
                )
                chunk_embeddings_list.append(
                    torch.zeros(0, self._transformer_dim, device=self._device)
                )
                attention_weights_list.append(
                    torch.zeros(0, device=self._device)
                )
                continue

            # 3. Project to transformer dim
            chunk_embeddings = self._input_projection(chunk_embeddings)  # (C, transformer_dim)

            # 4. Prepend [POOL] token
            sequence = torch.cat([self._pool_token, chunk_embeddings], dim=0)  # (C+1, transformer_dim)

            # 5. Add positional encoding
            seq_len = sequence.size(0)
            sequence = sequence + self._positional_encoding[:seq_len].to(self._device)

            # 6. Run through transformer layers, collecting attention from last layer
            sequence = sequence.unsqueeze(0)  # (1, C+1, transformer_dim)
            attn_weights = None
            for layer in self._transformer_layers:
                sequence, attn_weights = layer(sequence, return_attention=True)

            # 7. Extract [POOL] output (position 0)
            pool_output = sequence[0, 0, :]  # (transformer_dim,)

            # 8. Extract chunk embeddings (positions 1:) after transformer
            transformer_chunk_embs = sequence[0, 1:, :]  # (C, transformer_dim)

            # 9. Extract [POOL] attention to chunks (from last layer)
            # attn_weights shape: (1, seq_len, seq_len)
            # We want attention FROM [POOL] (position 0) TO all chunks (positions 1:)
            if attn_weights is not None:
                pool_attention = attn_weights[0, 0, 1:]  # (C,)
                # Normalize to sum to 1
                pool_attention = pool_attention / (pool_attention.sum() + 1e-9)
            else:
                # Fallback: uniform attention
                num_chunks = transformer_chunk_embs.size(0)
                pool_attention = torch.ones(num_chunks, device=self._device) / num_chunks

            # Output projection for pool token
            output = self._output_projection(pool_output.unsqueeze(0)).squeeze(0)  # (projection_dim,)
            batch_outputs.append(output)

            # Store chunk-level info for instance loss
            chunk_embeddings_list.append(transformer_chunk_embs)
            attention_weights_list.append(pool_attention)

        # Stack batch
        doc_features = torch.stack(batch_outputs)  # (B, projection_dim)

        return doc_features, chunk_embeddings_list, attention_weights_list

    def init_extractor(self, texts: List[str]) -> 'HierarchicalTransformerExtractor':
        """
        Initialize the feature extractor (triggers lazy initialization).

        For HierarchicalTransformerExtractor, this loads the pretrained sentence
        encoder and initializes the transformer pooling layers. The texts argument
        is not used since we use pretrained tokenizers.

        Args:
            texts: List of training text strings (not used, kept for API compatibility)

        Returns:
            self for method chaining
        """
        self._ensure_initialized()
        return self

    def fit_tokenizer(self, texts: List[str]) -> 'HierarchicalTransformerExtractor':
        """Alias for init_extractor() for backward compatibility."""
        return self.init_extractor(texts)

    def to(self, device):
        """Override to track device properly."""
        self._device = device if isinstance(device, torch.device) else torch.device(device)

        if self._sentence_encoder is not None:
            self._sentence_encoder = self._sentence_encoder.to(self._device)
        if self._input_projection is not None:
            self._input_projection = self._input_projection.to(self._device)
        if self._transformer_layers is not None:
            self._transformer_layers = self._transformer_layers.to(self._device)
        if self._output_projection is not None:
            self._output_projection = self._output_projection.to(self._device)
        if hasattr(self, '_positional_encoding') and self._positional_encoding is not None:
            self._positional_encoding = self._positional_encoding.to(self._device)

        return super().to(device)

    def get_state(self) -> Dict[str, Any]:
        """
        Get extractor state for checkpoint saving.

        Returns:
            Dictionary containing configuration for reconstruction
        """
        return {
            'sentence_encoder_model': self._sentence_encoder_model,
            'freeze_sentence_encoder': self._freeze,
            'max_chunks': self._max_chunks,
            'chunk_size': self._chunk_size,
            'chunk_overlap': self._chunk_overlap,
            'num_transformer_layers': self._num_layers,
            'num_attention_heads': self._num_heads,
            'transformer_dim': self._transformer_dim,
            'transformer_dropout': self._dropout,
            'projection_dim': self._projection_dim,
        }

    def interpret_attention(
        self,
        texts: List[str],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get human-readable interpretation of chunk attention.

        This extracts attention weights from the [POOL] token to each chunk,
        showing which chunks contribute most to the final representation.

        Args:
            texts: List of document texts
            top_k: Number of top-attended chunks to show

        Returns:
            List of dicts per document with attention interpretations:
            - 'chunks': List of chunk strings
            - 'chunk_attention': Attention weights from [POOL] to each chunk
            - 'top_chunks': Top-k chunks by attention weight
        """
        self._ensure_initialized()
        interpretations = []

        with torch.no_grad():
            for text in texts:
                chunks = split_into_chunks_hf(
                    text,
                    self._tokenizer,
                    chunk_size=self._chunk_size,
                    chunk_overlap=self._chunk_overlap,
                    max_chunks=self._max_chunks
                )
                if not chunks:
                    chunks = [text[:500]]

                # Encode chunks
                chunk_embeddings = self._encode_chunks_batch(chunks)
                chunk_embeddings = self._input_projection(chunk_embeddings)

                # Prepend [POOL] and add positional encoding
                sequence = torch.cat([self._pool_token, chunk_embeddings], dim=0)
                seq_len = sequence.size(0)
                sequence = sequence + self._positional_encoding[:seq_len].to(self._device)
                sequence = sequence.unsqueeze(0)

                # Run through transformer layers, collecting attention from last layer
                attn_weights = None
                for layer in self._transformer_layers:
                    sequence, attn_weights = layer(sequence, return_attention=True)

                # attn_weights shape: (1, seq_len, seq_len)
                # We want attention FROM [POOL] (position 0) TO all chunks
                if attn_weights is not None and len(chunks) > 0:
                    pool_attention = attn_weights[0, 0, 1:].cpu()  # Skip position 0 (self-attention)

                    # Normalize
                    pool_attention = pool_attention / (pool_attention.sum() + 1e-9)

                    # Get top-k
                    k_actual = min(top_k, len(chunks))
                    top_vals, top_indices = torch.topk(pool_attention, k_actual)

                    top_chunks = [
                        {
                            'chunk': chunks[idx],
                            'attention': val.item(),
                            'idx': int(idx)
                        }
                        for val, idx in zip(top_vals, top_indices)
                    ]

                    interpretations.append({
                        'chunks': chunks,
                        'chunk_attention': pool_attention.tolist(),
                        'top_chunks': top_chunks
                    })
                else:
                    interpretations.append({
                        'chunks': chunks,
                        'chunk_attention': [],
                        'top_chunks': []
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
        interpretations = self.interpret_attention(texts, top_k=self._max_chunks)
        return {
            'interpretations': interpretations,
            'num_layers': self._num_layers,
            'num_heads': self._num_heads,
            'model': self._sentence_encoder_model
        }
