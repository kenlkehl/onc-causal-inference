# cdt/models/gru_pool_extractor.py
"""GRU + Transformer + Gated Attention Pooling feature extractor.

This module implements a hierarchical approach for extracting features from long
clinical text that combines:

1. Learned word embeddings + BiGRU with simple attention pooling for chunk encoding
2. Standard transformer layers for cross-chunk context (chunks attend to each other)
3. Gated attention pooling (tanh x sigmoid) for final document aggregation

Unlike gru_transformer_mil which uses K confounder queries and task-specific weighting,
this extractor produces a single feature vector via gated attention pooling over
transformer-processed chunk embeddings.

Architecture:
    Long Clinical Text
            |
    Split into Overlapping Token Chunks (C chunks)
            |
    [Per Chunk - Shared BiGRU]
    Word Embeddings -> BiGRU -> Attention Pooling (C x gru_output_dim)
            |
    Project to transformer_dim (C x transformer_dim)
            |
    Add Positional Encoding
            |
    Transformer Layer(s) - chunks attend to each other (cross-chunk context)
            |
    Gated Attention Pooling (tanh x sigmoid) -> single vector
            |
    Output Projection -> Final Representation (projection_dim)

REQUIRES: fit_tokenizer(texts) before use
"""

import logging
import math
from typing import Optional, List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cnn_extractor import WordTokenizer
from .gru_extractor import AttentionPooling
from .hierarchical_transformer_extractor import InterpretableTransformerLayer
from .chunking import split_into_chunks_vocab
from .numeric_features import NumericFeatureVector


logger = logging.getLogger(__name__)


class GatedAttentionPooling(nn.Module):
    """
    Gated attention pooling (tanh x sigmoid) for aggregating sequences.

    Used for final document aggregation (chunks -> document vector).
    The gating mechanism allows learning which chunks to suppress (via sigmoid gate)
    while extracting content (via tanh).

    Formula:
        g = tanh(V(h)) * sigmoid(U(h))  # Gated features
        g' = LayerNorm(g)
        s = v(W(g'))  # Attention scores
        w = softmax(s)  # Attention weights
        output = sum(w * h)  # Weighted sum of original embeddings
    """

    def __init__(self, hidden_dim: int, attention_dim: Optional[int] = None):
        """
        Initialize gated attention pooling.

        Args:
            hidden_dim: Dimension of input hidden states
            attention_dim: Dimension of attention hidden layer (default: hidden_dim)
        """
        super().__init__()
        attention_dim = attention_dim or hidden_dim

        # Gating transforms
        self.V = nn.Linear(hidden_dim, attention_dim)  # tanh branch (content)
        self.U = nn.Linear(hidden_dim, attention_dim)  # sigmoid branch (gate)

        # Attention computation
        self.W = nn.Linear(attention_dim, attention_dim, bias=False)
        self.v = nn.Linear(attention_dim, 1, bias=False)

        self.layer_norm = nn.LayerNorm(attention_dim)

    def forward(
        self,
        chunk_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply gated attention pooling.

        Args:
            chunk_embeddings: (C, D) or (B, C, D) - transformed chunk embeddings
            attention_mask: Optional mask for valid chunks (1 = valid, 0 = padding)

        Returns:
            pooled: (D,) or (B, D) - document representation
            weights: (C,) or (B, C) - attention weights over chunks
        """
        # Gated features: g = tanh(V(h)) * sigmoid(U(h))
        tanh_branch = torch.tanh(self.V(chunk_embeddings))
        sigmoid_branch = torch.sigmoid(self.U(chunk_embeddings))
        gated = tanh_branch * sigmoid_branch
        gated = self.layer_norm(gated)

        # Attention scores
        scores = self.v(self.W(gated)).squeeze(-1)

        # Apply mask if provided
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -1e9)

        # Softmax and weighted sum
        weights = F.softmax(scores, dim=-1)

        # Handle both single sample and batch
        if chunk_embeddings.dim() == 2:
            pooled = (weights.unsqueeze(-1) * chunk_embeddings).sum(dim=0)
        else:
            pooled = torch.bmm(weights.unsqueeze(1), chunk_embeddings).squeeze(1)

        return pooled, weights


class GRUPoolExtractor(nn.Module):
    """
    BiGRU + transformer + gated pooling feature extractor.

    Combines:
    - Learned word embeddings + BiGRU + attention for chunk encoding
    - Transformer layers for cross-chunk context
    - Gated attention pooling for final document aggregation

    This produces a single feature vector (not task-specific vectors like gru_transformer_mil).

    REQUIRES: fit_tokenizer(texts) before use

    Args:
        embedding_dim: Word embedding dimension
        gru_hidden_dim: GRU hidden state dimension per direction
        gru_num_layers: Number of stacked GRU layers
        gru_bidirectional: Use bidirectional GRU
        gru_dropout: Dropout rate for GRU
        max_chunks: Maximum number of chunks to process per document
        chunk_size: Number of tokens per chunk
        chunk_overlap: Number of overlapping tokens between chunks
        transformer_layers: Number of transformer layers for cross-chunk processing
        transformer_heads: Number of attention heads in transformer
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
        gru_hidden_dim: int = 128,
        gru_num_layers: int = 1,
        gru_bidirectional: bool = True,
        gru_dropout: float = 0.1,
        max_chunks: int = 100,
        chunk_size: int = 128,
        chunk_overlap: int = 32,
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
        self._gru_hidden_dim = gru_hidden_dim
        self._gru_num_layers = gru_num_layers
        self._gru_bidirectional = gru_bidirectional
        self._gru_dropout = gru_dropout
        self._max_chunks = max_chunks
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._transformer_layers = transformer_layers
        self._transformer_heads = transformer_heads
        self._transformer_dim = transformer_dim
        self._transformer_dropout = transformer_dropout
        self._gated_attention_dim = gated_attention_dim
        self._projection_dim = projection_dim
        self._max_vocab_size = max_vocab_size
        self._min_word_freq = min_word_freq

        # GRU output dimension
        self._gru_output_dim = gru_hidden_dim * (2 if gru_bidirectional else 1)

        # Tokenizer (fit during fit_tokenizer)
        self._tokenizer = WordTokenizer(
            max_length=chunk_size,
            min_freq=min_word_freq,
            max_vocab_size=max_vocab_size
        )

        # Embedding layer (initialized after tokenizer is fitted)
        self._embedding = None

        # BiGRU for chunk encoding
        self._gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=gru_hidden_dim,
            num_layers=gru_num_layers,
            batch_first=True,
            dropout=gru_dropout if gru_num_layers > 1 else 0,
            bidirectional=gru_bidirectional
        )

        # Attention pooling for tokens within chunks (simple attention, not gated)
        self._chunk_attention = AttentionPooling(
            hidden_dim=self._gru_output_dim,
            attention_dim=self._gru_output_dim
        )

        # Project GRU output to transformer dim
        self._input_projection = nn.Linear(self._gru_output_dim, transformer_dim)

        # Positional encoding for chunks
        self._register_positional_encoding()

        # Layer norm after embedding
        self._embed_layer_norm = nn.LayerNorm(embedding_dim)
        self._embed_dropout = nn.Dropout(gru_dropout)

        # Interpretable transformer layers for cross-chunk context
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

        logger.info(f"GRUPoolExtractor initialized:")
        logger.info(f"  Embedding dim: {embedding_dim}")
        logger.info(f"  GRU hidden dim: {gru_hidden_dim} x {2 if gru_bidirectional else 1}")
        logger.info(f"  GRU layers: {gru_num_layers}")
        logger.info(f"  Transformer layers: {transformer_layers}, dim: {transformer_dim}")
        logger.info(f"  Transformer heads: {transformer_heads}")
        logger.info(f"  Max chunks: {max_chunks}, chunk_size: {chunk_size}, overlap: {chunk_overlap}")
        logger.info(f"  Gated attention dim: {gated_attention_dim}")
        logger.info(f"  Projection dim: {projection_dim}")

    def _register_positional_encoding(self):
        """Create sinusoidal positional encoding for chunks."""
        max_len = self._max_chunks
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
        """Return the transformer hidden dimension (chunk embedding dimension)."""
        return self._transformer_dim

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return self._tokenizer.vocab_size

    def fit_tokenizer(self, texts: List[str]) -> 'GRUPoolExtractor':
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
        import re
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def _encode_chunks_batch(
        self,
        chunk_ids_list: List[List[int]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode chunks with BiGRU + attention pooling.

        Args:
            chunk_ids_list: List of token ID lists (one per chunk)

        Returns:
            chunk_embeddings: (num_chunks, gru_output_dim)
            chunk_attention_weights: (num_chunks, max_chunk_len) - attention weights within chunks
        """
        if not chunk_ids_list:
            return (
                torch.zeros(0, self._gru_output_dim, device=self._device),
                torch.zeros(0, 0, device=self._device)
            )

        # Pad chunks to same length
        max_len = max(len(c) for c in chunk_ids_list)
        padded = []
        masks = []

        for chunk_ids in chunk_ids_list:
            pad_len = max_len - len(chunk_ids)
            padded.append(chunk_ids + [self._tokenizer.pad_token] * pad_len)
            masks.append([1] * len(chunk_ids) + [0] * pad_len)

        input_ids = torch.tensor(padded, dtype=torch.long, device=self._device)
        attention_mask = torch.tensor(masks, dtype=torch.float, device=self._device)

        # Embed tokens
        embeddings = self._embedding(input_ids)  # (num_chunks, max_len, embedding_dim)
        embeddings = self._embed_layer_norm(embeddings)
        embeddings = self._embed_dropout(embeddings)

        # GRU forward
        gru_output, _ = self._gru(embeddings)  # (num_chunks, max_len, gru_output_dim)

        # Attention pooling within each chunk
        chunk_embeddings = self._chunk_attention(gru_output, attention_mask)  # (num_chunks, gru_output_dim)

        # Get attention weights for interpretability
        with torch.no_grad():
            scores = self._chunk_attention.v(
                torch.tanh(self._chunk_attention.W(gru_output))
            ).squeeze(-1)
            scores = scores.masked_fill(attention_mask == 0, -1e9)
            chunk_attention_weights = F.softmax(scores, dim=1)

        return chunk_embeddings, chunk_attention_weights

    def _encode_chunks_from_tensors(
        self,
        chunk_token_ids: torch.Tensor,
        chunk_lengths: torch.Tensor,
        max_sub_batch: int = 128
    ) -> torch.Tensor:
        """
        Encode pre-padded chunk tensors through embedding -> GRU -> attention pooling.

        Sub-batches to avoid OOM when total_chunks is large.

        Args:
            chunk_token_ids: (total_chunks, max_len) pre-padded token IDs from collator
            chunk_lengths: (total_chunks,) actual lengths per chunk
            max_sub_batch: Maximum chunks per forward pass

        Returns:
            (total_chunks, gru_output_dim) chunk embeddings
        """
        total = chunk_token_ids.size(0)
        max_len = chunk_token_ids.size(1)
        all_embeddings = []

        for start in range(0, total, max_sub_batch):
            end = min(start + max_sub_batch, total)
            batch_ids = chunk_token_ids[start:end].to(self._device)      # (sub_B, max_len)
            batch_lens = chunk_lengths[start:end].to(self._device)       # (sub_B,)

            # Build attention mask from lengths
            sub_B = batch_ids.size(0)
            attention_mask = torch.arange(max_len, device=self._device).unsqueeze(0).expand(sub_B, -1)
            attention_mask = (attention_mask < batch_lens.unsqueeze(1)).float()  # (sub_B, max_len)

            # Embed tokens
            embeddings = self._embedding(batch_ids)  # (sub_B, max_len, embedding_dim)
            embeddings = self._embed_layer_norm(embeddings)
            embeddings = self._embed_dropout(embeddings)

            # GRU forward
            gru_output, _ = self._gru(embeddings)  # (sub_B, max_len, gru_output_dim)

            # Attention pooling within each chunk
            chunk_embs = self._chunk_attention(gru_output, attention_mask)  # (sub_B, gru_output_dim)
            all_embeddings.append(chunk_embs)

        return torch.cat(all_embeddings, dim=0)  # (total_chunks, gru_output_dim)

    def _pad_chunks_to_batch(
        self,
        chunk_embeddings: torch.Tensor,
        doc_chunk_counts: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reshape flat chunk embeddings into padded batch tensor with positional encoding.

        Args:
            chunk_embeddings: (total_chunks, D) flat tensor
            doc_chunk_counts: List[int] of per-document chunk counts

        Returns:
            padded: (B, max_C, D) padded batch tensor with positional encoding
            mask: (B, max_C) attention mask (1=valid, 0=pad)
        """
        B = len(doc_chunk_counts)
        max_C = max(doc_chunk_counts)
        D = chunk_embeddings.size(1)

        padded = torch.zeros(B, max_C, D, device=self._device)
        mask = torch.zeros(B, max_C, device=self._device)

        offset = 0
        for i, count in enumerate(doc_chunk_counts):
            padded[i, :count] = chunk_embeddings[offset:offset + count]
            mask[i, :count] = 1.0
            # Add positional encoding
            padded[i, :count] = padded[i, :count] + self._positional_encoding[:count].to(self._device)
            offset += count

        return padded, mask

    def _forward_preprocessed(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        GPU-only forward pass on pre-tokenized batch from VocabChunkCollator.

        Args:
            batch: Dict with 'chunk_token_ids', 'chunk_lengths', 'doc_chunk_counts', 'texts'

        Returns:
            Feature tensor of shape (B, projection_dim)
        """
        chunk_token_ids = batch['chunk_token_ids']      # (total_chunks, max_len)
        chunk_lengths = batch['chunk_lengths']           # (total_chunks,)
        doc_chunk_counts = batch['doc_chunk_counts']     # List[int], len B
        texts = batch['texts']

        # 1. Encode all chunks through embedding -> GRU -> attention pooling (sub-batched)
        chunk_embeddings = self._encode_chunks_from_tensors(
            chunk_token_ids, chunk_lengths
        )  # (total_chunks, gru_output_dim)

        # 2. Project to transformer dim
        chunk_embeddings = self._input_projection(chunk_embeddings)  # (total_chunks, transformer_dim)

        # 3. Pad into (B, max_C, transformer_dim) with positional encoding
        padded, mask = self._pad_chunks_to_batch(chunk_embeddings, doc_chunk_counts)

        # 4. Run through transformer layers with key_padding_mask
        key_padding_mask = (mask == 0)  # True = IGNORE for nn.MultiheadAttention
        for layer in self._transformer_layer_modules:
            padded, _ = layer(padded, return_attention=False, key_padding_mask=key_padding_mask)

        # 5. Apply gated attention pooling (batched)
        pooled, _ = self._gated_pooling(padded, attention_mask=mask)  # (B, transformer_dim)

        # 6. Add numeric features if enabled
        if self._numeric_features_enabled and self._numeric_feature_vector is not None:
            numeric_feats = self._numeric_feature_vector(texts)  # (B, numeric_dim)
            pooled = self._numeric_merge(
                torch.cat([pooled, numeric_feats], dim=1)
            )

        # 7. Output projection
        features = self._output_projection(pooled)  # (B, projection_dim)
        return features

    def _forward_from_texts(self, texts: List[str]) -> torch.Tensor:
        """
        Legacy forward path: chunk + tokenize + encode per document.

        Args:
            texts: List of document texts

        Returns:
            Feature tensor of shape (batch_size, projection_dim)
        """
        batch_outputs = []

        for text in texts:
            # 1. Split text into overlapping token chunks
            chunks = split_into_chunks_vocab(
                text,
                word_to_idx=self._tokenizer.word_to_id,
                tokenize_fn=self._tokenize_fn,
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
                max_chunks=self._max_chunks
            )

            # 2. Encode chunks with BiGRU + attention
            chunk_embeddings, _ = self._encode_chunks_batch(chunks)  # (C, gru_output_dim)

            if chunk_embeddings.size(0) == 0:
                # Fallback for empty text
                batch_outputs.append(
                    torch.zeros(self._projection_dim, device=self._device)
                )
                continue

            # 3. Project to transformer dim
            chunk_embeddings = self._input_projection(chunk_embeddings)  # (C, transformer_dim)

            # 4. Add positional encoding
            num_chunks = chunk_embeddings.size(0)
            chunk_embeddings = chunk_embeddings + self._positional_encoding[:num_chunks].to(self._device)

            # 5. Run through transformer layers
            sequence = chunk_embeddings.unsqueeze(0)  # (1, C, transformer_dim)
            for layer in self._transformer_layer_modules:
                sequence, _ = layer(sequence, return_attention=False)

            # 6. Apply gated attention pooling
            pooled, _ = self._gated_pooling(sequence.squeeze(0))  # (transformer_dim,)

            # 6.5. Add numeric features if enabled
            if self._numeric_features_enabled and self._numeric_feature_vector is not None:
                numeric_feats = self._numeric_feature_vector([text])  # (1, numeric_dim)
                pooled = self._numeric_merge(
                    torch.cat([pooled.unsqueeze(0), numeric_feats], dim=1)
                ).squeeze(0)

            # 7. Output projection
            output = self._output_projection(pooled)  # (projection_dim,)
            batch_outputs.append(output)

        # Stack batch
        return torch.stack(batch_outputs)  # (B, projection_dim)

    def forward(self, texts_or_batch) -> torch.Tensor:
        """
        Extract features from texts or preprocessed batch.

        Accepts either:
        - List[str]: Raw text strings (legacy path, chunks + tokenizes internally)
        - Dict with 'chunk_token_ids': Preprocessed batch from VocabChunkCollator

        Args:
            texts_or_batch: List of document texts or preprocessed batch dict

        Returns:
            Feature tensor of shape (batch_size, projection_dim)
        """
        if not self._initialized or self._embedding is None:
            raise RuntimeError("Must call fit_tokenizer() before forward()")

        if isinstance(texts_or_batch, dict) and 'chunk_token_ids' in texts_or_batch:
            return self._forward_preprocessed(texts_or_batch)
        return self._forward_from_texts(texts_or_batch)

    def _forward_with_instances_preprocessed(
        self,
        batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Batched forward pass returning chunk-level info for CLAM-style instance loss.

        Args:
            batch: Preprocessed batch dict from VocabChunkCollator

        Returns:
            doc_features: (B, projection_dim) - document-level features
            chunk_embeddings_list: List of (C_i, transformer_dim) tensors per doc
            attention_weights_list: List of (C_i,) tensors - gated attention weights per doc
        """
        chunk_token_ids = batch['chunk_token_ids']      # (total_chunks, max_len)
        chunk_lengths = batch['chunk_lengths']           # (total_chunks,)
        doc_chunk_counts = batch['doc_chunk_counts']     # List[int], len B
        texts = batch['texts']

        # 1. Encode all chunks through embedding -> GRU -> attention pooling (sub-batched)
        chunk_embeddings = self._encode_chunks_from_tensors(
            chunk_token_ids, chunk_lengths
        )  # (total_chunks, gru_output_dim)

        # 2. Project to transformer dim
        chunk_embeddings = self._input_projection(chunk_embeddings)  # (total_chunks, transformer_dim)

        # 3. Pad into (B, max_C, transformer_dim) with positional encoding
        padded, mask = self._pad_chunks_to_batch(chunk_embeddings, doc_chunk_counts)

        # 4. Run through transformer layers with key_padding_mask
        key_padding_mask = (mask == 0)  # True = IGNORE for nn.MultiheadAttention
        for layer in self._transformer_layer_modules:
            padded, _ = layer(padded, return_attention=False, key_padding_mask=key_padding_mask)

        # 5. Split back per-doc, apply gated pooling per-doc for CLAM
        batch_outputs = []
        chunk_embeddings_list = []
        attention_weights_list = []

        for i, count in enumerate(doc_chunk_counts):
            doc_chunks = padded[i, :count]  # (C_i, transformer_dim)
            pooled, attn_weights = self._gated_pooling(doc_chunks)  # (transformer_dim,), (C_i,)

            # Numeric features
            if self._numeric_features_enabled and self._numeric_feature_vector is not None:
                numeric_feats = self._numeric_feature_vector([texts[i]])  # (1, numeric_dim)
                pooled = self._numeric_merge(
                    torch.cat([pooled.unsqueeze(0), numeric_feats], dim=1)
                ).squeeze(0)

            output = self._output_projection(pooled)
            batch_outputs.append(output)
            chunk_embeddings_list.append(doc_chunks)
            attention_weights_list.append(attn_weights)

        doc_features = torch.stack(batch_outputs)
        return doc_features, chunk_embeddings_list, attention_weights_list

    def _forward_with_instances_from_texts(
        self,
        texts: List[str]
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Legacy forward_with_instances path from raw text strings.

        Args:
            texts: List of document texts

        Returns:
            doc_features: (B, projection_dim) - document-level features
            chunk_embeddings_list: List of (C_i, transformer_dim) tensors per doc
            attention_weights_list: List of (C_i,) tensors - gated attention weights per doc
        """
        batch_outputs = []
        chunk_embeddings_list = []
        attention_weights_list = []

        for text in texts:
            # 1. Split text into overlapping token chunks
            chunks = split_into_chunks_vocab(
                text,
                word_to_idx=self._tokenizer.word_to_id,
                tokenize_fn=self._tokenize_fn,
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
                max_chunks=self._max_chunks
            )

            # 2. Encode chunks with BiGRU + attention
            chunk_embeddings, _ = self._encode_chunks_batch(chunks)  # (C, gru_output_dim)

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

            # 4. Add positional encoding
            num_chunks = chunk_embeddings.size(0)
            chunk_embeddings = chunk_embeddings + self._positional_encoding[:num_chunks].to(self._device)

            # 5. Run through transformer layers
            sequence = chunk_embeddings.unsqueeze(0)  # (1, C, transformer_dim)
            for layer in self._transformer_layer_modules:
                sequence, _ = layer(sequence, return_attention=False)

            # Extract transformer-processed chunk embeddings (before pooling)
            transformer_chunk_embs = sequence.squeeze(0)  # (C, transformer_dim)

            # 6. Apply gated attention pooling
            pooled, attn_weights = self._gated_pooling(transformer_chunk_embs)  # (transformer_dim,), (C,)

            # 6.5. Add numeric features if enabled
            if self._numeric_features_enabled and self._numeric_feature_vector is not None:
                numeric_feats = self._numeric_feature_vector([text])  # (1, numeric_dim)
                pooled = self._numeric_merge(
                    torch.cat([pooled.unsqueeze(0), numeric_feats], dim=1)
                ).squeeze(0)

            # 7. Output projection
            output = self._output_projection(pooled)  # (projection_dim,)
            batch_outputs.append(output)

            # Store chunk-level info for instance loss
            chunk_embeddings_list.append(transformer_chunk_embs)
            attention_weights_list.append(attn_weights)

        # Stack batch
        doc_features = torch.stack(batch_outputs)  # (B, projection_dim)

        return doc_features, chunk_embeddings_list, attention_weights_list

    def forward_with_instances(
        self,
        texts_or_batch
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass returning document features AND chunk-level info for CLAM-style instance loss.

        Accepts either List[str] or preprocessed batch dict.

        This method returns transformer-processed chunk embeddings (before gated pooling)
        and their gated attention weights, enabling instance-level supervision on
        top-attended chunks.

        Args:
            texts_or_batch: List of document texts or preprocessed batch dict

        Returns:
            doc_features: (B, projection_dim) - document-level features
            chunk_embeddings_list: List of (C_i, transformer_dim) tensors per doc
            attention_weights_list: List of (C_i,) tensors - gated attention weights per doc
        """
        if not self._initialized or self._embedding is None:
            raise RuntimeError("Must call fit_tokenizer() before forward_with_instances()")

        if isinstance(texts_or_batch, dict) and 'chunk_token_ids' in texts_or_batch:
            return self._forward_with_instances_preprocessed(texts_or_batch)
        return self._forward_with_instances_from_texts(texts_or_batch)

    def get_tokenizer_state(self) -> Dict[str, Any]:
        """Get tokenizer state for checkpoint saving."""
        return self._tokenizer.get_state()

    def load_tokenizer_state(self, state: Dict[str, Any]) -> 'GRUPoolExtractor':
        """
        Load tokenizer state and rebuild embedding layer.

        Args:
            state: Tokenizer state dictionary

        Returns:
            self for method chaining
        """
        self._tokenizer.load_state(state)

        # Rebuild embedding layer with loaded vocabulary size
        self._embedding = nn.Embedding(
            num_embeddings=self._tokenizer.vocab_size,
            embedding_dim=self._embedding_dim,
            padding_idx=self._tokenizer.pad_token
        )
        self._embedding.to(self._device)

        self._initialized = True
        logger.info(f"Tokenizer state loaded: vocab size = {self._tokenizer.vocab_size}")

        return self

    def get_state(self) -> Dict[str, Any]:
        """
        Get extractor state for checkpoint saving.

        Returns:
            Dictionary containing configuration for reconstruction
        """
        return {
            'embedding_dim': self._embedding_dim,
            'gru_hidden_dim': self._gru_hidden_dim,
            'gru_num_layers': self._gru_num_layers,
            'gru_bidirectional': self._gru_bidirectional,
            'gru_dropout': self._gru_dropout,
            'max_chunks': self._max_chunks,
            'chunk_size': self._chunk_size,
            'chunk_overlap': self._chunk_overlap,
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

        This extracts attention weights at both the token level (within chunks)
        and chunk level (gated pooling).

        Args:
            texts: List of document texts
            top_k: Number of top-attended items to show

        Returns:
            List of dicts per document with attention interpretations:
            - 'num_chunks': Number of chunks in document
            - 'chunk_attention': Gated attention weights over chunks
            - 'top_chunks': Top-k chunks by attention weight
            - 'top_tokens_per_chunk': Top tokens within each top chunk
        """
        if not self._initialized or self._embedding is None:
            raise RuntimeError("Must call fit_tokenizer() before interpret_attention()")

        interpretations = []

        with torch.no_grad():
            for text in texts:
                # Split into chunks
                chunks = split_into_chunks_vocab(
                    text,
                    word_to_idx=self._tokenizer.word_to_id,
                    tokenize_fn=self._tokenize_fn,
                    chunk_size=self._chunk_size,
                    chunk_overlap=self._chunk_overlap,
                    max_chunks=self._max_chunks
                )

                if not chunks or all(len(c) == 0 for c in chunks):
                    interpretations.append({
                        'num_chunks': 0,
                        'chunk_attention': [],
                        'top_chunks': [],
                        'top_tokens_per_chunk': []
                    })
                    continue

                # Encode chunks
                chunk_embeddings, token_attention = self._encode_chunks_batch(chunks)

                if chunk_embeddings.size(0) == 0:
                    interpretations.append({
                        'num_chunks': 0,
                        'chunk_attention': [],
                        'top_chunks': [],
                        'top_tokens_per_chunk': []
                    })
                    continue

                # Project and add positional encoding
                chunk_embeddings = self._input_projection(chunk_embeddings)
                num_chunks = chunk_embeddings.size(0)
                chunk_embeddings = chunk_embeddings + self._positional_encoding[:num_chunks].to(self._device)

                # Run through transformer
                sequence = chunk_embeddings.unsqueeze(0)
                for layer in self._transformer_layer_modules:
                    sequence, _ = layer(sequence, return_attention=False)

                # Get gated attention weights
                _, chunk_weights = self._gated_pooling(sequence.squeeze(0))
                chunk_weights = chunk_weights.cpu()

                # Get top-k chunks
                k_actual = min(top_k, num_chunks)
                top_vals, top_indices = torch.topk(chunk_weights, k_actual)

                # Convert token IDs back to words for interpretation
                id_to_word = self._tokenizer.id_to_word

                # Build chunk text representations
                chunk_texts = []
                for chunk_ids in chunks:
                    words = [id_to_word.get(tid, '<unk>') for tid in chunk_ids]
                    chunk_texts.append(' '.join(words))

                # Top chunks info
                top_chunks = [
                    {
                        'chunk_idx': int(idx),
                        'attention': float(val),
                        'text_preview': chunk_texts[idx][:200] + '...' if len(chunk_texts[idx]) > 200 else chunk_texts[idx]
                    }
                    for val, idx in zip(top_vals.tolist(), top_indices.tolist())
                ]

                # Top tokens within top chunks
                top_tokens_per_chunk = []
                for idx in top_indices.tolist():
                    chunk_ids = chunks[idx]
                    chunk_token_weights = token_attention[idx, :len(chunk_ids)].cpu()

                    k_tokens = min(5, len(chunk_ids))
                    top_token_vals, top_token_indices = torch.topk(chunk_token_weights, k_tokens)

                    top_tokens = [
                        {
                            'token': id_to_word.get(chunk_ids[int(ti)], '<unk>'),
                            'attention': float(tv)
                        }
                        for tv, ti in zip(top_token_vals.tolist(), top_token_indices.tolist())
                    ]
                    top_tokens_per_chunk.append(top_tokens)

                interpretations.append({
                    'num_chunks': num_chunks,
                    'chunk_attention': chunk_weights.tolist(),
                    'top_chunks': top_chunks,
                    'top_tokens_per_chunk': top_tokens_per_chunk
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
            'transformer_layers': self._transformer_layers,
            'transformer_heads': self._transformer_heads,
            'gru_hidden_dim': self._gru_hidden_dim,
            'gru_bidirectional': self._gru_bidirectional
        }

    def to(self, device):
        """Override to track device properly."""
        self._device = device if isinstance(device, torch.device) else torch.device(device)

        if self._embedding is not None:
            self._embedding = self._embedding.to(self._device)

        return super().to(device)
