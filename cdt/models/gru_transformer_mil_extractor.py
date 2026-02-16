# cdt/models/gru_transformer_mil_extractor.py
"""GRU-Transformer-MIL feature extractor combining learned GRU with transformer and gated MIL.

This module implements a hierarchical approach for extracting features from long
clinical text that learns entirely from scratch (no pretrained encoders):

1. Split text into overlapping token chunks
2. Encode each chunk with a shared BiGRU + attention pooling
3. Apply transformer layers for cross-chunk context
4. Apply gated MIL attention with K learnable confounder queries
5. Task-specific weighting of confounders (propensity, tau/y0, outcome/y1)
6. Concatenate and project to output dimension

Architecture:
    Long Clinical Text
            |
    Split into Overlapping Token Chunks (C chunks)
            |
    [Per Chunk - Shared GRU]
    Word Embeddings -> BiGRU -> Attention Pooling (C x gru_output_dim)
            |
    Transformer Layers (cross-chunk context) + Positional Encoding
            |
    Gated MIL Attention with K Confounder Queries
            |
    K Confounder Representations (K x D)
            |
    Task-Specific Weighting (propensity, tau/y0, outcome/y1)
            |
    Concatenate (3 x D) -> MLP Projection -> Final Output

Key insight: This architecture learns from scratch, which means:
- No pretrained encoder means all parameters learn together via causal loss
- BiGRU is O(N) vs transformer's O(N^2) per chunk
- Shared GRU across chunks provides parameter efficiency
- Transformer cross-chunk layers add global context at chunk level

Supports two forward paths:
- List[str]: Legacy per-doc chunking + tokenizing (interpret_attention, predict)
- Dict with 'chunk_token_ids': Preprocessed batch from VocabChunkCollator (training)

References:
- Ilse et al. (2018): "Attention-based Deep Multiple Instance Learning"
"""

import logging
import math
from typing import Optional, List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .chunking import split_into_chunks_vocab
from .cnn_extractor import WordTokenizer
from .gru_extractor import AttentionPooling
from .hierarchical_transformer_extractor import InterpretableTransformerLayer
from .gated_mil_attention import GatedMILAttention, TaskSpecificConfounderWeighting
from .numeric_features import NumericFeatureVector


logger = logging.getLogger(__name__)


class GRUTransformerMILExtractor(nn.Module):
    """
    Feature extractor combining BiGRU chunk encoding with transformer and gated MIL.

    This architecture:
    1. Uses learned BiGRU + attention for within-chunk token pooling
    2. Uses transformer layers for cross-chunk context
    3. Uses gated MIL attention with K confounder queries for aggregation
    4. Uses task-specific weighting of shared confounders

    IMPORTANT: Call fit_tokenizer(texts) before using this extractor.

    Args:
        embedding_dim: Dimension of word embeddings
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
        transformer_dropout: Dropout rate for transformer
        num_confounders: Number of confounder queries (K)
        mil_hidden_dim: Hidden dimension for gated MIL attention
        projection_dim: Final output dimension
        max_vocab_size: Maximum vocabulary size
        min_word_freq: Minimum word frequency for vocabulary
        model_type: "rlearner" or "dragonnet" (affects task-specific weights)
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
        num_confounders: int = 4,
        mil_hidden_dim: int = 128,
        projection_dim: int = 128,
        max_vocab_size: int = 50000,
        min_word_freq: int = 2,
        model_type: str = "rlearner",
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
        self._num_directions = 2 if gru_bidirectional else 1
        self._gru_output_dim = gru_hidden_dim * self._num_directions
        self._max_chunks = max_chunks
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._transformer_layers = transformer_layers
        self._transformer_heads = transformer_heads
        self._transformer_dim = transformer_dim
        self._transformer_dropout = transformer_dropout
        self._num_confounders = num_confounders
        self._mil_hidden_dim = mil_hidden_dim
        self._projection_dim = projection_dim
        self._max_vocab_size = max_vocab_size
        self._min_word_freq = min_word_freq
        self._model_type = model_type

        # Word tokenizer (builds vocabulary from training data)
        # Exposed as both self.tokenizer (legacy) and self._tokenizer (collator compat)
        self.tokenizer = WordTokenizer(
            max_length=chunk_size,
            min_freq=min_word_freq,
            max_vocab_size=max_vocab_size
        )

        # Lazy initialization
        self._embedding = None
        self._gru = None
        self._attention_pooling = None
        self._input_projection = None
        self._transformer_layer_stack = None
        self._gated_mil_attention = None
        self._task_weighting = None
        self._output_projection = None
        self._initialized = False

        logger.info(f"GRUTransformerMILExtractor initialized:")
        logger.info(f"  Embedding dim: {embedding_dim}")
        logger.info(f"  GRU hidden dim: {gru_hidden_dim} x {self._num_directions} directions")
        logger.info(f"  GRU layers: {gru_num_layers}")
        logger.info(f"  Max chunks: {max_chunks}, chunk_size: {chunk_size}, overlap: {chunk_overlap}")
        logger.info(f"  Transformer layers: {transformer_layers}, heads: {transformer_heads}")
        logger.info(f"  Transformer dim: {transformer_dim}")
        logger.info(f"  Num confounders: {num_confounders}")
        logger.info(f"  MIL hidden dim: {mil_hidden_dim}")
        logger.info(f"  Projection dim: {projection_dim}")
        logger.info(f"  Model type: {model_type}")

    @property
    def _tokenizer(self):
        """Alias for VocabChunkCollator compatibility (expects ext._tokenizer)."""
        return self.tokenizer

    def _ensure_initialized(self):
        """Ensure all components are initialized after tokenizer is fitted."""
        if self._initialized:
            return

        if self.tokenizer.vocab_size == 0:
            raise RuntimeError("Must call fit_tokenizer() before using this extractor")

        # Word embedding layer
        self._embedding = nn.Embedding(
            num_embeddings=self.tokenizer.vocab_size,
            embedding_dim=self._embedding_dim,
            padding_idx=self.tokenizer.pad_token
        ).to(self._device)

        self._embed_layer_norm = nn.LayerNorm(self._embedding_dim).to(self._device)
        self._embed_dropout = nn.Dropout(self._gru_dropout).to(self._device)

        # BiGRU for chunk encoding (shared across all chunks)
        self._gru = nn.GRU(
            input_size=self._embedding_dim,
            hidden_size=self._gru_hidden_dim,
            num_layers=self._gru_num_layers,
            batch_first=True,
            dropout=self._gru_dropout if self._gru_num_layers > 1 else 0,
            bidirectional=self._gru_bidirectional
        ).to(self._device)

        # Attention pooling for GRU outputs
        self._attention_pooling = AttentionPooling(
            hidden_dim=self._gru_output_dim,
            attention_dim=self._gru_output_dim
        ).to(self._device)

        # Project GRU output to transformer dim
        self._input_projection = nn.Linear(
            self._gru_output_dim,
            self._transformer_dim
        ).to(self._device)

        # Register positional encoding
        self._register_positional_encoding()

        # Transformer layers for cross-chunk context
        self._transformer_layer_stack = nn.ModuleList([
            InterpretableTransformerLayer(
                d_model=self._transformer_dim,
                nhead=self._transformer_heads,
                dim_feedforward=self._transformer_dim * 4,
                dropout=self._transformer_dropout
            )
            for _ in range(self._transformer_layers)
        ]).to(self._device)

        # Gated MIL attention for confounder extraction
        self._gated_mil_attention = GatedMILAttention(
            input_dim=self._transformer_dim,
            hidden_dim=self._mil_hidden_dim,
            num_confounders=self._num_confounders,
            dropout=self._transformer_dropout
        ).to(self._device)

        # Task-specific weighting of confounders
        self._task_weighting = TaskSpecificConfounderWeighting(
            confounder_dim=self._transformer_dim,
            num_confounders=self._num_confounders,
            model_type=self._model_type
        ).to(self._device)

        # Output projection: 3 * transformer_dim -> projection_dim
        self._output_projection = nn.Sequential(
            nn.Linear(3 * self._transformer_dim, self._projection_dim * 2),
            nn.LayerNorm(self._projection_dim * 2),
            nn.GELU(),
            nn.Dropout(self._transformer_dropout),
            nn.Linear(self._projection_dim * 2, self._projection_dim),
            nn.LayerNorm(self._projection_dim)
        ).to(self._device)

        # Numeric feature vector (merged into combined representation before output projection)
        self._numeric_feature_vector = None
        self._numeric_merge = None
        if self._numeric_features_enabled:
            self._numeric_feature_vector = NumericFeatureVector(
                num_magnitude_bins=self._numeric_magnitude_bins,
                num_type_categories=self._numeric_type_categories,
                output_dim=self._numeric_embedding_dim
            ).to(self._device)
            self._numeric_merge = nn.Sequential(
                nn.Linear(3 * self._transformer_dim + self._numeric_embedding_dim, 3 * self._transformer_dim),
                nn.LayerNorm(3 * self._transformer_dim),
                nn.ReLU(),
            ).to(self._device)

        self._initialized = True
        logger.info(f"GRUTransformerMILExtractor fully initialized: vocab_size={self.tokenizer.vocab_size}")

    def _register_positional_encoding(self):
        """Create sinusoidal positional encoding for chunk positions."""
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
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return self.tokenizer.vocab_size

    def fit_tokenizer(self, texts: List[str]) -> 'GRUTransformerMILExtractor':
        """
        Fit tokenizer on training texts and initialize all components.

        MUST be called before using the model for training or inference.

        Args:
            texts: List of training text strings

        Returns:
            self for method chaining
        """
        self.tokenizer.fit(texts)
        logger.info(f"Tokenizer fitted: vocab_size={self.tokenizer.vocab_size}")

        # Now initialize all components
        self._ensure_initialized()

        return self

    def _tokenize_fn(self, text: str) -> List[str]:
        """Tokenize text into words for chunk splitting."""
        return text.lower().split()

    def _encode_chunk_batch(
        self,
        chunk_ids_list: List[List[int]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a batch of chunks through shared GRU + attention pooling.

        Args:
            chunk_ids_list: List of token ID lists (one per chunk)

        Returns:
            chunk_embeddings: (num_chunks, gru_output_dim) chunk representations
            attention_weights: (num_chunks, max_len) attention weights for interpretability
        """
        if not chunk_ids_list:
            return (
                torch.zeros(0, self._gru_output_dim, device=self._device),
                torch.zeros(0, 0, device=self._device)
            )

        # Pad chunks to same length
        max_len = max(len(chunk) for chunk in chunk_ids_list)
        padded_chunks = []
        attention_masks = []

        pad_id = self.tokenizer.pad_token

        for chunk in chunk_ids_list:
            pad_length = max_len - len(chunk)
            padded = chunk + [pad_id] * pad_length
            mask = [1] * len(chunk) + [0] * pad_length
            padded_chunks.append(padded)
            attention_masks.append(mask)

        # Convert to tensors
        input_ids = torch.tensor(padded_chunks, dtype=torch.long, device=self._device)
        attention_mask = torch.tensor(attention_masks, dtype=torch.float, device=self._device)

        # Embed tokens
        embeddings = self._embedding(input_ids)  # (num_chunks, max_len, embedding_dim)
        embeddings = self._embed_layer_norm(embeddings)
        embeddings = self._embed_dropout(embeddings)

        # GRU forward pass
        gru_output, _ = self._gru(embeddings)  # (num_chunks, max_len, gru_output_dim)

        # Attention pooling
        chunk_embeddings = self._attention_pooling(gru_output, attention_mask)  # (num_chunks, gru_output_dim)

        # Get attention weights for interpretability
        scores = self._attention_pooling.v(torch.tanh(self._attention_pooling.W(gru_output))).squeeze(-1)
        scores = scores.masked_fill(attention_mask == 0, -1e9)
        attention_weights = F.softmax(scores, dim=1)

        return chunk_embeddings, attention_weights

    # ------------------------------------------------------------------
    # Preprocessed batch helpers (VocabChunkCollator path)
    # ------------------------------------------------------------------

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
            chunk_token_ids: (total_chunks, max_len) pre-padded token IDs
            chunk_lengths: (total_chunks,) actual length of each chunk
            max_sub_batch: Maximum chunks per GRU forward pass

        Returns:
            (total_chunks, gru_output_dim) chunk representations
        """
        total = chunk_token_ids.size(0)
        max_len = chunk_token_ids.size(1)
        all_chunk_embs = []

        for start in range(0, total, max_sub_batch):
            end = min(start + max_sub_batch, total)
            batch_ids = chunk_token_ids[start:end].to(self._device)     # (sub_B, max_len)
            batch_lens = chunk_lengths[start:end].to(self._device)      # (sub_B,)

            # Build attention mask from lengths
            sub_B = batch_ids.size(0)
            positions = torch.arange(max_len, device=self._device).unsqueeze(0)  # (1, max_len)
            attention_mask = (positions < batch_lens.unsqueeze(1)).float()  # (sub_B, max_len)

            # Embed tokens
            embeddings = self._embedding(batch_ids)  # (sub_B, max_len, embedding_dim)
            embeddings = self._embed_layer_norm(embeddings)
            embeddings = self._embed_dropout(embeddings)

            # GRU forward pass
            gru_output, _ = self._gru(embeddings)  # (sub_B, max_len, gru_output_dim)

            # Attention pooling
            chunk_embs = self._attention_pooling(gru_output, attention_mask)  # (sub_B, gru_output_dim)
            all_chunk_embs.append(chunk_embs)

        return torch.cat(all_chunk_embs, dim=0)  # (total_chunks, gru_output_dim)

    def _pad_chunks_to_batch(
        self,
        chunk_embeddings: torch.Tensor,
        doc_chunk_counts: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reshape flat chunk embeddings into padded batch tensor with positional encoding.

        Args:
            chunk_embeddings: (total_chunks, D) flat tensor of projected chunk embeddings
            doc_chunk_counts: List[int] of per-document chunk counts

        Returns:
            padded: (B, max_C, D) padded batch tensor with positional encoding added
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

        GRU encoding and transformer are batched across all chunks/documents.
        MIL attention + task weighting remain per-document (variable chunk counts
        + K confounder queries make true batching impractical without masking
        complexity that would negate the performance gain).

        Args:
            batch: Dict with 'chunk_token_ids', 'chunk_lengths', 'doc_chunk_counts', 'texts'

        Returns:
            Feature tensor of shape (B, projection_dim)
        """
        chunk_token_ids = batch['chunk_token_ids']          # (total_chunks, max_len)
        chunk_lengths = batch['chunk_lengths']               # (total_chunks,)
        doc_chunk_counts = batch['doc_chunk_counts']         # List[int], len B
        texts = batch['texts']

        # 1. Encode all chunks through GRU (sub-batched)
        chunk_gru_embs = self._encode_chunks_from_tensors(
            chunk_token_ids, chunk_lengths
        )  # (total_chunks, gru_output_dim)

        # 2. Project to transformer dim
        chunk_embeddings = self._input_projection(chunk_gru_embs)  # (total_chunks, transformer_dim)

        # 3. Pad into (B, max_C, transformer_dim) with positional encoding
        padded, mask = self._pad_chunks_to_batch(chunk_embeddings, doc_chunk_counts)

        # 4. Run through transformer layers with key_padding_mask
        key_padding_mask = (mask == 0)  # True = IGNORE for nn.MultiheadAttention
        for layer in self._transformer_layer_stack:
            padded, _ = layer(padded, return_attention=False, key_padding_mask=key_padding_mask)

        # 5. Split back per-doc for MIL attention + task weighting (per-doc operation)
        batch_outputs = []
        for i, count in enumerate(doc_chunk_counts):
            doc_chunks = padded[i, :count]  # (C_i, transformer_dim)

            # Apply gated MIL attention to get K confounders
            confounders, _ = self._gated_mil_attention(doc_chunks)  # (K, transformer_dim)

            # Apply task-specific weighting
            prop_repr, task2_repr, task3_repr = self._task_weighting(confounders)
            # Each is (transformer_dim,)

            combined = torch.cat([prop_repr, task2_repr, task3_repr], dim=0)  # (3 * transformer_dim,)
            batch_outputs.append(combined)

        # Stack batch
        batch_outputs = torch.stack(batch_outputs)  # (B, 3 * transformer_dim)

        # 6. Add numeric features if enabled
        if self._numeric_features_enabled and self._numeric_feature_vector is not None:
            numeric_feats = self._numeric_feature_vector(texts)  # (B, numeric_dim)
            batch_outputs = self._numeric_merge(
                torch.cat([batch_outputs, numeric_feats], dim=1)
            )

        # 7. Project to output dimension
        features = self._output_projection(batch_outputs)  # (B, projection_dim)
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
            # 1. Split into overlapping token chunks
            chunks = split_into_chunks_vocab(
                text,
                word_to_idx=self.tokenizer.word_to_id,
                tokenize_fn=self._tokenize_fn,
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
                max_chunks=self._max_chunks
            )

            # 2. Encode chunks through shared GRU + attention
            chunk_embeddings, _ = self._encode_chunk_batch(chunks)  # (C, gru_output_dim)

            # 3. Project to transformer dim
            chunk_embeddings = self._input_projection(chunk_embeddings)  # (C, transformer_dim)

            # 4. Add positional encoding
            num_chunks = chunk_embeddings.size(0)
            chunk_embeddings = chunk_embeddings + self._positional_encoding[:num_chunks].to(self._device)

            # 5. Run through transformer layers
            chunk_embeddings = chunk_embeddings.unsqueeze(0)  # (1, C, transformer_dim)
            for layer in self._transformer_layer_stack:
                chunk_embeddings, _ = layer(chunk_embeddings, return_attention=False)
            chunk_embeddings = chunk_embeddings.squeeze(0)  # (C, transformer_dim)

            # 6. Apply gated MIL attention to get K confounders
            confounders, _ = self._gated_mil_attention(chunk_embeddings)  # (K, transformer_dim)

            # 7. Apply task-specific weighting
            prop_repr, task2_repr, task3_repr = self._task_weighting(confounders)
            # Each is (transformer_dim,)

            # 8. Concatenate task representations
            combined = torch.cat([prop_repr, task2_repr, task3_repr], dim=0)  # (3 * transformer_dim,)
            batch_outputs.append(combined)

        # Stack batch
        batch_outputs = torch.stack(batch_outputs)  # (B, 3 * transformer_dim)

        # Add numeric features if enabled
        if self._numeric_features_enabled and self._numeric_feature_vector is not None:
            numeric_feats = self._numeric_feature_vector(texts)  # (B, numeric_dim)
            batch_outputs = self._numeric_merge(
                torch.cat([batch_outputs, numeric_feats], dim=1)
            )

        # 9. Project to output dimension
        features = self._output_projection(batch_outputs)  # (B, projection_dim)

        return features

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
        self._ensure_initialized()

        if isinstance(texts_or_batch, dict) and 'chunk_token_ids' in texts_or_batch:
            return self._forward_preprocessed(texts_or_batch)
        return self._forward_from_texts(texts_or_batch)

    def interpret_attention(
        self,
        texts: List[str],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get human-readable interpretation of attention patterns.

        Returns attention at three levels:
        1. Token attention within chunks (which words matter)
        2. Chunk attention per confounder (which chunks matter)
        3. Task-specific confounder weights

        Args:
            texts: List of document texts
            top_k: Number of top-attended chunks to show per confounder

        Returns:
            List of dicts per document with attention interpretations
        """
        self._ensure_initialized()
        interpretations = []

        # Get task weights (shared across all documents)
        task_weights = self._task_weighting.get_weights()

        with torch.no_grad():
            for text in texts:
                # Split into chunks
                chunks = split_into_chunks_vocab(
                    text,
                    word_to_idx=self.tokenizer.word_to_id,
                    tokenize_fn=self._tokenize_fn,
                    chunk_size=self._chunk_size,
                    chunk_overlap=self._chunk_overlap,
                    max_chunks=self._max_chunks
                )

                # Get chunk embeddings and token attention
                chunk_embeddings, token_attention = self._encode_chunk_batch(chunks)

                # Project and add positional encoding
                chunk_embeddings = self._input_projection(chunk_embeddings)
                num_chunks = chunk_embeddings.size(0)
                chunk_embeddings = chunk_embeddings + self._positional_encoding[:num_chunks].to(self._device)

                # Run through transformer
                chunk_embeddings = chunk_embeddings.unsqueeze(0)
                for layer in self._transformer_layer_stack:
                    chunk_embeddings, _ = layer(chunk_embeddings, return_attention=False)
                chunk_embeddings = chunk_embeddings.squeeze(0)

                # Get MIL attention weights
                _, mil_attention = self._gated_mil_attention(chunk_embeddings, return_attention=True)

                # Build interpretation
                confounder_attention = {}
                top_chunks_per_confounder = {}

                # Decode chunks back to text for display
                chunk_texts = []
                for chunk_ids in chunks:
                    words = [self.tokenizer.id_to_word.get(idx, '<UNK>') for idx in chunk_ids]
                    chunk_texts.append(' '.join(words))

                for k in range(self._num_confounders):
                    attn = mil_attention[k].cpu()  # (C,)
                    confounder_attention[f'confounder_{k}'] = attn.tolist()

                    # Get top-k chunks
                    k_actual = min(top_k, len(chunks))
                    top_vals, top_indices = torch.topk(attn, k_actual)

                    top_chunks_per_confounder[f'confounder_{k}'] = [
                        {
                            'chunk': chunk_texts[int(idx)],
                            'attention': float(val),
                            'idx': int(idx)
                        }
                        for val, idx in zip(top_vals, top_indices)
                    ]

                interpretations.append({
                    'chunks': chunk_texts,
                    'confounder_attention': confounder_attention,
                    'top_chunks_per_confounder': top_chunks_per_confounder,
                    'task_weights': task_weights,
                    'token_attention_shape': list(token_attention.shape)
                })

        return interpretations

    def get_task_weights(self) -> Dict[str, List[float]]:
        """
        Get the task-specific confounder weights for interpretability.

        Returns:
            Dictionary with normalized weights per task
        """
        self._ensure_initialized()
        return self._task_weighting.get_weights()

    def _get_tau_weights(self) -> torch.Tensor:
        """
        Get normalized tau weights for aggregating MIL attention across K confounders.

        For R-Learner: uses 'tau' weights (treatment effect modifiers).
        For DragonNet: uses average of 'y0' and 'y1' weights (potential outcomes).

        Returns:
            (K,) normalized weight tensor on self._device
        """
        task_weights = self._task_weighting.get_weights()
        if 'tau' in task_weights:
            tau_weights = torch.tensor(task_weights['tau'], device=self._device, dtype=torch.float32)
        else:
            # DragonNet: average of y0 and y1 weights as proxy for treatment effect importance
            y0_weights = torch.tensor(task_weights['y0'], device=self._device, dtype=torch.float32)
            y1_weights = torch.tensor(task_weights['y1'], device=self._device, dtype=torch.float32)
            tau_weights = (y0_weights + y1_weights) / 2
        return F.softmax(tau_weights, dim=0)  # Normalize to sum to 1

    def _forward_with_instances_preprocessed(
        self,
        batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Batched forward pass returning chunk-level info for CLAM-style instance loss.

        GRU + transformer are batched; MIL attention + CLAM outputs are per-doc.

        Args:
            batch: Preprocessed batch dict from VocabChunkCollator

        Returns:
            doc_features: (B, projection_dim) - document-level features
            chunk_embeddings_list: List of (C_i, transformer_dim) tensors per doc
            attention_weights_list: List of (C_i,) tensors - tau-weighted attention per doc
        """
        chunk_token_ids = batch['chunk_token_ids']
        chunk_lengths = batch['chunk_lengths']
        doc_chunk_counts = batch['doc_chunk_counts']
        texts = batch['texts']

        tau_weights = self._get_tau_weights()

        # 1. Encode all chunks through GRU (sub-batched)
        chunk_gru_embs = self._encode_chunks_from_tensors(
            chunk_token_ids, chunk_lengths
        )  # (total_chunks, gru_output_dim)

        # 2. Project to transformer dim
        chunk_embeddings = self._input_projection(chunk_gru_embs)  # (total_chunks, transformer_dim)

        # 3. Pad into (B, max_C, transformer_dim) with positional encoding
        padded, mask = self._pad_chunks_to_batch(chunk_embeddings, doc_chunk_counts)

        # 4. Run through transformer layers with key_padding_mask
        key_padding_mask = (mask == 0)
        for layer in self._transformer_layer_stack:
            padded, _ = layer(padded, return_attention=False, key_padding_mask=key_padding_mask)

        # 5. Split back per-doc for MIL + CLAM outputs
        batch_outputs = []
        chunk_embeddings_list = []
        attention_weights_list = []

        for i, count in enumerate(doc_chunk_counts):
            doc_chunks = padded[i, :count]  # (C_i, transformer_dim)

            if count == 0:
                batch_outputs.append(
                    torch.zeros(3 * self._transformer_dim, device=self._device)
                )
                chunk_embeddings_list.append(
                    torch.zeros(0, self._transformer_dim, device=self._device)
                )
                attention_weights_list.append(
                    torch.zeros(0, device=self._device)
                )
                continue

            # Store transformer-processed chunk embeddings for CLAM
            transformer_chunk_embs = doc_chunks.clone()

            # Apply gated MIL attention to get K confounders with attention weights
            confounders, mil_attn = self._gated_mil_attention(
                doc_chunks, return_attention=True
            )  # confounders: (K, transformer_dim), mil_attn: (K, C_i)

            # Aggregate attention using tau weights: (K,) @ (K, C_i) -> (C_i,)
            aggregated_attention = (tau_weights.unsqueeze(1) * mil_attn).sum(dim=0)

            # Apply task-specific weighting
            prop_repr, task2_repr, task3_repr = self._task_weighting(confounders)

            combined = torch.cat([prop_repr, task2_repr, task3_repr], dim=0)
            batch_outputs.append(combined)

            chunk_embeddings_list.append(transformer_chunk_embs)
            attention_weights_list.append(aggregated_attention)

        # Stack batch
        batch_outputs = torch.stack(batch_outputs)  # (B, 3 * transformer_dim)

        # 6. Add numeric features if enabled
        if self._numeric_features_enabled and self._numeric_feature_vector is not None:
            numeric_feats = self._numeric_feature_vector(texts)
            batch_outputs = self._numeric_merge(
                torch.cat([batch_outputs, numeric_feats], dim=1)
            )

        # 7. Project to output dimension
        features = self._output_projection(batch_outputs)  # (B, projection_dim)

        return features, chunk_embeddings_list, attention_weights_list

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
            attention_weights_list: List of (C_i,) tensors - tau-weighted attention per doc
        """
        batch_outputs = []
        chunk_embeddings_list = []
        attention_weights_list = []

        tau_weights = self._get_tau_weights()

        for text in texts:
            # 1. Split into overlapping token chunks
            chunks = split_into_chunks_vocab(
                text,
                word_to_idx=self.tokenizer.word_to_id,
                tokenize_fn=self._tokenize_fn,
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
                max_chunks=self._max_chunks
            )

            # 2. Encode chunks through shared GRU + attention
            chunk_embeddings, _ = self._encode_chunk_batch(chunks)  # (C, gru_output_dim)

            if chunk_embeddings.size(0) == 0:
                # Fallback for empty text
                batch_outputs.append(
                    torch.zeros(3 * self._transformer_dim, device=self._device)
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
            chunk_embeddings = chunk_embeddings.unsqueeze(0)  # (1, C, transformer_dim)
            for layer in self._transformer_layer_stack:
                chunk_embeddings, _ = layer(chunk_embeddings, return_attention=False)
            chunk_embeddings = chunk_embeddings.squeeze(0)  # (C, transformer_dim)

            # Store transformer-processed chunk embeddings for CLAM
            transformer_chunk_embs = chunk_embeddings.clone()

            # 6. Apply gated MIL attention to get K confounders with attention weights
            confounders, attention_weights = self._gated_mil_attention(
                chunk_embeddings, return_attention=True
            )  # confounders: (K, transformer_dim), attention_weights: (K, C)

            # Aggregate attention using tau weights: (K,) @ (K, C) -> (C,)
            # This gives higher weight to confounders important for treatment effect
            aggregated_attention = (tau_weights.unsqueeze(1) * attention_weights).sum(dim=0)

            # 7. Apply task-specific weighting
            prop_repr, task2_repr, task3_repr = self._task_weighting(confounders)
            # Each is (transformer_dim,)

            # 8. Concatenate task representations
            combined = torch.cat([prop_repr, task2_repr, task3_repr], dim=0)  # (3 * transformer_dim,)
            batch_outputs.append(combined)

            # Store chunk-level info for instance loss
            chunk_embeddings_list.append(transformer_chunk_embs)
            attention_weights_list.append(aggregated_attention)

        # Stack batch
        batch_outputs = torch.stack(batch_outputs)  # (B, 3 * transformer_dim)

        # Add numeric features if enabled
        if self._numeric_features_enabled and self._numeric_feature_vector is not None:
            numeric_feats = self._numeric_feature_vector(texts)
            batch_outputs = self._numeric_merge(
                torch.cat([batch_outputs, numeric_feats], dim=1)
            )

        # 9. Project to output dimension
        features = self._output_projection(batch_outputs)  # (B, projection_dim)

        return features, chunk_embeddings_list, attention_weights_list

    def forward_with_instances(
        self,
        texts_or_batch
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass returning document features AND chunk-level info for CLAM-style instance loss.

        This method returns transformer-processed chunk embeddings and tau-weighted aggregated
        attention weights across K confounders, enabling instance-level supervision on
        top-attended chunks.

        The tau-weighted aggregation uses the task-specific weights for tau (treatment effect)
        to prioritize confounders most relevant to treatment effect modification. This aligns
        CLAM supervision with the causal objective.

        Accepts either List[str] or preprocessed batch dict.

        Args:
            texts_or_batch: List of document texts or preprocessed batch dict

        Returns:
            doc_features: (B, projection_dim) - document-level features
            chunk_embeddings_list: List of (C_i, transformer_dim) tensors per doc
            attention_weights_list: List of (C_i,) tensors - tau-weighted attention per doc
        """
        self._ensure_initialized()

        if isinstance(texts_or_batch, dict) and 'chunk_token_ids' in texts_or_batch:
            return self._forward_with_instances_preprocessed(texts_or_batch)
        return self._forward_with_instances_from_texts(texts_or_batch)

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
            'num_confounders': self._num_confounders,
            'model_type': self._model_type,
            'gru_output_dim': self._gru_output_dim,
            'transformer_dim': self._transformer_dim
        }

    def to(self, device):
        """Override to track device properly."""
        self._device = device if isinstance(device, torch.device) else torch.device(device)

        if self._embedding is not None:
            self._embedding = self._embedding.to(self._device)
        if self._embed_layer_norm is not None:
            self._embed_layer_norm = self._embed_layer_norm.to(self._device)
        if self._embed_dropout is not None:
            self._embed_dropout = self._embed_dropout.to(self._device)
        if self._gru is not None:
            self._gru = self._gru.to(self._device)
        if self._attention_pooling is not None:
            self._attention_pooling = self._attention_pooling.to(self._device)
        if self._input_projection is not None:
            self._input_projection = self._input_projection.to(self._device)
        if self._transformer_layer_stack is not None:
            self._transformer_layer_stack = self._transformer_layer_stack.to(self._device)
        if self._gated_mil_attention is not None:
            self._gated_mil_attention = self._gated_mil_attention.to(self._device)
        if self._task_weighting is not None:
            self._task_weighting = self._task_weighting.to(self._device)
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
            'num_confounders': self._num_confounders,
            'mil_hidden_dim': self._mil_hidden_dim,
            'projection_dim': self._projection_dim,
            'max_vocab_size': self._max_vocab_size,
            'min_word_freq': self._min_word_freq,
            'model_type': self._model_type,
            'vocab_size': self.tokenizer.vocab_size
        }

    def get_tokenizer_state(self) -> Dict[str, Any]:
        """
        Get tokenizer state for checkpoint saving.

        Returns:
            Dictionary with word_to_id mapping and vocabulary info
        """
        return {
            'word_to_id': self.tokenizer.word_to_id,
            'id_to_word': self.tokenizer.id_to_word,
            'vocab_size': self.tokenizer.vocab_size
        }

    def load_tokenizer_state(self, state: Dict[str, Any]) -> 'GRUTransformerMILExtractor':
        """
        Load tokenizer state from checkpoint.

        Args:
            state: Dictionary with tokenizer state

        Returns:
            self for method chaining
        """
        self.tokenizer.word_to_id = state['word_to_id']
        self.tokenizer.id_to_word = state['id_to_word']
        self.tokenizer._vocab_size = state['vocab_size']
        self._ensure_initialized()
        return self
