# cdt/models/bert_pool_extractor.py
"""BERT + Transformer + Gated Attention Pooling feature extractor.

This module implements a hierarchical approach for extracting features from long
clinical text that combines:

1. Pretrained BERT per chunk: [CLS] token + attention-pooled token embeddings (or random init)
2. Standard transformer layers for cross-chunk context (chunks attend to each other)
3. Gated attention pooling (tanh x sigmoid) for final document aggregation

Unlike hierarchical_transformer which uses a [POOL] token for aggregation,
this extractor uses gated attention pooling (like gru_pool) for more expressive
document-level aggregation. BERT is unfrozen by default for end-to-end
fine-tuning, and supports random weight initialization.

Architecture:
    Long Clinical Text
            |
    Split into Overlapping Token Chunks (C chunks)
            |
    BERT per Chunk -> [CLS] token + AttentionPooled tokens (C x 2*bert_dim)
            |
    Project to transformer_dim (C x transformer_dim)
            |
    Add Sinusoidal Positional Encoding
            |
    Transformer Layer(s) - chunks attend to each other (cross-chunk context)
            |
    Gated Attention Pooling (tanh x sigmoid) -> single document vector
            |
    Output Projection MLP -> Final Representation (projection_dim)

DOES NOT require fit_tokenizer() - uses pretrained HF tokenizer.
"""

import logging
import math
import threading
from typing import Optional, List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .chunking import split_into_chunks_hf
from .gru_extractor import AttentionPooling
from .gru_pool_extractor import GatedAttentionPooling
from .hierarchical_transformer_extractor import InterpretableTransformerLayer
from .numeric_features import NumericFeatureVector


logger = logging.getLogger(__name__)

# Module-level lock for thread-safe HuggingFace model loading.
# AutoModel.from_pretrained with accelerate is not thread-safe (can leave
# meta tensors when called concurrently), so we serialize model loading.
_model_load_lock = threading.Lock()


class BertPoolExtractor(nn.Module):
    """
    BERT + Transformer + Gated Attention Pooling feature extractor.

    Combines:
    - Pretrained BERT [CLS] + attention-pooled tokens per chunk for chunk encoding
    - Transformer layers for cross-chunk context
    - Gated attention pooling for final document aggregation

    Each chunk is represented by concatenating BERT's [CLS] token (global summary)
    with a learned attention-pooled vector over all token hidden states (content-focused
    weighted average). This gives the model both BERT's global summary and a
    keyword-sensitive representation.

    This produces a single feature vector via gated attention pooling over
    transformer-processed chunk embeddings.

    No fit_tokenizer() required - uses pretrained HF tokenizer.

    Args:
        sentence_encoder_model: HuggingFace model name for chunk encoding
        freeze_sentence_encoder: Whether to freeze BERT weights (default: False, unfrozen)
        use_pretrained: If True, load pretrained weights; if False, random init
        max_chunks: Maximum number of chunks to process per document
        chunk_size: Number of tokens per chunk
        chunk_overlap: Number of overlapping tokens between chunks
        num_transformer_layers: Number of transformer layers for cross-chunk processing
        num_attention_heads: Number of attention heads in transformer
        transformer_dim: Hidden dimension for transformer layers
        transformer_dropout: Dropout rate for transformer layers
        gated_attention_dim: Hidden dimension for gated attention pooling
        projection_dim: Final output dimension
        device: PyTorch device
    """

    def __init__(
        self,
        sentence_encoder_model: str = "prajjwal1/bert-tiny",
        freeze_sentence_encoder: bool = False,
        use_pretrained: bool = True,
        max_chunks: int = 100,
        chunk_size: int = 128,
        chunk_overlap: int = 32,
        num_transformer_layers: int = 2,
        num_attention_heads: int = 4,
        transformer_dim: int = 256,
        transformer_dropout: float = 0.1,
        gated_attention_dim: int = 128,
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
        self._use_pretrained = use_pretrained
        self._max_chunks = max_chunks
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._num_layers = num_transformer_layers
        self._num_heads = num_attention_heads
        self._transformer_dim = transformer_dim
        self._dropout = transformer_dropout
        self._gated_attention_dim = gated_attention_dim
        self._projection_dim = projection_dim

        # Lazy initialization
        self._sentence_encoder = None
        self._tokenizer = None
        self._sentence_dim = None
        self._chunk_attention = None
        self._input_projection = None
        self._transformer_layer_modules = None
        self._gated_pooling = None
        self._output_projection = None
        self._initialized = False

        init_mode = "pretrained" if use_pretrained else "random init"
        logger.info(f"BertPoolExtractor initialized:")
        logger.info(f"  Chunk encoder: {sentence_encoder_model} ({init_mode})")
        logger.info(f"  Freeze encoder: {freeze_sentence_encoder}")
        logger.info(f"  Max chunks: {max_chunks}, chunk_size: {chunk_size}, overlap: {chunk_overlap}")
        logger.info(f"  Transformer layers: {num_transformer_layers}")
        logger.info(f"  Transformer dim: {transformer_dim}")
        logger.info(f"  Attention heads: {num_attention_heads}")
        logger.info(f"  Gated attention dim: {gated_attention_dim}")
        logger.info(f"  Projection dim: {projection_dim}")

    def _ensure_initialized(self):
        """Lazily initialize components."""
        if self._initialized:
            return

        from transformers import AutoModel, AutoTokenizer, AutoConfig

        init_mode = "pretrained" if self._use_pretrained else "random init"
        logger.info(f"Loading sentence encoder: {self._sentence_encoder_model} ({init_mode})")

        # Serialize model loading across threads
        with _model_load_lock:
            # Always load pretrained tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self._sentence_encoder_model)

            if self._use_pretrained:
                self._sentence_encoder = AutoModel.from_pretrained(self._sentence_encoder_model)
            else:
                # Random init: use architecture only
                config = AutoConfig.from_pretrained(self._sentence_encoder_model)
                self._sentence_encoder = AutoModel.from_config(config)
                logger.info(f"  Created model from config with random weights")

        self._sentence_encoder = self._sentence_encoder.to(self._device)
        self._sentence_dim = self._sentence_encoder.config.hidden_size
        logger.info(f"  Sentence encoder dim: {self._sentence_dim}")

        if self._freeze:
            for param in self._sentence_encoder.parameters():
                param.requires_grad = False
            logger.info("  Sentence encoder frozen")

        # Attention pooling over token hidden states within each chunk
        self._chunk_attention = AttentionPooling(
            hidden_dim=self._sentence_dim,
            attention_dim=self._sentence_dim
        ).to(self._device)

        # Input projection: [CLS] || attn_pooled -> transformer_dim
        self._input_projection = nn.Linear(2 * self._sentence_dim, self._transformer_dim).to(self._device)

        # Positional encoding (sinusoidal)
        self._register_positional_encoding()

        # Interpretable transformer layers for cross-chunk context
        self._transformer_layer_modules = nn.ModuleList([
            InterpretableTransformerLayer(
                d_model=self._transformer_dim,
                nhead=self._num_heads,
                dim_feedforward=self._transformer_dim * 4,
                dropout=self._dropout
            )
            for _ in range(self._num_layers)
        ]).to(self._device)

        # Gated attention pooling for final aggregation
        self._gated_pooling = GatedAttentionPooling(
            hidden_dim=self._transformer_dim,
            attention_dim=self._gated_attention_dim
        ).to(self._device)

        # Output projection
        self._output_projection = nn.Sequential(
            nn.Linear(self._transformer_dim, self._transformer_dim),
            nn.LayerNorm(self._transformer_dim),
            nn.GELU(),
            nn.Dropout(self._dropout),
            nn.Linear(self._transformer_dim, self._projection_dim),
            nn.LayerNorm(self._projection_dim)
        ).to(self._device)

        # Numeric feature vector (merged after pooling, before output projection)
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
        logger.info("BertPoolExtractor initialization complete")

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

    def _encode_chunks_batch(
        self, chunks: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode chunks with BERT, returning [CLS] tokens, token hidden states,
        and the attention mask.

        Args:
            chunks: List of chunk strings

        Returns:
            cls_embeddings: (num_chunks, sentence_dim) [CLS] embeddings
            token_hidden_states: (num_chunks, seq_len, sentence_dim) all token hidden states
            attention_mask: (num_chunks, seq_len) 1=valid, 0=pad
        """
        if not chunks:
            self._ensure_initialized()
            empty = torch.zeros(0, self._sentence_dim, device=self._device)
            return empty, torch.zeros(0, 0, self._sentence_dim, device=self._device), torch.zeros(0, 0, device=self._device)

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
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        return cls_embeddings, outputs.last_hidden_state, attention_mask

    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        Extract features from texts.

        Args:
            texts: List of document texts

        Returns:
            Feature tensor of shape (batch_size, projection_dim)
        """
        self._ensure_initialized()
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
            cls_embeddings, token_hidden_states, token_attn_mask = self._encode_chunks_batch(chunks)

            if cls_embeddings.size(0) == 0:
                batch_outputs.append(
                    torch.zeros(self._projection_dim, device=self._device)
                )
                continue

            # 3. Attention-pool token hidden states and concatenate with [CLS]
            attn_pooled = self._chunk_attention(token_hidden_states, token_attn_mask)  # (C, sentence_dim)
            chunk_embeddings = torch.cat([cls_embeddings, attn_pooled], dim=1)  # (C, 2*sentence_dim)

            # 4. Project to transformer dim
            chunk_embeddings = self._input_projection(chunk_embeddings)  # (C, transformer_dim)

            # 5. Add positional encoding
            num_chunks = chunk_embeddings.size(0)
            chunk_embeddings = chunk_embeddings + self._positional_encoding[:num_chunks].to(self._device)

            # 6. Run through transformer layers
            sequence = chunk_embeddings.unsqueeze(0)  # (1, C, transformer_dim)
            for layer in self._transformer_layer_modules:
                sequence, _ = layer(sequence, return_attention=False)

            # 7. Apply gated attention pooling
            pooled, _ = self._gated_pooling(sequence.squeeze(0))  # (transformer_dim,)

            # 7.5. Add numeric features if enabled
            if self._numeric_features_enabled and self._numeric_feature_vector is not None:
                numeric_feats = self._numeric_feature_vector([text])  # (1, numeric_dim)
                pooled = self._numeric_merge(
                    torch.cat([pooled.unsqueeze(0), numeric_feats], dim=1)
                ).squeeze(0)

            # 8. Output projection
            output = self._output_projection(pooled)  # (projection_dim,)
            batch_outputs.append(output)

        # Stack batch
        return torch.stack(batch_outputs)  # (B, projection_dim)

    def forward_with_instances(
        self,
        texts: List[str]
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass returning document features AND chunk-level info for CLAM-style instance loss.

        This method returns transformer-processed chunk embeddings (before gated pooling)
        and their gated attention weights, enabling instance-level supervision on
        top-attended chunks.

        Args:
            texts: List of document texts

        Returns:
            doc_features: (B, projection_dim) - document-level features
            chunk_embeddings_list: List of (C_i, transformer_dim) tensors per doc
            attention_weights_list: List of (C_i,) tensors - gated attention weights per doc
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
                chunks = [text[:500]]

            # 2. Encode chunks with BERT
            cls_embeddings, token_hidden_states, token_attn_mask = self._encode_chunks_batch(chunks)

            if cls_embeddings.size(0) == 0:
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

            # 3. Attention-pool token hidden states and concatenate with [CLS]
            attn_pooled = self._chunk_attention(token_hidden_states, token_attn_mask)  # (C, sentence_dim)
            chunk_embeddings = torch.cat([cls_embeddings, attn_pooled], dim=1)  # (C, 2*sentence_dim)

            # 4. Project to transformer dim
            chunk_embeddings = self._input_projection(chunk_embeddings)  # (C, transformer_dim)

            # 5. Add positional encoding
            num_chunks = chunk_embeddings.size(0)
            chunk_embeddings = chunk_embeddings + self._positional_encoding[:num_chunks].to(self._device)

            # 6. Run through transformer layers
            sequence = chunk_embeddings.unsqueeze(0)  # (1, C, transformer_dim)
            for layer in self._transformer_layer_modules:
                sequence, _ = layer(sequence, return_attention=False)

            # Extract transformer-processed chunk embeddings (before pooling)
            transformer_chunk_embs = sequence.squeeze(0)  # (C, transformer_dim)

            # 7. Apply gated attention pooling
            pooled, attn_weights = self._gated_pooling(transformer_chunk_embs)  # (transformer_dim,), (C,)

            # 7.5. Add numeric features if enabled
            if self._numeric_features_enabled and self._numeric_feature_vector is not None:
                numeric_feats = self._numeric_feature_vector([text])  # (1, numeric_dim)
                pooled = self._numeric_merge(
                    torch.cat([pooled.unsqueeze(0), numeric_feats], dim=1)
                ).squeeze(0)

            # 8. Output projection
            output = self._output_projection(pooled)  # (projection_dim,)
            batch_outputs.append(output)

            # Store chunk-level info for instance loss
            chunk_embeddings_list.append(transformer_chunk_embs)
            attention_weights_list.append(attn_weights)

        # Stack batch
        doc_features = torch.stack(batch_outputs)  # (B, projection_dim)

        return doc_features, chunk_embeddings_list, attention_weights_list

    def init_extractor(self, texts: List[str]) -> 'BertPoolExtractor':
        """
        Initialize the feature extractor (triggers lazy initialization).

        For BertPoolExtractor, this loads the BERT encoder and initializes
        the transformer and pooling layers. The texts argument is not used
        since we use pretrained tokenizers.

        Args:
            texts: List of training text strings (not used, kept for API compatibility)

        Returns:
            self for method chaining
        """
        self._ensure_initialized()
        return self

    def fit_tokenizer(self, texts: List[str]) -> 'BertPoolExtractor':
        """Alias for init_extractor() for backward compatibility."""
        return self.init_extractor(texts)

    def get_state(self) -> Dict[str, Any]:
        """
        Get extractor state for checkpoint saving.

        Returns:
            Dictionary containing configuration for reconstruction
        """
        return {
            'sentence_encoder_model': self._sentence_encoder_model,
            'freeze_sentence_encoder': self._freeze,
            'use_pretrained': self._use_pretrained,
            'max_chunks': self._max_chunks,
            'chunk_size': self._chunk_size,
            'chunk_overlap': self._chunk_overlap,
            'num_transformer_layers': self._num_layers,
            'num_attention_heads': self._num_heads,
            'transformer_dim': self._transformer_dim,
            'transformer_dropout': self._dropout,
            'gated_attention_dim': self._gated_attention_dim,
            'projection_dim': self._projection_dim,
        }

    def interpret_attention(
        self,
        texts: List[str],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get human-readable interpretation of chunk attention.

        This extracts gated attention weights over chunks, showing which
        chunks contribute most to the final representation.

        Args:
            texts: List of document texts
            top_k: Number of top-attended chunks to show

        Returns:
            List of dicts per document with attention interpretations:
            - 'num_chunks': Number of chunks in document
            - 'chunk_attention': Gated attention weights over chunks
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
                cls_embeddings, token_hidden_states, token_attn_mask = self._encode_chunks_batch(chunks)

                if cls_embeddings.size(0) == 0:
                    interpretations.append({
                        'num_chunks': 0,
                        'chunk_attention': [],
                        'top_chunks': []
                    })
                    continue

                attn_pooled = self._chunk_attention(token_hidden_states, token_attn_mask)
                chunk_embeddings = torch.cat([cls_embeddings, attn_pooled], dim=1)
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

                top_chunks = [
                    {
                        'chunk_idx': int(idx),
                        'attention': float(val),
                        'text_preview': chunks[idx][:200] + '...' if len(chunks[idx]) > 200 else chunks[idx]
                    }
                    for val, idx in zip(top_vals.tolist(), top_indices.tolist())
                ]

                interpretations.append({
                    'num_chunks': num_chunks,
                    'chunk_attention': chunk_weights.tolist(),
                    'top_chunks': top_chunks
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
            'model': self._sentence_encoder_model,
            'use_pretrained': self._use_pretrained
        }

    def to(self, device):
        """Override to track device properly."""
        self._device = device if isinstance(device, torch.device) else torch.device(device)

        if self._sentence_encoder is not None:
            self._sentence_encoder = self._sentence_encoder.to(self._device)
        if self._chunk_attention is not None:
            self._chunk_attention = self._chunk_attention.to(self._device)
        if self._input_projection is not None:
            self._input_projection = self._input_projection.to(self._device)
        if self._transformer_layer_modules is not None:
            self._transformer_layer_modules = self._transformer_layer_modules.to(self._device)
        if self._gated_pooling is not None:
            self._gated_pooling = self._gated_pooling.to(self._device)
        if self._output_projection is not None:
            self._output_projection = self._output_projection.to(self._device)
        if hasattr(self, '_positional_encoding') and self._positional_encoding is not None:
            self._positional_encoding = self._positional_encoding.to(self._device)

        return super().to(device)
