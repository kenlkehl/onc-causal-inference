# cdt/models/bert_gated_pool_extractor.py
"""BERT + Transformer + Gated Attention Pooling feature extractor.

This module implements a hierarchical approach for extracting features from long
clinical text using pretrained BERT for chunk encoding:

1. Split text into overlapping token chunks using HuggingFace tokenizer
2. Encode each chunk with BERT (frozen or fine-tuned) -> [CLS] or mean pooling
3. Apply transformer layers for cross-chunk context (chunks attend to each other)
4. Use gated attention pooling (tanh x sigmoid) for final document aggregation

Architecture:
    Long Clinical Text
            |
    Split into Overlapping Token Chunks (C chunks)
            |
    [Per Chunk - Shared BERT]
    Tokens -> BERT -> [CLS] or mean pool (C x bert_hidden_dim)
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
"""

import logging
import math
from typing import Optional, List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hierarchical_transformer_extractor import InterpretableTransformerLayer
from .chunking import split_into_chunks_hf
from .gated_attention import GatedAttentionPooling


logger = logging.getLogger(__name__)


class BERTGatedPoolExtractor(nn.Module):
    """
    BERT + Transformer + Gated Attention Pooling feature extractor.

    Uses pretrained BERT for chunk encoding with gated attention pooling
    for final document aggregation. Supports both frozen and fine-tuned BERT.

    Args:
        bert_model: HuggingFace model name for chunk encoding
        freeze_encoder: Whether to freeze BERT weights
        use_mean_pooling: Use mean pooling instead of [CLS] for chunk embeddings
        max_chunks: Maximum number of chunks to process per document
        chunk_size: Number of tokens per chunk
        chunk_overlap: Number of overlapping tokens between chunks
        transformer_layers: Number of transformer layers for cross-chunk processing
        transformer_heads: Number of attention heads in transformer
        transformer_dim: Hidden dimension for transformer layers
        transformer_dropout: Dropout rate for transformer layers
        gated_attention_dim: Hidden dimension for gated attention pooling
        projection_dim: Final output dimension
        device: PyTorch device
    """

    def __init__(
        self,
        bert_model: str = "prajjwal1/bert-tiny",
        freeze_encoder: bool = True,
        use_mean_pooling: bool = False,
        max_chunks: int = 100,
        chunk_size: int = 128,
        chunk_overlap: int = 32,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        transformer_dim: int = 256,
        transformer_dropout: float = 0.1,
        gated_attention_dim: int = 128,
        projection_dim: int = 128,
        device: Optional[torch.device] = None
    ):
        super().__init__()

        self._device = device or torch.device('cpu')
        self._bert_model_name = bert_model
        self._freeze_encoder = freeze_encoder
        self._use_mean_pooling = use_mean_pooling
        self._max_chunks = max_chunks
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._transformer_layers = transformer_layers
        self._transformer_heads = transformer_heads
        self._transformer_dim = transformer_dim
        self._transformer_dropout = transformer_dropout
        self._gated_attention_dim = gated_attention_dim
        self._projection_dim = projection_dim

        # Lazy initialization for BERT
        self._bert_encoder = None
        self._tokenizer = None
        self._bert_hidden_dim = None
        self._input_projection = None
        self._initialized = False

        # Will be initialized after BERT is loaded
        self._transformer_layer_modules = None
        self._gated_pooling = None
        self._output_projection = None

        logger.info(f"BERTGatedPoolExtractor initialized:")
        logger.info(f"  BERT model: {bert_model}")
        logger.info(f"  Freeze encoder: {freeze_encoder}")
        logger.info(f"  Use mean pooling: {use_mean_pooling}")
        logger.info(f"  Max chunks: {max_chunks}, chunk_size: {chunk_size}, overlap: {chunk_overlap}")
        logger.info(f"  Transformer layers: {transformer_layers}, dim: {transformer_dim}")
        logger.info(f"  Transformer heads: {transformer_heads}")
        logger.info(f"  Gated attention dim: {gated_attention_dim}")
        logger.info(f"  Projection dim: {projection_dim}")

    def _ensure_initialized(self):
        """Lazily initialize BERT encoder and downstream components."""
        if self._initialized:
            return

        from transformers import AutoModel, AutoTokenizer

        logger.info(f"Loading BERT encoder: {self._bert_model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(self._bert_model_name)
        self._bert_encoder = AutoModel.from_pretrained(self._bert_model_name)
        self._bert_encoder = self._bert_encoder.to(self._device)
        self._bert_hidden_dim = self._bert_encoder.config.hidden_size
        logger.info(f"  BERT hidden dim: {self._bert_hidden_dim}")

        if self._freeze_encoder:
            for param in self._bert_encoder.parameters():
                param.requires_grad = False
            logger.info("  BERT encoder frozen")

        # Input projection: bert_hidden_dim -> transformer_dim
        self._input_projection = nn.Linear(self._bert_hidden_dim, self._transformer_dim).to(self._device)

        # Positional encoding for chunks
        self._register_positional_encoding()

        # Interpretable transformer layers for cross-chunk context
        self._transformer_layer_modules = nn.ModuleList([
            InterpretableTransformerLayer(
                d_model=self._transformer_dim,
                nhead=self._transformer_heads,
                dim_feedforward=self._transformer_dim * 4,
                dropout=self._transformer_dropout
            )
            for _ in range(self._transformer_layers)
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
            nn.Dropout(self._transformer_dropout),
            nn.Linear(self._transformer_dim, self._projection_dim),
            nn.LayerNorm(self._projection_dim)
        ).to(self._device)

        self._initialized = True
        logger.info("BERTGatedPoolExtractor initialization complete")

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

    def fit_tokenizer(self, texts: List[str]) -> 'BERTGatedPoolExtractor':
        """
        Initialize the BERT encoder (triggers lazy initialization).

        For BERTGatedPoolExtractor, this just loads the pretrained BERT model
        and tokenizer. The texts argument is not used since we use pretrained
        tokenizers.

        Args:
            texts: List of training text strings (not used, kept for API compatibility)

        Returns:
            self for method chaining
        """
        self._ensure_initialized()
        return self

    def init_extractor(self, texts: List[str]) -> 'BERTGatedPoolExtractor':
        """Alias for fit_tokenizer() for API compatibility."""
        return self.fit_tokenizer(texts)

    def _encode_chunks_batch(self, chunks: List[str]) -> torch.Tensor:
        """
        Encode chunks with BERT, returning [CLS] or mean pooled embeddings.

        Args:
            chunks: List of chunk text strings

        Returns:
            Tensor of shape (num_chunks, bert_hidden_dim)
        """
        if not chunks:
            return torch.zeros(0, self._bert_hidden_dim, device=self._device)

        encoded = self._tokenizer(
            chunks,
            padding=True,
            truncation=True,
            max_length=self._chunk_size,
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].to(self._device)
        attention_mask = encoded['attention_mask'].to(self._device)

        with torch.set_grad_enabled(not self._freeze_encoder):
            outputs = self._bert_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        if self._use_mean_pooling:
            # Mean pooling over non-padding tokens
            hidden_states = outputs.last_hidden_state  # (B, L, D)
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            chunk_embeddings = sum_embeddings / sum_mask
        else:
            # [CLS] token at position 0
            chunk_embeddings = outputs.last_hidden_state[:, 0, :]

        return chunk_embeddings

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
            # 1. Split text into overlapping token chunks
            chunks = split_into_chunks_hf(
                text,
                self._tokenizer,
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
                max_chunks=self._max_chunks
            )

            # 2. Encode chunks with BERT
            chunk_embeddings = self._encode_chunks_batch(chunks)  # (C, bert_hidden_dim)

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

            # 7. Output projection
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
            # 1. Split text into overlapping token chunks
            chunks = split_into_chunks_hf(
                text,
                self._tokenizer,
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
                max_chunks=self._max_chunks
            )

            # 2. Encode chunks with BERT
            chunk_embeddings = self._encode_chunks_batch(chunks)  # (C, bert_hidden_dim)

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

            # 7. Output projection
            output = self._output_projection(pooled)  # (projection_dim,)
            batch_outputs.append(output)

            # Store chunk-level info for instance loss
            chunk_embeddings_list.append(transformer_chunk_embs)
            attention_weights_list.append(attn_weights)

        # Stack batch
        doc_features = torch.stack(batch_outputs)  # (B, projection_dim)

        return doc_features, chunk_embeddings_list, attention_weights_list

    def get_state(self) -> Dict[str, Any]:
        """
        Get extractor state for checkpoint saving.

        Returns:
            Dictionary containing configuration for reconstruction
        """
        return {
            'bert_model': self._bert_model_name,
            'freeze_encoder': self._freeze_encoder,
            'use_mean_pooling': self._use_mean_pooling,
            'max_chunks': self._max_chunks,
            'chunk_size': self._chunk_size,
            'chunk_overlap': self._chunk_overlap,
            'transformer_layers': self._transformer_layers,
            'transformer_heads': self._transformer_heads,
            'transformer_dim': self._transformer_dim,
            'transformer_dropout': self._transformer_dropout,
            'gated_attention_dim': self._gated_attention_dim,
            'projection_dim': self._projection_dim,
        }

    def interpret_attention(
        self,
        texts: List[str],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get human-readable interpretation of attention weights.

        This extracts gated attention weights over chunks.

        Args:
            texts: List of document texts
            top_k: Number of top-attended chunks to show

        Returns:
            List of dicts per document with attention interpretations:
            - 'num_chunks': Number of chunks in document
            - 'chunk_attention': Gated attention weights over chunks
            - 'top_chunks': Top-k chunks by attention weight with text preview
        """
        self._ensure_initialized()
        interpretations = []

        with torch.no_grad():
            for text in texts:
                # Split into chunks
                chunks = split_into_chunks_hf(
                    text,
                    self._tokenizer,
                    chunk_size=self._chunk_size,
                    chunk_overlap=self._chunk_overlap,
                    max_chunks=self._max_chunks
                )

                if not chunks:
                    interpretations.append({
                        'num_chunks': 0,
                        'chunk_attention': [],
                        'top_chunks': []
                    })
                    continue

                # Encode chunks
                chunk_embeddings = self._encode_chunks_batch(chunks)

                if chunk_embeddings.size(0) == 0:
                    interpretations.append({
                        'num_chunks': 0,
                        'chunk_attention': [],
                        'top_chunks': []
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

                # Top chunks info
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
            'transformer_layers': self._transformer_layers,
            'transformer_heads': self._transformer_heads,
            'bert_model': self._bert_model_name,
            'use_mean_pooling': self._use_mean_pooling
        }

    def to(self, device):
        """Override to track device properly."""
        self._device = device if isinstance(device, torch.device) else torch.device(device)

        if self._bert_encoder is not None:
            self._bert_encoder = self._bert_encoder.to(self._device)
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
