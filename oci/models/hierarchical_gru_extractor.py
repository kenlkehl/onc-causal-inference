# oci/models/hierarchical_gru_extractor.py
"""Hierarchical BiGRU feature extractor with two-level gated attention pooling.

Architecture:
    Raw text
      -> LearnedTokenizer (full text)
      -> Overlapping token-based chunking
      -> For each chunk: Embedding -> BiGRU -> GatedAttentionPooling -> chunk_vector
      -> All chunk vectors -> GatedAttentionPooling (document-level) -> document_vector
      -> Projection MLP -> (batch, output_dim)

Trains from scratch. Requires fit_tokenizer() before training.
"""

import logging
from typing import Dict, List, Any, Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .gated_attention_pooling import GatedAttentionPooling
from .learned_tokenizer import LearnedTokenizer
from .text_chunking import chunk_token_ids, pad_and_batch_chunks

logger = logging.getLogger(__name__)


class HierarchicalGRUExtractor(nn.Module):
    """Hierarchical BiGRU: chunk-level BiGRU encoding + document-level gated attention pooling.

    Args:
        embedding_dim: Word embedding dimension.
        gru_hidden_dim: Hidden dimension per GRU direction (output = 2 * gru_hidden_dim).
        num_gru_layers: Number of stacked BiGRU layers.
        chunk_size: Tokens per chunk.
        chunk_overlap: Overlapping tokens between consecutive chunks.
        max_chunks: Maximum number of chunks per document.
        vocab_size: Vocabulary size (set after fit_tokenizer).
        gated_attention_dim: Hidden dim for gated attention pooling.
        projection_dim: Final output dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        gru_hidden_dim: int = 256,
        num_gru_layers: int = 2,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        max_chunks: int = 32,
        vocab_size: int = 50000,
        gated_attention_dim: int = 128,
        projection_dim: int = 128,
        dropout: float = 0.1,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self._device = device or torch.device('cpu')
        self._embedding_dim = embedding_dim
        self._gru_hidden_dim = gru_hidden_dim
        self._num_gru_layers = num_gru_layers
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._max_chunks = max_chunks
        self._vocab_size = vocab_size
        self._gated_attention_dim = gated_attention_dim
        self._projection_dim = projection_dim
        self._dropout = dropout

        # BiGRU output dim = 2 * gru_hidden_dim (forward + backward)
        self._gru_output_dim = 2 * gru_hidden_dim

        self._tokenizer = LearnedTokenizer()

        # Embedding
        self._embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Shared BiGRU for all chunks
        self._gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=gru_hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_gru_layers > 1 else 0.0,
        )

        # Token-level pooling within each chunk
        self._token_pooling = GatedAttentionPooling(
            hidden_dim=self._gru_output_dim,
            attention_dim=gated_attention_dim,
        )

        # Document-level pooling across chunks
        self._chunk_pooling = GatedAttentionPooling(
            hidden_dim=self._gru_output_dim,
            attention_dim=gated_attention_dim,
        )

        # Projection MLP
        self._output_dim = projection_dim
        self._projection = nn.Sequential(
            nn.Linear(self._gru_output_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, projection_dim),
            nn.LayerNorm(projection_dim),
        )

        logger.info(
            f"HierarchicalGRUExtractor initialized: "
            f"embedding_dim={embedding_dim}, gru_hidden_dim={gru_hidden_dim}, "
            f"num_layers={num_gru_layers}, gru_output_dim={self._gru_output_dim}, "
            f"chunk_size={chunk_size}, chunk_overlap={chunk_overlap}, "
            f"max_chunks={max_chunks}, projection_dim={projection_dim}"
        )

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def fit_tokenizer(self, texts: List[str], min_freq: int = 2) -> None:
        """Build vocabulary from training texts."""
        self._tokenizer.fit(texts, vocab_size=self._vocab_size, min_freq=min_freq)
        actual_vocab = self._tokenizer.vocab_size
        if actual_vocab != self._embedding.num_embeddings:
            old_weight = self._embedding.weight.data
            self._embedding = nn.Embedding(actual_vocab, self._embedding_dim, padding_idx=0)
            copy_size = min(old_weight.shape[0], actual_vocab)
            self._embedding.weight.data[:copy_size] = old_weight[:copy_size]
            self._embedding = self._embedding.to(self._device)
        logger.info(f"HierarchicalGRUExtractor: tokenizer fitted, vocab_size={actual_vocab}")

    def forward(self, texts_or_batch) -> torch.Tensor:
        """Extract features from raw texts with hierarchical chunking.

        Args:
            texts_or_batch: List[str] of document texts.

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

        # Tokenize and chunk
        max_token_length = self._chunk_size * self._max_chunks
        batch_chunk_ids = []
        for text in texts:
            token_ids = self._tokenizer.encode(text, max_length=max_token_length)
            chunks = chunk_token_ids(
                token_ids, self._chunk_size, self._chunk_overlap, self._max_chunks
            )
            batch_chunk_ids.append(chunks)

        # Pad to uniform (B, max_chunks, max_chunk_len)
        input_ids, attn_mask, chunk_mask = pad_and_batch_chunks(
            batch_chunk_ids, self._tokenizer.pad_token_id
        )
        input_ids = input_ids.to(self._device)
        attn_mask = attn_mask.to(self._device)
        chunk_mask = chunk_mask.to(self._device)

        B, C, L = input_ids.shape

        # Flatten chunks: (B*C, L)
        flat_ids = input_ids.view(B * C, L)
        flat_mask = attn_mask.view(B * C, L)

        # Embed: (B*C, L, embedding_dim)
        embedded = self._embedding(flat_ids)

        # BiGRU: pack padded sequences for efficiency
        lengths = flat_mask.sum(dim=1).long().clamp(min=1).cpu()
        packed = pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        gru_out, _ = self._gru(packed)
        gru_out, _ = pad_packed_sequence(gru_out, batch_first=True, total_length=L)
        # gru_out: (B*C, L, 2*gru_hidden_dim)

        # Token-level pooling within each chunk: (B*C, gru_output_dim)
        chunk_vectors, self._last_token_weights = self._token_pooling(
            gru_out, attention_mask=flat_mask
        )

        # Reshape to (B, C, gru_output_dim) for document-level pooling
        chunk_vectors = chunk_vectors.view(B, C, -1)

        # Document-level pooling across chunks: (B, gru_output_dim)
        doc_vector, self._last_chunk_weights = self._chunk_pooling(
            chunk_vectors, attention_mask=chunk_mask
        )

        # Project to output dim: (B, projection_dim)
        features = self._projection(doc_vector)

        return features

    def get_state(self) -> Dict[str, Any]:
        return {
            'extractor_type': 'hierarchical_gru',
            'embedding_dim': self._embedding_dim,
            'gru_hidden_dim': self._gru_hidden_dim,
            'num_gru_layers': self._num_gru_layers,
            'chunk_size': self._chunk_size,
            'chunk_overlap': self._chunk_overlap,
            'max_chunks': self._max_chunks,
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
