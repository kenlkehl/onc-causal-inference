"""Byte-level hierarchical CNN extractor for raw clinical text.

This extractor intentionally avoids word vocabularies and clinical concept
rules.  It encodes UTF-8 bytes, learns local convolutional filters over byte
chunks, and pools chunks with learned attention.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .gated_attention_pooling import GatedAttentionPooling
from .simple_cnn_extractor import DilatedConvStack
from .text_chunking import chunk_token_ids, pad_and_batch_chunks

logger = logging.getLogger(__name__)


class ByteCNNExtractor(nn.Module):
    """Hierarchical byte CNN with token- and chunk-level gated attention."""

    def __init__(
        self,
        embedding_dim: int = 32,
        conv_dim: int = 64,
        kernel_size: int = 7,
        num_conv_blocks: int = 4,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        max_chunks: int = 128,
        gated_attention_dim: int = 64,
        projection_dim: int = 128,
        dropout: float = 0.1,
        lowercase: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        if chunk_overlap >= chunk_size:
            raise ValueError("byte chunk_overlap must be smaller than chunk_size")

        self._device = device or torch.device("cpu")
        self._embedding_dim = embedding_dim
        self._conv_dim = conv_dim
        self._kernel_size = kernel_size
        self._num_conv_blocks = num_conv_blocks
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._max_chunks = max_chunks
        self._gated_attention_dim = gated_attention_dim
        self._projection_dim = projection_dim
        self._dropout = dropout
        self._lowercase = lowercase

        # Byte ids are 1..256; 0 is padding.
        self._embedding = nn.Embedding(257, embedding_dim, padding_idx=0)
        self._cnn = DilatedConvStack(
            input_dim=embedding_dim,
            conv_dim=conv_dim,
            kernel_size=kernel_size,
            num_blocks=num_conv_blocks,
            dropout=dropout,
        )
        self._token_pooling = GatedAttentionPooling(
            hidden_dim=conv_dim,
            attention_dim=gated_attention_dim,
        )
        self._chunk_pooling = GatedAttentionPooling(
            hidden_dim=conv_dim,
            attention_dim=gated_attention_dim,
        )
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
            "ByteCNNExtractor initialized: embedding_dim=%d, conv_dim=%d, "
            "chunk_size=%d, max_chunks=%d, projection_dim=%d",
            embedding_dim,
            conv_dim,
            chunk_size,
            max_chunks,
            projection_dim,
        )

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def fit_tokenizer(self, texts: List[str]) -> None:
        """No-op for byte vocabulary; kept for trainable-extractor compatibility."""
        del texts

    @staticmethod
    def _texts_from_input(texts_or_batch: Any) -> List[str]:
        if isinstance(texts_or_batch, dict):
            return [str(text) for text in texts_or_batch.get("texts", [])]
        return [str(text) for text in texts_or_batch]

    def _encode_text(self, text: str) -> List[int]:
        if self._lowercase:
            text = text.lower()
        max_len = self._chunk_size * self._max_chunks
        raw = text.encode("utf-8", errors="ignore")[:max_len]
        return [byte + 1 for byte in raw]

    def _batch_chunks(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_chunk_ids = []
        for text in texts:
            token_ids = self._encode_text(text)
            chunks = chunk_token_ids(
                token_ids,
                self._chunk_size,
                self._chunk_overlap,
                self._max_chunks,
            )
            if chunks == [[]]:
                chunks = [[0]]
            batch_chunk_ids.append(chunks)

        input_ids, attn_mask, chunk_mask = pad_and_batch_chunks(batch_chunk_ids, pad_token_id=0)
        return (
            input_ids.to(self._device),
            attn_mask.to(self._device),
            chunk_mask.to(self._device),
        )

    def forward(self, texts_or_batch: Any) -> torch.Tensor:
        texts = self._texts_from_input(texts_or_batch)
        input_ids, attn_mask, chunk_mask = self._batch_chunks(texts)

        batch_size, n_chunks, chunk_len = input_ids.shape
        flat_ids = input_ids.view(batch_size * n_chunks, chunk_len)
        flat_mask = attn_mask.view(batch_size * n_chunks, chunk_len)

        embedded = self._embedding(flat_ids)
        encoded = self._cnn(embedded)
        chunk_vectors, self._last_token_weights = self._token_pooling(
            encoded,
            attention_mask=flat_mask,
        )
        chunk_vectors = chunk_vectors.view(batch_size, n_chunks, -1)
        doc_vector, self._last_chunk_weights = self._chunk_pooling(
            chunk_vectors,
            attention_mask=chunk_mask,
        )
        return self._projection(doc_vector)

    def extract_shared_forest_features(
        self,
        texts_or_batch: Any,
        text_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Expose learned neural features to both X and W forest matrices."""
        if text_features is not None:
            return text_features
        return self.forward(texts_or_batch)

    def get_state(self) -> Dict[str, Any]:
        return {
            "extractor_type": "byte_cnn",
            "embedding_dim": self._embedding_dim,
            "conv_dim": self._conv_dim,
            "kernel_size": self._kernel_size,
            "num_conv_blocks": self._num_conv_blocks,
            "chunk_size": self._chunk_size,
            "chunk_overlap": self._chunk_overlap,
            "max_chunks": self._max_chunks,
            "gated_attention_dim": self._gated_attention_dim,
            "projection_dim": self._projection_dim,
            "dropout": self._dropout,
            "lowercase": self._lowercase,
            "output_dim": self._output_dim,
        }

    def get_num_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}

    def to(self, device):
        self._device = device if isinstance(device, torch.device) else torch.device(device)
        return super().to(device)
