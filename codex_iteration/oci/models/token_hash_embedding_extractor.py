"""Neural token-hash embedding extractor.

This extractor preserves subword identity without hard-coding any clinical
concept strings. Raw text is tokenized with a pretrained tokenizer, generic
token n-grams are hashed to buckets, and learned EmbeddingBag layers pool those
bucket IDs into a document representation.
"""

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TokenHashEmbeddingExtractor(nn.Module):
    """Learned embedding-bag representation over generic token-ID hashes."""

    def __init__(
        self,
        model_name: str = "google/medgemma-1.5-4b-it",
        max_length: int = 50000,
        num_hash_buckets: int = 32768,
        projection_dim: int = 128,
        ngram_orders: Sequence[int] = (1, 2, 3),
        front_tokens: int = 4096,
        dropout: float = 0.1,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        if num_hash_buckets < 1024:
            raise ValueError("num_hash_buckets must be at least 1024")
        if projection_dim < 1:
            raise ValueError("projection_dim must be positive")
        if not ngram_orders:
            raise ValueError("ngram_orders must contain at least one order")

        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers is required for TokenHashEmbeddingExtractor"
            ) from exc

        self._device = device or torch.device("cpu")
        self._model_name = model_name
        self._max_length = max_length
        self._num_hash_buckets = num_hash_buckets
        self._projection_dim = projection_dim
        self._ngram_orders = tuple(int(n) for n in ngram_orders)
        self._front_tokens = front_tokens
        self._dropout = dropout
        self._hash_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="right",
            truncation_side="left",
        )
        if self._tokenizer.pad_token is None and self._tokenizer.eos_token is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._all_bag = nn.EmbeddingBag(
            num_embeddings=num_hash_buckets,
            embedding_dim=projection_dim,
            mode="sum",
        )
        self._front_bag = nn.EmbeddingBag(
            num_embeddings=num_hash_buckets,
            embedding_dim=projection_dim,
            mode="sum",
        )
        self._output_dim = projection_dim
        self._projection = nn.Sequential(
            nn.LayerNorm(projection_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim * 2, projection_dim),
            nn.LayerNorm(projection_dim),
        )

        logger.info(
            "TokenHashEmbeddingExtractor initialized: model=%s, max_length=%d, "
            "hash_buckets=%d, projection_dim=%d, ngrams=%s, front_tokens=%d",
            model_name,
            max_length,
            num_hash_buckets,
            projection_dim,
            self._ngram_orders,
            front_tokens,
        )
        self.to(self._device)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @staticmethod
    def _texts_from_input(texts_or_batch: Any) -> List[str]:
        if isinstance(texts_or_batch, dict):
            return [str(text) for text in texts_or_batch.get("texts", [])]
        return [str(text) for text in texts_or_batch]

    def _hash_ngrams(self, ids: np.ndarray) -> np.ndarray:
        parts = []
        for order in self._ngram_orders:
            if len(ids) < order:
                continue
            if order == 1:
                hashed = ids * 1_000_003 + 9_176
            else:
                hashed = np.zeros(len(ids) - order + 1, dtype=np.int64)
                for offset in range(order):
                    hashed += ids[offset: len(ids) - order + 1 + offset] * (
                        1_000_003 + 9_176 * (offset + 1)
                    )
                hashed += 19_260_817 * order
            parts.append(np.remainder(hashed, self._num_hash_buckets))
        if not parts:
            return np.zeros(1, dtype=np.int64)
        return np.unique(np.concatenate(parts)).astype(np.int64, copy=False)

    def _hash_text(self, text: str) -> Tuple[np.ndarray, np.ndarray]:
        cached = self._hash_cache.get(text)
        if cached is not None:
            return cached

        encoding = self._tokenizer(
            text,
            truncation=True,
            max_length=self._max_length,
            padding=False,
        )
        ids = np.asarray(encoding["input_ids"], dtype=np.int64)
        if ids.size == 0:
            ids = np.zeros(1, dtype=np.int64)

        all_hashes = self._hash_ngrams(ids)
        front_ids = ids[: self._front_tokens] if self._front_tokens > 0 else ids[:0]
        front_hashes = self._hash_ngrams(front_ids) if front_ids.size else np.zeros(1, dtype=np.int64)

        result = (all_hashes, front_hashes)
        self._hash_cache[text] = result
        return result

    def _bag(
        self,
        hash_lists: List[np.ndarray],
        embedding: nn.EmbeddingBag,
    ) -> torch.Tensor:
        lengths = [max(len(ids), 1) for ids in hash_lists]
        offsets = np.zeros(len(hash_lists), dtype=np.int64)
        if len(hash_lists) > 1:
            offsets[1:] = np.cumsum(lengths[:-1])
        flat = np.concatenate([
            ids if len(ids) else np.zeros(1, dtype=np.int64)
            for ids in hash_lists
        ])

        flat_t = torch.as_tensor(flat, dtype=torch.long, device=self._device)
        offsets_t = torch.as_tensor(offsets, dtype=torch.long, device=self._device)
        pooled = embedding(flat_t, offsets_t)
        denom = torch.as_tensor(lengths, dtype=pooled.dtype, device=self._device)
        return pooled / denom.sqrt().unsqueeze(1).clamp_min(1.0)

    def forward(self, texts_or_batch: Any) -> torch.Tensor:
        texts = self._texts_from_input(texts_or_batch)
        all_hashes = []
        front_hashes = []
        for text in texts:
            all_ids, front_ids = self._hash_text(text)
            all_hashes.append(all_ids)
            front_hashes.append(front_ids)

        all_vec = self._bag(all_hashes, self._all_bag)
        front_vec = self._bag(front_hashes, self._front_bag)
        combined = torch.cat([all_vec, front_vec], dim=1)
        return self._projection(combined)

    def extract_shared_forest_features(
        self,
        texts_or_batch: Any,
        text_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Expose learned neural features symmetrically to X and W."""
        if text_features is not None:
            return text_features
        return self.forward(texts_or_batch)

    def fit_tokenizer(self, texts: List[str]) -> None:
        """No-op; this extractor uses a pretrained tokenizer."""
        del texts

    def get_state(self) -> Dict[str, Any]:
        return {
            "extractor_type": "token_hash_embedding",
            "model_name": self._model_name,
            "max_length": self._max_length,
            "num_hash_buckets": self._num_hash_buckets,
            "projection_dim": self._projection_dim,
            "ngram_orders": self._ngram_orders,
            "front_tokens": self._front_tokens,
            "dropout": self._dropout,
            "output_dim": self._output_dim,
        }

    def get_num_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}

    def to(self, device):
        self._device = device if isinstance(device, torch.device) else torch.device(device)
        return super().to(device)
