"""Focused coverage for the HTR extractor retained by Stage 1."""

from __future__ import annotations

import pytest

from oci.models.hierarchical_transformer_extractor import (
    HierarchicalTransformerExtractor,
    split_text_into_word_chunks,
)
from oci.models.lossless_tokenization import SemanticTruncationError


SAMPLE_TEXTS = (
    "Patient has metastatic lung cancer and performance status one.",
    "Treatment began after progression on first-line therapy.",
)


def _hash_extractor(**overrides):
    options = {
        "sentence_encoder_model": "hash",
        "chunk_size_words": 5,
        "chunk_overlap_words": 1,
        "max_chunks": 8,
        "num_transformer_layers": 1,
        "num_attention_heads": 2,
        "transformer_dim": 32,
        "projection_dim": 16,
        "hash_embedding_dim": 32,
        "transformer_dropout": 0.0,
    }
    options.update(overrides)
    return HierarchicalTransformerExtractor(**options)


def test_hash_backend_forward_shape():
    output = _hash_extractor()(list(SAMPLE_TEXTS))
    assert tuple(output.shape) == (2, 16)


def test_word_chunking_fails_closed_when_capacity_would_truncate():
    text = " ".join(f"word{index}" for index in range(12))
    with pytest.raises(SemanticTruncationError, match="semantic truncation is forbidden"):
        split_text_into_word_chunks(
            text,
            chunk_size_words=4,
            chunk_overlap_words=1,
            max_chunks=2,
        )
