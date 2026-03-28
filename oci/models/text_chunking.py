# oci/models/text_chunking.py
"""Shared token-based overlapping chunking utility.

Used by hierarchical extractors (hierarchical_llm, hierarchical_cnn, hierarchical_gru)
to split tokenized sequences into overlapping chunks for two-level pooling.
"""

from typing import List, Tuple

import torch


def chunk_token_ids(
    token_ids: List[int],
    chunk_size: int,
    chunk_overlap: int,
    max_chunks: int,
) -> List[List[int]]:
    """Split a token ID sequence into overlapping chunks.

    Args:
        token_ids: Full sequence of token IDs for one document.
        chunk_size: Number of tokens per chunk.
        chunk_overlap: Number of overlapping tokens between consecutive chunks.
        max_chunks: Maximum number of chunks to return (truncates the rest).

    Returns:
        List of token ID lists, one per chunk. Always returns at least one chunk.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be < chunk_size ({chunk_size})"
        )

    stride = chunk_size - chunk_overlap
    chunks = []
    start = 0

    while start < len(token_ids) and len(chunks) < max_chunks:
        end = start + chunk_size
        chunk = token_ids[start:end]
        if chunk:  # skip empty trailing chunk
            chunks.append(chunk)
        start += stride

    # Always return at least one chunk (possibly shorter than chunk_size)
    if not chunks:
        chunks = [token_ids[:chunk_size] if token_ids else []]

    return chunks


def pad_and_batch_chunks(
    batch_chunk_ids: List[List[List[int]]],
    pad_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad chunks across a batch to uniform dimensions.

    Args:
        batch_chunk_ids: List (batch) of list (chunks) of list (token IDs).
            batch_chunk_ids[b][c] is the token ID list for sample b, chunk c.
        pad_token_id: Token ID used for padding.

    Returns:
        input_ids: (B, max_chunks, max_chunk_len) long tensor
        attention_mask: (B, max_chunks, max_chunk_len) float tensor (1=real, 0=pad)
        chunk_mask: (B, max_chunks) float tensor (1=real chunk, 0=padding chunk)
    """
    B = len(batch_chunk_ids)
    max_chunks = max(len(sample_chunks) for sample_chunks in batch_chunk_ids)
    max_chunk_len = max(
        len(chunk)
        for sample_chunks in batch_chunk_ids
        for chunk in sample_chunks
    ) if max_chunks > 0 else 0

    input_ids = torch.full((B, max_chunks, max_chunk_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros(B, max_chunks, max_chunk_len)
    chunk_mask = torch.zeros(B, max_chunks)

    for b, sample_chunks in enumerate(batch_chunk_ids):
        for c, chunk in enumerate(sample_chunks):
            length = len(chunk)
            input_ids[b, c, :length] = torch.tensor(chunk, dtype=torch.long)
            attention_mask[b, c, :length] = 1.0
            chunk_mask[b, c] = 1.0

    return input_ids, attention_mask, chunk_mask
