# cdt/models/chunking.py
"""Token-based chunking utilities for hierarchical text processing.

This module provides functions to split text into overlapping token-based chunks
instead of sentence-based splitting. This approach:
1. Provides more consistent chunk sizes
2. Preserves context across chunk boundaries via overlap
3. Works uniformly with any text (not dependent on sentence structure)

Two chunking functions are provided:
- split_into_chunks_hf: For HuggingFace tokenizers (BERT-based extractors)
- split_into_chunks_vocab: For learned vocabularies (GRU-based extractors)
"""

import warnings
import re
from typing import List, Optional, Callable, Union


def split_into_chunks_hf(
    text: str,
    tokenizer,  # HuggingFace PreTrainedTokenizer
    chunk_size: int = 128,
    chunk_overlap: int = 32,
    max_chunks: int = 100
) -> List[str]:
    """
    Split text into overlapping token chunks using HuggingFace tokenizer.

    Used by: hierarchical_transformer, gated_mil_hierarchical, confounder (non-GRU mode)

    The text is tokenized, split into overlapping windows, and each window is
    decoded back to text strings for BERT encoding.

    Args:
        text: Input text string
        tokenizer: HuggingFace PreTrainedTokenizer (e.g., from AutoTokenizer)
        chunk_size: Number of tokens per chunk (N)
        chunk_overlap: Number of overlapping tokens between chunks (M)
        max_chunks: Maximum number of chunks to return

    Returns:
        List of chunk strings (decoded from token windows)

    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        >>> chunks = split_into_chunks_hf("This is a long text...", tokenizer, chunk_size=128, chunk_overlap=32)
    """
    if not text or not text.strip():
        return [""]

    # Tokenize entire text without special tokens (we'll add them during encoding)
    tokens = tokenizer.encode(text, add_special_tokens=False)

    if len(tokens) == 0:
        # Fallback for empty tokenization
        return [text[:500]] if text.strip() else [""]

    # Calculate stride (step size between chunk starts)
    stride = max(1, chunk_size - chunk_overlap)

    chunks = []
    for start in range(0, len(tokens), stride):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]

        # Decode back to text for BERT encoding
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)

        if chunk_text.strip():
            chunks.append(chunk_text)

        # Stop if we've reached max chunks or processed all tokens
        if len(chunks) >= max_chunks or end >= len(tokens):
            break

    # Fallback if no valid chunks were created
    return chunks if chunks else [text[:500]]


def split_into_chunks_vocab(
    text: str,
    word_to_idx: dict,
    tokenize_fn: Callable[[str], List[str]],
    chunk_size: int = 128,
    chunk_overlap: int = 32,
    max_chunks: int = 100
) -> List[List[int]]:
    """
    Split text into overlapping token chunks using learned vocabulary.

    Used by: GRU-based extractors that build vocabulary from training data.

    Unlike split_into_chunks_hf, this returns token ID lists directly (not decoded
    text strings), which is more efficient for learned vocabularies.

    Args:
        text: Input text string
        word_to_idx: Dictionary mapping words to token indices
        tokenize_fn: Function to tokenize text into words (e.g., lambda t: t.lower().split())
        chunk_size: Number of tokens per chunk (N)
        chunk_overlap: Number of overlapping tokens between chunks (M)
        max_chunks: Maximum number of chunks to return

    Returns:
        List of token ID lists (one list per chunk)

    Example:
        >>> word_to_idx = {'the': 1, 'cat': 2, 'sat': 3, '<unk>': 0}
        >>> tokenize_fn = lambda t: t.lower().split()
        >>> chunks = split_into_chunks_vocab("The cat sat on the mat", word_to_idx, tokenize_fn)
    """
    if not text or not text.strip():
        pad_idx = word_to_idx.get('<pad>', 0)
        return [[pad_idx]]

    # Tokenize text using the extractor's tokenization
    words = tokenize_fn(text)

    if len(words) == 0:
        unk_idx = word_to_idx.get('<unk>', 0)
        return [[unk_idx]]

    # Convert to indices with UNK handling
    unk_idx = word_to_idx.get('<unk>', 0)
    token_ids = [word_to_idx.get(w, unk_idx) for w in words]

    # Calculate stride
    stride = max(1, chunk_size - chunk_overlap)

    chunks = []
    for start in range(0, len(token_ids), stride):
        end = min(start + chunk_size, len(token_ids))
        chunk_ids = token_ids[start:end]
        chunks.append(chunk_ids)

        # Stop if we've reached max chunks or processed all tokens
        if len(chunks) >= max_chunks or end >= len(token_ids):
            break

    # Fallback if no chunks were created
    return chunks if chunks else [[unk_idx]]


def split_into_sentences(text: str, max_sentences: int = 100) -> List[str]:
    """
    DEPRECATED: Use split_into_chunks_hf or split_into_chunks_vocab instead.

    Split text into sentences using regex-based heuristics.
    Kept for backward compatibility only.

    Args:
        text: Input text string
        max_sentences: Maximum number of sentences to return

    Returns:
        List of sentence strings
    """
    warnings.warn(
        "split_into_sentences is deprecated. Use split_into_chunks_hf or "
        "split_into_chunks_vocab instead for token-based chunking.",
        DeprecationWarning,
        stacklevel=2
    )

    if not text or not text.strip():
        return [""]

    # Original regex-based sentence splitting
    # Matches period/exclamation/question followed by space and capital letter
    pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$'
    sentences = re.split(pattern, text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    return sentences[:max_sentences] if sentences else [text[:500]]


def get_chunk_boundaries(
    num_tokens: int,
    chunk_size: int,
    chunk_overlap: int,
    max_chunks: int = 100
) -> List[tuple]:
    """
    Calculate chunk boundaries for a given number of tokens.

    Useful for understanding how text will be split without actually
    performing tokenization.

    Args:
        num_tokens: Total number of tokens
        chunk_size: Tokens per chunk
        chunk_overlap: Overlap between chunks
        max_chunks: Maximum chunks

    Returns:
        List of (start, end) tuples for each chunk
    """
    stride = max(1, chunk_size - chunk_overlap)
    boundaries = []

    for start in range(0, num_tokens, stride):
        end = min(start + chunk_size, num_tokens)
        boundaries.append((start, end))

        if len(boundaries) >= max_chunks or end >= num_tokens:
            break

    return boundaries
