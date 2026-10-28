"""Lossless sentence-chunk utilities used by Stage 1 embedding evidence."""

from __future__ import annotations

import re
from typing import List

from .lossless_tokenization import SemanticTruncationError


_WORD_RE = re.compile(r"\S+")

# Chunks with fewer letters and digits than this carry no readable clinical
# text: a short tail already contained in the previous chunk's overlap, an
# empty record, or a run of symbols such as soft hyphens.
MIN_INFORMATIVE_CHUNK_ALNUM_CHARS = 12


def informative_chunk_text(text: str) -> bool:
    """Return True when a chunk has enough letters and digits to be evidence."""
    count = 0
    for character in str(text or ""):
        if character.isalnum():
            count += 1
            if count >= MIN_INFORMATIVE_CHUNK_ALNUM_CHARS:
                return True
    return False


def chunk_text_words(
    text: str,
    chunk_size_words: int,
    chunk_overlap_words: int,
    max_chunks: int,
    chunk_selection: str = "first",
) -> List[str]:
    """Split text into overlapping word chunks.

    This intentionally uses simple whitespace word spans. Sentence-transformer
    tokenizers handle subword details after chunking, while this function keeps
    the cache key and chunk boundaries model-independent.
    """
    if chunk_size_words < 1:
        raise ValueError("chunk_size_words must be >= 1")
    if max_chunks < 1:
        raise ValueError("max_chunks must be >= 1")
    if chunk_overlap_words >= chunk_size_words:
        raise ValueError(
            "chunk_overlap_words must be smaller than chunk_size_words"
        )
    if chunk_overlap_words < 0:
        raise ValueError("chunk_overlap_words must be >= 0")
    selection = str(chunk_selection).strip().lower()
    if selection not in {"first", "last"}:
        raise ValueError("chunk_selection must be 'first' or 'last'")

    words = [match.group(0) for match in _WORD_RE.finditer(text or "")]
    if not words:
        return [""]

    stride = chunk_size_words - chunk_overlap_words
    chunks: List[str] = []
    start = 0
    while start < len(words):
        chunk_words = words[start:start + chunk_size_words]
        if chunk_words:
            chunks.append(" ".join(chunk_words))
        start += stride

    if len(chunks) > max_chunks:
        raise SemanticTruncationError(
            "Concept embedding note requires "
            f"{len(chunks)} chunks but configured max_chunks={max_chunks}; "
            "semantic truncation is forbidden. Increase max_chunks so the "
            "capacity is nonbinding."
        )
    return chunks or [""]


def split_text_to_token_chunks(
    text: str,
    tokenizer,
    max_seq_length: int,
    chunk_overlap_tokens: int = 0,
    encoding_prefix: str = "",
) -> List[str]:
    """Split text into chunks whose exact re-encoding fits ``max_seq_length``.

    Tokenizer decode/encode round trips are not necessarily length preserving.
    In particular, a WordPiece slice that begins with a continuation token can
    decode to text beginning with ``##`` and then consume additional tokens when
    encoded again.  ``encoding_prefix`` represents framing added by the caller
    before model tokenization (for example, a model-specific document marker).
    Every returned chunk is checked with that prefix and special tokens included.
    """
    if max_seq_length < 1:
        raise ValueError("max_seq_length must be >= 1")
    if chunk_overlap_tokens < 0:
        raise ValueError("chunk_overlap_tokens must be >= 0")

    raw_text = text or ""
    prefix = str(encoding_prefix)
    if _encoded_length(tokenizer, f"{prefix}{raw_text}") <= max_seq_length:
        return [raw_text]

    token_ids = _encode_without_special_tokens(tokenizer, raw_text)
    if not token_ids:
        raise SemanticTruncationError(
            "Model input framing exceeds max_seq_length even though the text has "
            "no content tokens; semantic truncation is forbidden."
        )

    content_limit = max(1, max_seq_length - _num_special_tokens(tokenizer))
    overlap = min(int(chunk_overlap_tokens), content_limit - 1)
    chunks: List[str] = []
    start = 0
    while start < len(token_ids):
        end = min(start + content_limit, len(token_ids))
        decoded = ""
        while end > start:
            decoded = _decode_token_ids(tokenizer, token_ids[start:end]).strip()
            if (
                decoded
                and _encoded_length(tokenizer, f"{prefix}{decoded}")
                <= max_seq_length
            ):
                break
            end -= 1
        if end <= start:
            raise SemanticTruncationError(
                "A single decoded content token plus model input framing exceeds "
                f"max_seq_length={max_seq_length}; semantic truncation is forbidden."
            )
        chunks.append(decoded)
        if end >= len(token_ids):
            break
        # Advance from the verified end, not the original proposal: shrinking a
        # chunk must not create a gap in the source token stream.  Force forward
        # progress when the requested overlap is as large as the fitted chunk.
        start = max(start + 1, end - overlap)
    return chunks or [""]


def _encode_without_special_tokens(tokenizer, text: str) -> List[int]:
    return _encode_token_ids(tokenizer, text, add_special_tokens=False)


def _encoded_length(tokenizer, text: str) -> int:
    return len(_encode_token_ids(tokenizer, text, add_special_tokens=True))


def _encode_token_ids(
    tokenizer,
    text: str,
    *,
    add_special_tokens: bool,
) -> List[int]:
    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        try:
            # Hugging Face tokenizers warn when this boundary-discovery pass is
            # longer than model_max_length, even though it is never sent to the
            # model and is immediately split into bounded chunks below.
            token_ids = encode(
                text,
                add_special_tokens=add_special_tokens,
                verbose=False,
            )
        except TypeError:
            # Lightweight/custom tokenizers may not expose the optional
            # Hugging Face ``verbose`` argument.
            token_ids = encode(text, add_special_tokens=add_special_tokens)
    else:
        try:
            encoded = tokenizer(
                text,
                add_special_tokens=add_special_tokens,
                verbose=False,
            )
        except TypeError:
            encoded = tokenizer(text, add_special_tokens=add_special_tokens)
        token_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return [int(token_id) for token_id in token_ids]


def _decode_token_ids(tokenizer, token_ids: List[int]) -> str:
    try:
        return tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode(token_ids, skip_special_tokens=True)


def _num_special_tokens(tokenizer) -> int:
    fn = getattr(tokenizer, "num_special_tokens_to_add", None)
    if callable(fn):
        try:
            return max(0, int(fn(pair=False)))
        except TypeError:
            return max(0, int(fn(False)))
    return 0
