"""Generic token-window selection for long clinical documents."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch


VALID_DOCUMENT_WINDOWS = {"head", "tail", "head_tail"}


def validate_document_window(document_window: str) -> str:
    """Validate and normalize a document-window mode."""
    if document_window not in VALID_DOCUMENT_WINDOWS:
        valid = ", ".join(sorted(VALID_DOCUMENT_WINDOWS))
        raise ValueError(f"document_window must be one of {{{valid}}}, got {document_window!r}")
    return document_window


def select_token_window(
    input_ids: Sequence[int],
    max_length: int,
    document_window: str = "tail",
) -> List[int]:
    """Select a generic token window from a long document.

    ``head_tail`` keeps the beginning and end of the note. This is deliberately
    content-agnostic: it does not inspect token strings or clinical concepts.
    """
    document_window = validate_document_window(document_window)
    ids = [int(token_id) for token_id in input_ids]
    if max_length <= 0 or len(ids) <= max_length:
        return ids
    if document_window == "head":
        return ids[:max_length]
    if document_window == "tail":
        return ids[-max_length:]

    head_len = max_length // 2
    tail_len = max_length - head_len
    if head_len <= 0:
        return ids[-tail_len:]
    if tail_len <= 0:
        return ids[:head_len]
    return ids[:head_len] + ids[-tail_len:]


def tokenize_with_document_window(
    tokenizer: Any,
    texts: Sequence[str],
    max_length: int,
    document_window: str = "tail",
    pad_to_length: Optional[int] = None,
    return_tensors: Optional[str] = None,
    return_length: bool = False,
) -> Dict[str, Any]:
    """Tokenize texts and apply a content-agnostic document window.

    Args:
        tokenizer: HuggingFace tokenizer.
        texts: Batch of input strings.
        max_length: Maximum selected token count per document.
        document_window: ``head``, ``tail``, or ``head_tail``.
        pad_to_length: If set, pad all sequences to this length.
        return_tensors: Currently supports ``"pt"`` or ``None``.
        return_length: Include selected sequence lengths in the result.
    """
    document_window = validate_document_window(document_window)
    encodings = tokenizer(
        list(texts),
        truncation=False,
        padding=False,
    )
    selected = [
        select_token_window(ids, max_length=max_length, document_window=document_window)
        for ids in encodings["input_ids"]
    ]
    lengths = [len(ids) for ids in selected]

    if pad_to_length is None and return_tensors == "pt":
        pad_to_length = max(lengths) if lengths else 0

    attention_mask = [[1] * length for length in lengths]
    if pad_to_length is not None:
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        padded_ids = []
        padded_masks = []
        for ids, mask in zip(selected, attention_mask):
            clipped_ids = ids[:pad_to_length]
            clipped_mask = mask[:pad_to_length]
            pad_len = max(0, pad_to_length - len(clipped_ids))
            padded_ids.append(clipped_ids + [int(pad_token_id)] * pad_len)
            padded_masks.append(clipped_mask + [0] * pad_len)
        selected = padded_ids
        attention_mask = padded_masks

    result: Dict[str, Any] = {
        "input_ids": selected,
        "attention_mask": attention_mask,
    }
    if return_length:
        result["length"] = lengths

    if return_tensors is None:
        return result
    if return_tensors != "pt":
        raise ValueError("tokenize_with_document_window only supports return_tensors='pt'")
    result["input_ids"] = torch.as_tensor(result["input_ids"], dtype=torch.long)
    result["attention_mask"] = torch.as_tensor(result["attention_mask"], dtype=torch.long)
    if return_length:
        result["length"] = torch.as_tensor(lengths, dtype=torch.long)
    return result
