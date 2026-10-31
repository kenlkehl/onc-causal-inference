"""Masked-language-model pretraining of an HTR chunk encoder on given texts.

The encoder sees the chunks the HTR builds from the texts, tokenized as the HTR
tokenizes them, with BERT's masking: each non-special token is selected with
probability ``mask_probability``, and a selected token becomes the mask token
(80%), a random token (10%), or stays unchanged (10%). The prediction head
starts from the checkpoint's pretrained masked-LM head when it has one, and its
output layer is the encoder's input embedding matrix. Only the texts passed in
are read, so the caller decides whose text the encoder sees.
"""

from __future__ import annotations

import contextlib
import logging
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

MLM_HEAD_PREFIX = "cls.predictions."


class MaskedLMHead(nn.Module):
    """BERT's masked-LM head, decoding through the encoder's input embeddings."""

    def __init__(self, config: Any):
        super().__init__()
        from transformers.models.bert.modeling_bert import BertPredictionHeadTransform

        self.transform = BertPredictionHeadTransform(config)
        self.bias = nn.Parameter(torch.zeros(int(config.vocab_size)))

    def forward(self, hidden: torch.Tensor, embedding_weight: torch.Tensor) -> torch.Tensor:
        return F.linear(self.transform(hidden), embedding_weight, self.bias)


def _checkpoint_state(extractor: Any) -> Dict[str, torch.Tensor]:
    path = Path(extractor._resolve_sentence_encoder_path())
    if (path / "model.safetensors").exists():
        from safetensors.torch import load_file

        return load_file(str(path / "model.safetensors"))
    if (path / "pytorch_model.bin").exists():
        return torch.load(path / "pytorch_model.bin", map_location="cpu", weights_only=True)
    return {}


def load_pretrained_head(head: MaskedLMHead, extractor: Any) -> bool:
    """Copy the checkpoint's masked-LM head into ``head``; False when it has none."""

    state = _checkpoint_state(extractor)
    transform = {
        key[len(MLM_HEAD_PREFIX + "transform.") :]: value
        for key, value in state.items()
        if key.startswith(MLM_HEAD_PREFIX + "transform.")
    }
    bias = state.get(MLM_HEAD_PREFIX + "bias", state.get(MLM_HEAD_PREFIX + "decoder.bias"))
    if bias is None or set(transform) != set(head.transform.state_dict()):
        return False
    head.transform.load_state_dict(transform)
    with torch.no_grad():
        head.bias.copy_(bias)
    return True


def pretrain_chunk_encoder(
    extractor: Any,
    texts: Sequence[str],
    *,
    epochs: int,
    mask_probability: float,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    gradient_clip_norm: float = 1.0,
    label: str = "",
) -> Dict[str, Any]:
    """Pretrain ``extractor``'s chunk encoder in place; return its weights and history.

    The learning rate decays linearly to a tenth of ``learning_rate``. Masks and
    batch order come from a generator seeded from Torch's global generator, so
    a seeded caller gets the same corpus passes.
    """

    if extractor._effective_sentence_encoder_backend() != "transformers":
        raise ValueError("masked-language-model pretraining needs the HTR transformers backend")
    extractor._ensure_encoder_initialized()
    encoder = extractor._sentence_encoder
    tokenizer = extractor._tokenizer
    if tokenizer.mask_token_id is None:
        raise ValueError("masked-language-model pretraining needs a tokenizer with a mask token")
    device = next(encoder.parameters()).device
    chunks = [chunk for text_chunks in extractor._chunks_for_texts(list(texts)) for chunk in text_chunks]
    # The HTR's own tokenization, which refuses chunks longer than max_chunk_length.
    sequences = [extractor._tokenize_one_chunk_for_transformers(chunk)[0] for chunk in chunks]
    if not sequences:
        raise ValueError("masked-language-model pretraining received no text")
    special = torch.tensor(sorted(set(tokenizer.all_special_ids)), dtype=torch.long)
    vocab_size = int(encoder.config.vocab_size)
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)

    head = MaskedLMHead(encoder.config).to(device)
    pretrained_head = load_pretrained_head(head, extractor)
    parameters = [p for p in encoder.parameters() if p.requires_grad] + list(head.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=float(learning_rate), weight_decay=float(weight_decay))
    steps_per_epoch = math.ceil(len(sequences) / int(batch_size))
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1, total_iters=max(1, steps_per_epoch * int(epochs))
    )
    generator = torch.Generator().manual_seed(int(torch.randint(0, 2**31 - 1, (1,)).item()))
    logger.info(
        "%s masked-LM pretraining: %s chunks from %s texts, %s epoch(s), batch_size=%s, "
        "mask_probability=%.3g, lr=%.3g, weight_decay=%.3g, head=%s",
        label,
        len(sequences),
        len(texts),
        epochs,
        batch_size,
        mask_probability,
        learning_rate,
        weight_decay,
        "pretrained" if pretrained_head else "random",
    )
    encoder.train()
    head.train()
    history: List[Dict[str, float]] = []
    for epoch in range(1, int(epochs) + 1):
        order = torch.randperm(len(sequences), generator=generator).tolist()
        loss_sum, correct, masked = 0.0, 0, 0
        for start in range(0, len(order), int(batch_size)):
            batch = [sequences[index] for index in order[start : start + int(batch_size)]]
            width = max(len(sequence) for sequence in batch)
            ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
            attention = torch.zeros((len(batch), width), dtype=torch.long)
            for row, sequence in enumerate(batch):
                ids[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
                attention[row, : len(sequence)] = 1
            candidates = attention.bool() & ~torch.isin(ids, special)
            selected = candidates & (torch.rand(ids.shape, generator=generator) < mask_probability)
            roll = torch.rand(ids.shape, generator=generator)
            random_tokens = torch.randint(vocab_size, ids.shape, generator=generator)
            if not bool(selected.any()):
                continue
            labels = ids[selected]
            corrupted = ids.clone()
            corrupted[selected & (roll < 0.8)] = int(tokenizer.mask_token_id)
            swap = selected & (roll >= 0.8) & (roll < 0.9)
            corrupted[swap] = random_tokens[swap]
            rows, columns = selected.nonzero(as_tuple=True)
            hidden = encoder(
                input_ids=corrupted.to(device), attention_mask=attention.to(device)
            ).last_hidden_state
            logits = head(
                hidden[rows.to(device), columns.to(device)],
                encoder.embeddings.word_embeddings.weight,
            )
            labels = labels.to(device)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(parameters, float(gradient_clip_norm))
            optimizer.step()
            scheduler.step()
            loss_sum += float(loss.detach()) * len(labels)
            correct += int((logits.detach().argmax(dim=1) == labels).sum())
            masked += len(labels)
        history.append(
            {
                "epoch": epoch,
                "loss": loss_sum / max(1, masked),
                "accuracy": correct / max(1, masked),
                "masked_tokens": masked,
            }
        )
        logger.info(
            "%s masked-LM epoch %s/%s: loss=%.4f accuracy=%.4f masked_tokens=%s lr=%.3g",
            label,
            epoch,
            epochs,
            history[-1]["loss"],
            history[-1]["accuracy"],
            masked,
            optimizer.param_groups[0]["lr"],
        )
    encoder.eval()
    return {
        "encoder": {key: value.detach().cpu().numpy() for key, value in encoder.state_dict().items()},
        "history": history,
        "chunks": len(sequences),
        "texts": len(texts),
        "head": "pretrained" if pretrained_head else "random",
    }


@contextlib.contextmanager
def preserved_random_state(device: Any) -> Iterator[None]:
    """Leave Python's, NumPy's, and Torch's global random streams as they were."""

    states = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    cuda = torch.device(device).type == "cuda" and torch.cuda.is_available()
    cuda_state = torch.cuda.get_rng_state(device) if cuda else None
    try:
        yield
    finally:
        random.setstate(states[0])
        np.random.set_state(states[1])
        torch.set_rng_state(states[2])
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, device)


def load_encoder_weights(extractor: Any, weights: Dict[str, np.ndarray]) -> None:
    """Load weights from :func:`pretrain_chunk_encoder` into ``extractor``'s encoder."""

    extractor._ensure_encoder_initialized()
    encoder = extractor._sentence_encoder
    device = next(encoder.parameters()).device
    encoder.load_state_dict(
        {key: torch.as_tensor(np.asarray(value)).to(device) for key, value in weights.items()},
        strict=True,
    )
