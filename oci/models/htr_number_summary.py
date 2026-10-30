"""Per-chunk summaries of the numbers written in an HTR chunk.

Each number in a chunk becomes a mention: the chunk encoder's final hidden
state at the number's first token, which carries what the number measures from
the words around it, combined with an encoding of its magnitude. A few learned
queries attend over the chunk's mentions, and their projected output is added
to the chunk's vector before the document transformer. The projection starts
at zero and has no bias, so an untrained summary leaves every chunk vector
unchanged and a chunk without numbers always keeps its vector.
"""

from __future__ import annotations

import math
import re
from bisect import bisect_right
from typing import Any, Dict, Sequence, Tuple

import torch
import torch.nn as nn

# A number starts after a character that is not a letter, digit, underscore,
# or period, so identifiers such as "R05.9", "T2N0", and "BRCA1" give no value.
# Units may follow directly ("5mg"), and thousands separators stay with the
# number ("150,000").
HTR_NUMBER_PATTERN = re.compile(
    r"(?<![\w.])(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?"
)
# Larger values are identifiers rather than measurements; they share one
# encoding.
HTR_NUMBER_MAX_VALUE = 1e12
HTR_NUMBER_FREQUENCIES = 16

NumberMentions = Tuple[Tuple[int, float], ...]


def chunk_number_mentions(chunk: str, offsets: Sequence[Sequence[int]]) -> NumberMentions:
    """Return ``(token index, value)`` for each number, anchored at its first token.

    ``offsets`` is the tokenizer's character offset mapping for ``chunk``,
    including special tokens (which have empty spans).
    """
    spans = [
        (int(start), int(end), index)
        for index, (start, end) in enumerate(offsets)
        if int(end) > int(start)
    ]
    starts = [start for start, _, _ in spans]
    mentions = []
    for match in HTR_NUMBER_PATTERN.finditer(str(chunk or "")):
        position = bisect_right(starts, match.start()) - 1
        if position < 0 or spans[position][1] <= match.start():
            raise RuntimeError("HTR tokenizer offsets do not cover a number in the chunk")
        value = min(float(match.group(0).replace(",", "")), HTR_NUMBER_MAX_VALUE)
        mentions.append((spans[position][2], value))
    return tuple(mentions)


def tokenize_chunk_numbers(tokenizer: Any, chunk: str) -> NumberMentions:
    """Locate the numbers of one chunk in the encoder tokenizer's own tokens."""
    if HTR_NUMBER_PATTERN.search(str(chunk or "")) is None:
        return ()
    encoded = tokenizer(
        str(chunk or ""),
        padding=False,
        truncation=False,
        return_offsets_mapping=True,
    )
    offsets = encoded.get("offset_mapping")
    if offsets is None or len(offsets) != len(encoded["input_ids"]):
        raise RuntimeError("HTR number summaries need the tokenizer's character offsets")
    return chunk_number_mentions(chunk, offsets)


def collate_number_mentions(mentions_by_chunk: Sequence[NumberMentions]) -> Dict[str, torch.Tensor]:
    """Flatten per-chunk mentions in chunk order (token indices are unpadded)."""
    chunk_index, token_index, values = [], [], []
    for chunk, mentions in enumerate(mentions_by_chunk):
        for token, value in mentions:
            chunk_index.append(chunk)
            token_index.append(int(token))
            values.append(float(value))
    return {
        "number_chunk_index": torch.as_tensor(chunk_index, dtype=torch.long),
        "number_token_index": torch.as_tensor(token_index, dtype=torch.long),
        "number_value": torch.as_tensor(values, dtype=torch.float32),
    }


class ChunkNumberSummary(nn.Module):
    """Attend over each chunk's number mentions with ``num_queries`` queries."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        model_dim: int,
        num_queries: int,
        num_frequencies: int = HTR_NUMBER_FREQUENCIES,
    ):
        super().__init__()
        self._model_dim = int(model_dim)
        self._num_queries = int(num_queries)
        # Periodic features of log(1 + value) with trainable frequencies
        # (Gorishniy et al., 2022), plus the log value itself. Only magnitude
        # enters: a value's digit count or written precision does not.
        self.frequencies = nn.Parameter(torch.randn(int(num_frequencies)))
        self.value_projection = nn.Sequential(
            nn.Linear(2 * int(num_frequencies) + 1, self._model_dim),
            nn.ReLU(),
        )
        self.context_projection = nn.Linear(int(hidden_dim), self._model_dim)
        # The product term lets what a number measures gate how its value
        # matters.
        self.mention = nn.Sequential(
            nn.Linear(3 * self._model_dim, self._model_dim),
            nn.GELU(approximate="none"),
            nn.Linear(self._model_dim, self._model_dim),
            nn.LayerNorm(self._model_dim, eps=1e-5),
        )
        self.queries = nn.Parameter(torch.randn(self._num_queries, self._model_dim) * 0.02)
        self.keys = nn.Linear(self._model_dim, self._model_dim)
        self.values = nn.Linear(self._model_dim, self._model_dim)
        self.output = nn.Linear(self._num_queries * self._model_dim, self._model_dim, bias=False)
        nn.init.zeros_(self.output.weight)

    def encode_values(self, values: torch.Tensor) -> torch.Tensor:
        log_value = torch.log1p(values.clamp(min=0.0, max=HTR_NUMBER_MAX_VALUE))
        angles = 2.0 * math.pi * log_value[:, None] * self.frequencies[None, :]
        features = torch.cat(
            [log_value[:, None] / 10.0, torch.sin(angles), torch.cos(angles)],
            dim=1,
        )
        return self.value_projection(features)

    def forward(
        self,
        context: torch.Tensor,
        values: torch.Tensor,
        chunk_index: torch.Tensor,
        num_chunks: int,
    ) -> torch.Tensor:
        """Return one summary per chunk.

        ``context`` is ``(M, hidden_dim)`` and ``values`` is ``(M,)``.
        ``chunk_index`` is a sorted CPU tensor assigning each mention to one of
        ``num_chunks`` chunks.
        """
        device = context.device
        context = context.float()
        if context.shape[0] == 0:
            return context.new_zeros(int(num_chunks), self._model_dim)
        projected_context = self.context_projection(context)
        projected_value = self.encode_values(values)
        mentions = self.mention(
            torch.cat(
                [projected_context, projected_value, projected_context * projected_value],
                dim=1,
            )
        )
        # Lay the mentions out densely per chunk with a gather built on the
        # CPU, which keeps the pooling deterministic on CUDA.
        chunk_index = chunk_index.to("cpu", dtype=torch.long)
        counts = torch.bincount(chunk_index, minlength=int(num_chunks))
        width = int(counts.max().item())
        first = torch.cumsum(counts, dim=0) - counts
        slot = torch.arange(chunk_index.numel()) - first[chunk_index]
        gather = torch.full((int(num_chunks), width), chunk_index.numel(), dtype=torch.long)
        gather[chunk_index, slot] = torch.arange(chunk_index.numel())
        valid = (gather < chunk_index.numel()).to(device)
        gather = gather.to(device)
        padded = torch.cat([mentions, mentions.new_zeros(1, self._model_dim)], dim=0)[gather]

        scores = torch.einsum("cmd,qd->cmq", self.keys(padded), self.queries)
        scores = scores / math.sqrt(self._model_dim)
        # A finite fill keeps chunks without mentions free of NaN; zeroing
        # their weights gives them a zero summary (the output has no bias).
        scores = scores.masked_fill(~valid[:, :, None], torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1) * valid[:, :, None].to(scores.dtype)
        pooled = torch.einsum("cmq,cmd->cqd", weights, self.values(padded))
        return self.output(pooled.reshape(int(num_chunks), -1))
