"""Random-chunk null for the Stage 1 lexical recovery metric (post-hoc only).

The lexical recovery metric is the best token recall of an oracle feature's
name tokens over an architecture's evidence texts.  Long excerpts contain many
tokens, so a high value can arise by chance.  This module compares the
observed value in each discovery context with the same metric computed on
random text windows from that context's training rows, matched to the
evidence items in number and character length.

A window starts at a token boundary of a training row long enough to hold it
(a row shorter than every item contributes its whole text and is counted as
unmatched).  Token recall on a window equals the recall obtained by
lowercasing the window and splitting it into maximal ``[a-z0-9]`` runs, the
tokenization used for the observed metric; the vectorized implementation
below reproduces that exactly, including a token cut at the window's end.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> set[str]:
    return set(_TOKEN.findall(str(text or "").lower()))


def best_token_recall(texts: Iterable[str], tokens: frozenset[str]) -> float | None:
    """Return max over texts of |tokens ∩ tokens(text)| / |tokens|."""

    if not tokens:
        return None
    return max(
        (len(tokens & tokenize(text)) / len(tokens) for text in texts),
        default=0.0,
    )


def null_seed(
    *,
    run_seed: int,
    architecture: str,
    outer_fold: int,
    inner_fold: int | None,
    schema_version: str,
) -> int:
    """Derive a draw seed from the run seed, architecture, and context only."""

    material = "|".join(
        [
            str(schema_version),
            str(int(run_seed)),
            str(architecture),
            str(int(outer_fold)),
            "outer" if inner_fold is None else str(int(inner_fold)),
        ]
    )
    return int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:16], 16)


def _alnum_mask(text: str) -> np.ndarray:
    codes = np.frombuffer(text.encode("utf-32-le", errors="surrogatepass"), dtype=np.uint32)
    return ((codes >= 97) & (codes <= 122)) | ((codes >= 48) & (codes <= 57))


class TrainingTextCorpus:
    """Lowercased dataset texts with token-run boundaries, built once."""

    def __init__(self, texts: Sequence[str]) -> None:
        lowered = [str(text or "").lower() for text in texts]
        self.lengths = np.asarray([len(text) for text in lowered], dtype=np.int64)
        # One separator character keeps token runs from joining across rows.
        self.base = (np.cumsum(self.lengths + 1) - (self.lengths + 1)).astype(np.int64)
        self.text = "\n".join(lowered)
        masks = []
        for text in lowered:
            masks.append(_alnum_mask(text))
            masks.append(np.zeros(1, dtype=bool))
        self.alnum = (
            np.concatenate(masks)[: len(self.text)]
            if masks
            else np.zeros(0, dtype=bool)
        )
        previous = np.zeros(len(self.alnum), dtype=bool)
        previous[1:] = self.alnum[:-1]
        self.run_starts = np.flatnonzero(self.alnum & ~previous).astype(np.int64)
        self._positions: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def token_positions(self, token: str) -> tuple[np.ndarray, np.ndarray]:
        """Return sorted starts of whole-run and run-prefix occurrences."""

        if token not in self._positions:
            starts = np.asarray(
                [match.start() for match in re.finditer(re.escape(token), self.text)],
                dtype=np.int64,
            )
            if len(starts):
                ends = starts + len(token)
                left_clean = (starts == 0) | ~self.alnum[np.maximum(starts - 1, 0)]
                continues = np.zeros(len(starts), dtype=bool)
                inside = ends < len(self.text)
                continues[inside] = self.alnum[ends[inside]]
                exact = starts[left_clean & ~continues]
                prefix = starts[left_clean & continues]
            else:
                exact = prefix = starts
            self._positions[token] = (exact, prefix)
        return self._positions[token]


@dataclass(frozen=True)
class NullResult:
    observed: dict[str, float | None]
    null_values: dict[str, np.ndarray]
    items: int
    length_unmatched_items: int
    draws: int
    seed: int


def _draw_windows(
    corpus: TrainingTextCorpus,
    row_ids: Sequence[int],
    item_lengths: Sequence[int],
    *,
    draws: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    rows = np.asarray(sorted({int(value) for value in row_ids}), dtype=np.int64)
    lengths = np.asarray(item_lengths, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    row_draw = rng.random((int(draws), len(lengths)))
    start_draw = rng.random((int(draws), len(lengths)))
    row_lengths = corpus.lengths[rows]
    order = np.argsort(row_lengths, kind="stable")
    sorted_rows = rows[order]
    sorted_lengths = row_lengths[order]
    first_long_enough = np.searchsorted(sorted_lengths, lengths, side="left")
    eligible = len(rows) - first_long_enough
    matched = eligible > 0
    choice = np.where(
        matched[None, :],
        first_long_enough[None, :]
        + np.floor(row_draw * np.maximum(eligible, 1)[None, :]).astype(np.int64),
        np.floor(row_draw * len(rows)).astype(np.int64),
    )
    choice = np.minimum(choice, len(rows) - 1)
    chosen = sorted_rows[choice]
    base = corpus.base[chosen]
    row_length = corpus.lengths[chosen]
    window = np.minimum(lengths[None, :], row_length)
    low = np.searchsorted(corpus.run_starts, base, side="left")
    high = np.searchsorted(corpus.run_starts, base + row_length - window, side="right")
    count = high - low
    pick = low + np.floor(start_draw * np.maximum(count, 1)).astype(np.int64)
    pick = np.minimum(pick, max(len(corpus.run_starts) - 1, 0))
    starts = np.where(
        count > 0,
        corpus.run_starts[pick] if len(corpus.run_starts) else base,
        base,
    )
    return starts, starts + window, int(np.sum(~matched))


def window_token_presence(
    corpus: TrainingTextCorpus,
    starts: np.ndarray,
    ends: np.ndarray,
    token: str,
) -> np.ndarray:
    """Whether ``token`` is a token of each window ``corpus.text[start:end]``."""

    exact, prefix = corpus.token_positions(token)
    size = len(token)
    present = np.searchsorted(exact, ends - size, side="right") > np.searchsorted(
        exact, starts, side="left"
    )
    if len(prefix):
        cut = ends - size
        index = np.minimum(np.searchsorted(prefix, cut, side="left"), len(prefix) - 1)
        present |= (prefix[index] == cut) & (cut >= starts)
    return present


def random_chunk_null(
    *,
    corpus: TrainingTextCorpus,
    training_row_ids: Sequence[int],
    evidence_texts: Sequence[str],
    feature_tokens: Mapping[str, frozenset[str]],
    draws: int,
    seed: int,
) -> NullResult:
    """Observed and null best token recall for one architecture and context."""

    texts = [str(text or "") for text in evidence_texts]
    observed = {
        name: best_token_recall(texts, tokens) for name, tokens in feature_tokens.items()
    }
    null_values: dict[str, np.ndarray] = {}
    unmatched = 0
    if texts and int(draws) > 0 and len(training_row_ids):
        starts, ends, unmatched = _draw_windows(
            corpus,
            training_row_ids,
            [len(text) for text in texts],
            draws=int(draws),
            seed=int(seed),
        )
        presence: dict[str, np.ndarray] = {}
        for tokens in feature_tokens.values():
            for token in tokens:
                if token not in presence:
                    presence[token] = window_token_presence(corpus, starts, ends, token)
        for name, tokens in feature_tokens.items():
            if not tokens:
                continue
            hits = np.zeros(starts.shape, dtype=np.int64)
            for token in tokens:
                hits += presence[token]
            null_values[name] = hits.max(axis=1) / float(len(tokens))
    return NullResult(
        observed=observed,
        null_values=null_values,
        items=len(texts),
        length_unmatched_items=int(unmatched),
        draws=int(draws),
        seed=int(seed),
    )


def summarize_null(observed: float | None, null: np.ndarray | None) -> dict[str, float | None]:
    """Null mean, 95th percentile, and the add-one empirical p-value."""

    if observed is None or null is None or not len(null):
        return {"null_mean": None, "null_p95": None, "empirical_p_value": None}
    values = np.asarray(null, dtype=float)
    return {
        "null_mean": float(np.mean(values)),
        "null_p95": float(np.quantile(values, 0.95)),
        "empirical_p_value": float(
            (1 + int(np.sum(values >= float(observed) - 1e-12))) / (len(values) + 1)
        ),
    }


__all__ = [
    "NullResult",
    "TrainingTextCorpus",
    "best_token_recall",
    "null_seed",
    "random_chunk_null",
    "summarize_null",
    "tokenize",
    "window_token_presence",
]
