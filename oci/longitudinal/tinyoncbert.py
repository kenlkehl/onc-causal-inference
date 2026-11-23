"""TinyOncBERT: clinical oncology sentence & note distillation and MLM pretraining.

Distills a strong sentence embedding teacher (e.g. ``Qwen/Qwen3-Embedding-0.6B``)
into a compact 6-layer, 384-dim HuggingFace ``BertModel`` using a 4-term objective:

1. Projection alignment to teacher embeddings (cosine + MSE).
2. Unprojected relational similarity preservation in native 384D space (``S @ S.T`` vs ``T @ T.T``).
3. Clinical negation-flip contrastive separation + 3-class polarity classification head.
4. Masked language modeling on sentences plus interleaved 512-token contiguous note windows.

The saved checkpoint in ``--output-dir`` is a standard HuggingFace ``BertModel`` +
tokenizer (with distillation-only heads stripped) that loads cleanly via
``AutoModel.from_pretrained`` as a drop-in backbone for ``PatientEncoder``.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import contextlib
from dataclasses import asdict, dataclass
import hashlib
import inspect
import json
import logging
import math
import os
from pathlib import Path
import random
import re
import time
from typing import Any, Iterator, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    BertConfig,
    BertModel,
    PreTrainedTokenizerBase,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_PARQUET = "hf://datasets/ksg-dfci/mmai-synthetic-1226/patients/all_synthetic_notes.parquet"
DEFAULT_TEACHER_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_TOKENIZER_MODEL = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
DEFAULT_SVD_INIT_MODEL = "NeuML/pubmedbert-base-embeddings"

SENTENCE_SPLIT_RE = re.compile(r"(?:\n+|;\s+|(?<!\d)\.(?:\s+|$)|(?:\s+-\s+))")

POLARITY_AFFIRMED = 0
POLARITY_NEGATED = 1
POLARITY_UNCERTAIN = 2

_BIDIRECTIONAL_MODEL_TYPES = frozenset(
    {
        "albert",
        "bert",
        "camembert",
        "deberta",
        "deberta-v2",
        "distilbert",
        "electra",
        "mobilebert",
        "modernbert",
        "mpnet",
        "nomic_bert",
        "roberta",
        "xlm-roberta",
    }
)

CURATION_FORMAT_VERSION = 1


# ---------------------------------------------------------------------------
# Clinical text normalization, polarity classification & negation flipping
# ---------------------------------------------------------------------------

_MD_HR_RE = re.compile(r"^\s*(?:[-*_=\u2014]\s*){3,}\s*$", flags=re.MULTILINE)
_MD_HEADER_RE = re.compile(r"^\s*#{1,6}\s*", flags=re.MULTILINE)
_MD_HEADER_TRAIL_RE = re.compile(r"\s+#{1,6}\s*$", flags=re.MULTILINE)
_MD_BOLD_STAR_RE = re.compile(r"\*\*(.+?)\*\*", flags=re.DOTALL)
_MD_BOLD_UNDER_RE = re.compile(r"(?<!\w)__(.+?)__(?!\w)", flags=re.DOTALL)
_MD_ITALIC_STAR_RE = re.compile(r"(?<!\w)\*([^*\n]+?)\*(?!\w)")
_MD_BULLET_RE = re.compile(r"^\s*[-*\u2022]\s+", flags=re.MULTILINE)
_MD_BACKTICK_RE = re.compile(r"`+")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_MULTI_NL_RE = re.compile(r"\n{3,}")


def strip_markdown(text: str) -> str:
    """Strip common LLM/Nemotron markdown formatting while preserving clinical text."""
    if not text:
        return ""
    out = text.replace("\r\n", "\n").replace("\r", "\n")
    out = _MD_HR_RE.sub("", out)
    out = _MD_HEADER_RE.sub("", out)
    out = _MD_HEADER_TRAIL_RE.sub("", out)
    out = _MD_BOLD_STAR_RE.sub(r"\1", out)
    out = out.replace("**", "")
    out = _MD_BOLD_UNDER_RE.sub(r"\1", out)
    out = _MD_ITALIC_STAR_RE.sub(r"\1", out)
    out = _MD_BACKTICK_RE.sub("", out)
    out = _MD_BULLET_RE.sub("", out)
    lines = [_MULTI_SPACE_RE.sub(" ", line).strip() for line in out.split("\n")]
    out = "\n".join(lines)
    out = _MULTI_NL_RE.sub("\n\n", out)
    return out.strip()


_UNCERTAIN_RE = re.compile(
    r"(?:"
    r"\bcannot\s+(?:rule\s+out|exclude|assess|evaluate|determine)\b"
    r"|\bunable\s+to\s+(?:rule\s+out|exclude|assess|evaluate|determine)\b"
    r"|\brule\s+out\b|\br/o\b"
    r"|\bsuspicious\s+for\b|\bsuspicion\s+(?:for|of)\b"
    r"|\bconcern(?:ing)?\s+for\b"
    r"|\bpossibl(?:e|y)\b|\bprobabl(?:e|y)\b|\blikely\b"
    r"|\bquestionable\b|\bequivocal\b|\bindeterminate\b|\buncertain\b|\bunclear\b"
    r"|\bdifferential\s+(?:includes?|diagnosis)\b"
    r"|\b(?:may|might|could)\s+(?:represent|indicate|be|reflect|suggest)\b"
    r"|\bversus\b|\bvs\.?\b"
    r"|\bpending\b"
    r")",
    flags=re.IGNORECASE,
)

_NEGATED_RE = re.compile(
    r"(?:"
    r"\bden(?:ies|ied|ying)\b"
    r"|\bno\s+evidence\s+of\b"
    r"|\bno\s+signs?\s+of\b"
    r"|\bno\s+history\s+of\b"
    r"|\bno\s+findings?\s+of\b"
    r"|\bno\s+indication\s+of\b"
    r"|\bnegative\s+for\b"
    r"|\btested\s+negative\b"
    r"|\b(?:her2|er|pr|pd-?l1|ebv|hpv|egfr|alk|ros1|braf|kras|brca|msi)[-\s]?negative\b"
    r"|\btriple[-\s]negative\b"
    r"|\bwithout\b(?!\s+(?:iv\s+|intravenous\s+|oral\s+)?contrast\b)"
    r"|\babsen(?:t|ce\s+of)\b"
    r"|\bfree\s+of\b"
    r"|\bunremarkable\b"
    r"|\bruled\s+out\b"
    r"|\b(?:does|did|do|is|are|was|were|has|have|had)\s+not\b"
    r"|\b(?:doesn't|didn't|don't|isn't|aren't|wasn't|weren't|hasn't|haven't|hadn't)\b"
    r"|\bnot\s+(?:present|noted|seen|identified|detected|observed|evident|demonstrated|palpable|appreciated|elevated|enlarged|resectable|metastatic|positive|reported|endorsed)\b"
    r"|\bno\b(?!\s*(?:\.\s*\d|doubt\b))"
    r"|\bnone\b"
    r")",
    flags=re.IGNORECASE,
)


def classify_polarity(sentence: str) -> int:
    """Classify a clinical sentence as affirmed (0), negated (1), or uncertain/other (2)."""
    cleaned = sentence.strip()
    if not cleaned or cleaned.endswith("?"):
        return POLARITY_UNCERTAIN
    if _UNCERTAIN_RE.search(cleaned):
        return POLARITY_UNCERTAIN
    if _NEGATED_RE.search(cleaned):
        return POLARITY_NEGATED
    return POLARITY_AFFIRMED


def _match_case(template: str, source: str) -> str:
    """Match capitalization style of ``source`` onto ``template``."""
    if not source or not template:
        return template
    if source.isupper() and len(source) > 1:
        return template.upper()
    if source[0].isupper():
        return template[0].upper() + template[1:]
    return template.lower()


def _lowercase_initial_unless_acronym(text: str) -> str:
    """Lowercase the first character unless the leading token looks like an acronym."""
    if not text:
        return text
    first_word = text.split(" ", 1)[0]
    letters = [c for c in first_word if c.isalpha()]
    if len(letters) >= 2 and all(c.isupper() for c in letters):
        return text
    return text[0].lower() + text[1:]


def _capitalize_initial_preserving_acronym(text: str) -> str:
    if not text:
        return text
    return text[0].upper() + text[1:]


# Ordered (pattern, replacement_fn) pairs for negated -> affirmed flips
def _build_neg_to_aff_rules() -> list[tuple[re.Pattern[str], Any]]:
    def _sub_word(target: str):
        return lambda m: _match_case(target, m.group(0))

    def _keep_prefix_word(target: str):
        return lambda m: f"{m.group(1)} {_match_case(target, m.group(2))}"

    def _drop_no_after_verb(m: re.Match[str]) -> str:
        return m.group(1)

    def _leading_no_to_noted(m: re.Match[str]) -> str:
        body = m.group(1).strip()
        punct = m.group(2) or ""
        if not body:
            return m.group(0)
        body_cap = _capitalize_initial_preserving_acronym(body)
        return f"{body_cap} noted{punct}"

    return [
        (re.compile(r"\bno\s+evidence\s+of\b", re.IGNORECASE), _sub_word("evidence of")),
        (re.compile(r"\bno\s+signs\s+of\b", re.IGNORECASE), _sub_word("signs of")),
        (re.compile(r"\bno\s+sign\s+of\b", re.IGNORECASE), _sub_word("sign of")),
        (re.compile(r"\bno\s+history\s+of\b", re.IGNORECASE), _sub_word("history of")),
        (re.compile(r"\bno\s+findings\s+of\b", re.IGNORECASE), _sub_word("findings of")),
        (re.compile(r"\bno\s+indication\s+of\b", re.IGNORECASE), _sub_word("indication of")),
        (re.compile(r"\bdenies\b", re.IGNORECASE), _sub_word("reports")),
        (re.compile(r"\bdenied\b", re.IGNORECASE), _sub_word("reported")),
        (re.compile(r"\bdenying\b", re.IGNORECASE), _sub_word("reporting")),
        (re.compile(r"\btested\s+negative\s+for\b", re.IGNORECASE), _sub_word("tested positive for")),
        (re.compile(r"\bnegative\s+for\b", re.IGNORECASE), _sub_word("positive for")),
        (re.compile(r"\bwithout\s+evidence\s+of\b", re.IGNORECASE), _sub_word("with evidence of")),
        (
            re.compile(
                r"\bwithout\b(?!\s+(?:iv\s+|intravenous\s+|oral\s+)?contrast\b)",
                re.IGNORECASE,
            ),
            _sub_word("with"),
        ),
        (re.compile(r"\babsence\s+of\b", re.IGNORECASE), _sub_word("presence of")),
        (re.compile(r"\bfree\s+of\b", re.IGNORECASE), _sub_word("positive for")),
        (re.compile(r"\bunremarkable\b", re.IGNORECASE), _sub_word("abnormal")),
        (re.compile(r"\bruled\s+out\b", re.IGNORECASE), _sub_word("confirmed")),
        (
            re.compile(
                r"\b(her2|er|pr|pd-?l1|ebv|hpv|egfr|alk|ros1|braf|kras|brca|msi)([-\s])(negative)\b",
                re.IGNORECASE,
            ),
            lambda m: f"{m.group(1)}{m.group(2)}{_match_case('positive', m.group(3))}",
        ),
        (
            re.compile(r"\b(does\s+not\s+have)\b", re.IGNORECASE),
            _sub_word("has"),
        ),
        (
            re.compile(r"\b(did\s+not\s+have)\b", re.IGNORECASE),
            _sub_word("had"),
        ),
        (
            re.compile(r"\b(is|was|are|were)\s+not\b", re.IGNORECASE),
            lambda m: m.group(1),
        ),
        (
            re.compile(
                r"\bnot\s+(present|noted|seen|identified|detected|observed|evident|demonstrated|palpable|appreciated|elevated|enlarged|resectable|metastatic|positive)\b",
                re.IGNORECASE,
            ),
            lambda m: m.group(1),
        ),
        (
            re.compile(
                r"\b(there\s+(?:is|are|was|were)|has|have|had|shows?|showed|demonstrates?|demonstrated|reveals?|revealed|with)\s+no\b",
                re.IGNORECASE,
            ),
            _drop_no_after_verb,
        ),
        (
            re.compile(r"^\s*no\s+([^.!?]+?)([.!?])?\s*$", re.IGNORECASE),
            _leading_no_to_noted,
        ),
        (
            re.compile(r"\b(is|was|are|were)\s+(absent)\b", re.IGNORECASE),
            _keep_prefix_word("present"),
        ),
    ]


# Ordered (pattern, replacement_fn) pairs for affirmed -> negated flips
def _build_aff_to_neg_rules() -> list[tuple[re.Pattern[str], Any]]:
    def _sub_word(target: str):
        return lambda m: _match_case(target, m.group(0))

    def _noted_to_no(m: re.Match[str]) -> str:
        body = m.group(1).strip()
        tail = m.group(2) or ""
        if not body:
            return m.group(0)
        body_low = _lowercase_initial_unless_acronym(body)
        return f"No {body_low}{tail}"

    return [
        (re.compile(r"\bevidence\s+of\b", re.IGNORECASE), _sub_word("no evidence of")),
        (re.compile(r"\bsigns\s+of\b", re.IGNORECASE), _sub_word("no signs of")),
        (re.compile(r"\bsign\s+of\b", re.IGNORECASE), _sub_word("no sign of")),
        (re.compile(r"\bhistory\s+of\b", re.IGNORECASE), _sub_word("no history of")),
        (re.compile(r"\b(?:reports|endorses)\b", re.IGNORECASE), _sub_word("denies")),
        (re.compile(r"\b(?:reported|endorsed)\b", re.IGNORECASE), _sub_word("denied")),
        (re.compile(r"\b(?:reporting|endorsing)\b", re.IGNORECASE), _sub_word("denying")),
        (re.compile(r"\bcomplains\s+of\b", re.IGNORECASE), _sub_word("denies")),
        (re.compile(r"\bcomplained\s+of\b", re.IGNORECASE), _sub_word("denied")),
        (re.compile(r"\btested\s+positive\s+for\b", re.IGNORECASE), _sub_word("tested negative for")),
        (re.compile(r"\bpositive\s+for\b", re.IGNORECASE), _sub_word("negative for")),
        (re.compile(r"\bwith\s+evidence\s+of\b", re.IGNORECASE), _sub_word("without evidence of")),
        (re.compile(r"\bpresence\s+of\b", re.IGNORECASE), _sub_word("absence of")),
        (re.compile(r"\babnormal\b", re.IGNORECASE), _sub_word("unremarkable")),
        (
            re.compile(
                r"\b(her2|er|pr|pd-?l1|ebv|hpv|egfr|alk|ros1|braf|kras|brca|msi)([-\s])(positive)\b",
                re.IGNORECASE,
            ),
            lambda m: f"{m.group(1)}{m.group(2)}{_match_case('negative', m.group(3))}",
        ),
        (
            re.compile(
                r"^(\s*)with\b(?!\s+(?:iv\s+|intravenous\s+|oral\s+)?contrast\b)",
                re.IGNORECASE,
            ),
            lambda m: f"{m.group(1)}{_match_case('without', m.group(0).strip())}",
        ),
        (
            re.compile(
                r"\bwith\s+(associated|acute|new|worsening|progressive|persistent|mild|moderate|severe|significant|bilateral|diffuse|focal|palpable|visible|elevated|enlarged|metastatic|recurrent|residual)\b",
                re.IGNORECASE,
            ),
            lambda m: f"{_match_case('without', m.group(0).split()[0])} {m.group(1)}",
        ),
        (
            re.compile(
                r"\b(there\s+(?:is|are|was|were)|shows?|showed|demonstrates?|demonstrated|reveals?|revealed)\s+(?!no\b)",
                re.IGNORECASE,
            ),
            lambda m: f"{m.group(1)} no ",
        ),
        (
            re.compile(
                r"\b(has|have|had)\s+(?!(?:no|not|been|had|undergone|received|completed|started|continued|improved|resolved|progressed|developed|tolerated)\b)",
                re.IGNORECASE,
            ),
            lambda m: f"{m.group(1)} no ",
        ),
        (
            re.compile(
                r"\b(is|was|are|were)\s+(present|noted|seen|identified|detected|observed|evident|demonstrated|palpable|appreciated|elevated|enlarged|resectable|metastatic|positive)\b",
                re.IGNORECASE,
            ),
            lambda m: f"{m.group(1)} not {m.group(2)}",
        ),
        (
            re.compile(
                r"^\s*([A-Za-z][^.!?]{3,}?)\s+(?:noted|present|identified|detected|seen)(\s+(?:on|in|at|within|throughout)\b[^.!?]*[.!?]?|[.!?]?)\s*$",
                re.IGNORECASE,
            ),
            _noted_to_no,
        ),
    ]


_NEG_TO_AFF_RULES = _build_neg_to_aff_rules()
_AFF_TO_NEG_RULES = _build_aff_to_neg_rules()


def flip_clinical_negation(sentence: str) -> str | None:
    """Generate a grammatical polarity-flipped hard negative for a clinical sentence.

    Supports bidirectional flipping (negated <-> affirmed). Returns ``None`` for
    uncertain sentences or when no clean flip rule matches.
    """
    cleaned = " ".join(sentence.split())
    if not cleaned:
        return None

    polarity = classify_polarity(cleaned)
    if polarity == POLARITY_UNCERTAIN:
        return None

    rules = _NEG_TO_AFF_RULES if polarity == POLARITY_NEGATED else _AFF_TO_NEG_RULES
    expected_polarity = POLARITY_AFFIRMED if polarity == POLARITY_NEGATED else POLARITY_NEGATED

    for pattern, repl in rules:
        if not pattern.search(cleaned):
            continue
        candidate = pattern.sub(repl, cleaned, count=1)
        candidate = " ".join(candidate.split())
        if candidate != cleaned and classify_polarity(candidate) == expected_polarity:
            return candidate
    return None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TinyOncBertConfig:
    """Configuration for TinyOncBERT curation, teacher precomputation, and training."""

    # Data & paths
    parquet: str = DEFAULT_PARQUET
    work_dir: str = "artifacts/tinyoncbert_6l_384d/work"
    output_dir: str = "artifacts/tinyoncbert_6l_384d"
    split_filter: str = "train"
    num_sentences: int = 2_000_000
    num_windows: int = 250_000
    val_sentences: int = 4_096
    val_windows: int = 512
    markdown_strip_prob: float = 0.5
    sentence_max_length: int = 64
    window_max_length: int = 512

    # Teacher
    teacher_model: str = DEFAULT_TEACHER_MODEL
    teacher_pooling: str = "auto"
    teacher_batch_size: int = 1024
    teacher_max_length: int = 128

    # Tokenizer & student architecture
    tokenizer_model: str = DEFAULT_TOKENIZER_MODEL
    svd_init_model: str = DEFAULT_SVD_INIT_MODEL
    num_hidden_layers: int = 6
    hidden_size: int = 384
    num_attention_heads: int = 6
    intermediate_size: int = 1536
    max_position_embeddings: int = 512
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1

    # Loss weights
    lambda_emb: float = 1.0
    lambda_rel: float = 0.5
    lambda_neg: float = 0.5
    lambda_pol: float = 0.1
    lambda_mlm: float = 0.25
    neg_margin: float = 0.25
    mlm_probability: float = 0.15
    window_mlm_every: int = 6

    # Optimization & runtime
    epochs: int = 3
    batch_size: int = 2048
    window_batch_size: int = 256
    lr: float = 5e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    grad_clip: float = 1.0
    log_every: int = 50
    eval_every: int = 500
    num_workers: int = 4
    device: str = "cuda"
    compile_model: bool = False
    seed: int = 42


# ---------------------------------------------------------------------------
# Streaming parquet curation (resumable & event_type-balanced)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _open_parquet_file(parquet_path: str) -> Iterator[pq.ParquetFile]:
    """Open either a local parquet file or an ``hf://`` parquet file for row-group streaming."""
    if parquet_path.startswith("hf://"):
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem()
        with fs.open(parquet_path, mode="rb") as handle:
            yield pq.ParquetFile(handle)
    else:
        yield pq.ParquetFile(parquet_path)


def split_clinical_sentences(note: str) -> list[str]:
    """Split a clinical note into clauses with >= 20 chars and >= 10 alphabetic chars."""
    clauses: list[str] = []
    for raw in SENTENCE_SPLIT_RE.split(note):
        cleaned = " ".join(raw.split())
        if len(cleaned) < 20:
            continue
        if sum(1 for ch in cleaned if ch.isalpha()) < 10:
            continue
        clauses.append(cleaned)
    return clauses


def _sentence_digest(sentence: str) -> bytes:
    return hashlib.blake2b(sentence.lower().encode("utf-8"), digest_size=8).digest()


def _extract_contiguous_windows(
    text: str,
    tokenizer: PreTrainedTokenizerBase | None,
    window_max_length: int,
) -> tuple[list[str], str]:
    """Slice ``text`` into contiguous ``window_max_length - 2`` token windows plus remainder.

    Uses Rust fast-tokenizer offset mappings when available so emitted windows are exact
    substrings of the original note text without detokenization artifacts.
    """
    span = max(8, window_max_length - 2)
    if not text.strip():
        return [], ""

    if tokenizer is not None and hasattr(tokenizer, "_tokenizer"):
        enc = tokenizer._tokenizer.encode(text, add_special_tokens=False)
        n_toks = len(enc.ids)
        if n_toks < span:
            return [], text
        offsets = enc.offsets
        windows: list[str] = []
        pos = 0
        while pos + span <= n_toks:
            start_char = offsets[pos][0]
            end_char = offsets[pos + span - 1][1]
            chunk = text[start_char:end_char].strip()
            if chunk:
                windows.append(chunk)
            pos += span
        remainder = text[offsets[pos][0] :].strip() if pos < n_toks else ""
        return windows, remainder

    if tokenizer is not None:
        ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
        n_toks = len(ids)
        if n_toks < span:
            return [], text
        windows = []
        pos = 0
        while pos + span <= n_toks:
            chunk = tokenizer.decode(ids[pos : pos + span], skip_special_tokens=True).strip()
            if chunk:
                windows.append(chunk)
            pos += span
        remainder = (
            tokenizer.decode(ids[pos:], skip_special_tokens=True).strip() if pos < n_toks else ""
        )
        return windows, remainder

    # Fallback without tokenizer: approximate 4 chars/token on word boundaries
    words = text.split()
    word_span = max(16, int(span * 0.75))
    if len(words) < word_span:
        return [], text
    windows = []
    pos = 0
    while pos + word_span <= len(words):
        windows.append(" ".join(words[pos : pos + word_span]))
        pos += word_span
    remainder = " ".join(words[pos:]) if pos < len(words) else ""
    return windows, remainder


def _water_fill_select(
    buckets: dict[str, list[Any]],
    target: int,
    rng: random.Random,
) -> list[tuple[str, Any]]:
    """Select up to ``target`` items balanced across ``event_type`` buckets via water-filling."""
    if target <= 0 or not buckets:
        return []

    keys = sorted(buckets.keys())
    shuffled: dict[str, list[Any]] = {}
    for k in keys:
        items = list(buckets[k])
        rng.shuffle(items)
        shuffled[k] = items

    total_available = sum(len(v) for v in shuffled.values())
    if total_available <= target:
        combined = [(k, item) for k in keys for item in shuffled[k]]
        rng.shuffle(combined)
        return combined

    # Water-filling allocation: rare types contribute all items; richer types split remainder equally
    remaining_target = target
    remaining_keys = list(keys)
    quotas: dict[str, int] = {}
    while remaining_keys and remaining_target > 0:
        fair = remaining_target // len(remaining_keys)
        rem = remaining_target % len(remaining_keys)
        exhausted = [k for k in remaining_keys if len(shuffled[k]) <= fair]
        if not exhausted:
            for idx, k in enumerate(remaining_keys):
                quotas[k] = fair + (1 if idx < rem else 0)
            remaining_target = 0
            break
        for k in exhausted:
            quotas[k] = len(shuffled[k])
            remaining_target -= len(shuffled[k])
        remaining_keys = [k for k in remaining_keys if k not in quotas]

    combined = []
    for k in keys:
        take = quotas.get(k, 0)
        for item in shuffled[k][:take]:
            combined.append((k, item))
    rng.shuffle(combined)
    return combined


def _curated_config_key(cfg: TinyOncBertConfig) -> dict[str, Any]:
    return {
        "version": CURATION_FORMAT_VERSION,
        "parquet": str(cfg.parquet),
        "split_filter": cfg.split_filter,
        "num_sentences": int(cfg.num_sentences),
        "num_windows": int(cfg.num_windows),
        "markdown_strip_prob": float(cfg.markdown_strip_prob),
        "window_max_length": int(cfg.window_max_length),
        "seed": int(cfg.seed),
    }


def _write_parquet_shards(
    table: pa.Table,
    out_dir: Path,
    prefix: str,
    rows_per_shard: int,
) -> list[str]:
    shard_names: list[str] = []
    total_rows = table.num_rows
    if total_rows == 0:
        name = f"{prefix}_0000.parquet"
        pq.write_table(table, out_dir / name, compression="zstd")
        return [name]
    for shard_idx, start in enumerate(range(0, total_rows, rows_per_shard)):
        end = min(start + rows_per_shard, total_rows)
        name = f"{prefix}_{shard_idx:04d}.parquet"
        pq.write_table(table.slice(start, end - start), out_dir / name, compression="zstd")
        shard_names.append(name)
    return shard_names


def curate_corpus(
    cfg: TinyOncBertConfig,
    tokenizer: PreTrainedTokenizerBase | None = None,
    *,
    sentences_per_shard: int = 250_000,
    windows_per_shard: int = 100_000,
) -> dict[str, Any]:
    """Stream synthetic notes from parquet, curate balanced unique sentences + 512-tok windows.

    Results are cached as compressed parquet shards + ``manifest.json`` under
    ``<work_dir>/curated`` so reruns resume instantaneously.
    """
    curated_dir = Path(cfg.work_dir) / "curated"
    curated_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = curated_dir / "manifest.json"
    expected_key = _curated_config_key(cfg)

    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing.get("config_key") == expected_key:
                all_shards_exist = all(
                    (curated_dir / s).exists() for s in existing.get("sentence_shards", [])
                ) and all((curated_dir / w).exists() for w in existing.get("window_shards", []))
                if all_shards_exist:
                    LOGGER.info(
                        "Reusing cached curated corpus at %s (%d sentences, %d windows)",
                        curated_dir,
                        existing["num_sentences_actual"],
                        existing["num_windows_actual"],
                    )
                    return existing
        except Exception as exc:
            LOGGER.warning("Ignoring invalid curated manifest (%s); re-curating.", exc)

    rng = random.Random(cfg.seed)
    seen_hashes: set[bytes] = set()
    sent_buckets: dict[str, list[str]] = defaultdict(list)
    sent_seen_per_type: dict[str, int] = defaultdict(int)

    win_buckets: dict[str, list[str]] = defaultdict(list)
    win_seen_per_type: dict[str, int] = defaultdict(int)
    win_buffers: dict[str, str] = defaultdict(str)

    # Cap per-type reservoirs to bound memory while preserving balance & diversity
    def _sent_cap() -> int:
        k = max(1, len(sent_buckets))
        return max(min(cfg.num_sentences, 10_000), math.ceil(cfg.num_sentences * 1.35 / k))

    def _win_cap() -> int:
        if cfg.num_windows <= 0:
            return 0
        k = max(1, len(win_buckets) or len(sent_buckets))
        return max(min(cfg.num_windows, 2_000), math.ceil(cfg.num_windows * 1.35 / k))

    def _reservoir_add(
        buckets: dict[str, list[str]],
        seen_counts: dict[str, int],
        etype: str,
        item: str,
        cap: int,
    ) -> None:
        if cap <= 0:
            return
        seen_counts[etype] += 1
        bucket = buckets[etype]
        if len(bucket) < cap:
            bucket.append(item)
        else:
            j = rng.randrange(seen_counts[etype])
            if j < cap:
                bucket[j] = item

    def _has_sufficient_balanced_data(rg_idx: int, num_rgs: int) -> bool:
        min_rgs = min(num_rgs, max(4, num_rgs // 10))
        if rg_idx + 1 < min_rgs:
            return False
        if not sent_buckets:
            return False
        s_total = sum(len(v) for v in sent_buckets.values())
        if s_total < math.ceil(cfg.num_sentences * 1.15):
            return False
        per_type_s = math.ceil(cfg.num_sentences / max(1, len(sent_buckets)))
        if any(len(v) < per_type_s for v in sent_buckets.values()):
            # Still allow early stop once total has enough cushion across rich types
            if s_total < math.ceil(cfg.num_sentences * 1.35):
                return False
        if cfg.num_windows > 0:
            w_total = sum(len(v) for v in win_buckets.values())
            if w_total < math.ceil(cfg.num_windows * 1.1):
                return False
        return True

    notes_read = 0
    notes_kept = 0
    t0 = time.time()

    with _open_parquet_file(cfg.parquet) as pf:
        available_cols = set(pf.schema_arrow.names)
        if "synthetic_note" not in available_cols:
            raise ValueError(
                f"Parquet file {cfg.parquet} is missing required 'synthetic_note' column "
                f"(available: {sorted(available_cols)})"
            )
        cols_to_read = ["synthetic_note"]
        has_event_type = "event_type" in available_cols
        has_split = "split" in available_cols
        if has_event_type:
            cols_to_read.append("event_type")
        if has_split:
            cols_to_read.append("split")

        num_rgs = pf.num_row_groups
        LOGGER.info(
            "Streaming parquet %s (%d row groups, reading columns %s)",
            cfg.parquet,
            num_rgs,
            cols_to_read,
        )

        for rg_idx in range(num_rgs):
            table = pf.read_row_group(rg_idx, columns=cols_to_read)
            notes_col = table["synthetic_note"].to_pylist()
            etypes_col = (
                table["event_type"].to_pylist() if has_event_type else ["default"] * len(notes_col)
            )
            splits_col = table["split"].to_pylist() if has_split else [None] * len(notes_col)

            for raw_note, raw_etype, raw_split in zip(notes_col, etypes_col, splits_col):
                notes_read += 1
                if has_split and cfg.split_filter and raw_split != cfg.split_filter:
                    continue
                if not isinstance(raw_note, str) or not raw_note.strip():
                    continue
                notes_kept += 1
                etype = str(raw_etype) if raw_etype is not None else "default"

                note = (
                    strip_markdown(raw_note)
                    if rng.random() < cfg.markdown_strip_prob
                    else raw_note.strip()
                )
                if not note:
                    continue

                # Extract unique clinical sentences
                s_cap = _sent_cap()
                for clause in split_clinical_sentences(note):
                    digest = _sentence_digest(clause)
                    if digest in seen_hashes:
                        continue
                    seen_hashes.add(digest)
                    _reservoir_add(sent_buckets, sent_seen_per_type, etype, clause, s_cap)

                # Pack notes per event_type into contiguous 512-token windows
                if cfg.num_windows > 0:
                    w_cap = _win_cap()
                    # Only keep tokenizing windows if this event_type still needs diversity
                    if win_seen_per_type[etype] < w_cap * 2:
                        buf = win_buffers[etype]
                        buf = f"{buf}\n\n{note}" if buf else note
                        emitted, remainder = _extract_contiguous_windows(
                            buf, tokenizer, cfg.window_max_length
                        )
                        win_buffers[etype] = remainder
                        for win in emitted:
                            _reservoir_add(win_buckets, win_seen_per_type, etype, win, w_cap)

            if (rg_idx + 1) % 10 == 0 or rg_idx + 1 == num_rgs:
                LOGGER.info(
                    "Row group %d/%d | notes=%d | unique_sents_buffered=%d | windows_buffered=%d | %.1fs",
                    rg_idx + 1,
                    num_rgs,
                    notes_kept,
                    sum(len(v) for v in sent_buckets.values()),
                    sum(len(v) for v in win_buckets.values()),
                    time.time() - t0,
                )

            if _has_sufficient_balanced_data(rg_idx, num_rgs):
                LOGGER.info("Reached balanced target quotas early at row group %d/%d.", rg_idx + 1, num_rgs)
                break

    # Flush any remaining partial window buffers if we haven't met num_windows
    if cfg.num_windows > 0:
        w_cap = _win_cap()
        for etype, rem in win_buffers.items():
            rem_clean = rem.strip()
            if len(rem_clean) >= 40 and sum(1 for c in rem_clean if c.isalpha()) >= 10:
                _reservoir_add(win_buckets, win_seen_per_type, etype, rem_clean, w_cap)

    selected_sents = _water_fill_select(sent_buckets, cfg.num_sentences, rng)
    selected_wins = _water_fill_select(win_buckets, cfg.num_windows, rng)

    # Annotate sentences with polarity & negation flips
    sent_texts: list[str] = []
    sent_etypes: list[str] = []
    sent_polarities: list[int] = []
    sent_flips: list[str] = []
    event_type_counts: dict[str, int] = defaultdict(int)
    polarity_counts = {0: 0, 1: 0, 2: 0}
    num_flippable = 0

    for etype, text in selected_sents:
        pol = classify_polarity(text)
        flipped = flip_clinical_negation(text)
        sent_texts.append(text)
        sent_etypes.append(etype)
        sent_polarities.append(pol)
        sent_flips.append(flipped if flipped is not None else "")
        event_type_counts[etype] += 1
        polarity_counts[pol] += 1
        if flipped is not None:
            num_flippable += 1

    sent_table = pa.table(
        {
            "text": pa.array(sent_texts, type=pa.string()),
            "event_type": pa.array(sent_etypes, type=pa.string()),
            "polarity": pa.array(sent_polarities, type=pa.int8()),
            "flipped_text": pa.array(sent_flips, type=pa.string()),
        }
    )
    sent_shards = _write_parquet_shards(sent_table, curated_dir, "sentences", sentences_per_shard)

    win_table = pa.table(
        {
            "text": pa.array([w for _, w in selected_wins], type=pa.string()),
            "event_type": pa.array([e for e, _ in selected_wins], type=pa.string()),
        }
    )
    win_shards = _write_parquet_shards(win_table, curated_dir, "windows", windows_per_shard)

    sig_payload = json.dumps(
        {
            "config_key": expected_key,
            "num_sentences_actual": len(sent_texts),
            "num_windows_actual": len(selected_wins),
        },
        sort_keys=True,
    ).encode("utf-8")
    signature = hashlib.sha256(sig_payload).hexdigest()[:16]

    manifest = {
        "config_key": expected_key,
        "signature": signature,
        "notes_read": notes_read,
        "notes_kept": notes_kept,
        "num_sentences_actual": len(sent_texts),
        "num_windows_actual": len(selected_wins),
        "event_type_counts": dict(sorted(event_type_counts.items())),
        "polarity_counts": {str(k): v for k, v in polarity_counts.items()},
        "num_flippable": num_flippable,
        "sentence_shards": sent_shards,
        "window_shards": win_shards,
    }
    tmp_manifest = curated_dir / "manifest.json.tmp"
    tmp_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tmp_manifest.replace(manifest_path)

    LOGGER.info(
        "Curated %d sentences (%d flippable, %.1f%%) and %d windows across %d event types in %.1fs",
        len(sent_texts),
        num_flippable,
        100.0 * num_flippable / max(1, len(sent_texts)),
        len(selected_wins),
        len(event_type_counts),
        time.time() - t0,
    )
    return manifest


def load_curated_sentences(work_dir: str | Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Load curated sentence shards from ``<work_dir>/curated`` into memory."""
    curated_dir = Path(work_dir) / "curated"
    tables = [pq.read_table(curated_dir / shard) for shard in manifest["sentence_shards"]]
    full = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
    return {
        "texts": full["text"].to_pylist(),
        "event_types": full["event_type"].to_pylist(),
        "polarities": np.asarray(full["polarity"].to_numpy(zero_copy_only=False), dtype=np.int64),
        "flipped_texts": full["flipped_text"].to_pylist(),
    }


def load_curated_windows(work_dir: str | Path, manifest: dict[str, Any]) -> list[str]:
    """Load curated 512-token note windows from ``<work_dir>/curated``."""
    curated_dir = Path(work_dir) / "curated"
    tables = [pq.read_table(curated_dir / shard) for shard in manifest["window_shards"]]
    full = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
    return full["text"].to_pylist()


# ---------------------------------------------------------------------------
# Teacher embedding precomputation (length-sorted BF16 -> memmap FP16 .npy)
# ---------------------------------------------------------------------------


def resolve_teacher_pooling(teacher_model: nn.Module, requested: str) -> str:
    """Resolve ``auto`` pooling mode into ``mean`` (bidirectional) or ``last`` (decoder)."""
    if requested in ("mean", "last"):
        return requested
    if requested != "auto":
        raise ValueError(f"Unsupported teacher_pooling={requested!r}; expected 'auto', 'mean', or 'last'")
    model_type = str(getattr(getattr(teacher_model, "config", None), "model_type", "")).lower()
    if model_type in _BIDIRECTIONAL_MODEL_TYPES:
        return "mean"
    return "last"


def pool_embeddings(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
    pooling: str,
) -> torch.Tensor:
    """Pool token hidden states into sentence embeddings via ``mean`` or ``last`` token pooling."""
    if pooling == "mean":
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        summed = (last_hidden_state * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1e-6)
        return summed / denom
    if pooling == "last":
        # Handles both left-padding (last column is non-pad for all rows) and right-padding.
        left_padded = bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item())
        if left_padded:
            return last_hidden_state[:, -1]
        seq_lengths = attention_mask.sum(dim=1).clamp_min(1) - 1
        batch_idx = torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device)
        return last_hidden_state[batch_idx, seq_lengths]
    raise ValueError(f"Unknown pooling mode: {pooling!r}")


@torch.no_grad()
def precompute_teacher_embeddings(
    cfg: TinyOncBertConfig,
    sentences: Sequence[str],
    curated_signature: str,
    *,
    teacher_model_obj: nn.Module | None = None,
    teacher_tokenizer_obj: PreTrainedTokenizerBase | None = None,
) -> tuple[Path, int]:
    """Precompute L2-normalized FP16 teacher embeddings into a memory-mapped ``.npy`` file.

    Resumes cleanly if a matching completed cache already exists in ``cfg.work_dir``.
    """
    work_dir = Path(cfg.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    emb_path = work_dir / "teacher_embeddings.npy"
    meta_path = work_dir / "teacher_meta.json"
    num_sentences = len(sentences)

    if num_sentences == 0:
        raise ValueError("Cannot precompute teacher embeddings for an empty sentence list")

    expected_meta = {
        "teacher_model": cfg.teacher_model,
        "teacher_pooling_requested": cfg.teacher_pooling,
        "teacher_max_length": int(cfg.teacher_max_length),
        "num_sentences": int(num_sentences),
        "curated_signature": curated_signature,
    }

    if emb_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if (
                meta.get("complete") is True
                and all(meta.get(k) == v for k, v in expected_meta.items())
                and int(meta.get("teacher_dim", 0)) > 0
            ):
                arr = np.load(emb_path, mmap_mode="r")
                if arr.shape == (num_sentences, int(meta["teacher_dim"])) and arr.dtype == np.float16:
                    LOGGER.info(
                        "Reusing cached teacher embeddings %s with shape %s",
                        emb_path,
                        arr.shape,
                    )
                    return emb_path, int(meta["teacher_dim"])
        except Exception as exc:
            LOGGER.warning("Ignoring invalid teacher embedding cache (%s); recomputing.", exc)

    device = torch.device(cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    owns_model = teacher_model_obj is None
    if teacher_tokenizer_obj is None:
        teacher_tokenizer = AutoTokenizer.from_pretrained(cfg.teacher_model)
    else:
        teacher_tokenizer = teacher_tokenizer_obj

    if teacher_model_obj is None:
        load_dtype = torch.bfloat16 if use_bf16 else torch.float32
        teacher = AutoModel.from_pretrained(cfg.teacher_model, dtype=load_dtype)
    else:
        teacher = teacher_model_obj

    teacher.eval()
    teacher.to(device)

    pooling = resolve_teacher_pooling(teacher, cfg.teacher_pooling)
    orig_padding_side = getattr(teacher_tokenizer, "padding_side", "right")
    if pooling == "last":
        teacher_tokenizer.padding_side = "left"
    else:
        teacher_tokenizer.padding_side = "right"
    if teacher_tokenizer.pad_token is None:
        if teacher_tokenizer.eos_token is not None:
            teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
        else:
            teacher_tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    accepts_use_cache = "use_cache" in inspect.signature(teacher.forward).parameters
    teacher_dim = int(teacher.config.hidden_size)

    # Sort indices by character length for minimal padding waste
    order = sorted(range(num_sentences), key=lambda i: len(sentences[i]))
    batch_size = max(1, int(cfg.teacher_batch_size))
    num_batches = math.ceil(num_sentences / batch_size)

    # Check partial resume state
    start_batch = 0
    if emb_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if (
                meta.get("complete") is False
                and all(meta.get(k) == v for k, v in expected_meta.items())
                and int(meta.get("teacher_dim", 0)) == teacher_dim
                and int(meta.get("batch_size", 0)) == batch_size
            ):
                start_batch = int(meta.get("completed_batches", 0))
                LOGGER.info(
                    "Resuming teacher precomputation from batch %d/%d", start_batch, num_batches
                )
        except Exception:
            start_batch = 0

    if start_batch > 0 and emb_path.exists():
        mmap = np.lib.format.open_memmap(
            emb_path, mode="r+", dtype=np.float16, shape=(num_sentences, teacher_dim)
        )
    else:
        start_batch = 0
        mmap = np.lib.format.open_memmap(
            emb_path, mode="w+", dtype=np.float16, shape=(num_sentences, teacher_dim)
        )

    def _save_meta(completed_batches: int, complete: bool) -> None:
        payload = {
            **expected_meta,
            "teacher_pooling_resolved": pooling,
            "teacher_dim": teacher_dim,
            "batch_size": batch_size,
            "completed_batches": completed_batches,
            "complete": complete,
        }
        tmp = work_dir / "teacher_meta.json.tmp"
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(meta_path)

    amp_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16
        else contextlib.nullcontext()
    )

    t0 = time.time()
    LOGGER.info(
        "Precomputing teacher embeddings (%s, pooling=%s, dim=%d, batches=%d, bf16=%s)",
        cfg.teacher_model,
        pooling,
        teacher_dim,
        num_batches,
        use_bf16,
    )

    for b_idx in range(start_batch, num_batches):
        idx_slice = order[b_idx * batch_size : (b_idx + 1) * batch_size]
        batch_texts = [sentences[i] for i in idx_slice]
        encoded = teacher_tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=cfg.teacher_max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device, non_blocking=True) for k, v in encoded.items()}
        fwd_kwargs: dict[str, Any] = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }
        if "token_type_ids" in encoded and "token_type_ids" in inspect.signature(teacher.forward).parameters:
            fwd_kwargs["token_type_ids"] = encoded["token_type_ids"]
        if accepts_use_cache:
            fwd_kwargs["use_cache"] = False

        with amp_ctx:
            outputs = teacher(**fwd_kwargs)
            pooled = pool_embeddings(outputs.last_hidden_state, encoded["attention_mask"], pooling)

        normed = F.normalize(pooled.float(), p=2, dim=-1).cpu().numpy().astype(np.float16)
        mmap[idx_slice] = normed

        if (b_idx + 1) % 25 == 0 or (b_idx + 1) == num_batches:
            mmap.flush()
            _save_meta(b_idx + 1, complete=False)
            done = min((b_idx + 1) * batch_size, num_sentences)
            elapsed = max(time.time() - t0, 1e-6)
            LOGGER.info(
                "Teacher precompute: %d/%d (%.1f%%) | %.0f sents/s",
                done,
                num_sentences,
                100.0 * done / num_sentences,
                done / elapsed,
            )

    mmap.flush()
    del mmap
    _save_meta(num_batches, complete=True)
    if teacher_tokenizer_obj is not None:
        teacher_tokenizer.padding_side = orig_padding_side

    if owns_model:
        del teacher
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return emb_path, teacher_dim


# ---------------------------------------------------------------------------
# Student model, SVD warm-start & distillation heads
# ---------------------------------------------------------------------------


def build_student_config(
    cfg: TinyOncBertConfig,
    tokenizer: PreTrainedTokenizerBase,
) -> BertConfig:
    """Build the drop-in HuggingFace ``BertConfig`` for TinyOncBERT."""
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    return BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=cfg.hidden_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        intermediate_size=cfg.intermediate_size,
        hidden_act=cfg.hidden_act,
        hidden_dropout_prob=cfg.hidden_dropout_prob,
        attention_probs_dropout_prob=cfg.attention_probs_dropout_prob,
        max_position_embeddings=cfg.max_position_embeddings,
        type_vocab_size=2,
        pad_token_id=pad_token_id,
    )


@torch.no_grad()
def initialize_word_embeddings_svd(
    student: BertModel,
    svd_init_source: str | nn.Module | torch.Tensor,
    student_tokenizer: PreTrainedTokenizerBase | None = None,
) -> bool:
    """Warm-start student word embeddings via top-``hidden_size`` SVD projection.

    Accepts a HuggingFace model identifier/path, an instantiated ``nn.Module``,
    or a raw embedding ``torch.Tensor`` of shape ``(vocab_size, src_dim)``.
    """
    if isinstance(svd_init_source, str) and not svd_init_source.strip():
        return False

    src_tokenizer: PreTrainedTokenizerBase | None = None
    if isinstance(svd_init_source, torch.Tensor):
        src_weight = svd_init_source.detach().float().cpu()
    elif isinstance(svd_init_source, nn.Module):
        src_weight = svd_init_source.get_input_embeddings().weight.detach().float().cpu()
    else:
        LOGGER.info("Loading SVD warm-start word embeddings from %s", svd_init_source)
        src_model = AutoModel.from_pretrained(svd_init_source)
        src_weight = src_model.get_input_embeddings().weight.detach().float().cpu()
        del src_model
        if student_tokenizer is not None:
            try:
                src_tokenizer = AutoTokenizer.from_pretrained(svd_init_source)
            except Exception:
                src_tokenizer = None

    dst_weight = student.embeddings.word_embeddings.weight
    dst_vocab, dst_dim = dst_weight.shape

    aligned = src_weight
    if src_weight.shape[0] != dst_vocab:
        if student_tokenizer is not None and src_tokenizer is not None:
            src_vocab_map = src_tokenizer.get_vocab()
            dst_vocab_map = student_tokenizer.get_vocab()
            aligned = torch.randn(dst_vocab, src_weight.shape[1]) * float(src_weight.std().clamp_min(1e-3))
            matched = 0
            for tok, dst_id in dst_vocab_map.items():
                src_id = src_vocab_map.get(tok)
                if src_id is not None and dst_id < dst_vocab and src_id < src_weight.shape[0]:
                    aligned[dst_id] = src_weight[src_id]
                    matched += 1
            LOGGER.info(
                "Aligned %d/%d student tokens to SVD warm-start model vocab.",
                matched,
                dst_vocab,
            )
        else:
            LOGGER.warning(
                "Skipping SVD warm-start: vocab mismatch (src=%d vs student=%d) and no tokenizers to align.",
                src_weight.shape[0],
                dst_vocab,
            )
            return False

    # Compute compact SVD on float32 matrix
    u, s, _ = torch.linalg.svd(aligned, full_matrices=False)
    k = min(dst_dim, u.shape[1])
    proj = u[:, :k] * s[:k].unsqueeze(0)

    # Match source embedding standard deviation for stable BERT LayerNorm conditioning
    target_std = float(src_weight.std().clamp_min(1e-3))
    proj_std = float(proj.std().clamp_min(1e-6))
    proj = proj * (target_std / proj_std)

    if k < dst_dim:
        pad = torch.randn(dst_vocab, dst_dim - k) * float(student.config.initializer_range)
        proj = torch.cat([proj, pad], dim=1)

    dst_weight.copy_(proj.to(device=dst_weight.device, dtype=dst_weight.dtype))
    LOGGER.info(
        "Initialized student word embeddings (%dx%d) via top-%d SVD from %dx%d source.",
        dst_vocab,
        dst_dim,
        k,
        src_weight.shape[0],
        src_weight.shape[1],
    )
    return True


class BertMLMHead(nn.Module):
    """Standard BERT MLM prediction head with weights tied to word embeddings."""

    def __init__(self, config: BertConfig, word_embeddings_weight: nn.Parameter):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.act = nn.GELU()
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=True)
        self.decoder.weight = word_embeddings_weight

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(self.act(self.dense(hidden_states)))
        return self.decoder(x)


class TinyOncBertForDistillation(nn.Module):
    """Student ``BertModel`` plus distillation-only projection, polarity, and MLM heads."""

    def __init__(self, config: BertConfig, teacher_dim: int):
        super().__init__()
        self.bert = BertModel(config)
        self.teacher_proj = nn.Linear(config.hidden_size, teacher_dim)
        self.polarity_head = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, 3),
        )
        self.mlm_head = BertMLMHead(config, self.bert.embeddings.word_embeddings.weight)

    def encode_sentences(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return pool_embeddings(out.last_hidden_state, attention_mask, pooling="mean")

    def compute_mlm_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        masked_positions = labels != -100
        if not bool(masked_positions.any().item()):
            return self.bert.embeddings.word_embeddings.weight.new_tensor(0.0, dtype=torch.float32)
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Project only masked token states to vocab size (saves ~85% MLM head FLOPs/VRAM)
        selected_hidden = out.last_hidden_state[masked_positions]
        logits = self.mlm_head(selected_hidden).float()
        targets = labels[masked_positions]
        return F.cross_entropy(logits, targets)


# ---------------------------------------------------------------------------
# Masking & Datasets
# ---------------------------------------------------------------------------


def mask_tokens_bert(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    special_tokens_mask: torch.Tensor,
    *,
    vocab_size: int,
    mask_token_id: int,
    mlm_probability: float = 0.15,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply standard 15% BERT masking (80% [MASK], 10% random token, 10% unchanged)."""
    masked_inputs = input_ids.clone()
    labels = input_ids.clone()

    eligible = (attention_mask == 1) & (special_tokens_mask == 0)
    rand = torch.rand(input_ids.shape, generator=generator, device=input_ids.device)
    masked_indices = (rand < mlm_probability) & eligible

    # Ensure at least one eligible token is masked when possible so MLM gradient is active
    if not bool(masked_indices.any().item()) and bool(eligible.any().item()):
        eligible_flat = torch.nonzero(eligible.view(-1), as_tuple=False).squeeze(-1)
        pick_idx = int(
            torch.randint(len(eligible_flat), (1,), generator=generator, device=input_ids.device).item()
        )
        masked_indices.view(-1)[eligible_flat[pick_idx]] = True

    labels[~masked_indices] = -100

    # 80% of masked positions -> [MASK]
    decision = torch.rand(input_ids.shape, generator=generator, device=input_ids.device)
    replace_mask = masked_indices & (decision < 0.8)
    masked_inputs[replace_mask] = mask_token_id

    # 10% of masked positions -> random vocab token
    replace_random = masked_indices & (decision >= 0.8) & (decision < 0.9)
    if bool(replace_random.any().item()):
        random_words = torch.randint(
            low=0,
            high=vocab_size,
            size=input_ids.shape,
            dtype=input_ids.dtype,
            generator=generator,
            device=input_ids.device,
        )
        masked_inputs[replace_random] = random_words[replace_random]

    # Remaining 10% stay unchanged
    return masked_inputs, labels


class SentenceDistillationDataset(Dataset):
    """Indexed view over curated sentences, polarity labels, flips, and memmap teacher targets."""

    def __init__(
        self,
        texts: Sequence[str],
        polarities: np.ndarray,
        flipped_texts: Sequence[str],
        teacher_emb_path: Path,
        indices: Sequence[int],
    ):
        self.texts = texts
        self.polarities = polarities
        self.flipped_texts = flipped_texts
        self.teacher_emb_path = str(teacher_emb_path)
        self.indices = np.asarray(indices, dtype=np.int64)
        self._mmap: np.ndarray | None = None

    def _get_mmap(self) -> np.ndarray:
        if self._mmap is None:
            self._mmap = np.load(self.teacher_emb_path, mmap_mode="r")
        return self._mmap

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[str, int, str, np.ndarray]:
        row = int(self.indices[idx])
        emb = np.array(self._get_mmap()[row], dtype=np.float32, copy=True)
        return self.texts[row], int(self.polarities[row]), self.flipped_texts[row], emb


class WindowMLMDataset(Dataset):
    """Dataset of contiguous 512-token clinical note windows for MLM."""

    def __init__(self, windows: Sequence[str]):
        self.windows = list(windows)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> str:
        return self.windows[idx]


def _make_sentence_collator(
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    mlm_probability: float,
):
    vocab_size = len(tokenizer)
    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        raise ValueError("Student tokenizer must define a mask_token_id for MLM")

    def collate(batch: list[tuple[str, int, str, np.ndarray]]) -> dict[str, torch.Tensor]:
        texts = [item[0] for item in batch]
        polarities = torch.tensor([item[1] for item in batch], dtype=torch.long)
        teacher_embs = torch.from_numpy(np.stack([item[3] for item in batch], axis=0))

        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        masked_ids, mlm_labels = mask_tokens_bert(
            encoded["input_ids"],
            encoded["attention_mask"],
            encoded["special_tokens_mask"],
            vocab_size=vocab_size,
            mask_token_id=mask_token_id,
            mlm_probability=mlm_probability,
        )

        flip_indices: list[int] = []
        flip_texts: list[str] = []
        for idx, item in enumerate(batch):
            if item[2]:
                flip_indices.append(idx)
                flip_texts.append(item[2])

        out: dict[str, torch.Tensor] = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "mlm_input_ids": masked_ids,
            "mlm_labels": mlm_labels,
            "polarities": polarities,
            "teacher_embs": teacher_embs,
            "flip_indices": torch.tensor(flip_indices, dtype=torch.long),
        }
        if flip_texts:
            flip_enc = tokenizer(
                flip_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            out["flip_input_ids"] = flip_enc["input_ids"]
            out["flip_attention_mask"] = flip_enc["attention_mask"]
        return out

    return collate


def _make_window_collator(
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    mlm_probability: float,
):
    vocab_size = len(tokenizer)
    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        raise ValueError("Student tokenizer must define a mask_token_id for MLM")

    def collate(texts: list[str]) -> dict[str, torch.Tensor]:
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        masked_ids, mlm_labels = mask_tokens_bert(
            encoded["input_ids"],
            encoded["attention_mask"],
            encoded["special_tokens_mask"],
            vocab_size=vocab_size,
            mask_token_id=mask_token_id,
            mlm_probability=mlm_probability,
        )
        return {
            "mlm_input_ids": masked_ids,
            "attention_mask": encoded["attention_mask"],
            "mlm_labels": mlm_labels,
        }

    return collate


# ---------------------------------------------------------------------------
# Loss computation, validation & training loop
# ---------------------------------------------------------------------------


def compute_sentence_step_losses(
    model: TinyOncBertForDistillation,
    batch: dict[str, torch.Tensor],
    cfg: TinyOncBertConfig,
) -> dict[str, torch.Tensor]:
    """Compute all 4 sentence-level training objectives in float32."""
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    polarities = batch["polarities"]
    teacher_embs = F.normalize(batch["teacher_embs"].float(), p=2, dim=-1)

    # 1. Native student sentence representations
    s = model.encode_sentences(input_ids, attention_mask)
    s_native_norm = F.normalize(s.float(), p=2, dim=-1)

    # Term 1: Projected teacher alignment (cosine + MSE)
    s_proj = F.normalize(model.teacher_proj(s).float(), p=2, dim=-1)
    cos_teacher = (s_proj * teacher_embs).sum(dim=-1)
    emb_loss = (1.0 - cos_teacher).mean() + F.mse_loss(s_proj, teacher_embs)

    # Term 2: Unprojected relational similarity geometry in native 384D space
    rel_loss = F.mse_loss(s_native_norm @ s_native_norm.T, teacher_embs @ teacher_embs.T)

    # Term 3: Polarity classification + flipped hard-negative margin loss on native 384D embeddings
    pol_logits = model.polarity_head(s).float()
    pol_loss = F.cross_entropy(pol_logits, polarities)

    flip_indices = batch["flip_indices"]
    if flip_indices.numel() > 0 and "flip_input_ids" in batch:
        s_flip = model.encode_sentences(batch["flip_input_ids"], batch["flip_attention_mask"])
        s_flip_norm = F.normalize(s_flip.float(), p=2, dim=-1)
        cos_flip = (s_native_norm[flip_indices] * s_flip_norm).sum(dim=-1)
        neg_loss = F.relu(cos_flip - cfg.neg_margin).mean()
        # Flipped sentences have inverted binary polarity (0 <-> 1)
        flip_targets = (1 - polarities[flip_indices]).clamp(0, 1)
        pol_loss_flip = F.cross_entropy(model.polarity_head(s_flip).float(), flip_targets)
        pol_loss = 0.5 * (pol_loss + pol_loss_flip)
    else:
        neg_loss = s.new_tensor(0.0, dtype=torch.float32)

    # Term 4: Sentence-level MLM
    if cfg.lambda_mlm > 0.0:
        mlm_loss = model.compute_mlm_loss(
            batch["mlm_input_ids"], attention_mask, batch["mlm_labels"]
        )
    else:
        mlm_loss = s.new_tensor(0.0, dtype=torch.float32)

    total_loss = (
        cfg.lambda_emb * emb_loss
        + cfg.lambda_rel * rel_loss
        + cfg.lambda_neg * neg_loss
        + cfg.lambda_pol * pol_loss
        + cfg.lambda_mlm * mlm_loss
    )
    return {
        "loss": total_loss,
        "emb_loss": emb_loss,
        "rel_loss": rel_loss,
        "neg_loss": neg_loss,
        "pol_loss": pol_loss,
        "mlm_loss": mlm_loss,
    }


@torch.no_grad()
def evaluate_student(
    model: TinyOncBertForDistillation,
    val_sent_loader: DataLoader,
    val_win_loader: DataLoader | None,
    device: torch.device,
    use_bf16: bool,
) -> dict[str, float]:
    """Evaluate teacher cosine alignment, negation-flip cosine separation, and MLM loss."""
    model.eval()
    amp_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16
        else contextlib.nullcontext()
    )

    teacher_cos_sum = 0.0
    teacher_cos_count = 0
    neg_flip_cos_sum = 0.0
    neg_flip_count = 0
    pol_correct = 0
    pol_total = 0
    mlm_loss_sum = 0.0
    mlm_batches = 0

    for batch in val_sent_loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with amp_ctx:
            s = model.encode_sentences(batch["input_ids"], batch["attention_mask"])
            s_native_norm = F.normalize(s.float(), p=2, dim=-1)
            s_proj = F.normalize(model.teacher_proj(s).float(), p=2, dim=-1)
            t = F.normalize(batch["teacher_embs"].float(), p=2, dim=-1)

            cos_t = (s_proj * t).sum(dim=-1)
            teacher_cos_sum += float(cos_t.sum().item())
            teacher_cos_count += int(cos_t.numel())

            pol_preds = model.polarity_head(s).argmax(dim=-1)
            pol_correct += int((pol_preds == batch["polarities"]).sum().item())
            pol_total += int(batch["polarities"].numel())

            flip_indices = batch["flip_indices"]
            if flip_indices.numel() > 0 and "flip_input_ids" in batch:
                s_flip = model.encode_sentences(batch["flip_input_ids"], batch["flip_attention_mask"])
                s_flip_norm = F.normalize(s_flip.float(), p=2, dim=-1)
                cos_f = (s_native_norm[flip_indices] * s_flip_norm).sum(dim=-1)
                neg_flip_cos_sum += float(cos_f.sum().item())
                neg_flip_count += int(cos_f.numel())

            mlm_loss = model.compute_mlm_loss(
                batch["mlm_input_ids"], batch["attention_mask"], batch["mlm_labels"]
            )
            mlm_loss_sum += float(mlm_loss.item())
            mlm_batches += 1

    win_mlm_sum = 0.0
    win_mlm_batches = 0
    if val_win_loader is not None:
        for wbatch in val_win_loader:
            wbatch = {k: v.to(device, non_blocking=True) for k, v in wbatch.items()}
            with amp_ctx:
                wloss = model.compute_mlm_loss(
                    wbatch["mlm_input_ids"], wbatch["attention_mask"], wbatch["mlm_labels"]
                )
            win_mlm_sum += float(wloss.item())
            win_mlm_batches += 1

    model.train()
    metrics: dict[str, Any] = {
        "val_teacher_cos": teacher_cos_sum / max(1, teacher_cos_count),
        "val_neg_flip_cos": (neg_flip_cos_sum / neg_flip_count if neg_flip_count > 0 else None),
        "val_mlm_loss": mlm_loss_sum / max(1, mlm_batches),
        "val_pol_acc": pol_correct / max(1, pol_total),
    }
    if win_mlm_batches > 0:
        metrics["val_window_mlm_loss"] = win_mlm_sum / win_mlm_batches
    return metrics


def _cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    total = max(1, num_training_steps)
    warmup = max(0, min(num_warmup_steps, total))

    def lr_lambda(current_step: int) -> float:
        if warmup > 0 and current_step < warmup:
            return float(current_step + 1) / float(warmup)
        progress = float(current_step - warmup) / float(max(1, total - warmup))
        progress = min(max(progress, 0.0), 1.0)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _infinite_loader(loader: DataLoader) -> Iterator[dict[str, torch.Tensor]]:
    while True:
        for batch in loader:
            yield batch


def save_clean_student(
    model: TinyOncBertForDistillation,
    tokenizer: PreTrainedTokenizerBase,
    output_dir: str | Path,
    cfg: TinyOncBertConfig,
) -> Path:
    """Save only the clean ``BertModel`` + tokenizer (no distillation heads) via ``save_pretrained``."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    raw_bert = raw_model.bert._orig_mod if hasattr(raw_model.bert, "_orig_mod") else raw_model.bert
    raw_bert.save_pretrained(out_path)
    tokenizer.save_pretrained(out_path)
    (out_path / "tinyoncbert_train_config.json").write_text(
        json.dumps(asdict(cfg), indent=2), encoding="utf-8"
    )
    return out_path


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_tinyoncbert(
    cfg: TinyOncBertConfig,
    *,
    tokenizer_obj: PreTrainedTokenizerBase | None = None,
    teacher_model_obj: nn.Module | None = None,
    teacher_tokenizer_obj: PreTrainedTokenizerBase | None = None,
    svd_init_source: str | nn.Module | torch.Tensor | None = None,
    stage: str = "all",
) -> dict[str, Any]:
    """Run the full TinyOncBERT pipeline (curate -> teacher precompute -> train -> save)."""
    _seed_everything(cfg.seed)

    if cfg.num_workers > 0:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    tokenizer = (
        tokenizer_obj
        if tokenizer_obj is not None
        else AutoTokenizer.from_pretrained(cfg.tokenizer_model, use_fast=True)
    )

    manifest = curate_corpus(cfg, tokenizer=tokenizer)
    if stage == "curate":
        return {"manifest": manifest}

    sent_data = load_curated_sentences(cfg.work_dir, manifest)
    teacher_emb_path, teacher_dim = precompute_teacher_embeddings(
        cfg,
        sent_data["texts"],
        manifest["signature"],
        teacher_model_obj=teacher_model_obj,
        teacher_tokenizer_obj=teacher_tokenizer_obj,
    )
    if stage == "teacher":
        return {
            "manifest": manifest,
            "teacher_emb_path": str(teacher_emb_path),
            "teacher_dim": teacher_dim,
        }

    windows = load_curated_windows(cfg.work_dir, manifest) if manifest["num_windows_actual"] > 0 else []

    n_total = len(sent_data["texts"])
    if n_total < 2:
        raise ValueError(f"Need at least 2 curated sentences to train, got {n_total}")
    n_val = min(cfg.val_sentences, max(1, n_total // 20))
    if n_val >= n_total:
        n_val = 1
    n_train = n_total - n_val

    train_indices = list(range(0, n_train))
    val_indices = list(range(n_train, n_total))

    train_sent_ds = SentenceDistillationDataset(
        sent_data["texts"],
        sent_data["polarities"],
        sent_data["flipped_texts"],
        teacher_emb_path,
        train_indices,
    )
    val_sent_ds = SentenceDistillationDataset(
        sent_data["texts"],
        sent_data["polarities"],
        sent_data["flipped_texts"],
        teacher_emb_path,
        val_indices,
    )

    device = torch.device(cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    pin_mem = device.type == "cuda"
    sent_collator = _make_sentence_collator(tokenizer, cfg.sentence_max_length, cfg.mlm_probability)
    win_collator = _make_window_collator(tokenizer, cfg.window_max_length, cfg.mlm_probability)

    eff_batch = min(cfg.batch_size, n_train)
    train_sent_loader = DataLoader(
        train_sent_ds,
        batch_size=eff_batch,
        shuffle=True,
        drop_last=n_train > eff_batch,
        num_workers=cfg.num_workers,
        pin_memory=pin_mem,
        persistent_workers=cfg.num_workers > 0,
        collate_fn=sent_collator,
    )
    val_sent_loader = DataLoader(
        val_sent_ds,
        batch_size=min(cfg.batch_size, n_val),
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_mem,
        persistent_workers=cfg.num_workers > 0,
        collate_fn=sent_collator,
    )

    train_win_iter: Iterator[dict[str, torch.Tensor]] | None = None
    val_win_loader: DataLoader | None = None
    if windows:
        n_win_val = min(cfg.val_windows, max(1, len(windows) // 20)) if len(windows) >= 2 else 0
        train_windows = windows[:-n_win_val] if n_win_val > 0 else windows
        val_windows = windows[-n_win_val:] if n_win_val > 0 else []
        eff_win_batch = min(cfg.window_batch_size, len(train_windows))
        train_win_loader = DataLoader(
            WindowMLMDataset(train_windows),
            batch_size=eff_win_batch,
            shuffle=True,
            drop_last=len(train_windows) > eff_win_batch,
            num_workers=cfg.num_workers,
            pin_memory=pin_mem,
            persistent_workers=cfg.num_workers > 0,
            collate_fn=win_collator,
        )
        train_win_iter = _infinite_loader(train_win_loader)
        if val_windows:
            val_win_loader = DataLoader(
                WindowMLMDataset(val_windows),
                batch_size=min(cfg.window_batch_size, len(val_windows)),
                shuffle=False,
                num_workers=cfg.num_workers,
                pin_memory=pin_mem,
                persistent_workers=cfg.num_workers > 0,
                collate_fn=win_collator,
            )

    tokenizer.padding_side = "right"
    # Build student model
    student_config = build_student_config(cfg, tokenizer)
    model = TinyOncBertForDistillation(student_config, teacher_dim=teacher_dim)

    svd_src = svd_init_source if svd_init_source is not None else cfg.svd_init_model
    if svd_src is not None and (not isinstance(svd_src, str) or svd_src.strip()):
        initialize_word_embeddings_svd(model.bert, svd_src, student_tokenizer=tokenizer)

    model.to(device)
    if cfg.compile_model and hasattr(torch, "compile"):
        LOGGER.info("Compiling TinyOncBERT BertModel backbone with torch.compile")
        model.bert = torch.compile(model.bert)  # type: ignore[assignment]

    # AdamW with standard BERT no-decay on biases and LayerNorm weights
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name for nd in no_decay):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
        betas=(0.9, 0.98),
        eps=1e-6,
    )

    total_steps = max(1, cfg.epochs * len(train_sent_loader))
    warmup_steps = int(round(total_steps * cfg.warmup_ratio))
    scheduler = _cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"

    amp_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16
        else contextlib.nullcontext()
    )

    LOGGER.info(
        "Starting TinyOncBERT training: %d train sents, %d val sents, %d windows, "
        "%d epochs (%d steps), batch=%d, window_batch=%d, bf16=%s",
        n_train,
        n_val,
        len(windows),
        cfg.epochs,
        total_steps,
        eff_batch,
        min(cfg.window_batch_size, len(windows)) if windows else 0,
        use_bf16,
    )

    model.train()
    global_step = 0
    history: list[dict[str, Any]] = []
    t_start = time.time()
    window_mlm_steps = 0

    for epoch in range(1, cfg.epochs + 1):
        for batch in train_sent_loader:
            global_step += 1
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)

            with amp_ctx:
                losses = compute_sentence_step_losses(model, batch, cfg)
            losses["loss"].backward()

            win_mlm_val: float | None = None
            if (
                train_win_iter is not None
                and cfg.window_mlm_every > 0
                and cfg.lambda_mlm > 0.0
                and global_step % cfg.window_mlm_every == 0
            ):
                wbatch = next(train_win_iter)
                wbatch = {k: v.to(device, non_blocking=True) for k, v in wbatch.items()}
                with amp_ctx:
                    w_mlm_loss = model.compute_mlm_loss(
                        wbatch["mlm_input_ids"], wbatch["attention_mask"], wbatch["mlm_labels"]
                    )
                (cfg.lambda_mlm * w_mlm_loss).backward()
                win_mlm_val = float(w_mlm_loss.item())
                window_mlm_steps += 1

            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip).item())
            optimizer.step()
            scheduler.step()

            if global_step == 1 or global_step % cfg.log_every == 0 or global_step == total_steps:
                rec: dict[str, Any] = {
                    "type": "train",
                    "step": global_step,
                    "epoch": epoch,
                    "lr": float(scheduler.get_last_lr()[0]),
                    "loss": float(losses["loss"].item()),
                    "emb_loss": float(losses["emb_loss"].item()),
                    "rel_loss": float(losses["rel_loss"].item()),
                    "neg_loss": float(losses["neg_loss"].item()),
                    "pol_loss": float(losses["pol_loss"].item()),
                    "mlm_loss": float(losses["mlm_loss"].item()),
                    "grad_norm": grad_norm,
                    "elapsed_sec": round(time.time() - t_start, 2),
                }
                if win_mlm_val is not None:
                    rec["window_mlm_loss"] = win_mlm_val
                history.append(rec)
                with metrics_path.open("a", encoding="utf-8") as mf:
                    mf.write(json.dumps(rec) + "\n")
                LOGGER.info(
                    "step %d/%d (ep %d) | loss=%.4f emb=%.4f rel=%.4f neg=%.4f pol=%.4f mlm=%.4f%s | lr=%.2e",
                    global_step,
                    total_steps,
                    epoch,
                    rec["loss"],
                    rec["emb_loss"],
                    rec["rel_loss"],
                    rec["neg_loss"],
                    rec["pol_loss"],
                    rec["mlm_loss"],
                    f" win_mlm={win_mlm_val:.4f}" if win_mlm_val is not None else "",
                    rec["lr"],
                )

            if cfg.eval_every > 0 and (global_step % cfg.eval_every == 0 or global_step == total_steps):
                val_metrics = evaluate_student(model, val_sent_loader, val_win_loader, device, use_bf16)
                val_rec = {
                    "type": "val",
                    "step": global_step,
                    "epoch": epoch,
                    **val_metrics,
                    "elapsed_sec": round(time.time() - t_start, 2),
                }
                history.append(val_rec)
                with metrics_path.open("a", encoding="utf-8") as mf:
                    mf.write(json.dumps(val_rec) + "\n")
                LOGGER.info(
                    "VAL step %d | val_teacher_cos=%.4f | val_neg_flip_cos=%s | val_mlm_loss=%.4f | val_pol_acc=%.4f",
                    global_step,
                    val_metrics["val_teacher_cos"],
                    (
                        f"{val_metrics['val_neg_flip_cos']:.4f}"
                        if val_metrics["val_neg_flip_cos"] is not None
                        else "n/a"
                    ),
                    val_metrics["val_mlm_loss"],
                    val_metrics["val_pol_acc"],
                )
                save_clean_student(model, tokenizer, output_dir, cfg)

    saved_path = save_clean_student(model, tokenizer, output_dir, cfg)
    LOGGER.info("Saved clean TinyOncBERT model + tokenizer to %s", saved_path)

    return {
        "output_dir": str(saved_path),
        "manifest": manifest,
        "teacher_emb_path": str(teacher_emb_path),
        "teacher_dim": teacher_dim,
        "total_steps": global_step,
        "window_mlm_steps": window_mlm_steps,
        "history": history,
    }


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m oci.longitudinal.tinyoncbert",
        description="Curate oncology notes, precompute teacher embeddings, and distill TinyOncBERT.",
    )
    parser.add_argument(
        "--stage",
        choices=("all", "curate", "teacher", "train"),
        default="all",
        help="Pipeline stage to run (curation and teacher precomputation auto-resume if cached).",
    )
    parser.add_argument("--parquet", default=DEFAULT_PARQUET, help="Local or hf:// parquet path.")
    parser.add_argument(
        "--work-dir",
        default="artifacts/tinyoncbert_6l_384d/work",
        help="Working cache directory for curated shards and memmap teacher embeddings.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/tinyoncbert_6l_384d",
        help="Output directory for clean HuggingFace BertModel + tokenizer + metrics.jsonl.",
    )
    parser.add_argument("--split-filter", default="train", help="Split column value to retain.")
    parser.add_argument("--num-sentences", type=int, default=2_000_000)
    parser.add_argument("--num-windows", type=int, default=250_000)
    parser.add_argument("--val-sentences", type=int, default=4_096)
    parser.add_argument("--val-windows", type=int, default=512)
    parser.add_argument("--markdown-strip-prob", type=float, default=0.5)
    parser.add_argument("--sentence-max-length", type=int, default=64)
    parser.add_argument("--window-max-length", type=int, default=512)

    parser.add_argument("--teacher-model", default=DEFAULT_TEACHER_MODEL)
    parser.add_argument(
        "--teacher-pooling", choices=("auto", "last", "mean"), default="auto"
    )
    parser.add_argument("--teacher-batch-size", type=int, default=1024)
    parser.add_argument("--teacher-max-length", type=int, default=128)

    parser.add_argument("--tokenizer-model", default=DEFAULT_TOKENIZER_MODEL)
    parser.add_argument(
        "--svd-init-model",
        default=DEFAULT_SVD_INIT_MODEL,
        help="Pretrained model for SVD word embedding warm-start (pass '' to skip).",
    )
    parser.add_argument("--num-hidden-layers", type=int, default=6)
    parser.add_argument("--hidden-size", type=int, default=384)
    parser.add_argument("--num-attention-heads", type=int, default=6)
    parser.add_argument("--intermediate-size", type=int, default=1536)
    parser.add_argument("--max-position-embeddings", type=int, default=512)
    parser.add_argument("--hidden-act", default="gelu")

    parser.add_argument("--lambda-emb", type=float, default=1.0)
    parser.add_argument("--lambda-rel", type=float, default=0.5)
    parser.add_argument("--lambda-neg", type=float, default=0.5)
    parser.add_argument("--lambda-pol", type=float, default=0.1)
    parser.add_argument("--lambda-mlm", type=float, default=0.25)
    parser.add_argument("--neg-margin", type=float, default=0.25)
    parser.add_argument("--mlm-probability", type=float, default=0.15)
    parser.add_argument("--window-mlm-every", type=int, default=6)

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--window-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compile", dest="compile_model", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    stage = args.stage
    cfg_kwargs = {k: v for k, v in vars(args).items() if k != "stage"}
    cfg = TinyOncBertConfig(**cfg_kwargs)
    train_tinyoncbert(cfg, stage=stage)


if __name__ == "__main__":
    main()
