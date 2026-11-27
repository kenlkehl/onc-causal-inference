"""Option A shared multimodal feature builder and Option B (v6) hybrid warm-started fine-tuning.

Architecture overview
---------------------
* **Option A (Zero-Shot Composition):**
  Combines a frozen distilled clinical encoder (``TinyOncBERT``) with lean unmixed
  longitudinal patient features (default ``sent_256``: 220-d structured summary + 256-d
  context-fitted spherical k-means sentence MaxSim over all notes; optional ``option_a_full``
  adds 153-d NaN-preserving nadir/peak/delta, 64-d ICD/CPT SVD, 256-d latest-note SentKM,
  and 128-d lexical 1-3gram SVD) and feeds them directly into a frozen
  ``SparseTabularCausalPFN`` head.

* **Option B (v6 Hybrid Warm-Started Fine-Tuning):**
  Keeps ``TinyOncBERT`` completely frozen (preserving full-LLM note generalization and avoiding
  10x BERT backprop overhead), keeps all explicit Option-A columns unmixed, and appends
  ``learned_temporal_dim`` (default 24) trainable trajectory columns produced by a lightweight
  ``StructuredTemporalEncoder`` (labs, vitals, encounters, PROs ``EventTransformer`` +
  ``CrossModalTimeline``, optionally warm-started from Longitudinal Run 5) alongside the explicit
  columns. A ``SparseTabularCausalPFN`` head warm-started from Tabular v4 is lightly fine-tuned
  jointly with the temporal encoder on a 50/50 mixture of:
    1. Longitudinal multimodal episodes (``nemotron_5000.json.gz`` ``split == "train"`` +
       ``nemotron_5000_paraphrases.json.gz``; outcome + CATE + propensity losses, no role loss).
    2. Pure-tabular replay episodes (``oci.tabular_pfn.prior.sample_tabular_study`` matching
       Tabular v4's exact ``PriorConfig`` on ``design_split == "train"``; full 6-class role +
       screener + causal losses), plus an optional L2-SP anchor toward the pretrained Tabular v4
       weights to prevent tabular forgetting.
"""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
import inspect
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from synthetic_data.paraphrase_bank import load_paraphrase_bank
from synthetic_data.study_designs import DEFAULT_BANK, load_bank

from oci.tabular_pfn.model import (
    SparseTabularCausalPFN,
    TabularMemory,
    TabularModelConfig,
    _apply_standardize,
    _apply_v3_basis,
    _column_sufficient_stats,
    _compute_v3_basis_context,
    _fit_ard_loo_cv_ridge,
    _pair_empirical_corr,
    _robinson_all_p_stats,
    _robinson_confounder_mod_stats,
    _robinson_epistasis_all_pairs,
    _standardize_context,
)
from oci.tabular_pfn.prior import PriorConfig, sample_tabular_study
from oci.tabular_pfn.training import evaluate_tabular, load_tabular_run

from .events import (
    CODE_BUCKETS,
    ENCOUNTER_SUMMARY_SIZE,
    EXTRA_LAB_BUCKETS,
    EXTRA_PRO_BUCKETS,
    LAB_INPUT_SIZE,
    LAB_NAMES,
    LAB_SCALES,
    LAB_SUMMARY_SIZE,
    LAB_UNITS,
    PRO_INPUT_SIZE,
    PRO_INSTRUMENT,
    PRO_ITEMS,
    PRO_SUMMARY_SIZE,
    RECALL_DAYS,
    VITAL_INPUT_SIZE,
    VITAL_SCALES,
    VITAL_SUMMARY_SIZE,
    _numeric,
    binary_observations,
    encounter_codes,
    encounter_stream_summary,
    event_times,
    lab_features,
    lab_stream_summary,
    note_lexical_vector,
    note_sentences,
    patient_summary_vector,
    pro_features,
    pro_stream_summary,
    vital_stream_summary,
    vital_vector,
)
from .model import (
    _BIDIRECTIONAL_MODELS,
    ArmScale,
    CrossModalTimeline,
    EventTransformer,
    SentenceBatch,
    _SummaryProjector,
    _fit_lexical_basis,
    _fit_sentence_concepts,
    _predict_ridge,
    _project_sentence_maxsim,
)
from .training import (
    DESIGN_BANK_FILENAME,
    PARAPHRASE_BANK_FILENAME,
    PROPENSITY_COL,
    TARGETS,
    _aggregate_eval,
    _autocast,
    _release_host_memory,
    _ridge_t_learner_cate,
    _select_episode_design,
    _simulate_for_design,
    _width_bucket,
    _write_gzip_json,
    load_run_design_bank,
    load_run_paraphrase_bank_path,
    split_designs,
)

OPTION_B_CHECKPOINT_VERSION = 1
STRUCTURED_STREAMS = ("labs", "vitals", "encounters", "pros")


# ==============================================================================
# 1. SHARED OPTION-A DETERMINISTIC MULTIMODAL FEATURE PIPELINE
# ==============================================================================
@dataclass(frozen=True)
class OptionAFeatureConfig:
    """Configuration for deterministic explicit longitudinal multimodal feature extraction.

    Defaults to the lean ``sent_256`` structure (220-d structured summary + 256-d context-fitted
    TinyOncBERT spherical k-means sentence MaxSim = 476 columns), which outperformed wider
    collinear feature tables zero-shot on full-LLM cohorts.
    """

    include_nadir_peak: bool = False
    icd_svd_dim: int = 0
    sent_km_k: int = 256
    include_latest_sent_km: bool = False
    lex_svd_dim: int = 0
    sent_batch_size: int = 256
    sent_max_length: int = 96


OPTION_A_PRESETS: dict[str, OptionAFeatureConfig] = {
    "struct_only": OptionAFeatureConfig(
        include_nadir_peak=False,
        icd_svd_dim=0,
        sent_km_k=0,
        include_latest_sent_km=False,
        lex_svd_dim=0,
    ),
    "sent_256": OptionAFeatureConfig(
        include_nadir_peak=False,
        icd_svd_dim=0,
        sent_km_k=256,
        include_latest_sent_km=False,
        lex_svd_dim=0,
    ),
    "sent_256_lex_128": OptionAFeatureConfig(
        include_nadir_peak=False,
        icd_svd_dim=0,
        sent_km_k=256,
        include_latest_sent_km=False,
        lex_svd_dim=128,
    ),
    "option_a_temp128": OptionAFeatureConfig(
        include_nadir_peak=True,
        icd_svd_dim=64,
        sent_km_k=128,
        include_latest_sent_km=True,
        lex_svd_dim=0,
    ),
    "option_a_full": OptionAFeatureConfig(
        include_nadir_peak=True,
        icd_svd_dim=64,
        sent_km_k=256,
        include_latest_sent_km=True,
        lex_svd_dim=128,
    ),
}


def nadir_peak_features(row: dict) -> list[float]:
    """Extract 153-d min/max/delta trajectory features with genuine NaNs for missing panels."""
    feats: list[float] = []
    lab_series = {name: [] for name in LAB_NAMES}
    for panel in row.get("labs") or ():
        for item in panel.get("results") or ():
            name = item.get("component")
            val = item.get("value")
            if name in lab_series and val is not None and item.get("unit") == LAB_UNITS[name]:
                lab_series[name].append(_numeric(val, *LAB_SCALES[name]))
    for name in LAB_NAMES:
        vs = lab_series[name]
        feats.extend([min(vs), max(vs), vs[-1] - vs[0]] if vs else [float("nan")] * 3)

    vit_series = {name: [] for name in VITAL_SCALES}
    for vit in row.get("vitals") or ():
        for name, scale in VITAL_SCALES.items():
            if vit.get(name) is not None:
                vit_series[name].append(_numeric(vit[name], *scale))
    for name in VITAL_SCALES:
        vs = vit_series[name]
        feats.extend([min(vs), max(vs), vs[-1] - vs[0]] if vs else [float("nan")] * 3)

    pro_series = {item: [] for item in PRO_ITEMS}
    for form in row.get("pros") or ():
        if str(form.get("instrument", PRO_INSTRUMENT)) != PRO_INSTRUMENT:
            continue
        if form.get("recall_days", RECALL_DAYS) != RECALL_DAYS:
            continue
        for resp in form.get("responses") or ():
            key = (resp.get("symptom"), resp.get("attribute"))
            sc = resp.get("score")
            if key in pro_series and sc is not None:
                pro_series[key].append(float(sc) / 4.0)
    for item in PRO_ITEMS:
        vs = pro_series[item]
        feats.extend([min(vs), max(vs), vs[-1] - vs[0]] if vs else [float("nan")] * 3)
    return feats


def encounter_code_vector(row: dict) -> list[float]:
    """Hashed log1p encounter ICD/CPT/setting bucket counts (``CODE_BUCKETS`` dims)."""
    counts = [0.0] * CODE_BUCKETS
    for enc in row.get("encounters") or ():
        for cid in encounter_codes(enc):
            counts[cid] += 1.0
    return [math.log1p(c) if c else 0.0 for c in counts]


def _nadir_peak_feature_names() -> list[str]:
    names: list[str] = []
    for name in LAB_NAMES:
        names.extend([f"lab_{name}_min", f"lab_{name}_max", f"lab_{name}_delta"])
    for name in VITAL_SCALES:
        names.extend([f"vital_{name}_min", f"vital_{name}_max", f"vital_{name}_delta"])
    for sym, attr in PRO_ITEMS:
        names.extend([f"pro_{sym}_{attr}_min", f"pro_{sym}_{attr}_max", f"pro_{sym}_{attr}_delta"])
    return names


@torch.no_grad()
def encode_cohort_sentences(
    rows: list[dict],
    backbone,
    tokenizer,
    device: str | torch.device,
    *,
    batch_size: int = 256,
    max_length: int = 96,
    include_latest: bool = True,
) -> tuple[SentenceBatch, SentenceBatch | None]:
    """Embed deduplicated note sentences with a frozen clinical encoder (eval mode, zero leakage)."""
    dev = torch.device(device)
    per_pat_all = [note_sentences(row.get("notes") or ()) for row in rows]
    per_pat_latest = (
        [note_sentences([row["notes"][-1]]) if (row.get("notes") or ()) else [] for row in rows]
        if include_latest
        else None
    )
    uniq_set = {s for sents in per_pat_all for s in sents}
    if per_pat_latest is not None:
        uniq_set.update(s for sents in per_pat_latest for s in sents)
    uniq_texts = tuple(sorted(uniq_set))
    hidden_size = int(backbone.config.hidden_size)
    if not uniq_texts:
        empty = SentenceBatch(
            unique_embs=torch.empty((0, hidden_size), device=dev, dtype=torch.float32),
            flat_idx=torch.empty((0,), device=dev, dtype=torch.long),
            patient_ids=torch.empty((0,), device=dev, dtype=torch.long),
            num_patients=len(rows),
            unique_texts=(),
        )
        return empty, (empty if include_latest else None)

    is_bidir = getattr(backbone.config, "model_type", "") in _BIDIRECTIONAL_MODELS
    accepts_use_cache = "use_cache" in inspect.signature(backbone.forward).parameters
    limit = int(getattr(backbone.config, "max_position_embeddings", max_length))
    eff_len = min(max_length, limit)

    text_to_id = {text: idx for idx, text in enumerate(uniq_texts)}
    order = sorted(range(len(uniq_texts)), key=lambda i: (len(uniq_texts[i]), uniq_texts[i]))
    unique_embs = torch.empty((len(uniq_texts), hidden_size), device=dev, dtype=torch.float32)
    use_bf16 = dev.type == "cuda" and torch.cuda.is_bf16_supported()

    for start in range(0, len(order), batch_size):
        batch_order = order[start : start + batch_size]
        batch_texts = [uniq_texts[i] for i in batch_order]
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=eff_len,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(dev)
        kwargs = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
        if accepts_use_cache:
            kwargs["use_cache"] = False
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            hidden = backbone(**kwargs).last_hidden_state
        hidden = hidden.float()
        mask = enc["attention_mask"]
        if is_bidir:
            w = mask.unsqueeze(-1).float()
            pooled = (hidden * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)
        else:
            last_pos = mask.sum(dim=1).clamp_min(1) - 1
            pooled = hidden[torch.arange(len(batch_texts), device=dev), last_pos]
        pooled = F.normalize(pooled, p=2, dim=-1)
        unique_embs[torch.tensor(batch_order, device=dev, dtype=torch.long)] = pooled

    def _build(per_pat):
        f_idx, p_ids = [], []
        for p_idx, sents in enumerate(per_pat):
            for s in sents:
                f_idx.append(text_to_id[s])
                p_ids.append(p_idx)
        return SentenceBatch(
            unique_embs=unique_embs,
            flat_idx=torch.tensor(f_idx, device=dev, dtype=torch.long),
            patient_ids=torch.tensor(p_ids, device=dev, dtype=torch.long),
            num_patients=len(rows),
            unique_texts=uniq_texts,
        )

    sb_all = _build(per_pat_all)
    sb_latest = _build(per_pat_latest) if per_pat_latest is not None else None
    return sb_all, sb_latest


def slice_sentence_batch(sb: SentenceBatch, start: int, end: int) -> SentenceBatch:
    """Slice a ``SentenceBatch`` to patient index range ``[start, end)`` without leakage."""
    mask = (sb.patient_ids >= start) & (sb.patient_ids < end)
    sub_flat = sb.flat_idx[mask]
    sub_pat = sb.patient_ids[mask] - start
    if len(sub_flat) == 0:
        return SentenceBatch(
            unique_embs=sb.unique_embs.new_empty((0, sb.unique_embs.shape[1])),
            flat_idx=sub_flat,
            patient_ids=sub_pat,
            num_patients=end - start,
            unique_texts=(),
        )
    used_uniq, inv = torch.unique(sub_flat, sorted=True, return_inverse=True)
    return SentenceBatch(
        unique_embs=sb.unique_embs[used_uniq],
        flat_idx=inv,
        patient_ids=sub_pat,
        num_patients=end - start,
        unique_texts=(),
    )


@dataclass
class PrecomputedCohortFeatures:
    """Per-patient raw features computed once per cohort for fast multi-preset evaluation."""

    x_struct: torch.Tensor  # (N, 220) float32
    x_nadir_peak: torch.Tensor | None  # (N, 96) float32 with genuine NaNs
    code_raw: torch.Tensor | None  # (N, CODE_BUCKETS) float32
    lex_raw: torch.Tensor | None  # (N, LEX_BUCKETS) float32
    sb_all: SentenceBatch | None
    sb_latest: SentenceBatch | None


@torch.no_grad()
def precompute_cohort_raw_features(
    rows: list[dict],
    device: str | torch.device,
    *,
    backbone=None,
    tokenizer=None,
    need_nadir_peak: bool = True,
    need_icd: bool = True,
    need_lex: bool = True,
    need_sent: bool = True,
    need_latest_sent: bool = True,
    sent_batch_size: int = 256,
    sent_max_length: int = 96,
) -> PrecomputedCohortFeatures:
    """Precompute row-independent raw patient representations on ``device``."""
    dev = torch.device(device)
    x_struct = torch.from_numpy(
        np.asarray([patient_summary_vector(r) for r in rows], dtype=np.float32)
    ).to(dev)
    x_np = (
        torch.from_numpy(np.asarray([nadir_peak_features(r) for r in rows], dtype=np.float32)).to(
            dev
        )
        if need_nadir_peak
        else None
    )
    code_raw = (
        torch.from_numpy(np.asarray([encounter_code_vector(r) for r in rows], dtype=np.float32)).to(
            dev
        )
        if need_icd
        else None
    )
    lex_raw = (
        torch.from_numpy(
            np.asarray(
                [note_lexical_vector(r.get("notes") or ()) for r in rows], dtype=np.float32
            )
        ).to(dev)
        if need_lex
        else None
    )
    sb_all = sb_latest = None
    if need_sent:
        if backbone is None or tokenizer is None:
            raise ValueError("backbone and tokenizer are required when need_sent=True")
        sb_all, sb_latest = encode_cohort_sentences(
            rows,
            backbone,
            tokenizer,
            dev,
            batch_size=sent_batch_size,
            max_length=sent_max_length,
            include_latest=need_latest_sent,
        )
    return PrecomputedCohortFeatures(
        x_struct=x_struct,
        x_nadir_peak=x_np,
        code_raw=code_raw,
        lex_raw=lex_raw,
        sb_all=sb_all,
        sb_latest=sb_latest,
    )


@torch.no_grad()
def assemble_option_a_features_from_precomputed(
    pre: PrecomputedCohortFeatures,
    ctx_start: int,
    ctx_end: int,
    q_start: int,
    q_end: int,
    cfg: OptionAFeatureConfig,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Assemble strictly context-fitted explicit multimodal features ``(X_ctx, X_q, names)``."""
    ctx_parts: list[torch.Tensor] = [pre.x_struct[ctx_start:ctx_end]]
    q_parts: list[torch.Tensor] = [pre.x_struct[q_start:q_end]]
    names: list[str] = [f"struct_{i}" for i in range(pre.x_struct.shape[1])]

    if cfg.include_nadir_peak:
        if pre.x_nadir_peak is None:
            raise ValueError("PrecomputedCohortFeatures missing x_nadir_peak")
        ctx_parts.append(pre.x_nadir_peak[ctx_start:ctx_end])
        q_parts.append(pre.x_nadir_peak[q_start:q_end])
        names.extend(_nadir_peak_feature_names())

    if cfg.icd_svd_dim > 0:
        if pre.code_raw is None:
            raise ValueError("PrecomputedCohortFeatures missing code_raw")
        c_raw = pre.code_raw[ctx_start:ctx_end]
        q_raw = pre.code_raw[q_start:q_end]
        mu, basis, rms = _fit_lexical_basis(c_raw, cfg.icd_svd_dim)
        z_c = ((c_raw - mu) @ basis) / rms
        z_q = ((q_raw - mu) @ basis) / rms
        ctx_parts.append(z_c)
        q_parts.append(z_q)
        names.extend(f"icd_svd_{i}" for i in range(z_c.shape[1]))

    if cfg.sent_km_k > 0:
        if pre.sb_all is None:
            raise ValueError("PrecomputedCohortFeatures missing sb_all")
        ctx_sb = slice_sentence_batch(pre.sb_all, ctx_start, ctx_end)
        q_sb = slice_sentence_batch(pre.sb_all, q_start, q_end)
        centroids, _, _ = _fit_sentence_concepts(ctx_sb, cfg.sent_km_k)
        if centroids is not None and len(centroids) > 0:
            s_c = _project_sentence_maxsim(ctx_sb, centroids)
            s_q = _project_sentence_maxsim(q_sb, centroids)
            ctx_parts.append(s_c)
            q_parts.append(s_q)
            names.extend(f"sent_all_km_{i}" for i in range(s_c.shape[1]))
            if cfg.include_latest_sent_km:
                if pre.sb_latest is None:
                    raise ValueError("PrecomputedCohortFeatures missing sb_latest")
                ctx_lat = slice_sentence_batch(pre.sb_latest, ctx_start, ctx_end)
                q_lat = slice_sentence_batch(pre.sb_latest, q_start, q_end)
                sl_c = _project_sentence_maxsim(ctx_lat, centroids)
                sl_q = _project_sentence_maxsim(q_lat, centroids)
                ctx_parts.append(sl_c)
                q_parts.append(sl_q)
                names.extend(f"sent_latest_km_{i}" for i in range(sl_c.shape[1]))

    if cfg.lex_svd_dim > 0:
        if pre.lex_raw is None:
            raise ValueError("PrecomputedCohortFeatures missing lex_raw")
        c_raw = pre.lex_raw[ctx_start:ctx_end]
        q_raw = pre.lex_raw[q_start:q_end]
        mu, basis, rms = _fit_lexical_basis(c_raw, cfg.lex_svd_dim)
        z_c = ((c_raw - mu) @ basis) / rms
        z_q = ((q_raw - mu) @ basis) / rms
        ctx_parts.append(z_c)
        q_parts.append(z_q)
        names.extend(f"lex_svd_{i}" for i in range(z_c.shape[1]))

    X_ctx = torch.cat(ctx_parts, dim=1)
    X_q = torch.cat(q_parts, dim=1)
    return X_ctx, X_q, names


@torch.no_grad()
def extract_option_a_features(
    context_rows: list[dict],
    query_rows: list[dict],
    backbone,
    tokenizer,
    device: str | torch.device,
    cfg: OptionAFeatureConfig = OptionAFeatureConfig(),
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Extract strictly context-fitted explicit multimodal features for ``(context_rows, query_rows)``."""
    all_rows = list(context_rows) + list(query_rows)
    n_ctx = len(context_rows)
    n_tot = len(all_rows)
    pre = precompute_cohort_raw_features(
        all_rows,
        device,
        backbone=backbone,
        tokenizer=tokenizer,
        need_nadir_peak=cfg.include_nadir_peak,
        need_icd=cfg.icd_svd_dim > 0,
        need_lex=cfg.lex_svd_dim > 0,
        need_sent=cfg.sent_km_k > 0,
        need_latest_sent=cfg.sent_km_k > 0 and cfg.include_latest_sent_km,
        sent_batch_size=cfg.sent_batch_size,
        sent_max_length=cfg.sent_max_length,
    )
    return assemble_option_a_features_from_precomputed(pre, 0, n_ctx, n_ctx, n_tot, cfg)


# ==============================================================================
# 2. TRAINABLE LIGHTWEIGHT STRUCTURED TEMPORAL ENCODER (NO BERT BACKPROP!)
# ==============================================================================
@dataclass(frozen=True)
class OptionBConfig:
    """Configuration for the Option B hybrid multimodal tabular PFN model."""

    learned_temporal_dim: int = 24
    event_width: int = 128
    event_layers: int = 2
    event_heads: int = 4
    temporal_chunk_size: int = 1024
    always_screen_learned_cols: bool = True

    def __post_init__(self):
        if self.learned_temporal_dim < 0:
            raise ValueError("learned_temporal_dim must be non-negative")
        if self.learned_temporal_dim > 0:
            if min(self.event_width, self.event_layers, self.event_heads, self.temporal_chunk_size) < 1:
                raise ValueError("Invalid StructuredTemporalEncoder dimensions")
            if self.event_width % self.event_heads != 0:
                raise ValueError("event_width must be divisible by event_heads")


class StructuredTemporalEncoder(nn.Module):
    """Lightweight trainable cross-modal encoder over structured longitudinal event streams.

    Encodes timestamped ``labs``, ``vitals``, ``encounters``, and ``pros`` via ``EventTransformer``
    + ``CrossModalTimeline`` and projects to ``learned_temporal_dim`` strictly finite columns that
    append cleanly alongside unmixed explicit Option-A columns.
    """

    def __init__(self, cfg: OptionBConfig):
        super().__init__()
        self.cfg = cfg
        width = cfg.event_width
        ev_cfg = SimpleNamespace(
            event_width=width,
            event_layers=cfg.event_layers,
            heads=cfg.event_heads,
        )

        self.labs = nn.Linear(LAB_INPUT_SIZE, width)
        self.extra_labs = nn.Embedding(EXTRA_LAB_BUCKETS, width)
        self.extra_lab_scale = nn.Linear(1, width)
        self.lab_summary = _SummaryProjector(LAB_SUMMARY_SIZE, width)

        self.vitals = nn.Linear(VITAL_INPUT_SIZE, width)
        self.vital_summary = _SummaryProjector(VITAL_SUMMARY_SIZE, width)

        self.codes = nn.EmbeddingBag(CODE_BUCKETS, width, mode="mean")
        self.code_count = nn.Linear(1, width)
        self.encounter_summary = _SummaryProjector(ENCOUNTER_SUMMARY_SIZE, width)

        self.pros = nn.Linear(PRO_INPUT_SIZE, width)
        self.extra_pros = nn.Embedding(EXTRA_PRO_BUCKETS, width)
        self.extra_pro_scale = nn.Linear(1, width)
        self.pro_summary = _SummaryProjector(PRO_SUMMARY_SIZE, width)

        self.streams = nn.ModuleDict(
            {name: EventTransformer(ev_cfg, num_queries=1) for name in STRUCTURED_STREAMS}
        )
        self.cross_modal = CrossModalTimeline(ev_cfg, len(STRUCTURED_STREAMS))

        in_dim = (len(STRUCTURED_STREAMS) + 1) * width
        self.out_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, width),
            nn.GELU(),
            nn.Linear(width, cfg.learned_temporal_dim),
            nn.LayerNorm(cfg.learned_temporal_dim),
        )

    def _extra_bag(self, extra_lists, embedding, value_proj):
        device = embedding.weight.device
        dtype = embedding.weight.dtype
        out = torch.zeros(len(extra_lists), embedding.embedding_dim, device=device, dtype=dtype)
        nonempty_rows = [i for i, pairs in enumerate(extra_lists) if pairs]
        if not nonempty_rows:
            return out
        flat_ids = []
        flat_vals = []
        counts = []
        for row_idx in nonempty_rows:
            pairs = extra_lists[row_idx]
            counts.append(len(pairs))
            for bucket, val in pairs:
                flat_ids.append(bucket)
                flat_vals.append(val)
        ids = torch.tensor(flat_ids, device=device, dtype=torch.long)
        vals = torch.tensor(flat_vals, device=device, dtype=dtype).unsqueeze(-1)
        item_vecs = embedding(ids) + value_proj(vals)
        for row_idx, chunk in zip(nonempty_rows, item_vecs.split(counts), strict=True):
            out[row_idx] = chunk.mean(dim=0)
        return out

    def _values(self, name: str, events: list[dict]) -> torch.Tensor:
        if not events:
            return self.labs.weight.new_empty(0, self.cfg.event_width)
        if name == "encounters":
            codes = [encounter_codes(e) for e in events]
            device = self.codes.weight.device
            ids = torch.tensor([i for bag in codes for i in bag], device=device, dtype=torch.long)
            offsets = torch.tensor([0] + [len(bag) for bag in codes[:-1]], device=device).cumsum(0)
            counts = self.code_count.weight.new_tensor([[math.log1p(len(bag))] for bag in codes])
            return self.codes(ids, offsets) + self.code_count(counts)
        if name == "labs":
            parsed = [lab_features(e, strict=False) for e in events]
            dense = self.labs(self.labs.weight.new_tensor([v for v, _ in parsed]))
            extra = self._extra_bag([x for _, x in parsed], self.extra_labs, self.extra_lab_scale)
            return dense + extra
        if name == "pros":
            parsed = [pro_features(e, strict=False) for e in events]
            dense = self.pros(self.pros.weight.new_tensor([v for v, _ in parsed]))
            extra = self._extra_bag([x for _, x in parsed], self.extra_pros, self.extra_pro_scale)
            return dense + extra
        return self.vitals(self.vitals.weight.new_tensor([vital_vector(e) for e in events]))

    def _stream_trajectory_residual(self, name: str, streams: list[list[dict]]) -> torch.Tensor:
        device = self.labs.weight.device
        dtype = self.labs.weight.dtype
        if name == "labs":
            vec = torch.tensor([lab_stream_summary(s) for s in streams], device=device, dtype=dtype)
            return self.lab_summary(vec).unsqueeze(1)
        if name == "vitals":
            vec = torch.tensor(
                [vital_stream_summary(s) for s in streams], device=device, dtype=dtype
            )
            return self.vital_summary(vec).unsqueeze(1)
        if name == "encounters":
            vec = torch.tensor(
                [encounter_stream_summary(s) for s in streams], device=device, dtype=dtype
            )
            return self.encounter_summary(vec).unsqueeze(1)
        vec = torch.tensor([pro_stream_summary(s) for s in streams], device=device, dtype=dtype)
        return self.pro_summary(vec).unsqueeze(1)

    def _encode_chunk(self, rows: list[dict]) -> torch.Tensor:
        n_patients = len(rows)
        stream_summaries = []
        per_patient_tokens: list[list[torch.Tensor]] = [[] for _ in range(n_patients)]
        per_patient_days: list[list[float]] = [[] for _ in range(n_patients)]

        for modality_idx, (name, encoder) in enumerate(self.streams.items()):
            streams = [row.get(name) or [] for row in rows]
            lengths = [len(s) for s in streams]
            times = [t for stream in streams for t in event_times(stream)]
            flat_events = [event for stream in streams for event in stream]
            values = self._values(name, flat_events)
            encoded = encoder(values, times, lengths)
            traj = self._stream_trajectory_residual(name, streams)
            stream_summaries.append(encoded + traj)

            if flat_events:
                mod_vec = self.cross_modal.modality.weight[modality_idx]
                time_vec = self.cross_modal.clock(values.new_tensor(times).reshape(-1, 3))
                aligned = values + time_vec + mod_vec
                for p_idx, (chunk, stream) in enumerate(
                    zip(aligned.split(lengths), streams, strict=True)
                ):
                    if len(stream):
                        per_patient_tokens[p_idx].append(chunk)
                        per_patient_days[p_idx].extend(float(e["day_from_index"]) for e in stream)

        fused_inputs = [
            (
                torch.cat(chunks, dim=0)
                if chunks
                else self.labs.weight.new_empty(0, self.cfg.event_width),
                days,
            )
            for chunks, days in zip(per_patient_tokens, per_patient_days, strict=True)
        ]
        fused = self.cross_modal(fused_inputs)  # (N, 1, D)
        slots = torch.cat([*stream_summaries, fused], dim=1)  # (N, 5, D)
        return self.out_proj(slots.reshape(n_patients, -1)).float()

    def forward(self, rows: list[dict]) -> torch.Tensor:
        if not rows:
            return self.labs.weight.new_empty((0, self.cfg.learned_temporal_dim), dtype=torch.float32)
        chunk = self.cfg.temporal_chunk_size
        if len(rows) <= chunk:
            return self._encode_chunk(rows)
        outputs = []
        for start in range(0, len(rows), chunk):
            sub = rows[start : start + chunk]
            if self.training and torch.is_grad_enabled():
                # Dummy requires_grad tensor satisfies PyTorch checkpoint when inputs are Python dicts.
                dummy = self.labs.weight.new_zeros(1, requires_grad=True)
                out = checkpoint(
                    lambda _d, _sub=sub: self._encode_chunk(_sub),
                    dummy,
                    use_reentrant=False,
                )
            else:
                out = self._encode_chunk(sub)
            outputs.append(out)
        return torch.cat(outputs, dim=0)

    def load_from_longitudinal_checkpoint(self, run_dir_or_ckpt: str | Path) -> dict:
        """Warm-start structured stream encoders cleanly from a ``CausalStudyModel`` checkpoint."""
        path = Path(run_dir_or_ckpt)
        if path.is_dir():
            best = path / "best_checkpoint.pt"
            path = best if best.is_file() else (path / "checkpoint.pt")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        src = saved.get("weights", saved)
        dst = self.state_dict()
        loaded = 0
        skipped = 0
        mismatched: list[str] = []

        for k, v in dst.items():
            src_key = f"patient.{k}"
            if src_key in src and tuple(src[src_key].shape) == tuple(v.shape):
                dst[k] = src[src_key].to(dtype=v.dtype).clone()
                loaded += 1
            elif k == "cross_modal.modality.weight":
                # CausalStudyModel cross_modal has modality 0 = notes, 1..4 = labs/vitals/encounters/pros
                if (
                    src_key in src
                    and src[src_key].shape[0] == v.shape[0] + 1
                    and src[src_key].shape[1] == v.shape[1]
                ):
                    dst[k] = src[src_key][1:].to(dtype=v.dtype).clone()
                    loaded += 1
                else:
                    skipped += 1
                    mismatched.append(k)
            else:
                skipped += 1
                if not k.startswith("out_proj."):
                    mismatched.append(k)

        if mismatched:
            raise ValueError(
                f"Shape/key mismatch loading {path} ({len(mismatched)} mismatched encoder tensors, "
                f"e.g. {mismatched[:3]}); verify event_width/event_layers/event_heads match"
            )
        self.load_state_dict(dst, strict=True)
        return {
            "source": str(path),
            "source_step": int(saved.get("step", -1)) if isinstance(saved, dict) else -1,
            "loaded_tensors": loaded,
            "skipped_tensors": skipped,
        }


# ==============================================================================
# 3. DIFFERENTIABLE HYBRID STANDARDIZATION & PAIR HELPERS (ZERO EDITS TO oci/tabular_pfn)
# ==============================================================================
def _standardize_learned_context_diff(Z_ctx: torch.Tensor):
    """Differentiable column-wise standardization for strictly finite learned features."""
    Z_f = Z_ctx.float()
    z_mean = Z_f.mean(dim=0, keepdim=True)
    z_diff = Z_f - z_mean
    z_var = z_diff.square().mean(dim=0, keepdim=True)
    valid_z = z_var.squeeze(0).detach() > 1e-10
    z_sd = z_var.clamp_min(1e-8).sqrt()  # exact floor 1e-4, finite zero-slope below floor
    Z_s = ((Z_f - z_mean) / z_sd).clamp(-6.0, 6.0) * valid_z.float().unsqueeze(0)

    Z2_raw = Z_s * Z_s - 1.0
    z2_mu = Z2_raw.mean(dim=0, keepdim=True)
    Z2_c = Z2_raw - z2_mu
    z2_var = Z2_c.square().mean(dim=0, keepdim=True)
    z2_sd = z2_var.clamp_min(1e-8).sqrt()
    Z2_s = (Z2_c / z2_sd).clamp(-6.0, 6.0) * valid_z.float().unsqueeze(0)

    M_z = Z_s.new_zeros(Z_s.shape)
    return Z_s, Z2_s, M_z, z_mean, z_sd, z2_mu, z2_sd, valid_z


def _apply_learned_standardize_diff(
    Z_q: torch.Tensor,
    z_mean: torch.Tensor,
    z_sd: torch.Tensor,
    z2_mu: torch.Tensor,
    z2_sd: torch.Tensor,
    valid_z: torch.Tensor,
):
    Z_f = Z_q.float()
    Z_s = ((Z_f - z_mean) / z_sd).clamp(-6.0, 6.0) * valid_z.float().unsqueeze(0)
    Z2_raw = Z_s * Z_s - 1.0
    Z2_s = ((Z2_raw - z2_mu) / z2_sd).clamp(-6.0, 6.0) * valid_z.float().unsqueeze(0)
    M_z = Z_s.new_zeros(Z_s.shape)
    return Z_s, Z2_s, M_z


def _compute_learned_v3_basis_context_diff(Z_s: torch.Tensor, valid_z: torch.Tensor):
    zsat_raw = torch.tanh(1.5 * Z_s)
    zsat_mean = zsat_raw.mean(dim=0, keepdim=True)
    zsat_c = zsat_raw - zsat_mean
    zsat_var = zsat_c.square().mean(dim=0, keepdim=True)
    zsat_sd = zsat_var.clamp_min(1e-8).sqrt()
    Z_sat = (zsat_c / zsat_sd).clamp(-6.0, 6.0) * valid_z.float().unsqueeze(0)

    d_z = Z_s.shape[1]
    Mc_z = Z_s.new_zeros(Z_s.shape)
    miss_mean_z = Z_s.new_zeros((1, d_z))
    miss_sd_z = Z_s.new_full((1, d_z), 1e-3)
    miss_weight_z = Z_s.new_zeros((1, d_z))
    return Z_sat, Mc_z, zsat_mean, zsat_sd, miss_mean_z, miss_sd_z, miss_weight_z


def _apply_learned_v3_basis_diff(
    Z_s: torch.Tensor,
    valid_z: torch.Tensor,
    zsat_mean: torch.Tensor,
    zsat_sd: torch.Tensor,
):
    zsat_raw = torch.tanh(1.5 * Z_s)
    Z_sat = ((zsat_raw - zsat_mean) / zsat_sd).clamp(-6.0, 6.0) * valid_z.float().unsqueeze(0)
    Mc_z = Z_s.new_zeros(Z_s.shape)
    return Z_sat, Mc_z


def _build_pairs_context_diff(
    X_s: torch.Tensor,
    valid_col: torch.Tensor,
    col_a: torch.Tensor,
    col_b: torch.Tensor,
):
    """Differentiable counterpart to ``oci.tabular_pfn.model._build_pairs_context``."""
    raw = X_s[:, col_a] * X_s[:, col_b]
    p_mean = raw.mean(dim=0, keepdim=True)
    p_c = raw - p_mean
    p_var = p_c.square().mean(dim=0, keepdim=True)
    p_sd = p_var.clamp_min(1e-6).sqrt()  # exact floor 1e-3
    p_weight = (p_var / (p_var + 2e-3)).clamp_min(1e-12).sqrt() * (p_var.detach() > 0).float()
    p_valid = (valid_col[col_a] & valid_col[col_b]).float()
    P_s = ((p_c / p_sd) * p_weight).clamp(-6.0, 6.0) * p_valid.unsqueeze(0)
    return P_s, p_mean, p_sd, p_weight, p_valid


def _apply_pairs_diff(
    X_s: torch.Tensor,
    col_a: torch.Tensor,
    col_b: torch.Tensor,
    p_mean: torch.Tensor,
    p_sd: torch.Tensor,
    p_weight: torch.Tensor,
    p_valid: torch.Tensor,
) -> torch.Tensor:
    raw = X_s[:, col_a] * X_s[:, col_b]
    return (((raw - p_mean) / p_sd) * p_weight).clamp(-6.0, 6.0) * p_valid.unsqueeze(0)


# ==============================================================================
# 4. OPTION B HYBRID MODEL (FROZEN TINYONCBERT + TEMPORAL ENCODER + TABULAR V4)
# ==============================================================================
class MultimodalTabularPFN(nn.Module):
    """Option B (v6) hybrid multimodal CausalPFN with frozen clinical text encoder."""

    def __init__(
        self,
        backbone,
        tokenizer,
        tabular_head: SparseTabularCausalPFN,
        option_b_cfg: OptionBConfig = OptionBConfig(),
        feature_cfg: OptionAFeatureConfig = OptionAFeatureConfig(),
        *,
        backbone_path: str | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.backbone_path = backbone_path
        self.option_b_cfg = option_b_cfg
        self.feature_cfg = feature_cfg
        self.tabular_head = tabular_head
        self.schema_version = 4

        # Permanently freeze TinyOncBERT backbone (zero grads, permanent eval mode)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad_(False)

        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                raise ValueError("Tokenizer must define pad_token or eos_token")

        self.temporal_encoder: StructuredTemporalEncoder | None = (
            StructuredTemporalEncoder(option_b_cfg)
            if option_b_cfg.learned_temporal_dim > 0
            else None
        )

    @property
    def centers(self) -> torch.Tensor:
        return self.tabular_head.centers

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep clinical language backbone strictly frozen in eval mode at all times.
        self.backbone.eval()
        return self

    def risk_stats(self, logits: torch.Tensor, scale: ArmScale):
        return self.tabular_head.risk_stats(logits, scale)

    def risks(self, logits: torch.Tensor, scale: ArmScale):
        return self.tabular_head.risks(logits, scale)

    def forward_tabular(
        self,
        X_ctx: torch.Tensor,
        t_ctx: torch.Tensor,
        y_ctx: torch.Tensor,
        X_q: torch.Tensor,
        *,
        M_ctx: torch.Tensor | None = None,
        M_q: torch.Tensor | None = None,
    ):
        """Pure tabular forward pass directly through ``self.tabular_head``."""
        return self.tabular_head(X_ctx, t_ctx, y_ctx, X_q, M_ctx=M_ctx, M_q=M_q)

    def _condition_and_query_hybrid(
        self,
        X_exp_ctx: torch.Tensor,
        Z_ctx: torch.Tensor,
        t_ctx: torch.Tensor,
        y_ctx: torch.Tensor,
        X_exp_q: torch.Tensor,
        Z_q: torch.Tensor,
        *,
        M_exp_ctx: torch.Tensor | None = None,
        M_exp_q: torch.Tensor | None = None,
        always_screen_learned_cols: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, ArmScale, TabularMemory]:
        """Exact differentiable ``SparseTabularCausalPFN`` forward preserving gradients into ``Z``.

        Explicit features ``X_exp`` (which may contain genuine NaNs in Nadir/Peak) are standardized
        under ``torch.no_grad()`` identically to ``SparseTabularCausalPFN._standardize_context``,
        while finite learned temporal columns ``Z`` are standardized differentiably and appended.
        """
        head = self.tabular_head
        cfg = head.cfg
        v3 = bool(cfg.use_v3_solver_basis)
        v4 = bool(cfg.use_v4_interactions)
        v5 = bool(cfg.use_v5_epistasis)
        keep_learned = (
            self.option_b_cfg.always_screen_learned_cols
            if always_screen_learned_cols is None
            else bool(always_screen_learned_cols)
        )
        device_type = X_exp_ctx.device.type

        with torch.amp.autocast(device_type=device_type, enabled=False):
            t_f = t_ctx.float()
            y_f = y_ctx.float()
            n_ctx, p_exp = X_exp_ctx.shape
            d_l = Z_ctx.shape[1]
            p = p_exp + d_l
            scale = ArmScale.from_context(t_f, y_f)

            # 1. Context standardization: explicit columns (no_grad, NaN-safe) + learned columns (diff)
            Xe_s, Xe2_s, Me_s, ce_mu, ce_sd, xe2_mu, xe2_sd, valid_e = _standardize_context(
                X_exp_ctx, M_exp_ctx
            )
            Zl_s, Zl2_s, Ml_s, cl_mu, cl_sd, zl2_mu, zl2_sd, valid_l = (
                _standardize_learned_context_diff(Z_ctx)
            )
            X_s = torch.cat([Xe_s, Zl_s], dim=1)
            X2_s = torch.cat([Xe2_s, Zl2_s], dim=1)
            M_s = torch.cat([Me_s, Ml_s], dim=1)
            col_mean = torch.cat([ce_mu, cl_mu], dim=1)
            col_sd = torch.cat([ce_sd, cl_sd], dim=1)
            x2_mean = torch.cat([xe2_mu, zl2_mu], dim=1)
            x2_sd = torch.cat([xe2_sd, zl2_sd], dim=1)
            valid_col = torch.cat([valid_e, valid_l], dim=0)

            X_sat = M_c = xsat_mean = xsat_sd = miss_mean = miss_sd = miss_weight = None
            if v3:
                obs_exp_ctx = torch.isfinite(X_exp_ctx.float()).float()
                Xsat_e, Mc_e, xs_mu_e, xs_sd_e, mm_e, ms_e, mw_e = _compute_v3_basis_context(
                    Xe_s, Me_s, obs_exp_ctx, valid_e
                )
                Xsat_l, Mc_l, xs_mu_l, xs_sd_l, mm_l, ms_l, mw_l = (
                    _compute_learned_v3_basis_context_diff(Zl_s, valid_l)
                )
                X_sat = torch.cat([Xsat_e, Xsat_l], dim=1)
                M_c = torch.cat([Mc_e, Mc_l], dim=1)
                xsat_mean = torch.cat([xs_mu_e, xs_mu_l], dim=1)
                xsat_sd = torch.cat([xs_sd_e, xs_sd_l], dim=1)
                miss_mean = torch.cat([mm_e, mm_l], dim=1)
                miss_sd = torch.cat([ms_e, ms_l], dim=1)
                miss_weight = torch.cat([mw_e, mw_l], dim=1)

            col_stats, raw_nuis_prio = _column_sufficient_stats(
                X_s, X2_s, M_s, t_f, y_f, valid_col, X_sat=X_sat, M_c=M_c
            )

            # 2. O(P) Round-1 Column Role Encoder & Nuisance Screening
            h_col = head.col_in(col_stats)
            for blk in head.col_global_blocks:
                h_col = blk(h_col)
            nuis_screen_logits = head.nuis_screen_head(h_col).squeeze(-1)
            nuis_score = raw_nuis_prio + 1.5 * nuis_screen_logits
            nuis_score = torch.where(valid_col, nuis_score, nuis_score.new_full((), -1e5))

            if keep_learned and d_l > 0:
                k1_exp = min(p_exp, cfg.top_k_nuis)
                exp_idx = torch.topk(nuis_score[:p_exp], k1_exp, largest=True).indices
                learned_idx = torch.arange(p_exp, p, device=X_s.device, dtype=torch.long)
                nuis_idx = torch.cat([exp_idx, learned_idx], dim=0)
            else:
                k1 = min(p, cfg.top_k_nuis)
                nuis_idx = torch.topk(nuis_score, k1, largest=True).indices
            k1 = len(nuis_idx)

            # 3. Correlation-Biased Column Attention on Selected Nuisance Columns -> ARD Gates
            h_nuis = h_col[nuis_idx]
            X_nuis = X_s[:, nuis_idx]
            X2_nuis = X2_s[:, nuis_idx]
            corr_nuis = (
                head._pairwise_corr_features(X_nuis, M_s[:, nuis_idx])
                if v3
                else head._empirical_corr(X_nuis)
            )
            for layer in head.nuis_col_layers:
                h_nuis = layer(h_nuis, corr_nuis)
            nuis_gates = torch.sigmoid(head.nuis_gate_head(h_nuis)).clamp(1e-3, 1.0)
            v_nuis = valid_col[nuis_idx].float()

            g_e_sat = g_e_miss = g_er_sat = g_er_miss = g_m_sat = g_m_miss = None
            if v3:
                if cfg.decouple_rlearner_propensity:
                    (
                        g_e_lin, g_e_nl, g_e_sat, g_e_miss,
                        g_er_lin, g_er_nl, g_er_sat, g_er_miss,
                        g_m_lin, g_m_nl, g_m_sat, g_m_miss,
                    ) = nuis_gates.unbind(dim=-1)
                    g_er_lin = g_er_lin * v_nuis
                    g_er_nl = g_er_nl * v_nuis
                    g_er_sat = g_er_sat * v_nuis
                    g_er_miss = g_er_miss * v_nuis
                else:
                    (
                        g_e_lin, g_e_nl, g_e_sat, g_e_miss,
                        g_m_lin, g_m_nl, g_m_sat, g_m_miss,
                    ) = nuis_gates.unbind(dim=-1)
                g_e_lin = g_e_lin * v_nuis
                g_e_nl = g_e_nl * v_nuis
                g_e_sat = g_e_sat * v_nuis
                g_e_miss = g_e_miss * v_nuis
                g_m_lin = g_m_lin * v_nuis
                g_m_nl = g_m_nl * v_nuis
                g_m_sat = g_m_sat * v_nuis
                g_m_miss = g_m_miss * v_nuis
            else:
                if cfg.decouple_rlearner_propensity:
                    g_e_lin, g_e_nl, g_er_lin, g_er_nl, g_m_lin, g_m_nl = nuis_gates.unbind(dim=-1)
                    g_er_lin = g_er_lin * v_nuis
                    g_er_nl = g_er_nl * v_nuis
                else:
                    g_e_lin, g_e_nl, g_m_lin, g_m_nl = nuis_gates.unbind(dim=-1)
                g_e_lin = g_e_lin * v_nuis
                g_e_nl = g_e_nl * v_nuis
                g_m_lin = g_m_lin * v_nuis
                g_m_nl = g_m_nl * v_nuis

            # Round-1 symmetric pairwise interaction basis (differentiable wrt X_s)
            nuis_pair_a = nuis_pair_b = nuis_pair_mean = nuis_pair_sd = nuis_pair_weight = nuis_pair_valid = None
            g_pe = g_per = g_pm = P_nuis = None
            g_m_r1_max = g_er_r1_max = None
            if v4:
                g_m_r1_max = torch.stack([g_m_lin, g_m_nl, g_m_sat, g_m_miss], dim=-1).max(dim=-1).values
                g_er_r1_max = (
                    torch.stack([g_er_lin, g_er_nl, g_er_sat, g_er_miss], dim=-1).max(dim=-1).values
                    if cfg.decouple_rlearner_propensity
                    else torch.stack([g_e_lin, g_e_nl, g_e_sat, g_e_miss], dim=-1).max(dim=-1).values
                )
                if k1 >= 2:
                    r1_pair_prio = (
                        g_m_r1_max.detach()
                        + 0.35 * g_er_r1_max.detach()
                        + 1e-3 * raw_nuis_prio[nuis_idx].clamp(-10.0, 10.0)
                    )
                    r1_pair_prio = torch.where(
                        valid_col[nuis_idx], r1_pair_prio, r1_pair_prio.new_full((), -1e5)
                    )
                    k_pn = min(k1, cfg.pair_top_k)
                    pn_loc = torch.topk(r1_pair_prio, k_pn, largest=True).indices
                    pa_i, pb_i = torch.triu_indices(k_pn, k_pn, offset=1, device=X_s.device)
                    pa_nuis = pn_loc[pa_i]
                    pb_nuis = pn_loc[pb_i]
                    nuis_pair_a = nuis_idx[pa_nuis]
                    nuis_pair_b = nuis_idx[pb_nuis]
                    P_nuis, nuis_pair_mean, nuis_pair_sd, nuis_pair_weight, nuis_pair_valid = (
                        _build_pairs_context_diff(X_s, valid_col, nuis_pair_a, nuis_pair_b)
                    )
                    corr_ab_n = corr_nuis[pa_nuis, pb_nuis, 0]
                    nuis_p_stats = head._nuis_pair_stats(
                        P_nuis, t_f, y_f, corr_ab_n, nuis_pair_valid
                    )
                    ha = h_nuis[pa_nuis]
                    hb = h_nuis[pb_nuis]
                    nuis_p_in = torch.cat(
                        [ha + hb, ((ha - hb).square() + 1e-6).sqrt(), nuis_p_stats], dim=-1
                    )
                    pair_n_gates = torch.sigmoid(head.nuis_pair_gate_head(nuis_p_in)).clamp(1e-3, 1.0)
                    if cfg.decouple_rlearner_propensity:
                        g_pe, g_per, g_pm = pair_n_gates.unbind(dim=-1)
                        g_per = g_per * nuis_pair_valid
                    else:
                        g_pe, g_pm = pair_n_gates.unbind(dim=-1)
                        g_per = g_pe * nuis_pair_valid
                    g_pe = g_pe * nuis_pair_valid
                    g_pm = g_pm * nuis_pair_valid

            # 4. Differentiable ARD Closed-Form LOO-CV Nuisances
            ones = torch.ones((n_ctx, 1), device=X_s.device, dtype=torch.float32)
            lam_hi = 5e5 if v3 else 1e5
            if v3:
                Xsat_nuis = X_sat[:, nuis_idx]
                Mc_nuis = M_c[:, nuis_idx]
                e_parts = [
                    ones,
                    X_nuis * g_e_lin[None, :],
                    X2_nuis * g_e_nl[None, :],
                    Xsat_nuis * g_e_sat[None, :],
                    Mc_nuis * g_e_miss[None, :],
                ]
                m_parts = [
                    ones,
                    X_nuis * g_m_lin[None, :],
                    X2_nuis * g_m_nl[None, :],
                    Xsat_nuis * g_m_sat[None, :],
                    Mc_nuis * g_m_miss[None, :],
                ]
                if P_nuis is not None:
                    e_parts.append(P_nuis * g_pe[None, :])
                    m_parts.append(P_nuis * g_pm[None, :])
                A_e = torch.cat(e_parts, dim=1)
                A_m = torch.cat(m_parts, dim=1)
            else:
                A_e = torch.cat([ones, X_nuis * g_e_lin[None, :], X2_nuis * g_e_nl[None, :]], dim=1)
                A_m = torch.cat([ones, X_nuis * g_m_lin[None, :], X2_nuis * g_m_nl[None, :]], dim=1)
            cv_temp = F.softplus(head.log_cv_temp.float()) + 0.1

            lam_e = head.log_lam_e.float().exp().clamp(0.05, lam_hi)
            prop_fit, e_loo = _fit_ard_loo_cv_ridge(A_e, t_f, lam_e, cv_temp)
            e_loo = e_loo.clamp(0.02, 0.98)

            if cfg.decouple_rlearner_propensity:
                if v3:
                    er_parts = [
                        ones,
                        X_nuis * g_er_lin[None, :],
                        X2_nuis * g_er_nl[None, :],
                        Xsat_nuis * g_er_sat[None, :],
                        Mc_nuis * g_er_miss[None, :],
                    ]
                    if P_nuis is not None:
                        er_parts.append(P_nuis * g_per[None, :])
                    A_er = torch.cat(er_parts, dim=1)
                else:
                    A_er = torch.cat(
                        [ones, X_nuis * g_er_lin[None, :], X2_nuis * g_er_nl[None, :]], dim=1
                    )
                lam_er = head.log_lam_er.float().exp().clamp(0.05, lam_hi)
                r_prop_fit, e_r_loo = _fit_ard_loo_cv_ridge(A_er, t_f, lam_er, cv_temp)
                e_r_loo = e_r_loo.clamp(0.02, 0.98)
            else:
                g_er_lin, g_er_nl = g_e_lin, g_e_nl
                g_er_sat, g_er_miss = g_e_sat, g_e_miss
                r_prop_fit, e_r_loo = prop_fit, e_loo

            lam_m = head.log_lam_m.float().exp().clamp(0.05, lam_hi)
            out_fit, m_loo = _fit_ard_loo_cv_ridge(A_m, y_f, lam_m, cv_temp)

            # 5. Round-2 ALL-P Adjusted Robinson Interaction Scan
            t_res = t_f - e_r_loo
            y_res = y_f - m_loo
            rob_feats, raw_mod_prio = _robinson_all_p_stats(
                X_s, X2_s, t_res, y_res, valid_col, X_sat=X_sat, M_c=M_c
            )
            epi_cand_a = epi_cand_b = None
            if v4:
                r_ref = min(8, k1)
                ref_score = (
                    g_m_r1_max.detach()
                    + 0.5 * g_er_r1_max.detach()
                    + 1e-3 * raw_nuis_prio[nuis_idx].clamp(-10.0, 10.0)
                )
                ref_score = torch.where(
                    valid_col[nuis_idx], ref_score, ref_score.new_full((), -1e5)
                )
                ref_cols = nuis_idx[torch.topk(ref_score, r_ref, largest=True).indices]
                cm_feats, cm_prio = _robinson_confounder_mod_stats(
                    X_s, t_res, y_res, valid_col, ref_cols
                )
                rob_feats = torch.cat([rob_feats, cm_feats], dim=-1)
                raw_mod_prio = torch.where(
                    valid_col, raw_mod_prio + cm_prio, raw_mod_prio.new_full((), -1e4)
                )
                if v5:
                    k_cand = (
                        cfg.epi_pair_k + cfg.pair_top_k * (cfg.pair_top_k - 1) // 2
                        if cfg.epi_pair_k > 0
                        else 0
                    )
                    epi_feats, epi_prio, epi_cand_a, epi_cand_b = _robinson_epistasis_all_pairs(
                        X_s, t_res, y_res, valid_col, top_pairs=k_cand
                    )
                    rob_feats = torch.cat([rob_feats, epi_feats], dim=-1)
                    raw_mod_prio = torch.where(
                        valid_col, raw_mod_prio + epi_prio, raw_mod_prio.new_full((), -1e4)
                    )
            h_col2 = h_col + head.rob_update(torch.cat([h_col, rob_feats], dim=-1))
            h_col2 = head.rob_global_block(h_col2)
            if v3:
                h_col2 = head.rob_global_block2(h_col2)
            role_logits = head.role_head(h_col2)

            mod_screen_logits = head.mod_screen_head(h_col2).squeeze(-1)
            mod_score = raw_mod_prio + 1.5 * mod_screen_logits
            mod_score = torch.where(valid_col, mod_score, mod_score.new_full((), -1e5))

            if keep_learned and d_l > 0:
                k2_exp = min(p_exp, cfg.top_k_cate)
                exp_cate_idx = torch.topk(mod_score[:p_exp], k2_exp, largest=True).indices
                learned_idx = torch.arange(p_exp, p, device=X_s.device, dtype=torch.long)
                cate_idx = torch.cat([exp_cate_idx, learned_idx], dim=0)
            else:
                k2 = min(p, cfg.top_k_cate)
                cate_idx = torch.topk(mod_score, k2, largest=True).indices

            # 6. Correlation-Biased Attention on Selected CATE Columns -> ARD Robinson R-Learner
            h_cate = h_col2[cate_idx]
            X_cate = X_s[:, cate_idx]
            X2_cate = X2_s[:, cate_idx]
            corr_cate = (
                head._pairwise_corr_features(X_cate, M_s[:, cate_idx])
                if v3
                else head._empirical_corr(X_cate)
            )
            for layer in head.cate_col_layers:
                h_cate = layer(h_cate, corr_cate)
            cate_gates = torch.sigmoid(head.cate_gate_head(h_cate)).clamp(1e-3, 1.0)
            v_cate = valid_col[cate_idx].float()
            g_tau_sat = g_tau_miss = None
            cate_pair_a = cate_pair_b = cate_pair_mean = cate_pair_sd = cate_pair_weight = cate_pair_valid = None
            g_ptau = P_cate = None

            full_ge = torch.zeros(p, device=X_s.device, dtype=torch.float32)
            full_gm = torch.zeros(p, device=X_s.device, dtype=torch.float32)
            full_gt = torch.zeros(p, device=X_s.device, dtype=torch.float32)
            if v3:
                g_tau_lin, g_tau_nl, g_tau_sat, g_tau_miss = cate_gates.unbind(dim=-1)
                g_tau_lin = g_tau_lin * v_cate
                g_tau_nl = g_tau_nl * v_cate
                g_tau_sat = g_tau_sat * v_cate
                g_tau_miss = g_tau_miss * v_cate
                full_ge[nuis_idx] = torch.stack(
                    [g_er_lin, g_er_nl, g_er_sat, g_er_miss], dim=-1
                ).max(dim=-1).values
                full_gm[nuis_idx] = torch.stack(
                    [g_m_lin, g_m_nl, g_m_sat, g_m_miss], dim=-1
                ).max(dim=-1).values
                full_gt[cate_idx] = torch.stack(
                    [g_tau_lin, g_tau_nl, g_tau_sat, g_tau_miss], dim=-1
                ).max(dim=-1).values
                tau_parts = [
                    ones,
                    X_cate * g_tau_lin[None, :],
                    X2_cate * g_tau_nl[None, :],
                    X_sat[:, cate_idx] * g_tau_sat[None, :],
                    M_c[:, cate_idx] * g_tau_miss[None, :],
                ]
                if v4 and p >= 2:
                    pair_tau_prio = (
                        1.5 * full_gt.detach()
                        + 0.9 * full_gm.detach()
                        + 0.5 * full_ge.detach()
                        + 1e-3 * (raw_nuis_prio.clamp(-10.0, 10.0) + raw_mod_prio.clamp(-10.0, 10.0))
                    )
                    pair_tau_prio = torch.where(
                        valid_col, pair_tau_prio, pair_tau_prio.new_full((), -1e5)
                    )
                    k_pc = min(p, cfg.pair_top_k)
                    cp_nodes = torch.topk(pair_tau_prio, k_pc, largest=True).indices
                    ca_i, cb_i = torch.triu_indices(k_pc, k_pc, offset=1, device=X_s.device)
                    cate_pair_a = cp_nodes[ca_i]
                    cate_pair_b = cp_nodes[cb_i]
                    if (
                        v5
                        and cfg.epi_pair_k > 0
                        and epi_cand_a is not None
                        and len(epi_cand_a) > 0
                    ):
                        in_cp = torch.zeros(p, device=X_s.device, dtype=torch.bool)
                        in_cp[cp_nodes] = True
                        novel = ~(in_cp[epi_cand_a] & in_cp[epi_cand_b])
                        ea = epi_cand_a[novel][: cfg.epi_pair_k]
                        eb = epi_cand_b[novel][: cfg.epi_pair_k]
                        if len(ea) > 0:
                            cate_pair_a = torch.cat([cate_pair_a, ea], dim=0)
                            cate_pair_b = torch.cat([cate_pair_b, eb], dim=0)
                    P_cate, cate_pair_mean, cate_pair_sd, cate_pair_weight, cate_pair_valid = (
                        _build_pairs_context_diff(X_s, valid_col, cate_pair_a, cate_pair_b)
                    )
                    if v5:
                        corr_ab_c = _pair_empirical_corr(X_s, cate_pair_a, cate_pair_b)
                    else:
                        corr_cp = head._empirical_corr(X_s[:, cp_nodes])
                        corr_ab_c = corr_cp[ca_i, cb_i]
                    cate_p_stats = head._cate_pair_stats(
                        P_cate, t_res, y_res, corr_ab_c, cate_pair_valid
                    )
                    h2a = h_col2[cate_pair_a]
                    h2b = h_col2[cate_pair_b]
                    cate_p_in = torch.cat(
                        [h2a + h2b, ((h2a - h2b).square() + 1e-6).sqrt(), cate_p_stats], dim=-1
                    )
                    g_ptau = (
                        torch.sigmoid(head.cate_pair_gate_head(cate_p_in)).squeeze(-1).clamp(1e-3, 1.0)
                        * cate_pair_valid
                    )
                    tau_parts.append(P_cate * g_ptau[None, :])
                A_tau = torch.cat(tau_parts, dim=1)
            else:
                g_tau_lin, g_tau_nl = cate_gates.unbind(dim=-1)
                g_tau_lin = g_tau_lin * v_cate
                g_tau_nl = g_tau_nl * v_cate
                full_ge[nuis_idx] = torch.maximum(g_er_lin, g_er_nl)
                full_gm[nuis_idx] = torch.maximum(g_m_lin, g_m_nl)
                full_gt[cate_idx] = torch.maximum(g_tau_lin, g_tau_nl)
                A_tau = torch.cat(
                    [ones, X_cate * g_tau_lin[None, :], X2_cate * g_tau_nl[None, :]], dim=1
                )
            B_ctx = t_res.unsqueeze(-1) * A_tau
            lam_tau = head.log_lam_tau.float().exp().clamp(0.05, lam_hi)
            cate_fit, _ = _fit_ard_loo_cv_ridge(B_ctx, y_res, lam_tau, cv_temp, intercept_reg=0.05)

            # 7. Select top attn_slots columns for residual Two-Way TabPFN
            if v4 and g_ptau is not None and len(g_ptau) > 0:
                pair_gt_node = torch.zeros(p, device=X_s.device, dtype=torch.float32)
                pair_gt_node.scatter_reduce_(0, cate_pair_a, g_ptau.detach(), reduce="amax", include_self=True)
                pair_gt_node.scatter_reduce_(0, cate_pair_b, g_ptau.detach(), reduce="amax", include_self=True)
                full_gt = torch.maximum(full_gt, pair_gt_node)
            attn_prio = (
                full_gm
                + 1.5 * full_gt
                + 0.5 * full_ge
                + 1e-3 * (raw_nuis_prio.clamp(-10, 10) + raw_mod_prio.clamp(-10, 10))
            )
            if v5 and v3:
                full_ge_prop = torch.zeros(p, device=X_s.device, dtype=torch.float32)
                full_ge_prop[nuis_idx] = (
                    torch.stack([g_e_lin, g_e_nl, g_e_sat, g_e_miss], dim=-1).max(dim=-1).values.detach()
                )
                attn_prio = attn_prio + 0.35 * full_ge_prop
            attn_prio = torch.where(valid_col, attn_prio, attn_prio.new_full((), -1e5))
            k_attn = min(p, cfg.attn_slots)
            attn_idx = torch.topk(attn_prio, k_attn, largest=True).indices
            attn_gate_feats = torch.stack(
                [full_ge[attn_idx], full_gm[attn_idx], full_gt[attn_idx]], dim=-1
            )
            if cfg.decouple_rlearner_propensity:
                attn_gate_feats = attn_gate_feats.detach()

            mem = TabularMemory(
                col_mean=col_mean,
                col_sd=col_sd,
                x2_mean=x2_mean,
                x2_sd=x2_sd,
                valid_col=valid_col,
                col_embs=h_col2,
                role_logits=role_logits,
                nuis_screen_logits=nuis_screen_logits,
                mod_screen_logits=mod_screen_logits,
                nuis_idx=nuis_idx,
                cate_idx=cate_idx,
                attn_idx=attn_idx,
                g_e_lin=g_e_lin,
                g_e_nl=g_e_nl,
                g_er_lin=g_er_lin,
                g_er_nl=g_er_nl,
                g_m_lin=g_m_lin,
                g_m_nl=g_m_nl,
                g_tau_lin=g_tau_lin,
                g_tau_nl=g_tau_nl,
                attn_gate_feats=attn_gate_feats,
                propensity_fit=prop_fit,
                rlearner_prop_fit=r_prop_fit,
                outcome_fit=out_fit,
                cate_fit=cate_fit,
                attn_memories=[],
                n_ctx=n_ctx,
                n_features=p,
                xsat_mean=xsat_mean,
                xsat_sd=xsat_sd,
                miss_mean=miss_mean,
                miss_sd=miss_sd,
                miss_weight=miss_weight,
                g_e_sat=g_e_sat,
                g_e_miss=g_e_miss,
                g_er_sat=g_er_sat,
                g_er_miss=g_er_miss,
                g_m_sat=g_m_sat,
                g_m_miss=g_m_miss,
                g_tau_sat=g_tau_sat,
                g_tau_miss=g_tau_miss,
                nuis_pair_a=nuis_pair_a,
                nuis_pair_b=nuis_pair_b,
                nuis_pair_mean=nuis_pair_mean,
                nuis_pair_sd=nuis_pair_sd,
                nuis_pair_weight=nuis_pair_weight,
                nuis_pair_valid=nuis_pair_valid,
                g_pe=g_pe,
                g_per=g_per,
                g_pm=g_pm,
                cate_pair_a=cate_pair_a,
                cate_pair_b=cate_pair_b,
                cate_pair_mean=cate_pair_mean,
                cate_pair_sd=cate_pair_sd,
                cate_pair_weight=cate_pair_weight,
                cate_pair_valid=cate_pair_valid,
                g_ptau=g_ptau,
            )

            ctx_slots = head._build_patient_slots(X_s, X2_s, M_s, mem, X_sat=X_sat, M_c=M_c)
            context = head._assemble_context(ctx_slots, t_f, y_f, scale)
            attn_memories = []
            use_bf16 = context.is_cuda and torch.cuda.is_bf16_supported()
            use_ckpt = head.training and ((v5 and n_ctx > 1200) or (v4 and n_ctx > 4000))
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                for i, layer in enumerate(head.study_layers):
                    if use_ckpt:
                        context = checkpoint(layer.slot_mix, context, use_reentrant=False)
                    else:
                        context = layer.slot_mix(context)
                    memory = layer.sample_norm(context).transpose(0, 1)
                    attn_memories.append(memory)
                    if i + 1 < len(head.study_layers):
                        if use_ckpt:
                            context = checkpoint(
                                layer.sample_mix, context, memory, use_reentrant=False
                            )
                        else:
                            context = layer.sample_mix(context, memory)
            mem.attn_memories = attn_memories

            # 8. Differentiable query scoring
            Xe_qs, Xe2_qs, Me_qs = _apply_standardize(
                X_exp_q, M_exp_q, ce_mu, ce_sd, xe2_mu, xe2_sd, valid_e
            )
            Zl_qs, Zl2_qs, Ml_qs = _apply_learned_standardize_diff(
                Z_q, cl_mu, cl_sd, zl2_mu, zl2_sd, valid_l
            )
            Xq_s = torch.cat([Xe_qs, Zl_qs], dim=1)
            Xq2_s = torch.cat([Xe2_qs, Zl2_qs], dim=1)
            Mq_s = torch.cat([Me_qs, Ml_qs], dim=1)

            Xq_sat = Mq_c = None
            if v3:
                obs_exp_q = torch.isfinite(X_exp_q.float()).float()
                Xsat_eq, Mc_eq = _apply_v3_basis(
                    Xe_qs, Me_qs, obs_exp_q, valid_e, xs_mu_e, xs_sd_e, mm_e, ms_e, mw_e
                )
                Xsat_lq, Mc_lq = _apply_learned_v3_basis_diff(Zl_qs, valid_l, xs_mu_l, xs_sd_l)
                Xq_sat = torch.cat([Xsat_eq, Xsat_lq], dim=1)
                Mq_c = torch.cat([Mc_eq, Mc_lq], dim=1)

            n_q = len(Xq_s)
            ones_q = torch.ones((n_q, 1), device=Xq_s.device, dtype=torch.float32)
            X_n = Xq_s[:, mem.nuis_idx]
            X2_n = Xq2_s[:, mem.nuis_idx]
            X_c = Xq_s[:, mem.cate_idx]
            X2_c = Xq2_s[:, mem.cate_idx]

            P_nuis_q = P_cate_q = None
            if mem.nuis_pair_a is not None:
                P_nuis_q = _apply_pairs_diff(
                    Xq_s,
                    mem.nuis_pair_a,
                    mem.nuis_pair_b,
                    mem.nuis_pair_mean,
                    mem.nuis_pair_sd,
                    mem.nuis_pair_weight,
                    mem.nuis_pair_valid,
                )
            if mem.cate_pair_a is not None:
                P_cate_q = _apply_pairs_diff(
                    Xq_s,
                    mem.cate_pair_a,
                    mem.cate_pair_b,
                    mem.cate_pair_mean,
                    mem.cate_pair_sd,
                    mem.cate_pair_weight,
                    mem.cate_pair_valid,
                )

            if v3:
                Xsat_n = Xq_sat[:, mem.nuis_idx]
                Mc_n = Mq_c[:, mem.nuis_idx]
                e_parts_q = [
                    ones_q,
                    X_n * mem.g_e_lin[None, :],
                    X2_n * mem.g_e_nl[None, :],
                    Xsat_n * mem.g_e_sat[None, :],
                    Mc_n * mem.g_e_miss[None, :],
                ]
                m_parts_q = [
                    ones_q,
                    X_n * mem.g_m_lin[None, :],
                    X2_n * mem.g_m_nl[None, :],
                    Xsat_n * mem.g_m_sat[None, :],
                    Mc_n * mem.g_m_miss[None, :],
                ]
                t_parts_q = [
                    ones_q,
                    X_c * mem.g_tau_lin[None, :],
                    X2_c * mem.g_tau_nl[None, :],
                    Xq_sat[:, mem.cate_idx] * mem.g_tau_sat[None, :],
                    Mq_c[:, mem.cate_idx] * mem.g_tau_miss[None, :],
                ]
                if P_nuis_q is not None:
                    e_parts_q.append(P_nuis_q * mem.g_pe[None, :])
                    m_parts_q.append(P_nuis_q * mem.g_pm[None, :])
                if P_cate_q is not None:
                    t_parts_q.append(P_cate_q * mem.g_ptau[None, :])
                A_e_q = torch.cat(e_parts_q, dim=1)
                A_m_q = torch.cat(m_parts_q, dim=1)
                A_t_q = torch.cat(t_parts_q, dim=1)
            else:
                A_e_q = torch.cat(
                    [ones_q, X_n * mem.g_e_lin[None, :], X2_n * mem.g_e_nl[None, :]], dim=1
                )
                A_m_q = torch.cat(
                    [ones_q, X_n * mem.g_m_lin[None, :], X2_n * mem.g_m_nl[None, :]], dim=1
                )
                A_t_q = torch.cat(
                    [ones_q, X_c * mem.g_tau_lin[None, :], X2_c * mem.g_tau_nl[None, :]], dim=1
                )

            e_q, h_e = _predict_ridge(mem.propensity_fit, A_e_q)
            m_q, h_m = _predict_ridge(mem.outcome_fit, A_m_q)
            tau_q, h_tau = _predict_ridge(mem.cate_fit, A_t_q)
            e_prob = e_q.clamp(0.01, 0.99)
            if cfg.decouple_rlearner_propensity:
                if v3:
                    er_parts_q = [
                        ones_q,
                        X_n * mem.g_er_lin[None, :],
                        X2_n * mem.g_er_nl[None, :],
                        Xsat_n * mem.g_er_sat[None, :],
                        Mc_n * mem.g_er_miss[None, :],
                    ]
                    if P_nuis_q is not None:
                        er_parts_q.append(P_nuis_q * mem.g_per[None, :])
                    A_er_q = torch.cat(er_parts_q, dim=1)
                else:
                    A_er_q = torch.cat(
                        [ones_q, X_n * mem.g_er_lin[None, :], X2_n * mem.g_er_nl[None, :]], dim=1
                    )
                e_r_q, _ = _predict_ridge(mem.rlearner_prop_fit, A_er_q)
                e_r_prob = e_r_q.clamp(0.01, 0.99)
            else:
                e_r_prob = e_prob
            y0_raw = (m_q - e_r_prob * tau_q).clamp(0.01, 0.99)
            y1_raw = (m_q + (1.0 - e_r_prob) * tau_q).clamp(0.01, 0.99)

            log_n = y0_raw.new_full((n_q,), math.log1p(mem.n_ctx))
            arm_feats = torch.stack(
                [y0_raw, y1_raw, m_q, e_r_prob, tau_q, h_m.sqrt(), h_tau.sqrt(), log_n], dim=-1
            )
            cal = head.arm_cal(arm_feats)
            delta_shift = cal[:, :2]
            log_sigma = cal[:, 2:]
            base_delta = scale.logit_offsets(torch.stack([y0_raw, y1_raw], dim=1))
            center_delta = (base_delta + delta_shift).clamp(head.centers[0], head.centers[-1])
            sigma = F.softplus(log_sigma + 0.5) + 0.15
            solver_bin_logits = -0.5 * (
                (head.centers[None, None, :] - center_delta[..., None]) / sigma[..., None]
            ).square()

            e_logit = e_prob.logit()
            prop_feats = torch.stack([e_prob, e_logit, h_e.sqrt(), log_n], dim=-1)
            base_prop_logit = e_logit + head.prop_cal(prop_feats).squeeze(-1)

            q_slots = head._build_patient_slots(Xq_s, Xq2_s, Mq_s, mem, X_sat=Xq_sat, M_c=Mq_c)
            arm_q = head._assemble_arm_queries(q_slots, scale)
            arm_pooled = head._attend_queries(arm_q, mem.attn_memories)
            attn_logits = head.head(arm_pooled).reshape(n_q, 2, -1)

            prop_q = head._assemble_propensity_queries(q_slots, scale)
            prop_pooled = head._attend_queries(prop_q, mem.attn_memories)
            attn_prop = head.propensity_head(prop_pooled).squeeze(-1)

            outcome_logits = solver_bin_logits + head.attn_residual_scale.float() * attn_logits
            prop_logits = base_prop_logit + head.prop_residual_scale.float() * attn_prop
            return outcome_logits, prop_logits, scale, mem

    def forward_multimodal_from_explicit(
        self,
        X_exp_ctx: torch.Tensor,
        X_exp_q: torch.Tensor,
        context_rows: list[dict],
        query_rows: list[dict],
        *,
        t_ctx: torch.Tensor | None = None,
        y_ctx: torch.Tensor | None = None,
        precision: str = "auto",
    ):
        """Run Option B given precomputed explicit Option-A features (fast for multi-preset eval)."""
        device = self.centers.device
        if t_ctx is None or y_ctx is None:
            t_ctx, y_ctx = binary_observations(context_rows, device)
        if self.temporal_encoder is None:
            return self.tabular_head(X_exp_ctx, t_ctx, y_ctx, X_exp_q)

        all_rows = list(context_rows) + list(query_rows)
        n_ctx = len(context_rows)
        with _autocast(device, precision):
            Z_all = self.temporal_encoder(all_rows)
        Z_ctx = Z_all[:n_ctx]
        Z_q = Z_all[n_ctx:]
        return self._condition_and_query_hybrid(X_exp_ctx, Z_ctx, t_ctx, y_ctx, X_exp_q, Z_q)

    def forward(
        self,
        context_rows: list[dict],
        query_rows: list[dict],
        *,
        precision: str = "auto",
    ):
        """End-to-end longitudinal multimodal forward pass (frozen TinyOncBERT + temporal + Tabular v4)."""
        device = self.centers.device
        X_exp_ctx, X_exp_q, _ = extract_option_a_features(
            context_rows,
            query_rows,
            self.backbone,
            self.tokenizer,
            device,
            self.feature_cfg,
        )
        return self.forward_multimodal_from_explicit(
            X_exp_ctx, X_exp_q, context_rows, query_rows, precision=precision
        )


# ==============================================================================
# 5. CHECKPOINTING, LOADING & ANCHOR REGULARIZATION
# ==============================================================================
def _snapshot_tabular_weights(tabular_head: SparseTabularCausalPFN) -> dict[str, torch.Tensor]:
    return {k: v.detach().float().clone() for k, v in tabular_head.named_parameters() if v.requires_grad}


def _anchor_penalty(
    tabular_head: SparseTabularCausalPFN,
    anchor_snapshot: dict[str, torch.Tensor] | None,
) -> tuple[torch.Tensor, float]:
    """Compute proximal L2-SP penalty ``0.5 * sum((w - w0)^2)`` and RMS parameter drift."""
    zero = tabular_head.centers.new_tensor(0.0)
    if not anchor_snapshot:
        return zero, 0.0
    total_sq = zero
    n_elems = 0
    for name, param in tabular_head.named_parameters():
        ref = anchor_snapshot.get(name)
        if ref is None or not param.requires_grad:
            continue
        diff = param.float() - ref
        total_sq = total_sq + diff.square().sum()
        n_elems += param.numel()
    rms = float(math.sqrt(max(float(total_sq.detach().item()) / max(n_elems, 1), 0.0)))
    return 0.5 * total_sq, rms


def save_option_b_checkpoint(
    path: Path,
    model: MultimodalTabularPFN,
    optimizer,
    scheduler,
    step: int,
    settings: dict,
    prior: PriorConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "format_version": OPTION_B_CHECKPOINT_VERSION,
        "step": int(step),
        "option_b": asdict(model.option_b_cfg),
        "feature_cfg": asdict(model.feature_cfg),
        "tabular_model": asdict(model.tabular_head.cfg),
        "prior": asdict(prior),
        "backbone_path": model.backbone_path,
        "backbone_config": model.backbone.config.to_dict(),
        "settings": settings,
        "tabular_weights": model.tabular_head.state_dict(),
        "temporal_weights": (
            model.temporal_encoder.state_dict() if model.temporal_encoder is not None else None
        ),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _load_frozen_backbone(backbone_path: str | Path, device: str | torch.device):
    tokenizer = AutoTokenizer.from_pretrained(backbone_path)
    backbone = AutoModel.from_pretrained(backbone_path).to(device).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    return backbone, tokenizer


def load_option_b_run(
    run_dir: str | Path,
    device: str = "cpu",
    *,
    prefer_best: bool = True,
    backbone_override: str | Path | None = None,
) -> tuple[MultimodalTabularPFN, dict]:
    """Load a trained Option B (v6) checkpoint cleanly for evaluation or resuming."""
    run_dir = Path(run_dir)
    ckpt_path = run_dir / "best_checkpoint.pt"
    if not prefer_best or not ckpt_path.is_file():
        ckpt_path = run_dir / "checkpoint.pt"
    saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    bpath = str(backbone_override or saved.get("backbone_path") or "")
    if not bpath:
        raise FileNotFoundError(f"Checkpoint {ckpt_path} has no valid backbone_path")
    backbone, tokenizer = _load_frozen_backbone(bpath, device)

    tab_cfg = TabularModelConfig(**saved["tabular_model"])
    tab_head = SparseTabularCausalPFN(tab_cfg).to(device)
    tab_head.load_state_dict(saved["tabular_weights"], strict=True)

    opt_b_cfg = OptionBConfig(**saved["option_b"])
    feat_cfg = OptionAFeatureConfig(**saved["feature_cfg"])
    model = MultimodalTabularPFN(
        backbone,
        tokenizer,
        tab_head,
        opt_b_cfg,
        feat_cfg,
        backbone_path=bpath,
    ).to(device)
    if model.temporal_encoder is not None and saved.get("temporal_weights") is not None:
        model.temporal_encoder.load_state_dict(saved["temporal_weights"], strict=True)
    model.eval()
    return model, saved


# ==============================================================================
# 6. UNIFIED 50/50 MIXED EPISODE PREFETCHER (MULTIMODAL + TABULAR REPLAY)
# ==============================================================================
def _mixed_episode_worker(job):
    kind = job[0]
    if kind == "multimodal":
        _, args = job
        rows, prov = _simulate_for_design(*args)
        return "multimodal", rows, prov
    if kind == "tabular":
        _, seed, n_ctx, n_q, num_features, prior_dict = job
        prior = PriorConfig(**prior_dict)
        ep = sample_tabular_study(seed, n_ctx + n_q, num_features=num_features, prior=prior)
        return "tabular", ep, {"n_ctx": n_ctx, "n_q": n_q, "p": int(ep["X"].shape[1])}
    raise ValueError(f"Unknown mixed episode kind: {kind!r}")


class _MixedEpisodePrefetcher:
    """Overlap CPU episode simulation (longitudinal + tabular) with GPU compute deterministically."""

    def __init__(self, jobs, *, num_workers: int = 0, max_inflight: int | None = None):
        self._jobs = iter(jobs)
        self._num_workers = int(num_workers)
        self._executor = None
        self._pending: deque = deque()
        if self._num_workers <= 0:
            self._max_inflight = 0
        elif max_inflight is not None:
            self._max_inflight = max(1, int(max_inflight))
        else:
            self._max_inflight = min(max(self._num_workers, 4), 8)

    def __enter__(self):
        if self._num_workers > 0:
            ctx = multiprocessing.get_context("spawn")
            self._executor = ProcessPoolExecutor(max_workers=self._num_workers, mp_context=ctx)
            self._fill()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
        self._pending.clear()

    def _fill(self):
        while len(self._pending) < self._max_inflight:
            try:
                meta, job = next(self._jobs)
            except StopIteration:
                break
            self._pending.append((meta, self._executor.submit(_mixed_episode_worker, job)))

    def next_item(self):
        if self._executor is None:
            meta, job = next(self._jobs)
            kind, payload, prov = _mixed_episode_worker(job)
            return meta, kind, payload, prov
        meta, fut = self._pending.popleft()
        self._fill()
        kind, payload, prov = fut.result()
        return meta, kind, payload, prov


# ==============================================================================
# 7. EVALUATION (32-STUDY DEV + 1,000-STUDY HELD-OUT PARAPHRASED BENCHMARK)
# ==============================================================================
def _score_option_b_episode(
    model: MultimodalTabularPFN,
    rows: list[dict],
    context_size: int,
    *,
    precision: str = "auto",
) -> dict:
    device = model.centers.device
    ctx_rows = rows[:context_size]
    q_rows = rows[context_size:]
    outcome_logits, prop_logits, scale, _ = model(ctx_rows, q_rows, precision=precision)
    pred = model.risks(outcome_logits, scale).detach().cpu().numpy()
    truth = np.asarray([[r[k] for k in TARGETS] for r in q_rows], dtype=np.float32)
    t, y = binary_observations(ctx_rows, device)
    rates = ArmScale.from_context(t, y).rate
    arm_mean_ite = float((rates[1] - rates[0]).item())
    ridge_ite = _ridge_t_learner_cate(ctx_rows, q_rows)

    prop_err = None
    if all(r.get(PROPENSITY_COL) is not None for r in q_rows):
        prop_pred = prop_logits.float().sigmoid().detach().cpu().numpy()
        prop_true = np.asarray([r[PROPENSITY_COL] for r in q_rows], dtype=np.float32)
        prop_err = prop_pred - prop_true

    pred_ite = pred[:, 1] - pred[:, 0]
    true_ite = truth[:, 1] - truth[:, 0]
    return dict(
        pred_ite=pred_ite,
        true_ite=true_ite,
        arm_err=(pred - truth).ravel(),
        arm_mean_err=arm_mean_ite - true_ite,
        ridge_err=ridge_ite - true_ite,
        prop_err=prop_err,
    )


@torch.inference_mode()
def evaluate_option_b(
    model: MultimodalTabularPFN,
    *,
    seed: int = 42,
    studies: int = 32,
    context_size: int = 5000,
    query_size: int = 64,
    bank: dict | None = None,
    precision: str = "auto",
    num_workers: int = 0,
    max_inflight: int | None = None,
    paraphrase_bank_path: str | Path | None = None,
) -> dict:
    """Evaluate ``MultimodalTabularPFN`` on held-out longitudinal studies (exact Run 1-5 schema)."""
    was_training = model.training
    model.eval()
    designs = split_designs(bank, "held_out")
    train_cancers = (
        {d["question"]["cancer"] for d in bank["designs"] if d["question"]["split"] == "train"}
        if bank is not None
        else set()
    )
    records = []
    by_design: dict[str, list[dict]] = {}
    by_width: dict[str, list[dict]] = {}
    by_gen: dict[str, list[dict]] = {}

    pbank_str = str(paraphrase_bank_path) if paraphrase_bank_path else None

    def _eval_jobs():
        for step in range(1, studies + 1):
            design = _select_episode_design(seed, step, True, designs)
            yield (
                step,
                (
                    "multimodal",
                    (
                        design,
                        seed,
                        step,
                        True,
                        context_size + query_size,
                        model.schema_version,
                        0,
                        False,
                        pbank_str,
                    ),
                ),
            )

    progress_every = max(studies // 20, 25) if studies >= 50 else 0
    t0 = time.monotonic()
    with _MixedEpisodePrefetcher(
        _eval_jobs(), num_workers=num_workers, max_inflight=max_inflight
    ) as prefetcher:
        for done_idx in range(1, studies + 1):
            _, _, rows, prov = prefetcher.next_item()
            rec = _score_option_b_episode(model, rows, context_size, precision=precision)
            del rows
            records.append(rec)
            if designs:
                by_design.setdefault(prov["design_id"], []).append(rec)
                by_width.setdefault(_width_bucket(prov["width"]), []).append(rec)
                gen_key = (
                    "seen_cancer_new_question"
                    if prov["cancer"] in train_cancers
                    else "unseen_cancer"
                )
                by_gen.setdefault(gen_key, []).append(rec)
            if progress_every and (done_idx % progress_every == 0 or done_idx == studies):
                elapsed = time.monotonic() - t0
                eta = elapsed * (studies - done_idx) / done_idx
                print(
                    f"[option_b eval] {done_idx}/{studies} studies "
                    f"({elapsed / 60.0:.1f}m elapsed, ~{eta / 60.0:.1f}m left)",
                    file=sys.stderr,
                    flush=True,
                )

    result = _aggregate_eval(records)
    if designs:
        result["by_width"] = {k: _aggregate_eval(v) for k, v in sorted(by_width.items())}
        result["by_generalization"] = {k: _aggregate_eval(v) for k, v in sorted(by_gen.items())}
        result["held_out_designs"] = {
            key: dict(
                studies=len(vals),
                cate_rmse=float(
                    np.sqrt(
                        np.mean(
                            np.concatenate([r["pred_ite"] - r["true_ite"] for r in vals]) ** 2
                        )
                    )
                ),
            )
            for key, vals in by_design.items()
        }
    if was_training:
        model.train()
    return result


# ==============================================================================
# 8. 50/50 MIXED EPISODE TRAINING LOOP (WITH RESUME & DUAL HELD-OUT GUARD)
# ==============================================================================
def _sample_log_uniform_int(rng: np.random.Generator, lo: int, hi: int) -> int:
    if lo >= hi:
        return int(lo)
    val = math.exp(rng.uniform(math.log(lo), math.log(hi)))
    return int(max(lo, min(hi, round(val))))


def train_option_b(
    *,
    run_dir: str | Path,
    tabular_init_dir: str | Path,
    backbone_path: str | Path,
    structured_init_dir: str | Path | None = None,
    design_bank_path: str | Path | None = DEFAULT_BANK,
    paraphrase_bank_path: str | Path | None = None,
    option_b_cfg: OptionBConfig = OptionBConfig(),
    feature_cfg: OptionAFeatureConfig = OptionAFeatureConfig(),
    steps: int = 1200,
    grad_accum_steps: int = 8,
    multimodal_prob: float = 0.5,
    min_context_size: int = 100,
    context_size: int = 5000,
    query_size: int = 64,
    tabular_min_context: int = 80,
    tabular_max_context: int = 4000,
    tabular_query_size: int = 128,
    tabular_min_features: int = 20,
    tabular_max_features: int = 2500,
    temporal_lr: float = 2e-4,
    tabular_lr: float = 3e-5,
    weight_decay: float = 1e-2,
    tabular_weight_decay: float = 1e-4,
    anchor_weight: float = 1e-4,
    warmup_steps: int = 30,
    save_every: int = 100,
    eval_every: int = 100,
    eval_studies: int = 32,
    eval_context_size: int = 5000,
    eval_query_size: int = 64,
    tabular_eval_studies: int = 32,
    tabular_eval_context: int = 800,
    tabular_eval_features: int = 500,
    eval_step_zero: bool = True,
    seed: int = 42,
    precision: str = "auto",
    num_workers: int = 6,
    max_inflight: int | None = None,
    device: str = "cuda",
    resume: bool = True,
) -> dict:
    """Train Option B (v6) hybrid multimodal CausalPFN with 50/50 tabular replay & resume."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device)
    ckpt_path = run_dir / "checkpoint.pt"
    best_path = run_dir / "best_checkpoint.pt"
    metrics_path = run_dir / "metrics.jsonl"

    # Load Tabular v4 init (always needed for initial weights and/or anchor snapshot!)
    tab_init_model, tab_init_saved = load_tabular_run(tabular_init_dir, device="cpu", prefer_best=True)
    prior_dict = dict(tab_init_saved.get("prior") or asdict(PriorConfig()))
    prior_dict["design_split"] = "train"  # strictly enforce train-only designs during training!
    prior = PriorConfig(**prior_dict)
    anchor_snapshot_cpu = _snapshot_tabular_weights(tab_init_model) if anchor_weight > 0.0 else None

    start_step = 0
    best_eval_cate = float("inf")
    init_info: dict = {
        "tabular_init_dir": str(tabular_init_dir),
        "tabular_init_step": int(tab_init_saved.get("step", -1)),
        "structured_init": None,
    }

    if resume and ckpt_path.is_file():
        model, saved = load_option_b_run(
            run_dir, device=str(dev), prefer_best=False, backbone_override=backbone_path
        )
        start_step = int(saved["step"])
        prior = PriorConfig(**{**saved["prior"], "design_split": "train"})
        settings = dict(saved.get("settings") or {})
        best_eval_cate = float(settings.get("best_eval_cate_rmse", float("inf")))
        saved_opt_state = saved.get("optimizer")
        saved_sched_state = saved.get("scheduler")
        bank = load_run_design_bank(run_dir, saved)
        pbank_local = load_run_paraphrase_bank_path(run_dir, saved)
        if bank is None and design_bank_path is not None:
            bank = load_bank(design_bank_path)
        if pbank_local is None and paraphrase_bank_path is not None:
            pbank_local = Path(paraphrase_bank_path)
        print(
            f"[option_b] Resuming {run_dir} from step {start_step}/{steps} "
            f"(best_eval_cate_rmse={best_eval_cate:.4f})",
            file=sys.stderr,
            flush=True,
        )
    else:
        backbone, tokenizer = _load_frozen_backbone(backbone_path, dev)
        tab_head = tab_init_model.to(dev)
        model = MultimodalTabularPFN(
            backbone,
            tokenizer,
            tab_head,
            option_b_cfg,
            feature_cfg,
            backbone_path=str(backbone_path),
        ).to(dev)
        if model.temporal_encoder is not None and structured_init_dir:
            init_info["structured_init"] = model.temporal_encoder.load_from_longitudinal_checkpoint(
                structured_init_dir
            )
            print(
                f"[option_b] Warm-started StructuredTemporalEncoder from {structured_init_dir}: "
                f"{init_info['structured_init']}",
                file=sys.stderr,
                flush=True,
            )
        saved_opt_state = None
        saved_sched_state = None
        bank = load_bank(design_bank_path) if design_bank_path is not None else None
        settings = {
            "init_info": init_info,
            "seed": seed,
            "steps": steps,
            "grad_accum_steps": grad_accum_steps,
            "multimodal_prob": multimodal_prob,
            "min_context_size": min_context_size,
            "context_size": context_size,
            "query_size": query_size,
            "tabular_min_context": tabular_min_context,
            "tabular_max_context": tabular_max_context,
            "tabular_query_size": tabular_query_size,
            "tabular_min_features": tabular_min_features,
            "tabular_max_features": tabular_max_features,
            "temporal_lr": temporal_lr,
            "tabular_lr": tabular_lr,
            "anchor_weight": anchor_weight,
            "best_eval_cate_rmse": best_eval_cate,
        }
        if bank is not None:
            sha = _write_gzip_json(run_dir / DESIGN_BANK_FILENAME, bank)
            settings["design_bank_file"] = DESIGN_BANK_FILENAME
            settings["design_bank_sha256"] = sha
        pbank_local = None
        if paraphrase_bank_path is not None:
            pbank_obj = load_paraphrase_bank(paraphrase_bank_path)
            sha_p = _write_gzip_json(run_dir / PARAPHRASE_BANK_FILENAME, pbank_obj)
            settings["paraphrase_bank_file"] = PARAPHRASE_BANK_FILENAME
            settings["paraphrase_bank_sha256"] = sha_p
            pbank_local = run_dir / PARAPHRASE_BANK_FILENAME

    del tab_init_model
    anchor_snapshot = (
        {k: v.to(dev) for k, v in anchor_snapshot_cpu.items()}
        if anchor_snapshot_cpu is not None
        else None
    )
    del anchor_snapshot_cpu

    # Separate parameter groups: faster LR for lightweight temporal encoder, gentle LR for Tabular v4 head
    param_groups = []
    if model.temporal_encoder is not None:
        temp_params = [p for p in model.temporal_encoder.parameters() if p.requires_grad]
        if temp_params:
            param_groups.append(
                {"params": temp_params, "lr": temporal_lr, "weight_decay": weight_decay}
            )
    tab_params = [p for p in model.tabular_head.parameters() if p.requires_grad]
    if tab_params:
        param_groups.append(
            {"params": tab_params, "lr": tabular_lr, "weight_decay": tabular_weight_decay}
        )
    if not param_groups:
        raise RuntimeError("No trainable parameters found in MultimodalTabularPFN")

    optimizer = torch.optim.AdamW(param_groups)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(warmup_steps, max(steps, 1)),
        num_training_steps=max(steps, 1),
    )
    if saved_opt_state is not None:
        optimizer.load_state_dict(saved_opt_state)
    if saved_sched_state is not None:
        scheduler.load_state_dict(saved_sched_state)

    train_designs = split_designs(bank, "train") if bank is not None else None
    pbank_str = str(pbank_local) if pbank_local is not None else None
    prior_train_dict = asdict(replace(prior, design_split="train"))

    def _run_dual_eval(cur_step: int, elapsed_s: float) -> dict:
        t_ev = time.monotonic()
        long_ev = evaluate_option_b(
            model,
            seed=seed,
            studies=eval_studies,
            context_size=eval_context_size,
            query_size=eval_query_size,
            bank=bank,
            precision=precision,
            num_workers=min(num_workers, 4),
            paraphrase_bank_path=pbank_local,
        )
        tab_ev = None
        if tabular_eval_studies > 0:
            tab_ev = evaluate_tabular(
                model.tabular_head,
                studies=tabular_eval_studies,
                context_size=tabular_eval_context,
                query_size=tabular_query_size,
                num_features=tabular_eval_features,
                prior=prior,
                seed=900_001,
                compute_baselines=False,
                num_workers=min(num_workers, 4),
            )
        ev_rec = {
            "step": cur_step,
            "event": "eval",
            "elapsed_s": round(elapsed_s, 2),
            "eval_time_s": round(time.monotonic() - t_ev, 2),
            "eval_cate_rmse": long_ev["cate_rmse"],
            "eval_within_r": long_ev["mean_within_study_cate_pearson"],
            "eval_pool_r": long_ev["cate_pearson"],
            "eval_ate_rmse": long_ev["ate_rmse"],
            "eval_risk_rmse": long_ev["risk_rmse"],
            "eval_prop_rmse": long_ev.get("propensity_rmse"),
            "tab_cate_rmse": tab_ev["cate_rmse"] if tab_ev else None,
            "tab_within_r": tab_ev.get("within_study_cate_pearson") if tab_ev else None,
            "tab_role_auroc": tab_ev.get("role_signal_auroc") if tab_ev else None,
        }
        with metrics_path.open("a", encoding="utf-8") as mf:
            mf.write(json.dumps(ev_rec) + "\n")
        print(
            f"[option_b eval step={cur_step}] "
            f"long_cate={ev_rec['eval_cate_rmse']:.4f} "
            f"within_r={ev_rec['eval_within_r'] or 0.0:+.3f} "
            f"ate={ev_rec['eval_ate_rmse']:.4f} "
            f"prop={ev_rec['eval_prop_rmse'] or 0.0:.4f} | "
            f"tab_cate={ev_rec['tab_cate_rmse'] or 0.0:.4f} "
            f"tab_wr={ev_rec['tab_within_r'] or 0.0:+.3f} "
            f"role_auc={ev_rec['tab_role_auroc'] or 0.0:.4f} "
            f"({ev_rec['eval_time_s']:.1f}s)",
            file=sys.stderr,
            flush=True,
        )
        return ev_rec

    t_start = time.monotonic()
    if start_step == 0 and eval_step_zero and eval_studies > 0:
        ev0 = _run_dual_eval(0, 0.0)
        if ev0["eval_cate_rmse"] < best_eval_cate:
            best_eval_cate = float(ev0["eval_cate_rmse"])
            settings["best_eval_cate_rmse"] = best_eval_cate
            save_option_b_checkpoint(best_path, model, optimizer, scheduler, 0, settings, prior)

    if start_step >= steps:
        return {"step": start_step, "best_eval_cate_rmse": best_eval_cate}

    # Build deterministic job stream for steps (start_step + 1) .. steps
    n_mm_per_step = int(round(grad_accum_steps * multimodal_prob))
    n_mm_per_step = max(0, min(grad_accum_steps, n_mm_per_step))

    def _train_jobs():
        for step in range(start_step + 1, steps + 1):
            step_rng = np.random.default_rng([seed, step, 4099])
            # Exact deterministic interleaved schedule per gradient accumulation step
            kinds = ["multimodal"] * n_mm_per_step + ["tabular"] * (grad_accum_steps - n_mm_per_step)
            step_rng.shuffle(kinds)
            for micro_idx, kind in enumerate(kinds):
                ep_rng = np.random.default_rng([seed, step, micro_idx, 7919])
                if kind == "multimodal":
                    n_ctx_ep = _sample_log_uniform_int(ep_rng, min_context_size, context_size)
                    design = _select_episode_design(
                        seed, step, False, train_designs, micro_step=micro_idx + 1
                    )
                    yield (
                        (step, micro_idx, n_ctx_ep),
                        (
                            "multimodal",
                            (
                                design,
                                seed,
                                step,
                                False,
                                n_ctx_ep + query_size,
                                model.schema_version,
                                micro_idx + 1,
                                True,
                                pbank_str,
                            ),
                        ),
                    )
                else:
                    n_ctx_ep = _sample_log_uniform_int(
                        ep_rng, tabular_min_context, tabular_max_context
                    )
                    p_ep = _sample_log_uniform_int(
                        ep_rng, tabular_min_features, tabular_max_features
                    )
                    # Even seeds strictly reserved for training in oci.tabular_pfn!
                    tab_seed = 2 * (seed * 100_000 + step * 100 + micro_idx + 1)
                    yield (
                        (step, micro_idx, n_ctx_ep),
                        (
                            "tabular",
                            tab_seed,
                            n_ctx_ep,
                            tabular_query_size,
                            p_ep,
                            prior_train_dict,
                        ),
                    )

    model.train()
    with _MixedEpisodePrefetcher(
        _train_jobs(), num_workers=num_workers, max_inflight=max_inflight
    ) as prefetcher:
        for step in range(start_step + 1, steps + 1):
            optimizer.zero_grad(set_to_none=True)
            mm_losses, tab_losses = [], []
            mm_cates, tab_cates = [], []
            mm_props, tab_roles = [], []
            mm_n_list, tab_n_list, tab_p_list = [], [], []
            valid_micros = 0

            for _ in range(grad_accum_steps):
                meta, kind, payload, prov = prefetcher.next_item()
                _, _, n_ctx_ep = meta
                if kind == "multimodal":
                    rows = payload
                    ctx_rows = rows[:n_ctx_ep]
                    q_rows = rows[n_ctx_ep:]
                    oracle = torch.tensor(
                        [[r[k] for k in TARGETS] for r in q_rows],
                        device=dev,
                        dtype=torch.float32,
                    )
                    true_prop = torch.tensor(
                        [r[PROPENSITY_COL] for r in q_rows],
                        device=dev,
                        dtype=torch.float32,
                    )
                    ol, pl, sc, _ = model(ctx_rows, q_rows, precision=precision)
                    ep_loss, parts = model.tabular_head.loss_components(
                        ol, pl, sc, oracle, true_prop, true_roles=None
                    )
                    del rows, ctx_rows, q_rows
                    if torch.isfinite(ep_loss):
                        (ep_loss / grad_accum_steps).backward()
                        valid_micros += 1
                        mm_losses.append(float(ep_loss.detach().item()))
                        mm_cates.append(float(parts["cate_loss"]))
                        mm_props.append(float(parts["propensity_loss"]))
                        mm_n_list.append(n_ctx_ep)
                else:
                    ep = payload
                    X = ep["X"]
                    M = ep["M"]
                    t = ep["t"]
                    y = ep["y"]
                    X_ctx_t = torch.from_numpy(X[:n_ctx_ep]).to(dev)
                    M_ctx_t = torch.from_numpy(M[:n_ctx_ep]).to(dev)
                    t_ctx_t = torch.from_numpy(t[:n_ctx_ep]).to(dev)
                    y_ctx_t = torch.from_numpy(y[:n_ctx_ep]).to(dev)
                    X_q_t = torch.from_numpy(X[n_ctx_ep:]).to(dev)
                    M_q_t = torch.from_numpy(M[n_ctx_ep:]).to(dev)
                    oracle = torch.from_numpy(
                        np.stack([ep["true_y0"][n_ctx_ep:], ep["true_y1"][n_ctx_ep:]], axis=1)
                    ).to(dev)
                    true_prop = torch.from_numpy(ep["true_propensity"][n_ctx_ep:]).to(dev)
                    true_roles = torch.from_numpy(ep["roles"]).to(dev)
                    ol, pl, sc, mem = model.forward_tabular(
                        X_ctx_t, t_ctx_t, y_ctx_t, X_q_t, M_ctx=M_ctx_t, M_q=M_q_t
                    )
                    ep_loss, parts = model.tabular_head.loss_components(
                        ol,
                        pl,
                        sc,
                        oracle,
                        true_prop,
                        role_logits=mem.role_logits,
                        nuis_screen_logits=mem.nuis_screen_logits,
                        mod_screen_logits=mem.mod_screen_logits,
                        true_roles=true_roles,
                    )
                    del ep
                    if torch.isfinite(ep_loss):
                        (ep_loss / grad_accum_steps).backward()
                        valid_micros += 1
                        tab_losses.append(float(ep_loss.detach().item()))
                        tab_cates.append(float(parts["cate_loss"]))
                        tab_roles.append(float(parts.get("role_loss", 0.0)))
                        tab_n_list.append(n_ctx_ep)
                        tab_p_list.append(int(prov["p"]))

            anchor_val = 0.0
            drift_rms = 0.0
            if anchor_weight > 0.0 and anchor_snapshot is not None:
                anc_pen, drift_rms = _anchor_penalty(model.tabular_head, anchor_snapshot)
                anc_loss = anchor_weight * anc_pen
                if torch.isfinite(anc_loss):
                    anc_loss.backward()
                    anchor_val = float(anc_loss.detach().item())

            if valid_micros > 0:
                all_trainable = [p for grp in param_groups for p in grp["params"]]
                torch.nn.utils.clip_grad_norm_(all_trainable, max_norm=1.0)
                optimizer.step()
            scheduler.step()

            elapsed = time.monotonic() - t_start
            rec = {
                "step": step,
                "event": "train",
                "temporal_lr": float(optimizer.param_groups[0]["lr"]),
                "tabular_lr": float(optimizer.param_groups[-1]["lr"]),
                "mm_loss": float(np.mean(mm_losses)) if mm_losses else None,
                "mm_cate_loss": float(np.mean(mm_cates)) if mm_cates else None,
                "mm_prop_loss": float(np.mean(mm_props)) if mm_props else None,
                "tab_loss": float(np.mean(tab_losses)) if tab_losses else None,
                "tab_cate_loss": float(np.mean(tab_cates)) if tab_cates else None,
                "tab_role_loss": float(np.mean(tab_roles)) if tab_roles else None,
                "anchor_loss": anchor_val,
                "head_drift_rms": drift_rms,
                "mean_mm_n": float(np.mean(mm_n_list)) if mm_n_list else None,
                "mean_tab_n": float(np.mean(tab_n_list)) if tab_n_list else None,
                "mean_tab_p": float(np.mean(tab_p_list)) if tab_p_list else None,
                "elapsed_s": round(elapsed, 2),
            }
            with metrics_path.open("a", encoding="utf-8") as mf:
                mf.write(json.dumps(rec) + "\n")

            if step == 1 or step % 10 == 0 or step == steps:
                print(
                    f"[option_b step={step:4d}/{steps}] "
                    f"mm_loss={rec['mm_loss'] or 0.0:.3f} "
                    f"mm_cate={rec['mm_cate_loss'] or 0.0:.4f} | "
                    f"tab_loss={rec['tab_loss'] or 0.0:.3f} "
                    f"tab_role={rec['tab_role_loss'] or 0.0:.3f} | "
                    f"drift={drift_rms:.5f} ({elapsed:.1f}s)",
                    file=sys.stderr,
                    flush=True,
                )

            if step % 25 == 0:
                _release_host_memory()

            if eval_every > 0 and eval_studies > 0 and (step % eval_every == 0 or step == steps):
                ev = _run_dual_eval(step, elapsed)
                if ev["eval_cate_rmse"] < best_eval_cate:
                    best_eval_cate = float(ev["eval_cate_rmse"])
                    settings["best_eval_cate_rmse"] = best_eval_cate
                    save_option_b_checkpoint(
                        best_path, model, optimizer, scheduler, step, settings, prior
                    )
                model.train()

            if save_every > 0 and (step % save_every == 0 or step == steps):
                save_option_b_checkpoint(
                    ckpt_path, model, optimizer, scheduler, step, settings, prior
                )

    return {"step": steps, "best_eval_cate_rmse": best_eval_cate}


# ==============================================================================
# 9. CLI ENTRYPOINT (python -m oci.longitudinal.option_b train|evaluate)
# ==============================================================================
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Option B (v6) hybrid warm-started multimodal CausalPFN trainer & evaluator"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train", help="Train or resume Option B (v6) hybrid multimodal model")
    tr.add_argument("--run-dir", required=True, type=Path)
    tr.add_argument("--tabular-init-dir", required=True, type=Path)
    tr.add_argument("--backbone-path", required=True, type=Path)
    tr.add_argument("--structured-init-dir", default=None, type=Path)
    tr.add_argument("--design-bank", default=str(DEFAULT_BANK))
    tr.add_argument("--paraphrase-bank", default=None)
    tr.add_argument("--learned-temporal-dim", type=int, default=24)
    tr.add_argument("--event-width", type=int, default=128)
    tr.add_argument("--event-layers", type=int, default=2)
    tr.add_argument("--event-heads", type=int, default=4)
    tr.add_argument("--temporal-chunk-size", type=int, default=1024)
    tr.add_argument("--no-always-screen-learned", action="store_true")
    tr.add_argument("--feature-preset", choices=sorted(OPTION_A_PRESETS), default="sent_256")
    tr.add_argument("--steps", type=int, default=1200)
    tr.add_argument("--grad-accum-steps", type=int, default=8)
    tr.add_argument("--multimodal-prob", type=float, default=0.5)
    tr.add_argument("--min-context-size", type=int, default=100)
    tr.add_argument("--context-size", type=int, default=5000)
    tr.add_argument("--query-size", type=int, default=64)
    tr.add_argument("--tabular-min-context", type=int, default=80)
    tr.add_argument("--tabular-max-context", type=int, default=4000)
    tr.add_argument("--tabular-query-size", type=int, default=128)
    tr.add_argument("--tabular-min-features", type=int, default=20)
    tr.add_argument("--tabular-max-features", type=int, default=2500)
    tr.add_argument("--temporal-lr", type=float, default=2e-4)
    tr.add_argument("--tabular-lr", type=float, default=3e-5)
    tr.add_argument("--weight-decay", type=float, default=1e-2)
    tr.add_argument("--tabular-weight-decay", type=float, default=1e-4)
    tr.add_argument("--anchor-weight", type=float, default=1e-4)
    tr.add_argument("--warmup-steps", type=int, default=30)
    tr.add_argument("--save-every", type=int, default=100)
    tr.add_argument("--eval-every", type=int, default=100)
    tr.add_argument("--eval-studies", type=int, default=32)
    tr.add_argument("--eval-context-size", type=int, default=5000)
    tr.add_argument("--eval-query-size", type=int, default=64)
    tr.add_argument("--tabular-eval-studies", type=int, default=32)
    tr.add_argument("--tabular-eval-context", type=int, default=800)
    tr.add_argument("--tabular-eval-features", type=int, default=500)
    tr.add_argument("--skip-step-zero-eval", action="store_true")
    tr.add_argument("--seed", type=int, default=42)
    tr.add_argument("--precision", choices=("auto", "bf16", "fp32"), default="auto")
    tr.add_argument("--num-workers", type=int, default=6)
    tr.add_argument("--max-inflight", type=int, default=None)
    tr.add_argument("--device", default="cuda")
    tr.add_argument("--no-resume", action="store_true")

    ev = sub.add_parser("evaluate", help="Evaluate trained Option B (v6) on held-out studies")
    ev.add_argument("--run-dir", required=True, type=Path)
    ev.add_argument("--backbone-path", default=None, type=Path)
    ev.add_argument("--studies", type=int, default=1000)
    ev.add_argument("--context-size", type=int, default=5000)
    ev.add_argument("--query-size", type=int, default=64)
    ev.add_argument("--seed", type=int, default=42)
    ev.add_argument("--design-bank", default=None)
    ev.add_argument("--paraphrase-bank", default=None)
    ev.add_argument("--precision", choices=("auto", "bf16", "fp32"), default="auto")
    ev.add_argument("--num-workers", type=int, default=6)
    ev.add_argument("--max-inflight", type=int, default=None)
    ev.add_argument("--device", default="cuda")
    ev.add_argument("--use-latest", action="store_true")
    ev.add_argument("--output", default=None, type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.cmd == "train":
        opt_b_cfg = OptionBConfig(
            learned_temporal_dim=args.learned_temporal_dim,
            event_width=args.event_width,
            event_layers=args.event_layers,
            event_heads=args.event_heads,
            temporal_chunk_size=args.temporal_chunk_size,
            always_screen_learned_cols=not args.no_always_screen_learned,
        )
        feat_cfg = OPTION_A_PRESETS[args.feature_preset]
        summary = train_option_b(
            run_dir=args.run_dir,
            tabular_init_dir=args.tabular_init_dir,
            backbone_path=args.backbone_path,
            structured_init_dir=args.structured_init_dir,
            design_bank_path=args.design_bank,
            paraphrase_bank_path=args.paraphrase_bank,
            option_b_cfg=opt_b_cfg,
            feature_cfg=feat_cfg,
            steps=args.steps,
            grad_accum_steps=args.grad_accum_steps,
            multimodal_prob=args.multimodal_prob,
            min_context_size=args.min_context_size,
            context_size=args.context_size,
            query_size=args.query_size,
            tabular_min_context=args.tabular_min_context,
            tabular_max_context=args.tabular_max_context,
            tabular_query_size=args.tabular_query_size,
            tabular_min_features=args.tabular_min_features,
            tabular_max_features=args.tabular_max_features,
            temporal_lr=args.temporal_lr,
            tabular_lr=args.tabular_lr,
            weight_decay=args.weight_decay,
            tabular_weight_decay=args.tabular_weight_decay,
            anchor_weight=args.anchor_weight,
            warmup_steps=args.warmup_steps,
            save_every=args.save_every,
            eval_every=args.eval_every,
            eval_studies=args.eval_studies,
            eval_context_size=args.eval_context_size,
            eval_query_size=args.eval_query_size,
            tabular_eval_studies=args.tabular_eval_studies,
            tabular_eval_context=args.tabular_eval_context,
            tabular_eval_features=args.tabular_eval_features,
            eval_step_zero=not args.skip_step_zero_eval,
            seed=args.seed,
            precision=args.precision,
            num_workers=args.num_workers,
            max_inflight=args.max_inflight,
            device=args.device,
            resume=not args.no_resume,
        )
        print(json.dumps(summary, indent=2))
    elif args.cmd == "evaluate":
        model, saved = load_option_b_run(
            args.run_dir,
            device=args.device,
            prefer_best=not args.use_latest,
            backbone_override=args.backbone_path,
        )
        bank = (
            load_bank(args.design_bank)
            if args.design_bank
            else load_run_design_bank(args.run_dir, saved)
        )
        pbank_path = (
            Path(args.paraphrase_bank)
            if args.paraphrase_bank
            else load_run_paraphrase_bank_path(args.run_dir, saved)
        )
        res = evaluate_option_b(
            model,
            seed=args.seed,
            studies=args.studies,
            context_size=args.context_size,
            query_size=args.query_size,
            bank=bank,
            precision=args.precision,
            num_workers=args.num_workers,
            max_inflight=args.max_inflight,
            paraphrase_bank_path=pbank_path,
        )
        res["checkpoint_step"] = int(saved.get("step", -1))
        text = json.dumps(res, indent=2)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + "\n", encoding="utf-8")
        print(text)


if __name__ == "__main__":
    main()
