"""A cross-modal longitudinal patient encoder and hybrid differentiable CausalPFN study model."""

from dataclasses import dataclass
import inspect
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.checkpoint import checkpoint

from oci.models.lossless_tokenization import SemanticTruncationError
from .events import (
    CODE_BUCKETS,
    ENCOUNTER_SUMMARY_SIZE,
    EXTRA_LAB_BUCKETS,
    EXTRA_PRO_BUCKETS,
    LAB_INPUT_SIZE,
    LAB_SUMMARY_SIZE,
    LEGACY_STREAMS,
    PRO_INPUT_SIZE,
    PRO_SUMMARY_SIZE,
    STREAMS,
    VITAL_INPUT_SIZE,
    VITAL_SUMMARY_SIZE,
    binary_observations,
    encounter_codes,
    encounter_stream_summary,
    event_times,
    lab_features,
    lab_stream_summary,
    note_lexical_vector,
    note_sentences,
    pro_features,
    pro_stream_summary,
    vital_stream_summary,
    vital_vector,
)

BASE_MODEL = "huawei-noah/TinyBERT_General_4L_312D"
BASE_REVISION = None
PINNED_REVISIONS = {
    "LiquidAI/LFM2.5-230M-Base": "9d2be5519834990d30996f878b6771cccbd24f2c",
}
_BIDIRECTIONAL_MODELS = frozenset(
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
        "roberta",
        "xlm-roberta",
    }
)
CHECKPOINT_FORMAT_VERSION = 4
NOTE_WINDOW_OVERLAP = 32


@dataclass(frozen=True)
class ModelConfig:
    event_width: int = 128
    event_layers: int = 2
    study_width: int = 192
    study_layers: int = 2
    heads: int = 4
    bins: int = 129
    note_batch_size: int = 64
    note_slots: int = 8
    solver_dim: int = 64
    lexical_dim: int = 0
    sentence_concept_dim: int = 0
    ridge_grid_size: int = 5
    block_ridge: bool = False
    per_slot_note_temporal: bool = False
    cate_loss_weight: float = 2.0
    propensity_loss_weight: float = 0.5

    def __post_init__(self):
        if (
            min(
                self.event_width,
                self.study_width,
                self.heads,
                self.event_layers,
                self.study_layers,
                self.note_batch_size,
                self.note_slots,
                self.solver_dim,
            )
            < 1
            or self.lexical_dim < 0
            or self.sentence_concept_dim < 0
            or self.ridge_grid_size < 2
            or self.bins < 3
            or self.cate_loss_weight < 0
            or self.propensity_loss_weight < 0
        ):
            raise ValueError("Invalid model dimensions, loss weights, or note batch size")
        if self.event_width % self.heads or self.study_width % self.heads:
            raise ValueError("Both transformer widths must be divisible by heads")


def _sinh_grid(bins, max_offset=6.0, warp=2.0):
    """Non-uniform logit-offset grid concentrating resolution near the center."""
    u = torch.linspace(-1.0, 1.0, bins)
    return max_offset * torch.sinh(warp * u) / math.sinh(warp)


class EventTransformer(nn.Module):
    """Summarize a variable-length timestamped stream into one or more concept tokens."""

    def __init__(self, cfg, *, num_queries=1):
        super().__init__()
        width = cfg.event_width
        self.num_queries = num_queries
        self.clock = nn.Sequential(nn.Linear(3, width), nn.GELU(), nn.Linear(width, width))
        self.summary = nn.Parameter(torch.randn(num_queries, width) * 0.02)
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                width,
                cfg.heads,
                4 * width,
                dropout=0.0,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
            for _ in range(cfg.event_layers)
        )
        self.norm = nn.LayerNorm(width)

    def forward(self, values, times, lengths):
        if len(times):
            time_tensor = values.new_tensor(times).reshape(-1, 3)
            events = values + self.clock(time_tensor)
        else:
            events = values
        sequences = [torch.cat([self.summary, s], dim=0) for s in events.split(lengths)]
        tokens = pad_sequence(sequences, batch_first=True)
        total_lengths = torch.tensor(lengths, device=tokens.device) + self.num_queries
        mask = torch.arange(tokens.shape[1], device=tokens.device)[None, :] >= total_lengths[:, None]
        for layer in self.layers:
            tokens = layer(tokens, src_key_padding_mask=mask)
        return self.norm(tokens[:, : self.num_queries])


class CrossModalTimeline(nn.Module):
    """Attend jointly across chronologically aligned events from every modality."""

    def __init__(self, cfg, num_modalities):
        super().__init__()
        width = cfg.event_width
        self.modality = nn.Embedding(num_modalities, width)
        self.clock = nn.Sequential(nn.Linear(3, width), nn.GELU(), nn.Linear(width, width))
        self.summary = nn.Parameter(torch.randn(1, width) * 0.02)
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                width,
                cfg.heads,
                4 * width,
                dropout=0.0,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
            for _ in range(cfg.event_layers)
        )
        self.norm = nn.LayerNorm(width)

    def forward(self, per_patient_events):
        sequences = []
        lengths = []
        for tokens, days in per_patient_events:
            if len(days):
                order = sorted(range(len(days)), key=days.__getitem__)
                ordered = tokens[order]
                sequences.append(torch.cat([self.summary, ordered], dim=0))
                lengths.append(len(days))
            else:
                sequences.append(self.summary)
                lengths.append(0)
        padded = pad_sequence(sequences, batch_first=True)
        total_lengths = torch.tensor(lengths, device=padded.device) + 1
        mask = torch.arange(padded.shape[1], device=padded.device)[None, :] >= total_lengths[:, None]
        for layer in self.layers:
            padded = layer(padded, src_key_padding_mask=mask)
        return self.norm(padded[:, :1])


class _SummaryProjector(nn.Module):
    """Linear trajectory projection plus small-init nonlinear residual to preserve slopes."""

    def __init__(self, in_dim, width):
        super().__init__()
        self.linear = nn.Linear(in_dim, width)
        self.residual = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        nn.init.normal_(self.residual[-1].weight, std=0.01)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, x):
        return self.linear(x) + self.residual(x)


def _summary_mlp(in_dim, width):
    return _SummaryProjector(in_dim, width)


class PatientEncoder(nn.Module):
    """Cross-modal longitudinal patient encoder with multi-query note attention pooling."""

    def __init__(self, backbone, tokenizer, cfg, *, include_pros=True):
        super().__init__()
        self.backbone, self.tokenizer = backbone, tokenizer
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                raise ValueError("Note tokenizer must define a pad_token or eos_token")
        self.note_batch_size = cfg.note_batch_size
        self.note_slots = cfg.note_slots
        self.per_slot_note_temporal = bool(cfg.per_slot_note_temporal)
        self._accepts_use_cache = (
            "use_cache" in inspect.signature(self.backbone.forward).parameters
        )
        self._is_bidirectional = (
            getattr(self.backbone.config, "model_type", "") in _BIDIRECTIONAL_MODELS
        )
        width = cfg.event_width
        hidden_size = backbone.config.hidden_size

        # Multi-query learned token attention pooling + max pooling + mean gestalt for clinical notes.
        self.note_pool_norm = nn.LayerNorm(hidden_size)
        self.note_key = nn.Linear(hidden_size, width)
        self.note_val = nn.Linear(hidden_size, width)
        self.note_queries = nn.Parameter(torch.randn(cfg.note_slots, width) * 0.02)
        self.note_max_proj = nn.Linear(hidden_size, width)
        self.notes = nn.Linear(hidden_size, width)
        self.note_cross_norm = nn.LayerNorm(width)
        if self.per_slot_note_temporal:
            self.note_slot_clock = nn.Sequential(
                nn.Linear(3, width), nn.GELU(), nn.Linear(width, width)
            )
            self.note_slot_time_q = nn.Parameter(torch.randn(cfg.note_slots, width) * 0.02)
            self.note_slot_temporal_fuse = nn.Linear(4 * width, width)
            self.note_slot_temporal_norm = nn.LayerNorm(width)

        self.labs = nn.Linear(LAB_INPUT_SIZE, width)
        self.extra_labs = nn.Embedding(EXTRA_LAB_BUCKETS, width)
        self.extra_lab_scale = nn.Linear(1, width)
        self.lab_summary = _summary_mlp(LAB_SUMMARY_SIZE, width)

        self.vitals = nn.Linear(VITAL_INPUT_SIZE, width)
        self.vital_summary = _summary_mlp(VITAL_SUMMARY_SIZE, width)

        # Mean pooling plus explicit log-count keeps encounter norms stable on real EHRs.
        self.codes = nn.EmbeddingBag(CODE_BUCKETS, width, mode="mean")
        self.code_count = nn.Linear(1, width)
        self.encounter_summary = _summary_mlp(ENCOUNTER_SUMMARY_SIZE, width)

        if include_pros:
            self.pros = nn.Linear(PRO_INPUT_SIZE, width)
            self.extra_pros = nn.Embedding(EXTRA_PRO_BUCKETS, width)
            self.extra_pro_scale = nn.Linear(1, width)
            self.pro_summary = _summary_mlp(PRO_SUMMARY_SIZE, width)

        names = STREAMS if include_pros else LEGACY_STREAMS
        self.streams = nn.ModuleDict(
            {
                name: EventTransformer(cfg, num_queries=cfg.note_slots if name == "notes" else 1)
                for name in names
            }
        )
        self.cross_modal = CrossModalTimeline(cfg, len(names))
        self.num_slots = cfg.note_slots + (len(names) - 1) + 1

    def _backbone_hidden(self, ids, mask):
        kwargs = {"input_ids": ids, "attention_mask": mask}
        if self._accepts_use_cache:
            kwargs["use_cache"] = False
        return self.backbone(**kwargs).last_hidden_state

    def _pool_window_hidden(self, hidden, mask):
        """Extract exact per-window multi-query attention, max, and token-sum statistics."""
        h = hidden.to(self.note_pool_norm.weight.dtype)
        valid = mask.bool()
        h_norm = self.note_pool_norm(h)
        k = self.note_key(h_norm)
        v = self.note_val(h_norm)
        scores = torch.einsum("md,bld->bml", self.note_queries, k).float() / math.sqrt(k.shape[-1])
        scores = scores.masked_fill(~valid[:, None, :], -1e9)
        win_lse = torch.logsumexp(scores, dim=-1)  # (B, M) float32
        attn_w = torch.softmax(scores, dim=-1).to(v.dtype)
        win_attn = torch.einsum("bml,bld->bmd", attn_w, v)  # (B, M, D)
        win_max = h.float().masked_fill(~valid[:, :, None], -1e9).max(dim=1).values.to(h.dtype)
        w_f = valid.to(h.dtype).unsqueeze(-1)
        win_sum = (h * w_f).sum(dim=1)
        win_cnt = w_f.sum(dim=1).clamp_min(1.0)
        return win_attn, win_lse, win_max, win_sum, win_cnt

    def _forward_and_pool_window(self, ids, mask):
        hidden = self._backbone_hidden(ids, mask)
        return self._pool_window_hidden(hidden, mask)

    def _window_note_ids(self, ids, limit, *, overlap=NOTE_WINDOW_OVERLAP):
        """Split notes longer than ``limit`` into lossless overlapping windows."""
        cls_id = self.tokenizer.cls_token_id
        if cls_id is None:
            cls_id = self.tokenizer.bos_token_id
        sep_id = self.tokenizer.sep_token_id
        if sep_id is None:
            sep_id = self.tokenizer.eos_token_id

        windows = []
        counts = []
        for seq in ids:
            if not seq:
                raise SemanticTruncationError("Expected at least 1 token in every clinical note")
            if len(seq) <= limit:
                windows.append(seq)
                counts.append(1)
                continue
            prefix = [cls_id] if cls_id is not None and seq[0] == cls_id else []
            suffix = [sep_id] if sep_id is not None and seq[-1] == sep_id else []
            body = seq[len(prefix) : len(seq) - len(suffix) if suffix else len(seq)]
            body_limit = limit - len(prefix) - len(suffix)
            if body_limit < 1:
                raise SemanticTruncationError(
                    f"Backbone max_position_embeddings ({limit}) is too small for windowing"
                )
            eff_overlap = min(max(int(overlap), 0), body_limit // 2)
            stride = body_limit - eff_overlap
            note_windows = []
            pos = 0
            while True:
                end = min(pos + body_limit, len(body))
                note_windows.append([*prefix, *body[pos:end], *suffix])
                if end >= len(body):
                    break
                pos += stride
            windows.extend(note_windows)
            counts.append(len(note_windows))
        return windows, counts

    def _encode_note_subset(self, ids, indices, *, backbone_grad):
        """Run backbone (optionally frozen/no_grad) and trainable token attention pooler."""
        device = self.notes.weight.device
        grad_enabled = self.training and torch.is_grad_enabled()
        with_bb_grad = backbone_grad and grad_enabled
        batch_size = self.note_batch_size if with_bb_grad else max(self.note_batch_size * 4, 32)
        attns, lses, maxs, sums, cnts = [], [], [], [], []
        for start in range(0, len(indices), batch_size):
            sub_ids = [ids[i] for i in indices[start : start + batch_size]]
            batch = self.tokenizer.pad(
                {"input_ids": sub_ids},
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            ).to(device)
            input_ids, attn_mask = batch["input_ids"], batch["attention_mask"]
            if with_bb_grad:
                out = checkpoint(
                    self._forward_and_pool_window, input_ids, attn_mask, use_reentrant=False
                )
            else:
                with torch.no_grad():
                    hidden = self._backbone_hidden(input_ids, attn_mask)
                if grad_enabled:
                    out = checkpoint(
                        self._pool_window_hidden, hidden, attn_mask, use_reentrant=False
                    )
                else:
                    out = self._pool_window_hidden(hidden, attn_mask)
            attns.append(out[0])
            lses.append(out[1])
            maxs.append(out[2])
            sums.append(out[3])
            cnts.append(out[4])
        return (
            torch.cat(attns, dim=0),
            torch.cat(lses, dim=0),
            torch.cat(maxs, dim=0),
            torch.cat(sums, dim=0),
            torch.cat(cnts, dim=0),
        )

    def _aggregate_windows(self, win_attn, win_lse, win_max, win_sum, win_cnt, counts):
        """Exact global attention, max, and token-weighted mean pooling across lossless windows."""
        if all(c == 1 for c in counts):
            note_attn = win_attn
            note_max_raw = win_max
            note_mean_raw = win_sum / win_cnt.clamp_min(1.0)
        else:
            agg_attn, agg_max, agg_mean = [], [], []
            for a_g, lse_g, mx_g, sm_g, ct_g in zip(
                win_attn.split(counts),
                win_lse.split(counts),
                win_max.split(counts),
                win_sum.split(counts),
                win_cnt.split(counts),
                strict=True,
            ):
                if len(a_g) == 1:
                    agg_attn.append(a_g[0])
                    agg_max.append(mx_g[0])
                    agg_mean.append(sm_g[0] / ct_g[0].clamp_min(1.0))
                else:
                    w_win = torch.softmax(lse_g, dim=0).to(a_g.dtype).unsqueeze(-1)
                    agg_attn.append((w_win * a_g).sum(dim=0))
                    agg_max.append(mx_g.max(dim=0).values)
                    agg_mean.append(sm_g.sum(dim=0) / ct_g.sum(dim=0).clamp_min(1.0))
            note_attn = torch.stack(agg_attn, dim=0)
            note_max_raw = torch.stack(agg_max, dim=0)
            note_mean_raw = torch.stack(agg_mean, dim=0)
        return note_attn, self.note_max_proj(note_max_raw), self.notes(note_mean_raw)

    def _encode_notes(self, events, grad_flags=None):
        """Return per-note ``(note_attn [N, M, D], note_max [N, D], note_mean [N, D])``."""
        device = self.notes.weight.device
        if not events:
            width = self.notes.out_features
            dtype = self.notes.weight.dtype
            return (
                torch.empty((0, self.note_slots, width), device=device, dtype=dtype),
                torch.empty((0, width), device=device, dtype=dtype),
                torch.empty((0, width), device=device, dtype=dtype),
            )
        ids = self.tokenizer(
            [e["text"] for e in events], truncation=False, return_attention_mask=False
        )["input_ids"]
        limit = self.backbone.config.max_position_embeddings
        windows, counts = self._window_note_ids(ids, limit)

        if grad_flags is None or not (self.training and torch.is_grad_enabled()):
            bb_grad = self.training and torch.is_grad_enabled()
            stats = self._encode_note_subset(
                windows, list(range(len(windows))), backbone_grad=bb_grad
            )
            return self._aggregate_windows(*stats, counts)

        win_grad_flags = [
            flag for flag, count in zip(grad_flags, counts, strict=True) for _ in range(count)
        ]
        grad_idx = [i for i, flag in enumerate(win_grad_flags) if flag]
        nograd_idx = [i for i, flag in enumerate(win_grad_flags) if not flag]
        if not nograd_idx:
            stats = self._encode_note_subset(windows, grad_idx, backbone_grad=True)
            return self._aggregate_windows(*stats, counts)
        if not grad_idx:
            stats = self._encode_note_subset(windows, nograd_idx, backbone_grad=False)
            return self._aggregate_windows(*stats, counts)

        g_stats = self._encode_note_subset(windows, grad_idx, backbone_grad=True)
        ng_stats = self._encode_note_subset(windows, nograd_idx, backbone_grad=False)
        g_idx_t = torch.tensor(grad_idx, device=device, dtype=torch.long)
        ng_idx_t = torch.tensor(nograd_idx, device=device, dtype=torch.long)
        combined = []
        for g_t, ng_t in zip(g_stats, ng_stats, strict=True):
            buf = g_t.new_empty((len(windows), *g_t.shape[1:]))
            buf[g_idx_t] = g_t
            buf[ng_idx_t] = ng_t
            combined.append(buf)
        return self._aggregate_windows(*combined, counts)

    def _note_vectors(self, events, grad_flags=None):
        """Combined per-note summary vector (attention + max + mean)."""
        note_attn, note_max, note_mean = self._encode_notes(events, grad_flags=grad_flags)
        return note_attn.mean(dim=1) + note_max + note_mean

    def lexical_features(self, rows):
        """Hashed 1-3gram log-count matrix over each patient's pretreatment notes."""
        if not rows:
            raise ValueError("Expected at least one patient")
        device = self.notes.weight.device
        return torch.tensor(
            [note_lexical_vector(row.get("notes")) for row in rows],
            device=device,
            dtype=torch.float32,
        )

    @torch.no_grad()
    def sentence_features(self, rows, *, max_length=64, max_sentences=128):
        """Deduplicated, L2-normalized per-sentence backbone embeddings for study-adaptive pooling."""
        if not rows:
            raise ValueError("Expected at least one patient")
        device = self.notes.weight.device
        hidden_size = int(self.backbone.config.hidden_size)
        per_patient = [
            note_sentences(row.get("notes"), max_sentences=max_sentences) for row in rows
        ]
        uniq_texts = tuple(sorted({s for sents in per_patient for s in sents}))
        if not uniq_texts:
            return SentenceBatch(
                unique_embs=torch.empty((0, hidden_size), device=device, dtype=torch.float32),
                flat_idx=torch.empty((0,), device=device, dtype=torch.long),
                patient_ids=torch.empty((0,), device=device, dtype=torch.long),
                num_patients=len(rows),
                unique_texts=(),
            )

        text_to_id = {text: idx for idx, text in enumerate(uniq_texts)}
        flat_idx_list = []
        patient_id_list = []
        for p_idx, sents in enumerate(per_patient):
            for s in sents:
                flat_idx_list.append(text_to_id[s])
                patient_id_list.append(p_idx)

        limit = int(getattr(self.backbone.config, "max_position_embeddings", max_length))
        eff_len = min(max_length, limit)
        batch_size = max(self.note_batch_size * 4, 256)
        order = sorted(range(len(uniq_texts)), key=lambda i: (len(uniq_texts[i]), uniq_texts[i]))
        unique_embs = torch.empty((len(uniq_texts), hidden_size), device=device, dtype=torch.float32)

        for start in range(0, len(order), batch_size):
            batch_order = order[start : start + batch_size]
            batch_texts = [uniq_texts[i] for i in batch_order]
            enc = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=eff_len,
                return_tensors="pt",
                return_attention_mask=True,
            ).to(device)
            hidden = self._backbone_hidden(enc["input_ids"], enc["attention_mask"]).float()
            mask = enc["attention_mask"]
            if self._is_bidirectional:
                w = mask.unsqueeze(-1).float()
                pooled = (hidden * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)
            else:
                last_pos = mask.sum(dim=1).clamp_min(1) - 1
                pooled = hidden[torch.arange(len(batch_texts), device=device), last_pos]
            pooled = F.normalize(pooled, p=2, dim=-1)
            dest = torch.tensor(batch_order, device=device, dtype=torch.long)
            unique_embs[dest] = pooled

        return SentenceBatch(
            unique_embs=unique_embs,
            flat_idx=torch.tensor(flat_idx_list, device=device, dtype=torch.long),
            patient_ids=torch.tensor(patient_id_list, device=device, dtype=torch.long),
            num_patients=len(rows),
            unique_texts=uniq_texts,
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
        splits = item_vecs.split(counts)
        for row_idx, chunk in zip(nonempty_rows, splits, strict=True):
            out[row_idx] = chunk.mean(dim=0)
        return out

    def _values(self, name, events):
        if not events:
            return self.notes.weight.new_empty(0, self.notes.out_features)
        if name == "encounters":
            codes = [encounter_codes(e) for e in events]
            device = self.codes.weight.device
            ids = torch.tensor([i for bag in codes for i in bag], device=device, dtype=torch.long)
            offsets = torch.tensor([0] + [len(bag) for bag in codes[:-1]], device=device).cumsum(0)
            counts = self.code_count.weight.new_tensor(
                [[math.log1p(len(bag))] for bag in codes]
            )
            return self.codes(ids, offsets) + self.code_count(counts)
        if name == "labs":
            parsed = [lab_features(e, strict=False) for e in events]
            dense = self.labs(self.labs.weight.new_tensor([v for v, _ in parsed]))
            extra = self._extra_bag(
                [x for _, x in parsed], self.extra_labs, self.extra_lab_scale
            )
            return dense + extra
        if name == "pros":
            parsed = [pro_features(e, strict=False) for e in events]
            dense = self.pros(self.pros.weight.new_tensor([v for v, _ in parsed]))
            extra = self._extra_bag(
                [x for _, x in parsed], self.extra_pros, self.extra_pro_scale
            )
            return dense + extra
        return self.vitals(self.vitals.weight.new_tensor([vital_vector(e) for e in events]))

    def _stream_trajectory_residual(self, name, streams):
        device = self.notes.weight.device
        dtype = self.notes.weight.dtype
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
        if name == "pros":
            vec = torch.tensor([pro_stream_summary(s) for s in streams], device=device, dtype=dtype)
            return self.pro_summary(vec).unsqueeze(1)
        return None

    def forward(self, rows, *, backbone_grad_mask=None):
        if not rows:
            raise ValueError("Expected at least one patient")
        if backbone_grad_mask is not None and len(backbone_grad_mask) != len(rows):
            raise ValueError("backbone_grad_mask length must match number of patients")
        n_patients = len(rows)
        stream_summaries = []
        per_patient_tokens = [[] for _ in range(n_patients)]
        per_patient_days = [[] for _ in range(n_patients)]

        for modality_idx, (name, encoder) in enumerate(self.streams.items()):
            streams = [
                [] if name == "pros" and row.get(name) is None else row[name] for row in rows
            ]
            lengths = [len(s) for s in streams]
            times = [time for stream in streams for time in event_times(stream)]
            flat_events = [event for stream in streams for event in stream]
            if name == "notes":
                note_grad_flags = None
                if backbone_grad_mask is not None:
                    note_grad_flags = [
                        flag
                        for flag, length in zip(backbone_grad_mask, lengths, strict=True)
                        for _ in range(length)
                    ]
                note_attn, note_max, note_mean = self._encode_notes(
                    flat_events, grad_flags=note_grad_flags
                )
                values = note_attn.mean(dim=1) + note_max + note_mean
                temporal = encoder(values, times, lengths)  # (N_patients, M, D)
                # Preserve rare findings (and optional per-slot progression) across longitudinal notes.
                cross_note = temporal.new_zeros((n_patients, self.note_slots, values.shape[-1]))
                if self.per_slot_note_temporal:
                    if len(times):
                        slot_time_emb = self.note_slot_clock(
                            values.new_tensor(times).reshape(-1, 3)
                        )
                    else:
                        slot_time_emb = values.new_empty((0, values.shape[-1]))
                    inv_sqrt_d = 1.0 / math.sqrt(values.shape[-1])
                    for p_idx, (a_chunk, mx_chunk, t_chunk) in enumerate(
                        zip(
                            note_attn.split(lengths),
                            note_max.split(lengths),
                            slot_time_emb.split(lengths),
                            strict=True,
                        )
                    ):
                        if len(a_chunk):
                            slot_max = a_chunk.max(dim=0).values
                            slot_latest = a_chunk[-1]
                            slot_delta = (
                                a_chunk[-1] - a_chunk[0]
                                if len(a_chunk) >= 2
                                else torch.zeros_like(slot_latest)
                            )
                            timed = a_chunk + t_chunk.unsqueeze(1)
                            t_scores = (
                                torch.einsum("md,lmd->ml", self.note_slot_time_q, timed).float()
                                * inv_sqrt_d
                            )
                            t_w = torch.softmax(t_scores, dim=-1).to(a_chunk.dtype)
                            slot_attn = torch.einsum("ml,lmd->md", t_w, a_chunk)
                            fused_slot = self.note_slot_temporal_norm(
                                self.note_slot_temporal_fuse(
                                    torch.cat(
                                        [slot_max, slot_latest, slot_delta, slot_attn], dim=-1
                                    )
                                )
                            )
                            cross_note[p_idx] = fused_slot + mx_chunk.max(dim=0).values.unsqueeze(0)
                else:
                    for p_idx, (a_chunk, mx_chunk) in enumerate(
                        zip(note_attn.split(lengths), note_max.split(lengths), strict=True)
                    ):
                        if len(a_chunk):
                            cross_note[p_idx] = a_chunk.max(dim=0).values + mx_chunk.max(
                                dim=0
                            ).values.unsqueeze(0)
                stream_summaries.append(temporal + self.note_cross_norm(cross_note))
            else:
                values = self._values(name, flat_events)
                encoded = encoder(values, times, lengths)
                traj = self._stream_trajectory_residual(name, streams)
                stream_summaries.append(encoded + traj if traj is not None else encoded)

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
                else self.notes.weight.new_empty(0, self.notes.out_features),
                days,
            )
            for chunks, days in zip(per_patient_tokens, per_patient_days, strict=True)
        ]
        fused = self.cross_modal(fused_inputs)
        # Shape: (N_patients, num_slots, event_width)
        return torch.cat([*stream_summaries, fused], dim=1)


class TwoWayStudyLayer(nn.Module):
    """Alternating feature/slot-wise self-attention and context-only sample attention."""

    def __init__(self, cfg):
        super().__init__()
        d = cfg.study_width
        self.slot_norm = nn.LayerNorm(d)
        self.slot_attn = nn.MultiheadAttention(d, cfg.heads, batch_first=True)
        self.sample_norm = nn.LayerNorm(d)
        self.sample_attn = nn.MultiheadAttention(d, cfg.heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)
        )

    def slot_mix(self, patients):
        """Within-patient interaction across modality/concept slots and the (T, Y) token."""
        normed = self.slot_norm(patients)
        return patients + self.slot_attn(normed, normed, normed, need_weights=False)[0]

    def sample_mix(self, patients, memory):
        """Across-patient attention per slot; only context patients supply keys and values.

        ``patients`` has shape ``(N, S, D)`` and ``memory`` has shape ``(S, N_ctx, D)``.
        """
        q = self.sample_norm(patients).transpose(0, 1)  # (S, N, D)
        update = self.sample_attn(q, memory, memory, need_weights=False)[0].transpose(0, 1)
        patients = patients + update
        return patients + self.ff(patients)


@dataclass
class ArmScale:
    rate: torch.Tensor
    sd: torch.Tensor
    counts: torch.Tensor

    @classmethod
    def from_context(cls, t, y):
        if t.ndim != 1 or t.shape != y.shape or not len(t):
            raise ValueError("Expected aligned nonempty context observations")
        arms = torch.arange(2, device=t.device)
        mask = t[:, None] == arms
        counts = mask.sum(0).to(dtype=torch.float32)
        rates = ((mask * y[:, None]).sum(0) + 0.5) / (counts + 1.0)
        return cls(rates, (rates * (1.0 - rates)).sqrt(), counts)

    def logit_offsets(self, risks):
        """Centered log-odds residual relative to each arm's empirical rate."""
        return torch.logit(risks, eps=1e-6) - self.rate.logit()

    def bin_risks(self, centers):
        """Map centered logit bin centers back to arm-specific probabilities."""
        return (self.rate.logit()[:, None] + centers[None, :]).sigmoid()


@dataclass
class RidgeFit:
    """Cached closed-form LOO-CV Ridge solution for repeated query batch scoring."""

    weights: torch.Tensor  # (D + 1,)
    chol: torch.Tensor  # (K, D + 1, D + 1)
    alpha: torch.Tensor  # (K,)


@dataclass
class SentenceBatch:
    """Deduplicated per-sentence backbone embeddings mapped back to patients."""

    unique_embs: torch.Tensor  # (U, H) float32 L2-normalized
    flat_idx: torch.Tensor  # (M,) long indices into unique_embs
    patient_ids: torch.Tensor  # (M,) long indices in [0, num_patients)
    num_patients: int
    unique_texts: tuple[str, ...] = ()

    @classmethod
    def concat(cls, batches):
        """Merge chunked SentenceBatch objects while preserving canonical deduplicated order."""
        if not batches:
            raise ValueError("Expected at least one SentenceBatch")
        if len(batches) == 1:
            return batches[0]
        device = batches[0].unique_embs.device
        hidden_size = batches[0].unique_embs.shape[1]
        if all(len(b.unique_texts) == len(b.unique_embs) for b in batches):
            text_to_emb = {}
            for b in batches:
                for idx, text in enumerate(b.unique_texts):
                    if text not in text_to_emb:
                        text_to_emb[text] = b.unique_embs[idx]
            merged_texts = tuple(sorted(text_to_emb))
            if not merged_texts:
                return cls(
                    unique_embs=torch.empty((0, hidden_size), device=device, dtype=torch.float32),
                    flat_idx=torch.empty((0,), device=device, dtype=torch.long),
                    patient_ids=torch.empty((0,), device=device, dtype=torch.long),
                    num_patients=sum(b.num_patients for b in batches),
                    unique_texts=(),
                )
            text_to_id = {t: i for i, t in enumerate(merged_texts)}
            merged_embs = torch.stack([text_to_emb[t] for t in merged_texts], dim=0)
            flat_parts = []
            pat_parts = []
            pat_offset = 0
            for b in batches:
                if len(b.flat_idx):
                    remap = torch.tensor(
                        [text_to_id[t] for t in b.unique_texts], device=device, dtype=torch.long
                    )
                    flat_parts.append(remap[b.flat_idx])
                    pat_parts.append(b.patient_ids + pat_offset)
                pat_offset += b.num_patients
            return cls(
                unique_embs=merged_embs,
                flat_idx=torch.cat(flat_parts, dim=0) if flat_parts else torch.empty((0,), device=device, dtype=torch.long),
                patient_ids=torch.cat(pat_parts, dim=0) if pat_parts else torch.empty((0,), device=device, dtype=torch.long),
                num_patients=pat_offset,
                unique_texts=merged_texts,
            )

        emb_parts = []
        flat_parts = []
        pat_parts = []
        emb_offset = 0
        pat_offset = 0
        for b in batches:
            if len(b.unique_embs):
                emb_parts.append(b.unique_embs)
            if len(b.flat_idx):
                flat_parts.append(b.flat_idx + emb_offset)
                pat_parts.append(b.patient_ids + pat_offset)
            emb_offset += len(b.unique_embs)
            pat_offset += b.num_patients
        return cls(
            unique_embs=torch.cat(emb_parts, dim=0) if emb_parts else torch.empty((0, hidden_size), device=device, dtype=torch.float32),
            flat_idx=torch.cat(flat_parts, dim=0) if flat_parts else torch.empty((0,), device=device, dtype=torch.long),
            patient_ids=torch.cat(pat_parts, dim=0) if pat_parts else torch.empty((0,), device=device, dtype=torch.long),
            num_patients=pat_offset,
            unique_texts=(),
        )


@dataclass
class SolverMemory:
    """Context-conditioned state for the differentiable in-context orthogonalized R-learner."""

    mu: torch.Tensor  # (1, D)
    sd: torch.Tensor  # (1, D)
    propensity_fit: RidgeFit
    outcome_fit: RidgeFit
    cate_fit: RidgeFit
    n_ctx: int
    lex_mu: torch.Tensor | None = None
    lex_basis: torch.Tensor | None = None
    lex_rms: torch.Tensor | None = None
    sent_centroids: torch.Tensor | None = None
    sent_mu: torch.Tensor | None = None
    sent_rms: torch.Tensor | None = None


@dataclass
class StudyMemory:
    """Combined context memory for both the differentiable R-learner and residual attention."""

    solver: SolverMemory
    attn_memories: list


@torch.no_grad()
def _fit_lexical_basis(lex_ctx, max_dim):
    """Unsupervised, permutation-invariant top-k SVD basis on context note 1-3grams."""
    mu = lex_ctx.mean(dim=0, keepdim=True)
    xc = lex_ctx - mu
    n, b = xc.shape
    k = min(max_dim, n, b)
    if n <= b:
        evals, evecs = torch.linalg.eigh(xc @ xc.T)
        top_vals = evals[-k:].flip(0)
        top_u = evecs[:, -k:].flip(1)
        valid = top_vals > 1e-6
        inv_s = torch.where(valid, top_vals.clamp_min(1e-6).rsqrt(), torch.zeros_like(top_vals))
        basis = (xc.T @ top_u) * inv_s.unsqueeze(0)
    else:
        evals, evecs = torch.linalg.eigh(xc.T @ xc)
        top_vals = evals[-k:].flip(0)
        top_v = evecs[:, -k:].flip(1)
        basis = top_v * (top_vals > 1e-6).to(top_v.dtype).unsqueeze(0)
    proj = xc @ basis
    rms = proj.square().mean().sqrt().clamp_min(1e-4)
    return mu, basis, rms


@torch.no_grad()
def _project_sentence_maxsim(sent_batch, centroids):
    """Per-patient maximum cosine similarity to each study-specific sentence prototype."""
    n_pat = sent_batch.num_patients
    k = centroids.shape[0]
    device = centroids.device
    if n_pat == 0 or k == 0:
        return torch.zeros((n_pat, k), device=device, dtype=torch.float32)
    if len(sent_batch.flat_idx) == 0 or len(sent_batch.unique_embs) == 0:
        return torch.zeros((n_pat, k), device=device, dtype=torch.float32)

    sim_uniq = sent_batch.unique_embs.float() @ centroids.T  # (U, K)
    vals = sim_uniq[sent_batch.flat_idx]  # (M, K)
    out = torch.full((n_pat, k), -1.0, device=device, dtype=torch.float32)
    idx = sent_batch.patient_ids.unsqueeze(-1).expand(-1, k)
    out.scatter_reduce_(0, idx, vals, reduce="amax", include_self=False)
    has_sent = torch.zeros(n_pat, device=device, dtype=torch.bool)
    has_sent[sent_batch.patient_ids] = True
    return torch.where(has_sent.unsqueeze(-1), out, torch.zeros_like(out))


@torch.no_grad()
def _fit_sentence_concepts(ctx_sent, max_dim, *, max_pool=4096, n_iter=10):
    """Unsupervised, permutation-invariant spherical k-means prototypes on context sentences."""
    if ctx_sent is None or len(ctx_sent.unique_embs) == 0 or max_dim <= 0:
        return None, None, None
    pool = ctx_sent.unique_embs.float()
    if len(pool) > max_pool:
        mean_v = F.normalize(pool.mean(dim=0, keepdim=True), p=2, dim=-1)
        dist = 1.0 - (pool @ mean_v.T).squeeze(-1)
        top = torch.topk(dist, max_pool, largest=True).indices
        pool = pool[top]

    k = min(max_dim, len(pool))
    mean_v = F.normalize(pool.mean(dim=0, keepdim=True), p=2, dim=-1)
    first_idx = int((1.0 - (pool @ mean_v.T).squeeze(-1)).argmax().item())
    centroids = torch.empty((k, pool.shape[1]), device=pool.device, dtype=torch.float32)
    centroids[0] = pool[first_idx]
    max_sim = pool @ centroids[0]
    for ci in range(1, k):
        nxt = int(max_sim.argmin().item())
        centroids[ci] = pool[nxt]
        max_sim = torch.maximum(max_sim, pool @ centroids[ci])

    for _ in range(n_iter):
        sims = pool @ centroids.T
        labels = sims.argmax(dim=1)
        new_c = torch.zeros_like(centroids)
        new_c.index_add_(0, labels, pool)
        norms = new_c.norm(dim=1, keepdim=True)
        empty = norms[:, 0] < 1e-6
        if empty.any():
            n_empty = int(empty.sum().item())
            worst = torch.topk(sims.max(dim=1).values, n_empty, largest=False).indices
            new_c[empty] = pool[worst]
            norms = new_c.norm(dim=1, keepdim=True)
        centroids = new_c / norms.clamp_min(1e-8)

    raw_ctx = _project_sentence_maxsim(ctx_sent, centroids)
    mu = raw_ctx.mean(dim=0, keepdim=True)
    rms = (raw_ctx - mu).square().mean().sqrt().clamp_min(1e-4)
    return centroids, mu, rms


def _fit_loo_cv_ridge(
    A_ctx, y_ctx, lambdas, cv_temp, *, intercept_reg=1e-4, text_mask=None, text_mults=None
):
    """Differentiable batched LOO-CV Ridge via Cholesky (exact, strictly PD, fp32).

    When ``text_mask`` and ``text_mults`` are provided, searches the 2D grid of
    ``(lambda_k, text_mult_g)`` so observational LOO-CV can independently shrink
    high-dimensional text blocks relative to structured features.
    """
    d = A_ctx.shape[1]
    k = len(lambdas)
    G = A_ctx.T @ A_ctx  # (D, D)

    if text_mask is not None and text_mults is not None and len(text_mults) > 0:
        g = len(text_mults)
        m = k * g
        lam_2d = lambdas[:, None]
        mult_2d = text_mults[None, :]
        base_pen = lam_2d.expand(k, g).reshape(m, 1)
        text_pen = (lam_2d * mult_2d).clamp(0.05, 1e8).reshape(m, 1)
        diag_pen = torch.where(text_mask[None, :], text_pen, base_pen)
        diag_pen[:, 0] = intercept_reg
        pen = torch.diag_embed(diag_pen)
    else:
        m = k
        eye = torch.eye(d, device=A_ctx.device, dtype=A_ctx.dtype)
        pen = lambdas[:, None, None] * eye[None]
        pen[:, 0, 0] = intercept_reg

    rhs = (A_ctx.T @ y_ctx).unsqueeze(0).expand(m, -1).unsqueeze(-1)  # (M, D, 1)
    chol = torch.linalg.cholesky(G.unsqueeze(0) + pen)  # (M, D, D)
    w = torch.cholesky_solve(rhs, chol).squeeze(-1)  # (M, D)

    # Exact LOO hat diagonal: h_{m,i} = ||L_m^{-1} a_i||^2 >= 0
    V = torch.linalg.solve_triangular(
        chol, A_ctx.T.unsqueeze(0).expand(m, -1, -1), upper=False
    )  # (M, D, N)
    h = (V * V).sum(dim=1)  # (M, N)
    y_hat = (A_ctx @ w.T).T  # (M, N)
    denom = (1.0 - h).clamp_min(1e-2)
    loo_resid = (y_ctx.unsqueeze(0) - y_hat) / denom
    loo_mse = loo_resid.square().mean(dim=1)  # (M,)
    y_loo_k = y_ctx.unsqueeze(0) - loo_resid

    scale = loo_mse.detach().std(unbiased=False).clamp_min(1e-5)
    alpha = F.softmax(-cv_temp * (loo_mse - loo_mse.min()) / scale, dim=0)  # (M,)
    w_blend = (alpha[:, None] * w).sum(dim=0)  # (D,)
    y_loo = (alpha[:, None] * y_loo_k).sum(dim=0)  # (N,)
    return RidgeFit(weights=w_blend, chol=chol, alpha=alpha), y_loo


def _predict_ridge(fit, A_q):
    """Return query point predictions and Bayesian leverage h_q from a cached RidgeFit."""
    pred = A_q @ fit.weights
    k = fit.chol.shape[0]
    Vq = torch.linalg.solve_triangular(
        fit.chol, A_q.T.unsqueeze(0).expand(k, -1, -1), upper=False
    )  # (K, D, N_q)
    h_q = (fit.alpha[:, None] * (Vq * Vq).sum(dim=1)).sum(dim=0).clamp_min(0.0)
    return pred, h_q


class CausalStudyModel(nn.Module):
    def __init__(self, backbone, tokenizer, cfg, *, include_pros=True):
        super().__init__()
        self.cfg = cfg
        self.include_pros = include_pros
        self.schema_version = CHECKPOINT_FORMAT_VERSION if include_pros else 1
        self.patient = PatientEncoder(backbone, tokenizer, cfg, include_pros=include_pros)

        # Compact solver representation: dedicated note + structured subspaces + cross-modal MLP.
        flat_dim = self.patient.num_slots * cfg.event_width
        sol_dim = cfg.solver_dim
        self._note_sol_dim = sol_dim // 2
        self._struct_sol_dim = sol_dim - self._note_sol_dim
        note_flat_dim = cfg.note_slots * cfg.event_width
        struct_flat_dim = (self.patient.num_slots - cfg.note_slots) * cfg.event_width
        self.solver_note_proj = (
            nn.Linear(note_flat_dim, self._note_sol_dim) if self._note_sol_dim > 0 else None
        )
        self.solver_struct_proj = nn.Linear(struct_flat_dim, self._struct_sol_dim)
        self.solver_mlp = nn.Sequential(
            nn.LayerNorm(flat_dim),
            nn.Linear(flat_dim, 2 * sol_dim),
            nn.GELU(),
            nn.Linear(2 * sol_dim, sol_dim),
        )
        nn.init.normal_(self.solver_mlp[-1].weight, std=0.01)
        nn.init.zeros_(self.solver_mlp[-1].bias)

        # Learnable log-penalty grids for closed-form LOO-CV propensity, outcome, and R-learner.
        k_grid = cfg.ridge_grid_size
        base_grid = torch.linspace(0.5, 0.5 + 1.5 * (k_grid - 1), k_grid)
        tau_grid = torch.linspace(1.0, 1.0 + 1.5 * (k_grid - 1), k_grid)
        self.log_lam_e = nn.Parameter(base_grid.clone())
        self.log_lam_m = nn.Parameter(base_grid.clone())
        self.log_lam_tau = nn.Parameter(tau_grid.clone())
        self.log_cv_temp = nn.Parameter(torch.tensor(1.5))
        if cfg.lexical_dim > 0:
            self.lexical_log_weight = nn.Parameter(torch.tensor(0.0))
        else:
            self.lexical_log_weight = None
        if cfg.sentence_concept_dim > 0:
            self.sentence_log_weight = nn.Parameter(torch.tensor(0.0))
        else:
            self.sentence_log_weight = None
        if cfg.block_ridge:
            # 3-level study-adaptive text-to-structured penalty multipliers: ~1x, ~10x, ~1000x.
            init_mult = torch.tensor([0.0, 2.3, 6.9])
            self.log_text_mult_e = nn.Parameter(init_mult.clone())
            self.log_text_mult_m = nn.Parameter(init_mult.clone())
            self.log_text_mult_tau = nn.Parameter(init_mult.clone())
        else:
            self.log_text_mult_e = None
            self.log_text_mult_m = None
            self.log_text_mult_tau = None

        # Calibration and epistemic-uncertainty heads on solver predictions + query leverages.
        self.arm_cal = nn.Sequential(nn.Linear(8, 32), nn.GELU(), nn.Linear(32, 4))
        self.prop_cal = nn.Sequential(nn.Linear(4, 16), nn.GELU(), nn.Linear(16, 1))
        nn.init.normal_(self.arm_cal[-1].weight, std=0.01)
        nn.init.zeros_(self.arm_cal[-1].bias)
        nn.init.normal_(self.prop_cal[-1].weight, std=0.01)
        nn.init.zeros_(self.prop_cal[-1].bias)

        # Lightweight residual two-way CausalPFN study layers for nonlinear/posterior refinement.
        self.slot_proj = nn.Linear(cfg.event_width, cfg.study_width)
        self.slot_embed = nn.Parameter(
            torch.randn(1, self.patient.num_slots, cfg.study_width) * 0.02
        )
        # Encodes [t, standardized_y, baseline_logit, log1p(n), has_y, has_t].
        self.outcome_token = nn.Linear(6, cfg.study_width)
        self.layers = nn.ModuleList(TwoWayStudyLayer(cfg) for _ in range(cfg.study_layers))
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.study_width), nn.Linear(cfg.study_width, cfg.bins)
        )
        self.propensity_head = nn.Sequential(
            nn.LayerNorm(cfg.study_width), nn.Linear(cfg.study_width, 1)
        )
        self.attn_residual_scale = nn.Parameter(torch.tensor(0.1))
        self.prop_residual_scale = nn.Parameter(torch.tensor(0.1))

        centers = _sinh_grid(cfg.bins)
        spacings = centers[1:] - centers[:-1]
        local_width = torch.cat(
            [spacings[:1], 0.5 * (spacings[:-1] + spacings[1:]), spacings[-1:]]
        )
        self.register_buffer("centers", centers)
        self.register_buffer("bin_widths", local_width)

    def _patient_slots(self, embeddings):
        return self.slot_proj(embeddings) + self.slot_embed

    def _solver_features(self, embeddings):
        n = len(embeddings)
        n_note = self.patient.note_slots
        struct_part = self.solver_struct_proj(embeddings[:, n_note:].reshape(n, -1))
        if self.solver_note_proj is not None:
            note_part = self.solver_note_proj(embeddings[:, :n_note].reshape(n, -1))
            base = torch.cat([note_part, struct_part], dim=-1)
        else:
            base = struct_part
        return base + self.solver_mlp(embeddings.reshape(n, -1))

    def _assemble_context(self, embeddings, t, y, scale):
        arm = t.long()
        standardized = (y - scale.rate[arm]) / scale.sd[arm]
        base_logit = scale.rate.logit()[arm]
        log_n = torch.log1p(scale.counts[arm])
        ones = torch.ones_like(t)
        token_in = torch.stack([t, standardized, base_logit, log_n, ones, ones], dim=-1)
        tok = self.outcome_token(token_in).unsqueeze(1)
        return torch.cat([self._patient_slots(embeddings), tok], dim=1)

    def _assemble_arm_queries(self, embeddings, scale):
        n_q = len(embeddings)
        slots = self._patient_slots(embeddings)[:, None, :, :].expand(-1, 2, -1, -1)
        arms = embeddings.new_tensor([0.0, 1.0]).expand(n_q, 2)
        zeros = torch.zeros_like(arms)
        ones = torch.ones_like(arms)
        base_logit = scale.rate.logit()[None, :].expand(n_q, 2)
        log_n = torch.log1p(scale.counts)[None, :].expand(n_q, 2)
        token_in = torch.stack([arms, zeros, base_logit, log_n, zeros, ones], dim=-1)
        tok = self.outcome_token(token_in).unsqueeze(2)
        return torch.cat([slots, tok], dim=2).flatten(0, 1)

    def _assemble_propensity_queries(self, embeddings, scale):
        n_q = len(embeddings)
        total_n = scale.counts.sum()
        tx_rate = (scale.counts[1] + 0.5) / (total_n + 1.0)
        half = embeddings.new_full((n_q,), 0.5)
        zeros = embeddings.new_zeros((n_q,))
        base_logit = tx_rate.logit().expand(n_q)
        log_n = torch.log1p(total_n).expand(n_q)
        token_in = torch.stack([half, zeros, base_logit, log_n, zeros, zeros], dim=-1)
        tok = self.outcome_token(token_in).unsqueeze(1)
        return torch.cat([self._patient_slots(embeddings), tok], dim=1)

    def _fit_solver(self, embeddings, t, y, *, lexical=None, sentences=None):
        device_type = embeddings.device.type
        with torch.amp.autocast(device_type=device_type, enabled=False):
            phi_ctx = self._solver_features(embeddings).float()
            t_f = t.float()
            y_f = y.float()
            n_ctx = len(phi_ctx)
            mu = phi_ctx.mean(dim=0, keepdim=True)
            sd = phi_ctx.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-4)
            z_ctx = (phi_ctx - mu) / sd
            lex_mu = lex_basis = lex_rms = None
            if self.cfg.lexical_dim > 0 and lexical is not None:
                lex_f = lexical.float()
                lex_mu, lex_basis, lex_rms = _fit_lexical_basis(lex_f, self.cfg.lexical_dim)
                w_lex = F.softplus(self.lexical_log_weight.float())
                z_lex = w_lex * ((lex_f - lex_mu) @ lex_basis) / lex_rms
                z_ctx = torch.cat([z_ctx, z_lex], dim=1)
            sent_centroids = sent_mu = sent_rms = None
            if self.cfg.sentence_concept_dim > 0 and sentences is not None:
                sent_centroids, sent_mu, sent_rms = _fit_sentence_concepts(
                    sentences, self.cfg.sentence_concept_dim
                )
                if sent_centroids is not None:
                    w_sent = F.softplus(self.sentence_log_weight.float())
                    raw_sent = _project_sentence_maxsim(sentences, sent_centroids)
                    z_sent = w_sent * (raw_sent - sent_mu) / sent_rms
                    z_ctx = torch.cat([z_ctx, z_sent], dim=1)
            ones = torch.ones((n_ctx, 1), device=phi_ctx.device, dtype=torch.float32)
            A_ctx = torch.cat([ones, z_ctx], dim=1)
            cv_temp = F.softplus(self.log_cv_temp.float()) + 0.1

            text_mask = None
            mult_e = mult_m = mult_tau = None
            if self.cfg.block_ridge and self.log_text_mult_e is not None:
                text_mask = torch.zeros(A_ctx.shape[1], device=A_ctx.device, dtype=torch.bool)
                if self._note_sol_dim > 0:
                    text_mask[1 : 1 + self._note_sol_dim] = True
                if A_ctx.shape[1] > 1 + self.cfg.solver_dim:
                    text_mask[1 + self.cfg.solver_dim :] = True
                mult_e = self.log_text_mult_e.float().exp().clamp(0.1, 1e5)
                mult_m = self.log_text_mult_m.float().exp().clamp(0.1, 1e5)
                mult_tau = self.log_text_mult_tau.float().exp().clamp(0.1, 1e5)

            lam_e = self.log_lam_e.float().exp().clamp(0.05, 1e5)
            prop_fit, e_loo = _fit_loo_cv_ridge(
                A_ctx, t_f, lam_e, cv_temp, text_mask=text_mask, text_mults=mult_e
            )
            e_loo = e_loo.clamp(0.02, 0.98)

            lam_m = self.log_lam_m.float().exp().clamp(0.05, 1e5)
            out_fit, m_loo = _fit_loo_cv_ridge(
                A_ctx, y_f, lam_m, cv_temp, text_mask=text_mask, text_mults=mult_m
            )

            t_res = t_f - e_loo
            y_res = y_f - m_loo
            B_ctx = t_res.unsqueeze(-1) * A_ctx
            lam_tau = self.log_lam_tau.float().exp().clamp(0.05, 1e5)
            cate_fit, _ = _fit_loo_cv_ridge(
                B_ctx,
                y_res,
                lam_tau,
                cv_temp,
                intercept_reg=0.05,
                text_mask=text_mask,
                text_mults=mult_tau,
            )

            return SolverMemory(
                mu=mu,
                sd=sd,
                propensity_fit=prop_fit,
                outcome_fit=out_fit,
                cate_fit=cate_fit,
                n_ctx=n_ctx,
                lex_mu=lex_mu,
                lex_basis=lex_basis,
                lex_rms=lex_rms,
                sent_centroids=sent_centroids,
                sent_mu=sent_mu,
                sent_rms=sent_rms,
            )

    def _solver_query_stats(self, embeddings, solver_mem, *, lexical=None, sentences=None):
        device_type = embeddings.device.type
        with torch.amp.autocast(device_type=device_type, enabled=False):
            phi_q = self._solver_features(embeddings).float()
            z_q = (phi_q - solver_mem.mu) / solver_mem.sd
            if solver_mem.lex_basis is not None:
                if lexical is None:
                    raise ValueError("Expected query lexical features for lexical_dim > 0")
                w_lex = F.softplus(self.lexical_log_weight.float())
                z_lex_q = (
                    w_lex * ((lexical.float() - solver_mem.lex_mu) @ solver_mem.lex_basis)
                    / solver_mem.lex_rms
                )
                z_q = torch.cat([z_q, z_lex_q], dim=1)
            if solver_mem.sent_centroids is not None:
                if sentences is None:
                    raise ValueError(
                        "Expected query sentence features for sentence_concept_dim > 0"
                    )
                w_sent = F.softplus(self.sentence_log_weight.float())
                raw_sent_q = _project_sentence_maxsim(sentences, solver_mem.sent_centroids)
                z_sent_q = w_sent * (raw_sent_q - solver_mem.sent_mu) / solver_mem.sent_rms
                z_q = torch.cat([z_q, z_sent_q], dim=1)
            ones_q = torch.ones((len(z_q), 1), device=z_q.device, dtype=torch.float32)
            A_q = torch.cat([ones_q, z_q], dim=1)
            e_q, h_e = _predict_ridge(solver_mem.propensity_fit, A_q)
            m_q, h_m = _predict_ridge(solver_mem.outcome_fit, A_q)
            tau_q, h_tau = _predict_ridge(solver_mem.cate_fit, A_q)
            e_prob = e_q.clamp(0.01, 0.99)
            y0_raw = (m_q - e_prob * tau_q).clamp(0.01, 0.99)
            y1_raw = (m_q + (1.0 - e_prob) * tau_q).clamp(0.01, 0.99)
            return y0_raw, y1_raw, m_q, e_prob, tau_q, h_m, h_tau, h_e

    def condition(self, embeddings, t, y, *, lexical=None, sentences=None):
        scale = ArmScale.from_context(t, y)
        solver_mem = self._fit_solver(
            embeddings, t, y, lexical=lexical, sentences=sentences
        )
        context = self._assemble_context(embeddings, t, y, scale)
        attn_memories = []
        for i, layer in enumerate(self.layers):
            context = layer.slot_mix(context)
            memory = layer.sample_norm(context).transpose(0, 1)  # (S, N_ctx, D)
            attn_memories.append(memory)
            if i + 1 < len(self.layers):
                context = layer.sample_mix(context, memory)
        return StudyMemory(solver=solver_mem, attn_memories=attn_memories), scale

    def _attend_queries(self, queries, attn_memories):
        for layer, memory in zip(self.layers, attn_memories, strict=True):
            queries = layer.slot_mix(queries)
            queries = layer.sample_mix(queries, memory)
        return queries[:, -1, :]

    def query_logits(self, embeddings, memories, scale, *, lexical=None, sentences=None):
        y0_raw, y1_raw, m_q, e_prob, tau_q, h_m, h_tau, _ = self._solver_query_stats(
            embeddings, memories.solver, lexical=lexical, sentences=sentences
        )
        device_type = embeddings.device.type
        with torch.amp.autocast(device_type=device_type, enabled=False):
            n_q = len(embeddings)
            log_n = y0_raw.new_full((n_q,), math.log1p(memories.solver.n_ctx))
            arm_feats = torch.stack(
                [y0_raw, y1_raw, m_q, e_prob, tau_q, h_m.sqrt(), h_tau.sqrt(), log_n], dim=-1
            )
            cal = self.arm_cal(arm_feats.to(self.arm_cal[0].weight.dtype)).float()
            delta_shift = cal[:, :2]
            log_sigma = cal[:, 2:]
            base_delta = scale.logit_offsets(torch.stack([y0_raw, y1_raw], dim=1))
            center_delta = (base_delta + delta_shift).clamp(self.centers[0], self.centers[-1])
            sigma = F.softplus(log_sigma + 0.5) + 0.15
            solver_bin_logits = -0.5 * (
                (self.centers[None, None, :] - center_delta[..., None]) / sigma[..., None]
            ).square()

        queries = self._assemble_arm_queries(embeddings, scale)
        pooled = self._attend_queries(queries, memories.attn_memories)
        attn_logits = self.head(pooled).reshape(len(embeddings), 2, -1).float()
        return solver_bin_logits + self.attn_residual_scale.float() * attn_logits

    def query_propensity_logits(
        self, embeddings, memories, scale, *, lexical=None, sentences=None
    ):
        _, _, _, e_prob, _, _, _, h_e = self._solver_query_stats(
            embeddings, memories.solver, lexical=lexical, sentences=sentences
        )
        device_type = embeddings.device.type
        with torch.amp.autocast(device_type=device_type, enabled=False):
            n_q = len(embeddings)
            log_n = e_prob.new_full((n_q,), math.log1p(memories.solver.n_ctx))
            e_logit = e_prob.logit()
            prop_feats = torch.stack([e_prob, e_logit, h_e.sqrt(), log_n], dim=-1)
            prop_shift = self.prop_cal(prop_feats.to(self.prop_cal[0].weight.dtype)).float().squeeze(
                -1
            )
            base_prop_logit = e_logit + prop_shift

        queries = self._assemble_propensity_queries(embeddings, scale)
        pooled = self._attend_queries(queries, memories.attn_memories)
        attn_prop = self.propensity_head(pooled).squeeze(-1).float()
        return base_prop_logit + self.prop_residual_scale.float() * attn_prop

    def forward(
        self, context, queries, *, note_grad_patients=None, freeze_note_backbone=False
    ):
        if not queries:
            raise ValueError("Expected at least one query patient")
        n_ctx = len(context)
        n_q = len(queries)
        backbone_grad_mask = None
        if freeze_note_backbone:
            backbone_grad_mask = [False] * (n_ctx + n_q)
        elif (
            note_grad_patients is not None
            and self.training
            and torch.is_grad_enabled()
            and n_ctx > note_grad_patients
        ):
            if note_grad_patients < 0:
                raise ValueError("note_grad_patients must be nonnegative")
            ctx_mask = [False] * n_ctx
            if note_grad_patients > 0:
                for idx in torch.randperm(n_ctx)[:note_grad_patients].tolist():
                    ctx_mask[idx] = True
            backbone_grad_mask = ctx_mask + [True] * n_q
        t, y = binary_observations(context, self.centers.device)
        all_rows = [*context, *queries]
        embeddings = self.patient(all_rows, backbone_grad_mask=backbone_grad_mask)
        lex = self.patient.lexical_features(all_rows) if self.cfg.lexical_dim > 0 else None
        ctx_lex = lex[:n_ctx] if lex is not None else None
        q_lex = lex[n_ctx:] if lex is not None else None
        ctx_sent = (
            self.patient.sentence_features(context)
            if self.cfg.sentence_concept_dim > 0
            else None
        )
        q_sent = (
            self.patient.sentence_features(queries)
            if self.cfg.sentence_concept_dim > 0
            else None
        )
        memories, scale = self.condition(
            embeddings[:n_ctx], t, y, lexical=ctx_lex, sentences=ctx_sent
        )
        q_emb = embeddings[n_ctx:]
        outcome_logits = self.query_logits(
            q_emb, memories, scale, lexical=q_lex, sentences=q_sent
        )
        prop_logits = self.query_propensity_logits(
            q_emb, memories, scale, lexical=q_lex, sentences=q_sent
        )
        return outcome_logits, prop_logits, scale

    def risk_stats(self, logits, scale):
        """Return posterior predictive ``(mean, sd)`` on the probability scale."""
        probs = logits.float().softmax(dim=-1)
        values = scale.bin_risks(self.centers).float()
        mean = (probs * values).sum(dim=-1)
        var = (probs * (values - mean[..., None]).square()).sum(dim=-1).clamp_min(0.0)
        return mean, var.sqrt()

    def risks(self, logits, scale):
        mean, _ = self.risk_stats(logits, scale)
        return mean

    def loss_components(self, logits, scale, oracle, *, propensity_logits=None, true_propensity=None):
        """Histogram NLL on arm risks + direct CATE MSE + optional propensity BCE."""
        logits = logits.float()
        oracle = oracle.float()
        if not torch.isfinite(oracle).all() or ((oracle < 0) | (oracle > 1)).any():
            raise ValueError("Oracle targets must be probabilities")
        delta = scale.logit_offsets(oracle).clamp(self.centers[0], self.centers[-1])
        # Use local bin width around each target on the non-uniform sinh grid.
        nearest = (self.centers - delta[..., None]).abs().argmin(dim=-1)
        sigma = self.bin_widths[nearest].unsqueeze(-1)
        targets = (-0.5 * ((self.centers - delta[..., None]) / sigma).square()).softmax(dim=-1)
        outcome_loss = -(targets * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()

        pred_risks = self.risks(logits, scale)
        pred_ite = pred_risks[:, 1] - pred_risks[:, 0]
        true_ite = oracle[:, 1] - oracle[:, 0]
        cate_loss = (pred_ite - true_ite).square().mean()

        total = outcome_loss + self.cfg.cate_loss_weight * cate_loss
        prop_loss = logits.new_zeros(())
        if propensity_logits is not None and true_propensity is not None:
            propensity_logits = propensity_logits.float()
            true_propensity = true_propensity.float()
            if (
                not torch.isfinite(true_propensity).all()
                or ((true_propensity < 0) | (true_propensity > 1)).any()
            ):
                raise ValueError("Oracle propensities must be probabilities")
            prop_loss = F.binary_cross_entropy_with_logits(propensity_logits, true_propensity)
            total = total + self.cfg.propensity_loss_weight * prop_loss
        return total, dict(
            outcome_loss=outcome_loss.detach(),
            cate_loss=cate_loss.detach(),
            propensity_loss=prop_loss.detach(),
        )

    def loss(self, logits, scale, oracle, *, propensity_logits=None, true_propensity=None):
        total, _ = self.loss_components(
            logits,
            scale,
            oracle,
            propensity_logits=propensity_logits,
            true_propensity=true_propensity,
        )
        return total
