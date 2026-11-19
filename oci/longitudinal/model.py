"""A cross-modal longitudinal patient encoder and two-way CausalPFN study model."""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.checkpoint import checkpoint

from oci.models.lossless_tokenization import SemanticTruncationError
from .events import (
    CODE_BUCKETS,
    EXTRA_LAB_BUCKETS,
    EXTRA_PRO_BUCKETS,
    LAB_INPUT_SIZE,
    LEGACY_STREAMS,
    PRO_INPUT_SIZE,
    STREAMS,
    VITAL_INPUT_SIZE,
    binary_observations,
    encounter_codes,
    event_times,
    lab_features,
    pro_features,
    vital_vector,
)

BASE_MODEL = "LiquidAI/LFM2.5-230M-Base"
BASE_REVISION = "9d2be5519834990d30996f878b6771cccbd24f2c"
CHECKPOINT_FORMAT_VERSION = 3


@dataclass(frozen=True)
class ModelConfig:
    event_width: int = 128
    event_layers: int = 2
    study_width: int = 192
    study_layers: int = 4
    heads: int = 4
    bins: int = 129
    note_batch_size: int = 8
    note_slots: int = 4
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
            )
            < 1
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


class PatientEncoder(nn.Module):
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
        width = cfg.event_width
        self.notes = nn.Linear(backbone.config.hidden_size, width)
        self.labs = nn.Linear(LAB_INPUT_SIZE, width)
        self.extra_labs = nn.Embedding(EXTRA_LAB_BUCKETS, width)
        self.extra_lab_scale = nn.Linear(1, width)
        self.vitals = nn.Linear(VITAL_INPUT_SIZE, width)
        # Mean pooling plus explicit log-count keeps encounter norms stable on real EHRs.
        self.codes = nn.EmbeddingBag(CODE_BUCKETS, width, mode="mean")
        self.code_count = nn.Linear(1, width)
        if include_pros:
            self.pros = nn.Linear(PRO_INPUT_SIZE, width)
            self.extra_pros = nn.Embedding(EXTRA_PRO_BUCKETS, width)
            self.extra_pro_scale = nn.Linear(1, width)
        names = STREAMS if include_pros else LEGACY_STREAMS
        self.streams = nn.ModuleDict(
            {
                name: EventTransformer(cfg, num_queries=cfg.note_slots if name == "notes" else 1)
                for name in names
            }
        )
        self.cross_modal = CrossModalTimeline(cfg, len(names))
        self.num_slots = cfg.note_slots + (len(names) - 1) + 1

    def _last_token(self, ids, mask):
        hidden = self.backbone(
            input_ids=ids, attention_mask=mask, use_cache=False
        ).last_hidden_state
        last = mask.sum(dim=1) - 1
        return hidden[torch.arange(len(ids), device=ids.device), last].to(self.notes.weight.dtype)

    def _encode_note_subset(self, ids, indices, *, with_grad):
        device = self.notes.weight.device
        batch_size = self.note_batch_size if with_grad else max(self.note_batch_size * 4, 32)
        chunks = []
        ctx = torch.enable_grad() if with_grad else torch.no_grad()
        use_ckpt = with_grad and self.training and torch.is_grad_enabled()
        with ctx:
            for start in range(0, len(indices), batch_size):
                sub_ids = [ids[i] for i in indices[start : start + batch_size]]
                batch = self.tokenizer.pad(
                    {"input_ids": sub_ids},
                    padding=True,
                    return_tensors="pt",
                    return_attention_mask=True,
                ).to(device)
                args = batch["input_ids"], batch["attention_mask"]
                if use_ckpt:
                    last = checkpoint(self._last_token, *args, use_reentrant=False)
                else:
                    last = self._last_token(*args)
                chunks.append(last)
        return torch.cat(chunks, dim=0)

    def _note_vectors(self, events, grad_flags=None):
        device = self.notes.weight.device
        ids = self.tokenizer(
            [e["text"] for e in events], truncation=False, return_attention_mask=False
        )["input_ids"]
        limit = self.backbone.config.max_position_embeddings
        if any(len(seq) > limit or not seq for seq in ids):
            raise SemanticTruncationError(f"Expected 1 to {limit} tokens in every clinical note")

        if grad_flags is None or not (self.training and torch.is_grad_enabled()):
            with_grad = self.training and torch.is_grad_enabled()
            last = self._encode_note_subset(ids, list(range(len(ids))), with_grad=with_grad)
            return self.notes(last)

        grad_idx = [i for i, flag in enumerate(grad_flags) if flag]
        nograd_idx = [i for i, flag in enumerate(grad_flags) if not flag]
        if not nograd_idx:
            last = self._encode_note_subset(ids, grad_idx, with_grad=True)
            return self.notes(last)
        if not grad_idx:
            last = self._encode_note_subset(ids, nograd_idx, with_grad=False)
            return self.notes(last)

        grad_last = self._encode_note_subset(ids, grad_idx, with_grad=True)
        nograd_last = self._encode_note_subset(ids, nograd_idx, with_grad=False)
        combined = grad_last.new_empty((len(ids), grad_last.shape[-1]))
        combined[torch.tensor(grad_idx, device=device, dtype=torch.long)] = grad_last
        combined[torch.tensor(nograd_idx, device=device, dtype=torch.long)] = nograd_last
        return self.notes(combined)

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

    def _values(self, name, events, *, note_grad_flags=None):
        if not events:
            return self.notes.weight.new_empty(0, self.notes.out_features)
        if name == "notes":
            return self._note_vectors(events, grad_flags=note_grad_flags)
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
            note_grad_flags = None
            if name == "notes" and backbone_grad_mask is not None:
                note_grad_flags = [
                    flag
                    for flag, length in zip(backbone_grad_mask, lengths, strict=True)
                    for _ in range(length)
                ]
            values = self._values(name, flat_events, note_grad_flags=note_grad_flags)
            stream_summaries.append(encoder(values, times, lengths))

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


class CausalStudyModel(nn.Module):
    def __init__(self, backbone, tokenizer, cfg, *, include_pros=True):
        super().__init__()
        self.cfg = cfg
        self.include_pros = include_pros
        self.schema_version = CHECKPOINT_FORMAT_VERSION if include_pros else 1
        self.patient = PatientEncoder(backbone, tokenizer, cfg, include_pros=include_pros)
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
        centers = _sinh_grid(cfg.bins)
        spacings = centers[1:] - centers[:-1]
        local_width = torch.cat(
            [spacings[:1], 0.5 * (spacings[:-1] + spacings[1:]), spacings[-1:]]
        )
        self.register_buffer("centers", centers)
        self.register_buffer("bin_widths", local_width)

    def _patient_slots(self, embeddings):
        return self.slot_proj(embeddings) + self.slot_embed

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

    def condition(self, embeddings, t, y):
        scale = ArmScale.from_context(t, y)
        context = self._assemble_context(embeddings, t, y, scale)
        memories = []
        for i, layer in enumerate(self.layers):
            context = layer.slot_mix(context)
            memory = layer.sample_norm(context).transpose(0, 1)  # (S, N_ctx, D)
            memories.append(memory)
            if i + 1 < len(self.layers):
                context = layer.sample_mix(context, memory)
        return memories, scale

    def _attend_queries(self, queries, memories):
        for layer, memory in zip(self.layers, memories, strict=True):
            queries = layer.slot_mix(queries)
            queries = layer.sample_mix(queries, memory)
        return queries[:, -1, :]

    def query_logits(self, embeddings, memories, scale):
        queries = self._assemble_arm_queries(embeddings, scale)
        pooled = self._attend_queries(queries, memories)
        return self.head(pooled).reshape(len(embeddings), 2, -1)

    def query_propensity_logits(self, embeddings, memories, scale):
        queries = self._assemble_propensity_queries(embeddings, scale)
        pooled = self._attend_queries(queries, memories)
        return self.propensity_head(pooled).squeeze(-1)

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
        embeddings = self.patient([*context, *queries], backbone_grad_mask=backbone_grad_mask)
        memories, scale = self.condition(embeddings[:n_ctx], t, y)
        q_emb = embeddings[n_ctx:]
        outcome_logits = self.query_logits(q_emb, memories, scale)
        prop_logits = self.query_propensity_logits(q_emb, memories, scale)
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
