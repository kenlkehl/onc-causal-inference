"""A longitudinal patient encoder and a context-conditioned potential-outcome head."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.checkpoint import checkpoint

from oci.models.lossless_tokenization import SemanticTruncationError
from synthetic_data.longitudinal_history import LEGACY_STREAMS, STREAMS
from .events import (
    CODE_BUCKETS,
    LAB_NAMES,
    PANELS,
    PRO_INPUT_SIZE,
    VITAL_SCALES,
    binary_observations,
    encounter_codes,
    event_times,
    lab_vector,
    pro_vector,
    vital_vector,
)

BASE_MODEL = "LiquidAI/LFM2.5-230M-Base"
BASE_REVISION = "9d2be5519834990d30996f878b6771cccbd24f2c"


@dataclass(frozen=True)
class ModelConfig:
    event_width: int = 64
    event_layers: int = 2
    study_width: int = 128
    study_layers: int = 4
    heads: int = 4
    bins: int = 129
    note_batch_size: int = 8

    def __post_init__(self):
        if (
            min(
                self.event_width,
                self.study_width,
                self.heads,
                self.event_layers,
                self.study_layers,
                self.note_batch_size,
            )
            < 1
            or self.bins < 3
        ):
            raise ValueError("Invalid model dimensions or note batch size")
        if self.event_width % self.heads or self.study_width % self.heads:
            raise ValueError("Both transformer widths must be divisible by heads")


class EventTransformer(nn.Module):
    """A learned summary token can also represent an empty stream."""

    def __init__(self, cfg):
        super().__init__()
        width = cfg.event_width
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

    def forward(self, values, times, lengths):
        times = values.new_tensor(times).reshape(-1, 3)
        events = values + self.clock(times)
        sequences = [torch.cat([self.summary, s]) for s in events.split(lengths)]
        tokens = pad_sequence(sequences, batch_first=True)
        lengths = torch.tensor(lengths, device=tokens.device)
        mask = torch.arange(tokens.shape[1], device=tokens.device)[None, :] > lengths[:, None]
        for layer in self.layers:
            tokens = layer(tokens, src_key_padding_mask=mask)
        return self.norm(tokens[:, 0])


class PatientEncoder(nn.Module):
    def __init__(self, backbone, tokenizer, cfg, *, include_pros=True):
        super().__init__()
        self.backbone, self.tokenizer = backbone, tokenizer
        self.tokenizer.padding_side = "right"
        self.note_batch_size = cfg.note_batch_size
        width = cfg.event_width
        self.notes = nn.Linear(backbone.config.hidden_size, width)
        self.labs = nn.Linear(2 * len(LAB_NAMES) + len(PANELS), width)
        self.vitals = nn.Linear(2 * len(VITAL_SCALES), width)
        self.codes = nn.EmbeddingBag(CODE_BUCKETS, width, mode="sum")
        if include_pros:
            self.pros = nn.Linear(PRO_INPUT_SIZE, width)
        names = STREAMS if include_pros else LEGACY_STREAMS
        self.streams = nn.ModuleDict({name: EventTransformer(cfg) for name in names})

    def _last_token(self, ids, mask):
        hidden = self.backbone(
            input_ids=ids, attention_mask=mask, use_cache=False
        ).last_hidden_state
        last = mask.sum(dim=1) - 1
        return hidden[torch.arange(len(ids), device=ids.device), last].to(self.notes.weight.dtype)

    def _note_vectors(self, events):
        device = self.notes.weight.device
        ids = self.tokenizer(
            [e["text"] for e in events], truncation=False, return_attention_mask=False
        )["input_ids"]
        limit = self.backbone.config.max_position_embeddings
        if any(len(seq) > limit or not seq for seq in ids):
            raise SemanticTruncationError(f"Expected 1 to {limit} tokens in every clinical note")
        vectors = []
        for start in range(0, len(ids), self.note_batch_size):
            batch = self.tokenizer.pad(
                {"input_ids": ids[start : start + self.note_batch_size]},
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            ).to(device)
            args = batch["input_ids"], batch["attention_mask"]
            if self.training and torch.is_grad_enabled():
                last = checkpoint(self._last_token, *args, use_reentrant=False)
            else:
                last = self._last_token(*args)
            vectors.append(self.notes(last))
        return torch.cat(vectors)

    def _values(self, name, events):
        if not events:
            return self.notes.weight.new_empty(0, self.notes.out_features)
        if name == "notes":
            return self._note_vectors(events)
        if name == "encounters":
            codes = [encounter_codes(e) for e in events]
            device = self.codes.weight.device
            ids = torch.tensor([i for bag in codes for i in bag], device=device)
            offsets = torch.tensor([0] + [len(bag) for bag in codes[:-1]], device=device).cumsum(0)
            return self.codes(ids, offsets)
        vector = {"labs": lab_vector, "vitals": vital_vector, "pros": pro_vector}[name]
        layer = getattr(self, name)
        return layer(layer.weight.new_tensor([vector(e) for e in events]))

    def forward(self, rows):
        if not rows:
            raise ValueError("Expected at least one patient")
        summaries = []
        for name, encoder in self.streams.items():
            streams = [
                [] if name == "pros" and row.get(name) is None else row[name] for row in rows
            ]
            times = [time for stream in streams for time in event_times(stream)]
            values = self._values(name, [event for stream in streams for event in stream])
            summaries.append(encoder(values, times, [len(s) for s in streams]))
        return torch.cat(summaries, dim=-1)


class StudyLayer(nn.Module):
    """Every patient attends only to observed context patients."""

    def __init__(self, cfg):
        super().__init__()
        d = cfg.study_width
        self.norm = nn.LayerNorm(d)
        self.attention = nn.MultiheadAttention(d, cfg.heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)
        )

    def forward(self, patients, memory):
        update = self.attention(self.norm(patients), memory, memory, need_weights=False)[0]
        patients = patients + update
        return patients + self.ff(patients)


@dataclass
class ArmScale:
    rate: torch.Tensor
    sd: torch.Tensor

    @classmethod
    def from_context(cls, t, y):
        if t.ndim != 1 or t.shape != y.shape or not len(t):
            raise ValueError("Expected aligned nonempty context observations")
        arms = torch.arange(2, device=t.device)
        mask = t[:, None] == arms
        rates = ((mask * y[:, None]).sum(0) + 0.5) / (mask.sum(0) + 1.0)
        return cls(rates, (rates * (1.0 - rates)).sqrt())

    def standardized_logits(self, risks):
        return (torch.logit(risks, eps=1e-6) - self.rate.logit()) * self.sd

    def bin_risks(self, centers):
        return (self.rate.logit()[:, None] + centers / self.sd[:, None]).sigmoid()


class CausalStudyModel(nn.Module):
    def __init__(self, backbone, tokenizer, cfg, *, include_pros=True):
        super().__init__()
        self.cfg = cfg
        self.schema_version = 2 if include_pros else 1
        self.patient = PatientEncoder(backbone, tokenizer, cfg, include_pros=include_pros)
        self.row = nn.Linear(len(self.patient.streams) * cfg.event_width + 1, cfg.study_width)
        self.observed_y = nn.Linear(1, cfg.study_width)
        self.layers = nn.ModuleList(StudyLayer(cfg) for _ in range(cfg.study_layers))
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.study_width), nn.Linear(cfg.study_width, cfg.bins)
        )
        self.register_buffer("centers", torch.linspace(-6, 6, cfg.bins))

    def forward(self, context, queries):
        if not queries:
            raise ValueError("Expected at least one query patient")
        t, y = binary_observations(context, self.centers.device)
        embeddings = self.patient([*context, *queries])
        memories, scale = self.condition(embeddings[: len(context)], t, y)
        return self.query_logits(embeddings[len(context) :], memories), scale

    def condition(self, embeddings, t, y):
        scale = ArmScale.from_context(t, y)
        arm = t.long()
        standardized = (y - scale.rate[arm]) / scale.sd[arm]
        context = self.row(torch.cat([embeddings, t[:, None]], dim=1))
        context = context + self.observed_y(standardized[:, None])
        memories = []
        for i, layer in enumerate(self.layers):
            memory = layer.norm(context)
            memories.append(memory)
            # Only the next layer consumes updated context representations.
            if i + 1 < len(self.layers):
                context = layer(context, memory)
        return memories, scale

    def query_logits(self, embeddings, memories):
        arms = embeddings.new_tensor([0.0, 1.0]).expand(len(embeddings), 2)
        both = torch.cat([embeddings[:, None, :].expand(-1, 2, -1), arms[..., None]], dim=-1)
        queries = self.row(both.flatten(0, 1))
        for layer, memory in zip(self.layers, memories, strict=True):
            queries = layer(queries, memory)
        return self.head(queries).reshape(len(embeddings), 2, -1)

    def risks(self, logits, scale):
        return (logits.softmax(dim=-1) * scale.bin_risks(self.centers)).sum(dim=-1)

    def loss(self, logits, scale, oracle):
        # Histogram cross-entropy for both known conditional risks per simulated patient.
        if not torch.isfinite(oracle).all() or ((oracle < 0) | (oracle > 1)).any():
            raise ValueError("Oracle targets must be probabilities")
        z = scale.standardized_logits(oracle).clamp(self.centers[0], self.centers[-1])
        width = self.centers[1] - self.centers[0]
        targets = (-0.5 * ((self.centers - z[..., None]) / width).square()).softmax(dim=-1)
        return -(targets * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
