"""Neural mention-slot features discovered from raw clinical text.

The extractor is intentionally concept-agnostic. It first parses generic
label/value-like mentions from notes, embeds those mention strings with a
frozen encoder, then fits unsupervised slots over train-fold mention
embeddings. Patient-level X features are continuous soft slot activations and
slot-specific numeric summaries.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler


_INLINE_KV_RE = re.compile(
    r"(?:\*\*)?([A-Z][A-Za-z0-9 /_\-()]{1,56})(?:\*\*)?\s*:\s*"
    r"([^:|\n]{1,180})(?=(?:\s+(?:\*\*)?[A-Z][A-Za-z0-9 /_\-()]{1,56}"
    r"(?:\*\*)?\s*:)|$)"
)
_LINE_KV_RE = re.compile(
    r"(?m)^\s*(?:[-*]\s*)?(?:\*\*)?([A-Za-z][A-Za-z0-9 /_\-()]{1,80})"
    r"(?:\*\*)?\s*[:]\s*(.{1,260})$"
)
_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
_SPACE_RE = re.compile(r"\s+")


@dataclass
class MentionRecord:
    """A generic mention candidate parsed from one patient note."""

    patient_index: int
    label: str
    value: str
    source: str

    @property
    def encoder_text(self) -> str:
        return f"label: {self.label}\nvalue: {self.value}"


@dataclass
class NeuralMentionSlotConfig:
    """Configuration for raw-text neural mention slots."""

    n_slots: int = 64
    assignment_temperature: float = 0.35
    top_assignments: int = 4
    max_mentions_per_patient: int = 160
    min_label_chars: int = 2
    max_label_chars: int = 80
    max_value_chars: int = 260
    include_numeric_features: bool = True
    value_feature_dim: int = 0
    random_state: int = 0


def _normalize_text(text: object) -> str:
    value = unicodedata.normalize("NFKC", str(text))
    value = value.replace("**", " ")
    value = value.replace("‑", "-").replace("–", "-").replace("—", "-")
    return _SPACE_RE.sub(" ", value).strip()


def _clean_label(text: object) -> str:
    value = _normalize_text(text)
    value = re.sub(r"^[#*\-\s]+", "", value)
    value = re.sub(r"[*\s]+$", "", value)
    return value.strip()


def _clean_value(text: object, max_chars: int) -> str:
    value = _normalize_text(text)
    value = re.sub(r"^[#*\-\s]+", "", value)
    value = value.strip()
    if len(value) > max_chars:
        value = value[:max_chars].rsplit(" ", 1)[0]
    return value


def _first_number(text: object) -> Optional[float]:
    match = _NUMBER_RE.search(unicodedata.normalize("NFKC", str(text)).replace("−", "-"))
    if not match:
        return None
    try:
        value = float(match.group())
    except ValueError:
        return None
    if not np.isfinite(value) or abs(value) > 100000:
        return None
    return value


def _mention_quality(label: str, value: str) -> Tuple[int, int, int]:
    has_number = int(_first_number(value) is not None)
    value_alpha = int(bool(re.search(r"[A-Za-z]", value)))
    label_len = min(len(label), 80)
    return has_number + value_alpha, label_len, min(len(value), 260)


def extract_mention_records(
    texts: Sequence[str],
    max_mentions_per_patient: int = 160,
    min_label_chars: int = 2,
    max_label_chars: int = 80,
    max_value_chars: int = 260,
) -> List[MentionRecord]:
    """Extract generic label/value-like mention records from raw text."""
    records: List[MentionRecord] = []
    seen: set[Tuple[int, str, str]] = set()

    for patient_index, raw_text in enumerate(texts):
        text = str(raw_text)
        patient_records: List[MentionRecord] = []

        for line in text.splitlines():
            if "|" in line:
                cells = [
                    _clean_value(cell, max_value_chars)
                    for cell in line.split("|")
                    if _clean_value(cell, max_value_chars).strip(" -*")
                ]
                if len(cells) >= 2:
                    label = _clean_label(cells[0])
                    value = _clean_value(cells[1], max_value_chars)
                    patient_records.append(
                        MentionRecord(patient_index, label, value, "table")
                    )

            for match in _INLINE_KV_RE.finditer(line):
                patient_records.append(
                    MentionRecord(
                        patient_index,
                        _clean_label(match.group(1)),
                        _clean_value(match.group(2), max_value_chars),
                        "inline",
                    )
                )

        for match in _LINE_KV_RE.finditer(text):
            patient_records.append(
                MentionRecord(
                    patient_index,
                    _clean_label(match.group(1)),
                    _clean_value(match.group(2), max_value_chars),
                    "line",
                )
            )

        filtered: List[MentionRecord] = []
        for record in patient_records:
            label = record.label
            value = record.value
            if not (min_label_chars <= len(label) <= max_label_chars):
                continue
            if not value or not re.search(r"[A-Za-z0-9]", value):
                continue
            if set(label) <= {"-", " ", "|"}:
                continue
            key = (record.patient_index, label.lower(), value.lower())
            if key in seen:
                continue
            seen.add(key)
            filtered.append(record)

        filtered.sort(
            key=lambda record: _mention_quality(record.label, record.value),
            reverse=True,
        )
        records.extend(filtered[:max_mentions_per_patient])

    return records


def _cache_path(cache_dir: Path, model_name: str, texts: Sequence[str]) -> Path:
    digest = hashlib.sha256()
    digest.update(model_name.encode("utf-8"))
    for text in texts:
        digest.update(b"\0")
        digest.update(text.encode("utf-8", errors="ignore"))
    return cache_dir / f"mention_embeddings_{digest.hexdigest()[:24]}.npz"


@torch.no_grad()
def embed_mention_texts(
    texts: Sequence[str],
    model_name: str,
    batch_size: int = 64,
    max_length: int = 128,
    device: str = "cuda:0",
    cache_dir: Optional[str | Path] = None,
) -> np.ndarray:
    """Embed mention strings with a frozen HuggingFace model."""
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)

    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = _cache_path(cache_dir, model_name, texts)
        if path.exists():
            return np.load(path)["embeddings"].astype(np.float32)
    else:
        path = None

    from transformers import AutoModel, AutoTokenizer

    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        torch_device = torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if torch_device.type == "cuda" else torch.float32
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    model.to(torch_device)
    model.eval()

    outputs: List[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = list(texts[start: start + batch_size])
        batch = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batch = {key: value.to(torch_device) for key, value in batch.items()}
        hidden = model(**batch).last_hidden_state.float()
        mask = batch["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        outputs.append(pooled.cpu().numpy().astype(np.float32))

    embeddings = np.concatenate(outputs, axis=0).astype(np.float32)
    if path is not None:
        np.savez_compressed(
            path,
            embeddings=embeddings,
            meta=json.dumps({"model_name": model_name, "count": len(texts)}),
        )
    return embeddings


class NeuralMentionSlotExtractor:
    """Fit unsupervised soft slots and emit patient-level X features."""

    def __init__(self, config: Optional[NeuralMentionSlotConfig] = None):
        self.config = config or NeuralMentionSlotConfig()
        self.kmeans_: Optional[MiniBatchKMeans] = None
        self.numeric_scaler_: Optional[StandardScaler] = None
        self.value_svd_: Optional[TruncatedSVD] = None
        self.value_feature_dim_: int = 0
        self.n_features_: Optional[int] = None

    def fit(
        self,
        records: Sequence[MentionRecord],
        embeddings: np.ndarray,
        patient_indices: Iterable[int],
        value_embeddings: Optional[np.ndarray] = None,
    ) -> "NeuralMentionSlotExtractor":
        if len(records) != len(embeddings):
            raise ValueError("records and embeddings must have the same length")
        if len(records) == 0:
            raise ValueError("Cannot fit mention slots with no records")

        n_slots = min(int(self.config.n_slots), len(records))
        self.kmeans_ = MiniBatchKMeans(
            n_clusters=n_slots,
            random_state=self.config.random_state,
            batch_size=min(4096, max(256, len(records))),
            n_init="auto",
        )
        self.kmeans_.fit(embeddings)

        if self.config.include_numeric_features:
            values = [
                _first_number(record.value)
                for record in records
                if _first_number(record.value) is not None
            ]
            if values:
                self.numeric_scaler_ = StandardScaler().fit(
                    np.asarray(values, dtype=np.float32).reshape(-1, 1)
                )
            else:
                self.numeric_scaler_ = None

        self.value_svd_ = None
        self.value_feature_dim_ = 0
        if value_embeddings is not None and self.config.value_feature_dim > 0:
            if len(value_embeddings) != len(records):
                raise ValueError("value_embeddings must match records length")
            n_value_components = min(
                int(self.config.value_feature_dim),
                value_embeddings.shape[1] - 1,
                len(records) - 2,
            )
            if n_value_components >= 1:
                self.value_svd_ = TruncatedSVD(
                    n_components=n_value_components,
                    random_state=self.config.random_state,
                )
                self.value_svd_.fit(value_embeddings)
                self.value_feature_dim_ = int(n_value_components)

        feature_block_count = 3 if self.config.include_numeric_features else 2
        self.n_features_ = n_slots * feature_block_count
        if self.value_feature_dim_:
            self.n_features_ += n_slots * self.value_feature_dim_
        return self

    def _soft_assign(self, embeddings: np.ndarray) -> np.ndarray:
        if self.kmeans_ is None:
            raise RuntimeError("Extractor must be fit before transform")
        centers = self.kmeans_.cluster_centers_.astype(np.float32)
        distances = (
            np.sum(embeddings ** 2, axis=1, keepdims=True)
            - 2.0 * embeddings @ centers.T
            + np.sum(centers ** 2, axis=1)
        )
        distances = np.maximum(distances, 0.0)
        logits = -distances / max(float(self.config.assignment_temperature), 1e-4)

        top_k = min(int(self.config.top_assignments), logits.shape[1])
        if top_k < logits.shape[1]:
            keep = np.argpartition(-logits, kth=top_k - 1, axis=1)[:, :top_k]
            sparse_logits = np.full_like(logits, -np.inf, dtype=np.float32)
            rows = np.arange(logits.shape[0])[:, None]
            sparse_logits[rows, keep] = logits[rows, keep]
            logits = sparse_logits

        logits = logits - np.max(logits, axis=1, keepdims=True)
        weights = np.exp(logits)
        weights /= np.sum(weights, axis=1, keepdims=True).clip(min=1e-12)
        return weights.astype(np.float32)

    def transform(
        self,
        records: Sequence[MentionRecord],
        embeddings: np.ndarray,
        patient_indices: Sequence[int],
        value_embeddings: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if self.kmeans_ is None or self.n_features_ is None:
            raise RuntimeError("Extractor must be fit before transform")
        if len(records) != len(embeddings):
            raise ValueError("records and embeddings must have the same length")

        patient_indices = list(patient_indices)
        patient_to_row = {int(patient): row for row, patient in enumerate(patient_indices)}
        n_slots = int(self.kmeans_.n_clusters)
        presence = np.zeros((len(patient_indices), n_slots), dtype=np.float32)
        log_count = np.zeros((len(patient_indices), n_slots), dtype=np.float32)
        numeric_sum = np.zeros((len(patient_indices), n_slots), dtype=np.float32)
        numeric_weight = np.zeros((len(patient_indices), n_slots), dtype=np.float32)
        value_sum = None
        value_weight = None
        value_projected = None
        if self.value_svd_ is not None and value_embeddings is not None:
            if len(value_embeddings) != len(records):
                raise ValueError("value_embeddings must match records length")
            value_projected = self.value_svd_.transform(value_embeddings).astype(np.float32)
            value_sum = np.zeros(
                (len(patient_indices), n_slots, self.value_feature_dim_),
                dtype=np.float32,
            )
            value_weight = np.zeros((len(patient_indices), n_slots), dtype=np.float32)

        if len(records) > 0:
            weights = self._soft_assign(embeddings)
            for mention_index, record in enumerate(records):
                row = patient_to_row.get(int(record.patient_index))
                if row is None:
                    continue
                assignment = weights[mention_index]
                presence[row] = np.maximum(presence[row], assignment)
                log_count[row] += assignment

                number = _first_number(record.value)
                if (
                    self.config.include_numeric_features
                    and number is not None
                    and self.numeric_scaler_ is not None
                ):
                    scaled = float(
                        self.numeric_scaler_.transform([[number]])[0, 0]
                    )
                    numeric_sum[row] += assignment * scaled
                    numeric_weight[row] += assignment

                if value_projected is not None and value_sum is not None and value_weight is not None:
                    value_sum[row] += assignment[:, None] * value_projected[mention_index]
                    value_weight[row] += assignment

        log_count = np.log1p(log_count)
        blocks = [presence, log_count]
        if self.config.include_numeric_features:
            numeric_mean = np.divide(
                numeric_sum,
                np.maximum(numeric_weight, 1e-6),
                out=np.zeros_like(numeric_sum),
                where=numeric_weight > 0,
            )
            blocks.append(numeric_mean)
        if value_sum is not None and value_weight is not None:
            value_mean = np.divide(
                value_sum,
                np.maximum(value_weight[:, :, None], 1e-6),
                out=np.zeros_like(value_sum),
                where=value_weight[:, :, None] > 0,
            )
            blocks.append(value_mean.reshape(len(patient_indices), -1))
        features = np.concatenate(blocks, axis=1)
        return features.astype(np.float32)

    def fit_transform(
        self,
        records: Sequence[MentionRecord],
        embeddings: np.ndarray,
        patient_indices: Sequence[int],
        value_embeddings: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return self.fit(records, embeddings, patient_indices, value_embeddings).transform(
            records,
            embeddings,
            patient_indices,
            value_embeddings,
        )

    def diagnostics(self) -> Dict[str, int | float]:
        if self.kmeans_ is None:
            return {}
        return {
            "n_slots": int(self.kmeans_.n_clusters),
            "x_dim": int(self.n_features_ or 0),
            "value_feature_dim": int(self.value_feature_dim_),
            "assignment_temperature": float(self.config.assignment_temperature),
            "top_assignments": int(self.config.top_assignments),
        }
