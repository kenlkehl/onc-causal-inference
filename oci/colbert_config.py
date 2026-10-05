"""Lightweight configuration for OCI's independent ColBERT retriever."""

from dataclasses import dataclass, fields
from typing import Mapping

COLBERT_VERSION = "oci_colbert_v1"


@dataclass(frozen=True)
class ColBERTConfig:
    model_name: str = "lightonai/GTE-ModernColBERT-v1"
    revision: str | None = None
    # Logical CUDA indices, after CUDA_VISIBLE_DEVICES; auto uses all visible GPUs.
    devices: tuple[str, ...] = ("auto",)
    cache_dir: str = ".oci_cache/colbert"
    chunk_size: int = 64
    chunk_overlap: int = 0
    query_length: int = 512
    batch_size: int = 32
    score_batch_size: int = 32
    top_k: int = 20
    # Runtime concurrency; excluded from encoding and measurement identities.
    workers_per_device: int = 1
    # Runtime host-vector budget shared by the entire retrieval pool. Zero
    # disables retention; concurrent misses are still coalesced.
    query_cache_max_bytes: int = 1024 * 1024 * 1024

    def __post_init__(self):
        for name in ("model_name", "cache_dir"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"colbert.{name} must be nonempty")
        if self.revision is not None and (
            not isinstance(self.revision, str) or not self.revision.strip()
        ):
            raise ValueError("colbert.revision must be a nonempty string or null")
        if isinstance(self.devices, str):
            object.__setattr__(self, "devices", tuple(x.strip() for x in self.devices.split(",")))
        else:
            object.__setattr__(self, "devices", tuple(self.devices))
        if (
            not self.devices
            or len(set(self.devices)) != len(self.devices)
            or any(
                not isinstance(d, str)
                or not (d in {"auto", "cpu", "cuda"} or (d.startswith("cuda:") and d[5:].isdigit()))
                for d in self.devices
            )
            or ("auto" in self.devices and len(self.devices) != 1)
        ):
            raise ValueError("colbert.devices must be auto, cpu, or unique logical CUDA devices")
        for name in (
            "workers_per_device", "chunk_size", "query_length", "batch_size",
            "score_batch_size", "top_k",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"colbert.{name} must be a positive integer")
        if self.query_length < 8:
            raise ValueError("colbert.query_length must be at least 8")
        if type(self.query_cache_max_bytes) is not int or self.query_cache_max_bytes < 0:
            raise ValueError("colbert.query_cache_max_bytes must be a nonnegative integer")
        if type(self.chunk_overlap) is not int or not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError(
                "colbert.chunk_overlap must be nonnegative and smaller than chunk_size"
            )

    def encoding_identity(self):
        return {
            "version": COLBERT_VERSION,
            "model_name": self.model_name,
            "revision": self.revision,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "query_length": self.query_length,
        }

    def measurement_identity(self):
        return {**self.encoding_identity(), "top_k": self.top_k}


def colbert_config_from_mapping(value=None) -> ColBERTConfig:
    if isinstance(value, ColBERTConfig):
        return value
    if value is None:
        return ColBERTConfig()
    if not isinstance(value, Mapping):
        raise ValueError("colbert must be a configuration object")
    unknown = set(value) - {f.name for f in fields(ColBERTConfig)}
    if unknown:
        raise ValueError(f"Unknown colbert settings: {sorted(unknown)}")
    return ColBERTConfig(**dict(value))
