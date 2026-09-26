"""Deterministic mixed candidate grouping and merge-yield stopping rules.

Grouping only arranges LLM reviews. It never merges, filters, or assigns causal
roles to a candidate, and uses no patient measurements or outcomes.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

MIXED_SCHEMA = "mixed_semantic_alphabetical_random_candidate_batches_v1"


@dataclass(frozen=True)
class CandidateConsolidationPolicy:
    strategy: str = "mixed"
    semantic_fraction: float = 0.6
    random_fraction: float = 0.1
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_device: str = "cpu"
    early_stop_min_rounds: int = 3
    early_stop_patience: int = 2
    early_stop_min_reduction: float = 0.005

    def __post_init__(self):
        if self.strategy not in {"mixed", "legacy"}:
            raise ValueError("consolidation_policy.strategy must be mixed or legacy")
        for name in ("semantic_fraction", "random_fraction", "early_stop_min_reduction"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"consolidation_policy.{name} must be between 0 and 1")
        if self.semantic_fraction + self.random_fraction > 1:
            raise ValueError("consolidation_policy grouping fractions must sum to at most 1")
        for name in ("early_stop_min_rounds", "early_stop_patience"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"consolidation_policy.{name} must be a positive integer")
        if any(not isinstance(value, str) or not value.strip()
               for value in (self.embedding_model, self.embedding_device)):
            raise ValueError("consolidation_policy embedding model and device must be nonempty")

    def public_dict(self):
        return asdict(self)


def policy_from_mapping(raw: Mapping[str, Any] | None) -> CandidateConsolidationPolicy:
    if raw is None:
        return CandidateConsolidationPolicy()
    if not isinstance(raw, Mapping):
        raise ValueError("stage2.consolidation_policy must be an object")
    unknown = set(raw) - set(CandidateConsolidationPolicy.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown consolidation_policy settings: {sorted(unknown)}")
    return CandidateConsolidationPolicy(**raw)


class CandidateEmbeddingCache:
    """Persist vectors by exact description text, encoding only new/changed text."""

    def __init__(self, path: Path | None, policy: CandidateConsolidationPolicy,
                 encode: Callable[[Sequence[str], str, str], np.ndarray]):
        self.path, self.policy, self.encode = path, policy, encode
        self.vectors: dict[str, np.ndarray] = {}
        if path is not None and path.exists():
            with np.load(path, allow_pickle=False) as saved:
                if str(saved["model"].item()) != policy.embedding_model:
                    raise ValueError("Consolidation embedding cache model changed")
                keys, matrix = saved["keys"].tolist(), saved["vectors"]
                if matrix.ndim != 2 or len(keys) != len(matrix) or not np.isfinite(matrix).all():
                    raise ValueError("Invalid consolidation embedding cache")
                if len(set(keys)) != len(keys) or not np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-5):
                    raise ValueError("Invalid consolidation embedding cache keys or norms")
                self.vectors = dict(zip(keys, matrix))

    def matrix(self, texts: Sequence[str]) -> np.ndarray:
        keys = [hashlib.sha256(text.encode()).hexdigest() for text in texts]
        missing = dict((key, text) for key, text in zip(keys, texts) if key not in self.vectors)
        if missing:
            matrix = np.asarray(self.encode(list(missing.values()), self.policy.embedding_model,
                                            self.policy.embedding_device), dtype=np.float32)
            if matrix.ndim != 2 or matrix.shape[0] != len(missing) or matrix.shape[1] < 1 or not np.isfinite(matrix).all():
                raise ValueError("Invalid consolidation embeddings")
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            if (norms <= 0).any():
                raise ValueError("Zero consolidation embedding")
            if self.vectors and len(next(iter(self.vectors.values()))) != matrix.shape[1]:
                raise ValueError("Consolidation embedding dimension changed")
            self.vectors.update(zip(missing, matrix / norms))
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                temporary = self.path.with_suffix(".tmp")
                with temporary.open("wb") as stream:
                    np.savez_compressed(stream, model=self.policy.embedding_model,
                                        keys=list(self.vectors), vectors=np.stack(list(self.vectors.values())))
                temporary.replace(self.path)
        return np.stack([self.vectors[key] for key in keys])


def mixed_batches(groups: Sequence[Mapping[str, Any]], *, batch_size: int,
                  round_number: int, seed: int, policy: CandidateConsolidationPolicy,
                  embeddings: np.ndarray | None) -> tuple[list[list[dict[str, Any]]], list[str]]:
    """Partition a name-sorted pool once, interleaving weighted grouping methods.

    Semantic batches retrieve the nearest still-unassigned candidates to a
    pivot. Alphabetical batches follow a rotating lexical order. Random batches
    use seeded hashes. Round-dependent pivots and order expose new neighbors.
    """
    if batch_size < 2 or round_number < 1:
        raise ValueError("Invalid mixed consolidation batch size or round")
    if not groups:
        return [], []
    names = [str(group["name"]) for group in groups]
    if names != sorted(names) or len(names) != len(set(names)):
        raise ValueError("Mixed consolidation requires unique, name-sorted groups")
    if len(groups) <= batch_size:
        return [[dict(group) for group in groups]], ["complete_pool"]
    if policy.semantic_fraction and (embeddings is None or len(embeddings) != len(groups)):
        raise ValueError("Mixed consolidation requires an embedding for every candidate")
    weights = {"semantic": policy.semantic_fraction,
               "alphabetical": 1 - policy.semantic_fraction - policy.random_fraction,
               "random": policy.random_fraction}
    methods = [method for method, weight in weights.items() if weight > 0]
    # An odd step coprime to the pool length changes which semantic pivots and
    # alphabetical boundaries are considered first on subsequent rounds.
    step = max(1, len(groups) // 7)
    while math.gcd(step, len(groups)) != 1:
        step += 1
    offset = ((round_number - 1) * step + seed % len(groups)) % len(groups)
    lexical = list(range(offset, len(groups))) + list(range(offset))
    random_order = sorted(range(len(groups)), key=lambda i: (
        hashlib.sha256(f"{seed}\0{round_number}\0{names[i]}".encode()).hexdigest(), names[i]))
    available = set(lexical)
    counts: Counter[str] = Counter()
    batches, orderings = [], []
    while available:
        method = max(methods, key=lambda key: weights[key] * (len(batches) + 1) - counts[key])
        counts[method] += 1
        pivot = next(i for i in lexical if i in available)
        if method == "random":
            selected = [i for i in random_order if i in available][:batch_size]
        elif method == "alphabetical":
            selected = [i for i in lexical if i in available][:batch_size]
        else:
            similarities = embeddings @ embeddings[pivot]
            neighbors = sorted(available - {pivot}, key=lambda i: (-float(similarities[i]), names[i]))
            selected = [pivot, *neighbors[:batch_size - 1]]
        available.difference_update(selected)
        batches.append([dict(groups[i]) for i in sorted(selected)])
        orderings.append(method)
    return batches, orderings


def diminishing_returns(rounds: Sequence[Mapping[str, Any]], policy: CandidateConsolidationPolicy) -> bool:
    """Only fully successful, complete rounds count toward the low-yield streak."""
    if len(rounds) < max(policy.early_stop_min_rounds, policy.early_stop_patience):
        return False
    recent = rounds[-policy.early_stop_patience:]
    return all(not item["validation_fallback_batches"] and item["input_groups"] > 0
               and (item["input_groups"] - item["output_groups"]) / item["input_groups"] < policy.early_stop_min_reduction
               for item in recent)
