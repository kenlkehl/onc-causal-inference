"""Multi-device concept-embedding precompute writes every shard (F821 path)."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import torch

from oci.models import concept_embedding_cache
from oci.models.concept_embedding_cache import ConceptEmbeddingCache


class _FakeEncoder:
    max_seq_length = None

    def get_sentence_embedding_dimension(self) -> int:
        return 3

    def encode(self, chunks, **_kwargs):
        return np.asarray(
            [[len(chunk), chunk.count(" "), sum(map(ord, chunk)) % 97] for chunk in chunks],
            dtype=np.float32,
        )


def _cache(path: Path) -> ConceptEmbeddingCache:
    return ConceptEmbeddingCache(
        cache_dir=str(path),
        sentence_model_name="fake-model",
        dataset_path=str(path / "dataset.parquet"),
        chunk_size_words=3,
        chunk_overlap_words=1,
        max_chunks=16,
    )


def test_multi_device_precompute_matches_single_device(tmp_path, monkeypatch):
    monkeypatch.setattr(
        concept_embedding_cache,
        "load_sentence_transformer",
        lambda *_args, **_kwargs: _FakeEncoder(),
    )
    texts = [
        "one two three four five six seven",
        "alpha beta",
        "gamma delta epsilon zeta eta theta iota kappa lambda",
    ]
    single = _cache(tmp_path / "single")
    single.precompute(texts, device=torch.device("cpu"), batch_size=2)
    multi = _cache(tmp_path / "multi")
    multi.precompute_multi_gpu(
        texts,
        devices=[torch.device("cpu"), torch.device("meta")],
        batch_size=2,
    )

    single_embeddings = np.load(single.cache_path / "chunk_embeddings.npy")
    multi_embeddings = np.load(multi.cache_path / "chunk_embeddings.npy")
    assert len(multi_embeddings) > 2
    np.testing.assert_array_equal(multi_embeddings, single_embeddings)
    np.testing.assert_array_equal(
        np.load(multi.cache_path / "offsets.npy"),
        np.load(single.cache_path / "offsets.npy"),
    )
    metadata = json.loads((multi.cache_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["precompute_devices"] == ["cpu", "meta"]
    multi.open()
    assert len(multi.hidden_states_array) == len(texts)


def test_shard_encoder_does_not_close_over_the_deleted_memmap():
    source = Path(concept_embedding_cache.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    encode_shard = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_encode_shard"
    )
    names = {node.id for node in ast.walk(encode_shard) if isinstance(node, ast.Name)}
    assert "emb_mmap" not in names
