"""Persistent cache for sentence-transformer chunk embeddings."""

from __future__ import annotations

import concurrent.futures
import gc
import hashlib
import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from .concept_embedding_utils import chunk_text_words, split_text_to_token_chunks
from .lossless_tokenization import SemanticTruncationError


logger = logging.getLogger(__name__)


class _VariableLengthArray:
    """Index variable-length rows stored in one flat embedding array."""

    def __init__(self, flat_array: np.ndarray, offsets: np.ndarray) -> None:
        self.flat = flat_array
        self.offsets = offsets

    def __getitem__(self, index: int) -> np.ndarray:
        start = int(self.offsets[index])
        end = int(self.offsets[index + 1])
        return self.flat[start:end]

    def __len__(self) -> int:
        return len(self.offsets) - 1

    @property
    def shape(self) -> tuple[int, str, int]:
        return (len(self), "variable", int(self.flat.shape[-1]))


_SENTENCE_TRANSFORMER_CACHE = {}
_CHUNK_TEXTS_FILENAME = "chunk_texts.jsonl"
_CHUNKING_POLICY_VERSION = "word_then_token_bound_lossless_v2"


def _is_cuda_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    message = str(exc).lower()
    return "cuda" in message and "out of memory" in message


def _clear_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            logger.debug("torch.cuda.ipc_collect failed", exc_info=True)


def _apply_max_seq_length(encoder, max_seq_length: Optional[int]) -> None:
    if max_seq_length is None:
        return
    max_seq_length = int(max_seq_length)
    current = getattr(encoder, "max_seq_length", None)
    try:
        current_int = int(current) if current is not None else None
    except (TypeError, ValueError):
        current_int = None
    if current_int is not None and current_int <= max_seq_length:
        return
    try:
        encoder.max_seq_length = max_seq_length
        logger.info(
            "Capped SentenceTransformer max_seq_length at %d (model default was %s)",
            max_seq_length,
            current if current is not None else "unknown",
        )
    except Exception:
        logger.warning(
            "Failed to set SentenceTransformer max_seq_length=%d",
            max_seq_length,
            exc_info=True,
        )


def load_sentence_transformer(
    model_name: str,
    device: Optional[torch.device] = None,
    max_seq_length: Optional[int] = None,
):
    """Load a sentence-transformer model lazily."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for embedding-contrast evidence. "
            "Install the project dependency or run: pip install sentence-transformers"
        ) from exc
    model_device = str(device) if device is not None else None
    max_seq_key = int(max_seq_length) if max_seq_length is not None else None
    key = (model_name, model_device, max_seq_key)
    if key not in _SENTENCE_TRANSFORMER_CACHE:
        encoder = SentenceTransformer(
            model_name,
            device=model_device,
            model_kwargs={"torch_dtype": torch.float32},
        )
        _apply_max_seq_length(encoder, max_seq_key)
        encoder.float()
        encoder.eval()
        _SENTENCE_TRANSFORMER_CACHE[key] = encoder
    return _SENTENCE_TRANSFORMER_CACHE[key]


def clear_sentence_transformer_cache(
    model_name: Optional[str] = None,
    devices: Optional[List[torch.device]] = None,
) -> None:
    """Drop cached sentence-transformer instances so GPU memory can be reused."""
    device_names = {str(device) for device in devices} if devices is not None else None
    for key in list(_SENTENCE_TRANSFORMER_CACHE):
        cached_model_name, cached_device, _ = key
        if model_name is not None and cached_model_name != model_name:
            continue
        if device_names is not None and cached_device not in device_names:
            continue
        del _SENTENCE_TRANSFORMER_CACHE[key]
    _clear_cuda_memory()


def _coerce_embedding_matrix(
    batch_embeddings,
    expected_rows: int,
    expected_dim: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    batch_embeddings = np.asarray(batch_embeddings, dtype=np.float32)
    if batch_embeddings.ndim == 1:
        batch_embeddings = batch_embeddings.reshape(1, -1)
    if batch_embeddings.ndim != 2:
        raise RuntimeError(
            "Sentence transformer returned unexpected shape "
            f"{batch_embeddings.shape}"
        )
    if batch_embeddings.shape[0] != expected_rows:
        raise RuntimeError(
            "Sentence transformer returned "
            f"{batch_embeddings.shape[0]} embeddings for "
            f"{expected_rows} chunks"
        )
    embedding_dim = int(batch_embeddings.shape[1])
    if expected_dim is not None and embedding_dim != expected_dim:
        raise RuntimeError(
            "Sentence transformer embedding dimension changed from "
            f"{expected_dim} to {embedding_dim}"
        )
    return batch_embeddings, embedding_dim


def _get_sentence_transformer_tokenizer(encoder):
    tokenizer = getattr(encoder, "tokenizer", None)
    if tokenizer is not None:
        return tokenizer
    first_module = getattr(encoder, "_first_module", None)
    if callable(first_module):
        module = first_module()
        tokenizer = getattr(module, "tokenizer", None)
        if tokenizer is not None:
            return tokenizer
    try:
        module = encoder[0]
        tokenizer = getattr(module, "tokenizer", None)
        if tokenizer is not None:
            return tokenizer
    except Exception:
        pass
    return None


def _effective_max_seq_length(encoder, requested_max_seq_length: Optional[int]):
    current = getattr(encoder, "max_seq_length", None)
    try:
        current_int = int(current) if current is not None else None
    except (TypeError, ValueError):
        current_int = None
    if requested_max_seq_length is None:
        return current_int if current_int is not None and current_int > 0 else None
    requested = int(requested_max_seq_length)
    if current_int is not None and current_int > 0:
        return min(requested, current_int)
    return requested


class ConceptEmbeddingCache:
    """Cache of sentence-transformer embeddings for overlapping text chunks.

    The cache intentionally exposes the same array properties as HiddenStateCache:
    each sample indexes to a variable-length matrix. In this cache the sequence
    axis is chunks rather than tokens, with shape (n_chunks, embedding_dim).
    """

    def __init__(
        self,
        cache_dir: str,
        sentence_model_name: str,
        dataset_path: str,
        chunk_size_words: int,
        chunk_overlap_words: int,
        max_chunks: int,
        normalize_embeddings: bool = True,
        chunk_selection: str = "first",
        max_seq_length: Optional[int] = None,
    ):
        self._cache_dir = Path(cache_dir)
        self._sentence_model_name = sentence_model_name
        self._dataset_path = dataset_path
        self._chunk_size_words = chunk_size_words
        self._chunk_overlap_words = chunk_overlap_words
        self._max_chunks = max_chunks
        self._normalize_embeddings = normalize_embeddings
        self._chunk_selection = str(chunk_selection).strip().lower()
        self._max_seq_length = (
            int(max_seq_length) if max_seq_length is not None else None
        )
        self._cache_hash = self.compute_cache_hash(
            sentence_model_name=sentence_model_name,
            dataset_path=dataset_path,
            chunk_size_words=chunk_size_words,
            chunk_overlap_words=chunk_overlap_words,
            max_chunks=max_chunks,
            normalize_embeddings=normalize_embeddings,
            chunk_selection=self._chunk_selection,
            max_seq_length=self._max_seq_length,
        )
        self._cache_path = self._cache_dir / f"cecnn_chunk_embeddings_{self._cache_hash}"
        self._flat_mmap = None
        self._offsets = None
        self._metadata = None
        self._hs_array = None

    @staticmethod
    def compute_cache_hash(
        sentence_model_name: str,
        dataset_path: str,
        chunk_size_words: int,
        chunk_overlap_words: int,
        max_chunks: int,
        normalize_embeddings: bool = True,
        chunk_selection: str = "first",
        max_seq_length: Optional[int] = None,
    ) -> str:
        key = "|".join(
            [
                sentence_model_name,
                os.path.abspath(dataset_path),
                f"words{chunk_size_words}",
                f"overlap{chunk_overlap_words}",
                f"max{max_chunks}",
                f"norm{int(normalize_embeddings)}",
                f"select{str(chunk_selection).strip().lower()}",
                "tokmax"
                f"{int(max_seq_length) if max_seq_length is not None else 'model'}",
                _CHUNKING_POLICY_VERSION,
            ]
        )
        return hashlib.md5(key.encode()).hexdigest()[:12]

    @property
    def cache_hash(self) -> str:
        return self._cache_hash

    @property
    def cache_path(self) -> Path:
        return self._cache_path

    @property
    def hidden_size(self) -> int:
        if self._metadata is None:
            self._load_metadata()
        return int(self._metadata["hidden_size"])

    @property
    def actual_max_len(self) -> int:
        if self._metadata is None:
            self._load_metadata()
        return int(self._metadata["actual_max_len"])

    @property
    def chunk_counts(self) -> Optional[List[int]]:
        if self._metadata is None:
            self._load_metadata()
        return self._metadata.get("chunk_counts")

    @property
    def cache_size_gb(self) -> float:
        if self._metadata is None:
            self._load_metadata()
        return (
            int(self._metadata["total_chunks"])
            * int(self._metadata["hidden_size"])
            * 2
            / 1e9
        )

    @property
    def hidden_states_array(self):
        if self._hs_array is None:
            self.open()
        return self._hs_array

    def _load_metadata(self) -> None:
        meta_path = self._cache_path / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Cache metadata not found: {meta_path}")
        with open(meta_path) as f:
            self._metadata = json.load(f)

    def is_valid(self, expected_num_samples: int) -> bool:
        try:
            meta_path = self._cache_path / "metadata.json"
            emb_path = self._cache_path / "chunk_embeddings.npy"
            offsets_path = self._cache_path / "offsets.npy"
            chunks_path = self._cache_path / _CHUNK_TEXTS_FILENAME
            if not all(
                p.exists() for p in [meta_path, emb_path, offsets_path, chunks_path]
            ):
                return False

            self._load_metadata()
            if self._metadata.get("cache_hash") != self._cache_hash:
                return False
            if self._metadata.get("num_samples") != expected_num_samples:
                return False
            if self._metadata.get("storage_format") != "variable_length_chunks":
                return False
            if (
                self._metadata.get("chunking_policy_version")
                != _CHUNKING_POLICY_VERSION
                or self._metadata.get("semantic_truncation_allowed") is not False
            ):
                return False

            offsets = np.load(str(offsets_path))
            if len(offsets) != expected_num_samples + 1:
                return False
            embeddings = np.load(str(emb_path), mmap_mode="r")
            if embeddings.shape[0] != int(offsets[-1]):
                return False
            if embeddings.shape[1] != int(self._metadata["hidden_size"]):
                return False
            chunks_by_sample = self.load_chunks(expected_num_samples)
            if len(chunks_by_sample) != expected_num_samples:
                return False
            if sum(len(chunks) for chunks in chunks_by_sample) != int(offsets[-1]):
                return False
            return True
        except Exception as exc:
            logger.warning("Concept embedding cache validation failed: %s", exc)
            return False

    def load_chunks(self, expected_num_samples: Optional[int] = None) -> List[List[str]]:
        chunks_path = self._cache_path / _CHUNK_TEXTS_FILENAME
        chunks_by_sample: List[List[str]] = []
        with open(chunks_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    payload = json.loads(line)
                    chunks = payload.get("chunks", [])
                    chunks_by_sample.append([str(chunk) for chunk in chunks])
        if expected_num_samples is not None and len(chunks_by_sample) != int(
            expected_num_samples
        ):
            raise RuntimeError(
                "Chunk text cache has "
                f"{len(chunks_by_sample)} samples; expected {expected_num_samples}"
            )
        return chunks_by_sample

    def _write_chunks(self, sample_chunks: List[List[str]]) -> None:
        chunks_path = self._cache_path / _CHUNK_TEXTS_FILENAME
        with open(chunks_path, "w", encoding="utf-8") as f:
            for chunks in sample_chunks:
                f.write(json.dumps({"chunks": chunks}, ensure_ascii=False) + "\n")

    def _chunk_texts(
        self,
        texts: List[str],
        tokenizer=None,
        max_seq_length: Optional[int] = None,
    ) -> List[List[str]]:
        sample_chunks = [
            chunk_text_words(
                text,
                self._chunk_size_words,
                self._chunk_overlap_words,
                self._max_chunks,
                self._chunk_selection,
            )
            for text in texts
        ]
        if tokenizer is None or max_seq_length is None:
            return sample_chunks

        token_bounded_chunks: List[List[str]] = []
        for chunks in sample_chunks:
            split_chunks: List[str] = []
            for chunk in chunks:
                split_chunks.extend(
                    split_text_to_token_chunks(
                        chunk,
                        tokenizer,
                        max_seq_length=int(max_seq_length),
                        chunk_overlap_tokens=int(self._chunk_overlap_words),
                    )
                )
            if len(split_chunks) > self._max_chunks:
                raise SemanticTruncationError(
                    "ConceptEmbeddingCache token-bounded chunk plan exceeds "
                    "configured max_chunks; semantic truncation is forbidden "
                    f"({len(split_chunks)} > {self._max_chunks}). Increase "
                    "max_chunks so the cap is nonbinding."
                )
            token_bounded_chunks.append(split_chunks or [""])
        return token_bounded_chunks

    def precompute(
        self,
        texts: List[str],
        device: Optional[torch.device] = None,
        batch_size: int = 128,
    ) -> None:
        """Precompute sentence-transformer embeddings for all text chunks."""
        num_samples = len(texts)
        logger.info(
            "Precomputing concept chunk embeddings: samples=%d, model=%s",
            num_samples,
            self._sentence_model_name,
        )

        encoder = load_sentence_transformer(
            self._sentence_model_name,
            device=device,
            max_seq_length=self._max_seq_length,
        )
        tokenizer = _get_sentence_transformer_tokenizer(encoder)
        effective_max_seq_length = _effective_max_seq_length(
            encoder,
            self._max_seq_length,
        )
        if effective_max_seq_length is not None and tokenizer is None:
            raise RuntimeError(
                "Sentence-transformer exposes a finite sequence limit but no "
                "auditable tokenizer; refusing possible semantic truncation"
            )
        sample_chunks = self._chunk_texts(
            texts,
            tokenizer=tokenizer,
            max_seq_length=effective_max_seq_length,
        )
        chunk_counts = [len(chunks) for chunks in sample_chunks]
        total_chunks = int(sum(chunk_counts))
        offsets = np.zeros(num_samples + 1, dtype=np.int64)
        for i, count in enumerate(chunk_counts):
            offsets[i + 1] = offsets[i] + count

        flat_chunks = [chunk for chunks in sample_chunks for chunk in chunks]

        self._cache_path.mkdir(parents=True, exist_ok=True)
        emb_path = self._cache_path / "chunk_embeddings.npy"
        offsets_path = self._cache_path / "offsets.npy"
        np.save(str(offsets_path), offsets)
        self._write_chunks(sample_chunks)

        effective_batch_size = max(1, int(batch_size))
        logger.info(
            "  Total chunks: %d across %d samples (mean %.1f, max %d)",
            total_chunks,
            num_samples,
            total_chunks / max(num_samples, 1),
            max(chunk_counts) if chunk_counts else 0,
        )
        logger.info("  Encoding chunks with batch_size=%d", effective_batch_size)

        emb_mmap = None
        embedding_dim = None
        cursor = 0
        while cursor < total_chunks:
            end = min(cursor + effective_batch_size, total_chunks)
            batch_chunks = flat_chunks[cursor:end]
            try:
                batch_embeddings = encoder.encode(
                    batch_chunks,
                    batch_size=len(batch_chunks),
                    convert_to_numpy=True,
                    normalize_embeddings=self._normalize_embeddings,
                    show_progress_bar=False,
                )
            except Exception as exc:
                if _is_cuda_oom(exc):
                    _clear_cuda_memory()
                if _is_cuda_oom(exc) and effective_batch_size > 1:
                    new_batch_size = max(1, effective_batch_size // 2)
                    logger.warning(
                        "CUDA OOM while encoding concept chunks; reducing "
                        "batch_size from %d to %d and retrying",
                        effective_batch_size,
                        new_batch_size,
                    )
                    effective_batch_size = new_batch_size
                    continue
                if _is_cuda_oom(exc):
                    logger.error(
                        "CUDA OOM while encoding concept chunk at flat index %d "
                        "with batch_size=%d, chunk_chars=%d, chunk_words=%d; "
                        "try lowering embedding max_seq_length or chunk_size_words",
                        cursor,
                        effective_batch_size,
                        len(batch_chunks[0]) if batch_chunks else 0,
                        len(batch_chunks[0].split()) if batch_chunks else 0,
                    )
                raise

            batch_embeddings, batch_embedding_dim = _coerce_embedding_matrix(
                batch_embeddings,
                expected_rows=len(batch_chunks),
                expected_dim=embedding_dim,
            )

            if emb_mmap is None:
                embedding_dim = batch_embedding_dim
                emb_mmap = np.lib.format.open_memmap(
                    str(emb_path),
                    mode="w+",
                    dtype=np.float16,
                    shape=(total_chunks, embedding_dim),
                )

            emb_mmap[cursor:end] = batch_embeddings.astype(np.float16)
            del batch_embeddings
            cursor = end

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if emb_mmap is None or embedding_dim is None:
            raise RuntimeError("No chunk embeddings were generated")
        emb_mmap.flush()
        logger.info(
            "  Encoded %d chunks; final batch_size=%d",
            total_chunks,
            effective_batch_size,
        )

        metadata = {
            "sentence_model_name": self._sentence_model_name,
            "hidden_size": embedding_dim,
            "num_samples": num_samples,
            "total_chunks": total_chunks,
            "chunk_counts": chunk_counts,
            "chunk_size_words": self._chunk_size_words,
            "chunk_overlap_words": self._chunk_overlap_words,
            "max_chunks": self._max_chunks,
            "chunk_selection": self._chunk_selection,
            "normalize_embeddings": self._normalize_embeddings,
            "max_seq_length": self._max_seq_length,
            "effective_max_seq_length": effective_max_seq_length,
            "chunking_mode": "word_chunks_token_bounded",
            "actual_max_len": max(chunk_counts) if chunk_counts else 0,
            "storage_format": "variable_length_chunks",
            "dataset_path": os.path.abspath(self._dataset_path),
            "cache_hash": self._cache_hash,
            "chunking_policy_version": _CHUNKING_POLICY_VERSION,
            "semantic_truncation_allowed": False,
            "created_at": datetime.now().isoformat(),
            "dtype": "float16",
            "precompute_batch_size": effective_batch_size,
        }
        with open(self._cache_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        self._metadata = metadata
        logger.info(
            "Concept embedding cache created: %.3f GB, %d chunks",
            self.cache_size_gb,
            total_chunks,
        )

    def precompute_multi_gpu(
        self,
        texts: List[str],
        devices: List[torch.device],
        batch_size: int = 128,
    ) -> None:
        """Precompute sentence-transformer chunk embeddings across devices.

        Chunks are flattened in sample order, split into contiguous non-overlapping
        ranges, and encoded by one sentence-transformer copy per device. Each
        worker writes its range directly into the shared memmap.
        """
        unique_devices: List[torch.device] = []
        seen_devices = set()
        for device in devices:
            device_name = str(device)
            if device_name in seen_devices:
                continue
            seen_devices.add(device_name)
            unique_devices.append(device)

        if len(unique_devices) <= 1:
            device = unique_devices[0] if unique_devices else None
            self.precompute(texts, device=device, batch_size=batch_size)
            return

        num_samples = len(texts)
        logger.info(
            "Precomputing concept chunk embeddings: samples=%d, model=%s, devices=%s",
            num_samples,
            self._sentence_model_name,
            [str(device) for device in unique_devices],
        )

        probe_encoder = load_sentence_transformer(
            self._sentence_model_name,
            device=unique_devices[0],
            max_seq_length=self._max_seq_length,
        )
        tokenizer = _get_sentence_transformer_tokenizer(probe_encoder)
        effective_max_seq_length = _effective_max_seq_length(
            probe_encoder,
            self._max_seq_length,
        )
        if effective_max_seq_length is not None and tokenizer is None:
            raise RuntimeError(
                "Sentence-transformer exposes a finite sequence limit but no "
                "auditable tokenizer; refusing possible semantic truncation"
            )
        sample_chunks = self._chunk_texts(
            texts,
            tokenizer=tokenizer,
            max_seq_length=effective_max_seq_length,
        )
        chunk_counts = [len(chunks) for chunks in sample_chunks]
        total_chunks = int(sum(chunk_counts))
        offsets = np.zeros(num_samples + 1, dtype=np.int64)
        for i, count in enumerate(chunk_counts):
            offsets[i + 1] = offsets[i] + count

        flat_chunks = [chunk for chunks in sample_chunks for chunk in chunks]
        if total_chunks == 0:
            raise RuntimeError("No text chunks were generated")

        embedding_dim = int(
            getattr(probe_encoder, "get_sentence_embedding_dimension", lambda: 0)()
            or 0
        )
        if embedding_dim <= 0:
            probe_embeddings = probe_encoder.encode(
                [flat_chunks[0]],
                batch_size=1,
                convert_to_numpy=True,
                normalize_embeddings=self._normalize_embeddings,
                show_progress_bar=False,
            )
            _, embedding_dim = _coerce_embedding_matrix(
                probe_embeddings,
                expected_rows=1,
            )

        self._cache_path.mkdir(parents=True, exist_ok=True)
        emb_path = self._cache_path / "chunk_embeddings.npy"
        offsets_path = self._cache_path / "offsets.npy"
        np.save(str(offsets_path), offsets)
        self._write_chunks(sample_chunks)
        emb_mmap = np.lib.format.open_memmap(
            str(emb_path),
            mode="w+",
            dtype=np.float16,
            shape=(total_chunks, embedding_dim),
        )

        num_workers = min(len(unique_devices), total_chunks)
        shard_size = (total_chunks + num_workers - 1) // num_workers
        shards = []
        for idx in range(num_workers):
            start = idx * shard_size
            end = min(start + shard_size, total_chunks)
            if start >= end:
                continue
            shards.append((unique_devices[idx], start, end))

        logger.info(
            "  Total chunks: %d across %d samples (mean %.1f, max %d)",
            total_chunks,
            num_samples,
            total_chunks / max(num_samples, 1),
            max(chunk_counts) if chunk_counts else 0,
        )
        logger.info(
            "  Encoding chunks across %d device(s) with batch_size=%d per device",
            len(shards),
            max(1, int(batch_size)),
        )

        progress_lock = threading.Lock()
        progress = [0, 0]
        log_stride = max(1000, total_chunks // 10)

        def _encode_shard(device: torch.device, shard_start: int, shard_end: int) -> int:
            encoder = load_sentence_transformer(
                self._sentence_model_name,
                device=device,
                max_seq_length=self._max_seq_length,
            )
            effective_batch_size = max(1, int(batch_size))
            cursor = shard_start
            while cursor < shard_end:
                end = min(cursor + effective_batch_size, shard_end)
                batch_chunks = flat_chunks[cursor:end]
                try:
                    batch_embeddings = encoder.encode(
                        batch_chunks,
                        batch_size=len(batch_chunks),
                        convert_to_numpy=True,
                        normalize_embeddings=self._normalize_embeddings,
                        show_progress_bar=False,
                    )
                except Exception as exc:
                    if _is_cuda_oom(exc):
                        _clear_cuda_memory()
                    if _is_cuda_oom(exc) and effective_batch_size > 1:
                        new_batch_size = max(1, effective_batch_size // 2)
                        logger.warning(
                            "CUDA OOM while encoding concept chunks on %s; reducing "
                            "batch_size from %d to %d and retrying",
                            device,
                            effective_batch_size,
                            new_batch_size,
                        )
                        effective_batch_size = new_batch_size
                        continue
                    if _is_cuda_oom(exc):
                        logger.error(
                            "CUDA OOM while encoding concept chunk on %s at flat "
                            "index %d with batch_size=%d, chunk_chars=%d, "
                            "chunk_words=%d; try lowering embedding max_seq_length "
                            "or chunk_size_words",
                            device,
                            cursor,
                            effective_batch_size,
                            len(batch_chunks[0]) if batch_chunks else 0,
                            len(batch_chunks[0].split()) if batch_chunks else 0,
                        )
                    raise

                batch_embeddings, _ = _coerce_embedding_matrix(
                    batch_embeddings,
                    expected_rows=len(batch_chunks),
                    expected_dim=embedding_dim,
                )
                emb_mmap[cursor:end] = batch_embeddings.astype(np.float16)
                del batch_embeddings
                cursor = end

                with progress_lock:
                    progress[0] += len(batch_chunks)
                    if (
                        progress[0] == total_chunks
                        or progress[0] - progress[1] >= log_stride
                    ):
                        progress[1] = progress[0]
                        logger.info(
                            "  Encoded %d/%d concept chunks",
                            progress[0],
                            total_chunks,
                        )

                if device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            return effective_batch_size

        final_batch_sizes = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(shards)) as executor:
            futures = [
                executor.submit(_encode_shard, device, start, end)
                for device, start, end in shards
            ]
            for future in concurrent.futures.as_completed(futures):
                final_batch_sizes.append(future.result())

        emb_mmap.flush()
        del emb_mmap
        logger.info(
            "  Encoded %d chunks; final batch_size range=%d-%d",
            total_chunks,
            min(final_batch_sizes) if final_batch_sizes else max(1, int(batch_size)),
            max(final_batch_sizes) if final_batch_sizes else max(1, int(batch_size)),
        )

        metadata = {
            "sentence_model_name": self._sentence_model_name,
            "hidden_size": embedding_dim,
            "num_samples": num_samples,
            "total_chunks": total_chunks,
            "chunk_counts": chunk_counts,
            "chunk_size_words": self._chunk_size_words,
            "chunk_overlap_words": self._chunk_overlap_words,
            "max_chunks": self._max_chunks,
            "chunk_selection": self._chunk_selection,
            "normalize_embeddings": self._normalize_embeddings,
            "max_seq_length": self._max_seq_length,
            "effective_max_seq_length": effective_max_seq_length,
            "chunking_mode": "word_chunks_token_bounded",
            "actual_max_len": max(chunk_counts) if chunk_counts else 0,
            "storage_format": "variable_length_chunks",
            "dataset_path": os.path.abspath(self._dataset_path),
            "cache_hash": self._cache_hash,
            "chunking_policy_version": _CHUNKING_POLICY_VERSION,
            "semantic_truncation_allowed": False,
            "created_at": datetime.now().isoformat(),
            "dtype": "float16",
            "precompute_batch_size": max(final_batch_sizes)
            if final_batch_sizes
            else max(1, int(batch_size)),
            "num_gpus_used": sum(1 for device, _, _ in shards if device.type == "cuda"),
            "precompute_devices": [str(device) for device, _, _ in shards],
        }
        with open(self._cache_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        self._metadata = metadata
        logger.info(
            "Concept embedding cache created (%d device(s)): %.3f GB, %d chunks",
            len(shards),
            self.cache_size_gb,
            total_chunks,
        )

    def open(self) -> None:
        if self._metadata is None:
            self._load_metadata()
        emb_path = self._cache_path / "chunk_embeddings.npy"
        offsets_path = self._cache_path / "offsets.npy"
        flat = np.load(str(emb_path), mmap_mode="r")
        offsets = np.load(str(offsets_path))
        self._flat_mmap = flat
        self._offsets = offsets
        self._hs_array = _VariableLengthArray(flat, offsets)
