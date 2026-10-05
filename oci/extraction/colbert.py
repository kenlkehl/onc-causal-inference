"""Disk-backed, patient-isolated ColBERT retrieval with configurable device workers.

The shared worker pool bounds GPU replicas across concurrent folds and patient
threads. Each worker batches token encoding and MaxSim scoring. Patient indexes
are content addressed, atomically published, checksummed, and protected by file
locks across processes. No fitted cohort statistics or outcomes enter retrieval.
"""

from collections import OrderedDict
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile
import threading
import zipfile

import numpy as np

from ..colbert_config import colbert_config_from_mapping

LOGGER = logging.getLogger(__name__)
_PATIENT_CACHE_BYTES = 128 * 1024 * 1024
_PATIENT_CACHE_ENTRIES = 16


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def feature_query(feature):
    """Only measurement semantics enter a query; never causal roles or outcomes."""
    if not isinstance(feature, Mapping):
        from dataclasses import asdict

        feature = asdict(feature)
    keys = (
        "name",
        "description",
        "measurement_definition",
        "type",
        "value_type",
        "categories",
        "categories_or_unit",
        "value_aliases",
        "temporal_rule",
        "aggregation_rule",
        "conflict_resolution",
        "missing_value_rule",
    )
    return "\n".join(
        f"{key}: {json.dumps(feature[key], ensure_ascii=False)}"
        for key in keys
        if feature.get(key) is not None
    )


def chunk_text(text, tokenizer, config):
    """Token windows with verbatim Unicode character ranges; never truncate."""
    if not text.strip():
        return []
    offsets = tokenizer(
        text, add_special_tokens=False, truncation=False, return_offsets_mapping=True
    )["offset_mapping"]
    if not offsets:
        raise ValueError("Nonempty record has no encodable tokens")
    chunks, start = [], 0
    while start < len(offsets):
        end = min(start + config.chunk_size, len(offsets))
        while end < len(offsets) and offsets[end][0] < offsets[end - 1][1]:
            end += 1
        left = 0 if start == 0 else offsets[start][0]
        right = len(text) if end == len(offsets) else offsets[end][0]
        excerpt = text[left:right]
        if (
            len(tokenizer(excerpt, add_special_tokens=False, truncation=False)["input_ids"])
            > config.chunk_size + 5
        ):
            raise ValueError("ColBERT chunk boundary exceeds document capacity")
        chunks.append({"start": int(left), "end": int(right), "text": excerpt})
        if end == len(offsets):
            break
        start = max(start + 1, end - config.chunk_overlap)
        while start < end and offsets[start][0] < offsets[start - 1][1]:
            start += 1
    return chunks


def normalized_vectors(values):
    result = []
    for value in values:
        array = np.asarray(value, dtype=np.float32)
        if array.ndim != 2 or not all(array.shape) or not np.isfinite(array).all():
            raise ValueError("Invalid ColBERT token vectors")
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        if (norms <= 1e-12).any():
            raise ValueError("Zero ColBERT token vector")
        result.append(array / norms)
    if len({v.shape[1] for v in result}) > 1:
        raise ValueError("Inconsistent ColBERT vector dimensions")
    return result


@contextmanager
def cache_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(str(path) + ".lock", os.O_CREAT | os.O_RDWR, 0o600)
    with os.fdopen(fd, "a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def save_index(path, identity, chunks, vectors):
    lengths = np.array([len(v) for v in vectors], dtype=np.int64)
    packed = np.concatenate(vectors) if vectors else np.empty((0, 0), dtype=np.float32)
    manifest = {
        "identity": identity,
        "chunks": chunks,
        "vectors_sha256": hashlib.sha256(packed.tobytes() + lengths.tobytes()).hexdigest(),
    }
    # mkstemp creates owner-only files, also on a shared filesystem.
    fd, name = tempfile.mkstemp(prefix=".colbert-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez(
                stream,
                manifest=json.dumps(manifest, ensure_ascii=False),
                vectors=packed,
                lengths=lengths,
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def load_index(path, identity, text):
    with np.load(path, allow_pickle=False) as data:
        manifest = json.loads(str(data["manifest"].item()))
        vectors, lengths = data["vectors"], data["lengths"]
    if manifest["identity"] != identity:
        raise ValueError("Stale ColBERT index identity")
    chunks = manifest["chunks"]
    if (
        vectors.dtype != np.float32
        or lengths.dtype != np.int64
        or lengths.ndim != 1
        or vectors.ndim != 2
        or len(lengths) != len(chunks)
        or int(lengths.sum()) != len(vectors)
        or (lengths <= 0).any()
        or not np.isfinite(vectors).all()
        or hashlib.sha256(vectors.tobytes() + lengths.tobytes()).hexdigest()
        != manifest["vectors_sha256"]
    ):
        raise ValueError("Corrupt ColBERT vectors")
    previous = -1
    covered = 0
    for chunk in chunks:
        left, right = chunk["start"], chunk["end"]
        if (
            type(left) is not int
            or type(right) is not int
            or not 0 <= left < right <= len(text)
            or left <= previous
            or left > covered
            or text[left:right] != chunk["text"]
        ):
            raise ValueError("Corrupt ColBERT source spans")
        previous, covered = left, right
    if text.strip() and covered != len(text):
        raise ValueError("Incomplete ColBERT source coverage")
    return chunks, list(np.split(vectors, np.cumsum(lengths)[:-1])) if chunks else []


def _maxsim_batches(documents, *, device, batch_size):
    """Prepare unchanged document batches once for repeated feature searches."""
    import torch

    for offset in range(0, len(documents), batch_size):
        batch = documents[offset : offset + batch_size]
        longest = max(len(v) for v in batch)
        vectors = torch.zeros((len(batch), longest, batch[0].shape[1]), device=device)
        mask = torch.zeros((len(batch), longest), dtype=torch.bool, device=device)
        for i, doc in enumerate(batch):
            vectors[i, : len(doc)] = torch.as_tensor(doc, device=device)
            mask[i, : len(doc)] = True
        yield vectors, mask


def maxsim_scores(query, documents, *, device="cpu", batch_size=32, prepared_batches=None):
    """Exact sum-of-token-maxima cosine score, with bounded GPU working memory."""
    if not documents:
        return np.empty(0, dtype=np.float32)
    if device == "cpu":
        return np.array(
            [np.max(query @ doc.T, axis=1).sum() for doc in documents], dtype=np.float32
        )
    import torch

    scores = []
    with torch.inference_mode():
        q = torch.as_tensor(query, device=device)
        batches = prepared_batches if prepared_batches is not None else _maxsim_batches(
            documents, device=device, batch_size=batch_size)
        for vectors, mask in batches:
            similarity = torch.einsum("qd,btd->bqt", q, vectors)
            similarity.masked_fill_(~mask[:, None, :], -torch.inf)
            scores.extend(similarity.max(dim=-1).values.sum(dim=-1).cpu().tolist())
    return np.asarray(scores, dtype=np.float32)


def render_context(text, ranked):
    """Merge overlapping hits in source order so observations are not duplicated."""
    spans = []
    for hit in sorted(ranked, key=lambda h: (h["start"], h["end"])):
        if spans and hit["start"] <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], hit["end"])
        else:
            spans.append([hit["start"], hit["end"]])
    header = (
        "[oci_colbert_v1] Retrieved excerpts from this patient's record. "
        "Only these excerpts were reviewed. Missing means not found in the retrieved "
        "evidence, not absent from the full record. Source order is not clinical chronology. "
        "Apply temporal and conflict rules only when the excerpts support them. "
        "For counts or modes, only retrieved observations are available.\n"
    )
    return header + "\n\n".join(
        f"[Source characters {left}:{right}]\n{text[left:right]}" for left, right in spans
    )


class _DeviceWorker:
    def __init__(self, config, device, folder, signature):
        self.config, self.device, self.folder, self.signature = config, device, folder, signature
        self.encoder = None
        self.stream = None
        self.queries = OrderedDict()
        self.patients = OrderedDict()
        self.patient_cache_bytes = 0
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"colbert-{device}")

    @contextmanager
    def _execution_context(self):
        # Create and use the stream on this worker's executor thread. Its model
        # and cached tensors stay on the same stream for their entire lifetime.
        if self.device == "cpu":
            yield
        else:
            import torch

            if self.stream is None:
                self.stream = torch.cuda.Stream(device=self.device)
            with torch.cuda.stream(self.stream):
                yield

    @staticmethod
    def _index_version(path):
        try:
            stat = path.stat()
            return stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns
        except FileNotFoundError:
            return None

    def _patient_index(self, path, identity, text):
        cached = self.patients.pop(path, None)
        if cached is not None:
            self.patient_cache_bytes -= cached["bytes"]
            if cached["version"] == self._index_version(path):
                self.patients[path] = cached
                self.patient_cache_bytes += cached["bytes"]
                return cached["chunks"], cached["vectors"], cached["batches"], True
        with cache_lock(path):
            try:
                chunks, vectors = load_index(path, identity, text)
                cache_hit = True
            except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
                chunks = chunk_text(text, self.encoder.tokenizer, self.config)
                vectors = []
                for start in range(0, len(chunks), self.config.batch_size):
                    batch = chunks[start : start + self.config.batch_size]
                    encoded = normalized_vectors(self.encoder.encode([c["text"] for c in batch]))
                    if len(encoded) != len(batch):
                        raise ValueError("ColBERT encoder returned the wrong number of documents")
                    vectors.extend(encoded)
                save_index(path, identity, chunks, vectors)
                cache_hit = False
            version = self._index_version(path)
        # Bound both host vectors and padded GPU batches before allocating.
        size = sum(v.nbytes for v in vectors)
        if self.device != "cpu":
            for offset in range(0, len(vectors), self.config.score_batch_size):
                batch = vectors[offset:offset + self.config.score_batch_size]
                size += len(batch) * max(len(v) for v in batch) * (batch[0].shape[1] * 4 + 1)
        batches = None
        if size <= _PATIENT_CACHE_BYTES:
            while self.patients and (self.patient_cache_bytes + size > _PATIENT_CACHE_BYTES
                                    or len(self.patients) >= _PATIENT_CACHE_ENTRIES):
                _, removed = self.patients.popitem(last=False)
                self.patient_cache_bytes -= removed["bytes"]
                del removed
            if self.device != "cpu":
                import torch

                with torch.inference_mode():
                    batches = list(_maxsim_batches(vectors, device=self.device,
                                                   batch_size=self.config.score_batch_size))
            self.patients[path] = {"version": version, "chunks": chunks, "vectors": vectors,
                                   "batches": batches, "bytes": size}
            self.patient_cache_bytes += size
        return chunks, vectors, batches, cache_hit

    def retrieve(self, text, queries, top_k):
        with self._execution_context():
            return self._retrieve(text, queries, top_k)

    def _retrieve(self, text, queries, top_k):
        if self.encoder is None:
            from .colbert_encoder import ColBERTEncoder

            self.encoder = ColBERTEncoder(self.config, self.device, self.folder)
        identity = {**self.signature, "source_sha256": hashlib.sha256(text.encode()).hexdigest()}
        key = digest(identity)
        path = Path(self.config.cache_dir).expanduser() / "patients" / key[:2] / (key + ".npz")
        chunks, vectors, batches, cache_hit = self._patient_index(path, identity, text)
        missing = list(dict.fromkeys(q for q in queries if q not in self.queries))
        for start in range(0, len(missing), self.config.batch_size):
            batch = missing[start : start + self.config.batch_size]
            values = normalized_vectors(self.encoder.encode(batch, query=True))
            if len(values) != len(batch):
                raise ValueError("ColBERT encoder returned the wrong number of queries")
            self.queries.update(zip(batch, values, strict=True))
        hits = []
        for query in queries:
            scores = maxsim_scores(
                self.queries[query],
                vectors,
                device=self.device,
                batch_size=self.config.score_batch_size,
                prepared_batches=batches,
            )
            order = np.argsort(-scores, kind="stable")[:top_k]
            hits.append(
                [
                    {
                        **chunks[int(i)],
                        "chunk_index": int(i),
                        "rank": rank + 1,
                        "score": float(scores[i]),
                    }
                    for rank, i in enumerate(order)
                ]
            )
        while len(self.queries) > 512:
            self.queries.popitem(last=False)
        # The manifest is patient-independent; only vectors of exactly this text are searched.
        return {
            "context": render_context(text, [h for group in hits for h in group]),
            "queries": queries,
            "hits": hits,
            "source_sha256": identity["source_sha256"],
            "index_key": key,
            "index_path": str(path.resolve()),
            "cache_hit": cache_hit,
            "chunk_count": len(chunks),
            "model_signature": self.signature,
            "scope": "retrieved_excerpts",
        }


class ColBERTRetriever:
    def __init__(self, config):
        from .colbert_encoder import resolve_checkpoint
        import torch

        self.config = config
        folder, artifact = resolve_checkpoint(config)
        self.signature = {**config.encoding_identity(), **artifact}
        devices = config.devices
        if devices == ("auto",):
            devices = tuple(f"cuda:{i}" for i in range(torch.cuda.device_count())) or ("cpu",)
        self.workers = [
            _DeviceWorker(config, device, folder, self.signature)
            for _ in range(config.workers_per_device)
            for device in devices
        ]
        self.lock, self.next_worker = threading.Lock(), 0
        LOGGER.info(
            "ColBERT retrieval devices=%s workers_per_device=%s total_workers=%s cache=%s",
            devices, config.workers_per_device, len(self.workers), config.cache_dir,
        )

    def retrieve(self, text, features, *, top_k=None):
        top_k = self.config.top_k if top_k is None else top_k
        if type(top_k) is not int or top_k < 1:
            raise ValueError("ColBERT top_k must be a positive integer")
        if not str(text).strip():
            return {
                "context": "",
                "queries": [],
                "hits": [],
                "scope": "retrieved_excerpts",
                "source_sha256": hashlib.sha256(str(text).encode()).hexdigest(),
                "model_signature": self.signature,
                "chunk_count": 0,
            }
        with self.lock:
            worker = self.workers[self.next_worker % len(self.workers)]
            self.next_worker += 1
        queries = [feature_query(f) for f in features]
        return worker.executor.submit(worker.retrieve, str(text), queries, top_k).result()


_retrievers = {}
_retrievers_lock = threading.Lock()


def get_retriever(config=None):
    config = colbert_config_from_mapping(config)
    # Detect local checkpoint replacement even inside a long-running process.
    folder = Path(config.model_name).expanduser()
    local_state = None
    if folder.is_dir():
        local_state = digest(
            [
                (str(p.relative_to(folder)), p.stat().st_size, p.stat().st_mtime_ns)
                for p in sorted(folder.rglob("*"))
                if p.is_file() and p.suffix in {".json", ".txt", ".safetensors"}
            ]
        )
    # A pool is reused across concurrent fold/feature callers.
    key = (
        os.getpid(),
        local_state,
        digest(config.encoding_identity()),
        config.devices,
        config.workers_per_device,
        str(Path(config.cache_dir).expanduser().resolve()),
        config.batch_size,
        config.score_batch_size,
        config.top_k,
    )
    with _retrievers_lock:
        if key not in _retrievers:
            _retrievers[key] = ColBERTRetriever(config)
        return _retrievers[key]


def retrieval_identity(config):
    return {**config.measurement_identity(), **get_retriever(config).signature}
