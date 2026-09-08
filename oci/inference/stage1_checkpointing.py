"""Crash-safe, lightweight checkpoints for one exact Stage 1 context.

The checkpoint identity deliberately covers metadata rather than artifact
bytes.  Resume therefore reads only small JSON markers before loading the
payload that the caller actually needs; it never re-hashes a dataset, model,
or checkpoint file.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import random
import socket
import threading
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence, TypeVar

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

STAGE1_CHECKPOINT_SCHEMA_VERSION = "stage1_context_checkpoint_v1"
STAGE1_CHECKPOINT_IMPLEMENTATION_VERSION = 1

T = TypeVar("T")


class Stage1CheckpointBusyError(RuntimeError):
    """Raised when another process already owns an exact context."""


@dataclass(frozen=True)
class Stage1CheckpointResult:
    """The value and provenance of one checkpointed computation."""

    value: Any
    fingerprint: str
    reused: bool
    seed: int


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _canonical_value(value: Any) -> Any:
    """Return a stable, small JSON representation used only for metadata."""

    if is_dataclass(value) and not isinstance(value, type):
        return _canonical_value(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(child)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_value(child) for child in value]
    if isinstance(value, (set, frozenset)):
        canonical = [_canonical_value(child) for child in value]
        return sorted(canonical, key=lambda child: json.dumps(child, sort_keys=True))
    if isinstance(value, np.ndarray):
        return _canonical_value(value.tolist())
    if isinstance(value, np.generic):
        return _canonical_value(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        if np.isnan(value):
            return {"__nonfinite_float__": "nan"}
        return {"__nonfinite_float__": "inf" if value > 0 else "-inf"}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def lightweight_fingerprint(value: Any) -> str:
    """Hash canonical metadata, never the bytes of referenced artifacts."""

    encoded = json.dumps(
        _canonical_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def dataset_file_identity(path: Path | str) -> dict[str, Any]:
    """Return the inexpensive filesystem identity used by context checkpoints."""

    resolved = Path(path).expanduser().resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def row_id_identity(values: Sequence[Any] | np.ndarray) -> dict[str, Any]:
    """Summarize ordered row IDs without consulting any row payload bytes."""

    row_ids = [_canonical_value(value) for value in list(values)]
    return {
        "count": len(row_ids),
        "first": row_ids[0] if row_ids else None,
        "last": row_ids[-1] if row_ids else None,
        "ordered_id_fingerprint": lightweight_fingerprint(row_ids),
    }


def _write_json_file(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    _fsync_file(path)


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        _write_json_file(temporary, value)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _safe_unit_parts(unit: str) -> tuple[str, ...]:
    raw_parts = str(unit).strip("/").split("/")
    if not raw_parts or any(part in {"", ".", ".."} for part in raw_parts):
        raise ValueError(f"invalid Stage 1 checkpoint unit: {unit!r}")
    parts = []
    for part in raw_parts:
        normalized = "".join(
            character if character.isalnum() or character in {"-", "_", "."} else "_"
            for character in part
        )
        if not normalized:
            raise ValueError(f"invalid Stage 1 checkpoint unit segment: {part!r}")
        parts.append(normalized)
    return tuple(parts)


class _PayloadCodec:
    """Typed manifest codec using JSON, NPY, and Parquet; never pickle."""

    _TYPE_KEY = "__oci_stage1_checkpoint_type__"

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._counter = 0
        self.files: list[dict[str, Any]] = []

    def encode(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return self.encode(value.detach().cpu().numpy())
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                return {
                    self._TYPE_KEY: "object_array",
                    "shape": list(value.shape),
                    "values": self.encode(value.tolist()),
                }
            path = self._next_path("array", ".npy")
            with path.open("wb") as handle:
                np.save(handle, value, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            self._record(path, "npy")
            return {self._TYPE_KEY: "ndarray", "file": path.name}
        if isinstance(value, pd.DataFrame):
            path = self._next_path("frame", ".parquet")
            value.to_parquet(path, index=False)
            _fsync_file(path)
            self._record(path, "parquet")
            return {self._TYPE_KEY: "dataframe", "file": path.name}
        if isinstance(value, pd.Series):
            return {
                self._TYPE_KEY: "series",
                "name": self.encode(value.name),
                "values": self.encode(value.to_numpy()),
            }
        if isinstance(value, Mapping):
            if all(isinstance(key, str) for key in value) and self._TYPE_KEY not in value:
                return {str(key): self.encode(child) for key, child in value.items()}
            return {
                self._TYPE_KEY: "mapping",
                "items": [
                    [self.encode(key), self.encode(child)] for key, child in value.items()
                ],
            }
        if isinstance(value, tuple):
            return {
                self._TYPE_KEY: "tuple",
                "items": [self.encode(child) for child in value],
            }
        if isinstance(value, list):
            return [self.encode(child) for child in value]
        if isinstance(value, np.generic):
            return self.encode(value.item())
        if isinstance(value, Path):
            return {self._TYPE_KEY: "path", "value": str(value)}
        if isinstance(value, (datetime, date)):
            return {self._TYPE_KEY: "datetime", "value": value.isoformat()}
        if isinstance(value, float) and not np.isfinite(value):
            if np.isnan(value):
                label = "nan"
            else:
                label = "inf" if value > 0 else "-inf"
            return {self._TYPE_KEY: "nonfinite_float", "value": label}
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise TypeError(
            f"unsupported Stage 1 checkpoint payload type: {type(value).__name__}"
        )

    def decode(self, node: Any) -> Any:
        if isinstance(node, list):
            return [self.decode(child) for child in node]
        if not isinstance(node, Mapping):
            return node
        kind = node.get(self._TYPE_KEY)
        if kind is None:
            return {str(key): self.decode(value) for key, value in node.items()}
        if kind == "ndarray":
            path = self._validated_file(node, "npy")
            return np.load(path, allow_pickle=False)
        if kind == "object_array":
            values = np.asarray(self.decode(node["values"]), dtype=object)
            return values.reshape(tuple(int(size) for size in node["shape"]))
        if kind == "dataframe":
            path = self._validated_file(node, "parquet")
            return pd.read_parquet(path)
        if kind == "series":
            return pd.Series(self.decode(node["values"]), name=self.decode(node["name"]))
        if kind == "mapping":
            return {self.decode(key): self.decode(value) for key, value in node["items"]}
        if kind == "tuple":
            return tuple(self.decode(child) for child in node["items"])
        if kind == "list":
            return [self.decode(child) for child in node["items"]]
        if kind == "path":
            return Path(str(node["value"]))
        if kind == "datetime":
            return str(node["value"])
        if kind == "nonfinite_float":
            return {"nan": np.nan, "inf": np.inf, "-inf": -np.inf}[str(node["value"])]
        raise ValueError(f"unknown checkpoint payload kind: {kind!r}")

    def _next_path(self, prefix: str, suffix: str) -> Path:
        self._counter += 1
        return self.directory / f"{prefix}_{self._counter:06d}{suffix}"

    def _record(self, path: Path, kind: str) -> None:
        self.files.append(
            {"name": path.name, "kind": kind, "size": int(path.stat().st_size)}
        )

    def _validated_file(self, node: Mapping[str, Any], expected_kind: str) -> Path:
        name = str(node.get("file", ""))
        if not name or Path(name).name != name:
            raise ValueError("checkpoint payload contains an invalid file name")
        path = self.directory / name
        expected = next(
            (
                item
                for item in self.files
                if item.get("name") == name and item.get("kind") == expected_kind
            ),
            None,
        )
        if expected is None:
            raise ValueError(f"checkpoint manifest does not declare {name}")
        if not path.is_file() or int(path.stat().st_size) != int(expected["size"]):
            raise ValueError(f"checkpoint payload file is missing or truncated: {name}")
        return path


class Stage1CheckpointStore:
    """Versioned checkpoint store for one exact Stage 1 context."""

    def __init__(
        self,
        root: Path | str,
        *,
        context_identity: Mapping[str, Any],
        base_seed: int,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.context_identity = _canonical_value(dict(context_identity))
        self.base_seed = int(base_seed)
        self.base_fingerprint = lightweight_fingerprint(
            {
                "schema_version": STAGE1_CHECKPOINT_SCHEMA_VERSION,
                "implementation_version": STAGE1_CHECKPOINT_IMPLEMENTATION_VERSION,
                "context": self.context_identity,
                "base_seed": self.base_seed,
            }
        )
        self._tokens: dict[str, str] = {}
        self._progress_lock = threading.Lock()
        self._progress_path = self.root / "checkpoint_progress.json"

    @contextmanager
    def context_lock(self) -> Iterator[None]:
        """Own this context exclusively, with automatic release after a crash."""

        lock_path = self.root / "context.lock"
        with lock_path.open("a+", encoding="utf-8") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                handle.seek(0)
                owner = handle.read().strip() or "unknown owner"
                raise Stage1CheckpointBusyError(
                    f"Stage 1 context is already running ({owner})"
                ) from exc
            handle.seek(0)
            handle.truncate()
            handle.write(
                json.dumps(
                    {
                        "hostname": socket.gethostname(),
                        "pid": os.getpid(),
                        "started_at": _utc_now(),
                        "base_fingerprint": self.base_fingerprint,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
            try:
                self._mark_progress("context", "running")
                try:
                    yield
                except BaseException as exc:
                    self._mark_progress(
                        "context",
                        "failed",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    raise
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def run(
        self,
        unit: str,
        compute: Callable[[], T],
        *,
        parameters: Optional[Mapping[str, Any]] = None,
        dependencies: Sequence[str] = (),
        rows: Optional[Mapping[str, Any]] = None,
        validator: Optional[Callable[[T], None]] = None,
        deterministic_seed: bool = False,
    ) -> Stage1CheckpointResult:
        """Load a compatible unit or compute and publish it atomically."""

        fingerprint, seed, dependency_tokens = self._unit_identity(
            unit,
            parameters=parameters,
            dependencies=dependencies,
            rows=rows,
        )
        loaded = self._load(unit, fingerprint, rows=rows, validator=validator)
        if loaded is not None:
            self._tokens[unit] = fingerprint
            self._mark_progress(unit, "reused", fingerprint=fingerprint, seed=seed)
            logger.info("Reused Stage 1 checkpoint unit=%s", unit)
            return Stage1CheckpointResult(loaded, fingerprint, True, seed)

        self._mark_progress(unit, "running", fingerprint=fingerprint, seed=seed)
        logger.info("Computing Stage 1 checkpoint unit=%s", unit)
        try:
            if deterministic_seed:
                seed_everything(seed)
            value = compute()
            if validator is not None:
                validator(value)
            self._save(
                unit,
                fingerprint,
                value,
                parameters=parameters,
                dependency_tokens=dependency_tokens,
                rows=rows,
                seed=seed,
            )
        except BaseException as exc:
            self._mark_progress(
                unit,
                "failed",
                fingerprint=fingerprint,
                seed=seed,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        self._tokens[unit] = fingerprint
        self._mark_progress(unit, "completed", fingerprint=fingerprint, seed=seed)
        return Stage1CheckpointResult(value, fingerprint, False, seed)

    def token(self, unit: str) -> str:
        try:
            return self._tokens[unit]
        except KeyError as exc:
            raise RuntimeError(f"checkpoint dependency has not completed: {unit}") from exc

    def mark_context_complete(self) -> None:
        self._mark_progress("context", "completed")

    def _unit_identity(
        self,
        unit: str,
        *,
        parameters: Optional[Mapping[str, Any]],
        dependencies: Sequence[str],
        rows: Optional[Mapping[str, Any]],
    ) -> tuple[str, int, dict[str, str]]:
        _safe_unit_parts(unit)
        dependency_tokens = {dependency: self.token(dependency) for dependency in dependencies}
        identity = {
            "schema_version": STAGE1_CHECKPOINT_SCHEMA_VERSION,
            "implementation_version": STAGE1_CHECKPOINT_IMPLEMENTATION_VERSION,
            "base_fingerprint": self.base_fingerprint,
            "unit": unit,
            "parameters": _canonical_value(dict(parameters or {})),
            "dependencies": dependency_tokens,
            "rows": _canonical_value(dict(rows or {})),
        }
        fingerprint = lightweight_fingerprint(identity)
        seed_material = lightweight_fingerprint(
            {"base_seed": self.base_seed, "unit": unit, "fingerprint": fingerprint}
        )
        seed = int(seed_material[:8], 16) % (2**31 - 1)
        return fingerprint, seed, dependency_tokens

    def _checkpoint_dir(self, unit: str, fingerprint: str) -> Path:
        return self.root.joinpath("units", *_safe_unit_parts(unit), fingerprint)

    def _load(
        self,
        unit: str,
        fingerprint: str,
        *,
        rows: Optional[Mapping[str, Any]],
        validator: Optional[Callable[[T], None]],
    ) -> Optional[T]:
        directory = self._checkpoint_dir(unit, fingerprint)
        marker_path = directory / "complete.json"
        manifest_path = directory / "payload.json"
        if not marker_path.is_file():
            return None
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            if marker.get("status") != "complete":
                raise ValueError("checkpoint marker is not complete")
            if marker.get("schema_version") != STAGE1_CHECKPOINT_SCHEMA_VERSION:
                raise ValueError("checkpoint schema version changed")
            if marker.get("implementation_version") != STAGE1_CHECKPOINT_IMPLEMENTATION_VERSION:
                raise ValueError("checkpoint implementation version changed")
            if marker.get("unit") != unit or marker.get("fingerprint") != fingerprint:
                raise ValueError("checkpoint marker identity does not match its path")
            if marker.get("base_fingerprint") != self.base_fingerprint:
                raise ValueError("checkpoint context identity changed")
            if marker.get("rows") != _canonical_value(dict(rows or {})):
                raise ValueError("checkpoint row identity changed")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("schema_version") != STAGE1_CHECKPOINT_SCHEMA_VERSION:
                raise ValueError("checkpoint payload schema version changed")
            codec = _PayloadCodec(directory)
            codec.files = list(manifest.get("files") or [])
            value = codec.decode(manifest["root"])
            if validator is not None:
                validator(value)
            return value
        except Exception as exc:
            logger.warning("Ignoring incomplete or corrupt Stage 1 checkpoint %s: %s", unit, exc)
            self._mark_progress(
                unit,
                "invalid",
                fingerprint=fingerprint,
                error=f"{type(exc).__name__}: {exc}",
            )
            return None

    def _save(
        self,
        unit: str,
        fingerprint: str,
        value: Any,
        *,
        parameters: Optional[Mapping[str, Any]],
        dependency_tokens: Mapping[str, str],
        rows: Optional[Mapping[str, Any]],
        seed: int,
    ) -> None:
        target = self._checkpoint_dir(unit, fingerprint)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.parent / f".{fingerprint}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        temporary.mkdir()
        try:
            codec = _PayloadCodec(temporary)
            root = codec.encode(value)
            _write_json_file(
                temporary / "payload.json",
                {
                    "schema_version": STAGE1_CHECKPOINT_SCHEMA_VERSION,
                    "root": root,
                    "files": codec.files,
                },
            )
            _write_json_file(
                temporary / "complete.json",
                {
                    "status": "complete",
                    "schema_version": STAGE1_CHECKPOINT_SCHEMA_VERSION,
                    "implementation_version": STAGE1_CHECKPOINT_IMPLEMENTATION_VERSION,
                    "unit": unit,
                    "fingerprint": fingerprint,
                    "base_fingerprint": self.base_fingerprint,
                    "parameters": _canonical_value(dict(parameters or {})),
                    "dependencies": dict(dependency_tokens),
                    "rows": _canonical_value(dict(rows or {})),
                    "seed": int(seed),
                    "completed_at": _utc_now(),
                    "validation": "marker_identity_row_ids_file_presence_and_size",
                    "content_hashing": False,
                },
            )
            _fsync_directory(temporary)
            if target.exists():
                invalid = target.with_name(
                    f".{target.name}.invalid.{os.getpid()}.{uuid.uuid4().hex}"
                )
                os.replace(target, invalid)
            os.replace(temporary, target)
            _fsync_directory(target.parent)
        finally:
            if temporary.exists():
                for child in temporary.iterdir():
                    child.unlink(missing_ok=True)
                temporary.rmdir()

    def _mark_progress(
        self,
        unit: str,
        status: str,
        *,
        fingerprint: Optional[str] = None,
        seed: Optional[int] = None,
        error: Optional[str] = None,
    ) -> None:
        with self._progress_lock:
            progress: dict[str, Any]
            try:
                progress = json.loads(self._progress_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError):
                progress = {}
            if progress.get("base_fingerprint") != self.base_fingerprint:
                progress = {
                    "schema_version": STAGE1_CHECKPOINT_SCHEMA_VERSION,
                    "base_fingerprint": self.base_fingerprint,
                    "context_identity": self.context_identity,
                    "started_at": _utc_now(),
                    "units": {},
                }
            entry: dict[str, Any] = {"status": status, "updated_at": _utc_now()}
            if fingerprint is not None:
                entry["fingerprint"] = fingerprint
            if seed is not None:
                entry["seed"] = int(seed)
            if error is not None:
                entry["error"] = error[:2000]
            progress.setdefault("units", {})[unit] = entry
            progress["updated_at"] = _utc_now()
            _write_json_atomic(self._progress_path, progress)


def seed_everything(seed: int) -> None:
    """Seed one leaf so restored siblings cannot perturb neural initialization."""

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


__all__ = [
    "STAGE1_CHECKPOINT_IMPLEMENTATION_VERSION",
    "STAGE1_CHECKPOINT_SCHEMA_VERSION",
    "Stage1CheckpointBusyError",
    "Stage1CheckpointResult",
    "Stage1CheckpointStore",
    "dataset_file_identity",
    "lightweight_fingerprint",
    "row_id_identity",
    "seed_everything",
]
