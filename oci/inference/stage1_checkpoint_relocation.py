"""Verified path relocation for existing Stage 1 text-model checkpoints.

Only an explicit migration manifest enables this compatibility path. Numerical
settings, row identities, unit fingerprints, and deterministic seeds stay fixed.
"""
from __future__ import annotations

import copy
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

MANIFEST_NAME = "stage1_checkpoint_relocation.json"
SCHEMA_VERSION = "stage1_checkpoint_relocation_v1"


@lru_cache(maxsize=16)
def _dataset_digest(path: str, stat_identity: tuple[int, ...]) -> str:
    # Cache within a worker only, and invalidate when filesystem metadata changes.
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def relocated_checkpoint_identity(
    *, output_dir: Path, dataset_identity: Mapping[str, Any], scientific_config: Mapping[str, Any]
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Restore original logical paths only after verifying data and all settings.

    Runtime configs continue to use their physical paths. This function operates
    on a separate metadata copy solely for checkpoint fingerprints and seeds.
    """
    from .stage1_checkpointing import lightweight_fingerprint

    manifest_path = output_dir / MANIFEST_NAME
    if not manifest_path.exists():
        return dataset_identity, scientific_config
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported Stage 1 checkpoint relocation manifest")
    dataset = Path(str(dataset_identity["path"])).resolve(strict=True)
    if str(dataset) != manifest["dataset_path"]:
        raise ValueError("Relocated dataset path does not match the migration manifest")
    original = manifest["original_dataset_identity"]
    stat = dataset.stat()
    stat_key = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    if (
        stat.st_size != original["size"]
        or _dataset_digest(str(dataset), stat_key) != manifest["dataset_sha256"]
    ):
        raise ValueError("Relocated dataset checksum does not match the original snapshot")
    after = dataset.stat()
    after_key = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    if after_key != stat_key:
        raise ValueError("Relocated dataset changed while verifying its checksum")
    original_output = Path(manifest["original_output_dir"])
    physical_cache = str(output_dir.resolve() / "components/embedding_cache/cache")
    logical_cache = str(original_output / "components/embedding_cache/cache")
    normalized = copy.deepcopy(dict(scientific_config))
    architecture = normalized.get("architecture", {})
    for name in ("multi_model_forest", "multi_model_agentic_forest"):
        embedding = (architecture.get(name) or {}).get("embedding_contrast")
        if embedding and embedding.get("cache_dir") == physical_cache:
            embedding["cache_dir"] = logical_cache
    if lightweight_fingerprint(normalized) != manifest["original_applied_config_fingerprint"]:
        raise ValueError("Relocated Stage 1 scientific settings do not match the original checkpoints")
    return dict(original), normalized
