from __future__ import annotations

import copy
import hashlib
import json
import os

import pytest

from oci.inference.stage1_checkpointing import dataset_file_identity, lightweight_fingerprint
from oci.inference.stage1_checkpoint_relocation import (
    MANIFEST_NAME, SCHEMA_VERSION, relocated_checkpoint_identity,
)


def _case(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.write_bytes(b"original bytes")
    original = {"path": "/unmounted/source/dataset", "size": dataset.stat().st_size, "mtime_ns": 123}
    config = {"epochs": 12, "architecture": {"multi_model_forest": {
        "embedding_contrast": {"cache_dir": str(tmp_path / "components/embedding_cache/cache")}
    }}}
    logical = copy.deepcopy(config)
    logical["architecture"]["multi_model_forest"]["embedding_contrast"]["cache_dir"] = "/unmounted/output/components/embedding_cache/cache"
    manifest = dict(schema_version=SCHEMA_VERSION, dataset_path=str(dataset),
        original_dataset_identity=original, dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
        original_output_dir="/unmounted/output", original_applied_config_fingerprint=lightweight_fingerprint(logical))
    (tmp_path / MANIFEST_NAME).write_text(json.dumps(manifest))
    def verify(settings=config):
        return relocated_checkpoint_identity(output_dir=tmp_path,
            dataset_identity=dataset_file_identity(dataset), scientific_config=settings)
    return dataset, config, manifest, verify


def test_relocation_preserves_runtime_config_and_ignores_copy_timestamp(tmp_path):
    dataset, config, manifest, verify = _case(tmp_path)
    before = copy.deepcopy(config)
    identity, normalized = verify()
    assert identity == manifest["original_dataset_identity"]
    assert config == before
    assert normalized != config
    assert dataset_file_identity(dataset)["mtime_ns"] != identity["mtime_ns"]


def test_relocation_rejects_changed_bytes_even_with_same_size_and_mtime(tmp_path):
    dataset, _config, _manifest, verify = _case(tmp_path)
    verify()  # Populate the per-worker digest cache.
    stat = dataset.stat()
    dataset.write_bytes(b"modified bytes")
    os.utime(dataset, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(ValueError, match="checksum"):
        verify()


@pytest.mark.parametrize("change", ["epochs", "cache_dir", "path", "schema", "hash"])
def test_relocation_rejects_unverified_changes(tmp_path, change):
    _dataset, config, manifest, verify = _case(tmp_path)
    if change == "epochs":
        config["epochs"] += 1
    elif change == "cache_dir":
        config["architecture"]["multi_model_forest"]["embedding_contrast"]["cache_dir"] = "/other/cache"
    elif change == "path":
        manifest["dataset_path"] = "/other/dataset"
    elif change == "schema":
        manifest["schema_version"] = "unknown"
    else:
        manifest["dataset_sha256"] = "0" * 64
    (tmp_path / MANIFEST_NAME).write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        verify()


def test_no_manifest_retains_existing_identity_behavior(tmp_path):
    dataset, config, _manifest, verify = _case(tmp_path)
    (tmp_path / MANIFEST_NAME).unlink()
    identity, normalized = verify()
    assert identity == dataset_file_identity(dataset)
    assert normalized is config
