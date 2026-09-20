"""Record the user-authorized extraction reliability revision, preserving v1 inputs."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path

from compare import HERE, ROOT, immutable_json, now, read_json, sha256

OUT = HERE / "results"
REVISION = OUT / "revisions/reliability_v2"


def main():
    manifest_path = REVISION / "manifest.json"
    if manifest_path.exists():
        raise RuntimeError("Revision already frozen; validate it rather than rewriting it")
    stop = read_json(REVISION / "stop_original.json")
    # Preserve content hashes of every completed or partial scientific checkpoint
    # before the new worker starts. Logs and other mutable monitoring are excluded.
    inventory = {}
    schemas = Counter()
    def inspect(path):
        data = path.read_bytes()
        schema = json.loads(data).get("schema_version", "unspecified") if path.name == "complete.json" else None
        return str(path), {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}, schema

    names = ("complete.json", "result.json", "input.json", "serial_extraction.json")

    def inspect_tree(root):
        records = []
        for directory, _children, filenames in os.walk(root):
            records.extend(inspect(Path(directory) / name) for name in names if name in filenames)
        return records

    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = []
        for directory, children, filenames in os.walk(OUT / "refresh"):
            if Path(directory).name in {"batches", "pages"}:
                futures.extend(pool.submit(inspect_tree, Path(directory) / child) for child in children)
                children.clear()
            for name in names:
                if name in filenames:
                    futures.append(pool.submit(lambda p: [inspect(p)], Path(directory) / name))
        print({"checkpoint_groups_to_hash": len(futures)}, flush=True)
        for index, future in enumerate(as_completed(futures), start=1):
            for path, record, schema in future.result():
                inventory[path] = record
                if schema is not None:
                    schemas[schema] += 1
            if index % 16 == 0:
                print({"checkpoint_groups_hashed": index, "checkpoint_files_hashed": len(inventory)}, flush=True)
    inventory_path = REVISION / "checkpoint_inventory_before.json"
    immutable_json(inventory_path, {"recorded_at": now(), "completion_schemas": dict(schemas),
                                   "files": inventory})
    original_manifest = OUT / "inputs/source_manifest.json"
    original = read_json(original_manifest)
    overrides = {}
    for relative in ("oci/inference/plain_handoff_stage2.py",
                     "oci/inference/plain_handoff_stage2_analysis.py",
                     str((HERE / "compare.py").relative_to(ROOT))):
        path = ROOT / relative
        archive = REVISION / "before" / relative
        assert sha256(archive) == original[str(path)]["sha256"], relative
        overrides[str(path)] = {"before_sha256": sha256(archive), "before_archive": str(archive),
                                "after_sha256": sha256(path)}
    config_path = REVISION / "refresh_config.json"
    original_config = read_json(OUT / "inputs/refresh_config.json")
    changed = {"extraction_max_tokens": 4096, "extraction_reasoning_max_tokens": 32768,
               "extraction_stream": True, "extraction_deferred_retry_passes": 1}
    immutable_json(config_path, {**original_config, **changed})
    additional = [ROOT / "oci/inference/stage2_request_audit.py", HERE / "report.py",
                  Path(__file__).resolve(), HERE / "resume_reliability.py"]
    frozen_inputs = [*(OUT / "inputs").glob("*.json"), inventory_path,
                     REVISION / "stop_original.json"]
    immutable_json(manifest_path, {
        "schema": "selection_comparison_operational_revision_v1", "revision": "reliability_v2",
        "recorded_at": now(), "original_manifest_sha256": sha256(original_manifest),
        "authorization": "User requested recommended reliability changes, then a larger reasoning budget.",
        "prior_run_stop": stop,
        "source_overrides": overrides,
        "additional_sources": {str(p): {"sha256": sha256(p)} for p in additional},
        "frozen_inputs": {str(p.resolve()): {"sha256": sha256(p)} for p in frozen_inputs},
        "config": {"path": str(config_path), "sha256": sha256(config_path)},
        "config_changes": {k: {"before": original_config.get(k), "after": v} for k, v in changed.items()},
        "checkpoint_compatibility": {
            "patient_and_feature_results": "Normal model/text/definition fingerprint validation remains enforced.",
            "incomplete_serial_chunks": "Output reservation participates in fingerprint; changed plans re-extract normally.",
            "before_completion_schemas": dict(schemas),
            "source_scientific_artifacts_modified": False,
            "oracle_values_opened": False,
            "comparison_policy_changed": False,
        },
    })
    print({"manifest": str(manifest_path), "sha256": sha256(manifest_path),
           "checkpoint_files": len(inventory), "completion_schemas": dict(schemas)}, flush=True)


if __name__ == "__main__":
    main()
