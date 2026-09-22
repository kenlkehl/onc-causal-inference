"""Read-only source access and isolated artifacts for the query-edit pilot."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
RUN = HERE / "pilot_v1"
SOURCE = ROOT / "artifacts/research_all_evidence/five_conf_five_mod_nsclc_full"
SAVED = ROOT / "reports/2026-09-21/fold_1_extracted_logistic_interactions_2026-09-21/inputs"
CACHE = SOURCE / "components/embedding_cache/cache/cecnn_chunk_embeddings_e8d02eb07649"
BANKS = ("treatment", "outcome", "effect")
POLICY = {
    "schema": "outer1_query_perturbation_pilot_v1", "outer_fold": 1,
    "radii": [0.02, 0.05, 0.10, 0.20], "temperature": 0.05,
    "restarts": 3, "epochs": 50, "learning_rate": 0.005,
    "relative_sd_bounds": [0.5, 2.0], "distance_penalty": 0.002,
    "random_controls": 5, "seed": 20260922,
    "probe_logistic_C": 1.0, "probe_effect_penalty_per_row": 0.01,
    "spline_knots": 4, "spline_degree": 2,
    "llm_conditions": ["original", "targeted", "random"],
    "llm_radius": 0.20, "llm_random_control": 0,
    "llm_evidence_patients": 6, "llm_max_candidates": 2,
    "llm_workers": 4, "llm_model": "gemma4-31b",
    "llm_endpoint": "http://sn4622130540:8000/v1",
    "fold_workers": 2, "torch_threads_per_fold": 4,
    "nuisances": "frozen_exact_inner_tfidf_context_oof_and_external_predictions",
    "nuisance_note": "These are honest cached Stage 1 TF-IDF predictions, not the unpersisted original neural optimizer residuals.",
    "overlap_filter": None,
}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def status(path, phase, **extra):
    value = {"at": now(), "phase": phase, **extra}
    write(path, value)
    print(json.dumps(value), flush=True)


def split():
    return read(SAVED / "split.json")


def assert_scope(fit_ids, validation_ids, outer):
    fit, valid = set(map(int, fit_ids)), set(map(int, validation_ids))
    allowed, forbidden = set(outer["fit_row_ids"]), set(outer["heldout_row_ids"])
    assert len(fit) == len(fit_ids) and len(valid) == len(validation_ids)
    assert fit.isdisjoint(valid) and fit | valid == allowed
    assert (fit | valid).isdisjoint(forbidden)


def labels(row_ids):
    """Read only requested observed labels from the outer-training-only table."""
    import pandas as pd
    requested = list(map(int, row_ids))
    assert set(requested) <= set(split()["fit_row_ids"])
    frame = pd.read_parquet(SAVED / "training_labels.parquet",
                            columns=["_oci_row_id", "treatment", "outcome"],
                            filters=[("_oci_row_id", "in", requested)])
    assert frame._oci_row_id.is_unique and set(frame._oci_row_id) == set(requested)
    return frame.set_index("_oci_row_id").loc[requested]


def embeddings(row_ids):
    import numpy as np
    requested = list(map(int, row_ids))
    assert set(requested) <= set(split()["fit_row_ids"])
    offsets = np.load(CACHE / "offsets.npy", allow_pickle=False)
    values = np.load(CACHE / "chunk_embeddings.npy", mmap_mode="r", allow_pickle=False)
    return [np.asarray(values[offsets[i]:offsets[i + 1]], dtype=np.float32) for i in requested]


def chunk_texts(row_ids):
    requested = set(map(int, row_ids))
    assert requested <= set(split()["fit_row_ids"])
    result = {}
    with (CACHE / "chunk_texts.jsonl").open() as stream:
        for row_id, line in enumerate(stream):
            if row_id in requested:
                result[row_id] = read_line = json.loads(line)["chunks"]
                assert isinstance(read_line, list)
    assert set(result) == requested
    return result


def nuisance_path(position):
    return SOURCE / f"components/tfidf/stage1_tfidf_topics/contexts/outer_001_inner_{position:03d}/nuisance_predictions.parquet"


def verify_manifest():
    manifest = read(RUN / "manifest.json")
    assert manifest["policy"] == POLICY
    # Large read-only arrays were hashed once at preparation; verify stat binding.
    for name, record in manifest["files"].items():
        path = ROOT / name
        assert path.stat().st_size == record["bytes"] and path.stat().st_mtime_ns == record["mtime_ns"], name
        if record["bytes"] < 10_000_000:
            assert sha(path) == record["sha256"], name
    return manifest
