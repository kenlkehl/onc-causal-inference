"""Bind pilot inputs and audit exclusion of outer-test and inner-validation labels."""
import json
import numpy as np
import pandas as pd
from common import (HERE, ROOT, RUN, SOURCE, SAVED, CACHE, POLICY, assert_scope,
                    labels, nuisance_path, read, sha, split, write, now)


def main():
    from oci.inference.tfidf_topic_stage1 import _float_hex_sha256
    outer = split()
    assert len(outer["fit_row_ids"]) == 800 and len(outer["heldout_row_ids"]) == 200
    assert len(outer["inner_splits"]) == 5
    files = [SAVED / "split.json", SAVED / "training_labels.parquet",
             SOURCE / "run_config.json", SOURCE / "resolved_neural_query_config.json",
             *[CACHE / name for name in ("metadata.json", "offsets.npy", "chunk_embeddings.npy", "chunk_texts.jsonl")],
             ROOT / "oci/inference/neural_cohort_witness.py",
             *[HERE / name for name in ("common.py", "perturb.py", "probes.py", "prepare.py", "fit.py", "PILOT_PROTOCOL_2026-09-22.md")]]
    audit = []
    for position, part in enumerate(outer["inner_splits"], 1):
        train, valid = part["fit_row_ids"], part["heldout_row_ids"]
        assert_scope(train, valid, outer)
        qdir = SOURCE / f"components/neural_queries/outer_001_inner_{position:03d}"
        with np.load(qdir / "queries.npz", allow_pickle=False) as values:
            assert values["fit_row_ids"].tolist() == train
        paths = [qdir / name for name in ("queries.npz", "query_records.json", "scores.parquet")]
        paths += [nuisance_path(position), nuisance_path(position).parent / "context_metadata.json"]
        files += paths
        metadata = read(paths[-1])
        assert metadata["fit_row_ids"] == train and metadata["heldout_row_ids"] == valid
        assert metadata["registered_heldout_labels_accessed"] is False
        nesting = metadata["selection_nesting"]
        assert nesting["selection_frozen_before_registered_heldout_transform"] is True
        assert set(nesting["model_fit_row_ids"]).isdisjoint(nesting["calibration_row_ids"])
        assert set(nesting["model_fit_row_ids"]) | set(nesting["calibration_row_ids"]) == set(train)
        nuisances = pd.read_parquet(paths[-2])
        assert nuisances._oci_row_id.is_unique
        assert set(nuisances._oci_row_id) == set(train) | set(valid)
        assert set(nuisances.loc[nuisances.prediction_scope == "fit_oof", "_oci_row_id"]) == set(train)
        assert set(nuisances.loc[nuisances.prediction_scope == "external_heldout", "_oci_row_id"]) == set(valid)
        for _, row in nuisances.iterrows():
            used = set(map(int, row["fit_row_ids"]))
            assert int(row["_oci_row_id"]) not in used
            assert used <= set(train)
        inventory = metadata["artifact_inventory"]["entries"]
        entry = next(x for x in inventory if x["logical_name"] == "nuisance_predictions")
        assert entry["sha256"] == sha(paths[-2])
        training = labels(train)
        for target in ("treatment", "outcome"):
            assert _float_hex_sha256(training[target].to_numpy()) == metadata[f"registered_fit_{target}_sha256"]
        audit.append({"inner_fold": position, "fit_rows": len(train), "validation_rows": len(valid),
                      "nuisance_rows_with_fit_provenance_checked": len(nuisances),
                      "nuisance_self_training_violations": 0, "outside_outer_training_rows": 0,
                      "inner_validation_rows_in_any_nuisance_fit": 0, "training_label_hashes_match": True,
                      "upstream_registered_heldout_labels_accessed": False})
    bindings = {str(p.relative_to(ROOT)): {"bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns, "sha256": sha(p)}
                for p in files}
    manifest = {"created_at": now(), "policy": POLICY, "files": bindings,
                "outer_fit_row_ids": outer["fit_row_ids"], "forbidden_outer_test_row_ids": outer["heldout_row_ids"],
                "inner_splits": outer["inner_splits"], "scope_audit": audit,
                "validation_label_access": "evaluation only, after all edits, probe predictions, and LLM proposals freeze",
                "query_scope_allowlist": [f"outer_001_inner_{i:03d}" for i in range(1, 6)],
                "oracle_columns_loaded": False}
    path = RUN / "manifest.json"
    if path.exists():
        previous = read(path)
        for key in ("policy", "files", "inner_splits", "scope_audit"):
            assert previous[key] == manifest[key], key
    else:
        write(path, manifest)
    write(RUN / "ISOLATION_AUDIT_2026-09-22.json", {"at": now(), "outer_fold": 1, "outer_training_patients": 800,
          "outer_test_patients_excluded": 200, "folds": audit, "nuisance_provenance_rows_checked": 4000})
    print(json.dumps({"prepared": str(RUN), "audited_nuisance_rows": 4000, "outer_test_excluded": 200}), flush=True)


if __name__ == "__main__":
    main()
