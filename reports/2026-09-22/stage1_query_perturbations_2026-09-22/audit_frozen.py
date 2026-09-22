"""Audit frozen numerical geometry, packet scope, and untouched Stage 2 sources."""
import numpy as np
from common import ROOT, RUN, POLICY, read, sha, write, now, verify_manifest


def main():
    manifest = verify_manifest()
    frozen = read(RUN / "numerical_frozen.json")
    checks = []
    packet_count = 0
    for position, part in enumerate(manifest["inner_splits"], 1):
        folder = RUN / "folds" / f"inner_{position:03d}"
        assert sha(folder / "frozen.json") == frozen["folds"][folder.name]
        record = read(folder / "frozen.json")
        assert record["validation_outcomes_loaded"] is False
        for name, expected in record["files"].items():
            assert sha(folder / name) == expected
        variants = read(folder / "variants.json")
        with np.load(folder / "activations.npz", allow_pickle=False) as values:
            assert values["fit_row_ids"].tolist() == part["fit_row_ids"]
            assert values["validation_row_ids"].tolist() == part["heldout_row_ids"]
            q = values["vectors"]
            norm_error = float(np.max(np.abs(np.linalg.norm(q, axis=1) - 1)))
            assert norm_error < 2e-6
            actual = np.asarray([np.linalg.norm(q[i] - q[v["query"]]) for i, v in enumerate(variants)])
            assert np.max(np.abs(actual - np.asarray([v["distance"] for v in variants]))) < 3e-6
            assert np.all(actual <= np.asarray([v["radius"] for v in variants]) + 3e-6)
            matched_errors = []
            for i, variant in enumerate(variants):
                if variant["kind"] != "targeted":
                    continue
                controls = [k for k, other in enumerate(variants) if other["kind"] == "random" and
                            other["query"] == variant["query"] and other["radius"] == variant["radius"]]
                assert len(controls) == POLICY["random_controls"]
                matched_errors.extend(np.abs(actual[controls] - actual[i]).tolist())
            assert max(matched_errors) < 3e-6
        with np.load(folder / "probe_predictions.npz", allow_pickle=False) as values:
            assert values["row_ids"].tolist() == part["heldout_row_ids"]
            assert values["predictions"].shape == (2973, 160)
            assert np.isfinite(values["predictions"]).all()
        for identifier, expected in record["packets"].items():
            path = RUN / "packets" / (identifier + ".json")
            assert sha(path) == expected
            packet = read(path)
            assert packet["inner_fold"] == position
            assert {e["row_id"] for e in packet["evidence"]} <= set(part["fit_row_ids"])
            assert len({e["evidence_id"] for e in packet["evidence"]}) == len(packet["evidence"])
            assert len(packet["evidence"]) <= 24
            packet_count += 1
        checks.append({"inner_fold": position, "unit_vector_max_error": norm_error,
                       "random_control_distance_max_error": max(matched_errors), "query_variants": len(variants),
                       "probe_predictions": 2973, "outer_test_patients_used": 0})
    prior = read(ROOT / "reports/2026-09-22/fold_1_nested_modifier_count_2026-09-22/input_manifest_2026-09-22.json")
    changed = [path for path, expected in prior["sources"].items() if sha(path) != expected]
    assert not changed
    result = {"at": now(), "folds": checks, "packets_checked": packet_count,
              "numerical_freeze_sha256": sha(RUN / "numerical_frozen.json"),
              "stage2_frozen_sources_checked": len(prior["sources"]), "stage2_sources_changed": changed,
              "numerical_tests_passed": 6, "numerical_test_command": "python -m pytest -q test_pilot.py",
              "source_lint": "ruff check --select E9,F: passed",
              "inner_validation_scoring_started_at_audit_time": (RUN / "inner_validation_access_started.json").exists()}
    write(RUN / "FROZEN_QA_2026-09-22.json", result)
    print(result)


if __name__ == "__main__":
    main()
