"""Verify completed evidence integrity, partitions, coverage, and nuisance audits."""
import collections
import importlib.util
from pathlib import Path
import json

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("numerical_run", HERE / "numerical_2026-09-21.py")
n = importlib.util.module_from_spec(spec)
spec.loader.exec_module(n)


def main():
    from oci.inference.stage2_role_adjudication import _fingerprint

    frozen = n.read(HERE / f"numerical_frozen_{n.DATE}.json")
    manifest = n.read(HERE / f"input_manifest_{n.DATE}.json")
    assert frozen["input_manifest_sha256"] == n.sha(HERE / f"input_manifest_{n.DATE}.json")
    n.verify(frozen["files"])
    n.verify(manifest["sources"])
    n.verify(manifest["input_files"])
    report = n.read(HERE / "statistical_evidence.json")
    split = n.read(n.INPUTS / "inputs/split.json")
    definitions = n.read(n.INPUTS / "inputs/definitions.json")
    ids = {d["feature_id"] for d in definitions}
    assert len(ids) == 352 and set(report["multi_model_evidence"]) == ids
    assert len(report["cells"]) == 305
    train_ids = set(split["fit_row_ids"])
    outer_test = set(split["heldout_row_ids"])
    assert not train_ids & outer_test
    by_fold = {s.get("inner_fold", i): s for i, s in enumerate(split["inner_splits"], 1)}
    cell_counts, statuses, unsuccessful = collections.Counter(), collections.Counter(), []
    groups = collections.defaultdict(list)
    for cell in report["cells"]:
        fold = by_fold[cell["inner_fold"]]
        fit, hold = set(cell["fit_row_ids"]), set(cell["validation_row_ids"])
        assert fit <= set(fold["fit_row_ids"]) and hold <= set(fold["heldout_row_ids"])
        assert not fit & hold and not (fit | hold) & outer_test
        assert set(cell["candidate_ids"]) <= ids
        assert {r["feature_id"] for r in cell["records"]} == set(cell["candidate_ids"])
        cell_counts[cell["family"]] += 1
        groups[(cell["inner_fold"], cell["repeat"], cell["family"])].extend(cell["candidate_ids"])
        statuses.update((cell["family"], r["status"]) for r in cell["records"])
        if not any(r["status"] == "ok" for r in cell["records"]):
            unsuccessful.append({k: cell[k] for k in ["family", "inner_fold", "repeat", "audit"]})
    for key, members in groups.items():
        assert len(members) == len(ids) and set(members) == ids, f"Coverage changed: {key}"
    oof = report["cross_fitted_nuisance_models"]["predictions"]
    assert len(oof) == 800 and {r["_oci_row_id"] for r in oof} == train_ids
    folder = HERE / "numerical" / report["input_fingerprint"][:20]
    checked, nuisance_audits = 0, []
    for path in sorted(folder.glob("fold_*/**/*.json")):
        checkpoint = n.read(path)
        assert checkpoint["input_fingerprint"] == report["input_fingerprint"]
        assert checkpoint["result_sha256"] == _fingerprint(checkpoint["result"])
        checked += 1
        if path.name == "nuisances.json":
            fold = by_fold[int(path.parent.name.split("_")[-1])]
            result = checkpoint["result"]
            inner_oof = []
            for part in result["crossfit_audit"]:
                fit, hold = set(part["fit_row_ids"]), set(part["heldout_row_ids"])
                assert not fit & hold and fit | hold == set(fold["fit_row_ids"])
                assert not (fit | hold) & set(fold["heldout_row_ids"])
                inner_oof.extend(part["heldout_row_ids"])
                nuisance_audits.extend(part["models"])
            assert len(inner_oof) == len(set(inner_oof)) == len(fold["fit_row_ids"])
            nuisance_audits.extend(result["validation_fit_audits"])
    assert checked == 310 and len(nuisance_audits) == 40
    assert all(a["converged"] for a in nuisance_audits)
    output = {
        "verified_at": n.now(), "status": "passed", "source_sha256": n.sha(__file__),
        "checkpoint_hashes_verified": checked, "family_subset_cells": len(report["cells"]),
        "family_cell_counts": dict(cell_counts), "all_candidates_present_in_every_family_and_repeat": True,
        "all_outer_and_inner_partitions_preserved": True, "nuisance_fits_converged": len(nuisance_audits),
        "nuisance_penalties_at_grid_boundary": sum(a.get("selected_on_grid_boundary", False) for a in nuisance_audits),
        "nuisance_cv_nonconverged_grid_fits": sum(not x["converged"] for a in nuisance_audits
                                                 for fold in a.get("cv_optimizer_audit", []) for x in fold),
        "record_status_counts": [{"family": k[0], "status": k[1], "count": v} for k, v in sorted(statuses.items())],
        "wholly_unevaluable_cells": unsuccessful,
        "selection_effect_eligible_rows": sum(r["effect_eligible"] for r in oof),
        "statistical_evidence_sha256": n.sha(HERE / "statistical_evidence.json"),
    }
    n.write(HERE / f"numerical_validation_{n.DATE}.json", output)
    print(json.dumps(output), flush=True)


if __name__ == "__main__":
    main()
