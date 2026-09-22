"""Audit final experiment integrity, cohort alignment, and fitted nuisance clones."""
from pathlib import Path
import importlib.util
import json

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("numerical_run", HERE / "numerical_2026-09-21.py")
n = importlib.util.module_from_spec(spec)
spec.loader.exec_module(n)


def main():
    import numpy as np
    import pandas as pd

    selection = n.read(HERE / f"selection_frozen_{n.DATE}.json")
    predictions = n.read(HERE / f"predictions_frozen_{n.DATE}.json")
    oracle_access = n.read(HERE / "oracle_access_started.json")
    manifest = n.read(HERE / f"input_manifest_{n.DATE}.json")
    assert selection["frozen_at"] < predictions["frozen_at"] < oracle_access["at"]
    for files in [manifest["sources"], manifest["input_files"], selection["files"], predictions["files"]]:
        n.verify(files)
    assert n.read(HERE / "llm_runtime/parallel_review_validation.json")["native_replay_validated_all_responses_without_new_requests"]

    definitions = n.read(n.INPUTS / "inputs/definitions.json")
    decisions = n.read(HERE / "role_report.json")["decisions"]
    assert len(decisions) == len(definitions) == 352
    assert {d["feature_id"] for d in decisions} == {d["feature_id"] for d in definitions}
    selected = n.read(HERE / "selected_definitions.json")["features"]
    assert {d["feature_id"]: d["roles"] for d in selected} == {
        d["feature_id"]: d["roles"] for d in decisions if d["roles"]
    }

    heldout = pd.read_parquet(n.INPUTS / "inputs/heldout_labels.parquet")
    expected_ids = set(heldout._oci_row_id)
    matched_ids = set(heldout.loc[heldout.effect_eligible, "_oci_row_id"])
    clone_checks = []
    production_checks = []
    for seed in manifest["evaluation_plan"]["estimation_seeds"]:
        root = HERE / "production" / f"seed_{seed}"
        diag = n.read(root / "diagnostics.json")
        audit = diag["causal_forest_fit_audit"]
        assert diag["nuisance_model_family"] == "elastic_net"
        assert audit["effective_parameters"] == audit["configured_parameters"]
        assert not audit["tuning_attempted"]
        for role, clones in audit["fitted_nuisance_models"].items():
            assert clones
            for clone in clones:
                assert clone["estimator"] == "oci.models.elastic_net_nuisance.ElasticNetLogisticClassifier"
                clone_checks.append({"seed": seed, "role": role, **clone})
        frame = pd.read_csv(root / "predictions.csv", float_precision="round_trip")
        assert not frame._oci_row_id.duplicated().any()
        assert set(frame._oci_row_id) == expected_ids
        assert np.array_equal(frame.effect_eligible, frame.estimated_cate.notna())
        eligible = frame.loc[frame.effect_eligible]
        assert len(eligible) == diag["effect_estimation_rows"]
        cols = ["estimated_cate", "estimated_cate_lower_95", "estimated_cate_upper_95"]
        assert np.isfinite(eligible[cols]).all().all()
        assert (eligible[cols[1]] <= eligible[cols[2]]).all()
        production_checks.append({"seed": seed, "fit_n": diag["effect_fit_rows"], "test_n": len(eligible),
                                  "confounders": diag["confounders"], "modifiers": diag["effect_modifiers"],
                                  "pure_confounders_in_w": diag["pure_confounders_in_w"],
                                  "effective_parameters": audit["effective_parameters"]})

    matched_checks = []
    for seed in manifest["evaluation_plan"]["matched_forest_seeds"]:
        root = HERE / "matched_residuals" / f"seed_{seed}"
        frame = pd.read_csv(root / "predictions.csv", float_precision="round_trip")
        assert not frame._oci_row_id.duplicated().any()
        assert set(frame._oci_row_id) == matched_ids
        matched_checks.append(n.read(root / "metrics.json"))
    assert len({x["train_design_sha256"] for x in matched_checks}) == 1
    assert len({x["test_design_sha256"] for x in matched_checks}) == 1
    report = n.read(HERE / f"report_complete_{n.DATE}.json")
    assert report["report_sha256"] == n.sha(report["report"])
    assert report["figure_sha256"] == n.sha(HERE / f"ite_scatter_{n.DATE}.png")
    result = {"verified_at": n.now(), "status": "passed", "source_sha256": n.sha(__file__),
              "selection_before_predictions_before_oracle": True,
              "all_frozen_input_source_selection_prediction_hashes_verified": True,
              "every_candidate_has_one_decision": True,
              "production_checks": production_checks, "production_nuisance_clones": clone_checks,
              "nuisance_clones_at_iteration_limit": sum(c["optimization"]["iteration_limit_reached"] for c in clone_checks),
              "matched_design_and_cohort_consistent_across_seeds": True}
    n.write(HERE / f"final_validation_{n.DATE}.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in {"production_checks", "production_nuisance_clones"}}))


if __name__ == "__main__":
    main()
