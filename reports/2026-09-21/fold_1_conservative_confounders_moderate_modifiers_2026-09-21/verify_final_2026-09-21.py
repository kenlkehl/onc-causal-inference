"""Check saved fitted models, role routing, inputs, and comparison populations."""
import hashlib
import inspect
import json
from common import HERE, BASE, PRIOR, DATE, n, verify_inputs


def main():
    import numpy as np
    import pandas as pd
    from econml.dml import CausalForestDML
    from oci.inference.plain_handoff_stage2_analysis import _FeatureEncoder, _definitions_for_nuisance_role

    manifest, selection = verify_inputs()
    freeze = n.read(HERE / f"predictions_frozen_{DATE}.json")
    access = n.read(HERE / "oracle_access_started.json")
    assert selection["frozen_at"] < freeze["frozen_at"] < access["at"]
    assert access["predictions_freeze_sha256"] == n.sha(HERE / f"predictions_frozen_{DATE}.json")
    assert access["source_sha256"] == n.sha(HERE / f"evaluate_{DATE}.py")
    n.verify(freeze["files"])
    n.verify(n.read(PRIOR / f"predictions_frozen_{DATE}.json")["files"])
    assert inspect.signature(CausalForestDML).parameters["max_samples"].default == .45

    full = n.read(n.INPUTS / "inputs/definitions.json")
    by_id = {d["feature_id"]: d for d in full}
    selected = n.read(HERE / "selected_definitions.json")["features"]
    decisions = n.read(HERE / "role_report.json")["decisions"]
    assert len(decisions) == len(by_id) == 352
    assert {d["feature_id"] for d in decisions} == set(by_id)
    assert {d["feature_id"]: d["roles"] for d in selected} == {d["feature_id"]: d["roles"] for d in decisions if d["roles"]}
    original = n.read(BASE / "selected_definitions.json")["features"]
    c_ids = {d["feature_id"] for d in original if "confounder" in d["roles"]}
    m_ids = {"outer_001_feature_" + d["id"] for d in n.read(PRIOR / "global_response.json")["modifier_ranking"][:16]}
    assert {d["feature_id"] for d in selected if "confounder" in d["roles"]} == c_ids
    assert {d["feature_id"] for d in selected if "effect_modifier" in d["roles"]} == m_ids
    assert len(c_ids) == 189 and len(m_ids) == 16 and len(c_ids & m_ids) == 8
    assert len(selected) == 197
    for d in selected:
        for key, value in by_id[d["feature_id"]].items():
            if key not in {"roles", "nuisance_model_roles", "selection_source"}:
                assert d[key] == value
    assert {d["feature_id"] for d in _definitions_for_nuisance_role(selected, "treatment")} == c_ids
    assert {d["feature_id"] for d in _definitions_for_nuisance_role(selected, "outcome")} == c_ids | m_ids
    modifiers = [d for d in selected if d["feature_id"] in m_ids]
    controls = [d for d in selected if d["feature_id"] in c_ids - m_ids]

    train = pd.read_pickle(n.INPUTS / "inputs/training.pkl")
    test = pd.read_pickle(n.INPUTS / "inputs/heldout.pkl")
    tl = pd.read_parquet(n.INPUTS / "inputs/training_labels.parquet")
    vl = pd.read_parquet(n.INPUTS / "inputs/heldout_labels.parquet")
    split = n.read(n.INPUTS / "inputs/split.json")
    assert train._oci_row_id.tolist() == tl._oci_row_id.tolist() == split["fit_row_ids"]
    assert test._oci_row_id.tolist() == vl._oci_row_id.tolist() == split["heldout_row_ids"]
    oracle_path = n.ROOT / "synthetic_data/example_synthetic_datasets/five_confounders_five_effect_modifiers_nsclc_with_structured/dataset.parquet"
    observed = pd.read_parquet(oracle_path, columns=["treatment_indicator", "outcome_indicator"])
    for labels in (tl, vl):
        original_rows = observed.iloc[labels._oci_row_id.to_numpy(int)]
        assert np.array_equal(original_rows.treatment_indicator.to_numpy(), labels.treatment.to_numpy())
        assert np.array_equal(original_rows.outcome_indicator.to_numpy(), labels.outcome.to_numpy())
    lineage = n.read(HERE.parent / "fold_1_interim_oracle_review_2026-09-21/fold_1_feature_recovery_2026-09-21.json")
    recovery = n.read(HERE / f"oracle_recovery_{DATE}.json")["direct"]
    assert {frozenset(r["candidate_ids"]) for r in recovery} == {frozenset(r["direct_feature_ids"]) for r in lineage["concepts"]}

    expected_ids = set(vl._oci_row_id)
    matched_ids = set(vl.loc[vl.effect_eligible, "_oci_row_id"])
    clone_checks, production_checks = [], []
    for seed in manifest["evaluation_plan"]["estimation_seeds"]:
        root = HERE / "production" / f"seed_{seed}"
        diag = n.read(root / "diagnostics.json")
        audit = diag["causal_forest_fit_audit"]
        assert diag["nuisance_model_family"] == "elastic_net"
        assert audit["effective_parameters"] == audit["configured_parameters"]
        assert not audit["tuning_attempted"]
        assert (diag["confounders"], diag["effect_modifiers"], diag["pure_confounders_in_w"], diag["dual_role_features_in_x_only"]) == (189, 16, 181, 8)
        assert (diag["treatment_nuisance_features"], diag["outcome_nuisance_features"]) == (189, 197)
        fit = pd.read_csv(root / "nuisance_fit_predictions.csv", float_precision="round_trip")
        assert fit._oci_row_id.tolist() == train._oci_row_id.tolist()
        eligible_fit = train.loc[fit.effect_eligible.to_numpy()]
        x = _FeatureEncoder(modifiers).fit(eligible_fit).transform(eligible_fit)
        w = _FeatureEncoder(controls).fit(eligible_fit).transform(eligible_fit)
        for role, clones in audit["fitted_nuisance_models"].items():
            assert len(clones) == 2
            for clone in clones:
                assert clone["estimator"] == "oci.models.elastic_net_nuisance.ElasticNetLogisticClassifier"
                assert clone["n_features"] == x.shape[1] + w.shape[1]
                clone_checks.append({"seed": seed, "role": role, **clone})
        frame = pd.read_csv(root / "predictions.csv", float_precision="round_trip")
        assert not frame._oci_row_id.duplicated().any() and set(frame._oci_row_id) == expected_ids
        assert np.array_equal(frame.effect_eligible, frame.estimated_cate.notna())
        assert np.array_equal(frame.effect_eligible, frame.propensity.between(.1, .9))
        assert np.array_equal(fit.effect_eligible, fit.propensity.between(.1, .9))
        eligible = frame.loc[frame.effect_eligible]
        assert len(eligible) == diag["effect_estimation_rows"]
        assert len(eligible_fit) == diag["effect_fit_rows"]
        cols = ["estimated_cate", "estimated_cate_lower_95", "estimated_cate_upper_95"]
        assert np.isfinite(eligible[cols]).all().all()
        assert (eligible[cols[1]] <= eligible[cols[2]]).all()
        assert np.isclose(eligible.aipw_score.mean(), diag["ate_aipw"], atol=1e-12, rtol=0)
        production_checks.append({"seed": seed, "fit_n": len(eligible_fit), "test_n": len(eligible),
            "encoded_x_columns": x.shape[1], "encoded_w_columns": w.shape[1], "forest_parameters": audit["effective_parameters"]})

    matched_checks = []
    tr, te = tl.effect_eligible.to_numpy(bool), vl.effect_eligible.to_numpy(bool)
    enc = _FeatureEncoder(modifiers).fit(train.loc[tr].reset_index(drop=True))
    x = enc.transform(train.loc[tr].reset_index(drop=True))
    xv = enc.transform(test.loc[te].reset_index(drop=True))
    for seed in manifest["evaluation_plan"]["matched_forest_seeds"]:
        root = HERE / "matched_residuals" / f"seed_{seed}"
        frame = pd.read_csv(root / "predictions.csv", float_precision="round_trip")
        assert not frame._oci_row_id.duplicated().any() and set(frame._oci_row_id) == matched_ids
        detail = n.read(root / "metrics.json")
        assert detail["modifiers"] == 16 and detail["effective_parameters"]["max_samples"] == .45
        assert detail["train_design_sha256"] == hashlib.sha256(x.tobytes()).hexdigest()
        assert detail["test_design_sha256"] == hashlib.sha256(xv.tobytes()).hexdigest()
        for previous in ["matched_residuals", "broad_matched"]:
            old = pd.read_csv(PRIOR / previous / f"seed_{seed}/predictions.csv", float_precision="round_trip")
            assert old._oci_row_id.tolist() == frame._oci_row_id.tolist()
            for col in ["treatment", "outcome", "propensity", "outcome_prediction"]:
                assert np.array_equal(old[col], frame[col])
        matched_checks.append(detail)
    result = {"verified_at": n.now(), "status": "passed", "source_sha256": n.sha(__file__),
        "all_frozen_source_input_selection_and_prediction_hashes_verified": True,
        "new_selection_then_predictions_then_new_oracle_evaluation": True,
        "earlier_oracle_exposure_acknowledged": True, "unchanged_measurement_definitions": True,
        "exact_broad_confounder_set_and_frozen_top16_verified": True,
        "observed_label_row_alignment_verified": True, "inherited_direct_lineage_verified": True,
        "role_routing_verified": True, "production_checks": production_checks,
        "production_nuisance_clones": clone_checks,
        "nuisance_clones_at_iteration_limit": sum(c["optimization"]["iteration_limit_reached"] for c in clone_checks),
        "matched_comparison_same_rows_labels_residuals_and_hyperparameters": True,
        "matched_design": {"modifiers": 16, "encoded_columns": x.shape[1]}}
    n.write(HERE / f"final_validation_{DATE}.json", result)
    print(json.dumps({"status": "passed", "clones": len(clone_checks),
                      "iteration_limits": result["nuisance_clones_at_iteration_limit"], "production": production_checks}), flush=True)


if __name__ == "__main__":
    main()
