import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location("selection_comparison", Path(__file__).with_name("compare.py"))
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def evidence():
    definitions = [{"feature_id": str(i), "name": f"measurement_{i}"} for i in range(5)]
    votes = {str(i): int(i == 0) for i in range(5)}
    gains = [[-.1] * 5, [-.1] * 5, [.2, .2, .2, -.1, -.1], [.01, .01, .01, -.2, -.2], [-.1] * 5]
    folds = [{"tests": [{"feature_id": str(i), "status": "ok", "heldout_r_loss_improvement": gains[i][j]}
                        for i in range(5)]} for j in range(5)]
    stat = {"multivariable_modifier_elastic_net_screen": {"votes": votes}, "effect_modifier_screen": {"folds": folds}}
    llm = [{"feature_id": str(i), "roles": ["effect_modifier"] if i == 1 else []} for i in range(5)]
    return definitions, stat, llm


@pytest.mark.parametrize("budget,expect_split", [(20000, True), (100000, False)])
def test_role_prompt_budget_preserves_all_evidence_and_reuses_decisions(tmp_path, budget, expect_split):
    from oci.inference.stage2_role_adjudication import (
        Stage2RoleAdjudicationConfig, build_stage2_role_evidence,
    )
    definitions = [{"feature_id": f"feature_{i}", "name": f"measurement_{i}",
        "description": "Pretreatment finding. " * 50, "value_type": "categorical",
        "categories_or_unit": ["category " + str(j) + " long " * 35 for j in range(30)],
        "measurement_definition": "Record the documented category.",
        "oracle_role": "NEVER_SEND_THIS_MARKER"} for i in range(5)]
    stat = {"policy": {"selection_mode": "llm_roles"}}
    original = copy.deepcopy((definitions, stat))
    policy = Stage2RoleAdjudicationConfig()
    expected = build_stage2_role_evidence(definitions=definitions, statistical_report=stat, policy=policy)
    calls = []

    def request(messages, validate, **kwargs):
        assert sum(len(message["content"]) for message in messages) <= budget
        assert "NEVER_SEND_THIS_MARKER" not in json.dumps(messages)
        payload = json.loads(messages[1]["content"])
        calls.append(payload)
        return validate({"summary": "Fixture review.", "decisions": [
            {"feature_id": candidate["feature_id"], "roles": ["effect_modifier"],
             "evidence_for": [], "evidence_against": [], "inner_fold_consistency": "Fixture.",
             "cross_method_reconciliation": "Fixture.", "rationale": "Fixture decision."}
            for candidate in payload["role_evidence"]["candidates"]]})

    args = dict(definitions=definitions, statistical_report=stat, request_json=request,
                output_dir=tmp_path, policy=policy, max_prompt_chars=budget)
    first = comparison.bounded_role_adjudication(**args)
    audit = comparison.read_json(tmp_path / "request_budget.json")
    assert (audit["effective_max_candidates_per_request"] < 20) is expect_split
    assert [candidate for payload in calls for candidate in payload["role_evidence"]["candidates"]] == expected["candidates"]
    assert [payload["candidate_batch"]["batch_index"] for payload in calls] == list(range(1, len(calls) + 1))
    assert {payload["candidate_batch"]["batch_count"] for payload in calls} == {len(calls)}
    for payload in calls:
        assert {k: v for k, v in payload["role_evidence"].items() if k != "candidates"} == {
            k: v for k, v in expected.items() if k != "candidates"}
    count = len(calls)
    assert comparison.bounded_role_adjudication(**args) == first
    assert len(calls) == count
    assert (definitions, stat) == original
    assert len(first[1]["decisions"]) == len(definitions)
    with pytest.raises(ValueError, match="Incompatible resume input"):
        comparison.bounded_role_adjudication(**{**args, "max_prompt_chars": budget - 1})


def test_role_prompt_budget_fails_before_transport_when_single_candidate_cannot_fit(tmp_path):
    from oci.inference.stage2_role_adjudication import Stage2RoleAdjudicationConfig
    def unexpected(*args, **kwargs):
        pytest.fail("An oversized prompt reached transport")
    with pytest.raises(ValueError, match="One complete role-evidence candidate"):
        comparison.bounded_role_adjudication(definitions=[{"feature_id": "a", "name": "a"}],
            statistical_report={}, request_json=unexpected, output_dir=tmp_path,
            policy=Stage2RoleAdjudicationConfig(), max_prompt_chars=100)
    assert not (tmp_path / "request_budget.json").exists()


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_role_prompt_budget_rejects_invalid_limits(tmp_path, budget):
    from oci.inference.stage2_role_adjudication import Stage2RoleAdjudicationConfig
    with pytest.raises(ValueError, match="positive integer"):
        comparison.bounded_role_adjudication(definitions=[], statistical_report={},
            request_json=lambda *a, **k: pytest.fail("Unexpected request"), output_dir=tmp_path,
            policy=Stage2RoleAdjudicationConfig(), max_prompt_chars=budget)


def revision_fixture(tmp_path):
    out = tmp_path / "results"
    source = tmp_path / "source.py"
    archive = tmp_path / "before.py"
    source.write_text("old")
    archive.write_bytes(source.read_bytes())
    original = {str(source): {"sha256": comparison.sha256(source)}}
    original_config = {"model": "unchanged", "extraction_max_tokens": 75000}
    comparison.write_json(out / "inputs/source_manifest.json", original)
    comparison.write_json(out / "inputs/refresh_config.json", original_config)
    source.write_text("revised")
    revised_config = {**original_config, "extraction_max_tokens": 4096,
                      "extraction_reasoning_max_tokens": 32768}
    new_config = out / "revisions/v2/config.json"
    comparison.write_json(new_config, revised_config)
    revision = {
        "schema": "selection_comparison_operational_revision_v1",
        "original_manifest_sha256": comparison.sha256(out / "inputs/source_manifest.json"),
        "source_overrides": {str(source): {"before_sha256": original[str(source)]["sha256"],
                             "before_archive": str(archive), "after_sha256": comparison.sha256(source)}},
        "additional_sources": {},
        "frozen_inputs": {str(p): {"sha256": comparison.sha256(p)} for p in (out / "inputs").glob("*.json")},
        "config": {"path": str(new_config), "sha256": comparison.sha256(new_config)},
        "config_changes": {key: {"before": original_config.get(key), "after": revised_config[key]}
                           for key in ("extraction_max_tokens", "extraction_reasoning_max_tokens")},
    }
    manifest = out / "revisions/v2/manifest.json"
    comparison.write_json(manifest, revision)
    return out, source, archive, new_config, manifest


def test_revision_requires_explicit_opt_in_and_preserves_original_guard(tmp_path):
    out, source, archive, config, manifest = revision_fixture(tmp_path)
    with pytest.raises(ValueError, match="Source changed"):
        comparison.validate_run_inputs(out)
    assert comparison.validate_run_inputs(out, manifest)["extraction_reasoning_max_tokens"] == 32768


@pytest.mark.parametrize("target", ["source", "archive", "config", "input", "manifest"])
def test_revision_rejects_changed_sources_archives_or_inputs(tmp_path, target):
    out, source, archive, config, manifest = revision_fixture(tmp_path)
    path = {"source": source, "archive": archive, "config": config,
            "input": out / "inputs/refresh_config.json", "manifest": out / "inputs/source_manifest.json"}[target]
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError):
        comparison.validate_run_inputs(out, manifest)


def test_revision_cannot_change_scientific_configuration(tmp_path):
    out, source, archive, config, manifest = revision_fixture(tmp_path)
    changed = comparison.read_json(config)
    changed["model"] = "different"
    comparison.write_json(config, changed)
    record = comparison.read_json(manifest)
    record["config"]["sha256"] = comparison.sha256(config)
    record["config_changes"]["model"] = {"before": "unchanged", "after": "different"}
    comparison.write_json(manifest, record)
    with pytest.raises(ValueError, match="scientific configuration"):
        comparison.validate_run_inputs(out, manifest)


def test_distinct_admission_routes_and_negative_mean_not_rescued():
    actual = comparison.admissions(*evidence())["methods"]
    assert actual == {"current_joint": ["0"], "llm_adjudication": ["1"],
                      "permissive_union": ["0", "1", "2"], "all_candidates": ["0", "1", "2", "3", "4"]}


def test_locked_roles_survive_all_policies_without_effect_promotion():
    definitions, stat, llm = evidence()
    definitions[0].update(configured_explicit_feature=True, roles=["confounder"])
    definitions[4].update(configured_explicit_feature=True, roles=["effect_modifier"])
    for chosen in comparison.admissions(definitions, stat, llm)["methods"].values():
        assert "0" not in chosen and "4" in chosen


def test_incomplete_llm_output_fails_closed():
    definitions, stat, llm = evidence()
    with pytest.raises(ValueError, match="every candidate"):
        comparison.admissions(definitions, stat, llm[:-1])


def test_admissions_do_not_depend_on_names_or_extra_oracle_fields():
    original = evidence()
    changed = copy.deepcopy(original)
    for d in changed[0]:
        d.update(name="arbitrary", oracle_cate=999, oracle_role="effect_modifier")
    assert comparison.admissions(*original)["methods"] == comparison.admissions(*changed)["methods"]


def test_alignment_reorders_and_rejects_missing_or_duplicate_rows():
    frame = pd.DataFrame({"_oci_row_id": [2, 1], "x": [20, 10]})
    assert comparison.align(frame, [1, 2], ["x"]).x.tolist() == [10, 20]
    with pytest.raises(ValueError, match="coverage"):
        comparison.align(frame, [1, 3])
    with pytest.raises(ValueError, match="Duplicated"):
        comparison.align(pd.concat([frame, frame]), [1, 2])


def test_r_loss_uses_common_nuisance_residuals():
    m = comparison.loss_metrics([0, 1], [0, 1], [.5, .5], [.5, .5], [1, 1], 0)
    assert m["r_loss"] == 0 and m["constant_r_loss"] == .25 and m["r_score"] == 1


def test_direct_residual_forest_matches_causal_forest_dml_final_stage():
    from econml.dml import CausalForestDML
    from econml.grf import CausalForest
    from sklearn.linear_model import ElasticNet
    rng = np.random.default_rng(804)
    x = rng.normal(size=(180, 3))
    t = rng.normal(size=180) + x[:, 0]
    y = (.2 + .5 * x[:, 1]) * t + x[:, 2] + rng.normal(size=180)
    parameters = dict(n_estimators=20, min_samples_leaf=5, max_features="sqrt", honest=True,
                      inference=False, random_state=20, n_jobs=1)
    model = CausalForestDML(model_y=ElasticNet(alpha=.05), model_t=ElasticNet(alpha=.05),
                           cv=2, **parameters).fit(y, t, X=x, cache_values=True)
    y_res, t_res, x_res, _ = model.residuals_
    direct = CausalForest(**parameters).fit(x_res, t_res, y_res)
    np.testing.assert_allclose(model.effect(x[:20]), direct.predict(x[:20]).ravel(), rtol=1e-12, atol=1e-12)


def test_all_four_policies_share_exact_residuals_and_resume_predictions(tmp_path):
    rng = np.random.default_rng(451)
    def frame(ids):
        return pd.DataFrame({"_oci_row_id": ids, "a": rng.normal(size=len(ids)), "b": rng.normal(size=len(ids))})
    fit, heldout = frame(list(range(160))), frame(list(range(160, 200)))
    def nuisance(frame):
        t = rng.binomial(1, .5, len(frame))
        return pd.DataFrame({"_oci_row_id": frame._oci_row_id, "treatment": t,
            "outcome": rng.binomial(1, .45, len(frame)), "propensity": .5,
            "outcome_prediction": .45, "effect_eligible": True})
    nfit, ntest = nuisance(fit), nuisance(heldout)
    definitions = [{"feature_id": name, "name": name, "value_type": "continuous"} for name in ("a", "b")]
    routing = {"methods": {"current_joint": ["a"], "llm_adjudication": [],
                           "permissive_union": ["a", "b"], "all_candidates": ["a", "b"]}}
    args = (fit, heldout, definitions, routing, nfit, ntest,
            {"outer_fold": 1}, {"seed": 42}, tmp_path)
    first = comparison.fit_forests(*args)
    assert len(first) == 12
    assert len({r["nuisance_hash"] for r in first}) == 1
    assert {r["rows"] for r in first} == {40}
    assert comparison.fit_forests(*args) == first
    nfit.loc[0, "propensity"] = .6
    with pytest.raises(ValueError, match="fingerprint"):
        comparison.fit_forests(*args)


def test_common_nuisance_fits_ignore_test_outcomes_and_cache_exact_inputs(tmp_path):
    from types import SimpleNamespace
    from oci.inference.stage2_elastic_net_selection import Stage2ElasticNetSelectionConfig
    rng = np.random.default_rng(733)
    x = rng.normal(size=180)
    data = pd.DataFrame({"t": rng.binomial(1, 1/(1+np.exp(-x))),
                         "y": rng.binomial(1, 1/(1+np.exp(-.6*x)))})
    frame = pd.DataFrame({"_oci_row_id": range(180), "x": x})
    fit, held = frame.iloc[:140].copy(), frame.iloc[140:].copy().reset_index(drop=True)
    predictions = [{"_oci_row_id": i, "treatment": int(data.t.iloc[i]), "outcome": int(data.y.iloc[i]),
                    "propensity": .5, "outcome_prediction": .5, "effect_eligible": True} for i in range(140)]
    stat = {"policy": {"test": True}, "cross_fitted_nuisance_models": {"predictions": predictions, "folds": []}}
    cfg = SimpleNamespace(statistical_selection=Stage2ElasticNetSelectionConfig(internal_cv_folds=2,
        regularization_grid_size=3, max_iter=500), min_propensity=.1, max_propensity=.9)
    definitions = [{"feature_id": "x", "name": "x", "value_type": "continuous"}]
    split = {"outer_fold": 1, "fit_row_ids": list(range(140)), "heldout_row_ids": list(range(140, 180))}
    settings = {"seed": 42, "treatment_column": "t", "outcome_column": "y", "outcome_type": "binary"}
    args = (fit, held, definitions, stat, split, cfg, settings)
    first_fit, first_test = comparison.fit_common_nuisances(data, *args, tmp_path / "a")
    cached_fit, cached_test = comparison.fit_common_nuisances(data, *args, tmp_path / "a")
    np.testing.assert_allclose(first_test.propensity, cached_test.propensity)
    changed = data.copy()
    changed.loc[140:, "y"] = 1-changed.loc[140:, "y"]
    _, second_test = comparison.fit_common_nuisances(changed, *args, tmp_path / "b")
    np.testing.assert_allclose(first_test.propensity, second_test.propensity, atol=0, rtol=0)
    np.testing.assert_allclose(first_test.outcome_prediction, second_test.outcome_prediction, atol=0, rtol=0)
    with pytest.raises(ValueError, match="mismatch"):
        comparison.fit_common_nuisances(changed, *args, tmp_path / "a")
