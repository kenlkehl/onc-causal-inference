"""Predictive routing, immutable overrides, fold honesty and final refits."""

from copy import deepcopy

import numpy as np
import pytest

from oci.inference import plain_handoff_stage2_analysis as analysis
from oci.inference import stage2_multi_model_selection as numerical
from oci.inference.stage2_modifier_count import _apply_budget
from oci.inference.stage2_nuisance_selection import select_predictive_nuisances
from tests.test_stage2_multi_model import feature, sample_inputs, small_policy
from tests.test_stage2_final_modeling import inputs as final_inputs


def support_fixture():
    definitions = [feature(k) for k in ("t", "y", "both", "noise", "locked", "modifier")]
    definitions[-2].update(configured_explicit_feature=True, roles=["confounder"])
    definitions[-1].update(configured_explicit_feature=True, roles=["effect_modifier"])
    folds = [
        {
            "fit_row_ids": [0, 1],
            "heldout_row_ids": [2, 3],
            "task_feature_ids": {"treatment": ["t", "both"], "outcome": ["y"]},
        },
        {
            "fit_row_ids": [2, 3],
            "heldout_row_ids": [0, 1],
            "task_feature_ids": {"treatment": [], "outcome": ["both"]},
        },
    ]
    return definitions, folds


def test_any_fold_task_support_and_exact_locks_ignore_semantic_labels():
    definitions, folds = support_fixture()
    # A discovered (unlocked) semantic label cannot rescue a numerical zero.
    definitions[3]["roles"] = ["confounder", "effect_modifier"]
    selected, decisions, audit = select_predictive_nuisances(definitions, folds)
    by_id = {f["feature_id"]: f for f in selected}
    assert set(by_id) == {"t", "y", "both", "locked", "modifier"}
    assert by_id["t"]["nuisance_model_roles"] == ["treatment"]
    assert by_id["y"]["nuisance_model_roles"] == ["outcome"]
    assert by_id["both"]["nuisance_model_roles"] == ["treatment", "outcome"]
    assert by_id["locked"]["roles"] == ["confounder"]
    assert by_id["locked"]["nuisance_model_roles"] == ["treatment", "outcome"]
    assert by_id["modifier"]["roles"] == ["effect_modifier"]
    assert by_id["modifier"]["nuisance_model_roles"] == []
    assert audit["llm_roles_are_binding"] is False
    assert audit["cross_fold_union_used_for_scoring_nuisances"] is False
    assert len(decisions) == len(definitions)
    assert all(f["selection_authority"] == "multi_model" for f in selected)


@pytest.mark.parametrize("count", [0, 1])
def test_modifier_budget_preserves_separate_nuisance_support(count):
    definitions, folds = support_fixture()
    selected, decisions, _ = select_predictive_nuisances(definitions, folds)
    kept, audit = _apply_budget(
        definitions,
        selected,
        {"decisions": decisions},
        [{"feature_id": "noise", "evidence_ids": [], "rationale": "Effect evidence"}],
        count,
        ["modifier"],
    )
    by_id = {f["feature_id"]: f for f in kept}
    assert by_id["t"]["nuisance_model_roles"] == ["treatment"]
    assert by_id["y"]["nuisance_model_roles"] == ["outcome"]
    assert ("noise" in by_id) == bool(count)
    if count:
        assert by_id["noise"]["nuisance_model_roles"] == []
        assert by_id["noise"]["modeling_tasks"] == ["effect"]
    assert by_id["locked"]["roles"] == ["confounder"]
    assert by_id["modifier"]["roles"] == ["effect_modifier"]
    for row in audit["decisions"]:
        assert row["retained"] == (row["feature_id"] in by_id)


def test_invalid_nuisance_support_fails_closed():
    definitions, folds = support_fixture()
    bad = deepcopy(folds)
    bad[0]["task_feature_ids"]["outcome"].append("not_a_candidate")
    with pytest.raises(ValueError, match="unknown candidate"):
        select_predictive_nuisances(definitions, bad)
    bad = deepcopy(folds)
    bad[0]["fit_row_ids"].append(2)
    with pytest.raises(ValueError, match="overlap"):
        select_predictive_nuisances(definitions, bad)


@pytest.mark.parametrize("binary", [False, True])
def test_real_nuisance_support_uses_group_coefficients_and_training_rows_only(binary):
    args = sample_inputs()
    frame = args["extracted_fit"]
    definitions = args["definitions"]
    # A multi-level categorical variable must remain one routing decision.
    definitions[-1].update(value_type="categorical", categories_or_unit=["a", "b", "c"])
    frame["other"] = np.select([frame.other < -0.5, frame.other > 0.5], ["a", "b"], "c")
    train_ids, valid_ids = (
        args["inner_splits"][0]["fit_row_ids"],
        args["inner_splits"][0]["heldout_row_ids"],
    )
    t, y = args["dataset"].iloc[train_ids][["treatment", "outcome"]].to_numpy().T
    if binary:
        y = (y > np.median(y)).astype(float)
    kw = dict(definitions=definitions, t=t, y=y, binary=binary, policy=args["policy"], seed=12)
    fit = numerical._nuisances(frame.iloc[train_ids], frame.iloc[valid_ids], **kw)
    support = fit["predictive_support"]
    expected = {
        role: {r["feature_id"] for r in rows if r["selected"]}
        for role, rows in support["model_group_evidence"].items()
    }
    assert set(support["task_feature_ids"]["treatment"]) == expected["treatment"]
    assert set(support["task_feature_ids"]["outcome"]) == set().union(
        *(v for k, v in expected.items() if k != "treatment")
    )
    assert set(expected) == {"treatment", "outcome", "mu0", "mu1"}
    assert all(len(rows) == len(definitions) for rows in support["model_group_evidence"].values())
    # Even drastic changes in validation measurements cannot alter chosen supports.
    poisoned = frame.iloc[valid_ids].copy()
    poisoned.loc[:, ["c", "m", "noise"]] = 1e8
    other = numerical._nuisances(frame.iloc[train_ids], poisoned, **kw)
    assert other["predictive_support"] == support


@pytest.mark.parametrize("estimator", ["linear_interactions", "dina"])
def test_final_crossfit_uses_unscreened_catalog_and_dina_reuses_honest_nuisances(
    tmp_path, monkeypatch, estimator
):
    kw = final_inputs()
    all_fit = kw["extracted_fit"].copy()
    all_fit["discarded"] = np.arange(len(all_fit)) % 3
    all_heldout = kw["extracted_heldout"].copy()
    all_heldout["discarded"] = np.arange(len(all_heldout)) % 3
    candidates = [feature("x"), feature("discarded")]
    kw["definitions"][0].update(nuisance_model_roles=["outcome"])
    original = analysis._cross_fitted_nuisance
    crossfit_calls = []

    def check(**kwargs):
        assert kwargs["all_candidates"] is True
        assert [f["feature_id"] for f in kwargs["definitions"]] == ["x", "discarded"]
        assert set(kwargs["extracted"]._oci_row_id) == set(range(72))
        crossfit_calls.append(True)
        return original(**kwargs)

    monkeypatch.setattr(analysis, "_cross_fitted_nuisance", check)
    result = analysis.estimate_outer_fold(
        **kw,
        estimator=estimator,
        statistical_policy=small_policy(),
        output_dir=tmp_path,
        nuisance_candidate_definitions=candidates,
        nuisance_candidate_fit=all_fit,
        nuisance_candidate_heldout=all_heldout,
    )
    assert crossfit_calls == [True]
    assert result["nuisance_routing"]["treatment_feature_ids"] == []
    assert result["nuisance_routing"]["outcome_feature_ids"] == ["x"]
    assert result["nuisance_routing"]["training_crossfit_candidates"] == 2
    assert result["nuisance_routing"]["dina_reuses_external_nuisances"] == (estimator == "dina")
    # Compatible completed estimation must reuse its result, not refit.
    resumed = analysis.estimate_outer_fold(
        **kw,
        estimator=estimator,
        statistical_policy=small_policy(),
        output_dir=tmp_path,
        nuisance_candidate_definitions=candidates,
        nuisance_candidate_fit=all_fit,
        nuisance_candidate_heldout=all_heldout,
    )
    assert resumed == result and len(crossfit_calls) == 1
