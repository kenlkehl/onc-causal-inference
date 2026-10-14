"""Conventional score-adjusted tests and their role in modifier selection."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from scipy.stats import chi2

from oci.inference import stage2_interaction_screen as screen
from oci.inference import stage2_modifier_count as count
from oci.inference import stage2_multi_model_selection as numerical
from oci.inference import stage2_statistical_selection as tests
from oci.inference.stage2_elastic_net_selection import statistical_selection_config_from_mapping
from tests.test_stage2_multi_model import feature, small_policy
from tests.test_stage2_statistical_ranking import inputs, run, by_id


@pytest.mark.parametrize("binary", [False, True])
def test_reduced_and_full_models_share_crossfit_scores_and_population(monkeypatch, binary):
    import statsmodels.api as sm

    rng = np.random.default_rng(831)
    n = 700
    z, x = rng.normal(size=(2, n))
    e = expit(0.7 * z)
    t = rng.binomial(1, e).astype(float)
    mu0 = expit(-0.5 + z) if binary else z
    mu1 = expit(-0.5 + z + 0.5 + 1.5 * x) if binary else z + 0.5 + 1.5 * x
    m = (1 - e) * mu0 + e * mu1
    y = (
        rng.binomial(1, np.where(t, mu1, mu0))
        if binary
        else z + t * (0.5 + 1.5 * x) + rng.normal(size=n)
    )
    frame = pd.DataFrame({"_oci_row_id": np.arange(n), "x": x})
    keep = (e > 0.15) & (e < 0.85)
    captured = []
    original = tests._binary_nested_p_value if binary else tests._continuous_nested_p_value

    def inspect(target, reduced, additions):
        captured.append((target.copy(), reduced.copy(), additions.copy()))
        return original(target, reduced, additions)

    monkeypatch.setattr(
        tests, "_binary_nested_p_value" if binary else "_continuous_nested_p_value", inspect
    )
    result = numerical._univariable(
        frame,
        [feature("x")],
        t,
        y,
        binary=binary,
        policy=small_policy(),
        nuisance={"e": e, "m": m},
        effect_keep=keep,
    )
    record = next(r for r in result["records"] if r["role"] == "effect")
    assert record["interaction_adjusted"] and record["p_value"] < 0.01
    target, reduced, additions = captured[-1]
    assert np.array_equal(target, y[keep])
    expected_base, audit = screen.adjustment_design(t[keep], e[keep], m[keep], binary=binary)
    assert expected_base.shape[1] > 2
    assert np.array_equal(reduced[:, : expected_base.shape[1]], expected_base)
    assert np.array_equal(additions[:, 0], t[keep] * (x[keep] - np.median(x[keep])))
    full, _ = tests._rank_safe_columns(reduced, additions)
    assert np.array_equal(full[:, : reduced.shape[1]], reduced)
    assert full.shape[1] == reduced.shape[1] + 1
    assert result["audit"]["effect_fit_row_ids"] == np.flatnonzero(keep).tolist()
    assert result["audit"]["interaction_adjustment"] == audit
    if binary:
        reduced_fit = sm.GLM(target, reduced, family=sm.families.Binomial()).fit()
        full_fit = sm.GLM(target, full, family=sm.families.Binomial()).fit()
        assert record["p_value"] == pytest.approx(chi2.sf(2 * (full_fit.llf - reduced_fit.llf), 1))


def test_score_adjustment_handles_constant_redundant_and_boundary_probabilities():
    t = np.tile([0.0, 1.0], 20)
    design, audit = screen.adjustment_design(t, np.full(40, 0.5), np.full(40, 0.2), binary=True)
    assert np.array_equal(design, np.column_stack((np.ones(40), t)))
    assert all(v["constant"] for v in audit["scores"].values())
    e = np.linspace(0, 1, 40)
    design, _ = screen.adjustment_design(t, e, e, binary=True)
    assert np.isfinite(design).all() and np.linalg.matrix_rank(design) == design.shape[1]
    assert design.shape[1] == 4  # duplicated score bases are rank-filtered
    for invalid in [np.full(40, np.nan), np.ones(39), np.full(40, 1.5)]:
        with pytest.raises(ValueError):
            screen.adjustment_design(t, invalid, np.full(40, 0.5), binary=True)


def with_interactions(data, q_by_feature):
    definitions, report = deepcopy(data)
    report["model_families"].append("univariable")
    for fold in range(1, 4):
        for repeat in range(report["policy"]["multi_model"]["repeats"]):
            report["cells"].append(
                {
                    "family": "univariable",
                    "inner_fold": fold,
                    "repeat": repeat,
                    "records": [
                        {
                            "feature_id": key,
                            "role": "effect",
                            "status": "ok",
                            "interaction_adjusted": True,
                            "q_value": q,
                        }
                        for key, q in q_by_feature.items()
                    ],
                }
            )
    return definitions, report


def test_interaction_test_can_change_ranking_without_predictive_support(tmp_path):
    data = with_interactions(
        inputs({"interaction": (0, 0, 0), "predictive": (1, 1, 1)}),
        {"interaction": 1e-12, "predictive": 1.0},
    )
    result = run(tmp_path / "equal", data)
    assert result["ranking"][0]["feature_id"] == "interaction"
    c = by_id(result)["interaction"]
    assert c["evidence_groups"]["predictive"] == 0
    assert c["score"] == pytest.approx(0.5 * c["interaction_test"]["support_score"])
    assert "multi:interaction:univariable:effect" in c["evidence_ids"]
    data[1]["policy"]["multi_model"]["interaction_ranking_weight"] = 0
    prior = run(tmp_path / "predictive", data)
    assert prior["ranking"][0]["feature_id"] == "predictive"
    assert prior["identity"] != result["identity"]


def test_dependent_repeats_do_not_accumulate_significance_or_reward_lucky_fold(tmp_path):
    base = inputs({"lucky": (0, 0, 0), "stable": (0, 0, 0)}, repeats=1)
    data = with_interactions(base, {"lucky": 0.8, "stable": 0.02})
    for cell in data[1]["cells"]:
        if cell["family"] == "univariable" and cell["inner_fold"] == 1:
            cell["records"][0]["q_value"] = 1e-100
    first = run(tmp_path / "one", data)
    assert first["ranking"][0]["feature_id"] == "stable"
    repeated = deepcopy(data)
    repeated[1]["cells"] = [
        {**deepcopy(c), "repeat": r} for c in data[1]["cells"] for r in range(3)
    ]
    repeated[1]["policy"]["multi_model"]["repeats"] = 3
    second = run(tmp_path / "three", repeated)
    assert [(r["feature_id"], r["score"]) for r in first["ranking"]] == [
        (r["feature_id"], r["score"]) for r in second["ranking"]
    ]


def test_unadjusted_or_missing_tests_cannot_be_promoted_as_adjusted_support(tmp_path):
    data = with_interactions(inputs({"a": (1, 1, 1)}), {"a": 1e-20})
    first = run(tmp_path / "complete", data)
    for cell in data[1]["cells"]:
        if cell["family"] == "univariable":
            cell["records"][0].pop("interaction_adjusted")
    second = run(tmp_path / "old", data)
    a = by_id(second)["a"]
    assert a["interaction_test"]["status"] == "missing"
    assert a["score"] < by_id(first)["a"]["score"]
    assert a["evidence_groups"]["interaction_weight"] == 0.5
    assert a["interaction_test"]["selection_utility"] == 0


def test_interaction_supported_budget_cannot_override_predictive_guardrail():
    losses = {
        ("linear_interactions", 0): [0.2, 0.2, 0.2],
        ("linear_interactions", 4): [0.201, 0.199, 0.2],
        ("linear_interactions", 8): [0.3, 0.3, 0.3],
    }
    result = count.choose_effect_model(
        losses,
        rule="balanced_evidence",
        interaction_utilities={0: [0, 0, 0], 4: [0.1, 0.1, 0.1], 8: [1, 1, 1]},
    )
    assert result["chosen_additional_count"] == 4
    assert not result["options"]["linear_interactions"]["8"]["within_predictive_guardrail"]
    assert result["options"]["linear_interactions"]["4"]["within_predictive_guardrail"]
    # A tiny deterministic loss difference cannot discard well-supported modifiers.
    losses[("linear_interactions", 4)] = [0.200000001] * 3
    tiny = count.choose_effect_model(
        losses,
        rule="balanced_evidence",
        interaction_utilities={0: [0] * 3, 4: [0.1] * 3, 8: [1] * 3},
    )
    assert tiny["chosen_additional_count"] == 4
    weak = count.choose_effect_model(
        losses,
        rule="balanced_evidence",
        interaction_utilities={0: [0] * 3, 4: [-0.1] * 3, 8: [1] * 3},
    )
    assert weak["chosen_additional_count"] == 0
    assert not weak["formal_error_guarantee"]


def test_configuration_roundtrip_and_invalid_weights():
    cfg = statistical_selection_config_from_mapping({"selection_mode": "multi_model"})
    assert cfg.multi_model.interaction_ranking_weight == 0.5
    assert cfg.multi_model.modifier_count.selection_rule == "balanced_evidence"
    assert statistical_selection_config_from_mapping(cfg.public_dict()) == cfg
    for value in [-0.1, 1.1, True, float("nan")]:
        with pytest.raises(ValueError):
            replace(cfg.multi_model, interaction_ranking_weight=value).validate()
        with pytest.raises(ValueError):
            replace(cfg.multi_model.modifier_count, relative_r_loss_tolerance=value).validate()


def test_balanced_architecture_choice_and_legacy_configuration():
    losses = {
        ("linear_interactions", 0): [0.2] * 3,
        ("linear_interactions", 4): [0.2005, 0.1995, 0.2],
        ("dina", 4): [0.2005, 0.1995, 0.2],
        ("causal_forest", 4): [0.2005, 0.1995, 0.2],
    }
    utility = {0: [0] * 3, 4: [0.1] * 3}
    selected = count.choose_effect_model(
        losses, rule="balanced_evidence", interaction_utilities=utility
    )
    assert selected["chosen_estimator"] == "linear_interactions"
    assert selected["chosen_additional_count"] == 4
    # With the same eligible budget, a material fit advantage selects the forest.
    losses[("causal_forest", 4)] = [0.2003, 0.1993, 0.1998]
    selected = count.choose_effect_model(
        losses, rule="balanced_evidence", interaction_utilities=utility
    )
    assert selected["chosen_estimator"] == "causal_forest"
    old = statistical_selection_config_from_mapping(
        {
            "selection_mode": "multi_model",
            "multi_model": {
                "schema_version": "stage2_multi_model_selection_v6_predictive_nuisances",
                "modifier_count": {"selection_rule": "minimum_r_loss"},
            },
        }
    )
    assert old.multi_model.modifier_count.selection_rule == "minimum_r_loss"


def test_configured_interaction_family_controls_weight_and_cache(tmp_path):
    data = with_interactions(inputs({"a": (0, 0, 0), "b": (1, 1, 1)}), {"a": 1e-10, "b": 1.0})
    first = run(tmp_path, data)
    assert first["ranking"][0]["feature_id"] == "a"
    data[1]["model_families"].remove("univariable")
    second = run(tmp_path, data)
    assert second["ranking"][0]["feature_id"] == "b"
    assert by_id(second)["b"]["evidence_groups"]["interaction_weight"] == 0
    assert first["identity"] != second["identity"]
    data[1]["model_families"] = ["univariable"]
    data[1]["cells"] = [c for c in data[1]["cells"] if c["family"] == "univariable"]
    third = run(tmp_path, data)
    assert third["ranking"][0]["feature_id"] == "a"
    assert by_id(third)["a"]["evidence_groups"]["interaction_weight"] == 1
