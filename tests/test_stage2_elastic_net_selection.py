from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold

import oci.inference.stage2_elastic_net_selection as selection_module
from oci.inference.stage2_elastic_net_selection import (
    SCHEMA_VERSION,
    Stage2ElasticNetSelectionConfig,
    _encode_design,
    select_stage2_features_elastic_net,
)


def _continuous(feature_id: str, name: str) -> dict[str, object]:
    return {
        "feature_id": feature_id,
        "name": name,
        "description": name,
        "value_type": "continuous",
        "modeling_strategy": "continuous",
        "categories_or_unit": ["unitless"],
        "measurement_definition": name,
        "missing_value_rule": "null when absent",
    }


def _categorical(feature_id: str, name: str, levels: list[str]) -> dict[str, object]:
    return {
        "feature_id": feature_id,
        "name": name,
        "description": name,
        "value_type": "categorical",
        "categories_or_unit": levels,
        "measurement_definition": name,
        "missing_value_rule": "null when absent",
    }


def test_any_fold_nuisance_union_and_candidate_augmented_rlearner_selection():
    rng = np.random.default_rng(141)
    rows = 700
    confounder = rng.normal(size=rows)
    treatment_only = rng.normal(size=rows)
    outcome_only = rng.normal(size=rows)
    modifier = rng.normal(size=rows)
    noise = rng.normal(size=rows)
    treatment_probability = 1.0 / (
        1.0 + np.exp(-(1.5 * confounder + 1.4 * treatment_only))
    )
    treatment = rng.binomial(1, treatment_probability)
    outcome = (
        1.8 * confounder
        + 1.7 * outcome_only
        + treatment * (2.4 * modifier)
        + 0.45 * noise
    )
    dataset = pd.DataFrame(
        {
            "treatment": treatment,
            "outcome": outcome,
        }
    )
    extracted = pd.DataFrame(
        {
            "_oci_row_id": np.arange(rows),
            "confounder": confounder,
            "treatment_only": treatment_only,
            "outcome_only": outcome_only,
            "modifier": modifier,
            "noise": noise,
        }
    )
    definitions = [
        _continuous("conf", "confounder"),
        _continuous("t_only", "treatment_only"),
        _continuous("y_only", "outcome_only"),
        _continuous("mod", "modifier"),
        _continuous("noise", "noise"),
    ]
    splitter = KFold(n_splits=5, shuffle=True, random_state=19)
    inner_splits = [
        {
            "inner_fold": index,
            "fit_row_ids": fit.tolist(),
            "heldout_row_ids": heldout.tolist(),
        }
        for index, (fit, heldout) in enumerate(splitter.split(extracted), start=1)
    ]
    policy = Stage2ElasticNetSelectionConfig(
        l1_ratio=0.9,
        internal_cv_folds=3,
        regularization_grid_size=8,
        modifier_top_n_per_inner_fold=1,
        max_iter=3_000,
    )

    selected, report, dependencies, latent_states = select_stage2_features_elastic_net(
        dataset=dataset,
        extracted_fit=extracted,
        definitions=definitions,
        inner_splits=inner_splits,
        treatment_column="treatment",
        outcome_column="outcome",
        outcome_type="continuous",
        seed=91,
        policy=policy,
    )

    selected_by_id = {row["feature_id"]: row for row in selected}
    assert report["schema_version"] == SCHEMA_VERSION
    assert report["latent_construction"] == "disabled"
    assert report["pairwise_association_screen"] == "disabled"
    assert report["nuisance_screen"]["intersection_is_not_a_selection_gate"] is True
    assert "treatment" in selected_by_id["conf"]["nuisance_model_roles"]
    assert "outcome" in selected_by_id["conf"]["nuisance_model_roles"]
    assert selected_by_id["t_only"]["nuisance_model_roles"] == [
        "treatment",
        "outcome",
    ]
    assert selected_by_id["y_only"]["nuisance_model_roles"] == [
        "treatment",
        "outcome",
    ]
    assert (
        report["nuisance_screen"]["stable_treatment_feature_ids"]
        != report["nuisance_screen"]["stable_outcome_feature_ids"]
    )
    assert report["nuisance_screen"]["required_votes"] == 1
    assert report["nuisance_screen"]["union_is_used_by_both_nuisance_models"] is True
    assert (
        report["cross_fitted_nuisance_models"]["treatment_feature_ids"]
        == report["cross_fitted_nuisance_models"]["outcome_feature_ids"]
        == report["nuisance_screen"]["union_confounder_feature_ids"]
    )
    assert report["nuisance_screen"]["overall_treatment_auroc"] is not None
    assert report["nuisance_screen"]["overall_outcome_auroc"] is None
    assert report["confounder_univariable_screen"]["hard_selection_gate"] is False
    assert report["confounder_univariable_screen"][
        "multiplicity_adjustment"
    ] == "benjamini_hochberg_within_inner_fold_endpoint"
    assert report["confounder_univariable_screen"][
        "nominal_joint_support_votes"
    ]["conf"] >= 1
    for fold in report["confounder_univariable_screen"]["folds"]:
        confounder_test = next(
            row for row in fold["tests"] if row["feature_id"] == "conf"
        )
        assert confounder_test["treatment_p_value"] is not None
        assert confounder_test["outcome_adjusted_for_treatment_p_value"] is not None
        assert "treatment_q_value" in confounder_test
    for fold in report["nuisance_screen"]["folds"]:
        assert fold["treatment"]["heldout_auroc"] is not None
        assert fold["outcome"]["heldout_auroc"] is None
    assert "effect_modifier" in selected_by_id["mod"]["roles"]
    assert report["effect_modifier_screen"]["model_family"] == (
        "candidate_augmented_univariable_r_learner"
    )
    assert report["effect_modifier_screen"]["nuisance_model_family"] == (
        "group_elastic_net"
    )
    assert report["effect_modifier_screen"]["top_n_per_inner_fold"] == 1
    assert report["effect_modifier_screen"]["required_votes"] == 1
    assert report["effect_modifier_screen"][
        "r_loss_improvement_threshold_is_not_a_selection_gate"
    ]
    assert all(
        fold["selected_count"] == 1
        and fold["base_nuisance_model_family"] == "group_elastic_net"
        and fold["candidate_nuisance_model_family"] == "group_elastic_net"
        and fold["base_nuisance_predictions_for_fit_are_nested_cross_fitted"] is True
        and fold["candidate_nuisance_augmentations_are_cross_fitted"] is True
        and fold["modifier_evaluation_rows_are_untouched_inner_heldout"] is True
        for fold in report["effect_modifier_screen"]["folds"]
    )
    assert report["cross_fitted_nuisance_models"][
        "predictions_are_inner_fold_out_of_fold"
    ] is True
    assert report["cross_fitted_nuisance_models"]["model_family"] == (
        "group_elastic_net"
    )
    assert report["cross_fitted_nuisance_models"][
        "one_standard_error_rule"
    ] is False
    joint_modifier = report["multivariable_modifier_elastic_net_screen"]
    assert joint_modifier["hard_selection_gate"] is False
    assert joint_modifier["candidate_scope"] == (
        "all_candidate_groups_in_one_model_per_inner_fold"
    )
    assert len(joint_modifier["folds"]) == len(inner_splits)
    assert all(fold["status"] for fold in joint_modifier["folds"])
    assert [row["feature_id"] for row in dependencies] == [
        row["feature_id"] for row in selected
    ]
    assert latent_states == []


def test_investigator_locked_roles_are_preserved_without_latents():
    rng = np.random.default_rng(22)
    rows = 160
    treatment = rng.binomial(1, 0.5, size=rows)
    dataset = pd.DataFrame(
        {
            "treatment": treatment,
            "outcome": rng.normal(size=rows),
        }
    )
    extracted = pd.DataFrame(
        {
            "_oci_row_id": np.arange(rows),
            "locked": rng.normal(size=rows),
        }
    )
    definition = {
        **_continuous("locked", "locked"),
        "configured_explicit_feature": True,
        "roles": ["confounder", "effect_modifier"],
    }
    splitter = KFold(n_splits=4, shuffle=True, random_state=3)
    inner_splits = [
        {
            "inner_fold": index,
            "fit_row_ids": fit.tolist(),
            "heldout_row_ids": heldout.tolist(),
        }
        for index, (fit, heldout) in enumerate(splitter.split(extracted), start=1)
    ]
    policy = Stage2ElasticNetSelectionConfig(
        regularization_grid_size=5,
    )

    selected, _report, dependencies, latent_states = select_stage2_features_elastic_net(
        dataset=dataset,
        extracted_fit=extracted,
        definitions=[definition],
        inner_splits=inner_splits,
        treatment_column="treatment",
        outcome_column="outcome",
        outcome_type="continuous",
        seed=4,
        policy=policy,
    )

    assert selected[0]["roles"] == ["confounder", "effect_modifier"]
    assert selected[0]["nuisance_model_roles"] == ["treatment", "outcome"]
    assert dependencies[0]["feature_id"] == "locked"
    assert latent_states == []


def test_ordinal_measurement_is_one_standardized_score_not_dummy_columns():
    definition = {
        **_categorical("ecog", "ecog", ["0", "1", "2", "3"]),
        "value_type": "ordinal",
    }
    train = pd.DataFrame({"ecog": [0, 1, 2, 3, 1, 2, None, 0]})
    valid = pd.DataFrame({"ecog": [3, None, 1]})

    design = _encode_design(
        train,
        valid,
        [definition],
        categorical_min_count=1,
    )

    assert design.column_names == ("ecog:ordered_score", "ecog:missing")
    assert design.column_feature_ids == ("ecog", "ecog")
    np.testing.assert_allclose(np.mean(design.train, axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.std(design.train, axis=0), 1.0, atol=1e-12)


def test_nominal_factor_is_selected_as_one_group_in_both_nuisance_tasks():
    rng = np.random.default_rng(221)
    rows = 480
    levels = np.asarray(["A", "B", "C", "D"])
    category_index = rng.integers(0, len(levels), size=rows)
    category = levels[category_index]
    treatment_probability = 1.0 / (
        1.0 + np.exp(-np.asarray([-2.0, -0.7, 0.8, 2.0])[category_index])
    )
    treatment = rng.binomial(1, treatment_probability)
    outcome = np.asarray([-3.0, -1.0, 1.0, 3.0])[category_index] + rng.normal(
        scale=0.45,
        size=rows,
    )
    extracted = pd.DataFrame(
        {
            "_oci_row_id": np.arange(rows),
            "factor": category,
            "noise": rng.normal(size=rows),
        }
    )
    dataset = pd.DataFrame({"treatment": treatment, "outcome": outcome})
    definitions = [
        _categorical("factor", "factor", levels.tolist()),
        _continuous("noise", "noise"),
    ]
    splitter = KFold(n_splits=4, shuffle=True, random_state=12)
    inner_splits = [
        {
            "inner_fold": index,
            "fit_row_ids": fit.tolist(),
            "heldout_row_ids": heldout.tolist(),
        }
        for index, (fit, heldout) in enumerate(splitter.split(extracted), start=1)
    ]
    policy = Stage2ElasticNetSelectionConfig(
        regularization_grid_size=7,
        max_iter=2_000,
    )

    selected, report, _dependencies, _latents = select_stage2_features_elastic_net(
        dataset=dataset,
        extracted_fit=extracted,
        definitions=definitions,
        inner_splits=inner_splits,
        treatment_column="treatment",
        outcome_column="outcome",
        outcome_type="continuous",
        seed=18,
        policy=policy,
    )

    selected_factor = next(row for row in selected if row["feature_id"] == "factor")
    assert selected_factor["nuisance_model_roles"] == ["treatment", "outcome"]
    assert report["penalized_model_family"] == "group_lasso_plus_ridge"
    assert report["encoding"]["missing_indicator"] == (
        "same_penalty_group_as_measurement"
    )
    for fold in report["nuisance_screen"]["folds"]:
        assert "factor" in fold["treatment"]["feature_group_l2_norms"]
        assert "factor" in fold["outcome"]["feature_group_l2_norms"]


def test_categorical_modifier_uses_one_grouped_heldout_dina_loss_score():
    rng = np.random.default_rng(733)
    rows = 900
    levels = np.asarray(["A", "B", "C"])
    category_index = rng.integers(0, len(levels), size=rows)
    category = levels[category_index]
    treatment = rng.binomial(1, 0.5, size=rows)
    interaction_effect = np.asarray([-2.8, 0.0, 2.8])[category_index]
    outcome_probability = 1.0 / (
        1.0 + np.exp(-(-0.3 + 0.15 * category_index + treatment * interaction_effect))
    )
    outcome = rng.binomial(1, outcome_probability)
    extracted = pd.DataFrame(
        {
            "_oci_row_id": np.arange(rows),
            "factor": category,
            "noise": rng.normal(size=rows),
        }
    )
    dataset = pd.DataFrame({"treatment": treatment, "outcome": outcome})
    definitions = [
        _categorical("factor", "factor", levels.tolist()),
        _continuous("noise", "noise"),
    ]
    splitter = KFold(n_splits=3, shuffle=True, random_state=44)
    inner_splits = [
        {
            "inner_fold": index,
            "fit_row_ids": fit.tolist(),
            "heldout_row_ids": heldout.tolist(),
        }
        for index, (fit, heldout) in enumerate(splitter.split(extracted), start=1)
    ]

    selected, report, _dependencies, _latents = select_stage2_features_elastic_net(
        dataset=dataset,
        extracted_fit=extracted,
        definitions=definitions,
        inner_splits=inner_splits,
        treatment_column="treatment",
        outcome_column="outcome",
        outcome_type="binary",
        seed=8,
        policy=Stage2ElasticNetSelectionConfig(
            regularization_grid_size=5,
            modifier_top_n_per_inner_fold=1,
            max_iter=2_000,
        ),
    )

    selected_factor = next(row for row in selected if row["feature_id"] == "factor")
    assert "effect_modifier" in selected_factor["roles"]
    assert report["effect_modifier_screen"]["model_family"] == "candidate_group_dina"
    for fold in report["effect_modifier_screen"]["folds"]:
        assert fold["selected_feature_ids"] == ["factor"]
        factor_test = next(
            row for row in fold["tests"] if row["feature_id"] == "factor"
        )
        assert factor_test["categorical_interactions_are_grouped"] is True
        assert len(factor_test["tested_interaction_columns"]) == 2
        assert factor_test["coefficient_feature_ids"] == ["factor"] * 2
        assert factor_test["effect_objective"] == "bernoulli_dina"
        assert factor_test["heldout_dina_loss_improvement"] > 0
        assert factor_test["effect_scale"] == "conditional_log_odds_ratio"


def test_modifier_top_n_defaults_to_ten_and_has_no_score_gate():
    policy = Stage2ElasticNetSelectionConfig()
    assert policy.modifier_top_n_per_inner_fold == 10
    assert policy.public_dict()["modifier_top_n_per_inner_fold"] == 10
    assert (
        "modifier_min_fold_r_loss_improvement" not in policy.public_dict()
    )
    with pytest.raises(ValueError, match="modifier_top_n_per_inner_fold"):
        Stage2ElasticNetSelectionConfig(modifier_top_n_per_inner_fold=0).validate()
    with pytest.raises(ValueError, match="modifier_ridge_alpha"):
        Stage2ElasticNetSelectionConfig(modifier_ridge_alpha=0.0).validate()
    with pytest.raises(ValueError, match="modifier_continuous_winsor_quantile"):
        Stage2ElasticNetSelectionConfig(
            modifier_continuous_winsor_quantile=0.25
        ).validate()


def test_modifier_continuous_values_are_winsorized_from_training_only():
    rng = np.random.default_rng(5)
    train = pd.DataFrame({"lab": rng.normal(7.5, 1.2, size=800)})
    valid = pd.DataFrame({"lab": [7.0, 7_200.0]})
    design = _encode_design(
        train,
        valid,
        [_continuous("lab", "lab")],
        categorical_min_count=5,
    )

    bounded = selection_module._winsorize_modifier_design(
        design,
        quantile=0.005,
    )

    assert bounded.valid[-1, 0] <= np.max(bounded.train[:, 0])
    assert bounded.valid[-1, 0] < design.valid[-1, 0]


def test_candidate_augmented_nuisances_are_group_elastic_nets():
    rng = np.random.default_rng(921)
    rows = 600
    candidate = rng.normal(size=rows)
    treatment_probability = 1.0 / (1.0 + np.exp(-1.8 * candidate))
    treatment = rng.binomial(1, treatment_probability, size=rows).astype(float)
    outcome = 3.0 * candidate + 1.5 * treatment + rng.normal(scale=0.25, size=rows)
    fit = np.arange(450)
    heldout = np.arange(450, rows)
    augmentation_splits = [
        (train.astype(int), valid.astype(int))
        for train, valid in KFold(
            n_splits=3,
            shuffle=True,
            random_state=8,
        ).split(fit)
    ]

    result = selection_module._candidate_r_learner_test(
        train=pd.DataFrame({"candidate": candidate[fit]}),
        valid=pd.DataFrame({"candidate": candidate[heldout]}),
        treatment_train=treatment[fit],
        treatment_valid=treatment[heldout],
        outcome_train=outcome[fit],
        outcome_valid=outcome[heldout],
        base_e_train=np.full(len(fit), 0.5),
        base_e_valid=np.full(len(heldout), 0.5),
        base_m_train=np.full(len(fit), float(np.mean(outcome[fit]))),
        base_m_valid=np.full(len(heldout), float(np.mean(outcome[fit]))),
        augmentation_splits=augmentation_splits,
        feature=_continuous("candidate", "candidate"),
        binary_outcome=False,
        config=Stage2ElasticNetSelectionConfig(
            regularization_grid_size=5,
            max_iter=3_000,
        ),
        seed=17,
    )

    assert result["status"] == "ok"
    augmentation = result["candidate_augmented_nuisance_models"]
    assert augmentation["model_family"] == "group_elastic_net"
    assert augmentation["training_predictions_are_cross_fitted"] is True
    assert augmentation["heldout_augmented_treatment_log_loss"] < (
        augmentation["heldout_base_treatment_log_loss"]
    )
    assert augmentation["heldout_augmented_outcome_loss"] < (
        augmentation["heldout_base_outcome_loss"]
    )


def test_binary_nuisance_screens_report_inner_and_pooled_aurocs():
    rng = np.random.default_rng(808)
    rows = 300
    signal = rng.normal(size=rows)
    treatment = rng.binomial(1, 1.0 / (1.0 + np.exp(-1.8 * signal)))
    outcome = rng.binomial(1, 1.0 / (1.0 + np.exp(-1.4 * signal)))
    dataset = pd.DataFrame({"treatment": treatment, "outcome": outcome})
    extracted = pd.DataFrame(
        {"_oci_row_id": np.arange(rows), "signal": signal}
    )
    splitter = KFold(n_splits=3, shuffle=True, random_state=31)
    inner_splits = [
        {
            "inner_fold": index,
            "fit_row_ids": fit.tolist(),
            "heldout_row_ids": heldout.tolist(),
        }
        for index, (fit, heldout) in enumerate(splitter.split(extracted), start=1)
    ]

    _selected, report, _dependencies, _latents = select_stage2_features_elastic_net(
        dataset=dataset,
        extracted_fit=extracted,
        definitions=[_continuous("signal", "signal")],
        inner_splits=inner_splits,
        treatment_column="treatment",
        outcome_column="outcome",
        outcome_type="binary",
        seed=73,
        policy=Stage2ElasticNetSelectionConfig(
            regularization_grid_size=5,
            max_iter=2_000,
        ),
    )

    assert 0.5 < report["nuisance_screen"]["overall_treatment_auroc"] <= 1.0
    assert 0.5 < report["nuisance_screen"]["overall_outcome_auroc"] <= 1.0
    assert 0.5 < report["cross_fitted_nuisance_models"][
        "overall_treatment_auroc"
    ] <= 1.0
    assert 0.5 < report["cross_fitted_nuisance_models"][
        "overall_outcome_auroc"
    ] <= 1.0
    for fold in report["nuisance_screen"]["folds"]:
        assert fold["treatment"]["heldout_auroc"] is not None
        assert fold["outcome"]["heldout_auroc"] is not None


def test_outer_modifier_set_is_union_of_each_inner_folds_top_n(monkeypatch):
    rng = np.random.default_rng(99)
    rows = 90
    treatment = np.tile([0, 1], rows // 2)
    dataset = pd.DataFrame(
        {"treatment": treatment, "outcome": rng.normal(size=rows)}
    )
    extracted = pd.DataFrame({
        "_oci_row_id": np.arange(rows),
        "candidate_a": rng.normal(size=rows),
        "candidate_b": rng.normal(size=rows),
        "candidate_c": rng.normal(size=rows),
    })
    splitter = KFold(n_splits=3, shuffle=True, random_state=7)
    inner_splits = [
        {
            "inner_fold": index,
            "fit_row_ids": fit.tolist(),
            "heldout_row_ids": heldout.tolist(),
        }
        for index, (fit, heldout) in enumerate(splitter.split(extracted), start=1)
    ]

    def constant_logistic(train_x, train_y, valid_x, _column_feature_ids, **_kwargs):
        probability = float(np.mean(train_y))
        return selection_module._PenalizedFit(
            coefficients=np.zeros(train_x.shape[1]),
            train_prediction=np.full(len(train_y), probability),
            valid_prediction=np.full(len(valid_x), probability),
            regularization=0.1,
            cv_folds=2,
            status="ok",
            iterations=1,
            converged=True,
        )

    def controlled_squared(
        train_x,
        train_y,
        valid_x,
        _column_feature_ids,
        *,
        fit_intercept=True,
        **_kwargs,
    ):
        coefficients = np.zeros(train_x.shape[1])
        train_prediction = np.full(len(train_y), float(np.mean(train_y)))
        valid_prediction = np.full(len(valid_x), float(np.mean(train_y)))
        return selection_module._PenalizedFit(
            coefficients=coefficients,
            train_prediction=train_prediction,
            valid_prediction=valid_prediction,
            regularization=0.1,
            cv_folds=2,
            status="ok",
            iterations=1,
            converged=True,
        )

    monkeypatch.setattr(selection_module, "_logistic_elastic_net", constant_logistic)
    monkeypatch.setattr(selection_module, "_squared_error_elastic_net", controlled_squared)
    improvements = iter(
        [
            -0.01, -0.20, -0.30,
            -0.20, -0.01, -0.30,
            -0.30, -0.20, -0.01,
        ]
    )

    def controlled_r_learner_test(*, feature, **_kwargs):
        improvement = next(improvements)
        return {
            "feature_id": feature["feature_id"],
            "name": feature["name"],
            "status": "ok",
            "candidate_strategy": "continuous",
            "encoded_main_columns": [feature["name"]],
            "candidate_interaction_columns": [feature["name"]],
            "tested_interaction_columns": [feature["name"]],
            "categorical_interactions_are_grouped": False,
            "missingness_interactions": False,
            "heldout_reduced_r_loss": 1.0,
            "heldout_full_r_loss": 1.0 - improvement,
            "heldout_r_loss_improvement": improvement,
            "heldout_relative_r_loss_improvement": improvement,
        }

    monkeypatch.setattr(
        selection_module,
        "_candidate_r_learner_test",
        controlled_r_learner_test,
    )

    selected, report, _dependencies, _latents = select_stage2_features_elastic_net(
        dataset=dataset,
        extracted_fit=extracted,
        definitions=[
            _continuous("candidate_a", "candidate_a"),
            _continuous("candidate_b", "candidate_b"),
            _continuous("candidate_c", "candidate_c"),
        ],
        inner_splits=inner_splits,
        treatment_column="treatment",
        outcome_column="outcome",
        outcome_type="continuous",
        seed=15,
        policy=Stage2ElasticNetSelectionConfig(
            regularization_grid_size=5,
            modifier_top_n_per_inner_fold=1,
        ),
    )

    selected_by_id = {row["feature_id"]: row for row in selected}
    assert all(
        "effect_modifier" in selected_by_id[feature_id]["roles"]
        for feature_id in ("candidate_a", "candidate_b", "candidate_c")
    )
    assert report["effect_modifier_screen"]["required_votes"] == 1
    assert report["effect_modifier_screen"][
        "r_loss_improvement_threshold_is_not_a_selection_gate"
    ] is True
    assert report["effect_modifier_screen"]["votes"] == {
        "candidate_a": 1,
        "candidate_b": 1,
        "candidate_c": 1,
    }
    assert report["effect_modifier_screen"]["stable_effect_modifier_feature_ids"] == [
        "candidate_a",
        "candidate_b",
        "candidate_c",
    ]
    assert [fold["selected_count"] for fold in report["effect_modifier_screen"]["folds"]] == [
        1,
        1,
        1,
    ]
