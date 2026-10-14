"""The population treatment term survives regularization without forcing an effect."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy.special import logit

from oci.inference import stage2_effect_estimators as effects
from oci.inference import stage2_elastic_net_selection as linear
from oci.inference import stage2_multi_model_selection as numerical
from tests.test_stage2_multi_model import feature, small_policy


@pytest.mark.parametrize("ratio", [0.0, 0.8, 1.0])
def test_explicit_exemption_covers_group_lasso_and_ridge(ratio):
    # A two-column exempt group also checks expansion from groups to coefficients.
    values = np.array([2.0, -3.0, 4.0, -5.0])
    ids = ["covariate", "treatment", "treatment", "interaction"]
    kwargs = dict(step=0.5, regularization=100.0, group_ratio=ratio)
    actual = linear._group_ridge_prox(
        values, groups=linear._group_structure(ids, unpenalized_feature_ids=["treatment"]),
        **kwargs,
    )
    default = linear._group_ridge_prox(values, groups=linear._group_structure(ids), **kwargs)
    np.testing.assert_array_equal(actual[1:3], values[1:3])
    np.testing.assert_array_equal(actual[[0, 3]], default[[0, 3]])
    assert np.all(np.abs(default) < np.abs(values))


def strong_policy():
    return replace(
        small_policy(), minimum_log10_alpha=1, maximum_log10_alpha=2,
        max_iter=5000, optimization_tolerance=1e-9,
    )


def balanced_data(binary):
    treatment = np.repeat([0, 0, 1, 1], 80)
    z = np.repeat([-1, 1, -1, 1], 80)
    # A clinical feature called treatment must still be penalized; only the
    # explicit treatment-design column receives the exemption.
    frame = pd.DataFrame({"_oci_row_id": range(320), "treatment": z})
    outcome = (
        np.concatenate([np.r_[np.ones(k), np.zeros(80 - k)] for k in (10, 30, 50, 70)])
        if binary else 1.5 + 2 * treatment + 0.5 * z
    )
    return frame, treatment, outcome


@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("modifiers", [[], ["treatment"]])
def test_strong_penalty_preserves_treatment_in_every_cv_fit_and_final_refit(
    monkeypatch, binary, modifiers
):
    frame, treatment, outcome = balanced_data(binary)
    original = linear._fit_group_elastic_net
    fitted = []

    def checked_fit(x, y, **kwargs):
        state = original(x, y, **kwargs)
        assert state.converged
        means = [y[x[:, 0] == arm].mean() for arm in (0, 1)]
        expected = logit(means[1]) - logit(means[0]) if binary else means[1] - means[0]
        assert state.coefficients[0] == pytest.approx(expected, abs=1e-6)
        np.testing.assert_array_equal(state.coefficients[1:], 0)
        fitted.append(len(y))
        return state

    monkeypatch.setattr(linear, "_fit_group_elastic_net", checked_fit)
    policy = strong_policy()
    result = effects.fit_interaction_effect(
        train=frame, valid=frame, definitions=[feature("treatment")],
        modifier_ids=modifiers, treatment=treatment, outcome=outcome,
        binary=binary, policy=policy, seed=37,
    )
    assert fitted == [160] * (2 * policy.multi_model.regularization_grid_size) + [320]
    assert result["audit"]["unpenalized_groups"] == ["treatment"]
    np.testing.assert_allclose(result["tau"], 0.5 if binary else 2.0, atol=1e-6)


def test_residual_treatment_survives_while_modifier_coefficients_are_zero():
    frame, treatment, _ = balanced_data(False)
    residual_t = treatment - 0.5
    residual_y = residual_t * (2 + frame.treatment.to_numpy())
    result = numerical._fit_linear(
        frame, frame, [feature("treatment")], residual_y,
        binary=False, kind="rlearner", treatment=residual_t, valid_treatment=residual_t,
        policy=strong_policy(), seed=37, ratio=0.8,
    )
    assert result["coefficients"][0] == pytest.approx(2, abs=1e-6)
    np.testing.assert_array_equal(result["coefficients"][1:], 0)
    np.testing.assert_allclose(result["prediction"], 2 * residual_t, atol=1e-6)


def test_nuisance_covariate_named_treatment_remains_penalized():
    frame, treatment, _ = balanced_data(False)
    result = numerical._fit_linear(
        frame, frame, [feature("treatment")], 2 * frame.treatment.to_numpy(),
        binary=False, kind="main", treatment=treatment, valid_treatment=treatment,
        policy=strong_policy(), seed=37, ratio=0.8,
    )
    np.testing.assert_array_equal(result["coefficients"], 0)
    assert result["audit"]["unpenalized_groups"] == []


@pytest.mark.parametrize("binary", [False, True])
def test_unpenalized_treatment_can_still_be_estimated_as_zero(binary):
    frame, treatment, _ = balanced_data(binary)
    outcome = np.tile([0, 1], 160) if binary else frame.treatment.to_numpy()
    result = effects.fit_interaction_effect(
        train=frame, valid=frame, definitions=[feature("treatment")], modifier_ids=[],
        treatment=treatment, outcome=outcome, binary=binary, policy=strong_policy(), seed=37,
    )
    np.testing.assert_allclose(result["tau"], 0, atol=1e-6)
