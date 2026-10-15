import json

import numpy as np
import pytest
from sklearn.base import clone

from oci.models.causal_forest_head import CausalForestHead
from oci.models.elastic_net_nuisance import (
    ElasticNetLogisticClassifier,
    ElasticNetRegressor,
)


def _data():
    rng = np.random.RandomState(91)
    n = 96
    effect = rng.normal(size=(n, 3))
    control = rng.normal(size=(n, 2))
    treatment = np.tile(np.array([0, 1]), n // 2)
    outcome = (
        0.25 * treatment
        + effect[:, 0]
        + control[:, 0]
        + rng.normal(scale=0.8, size=n)
        > 0.1
    ).astype(float)
    return effect, control, treatment, outcome


def test_prediction_calls_explicit_binary_treatment_contrast():
    class EffectSpy:
        def __init__(self):
            self.calls = []

        def effect(self, X, **kwargs):
            self.calls.append((X, kwargs))
            return np.zeros(len(X), dtype=float)

    head = object.__new__(CausalForestHead)
    head._fitted = True
    head.model = EffectSpy()
    head.inference = False
    values = np.ones((3, 2), dtype=float)

    result = head.predict(values, return_ci=False)

    np.testing.assert_array_equal(result["tau_pred"], np.zeros(3))
    assert head.model.calls[0][1] == {"T0": 0, "T1": 1}


@pytest.mark.parametrize(
    ("outcome_type", "expected_class", "discrete_outcome", "prediction_interface"),
    [
        ("binary", ElasticNetLogisticClassifier, True, "predict_proba"),
        ("continuous", ElasticNetRegressor, False, "predict"),
    ],
)
def test_head_uses_outcome_typed_nuisance_model(
    outcome_type,
    expected_class,
    discrete_outcome,
    prediction_interface,
):
    head = CausalForestHead(
        outcome_type=outcome_type,
        n_estimators=8,
        subforest_size=4,
        min_samples_leaf=2,
        nuisance_regularization_grid_size=5,
        nuisance_max_iter=500,
        tune_model=False,
        n_jobs=1,
    )

    model = head._create_model()

    assert type(model.model_y) is expected_class
    assert model.discrete_outcome is discrete_outcome
    assert head._configured_forest_parameters()["discrete_outcome"] is discrete_outcome
    assert (
        head._configured_nuisance_parameters()["outcome_model_contract"][
            "prediction_interface"
        ]
        == prediction_interface
    )


def test_fitted_head_audits_every_crossfit_elastic_net_clone():
    effect, control, treatment, outcome = _data()
    head = CausalForestHead(
        outcome_type="binary",
        n_estimators=8,
        subforest_size=4,
        min_samples_leaf=2,
        nuisance_regularization_grid_size=5,
        nuisance_maximum_log10_c=2.0,
        nuisance_max_iter=10_000,
        nuisance_tolerance=1e-4,
        tune_model=False,
        n_jobs=1,
    ).fit(
        X=effect,
        W=control,
        T=treatment,
        Y=outcome,
    )

    audit = head.fit_audit()

    assert audit["effective_nuisance_parameters"]["model_family"] == "elastic_net"
    for role, fitted_models in audit["fitted_nuisance_models"].items():
        assert role in {"treatment", "outcome"}
        assert fitted_models
        assert all(item["crossfit_path"] for item in fitted_models)
        assert all(
            item["estimator"].endswith("ElasticNetLogisticClassifier")
            for item in fitted_models
        )
        assert all(item["fit_mode"] == "cross_validated" for item in fitted_models)
        assert all(item["effective_cv_folds"] >= 2 for item in fitted_models)
        assert all(
            item["selected_regularization"]["parameter"] == "C"
            for item in fitted_models
        )
        assert all(
            isinstance(item["optimization"]["iteration_limit_reached"], bool)
            for item in fitted_models
        )
    json.dumps(audit)


def test_elastic_net_wrappers_are_clone_safe_and_audit_empty_designs():
    design = np.empty((4, 0), dtype=float)
    classifier = clone(ElasticNetLogisticClassifier()).fit(
        design,
        np.array([0, 1, 0, 1]),
    )
    regressor = clone(ElasticNetRegressor()).fit(
        design,
        np.array([1.0, 2.0, 3.0, 4.0]),
    )

    np.testing.assert_allclose(classifier.predict_proba(design)[:, 1], 0.5)
    np.testing.assert_allclose(regressor.predict(design), 2.5)
    assert classifier.fit_audit()["fit_mode"] == "constant"
    assert classifier.fit_audit()["selected_regularization"] is None
    assert regressor.fit_audit()["fit_mode"] == "constant"
    assert regressor.fit_audit()["selected_regularization"] is None


def test_fitted_continuous_nuisance_audit_records_selected_alpha():
    rng = np.random.RandomState(7)
    design = rng.normal(size=(36, 3))
    target = 0.5 * design[:, 0] - 0.2 * design[:, 1] + rng.normal(
        scale=0.1,
        size=len(design),
    )
    regressor = ElasticNetRegressor(
        cv_folds=3,
        regularization_grid_size=5,
        max_iter=2_000,
        n_jobs=1,
    ).fit(design, target)

    audit = regressor.fit_audit()

    assert audit["fit_mode"] == "cross_validated"
    assert audit["effective_cv_folds"] == 3
    assert audit["selected_regularization"]["parameter"] == "alpha"
    assert audit["selected_regularization"]["value"] > 0
    assert audit["optimization"]["duality_gap"] is not None
    json.dumps(audit)


def test_binary_head_rejects_nonbinary_outcomes():
    head = CausalForestHead(
        outcome_type="binary",
        n_estimators=8,
        subforest_size=4,
        tune_model=False,
    )
    effect = np.ones((4, 1), dtype=float)
    treatment = np.array([0, 1, 0, 1], dtype=float)

    with pytest.raises(ValueError, match="must contain exactly both 0 and 1"):
        head.fit(effect, treatment, np.array([0.0, 0.2, 0.8, 1.0]))


def _continuous_data(n=1200, seed=4):
    rng = np.random.RandomState(seed)
    effect = rng.normal(size=(n, 2))
    control = rng.normal(size=(n, 2))
    propensity = 1.0 / (1.0 + np.exp(-0.8 * control[:, 0]))
    treatment = rng.binomial(1, propensity)
    tau = 1.0 + 0.5 * effect[:, 0]
    outcome = control[:, 0] + 0.5 * effect[:, 1] + tau * treatment + rng.normal(size=n)
    return effect, control, treatment, outcome


def _small_head(**overrides):
    values = dict(
        n_estimators=40,
        subforest_size=4,
        min_samples_leaf=5,
        nuisance_regularization_grid_size=5,
        n_jobs=1,
    )
    values.update(overrides)
    return CausalForestHead(**values)


def test_prediction_reports_finite_pointwise_standard_errors():
    effect, control, treatment, outcome = _data()
    head = _small_head(outcome_type="binary").fit(effect, treatment, outcome, W=control)

    result = head.predict(effect[:10], return_ci=True)

    assert result["tau_std"] is not None
    assert result["tau_std"].shape == (10,)
    assert np.isfinite(result["tau_std"]).all() and (result["tau_std"] > 0).all()
    np.testing.assert_allclose(
        result["tau_upper"] - result["tau_lower"],
        2 * 1.959964 * result["tau_std"],
        rtol=1e-4,
    )


def test_tuning_is_opt_in_and_every_tuned_parameter_is_audited():
    from oci.models.causal_forest_head import CAUSAL_FOREST_TUNING_GRID

    effect, control, treatment, outcome = _continuous_data(n=400)
    assert CausalForestHead().tune_model is False

    fixed = _small_head(outcome_type="continuous").fit(effect, treatment, outcome, W=control)
    fixed_audit = fixed.fit_audit()
    assert fixed_audit["tuning_attempted"] is False
    assert fixed_audit["tuning_params"] is None
    assert fixed_audit["tuned_parameter_changes"] == {}

    tuned = _small_head(outcome_type="continuous", tune_model=True).fit(
        effect, treatment, outcome, W=control
    )
    audit = tuned.fit_audit()
    assert audit["tuning_attempted"] is True
    assert audit["tuning_params"] == {k: list(v) for k, v in CAUSAL_FOREST_TUNING_GRID.items()}
    assert set(CAUSAL_FOREST_TUNING_GRID) <= set(audit["configured_parameters"])
    assert set(CAUSAL_FOREST_TUNING_GRID) <= set(audit["effective_parameters"])
    for name, change in audit["tuned_parameter_changes"].items():
        assert change["configured"] == audit["configured_parameters"][name]
        assert change["effective"] == audit["effective_parameters"][name]
    changed = {
        name
        for name in audit["configured_parameters"]
        if audit["configured_parameters"][name] != audit["effective_parameters"][name]
    }
    assert changed == set(audit["tuned_parameter_changes"])
    json.dumps(audit)


def test_doubly_robust_ate_contract():
    from oci.models.causal_forest_head import summarize_doubly_robust_scores

    effect, control, treatment, outcome = _continuous_data()
    fit, held = slice(0, 800), slice(800, None)
    head = _small_head(outcome_type="continuous").fit(
        effect[fit], treatment[fit], outcome[fit], W=control[fit]
    )

    training = head.training_doubly_robust_ate()
    assert training["ate"] == pytest.approx(float(np.ravel(head.model.ate_)[0]))
    assert training["ate_stderr"] == pytest.approx(float(np.ravel(head.model.ate_stderr_)[0]))
    assert training["n"] == 800

    scores = head.doubly_robust_scores(
        effect[held], treatment[held], outcome[held], W=control[held]
    )
    e_hat, m_hat = head.nuisance_predictions(effect[held], control[held])
    tau = head.predict(effect[held], return_ci=False)["tau_pred"]
    residual = treatment[held] - e_hat
    expected = tau + (outcome[held] - m_hat - tau * residual) * residual / np.maximum(
        e_hat * (1 - e_hat), 0.01
    )
    np.testing.assert_allclose(scores, expected)

    summary = summarize_doubly_robust_scores(scores)
    assert summary["n"] == 400
    assert summary["ate"] == pytest.approx(np.mean(scores))
    assert summary["ate_stderr"] == pytest.approx(np.std(scores, ddof=1) / np.sqrt(400))
    # True ATE is 1.0; both DR estimates cover it comfortably.
    assert abs(summary["ate"] - 1.0) < 3 * summary["ate_stderr"] + 0.05
    assert abs(training["ate"] - 1.0) < 3 * training["ate_stderr"] + 0.05
