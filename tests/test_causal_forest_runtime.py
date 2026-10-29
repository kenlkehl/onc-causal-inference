import json

import numpy as np
import pytest

from oci.models.causal_forest_head import CausalForestHead


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
        for item in fitted_models:
            # Every EconML clone fits its own label-free scaler and records
            # the explicit refit's iterations alongside the CV iterations.
            assert item["standardization"]["method"] == "StandardScaler"
            assert item["standardization"]["applied"] is True
            assert item["standardization"]["with_mean"] is True
            optimization = item["optimization"]
            assert isinstance(optimization["refit_iterations"], int)
            assert optimization["refit_iterations"] >= 1
            assert isinstance(optimization["cv_maximum_iterations_observed"], int)
            assert optimization["maximum_iterations_observed"] == max(
                optimization["refit_iterations"],
                optimization["cv_maximum_iterations_observed"],
            )
            assert optimization["iteration_limit_reached"] is (
                optimization["refit_iteration_limit_reached"]
                or optimization["cv_iteration_limit_reached"]
            )
    json.dumps(audit)


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
