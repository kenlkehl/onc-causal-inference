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
