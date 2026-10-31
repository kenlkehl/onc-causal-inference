"""Internal standardization and refit auditing of the elastic-net nuisances."""

import json
import warnings

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from oci.models.elastic_net_nuisance import (
    ElasticNetLogisticClassifier,
)


def _data(n=240, seed=3):
    rng = np.random.RandomState(seed)
    design = rng.normal(size=(n, 5))
    linear = design[:, 0] - 0.7 * design[:, 2] + 0.4 * design[:, 4]
    binary = rng.binomial(1, 1.0 / (1.0 + np.exp(-linear)))
    continuous = linear + rng.normal(scale=0.5, size=n)
    holdout = rng.normal(size=(60, 5))
    return design, binary, continuous, holdout


def _rescaled(values, column=2, factor=1000.0):
    result = np.array(values, dtype=float, copy=True)
    result[:, column] *= factor
    return result


def _fit_quietly(estimator, design, target):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return estimator.fit(design, target)


def test_classifier_predictions_are_invariant_to_column_scale():
    design, binary, _, holdout = _data()
    reference = ElasticNetLogisticClassifier(regularization_grid_size=6)
    rescaled = clone(reference)

    _fit_quietly(reference, design, binary)
    _fit_quietly(rescaled, _rescaled(design), binary)

    np.testing.assert_allclose(
        rescaled.predict_proba(_rescaled(design)),
        reference.predict_proba(design),
        atol=1e-6,
        rtol=0,
    )
    np.testing.assert_allclose(
        rescaled.predict_proba(_rescaled(holdout)),
        reference.predict_proba(holdout),
        atol=1e-6,
        rtol=0,
    )
    assert (
        rescaled.fit_audit()["selected_regularization"]
        == reference.fit_audit()["selected_regularization"]
    )


def test_classifier_explicit_refit_matches_sklearn_refit_rule_and_is_audited():
    design, binary, _, _ = _data()
    estimator = _fit_quietly(
        ElasticNetLogisticClassifier(regularization_grid_size=6), design, binary
    )
    standardized = StandardScaler().fit_transform(design)
    reference = LogisticRegressionCV(
        Cs=np.logspace(-2.0, 4.0, 6),
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        scoring="neg_log_loss",
        refit=True,
        l1_ratios=[0.8],
        penalty="elasticnet",
        solver="saga",
        fit_intercept=True,
        max_iter=5_000,
        tol=1e-4,
        random_state=42,
        n_jobs=1,
    ).fit(standardized, binary)

    audit = estimator.fit_audit()
    optimization = audit["optimization"]

    assert audit["selected_regularization"] == {
        "parameter": "C",
        "value": pytest.approx(float(reference.C_[0])),
    }
    np.testing.assert_allclose(estimator.model_.coef_, reference.coef_, atol=1e-10)
    np.testing.assert_allclose(estimator.model_.intercept_, reference.intercept_, atol=1e-10)
    assert optimization["cv_maximum_iterations_observed"] == int(np.max(reference.n_iter_))
    assert isinstance(optimization["refit_iterations"], int)
    assert optimization["refit_iterations"] >= 1
    assert optimization["refit_iterations"] == int(estimator.model_.n_iter_[0])
    assert optimization["maximum_iterations_observed"] == max(
        optimization["cv_maximum_iterations_observed"],
        optimization["refit_iterations"],
    )
    assert optimization["refit_iteration_limit_reached"] is False
    assert optimization["cv_iteration_limit_reached"] is False
    assert optimization["iteration_limit_reached"] is False
    assert audit["standardization"] == {
        "method": "StandardScaler",
        "fit_scope": "rows passed to fit (unsupervised; refit per cross-fitting fold)",
        "applied": True,
        "with_mean": True,
        "zero_variance_columns": 0,
    }
    json.dumps(audit)
