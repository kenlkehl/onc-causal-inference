"""Internal standardization and refit auditing of the elastic-net nuisances."""

import json
import warnings

import numpy as np
import pytest
from scipy import sparse
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from oci.models.elastic_net_nuisance import (
    ElasticNetLogisticClassifier,
    ElasticNetRegressor,
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


def test_regressor_predictions_are_invariant_to_column_scale():
    design, _, continuous, holdout = _data()
    reference = ElasticNetRegressor(regularization_grid_size=6).fit(design, continuous)
    rescaled = ElasticNetRegressor(regularization_grid_size=6).fit(_rescaled(design), continuous)

    np.testing.assert_allclose(
        rescaled.predict(_rescaled(design)),
        reference.predict(design),
        atol=1e-6,
        rtol=0,
    )
    np.testing.assert_allclose(
        rescaled.predict(_rescaled(holdout)),
        reference.predict(holdout),
        atol=1e-6,
        rtol=0,
    )
    assert (
        rescaled.fit_audit()["selected_regularization"]
        == reference.fit_audit()["selected_regularization"]
    )


def test_scaler_is_fit_on_fit_rows_only_and_reused_for_prediction():
    design, binary, continuous, holdout = _data()
    classifier = _fit_quietly(
        ElasticNetLogisticClassifier(regularization_grid_size=6), design, binary
    )
    regressor = ElasticNetRegressor(regularization_grid_size=6).fit(design, continuous)

    for estimator in (classifier, regressor):
        np.testing.assert_allclose(estimator.scaler_.mean_, design.mean(axis=0))
        np.testing.assert_allclose(estimator.scaler_.scale_, design.std(axis=0))
        before = estimator.scaler_.mean_.copy()
        # Prediction on shifted rows must not refit the scaler.
        _ = (
            estimator.predict_proba(holdout + 50.0)
            if hasattr(estimator, "predict_proba")
            else estimator.predict(holdout + 50.0)
        )
        np.testing.assert_array_equal(estimator.scaler_.mean_, before)

    np.testing.assert_allclose(
        classifier.predict_proba(holdout),
        classifier.model_.predict_proba(classifier.scaler_.transform(holdout)),
    )
    np.testing.assert_allclose(
        regressor.predict(holdout),
        regressor.model_.predict(regressor.scaler_.transform(holdout)),
    )


@pytest.mark.parametrize(
    ("estimator", "overrides"),
    [
        (
            ElasticNetLogisticClassifier(),
            {"l1_ratio": 0.5, "max_iter": 321, "minimum_log10_c": -1.0},
        ),
        (
            ElasticNetRegressor(),
            {"l1_ratio": 0.5, "max_iter": 321, "minimum_log10_alpha": -4.0},
        ),
    ],
)
def test_wrappers_round_trip_through_clone_and_get_params(estimator, overrides):
    configured = clone(estimator).set_params(**overrides)
    copied = clone(configured)

    assert copied is not configured
    assert copied.get_params(deep=False) == configured.get_params(deep=False)
    for name, value in overrides.items():
        assert copied.get_params()[name] == value
    assert type(copied)(**configured.get_params(deep=False)).get_params() == (
        configured.get_params()
    )

    design, binary, continuous, holdout = _data()
    target = binary if isinstance(estimator, ElasticNetLogisticClassifier) else continuous
    fitted = _fit_quietly(clone(configured), design, target)
    refitted = _fit_quietly(clone(fitted), design, target)
    # Cloning a fitted wrapper yields an unfitted copy with identical results.
    assert not hasattr(clone(fitted), "scaler_")
    predict = "predict_proba" if hasattr(fitted, "predict_proba") else "predict"
    np.testing.assert_array_equal(
        getattr(refitted, predict)(holdout), getattr(fitted, predict)(holdout)
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


def test_classifier_audit_reports_refit_iteration_limit():
    design, binary, _, _ = _data()
    estimator = ElasticNetLogisticClassifier(
        regularization_grid_size=4, max_iter=1, tolerance=1e-12
    )
    with pytest.warns(ConvergenceWarning):
        estimator.fit(design, binary)

    optimization = estimator.fit_audit()["optimization"]

    assert optimization["configured_max_iter"] == 1
    assert optimization["refit_iterations"] == 1
    assert optimization["refit_iteration_limit_reached"] is True
    assert optimization["cv_iteration_limit_reached"] is True
    assert optimization["iteration_limit_reached"] is True
    assert optimization["maximum_iterations_observed"] == 1


def test_fixed_grid_classifier_audits_refit_without_cv():
    design, _, _, _ = _data(n=30)
    target = np.zeros(len(design), dtype=int)
    target[0] = 1  # one positive row: too few for stratified CV
    estimator = _fit_quietly(ElasticNetLogisticClassifier(), design, target)

    audit = estimator.fit_audit()

    assert audit["fit_mode"] == "fixed_grid_midpoint"
    assert audit["optimization"]["cv_maximum_iterations_observed"] is None
    assert audit["optimization"]["cv_iteration_limit_reached"] is False
    assert audit["optimization"]["refit_iterations"] >= 1
    assert audit["standardization"]["applied"] is True


def test_regressor_audit_reports_refit_iterations_and_limit():
    design, _, continuous, _ = _data()
    converged = ElasticNetRegressor(regularization_grid_size=6).fit(design, continuous)
    with pytest.warns(ConvergenceWarning):
        limited = ElasticNetRegressor(regularization_grid_size=4, max_iter=1, tolerance=1e-12).fit(
            design, continuous
        )

    optimization = converged.fit_audit()["optimization"]
    assert optimization["refit_iterations"] == int(converged.model_.n_iter_)
    assert optimization["refit_iterations"] >= 1
    assert optimization["maximum_iterations_observed"] == optimization["refit_iterations"]
    assert optimization["refit_iteration_limit_reached"] is False
    assert optimization["iteration_limit_reached"] is False
    assert optimization["cv_maximum_iterations_observed"] is None
    assert optimization["cv_iteration_limit_reached"] is None
    assert converged.fit_audit()["standardization"]["with_mean"] is True

    limited_optimization = limited.fit_audit()["optimization"]
    assert limited_optimization["refit_iterations"] == 1
    assert limited_optimization["refit_iteration_limit_reached"] is True
    assert limited_optimization["iteration_limit_reached"] is True
    json.dumps(limited.fit_audit())


def test_classifier_accepts_sparse_designs_without_centering():
    design, binary, _, holdout = _data()
    rng = np.random.RandomState(11)
    mask = rng.uniform(size=design.shape) < 0.5
    dense = np.where(mask, 0.0, np.abs(design))
    dense_holdout = np.where(rng.uniform(size=holdout.shape) < 0.5, 0.0, np.abs(holdout))
    sparse_design = sparse.csr_matrix(dense)

    estimator = _fit_quietly(
        ElasticNetLogisticClassifier(regularization_grid_size=6), sparse_design, binary
    )
    rescaled = _fit_quietly(
        ElasticNetLogisticClassifier(regularization_grid_size=6),
        sparse.csr_matrix(_rescaled(dense)),
        binary,
    )

    audit = estimator.fit_audit()
    assert audit["standardization"]["with_mean"] is False
    assert sparse.issparse(estimator.scaler_.transform(sparse_design))
    probabilities = estimator.predict_proba(sparse.csr_matrix(dense_holdout))
    assert probabilities.shape == (len(dense_holdout), 2)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities, estimator.predict_proba(dense_holdout))
    np.testing.assert_allclose(
        rescaled.predict_proba(sparse.csr_matrix(_rescaled(dense_holdout))),
        probabilities,
        atol=1e-6,
        rtol=0,
    )

    # A dense-fitted (centered) model also accepts sparse prediction input.
    dense_fit = _fit_quietly(
        ElasticNetLogisticClassifier(regularization_grid_size=6), dense, binary
    )
    np.testing.assert_allclose(
        dense_fit.predict_proba(sparse.csr_matrix(dense_holdout)),
        dense_fit.predict_proba(dense_holdout),
    )


def test_constant_columns_do_not_produce_nan_or_change_predictions():
    design, binary, continuous, holdout = _data()

    def with_constants(values):
        return np.column_stack([values, np.full(len(values), 5.0), np.zeros(len(values))])

    classifier = _fit_quietly(
        ElasticNetLogisticClassifier(regularization_grid_size=6),
        with_constants(design),
        binary,
    )
    regressor = ElasticNetRegressor(regularization_grid_size=6).fit(
        with_constants(design), continuous
    )
    classifier_reference = _fit_quietly(
        ElasticNetLogisticClassifier(regularization_grid_size=6), design, binary
    )
    regressor_reference = ElasticNetRegressor(regularization_grid_size=6).fit(design, continuous)

    probabilities = classifier.predict_proba(with_constants(holdout))
    predictions = regressor.predict(with_constants(holdout))
    assert np.isfinite(probabilities).all()
    assert np.isfinite(predictions).all()
    assert np.isfinite(classifier.model_.coef_).all()
    assert np.isfinite(regressor.model_.coef_).all()
    np.testing.assert_allclose(
        probabilities, classifier_reference.predict_proba(holdout), atol=1e-8
    )
    np.testing.assert_allclose(predictions, regressor_reference.predict(holdout), atol=1e-8)
    assert classifier.fit_audit()["standardization"]["zero_variance_columns"] == 2
    assert regressor.fit_audit()["standardization"]["zero_variance_columns"] == 2

    # Every column constant: standardization leaves no signal, so both
    # wrappers use the exact intercept-only fit instead of an unconverged
    # saga intercept.
    all_constant = np.column_stack([np.ones(len(design)), np.zeros(len(design))])
    flat = ElasticNetLogisticClassifier().fit(all_constant, binary)
    flat_regressor = ElasticNetRegressor().fit(all_constant, continuous)
    np.testing.assert_allclose(flat.predict_proba(all_constant)[:, 1], binary.mean())
    np.testing.assert_allclose(flat_regressor.predict(all_constant), continuous.mean())
    for audit in (flat.fit_audit(), flat_regressor.fit_audit()):
        assert audit["fit_mode"] == "constant"
        assert audit["constant_reason"] == "no_varying_features"
        assert audit["standardization"]["applied"] is True
        assert audit["standardization"]["zero_variance_columns"] == 2
        assert audit["optimization"]["refit_iterations"] is None
    sparse_flat = ElasticNetLogisticClassifier().fit(
        sparse.csr_matrix(np.zeros((len(design), 3))), binary
    )
    assert sparse_flat.fit_audit()["constant_reason"] == "no_varying_features"


def test_constant_fit_mode_reports_unapplied_standardization():
    design = np.empty((4, 0), dtype=float)
    classifier = ElasticNetLogisticClassifier().fit(design, np.array([0, 1, 0, 1]))
    regressor = ElasticNetRegressor().fit(design, np.array([1.0, 2.0, 3.0, 4.0]))

    for audit in (classifier.fit_audit(), regressor.fit_audit()):
        assert audit["constant_reason"] == "no_features"
        assert audit["standardization"]["applied"] is False
        assert audit["standardization"]["with_mean"] is None
        assert audit["optimization"]["refit_iterations"] is None
        assert audit["optimization"]["refit_iteration_limit_reached"] is False
        assert audit["optimization"]["maximum_iterations_observed"] == 0
        assert audit["optimization"]["iteration_limit_reached"] is False
        json.dumps(audit)

    single_class = ElasticNetLogisticClassifier().fit(
        sparse.csr_matrix(np.ones((4, 2))), np.zeros(4, dtype=int)
    )
    assert single_class.fit_audit()["constant_reason"] == "single_class"
    np.testing.assert_allclose(
        single_class.predict_proba(sparse.csr_matrix(np.ones((3, 2))))[:, 1], 1e-6
    )
