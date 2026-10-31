"""Super-learner nuisance estimators and their Stage 2 integration."""

import json

import numpy as np
import pytest

from oci.models import super_learner_nuisance as sl


def _binary(n=240, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 5))
    logit = 1.2 * (x[:, 0] < 0) + x[:, 1] ** 2 - 1.0
    return x, rng.binomial(1, 1 / (1 + np.exp(-logit)))


def _continuous(n=240, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 5))
    return x, np.where(x[:, 0] < 0, 1.0, -1.0) + 0.5 * x[:, 1] + rng.normal(size=n)


@pytest.mark.parametrize("binary", [True])
def test_weights_lie_on_the_simplex_and_the_audit_is_complete(binary):
    x, y = _binary() if binary else _continuous()
    cls = sl.SuperLearnerClassifier if binary else sl.SuperLearnerRegressor
    model = cls(cv_folds=3, random_state=7).fit(x, y)
    audit = model.fit_audit()
    weights = np.array(list(audit["stack_weights"].values()))
    assert (weights >= 0).all() and weights.sum() == pytest.approx(1.0)
    assert list(audit["stack_weights"]) == list(sl.LEARNERS)
    assert audit["fit_mode"] == "stacked" and audit["effective_cv_folds"] == 3
    assert audit["loss"] == ("log_loss" if binary else "mean_squared_error")
    assert set(audit["learner_cv_loss"]) == set(sl.LEARNERS)
    assert audit["stack_cv_loss"] is not None
    assert audit["stack_cv_loss_method"] == "weights_refitted_on_the_other_internal_folds"
    assert audit["refitted_learners"] == [
        name for name in sl.LEARNERS if audit["stack_weights"][name] > 0]
    assert set(audit["base_learners"]) == set(audit["refitted_learners"])
    assert len(audit["internal_cv"]["elastic_net"]) == 3
    assert {"selected_regularization", "iteration_limit_reached"} <= set(
        audit["internal_cv"]["elastic_net"][0])
    assert isinstance(audit["elastic_net_iteration_limit_reached"], bool)
    assert audit["seeds"] == {"elastic_net": 7, "gradient_boosting": 8, "random_forest": 9,
                              "internal_cv": 10}
    assert audit["settings"]["random_forest"] == sl.RANDOM_FOREST_SETTINGS
    assert audit["settings"]["n_jobs"] == 1
    json.dumps(audit)  # JSON-safe
    if binary:
        probability = model.predict_proba(x)[:, 1]
        assert ((probability > 0) & (probability < 1)).all()
    else:
        assert np.isfinite(model.predict(x)).all()
