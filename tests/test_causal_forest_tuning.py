import logging

import numpy as np

import oci.models.causal_forest_head as causal_forest_head


class FakeNuisanceModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeCausalForestDML:
    instances = []
    fail_tune = False

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        self.__class__.instances.append(self)

    def tune(self, **kwargs):
        self.calls.append(("tune", kwargs))
        if self.fail_tune:
            raise RuntimeError("tuning failed")
        return self

    def fit(self, **kwargs):
        self.calls.append(("fit", kwargs))
        return self


def _patch_econml_dependencies(monkeypatch):
    FakeCausalForestDML.instances = []
    FakeCausalForestDML.fail_tune = False
    monkeypatch.setattr(causal_forest_head, "ECONML_AVAILABLE", True)
    monkeypatch.setattr(causal_forest_head, "CausalForestDML", FakeCausalForestDML)
    monkeypatch.setattr(causal_forest_head, "RandomForestClassifier", FakeNuisanceModel, raising=False)
    monkeypatch.setattr(causal_forest_head, "RandomForestRegressor", FakeNuisanceModel, raising=False)


def test_causal_forest_head_tunes_before_fit(monkeypatch):
    _patch_econml_dependencies(monkeypatch)
    X = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    T = np.array([[0], [1], [0], [1]])
    Y = np.array([[0], [1], [1], [0]])

    head = causal_forest_head.CausalForestHead(n_estimators=8, min_samples_leaf=2)
    head.fit(X=X, T=T, Y=Y)

    model = head.model
    assert model.calls[0][0] == "tune"
    assert model.calls[0][1]["params"] == "auto"
    assert model.calls[1][0] == "fit"
    np.testing.assert_array_equal(model.calls[0][1]["T"], T.flatten())
    np.testing.assert_array_equal(model.calls[0][1]["Y"], Y.flatten())
    np.testing.assert_array_equal(model.calls[0][1]["X"], X)


def test_causal_forest_head_warns_rebuilds_and_fits_when_tuning_fails(monkeypatch, caplog):
    _patch_econml_dependencies(monkeypatch)
    FakeCausalForestDML.fail_tune = True
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    T = np.array([0, 1, 0, 1])
    Y = np.array([0, 1, 1, 0])

    head = causal_forest_head.CausalForestHead(n_estimators=8, min_samples_leaf=2)

    with caplog.at_level(logging.WARNING):
        head.fit(X=X, T=T, Y=Y)

    first_model, second_model = FakeCausalForestDML.instances
    assert first_model.calls[0][0] == "tune"
    assert second_model is head.model
    assert second_model.calls[0][0] == "fit"
    assert "CausalForestDML hyperparameter tuning failed" in caplog.text


def test_tune_causal_forest_model_returns_false_on_failure(caplog):
    FakeCausalForestDML.instances = []
    FakeCausalForestDML.fail_tune = True
    model = FakeCausalForestDML()

    with caplog.at_level(logging.WARNING):
        tuned = causal_forest_head.tune_causal_forest_model(
            model,
            Y=np.array([0, 1]),
            T=np.array([0, 1]),
            X=np.array([[0.0], [1.0]]),
        )

    assert tuned is False
    assert model.calls[0][0] == "tune"
    assert "fitting with configured hyperparameters" in caplog.text
