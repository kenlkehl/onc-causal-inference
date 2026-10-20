"""Super-learner nuisance estimators and their Stage 2 integration."""

import io
import json
import pickle

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)

from oci.inference import plain_handoff_stage2_analysis as analysis
from oci.inference.plain_handoff_stage2 import plain_stage2_config_from_mapping
from oci.models import super_learner_nuisance as sl
from oci.models.elastic_net_nuisance import ElasticNetLogisticClassifier, ElasticNetRegressor
from tests.test_stage2_final_modeling import candidates as final_candidates
from tests.test_stage2_final_modeling import inputs as final_inputs
from tests.test_stage2_multi_model import small_policy
from tests.test_stage2_orthogonal_selection import _audits, _stored_nuisances

# Two internal folds keep the Stage 2 integration tests fast; the default is 5.
SUPER = {"name": "super_learner", "cv_folds": 2, "learners": list(sl.LEARNERS)}


def _binary(n=240, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 5))
    logit = 1.2 * (x[:, 0] < 0) + x[:, 1] ** 2 - 1.0
    return x, rng.binomial(1, 1 / (1 + np.exp(-logit)))


def _continuous(n=240, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 5))
    return x, np.where(x[:, 0] < 0, 1.0, -1.0) + 0.5 * x[:, 1] + rng.normal(size=n)


@pytest.mark.parametrize("binary", [True, False])
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


@pytest.mark.parametrize("binary", [True, False])
def test_deterministic_clone_safe_and_picklable(binary):
    x, y = _binary(seed=1) if binary else _continuous(seed=1)
    cls = sl.SuperLearnerClassifier if binary else sl.SuperLearnerRegressor
    model = cls(cv_folds=3, random_state=3).fit(x, y)

    def predict(m):
        return m.predict_proba(x) if binary else m.predict(x)

    np.testing.assert_array_equal(predict(model), predict(clone(model).fit(x, y)))
    np.testing.assert_array_equal(predict(model), predict(pickle.loads(pickle.dumps(model))))
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    buffer.seek(0)
    restored = joblib.load(buffer)
    np.testing.assert_array_equal(predict(model), predict(restored))
    assert restored.fit_audit() == model.fit_audit()
    assert clone(model).get_params() == model.get_params()


def test_single_learner_libraries_equal_that_learner():
    x, y = _binary(seed=2)
    seed = 11
    expected = {
        "elastic_net": ElasticNetLogisticClassifier(random_state=seed, n_jobs=1),
        "gradient_boosting": HistGradientBoostingClassifier(
            **sl.GRADIENT_BOOSTING_SETTINGS, early_stopping=True, random_state=seed + 1),
        "random_forest": RandomForestClassifier(
            **sl.RANDOM_FOREST_SETTINGS, n_jobs=1, random_state=seed + 2),
    }
    for name, reference in expected.items():
        model = sl.SuperLearnerClassifier(learners=(name,), random_state=seed).fit(x, y)
        assert model.fit_audit()["fit_mode"] == "single_learner"
        assert model.fit_audit()["stack_weights"] == {name: 1.0}
        np.testing.assert_array_equal(
            model.predict_proba(x)[:, 1], reference.fit(x, y).predict_proba(x)[:, 1])
    xr, yr = _continuous(seed=2)
    model = sl.SuperLearnerRegressor(learners=("elastic_net",), random_state=seed).fit(xr, yr)
    np.testing.assert_array_equal(
        model.predict(xr), ElasticNetRegressor(random_state=seed, n_jobs=1).fit(xr, yr).predict(xr))
    forest = sl.SuperLearnerRegressor(learners=("random_forest",), random_state=seed).fit(xr, yr)
    np.testing.assert_array_equal(forest.predict(xr), RandomForestRegressor(
        **sl.RANDOM_FOREST_SETTINGS, n_jobs=1, random_state=seed + 2).fit(xr, yr).predict(xr))


def test_small_arms_reduce_internal_folds_or_fall_back_to_the_elastic_net():
    x, _ = _binary(n=60, seed=3)
    y = np.r_[np.ones(3), np.zeros(57)].astype(int)
    reduced = sl.SuperLearnerClassifier(random_state=1).fit(x, y).fit_audit()
    assert reduced["requested_cv_folds"] == 5 and reduced["effective_cv_folds"] == 3
    assert reduced["cv_folds_reason"] == "smaller class has fewer rows than cv_folds"
    assert reduced["fit_mode"] == "stacked"
    lone = np.r_[np.ones(1), np.zeros(59)].astype(int)
    fallback = sl.SuperLearnerClassifier(random_state=1).fit(x, lone).fit_audit()
    assert fallback["fit_mode"] == "single_learner_fallback"
    assert fallback["stack_weights"]["elastic_net"] == 1.0
    assert fallback["effective_cv_folds"] == 0
    no_elastic = sl.SuperLearnerClassifier(
        learners=("gradient_boosting", "random_forest"), random_state=1).fit(x, lone)
    assert no_elastic.fit_audit()["stack_weights"] == {"gradient_boosting": 1.0,
                                                      "random_forest": 0.0}
    xr, yr = _continuous(n=7, seed=3)
    tiny = sl.SuperLearnerRegressor(random_state=1).fit(xr, yr).fit_audit()
    assert tiny["effective_cv_folds"] == 3
    assert tiny["cv_folds_reason"] == "fewer than two rows per internal fold"
    # Gradient boosting stops early only with enough rows (and class rows).
    assert {fold["early_stopping"] for fold in reduced["internal_cv"]["gradient_boosting"]} == {
        False}


def test_configuration_is_strict():
    assert sl.super_learner_config_from_mapping(None) == sl.SuperLearnerConfig()
    config = sl.super_learner_config_from_mapping(
        {"cv_folds": 3, "learners": ["random_forest", "elastic_net"]})
    assert config.learners == ("elastic_net", "random_forest")
    assert sl.learner_spec("super_learner", config) == {
        "name": "super_learner", "cv_folds": 3, "learners": ["elastic_net", "random_forest"]}
    assert sl.learner_spec("elastic_net", config) == {"name": "elastic_net"}
    for bad, message in (
        ({"folds": 3}, "unsupported fields"),
        ({"cv_folds": 1}, "cv_folds"),
        ({"learners": []}, "learners"),
        ({"learners": ["lasso"]}, "learners"),
        ({"learners": ["random_forest", "random_forest"]}, "learners"),
    ):
        with pytest.raises(ValueError, match=message):
            sl.super_learner_config_from_mapping(bad)
    with pytest.raises(ValueError, match="nuisance_learner"):
        sl.learner_spec("boosting", None)
    raw = {"endpoint": "http://stage2.test/v1", "model": "m"}
    stage2 = plain_stage2_config_from_mapping(raw, default_workers=1)
    assert stage2.nuisance_learner == "super_learner"
    assert stage2.nuisance_learner_spec() == {
        "name": "super_learner", "cv_folds": 5, "learners": list(sl.LEARNERS)}
    public = stage2.public_dict()
    assert public["super_learner"] == {"cv_folds": 5, "learners": list(sl.LEARNERS)}
    restored = plain_stage2_config_from_mapping(
        {**raw, "nuisance_learner": public["nuisance_learner"],
         "super_learner": public["super_learner"]}, default_workers=1)
    assert restored.nuisance_learner_spec() == stage2.nuisance_learner_spec()
    with pytest.raises(ValueError, match="nuisance_learner"):
        plain_stage2_config_from_mapping({**raw, "nuisance_learner": "forest"}, default_workers=1)
    with pytest.raises(ValueError, match="unsupported fields"):
        plain_stage2_config_from_mapping(
            {**raw, "super_learner": {"cv_folds": 3, "trees": 10}}, default_workers=1)


def test_elastic_net_learner_reproduces_the_elastic_net_nuisances(tmp_path):
    kw = final_inputs(True)
    default = analysis.estimate_outer_fold(
        **kw, **final_candidates(kw), estimator="linear_interactions",
        statistical_policy=small_policy(), output_dir=tmp_path / "default")
    explicit = analysis.estimate_outer_fold(
        **kw, **final_candidates(kw), estimator="linear_interactions",
        statistical_policy=small_policy(), output_dir=tmp_path / "explicit",
        nuisance_learner={"name": "elastic_net"})
    for name in ("predictions.csv", "nuisance_fit_predictions.csv"):
        pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "default" / name),
                                      pd.read_csv(tmp_path / "explicit" / name))
    assert explicit["nuisance_model_family"] == default["nuisance_model_family"] == "elastic_net"
    assert explicit["nuisance_learner"] == {"learner": "elastic_net",
                                            "configuration": {"name": "elastic_net"}}
    assert isinstance(analysis._fit_classifier(
        np.eye(4)[:, :2], np.array([0, 1, 0, 1]), seed=1), ElasticNetLogisticClassifier)


def test_final_estimation_with_the_super_learner_reuses_stored_stacks(tmp_path, monkeypatch):
    kw = final_inputs(True)
    store = tmp_path / "selection" / "nuisance_models"
    saved = _stored_nuisances(kw, store, learner=SUPER)
    manifest = json.loads((store / saved["model_store"]["root"].rsplit("/", 1)[-1]
                           / analysis.NUISANCE_MODEL_MANIFEST).read_text())
    assert manifest["identity"]["nuisance_learner"] == SUPER
    assert "../models/super_learner_nuisance.py" in manifest["identity"]["sources"]
    refit = analysis.estimate_outer_fold(
        **kw, **final_candidates(kw), estimator="linear_interactions",
        statistical_policy=small_policy(), output_dir=tmp_path / "refit",
        nuisance_learner=SUPER)

    def no_refit(*args, **kwargs):
        raise AssertionError("a stored stack was refitted")

    with monkeypatch.context() as patch:
        patch.setattr(analysis, "_fit_classifier", no_refit)
        patch.setattr(analysis, "_fit_regressor", no_refit)
        reused = analysis.estimate_outer_fold(
            **kw, **final_candidates(kw), estimator="linear_interactions",
            statistical_policy=small_policy(), output_dir=tmp_path / "reused",
            nuisance_model_store=store, nuisance_learner=SUPER)
    assert reused["nuisance_fit_audit_summary"]["model_sources"] == {"reused_stored_model": 9}
    for name in ("predictions.csv", "nuisance_fit_predictions.csv"):
        pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "reused" / name),
                                      pd.read_csv(tmp_path / "refit" / name))
    fitted = pd.read_csv(tmp_path / "reused" / "nuisance_fit_predictions.csv",
                         float_precision="round_trip")
    for column, values in zip(("propensity", "mu0", "mu1"), saved["recalibrated"]["fit"]):
        np.testing.assert_array_equal(fitted[column], values)
    stored = _audits(tmp_path / "selection" / "testing_audits")
    estimated = _audits(next((tmp_path / "reused" / "nuisance_fit_audits").iterdir()))
    assert {name: audit["fit_audit"] for name, audit in stored.items()} == {
        name: audit["fit_audit"] for name, audit in estimated.items()}
    assert all("stack_weights" in audit["fit_audit"] for audit in estimated.values())
    summary = refit["nuisance_learner"]
    assert summary["learner"] == "super_learner"
    assert set(summary["nuisances"]) == {"propensity", "mu0", "mu1"}
    for item in summary["nuisances"].values():
        assert sum(item["mean_stack_weights"].values()) == pytest.approx(1.0)
    assert refit["nuisance_model_family"] == "super_learner"
    assert "super_learner_nuisances" in refit["methodology_notes"]
    # An elastic-net store does not serve a super-learner estimation.
    other = tmp_path / "other" / "nuisance_models"
    _stored_nuisances(kw, other)
    absent = analysis.estimate_outer_fold(
        **kw, **final_candidates(kw), estimator="linear_interactions",
        statistical_policy=small_policy(), output_dir=tmp_path / "absent",
        nuisance_model_store=other, nuisance_learner=SUPER)
    assert absent["nuisance_model_store"]["status"] == "absent"


def test_learner_changes_require_reselection_only_under_orthogonal_tests(tmp_path):
    from dataclasses import replace
    from pathlib import Path
    from types import SimpleNamespace

    from oci.inference.research_all_evidence_workflow import (
        _stage2_reselection_policy_fingerprint,
        compile_config,
    )
    from oci.inference.stage2_multi_model_config import ModifierSelectionConfig
    from oci.inference.stage2_preflight import validate_selection_resume

    path = Path(__file__).resolve().parents[1] / "example_configs/research_all_evidence_multi_model.json"
    config = compile_config(json.loads(path.read_text()), config_dir=path.parent)
    root = tmp_path / "stage2"
    (root / "outer_001" / "selection").mkdir(parents=True)

    def check(stage2):
        (root / "config.json").write_text(json.dumps(stage2.public_dict()))
        validate_selection_resume(SimpleNamespace(stage2=stage2, output_dir=tmp_path))
        changed = replace(stage2, nuisance_learner="elastic_net")
        same = _stage2_reselection_policy_fingerprint(replace(config, stage2=stage2)) == (
            _stage2_reselection_policy_fingerprint(replace(config, stage2=changed)))
        try:
            validate_selection_resume(SimpleNamespace(stage2=changed, output_dir=tmp_path))
        except RuntimeError as error:
            return same, str(error)
        return same, None

    assert config.stage2.nuisance_learner == "super_learner"
    same, error = check(config.stage2)
    assert not same and "nuisance_learner or super_learner changed" in error
    selection = config.stage2.statistical_selection
    balanced = replace(config.stage2, statistical_selection=replace(
        selection, multi_model=replace(
            selection.multi_model,
            modifier_selection=ModifierSelectionConfig(rule="balanced_evidence"))))
    assert check(balanced) == (True, None)
