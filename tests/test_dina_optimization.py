"""Convergence, penalty selection, and failure isolation for DINA."""

from dataclasses import replace
import json
import pickle

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.special import expit

from oci.models import dina
from oci.inference import stage2_dina as adapter
from oci.inference import stage2_multi_model_selection as numerical
from oci.inference.stage2_elastic_net_selection import Stage2ElasticNetSelectionConfig
from tests.test_dina import simulated


def _kkt_residual(x, y, t, a, nu, model, alpha, ratio):
    """Independent block-by-block optimality calculation for the convex objective."""
    u = t - a
    residual = u * (expit(nu + u * model.predict(x)) - y)
    gradient = x.T @ residual / len(y) + alpha * (1 - ratio) * model.coefficients
    errors = [abs(residual.mean())]
    for group in set(model.group_ids):
        ix = np.asarray(model.group_ids) == group
        coefficients = model.coefficients[ix]
        penalty = alpha * ratio * np.sqrt(ix.sum())
        norm = np.linalg.norm(coefficients)
        errors.append(np.linalg.norm(gradient[ix] + penalty * coefficients / norm)
                      if norm else max(0, np.linalg.norm(gradient[ix]) - penalty))
    return max(errors)


@pytest.mark.parametrize("sparse_input", [False, True])
def test_flat_curvature_converges_with_noncontiguous_groups_and_certified_optimality(sparse_input):
    rng = np.random.default_rng(62)
    x = rng.normal(size=(180, 8))
    t = rng.binomial(1, 0.5, len(x))
    a = np.where(t, 0.98, 0.02)
    nu = np.zeros(len(x))
    y = rng.binomial(1, expit((t - a) * (x[:, 0] * 25 - x[:, 4] * 15)))
    groups = ["shared", "b", "shared", "c", "b", "d", "c", "shared"]
    alpha, ratio, tolerance = 2e-5, 0.6, 1e-7
    model = dina.fit(sparse.csr_matrix(x) if sparse_input else x, y, t, a, nu,
                     groups=groups, regularization=alpha, group_ratio=ratio,
                     tolerance=tolerance, max_iter=1000)
    assert model.converged and model.iterations < 1000
    residual = _kkt_residual(x, y, t, a, nu, model, alpha, ratio)
    assert residual <= tolerance
    assert model.diagnostics["stationarity_residual"] == pytest.approx(residual, abs=1e-12)
    assert model.diagnostics["step_size"] > 1


def test_warm_start_preserves_solution_and_rejects_incompatible_models():
    x, t, y, n, _ = simulated(500)
    args = dict(groups=["c", "c", "noise"], regularization=0.003, group_ratio=0.8)
    initial = dina.fit(x, y, t, n["a"], n["nu"], **{**args, "regularization": 0.004})
    saved = initial.coefficients.copy()
    warm = dina.fit(x, y, t, n["a"], n["nu"], **args, initial=initial)
    cold = dina.fit(x, y, t, n["a"], n["nu"], **args)
    np.testing.assert_allclose(warm.predict(x), cold.predict(x), atol=1e-4)
    np.testing.assert_array_equal(saved, initial.coefficients)
    assert warm.diagnostics["warm_started"] and not cold.diagnostics["warm_started"]
    for bad in [replace(initial, group_ids=("noise", "c", "c")),
                replace(initial, converged=False), replace(initial, intercept=np.nan)]:
        with pytest.raises(ValueError, match="warm start"):
            dina.fit(x, y, t, n["a"], n["nu"], **args, initial=bad)


def test_iteration_failure_has_serializable_numerical_diagnostics():
    x, t, y, n, _ = simulated(300)
    with pytest.raises(dina.DinaConvergenceError) as caught:
        dina.fit(x, y, t, n["a"], n["nu"], regularization=0.001, group_ratio=0.8, max_iter=1)
    error = pickle.loads(pickle.dumps(caught.value))
    audit = error.diagnostics
    assert not audit["converged"] and audit["reason"] == "iteration_limit"
    assert audit["iterations"] == audit["max_iter"] == 1
    assert audit["stationarity_residual"] > audit["tolerance"]
    assert audit["rows"] == 300 and audit["columns"] == 3
    assert "regularization=0.001" in str(error)
    json.dumps(audit, allow_nan=False)


@pytest.fixture
def adapter_inputs():
    x, t, y, n, _ = simulated(120)
    frame = pd.DataFrame({"_oci_row_id": range(120), "x": x[:, 0], "z": x[:, 2]})
    definitions = [{"feature_id": name, "name": name, "value_type": "continuous"} for name in ["x", "z"]]
    policy = Stage2ElasticNetSelectionConfig(internal_cv_folds=2, regularization_grid_size=3,
        minimum_log10_alpha=-3, maximum_log10_alpha=-1)
    policy = replace(policy, multi_model=replace(policy.multi_model, regularization_grid_size=3))
    return dict(train=frame.iloc[:80], valid=frame.iloc[80:], definitions=definitions,
        modifier_ids=["x", "z"], treatment=t[:80], outcome=y[:80], policy=policy, seed=51,
        nuisance=({k: v[:80] for k, v in n.items()}, {k: v[80:] for k, v in n.items()}))


def _failure():
    return dina.DinaConvergenceError("test iteration limit", {"converged": False,
        "reason": "iteration_limit", "iterations": 1, "stationarity_residual": 0.1})


@pytest.mark.parametrize("one_se", [False, True])
def test_failed_penalty_is_excluded_and_warm_starts_stay_within_cv_partition(monkeypatch, adapter_inputs, one_se):
    original = dina.fit
    calls = []
    models = []

    def controlled(*args, **kwargs):
        calls.append(kwargs.get("initial"))
        if len(calls) == 2:  # Middle penalty fails in just one CV partition.
            raise _failure()
        model = original(*args, **kwargs)
        models.append(model)
        return model

    monkeypatch.setattr(dina, "fit", controlled)
    inputs = {**adapter_inputs, "policy": replace(adapter_inputs["policy"], modifier_one_standard_error_rule=one_se)}
    result = adapter.fit_effect(**inputs)
    assert len(calls) == 7  # Three penalties per CV fold, then a fresh final refit.
    assert calls[0] is None and calls[3] is None and calls[6] is None
    assert calls[1] is models[0] and calls[2] is models[0]
    assert calls[4] is models[2] and calls[5] is models[3]
    assert result["audit"]["regularization"] != pytest.approx(0.01)
    failed = [d for d in result["audit"]["cv_fits"] if not d["converged"]]
    assert len(failed) == 1 and failed[0]["cv_fold"] == 1 and failed[0]["validation_loss"] is None
    assert np.isfinite(result["tau"]).all()
    json.dumps(result["audit"], allow_nan=False)


@pytest.mark.parametrize("final_only", [False, True])
def test_exhausted_or_final_fit_is_not_estimable_with_diagnostics(monkeypatch, adapter_inputs, final_only):
    original = dina.fit

    def fail(x, *args, **kwargs):
        if not final_only or len(x) == len(adapter_inputs["train"]):
            raise _failure()
        return original(x, *args, **kwargs)

    monkeypatch.setattr(dina, "fit", fail)
    with pytest.raises(numerical.NotEstimable) as caught:
        adapter.fit_effect(**adapter_inputs)
    diagnostics = caught.value.diagnostics
    assert len(diagnostics["cv_fits"]) == 6
    assert ("final_fit" in diagnostics) == final_only
    assert "final_dina_fit" in str(caught.value) if final_only else "every_cv_fold" in str(caught.value)
    json.dumps(diagnostics, allow_nan=False)


def test_invalid_data_errors_are_not_treated_as_convergence_failures(monkeypatch, adapter_inputs):
    def invalid(*args, **kwargs):
        raise ValueError("DINA design must be finite")

    monkeypatch.setattr(dina, "fit", invalid)
    with pytest.raises(ValueError, match="design must be finite") as caught:
        adapter.fit_effect(**adapter_inputs)
    assert not isinstance(caught.value, numerical.NotEstimable)


def test_failed_candidate_does_not_remove_other_candidate_evidence(monkeypatch, adapter_inputs):
    original = adapter.fit_effect

    def controlled(**kwargs):
        if kwargs["modifier_ids"] == ["x"]:
            raise adapter._not_estimable("final_dina_fit_did_not_converge", {"final_fit": _failure().diagnostics})
        return original(**kwargs)

    monkeypatch.setattr(adapter, "fit_effect", controlled)
    args = adapter_inputs
    result = adapter.candidate_evidence(args["train"], args["valid"], args["definitions"],
        args["treatment"], args["outcome"], np.zeros(40), np.zeros(40), policy=args["policy"],
        seed=args["seed"], nuisance=args["nuisance"])
    assert [r["status"] for r in result["records"]] == ["not_evaluable", "ok"]
    assert result["records"][0]["selected"] is None
    assert result["audit"]["failed_candidate_fits"]["x"]["diagnostics"]["final_fit"]["iterations"] == 1


def test_failed_dina_cell_keeps_diagnostics_and_other_model_evidence(monkeypatch, tmp_path):
    from tests.test_stage2_multi_model import sample_inputs
    from oci.inference.stage2_multi_model_config import Stage2MultiModelConfig

    args = sample_inputs()
    args["outcome_type"] = "binary"
    args["dataset"]["outcome"] = (args["dataset"]["outcome"] > 0).astype(int)
    args["policy"] = replace(args["policy"], multi_model=replace(args["policy"].multi_model, repeats=1))
    monkeypatch.setattr(Stage2MultiModelConfig, "active_families",
                        lambda self, outcome: ("univariable", "orthogonal_linear"))

    def nuisance(train, valid, *args, **kwargs):
        return {f"{partition}_{name}": [value] * len(frame)
                for partition, frame in [("training", train), ("validation", valid)]
                for name, value in [("propensity", 0.5), ("outcome", 0.4), ("mu0", 0.3), ("mu1", 0.5)]}

    def other_model(train, definitions, *args, **kwargs):
        return {"records": [numerical._record(d["feature_id"], "outcome", selected=True, score=0.1)
                            for d in definitions], "audit": {"status": "ok"}}

    def fail(*args, **kwargs):
        raise adapter._not_estimable("no_dina_regularization_converged_in_every_cv_fold",
                                     {"cv_fits": [_failure().diagnostics]})

    monkeypatch.setattr(numerical, "_nuisances", nuisance)
    monkeypatch.setattr(numerical, "_univariable", other_model)
    monkeypatch.setattr(adapter, "joint_evidence", fail)
    _, report, _, _ = numerical.select_stage2_features_multi_model(**args, checkpoint_dir=tmp_path)
    assert report["evaluable_cells_by_family"] == {"univariable": 2, "orthogonal_linear": 0}
    paths = list(tmp_path.glob("*/fold_*/repeat_*/orthogonal_linear_000.json"))
    assert len(paths) == 2
    for path in paths:
        cell = json.loads(path.read_text())["result"]
        assert cell["audit"]["optimizer_diagnostics"]["cv_fits"][0]["reason"] == "iteration_limit"
        assert all(r["status"] == "not_evaluable" and r["selected"] is None for r in cell["records"])
