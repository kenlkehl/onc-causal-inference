"""Scientific and integration contracts for binary DINA."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.special import expit, logit

from oci.models import dina


def test_counterfactual_reconstruction_and_nuisance_orthogonality():
    e = np.array([0.2, 0.5, 0.8])
    p0, p1 = np.array([0.05, 0.3, 0.8]), np.array([0.2, 0.6, 0.9])
    delta = logit(p1) - logit(p0)
    n = dina.nuisances(e, p0, p1)
    assert not np.allclose(n["a"], e)
    a, b = dina.counterfactuals(delta, n["a"], n["nu"])
    np.testing.assert_allclose(a, p0)
    np.testing.assert_allclose(b, p1)

    def expected_score(pred):
        return (1 - e) * (-pred["a"]) * (p0 - expit(pred["nu"] - pred["a"] * delta)) + e * (
            1 - pred["a"]
        ) * (p1 - expit(pred["nu"] + (1 - pred["a"]) * delta))

    # At the truth the score is orthogonal to each nuisance, not just propensity.
    for index in range(3):
        args = [e.copy(), p0.copy(), p1.copy()]
        args[index] += 1e-5
        plus = expected_score(dina.nuisances(*args))
        args[index] -= 2e-5
        minus = expected_score(dina.nuisances(*args))
        np.testing.assert_allclose((plus - minus) / 2e-5, 0, atol=1e-7)


def test_score_is_likelihood_derivative_and_fisher_centering():
    y, t = np.array([0, 1, 0, 1.0]), np.array([0, 0, 1, 1.0])
    n = dina.nuisances(
        np.full(4, 0.4), np.array([0.1, 0.2, 0.3, 0.4]), np.array([0.3, 0.4, 0.5, 0.6])
    )
    delta = 0.7
    score, weights, constant = dina.score(y, t, n["a"], n["nu"], delta=delta)
    h = 1e-6
    derivative = (
        dina.loss(y, t, n["a"], n["nu"], delta + h) - dina.loss(y, t, n["a"], n["nu"], delta - h)
    ) / (2 * h)
    np.testing.assert_allclose(score, -derivative, rtol=1e-7)
    x = np.arange(4.0)
    assert abs(np.sum(weights * (x - np.average(x, weights=weights)))) < 1e-12
    assert constant == delta and (weights > 0).all()


def simulated(n=1600, seed=85):
    rng = np.random.default_rng(seed)
    levels = rng.integers(0, 3, n)
    x = np.column_stack([levels == 1, levels == 2, rng.normal(size=n)]) * 1.0
    t = rng.binomial(1, 0.5, n)
    delta = 0.2 + 2 * x[:, 0] - 1.5 * x[:, 1]
    p0 = expit(-0.9 + 0.3 * x[:, 2])
    p1 = expit(logit(p0) + delta)
    y = rng.binomial(1, np.where(t, p1, p0))
    return x, t, y, dina.nuisances(np.full(n, 0.5), p0, p1), delta


def test_group_solver_sparse_dense_and_whole_categorical_selection():
    x, t, y, n, true = simulated()
    args = dict(groups=["category", "category", "noise"], regularization=0.003, group_ratio=0.8)
    model = dina.fit(x, y, t, n["a"], n["nu"], **args)
    other = dina.fit(sparse.csr_matrix(x), y, t, n["a"], n["nu"], **args)
    np.testing.assert_allclose(model.predict(x), other.predict(x), atol=1e-5)
    assert "category" in model.selected_groups()
    assert np.corrcoef(model.predict(x), true)[0, 1] > 0.98
    null = dina.fit(
        x, y, t, n["a"], n["nu"], groups=args["groups"], regularization=10, group_ratio=1
    )
    assert null.selected_groups() == []
    np.testing.assert_array_equal(null.coefficients, np.zeros(3))
    assert abs(null.intercept) > 0
    p0, p1 = dina.counterfactuals(model.predict(x), n["a"], n["nu"])
    assert (abs(p1 - p0) <= 1).all()


def test_dina_is_additional_binary_architecture_and_evidence_family():
    from oci.inference.stage2_multi_model_config import Stage2MultiModelConfig
    from oci.inference.stage2_effect_estimators import effect_model_family

    cfg = Stage2MultiModelConfig()
    assert {"causal_forest", "dina"} <= set(cfg.active_families("binary"))
    assert "dina" not in cfg.active_families("continuous")
    assert set(cfg.modifier_count.estimators) == {"causal_forest", "linear_interactions", "dina"}
    assert effect_model_family("dina", "binary") == "grouped_bernoulli_dina"
    with pytest.raises(ValueError):
        effect_model_family("dina", "continuous")


def test_stage2_adapter_uses_all_columns_of_categorical_feature():
    from oci.inference.stage2_dina import fit_effect, joint_evidence
    from oci.inference.stage2_elastic_net_selection import Stage2ElasticNetSelectionConfig

    x, t, y, n, true = simulated(400)
    frame = pd.DataFrame(
        {
            "_oci_row_id": np.arange(len(t)),
            "category": np.where(x[:, 0], "one", np.where(x[:, 1], "two", "zero")),
        }
    )
    defs = [
        {
            "feature_id": "category",
            "name": "category",
            "value_type": "categorical",
            "categories_or_unit": ["zero", "one", "two"],
        }
    ]
    p = Stage2ElasticNetSelectionConfig(
        internal_cv_folds=2, maximum_log10_alpha=-1, minimum_log10_alpha=-3
    )
    p = replace(
        p, multi_model=replace(p.multi_model, regularization_grid_size=3, permutation_repeats=2)
    )
    nt, nv = ({k: v[:300] for k, v in n.items()}, {k: v[300:] for k, v in n.items()})
    args = dict(
        train=frame.iloc[:300],
        valid=frame.iloc[300:],
        definitions=defs,
        treatment=t[:300],
        outcome=y[:300],
        modifier_ids=["category"],
        policy=p,
        seed=82,
        nuisance=(nt, nv),
    )
    f = fit_effect(**args)
    assert set(f["design"].column_feature_ids) == {"category"}
    assert len(f["design"].column_feature_ids) >= 2
    np.testing.assert_allclose(f["tau"], f["mu1"] - f["mu0"])
    assert f["audit"]["effect_scale"] == "conditional_log_odds_ratio"
    result = joint_evidence(
        frame.iloc[:300],
        frame.iloc[300:],
        defs,
        t[:300],
        y[:300],
        t[300:],
        y[300:],
        policy=p,
        seed=82,
        nuisance=(nt, nv),
        permutation=True,
    )
    assert len(result["records"]) == 1
    assert result["records"][0]["score"] > 0


def test_sparse_effect_fold_excludes_heldout_labels():
    from oci.inference.stage1_dina import crossfit_effect, splits

    rng = np.random.default_rng(2)
    n = 60
    t = np.tile([0, 1], n // 2)
    x = rng.integers(0, 3, n)
    y = rng.binomial(1, expit(-1 + t * (x - 1)), n)
    texts = ["patient has group " + ["alpha", "beta", "gamma"][i] for i in x]
    before = crossfit_effect(texts, t, y, folds=2, seed=25)
    hold = splits(t, 2, 25)[0][1]
    modified = y.copy()
    modified[hold] = 1 - y[hold]
    after = crossfit_effect(texts, t, modified, folds=2, seed=25)
    np.testing.assert_array_equal(before["tau"][hold], after["tau"][hold])
    np.testing.assert_array_equal(before["delta"][hold], after["delta"][hold])


def test_heldout_topic_score_uses_training_constant_and_native_weights():
    from oci.inference.tfidf_topic_score_selection import _bank_contribution

    x, t, y, n, _ = simulated(100)
    nt, nv = ({k: v[:60] for k, v in n.items()}, {k: v[60:] for k, v in n.items()})
    args = dict(
        bank="effect",
        fit_treatment=t[:60],
        fit_outcome=y[:60],
        heldout_treatment=t[60:],
        heldout_outcome=y[60:],
        fit_propensity=nt["e"],
        fit_outcome_prediction=nt["mu0"],
        heldout_propensity=nv["e"],
        heldout_outcome_prediction=nv["mu0"],
        dina_nuisance=(nt, nv),
    )
    contribution, weights, definition = _bank_contribution(**args)
    changed = _bank_contribution(**(args | {"heldout_outcome": 1 - y[60:]}))
    assert definition["fit_baseline"] == changed[2]["fit_baseline"]
    np.testing.assert_array_equal(weights, changed[1])
    assert definition["effect_scale"] == "conditional_log_odds_ratio"
    assert not np.allclose(contribution, changed[0])


def test_neural_objective_and_counterfactuals_match_numpy():
    import torch
    from types import SimpleNamespace
    from oci.inference.stage1_dina import neural_predictions

    x, t, y, n, _ = simulated(20)
    delta = np.linspace(-1, 1, len(y))
    td = torch.tensor(delta, dtype=torch.float64, requires_grad=True)
    value = dina.torch_loss(
        torch.tensor(y, dtype=torch.float64),
        torch.tensor(t, dtype=torch.float64),
        torch.tensor(n["a"]),
        torch.tensor(n["nu"]),
        td,
        reduction="none",
    )
    np.testing.assert_allclose(value.detach().numpy(), dina.loss(y, t, n["a"], n["nu"], delta))
    value.sum().backward()
    expected = (t - n["a"]) * (expit(n["nu"] + (t - n["a"]) * delta) - y)
    np.testing.assert_allclose(td.grad.numpy(), expected)
    runner = SimpleNamespace(config=SimpleNamespace(text_column="text"))
    model = SimpleNamespace(dina_nuisance=SimpleNamespace(predict=lambda _: n))
    pred = neural_predictions(runner, model, pd.DataFrame({"text": ["note"] * 20}), delta)
    np.testing.assert_allclose(pred["tau"], pred["mu1"] - pred["mu0"])
    assert not np.allclose(pred["mu0"], n["mu0"])


def test_missing_arm_nuisance_does_not_block_other_evidence():
    from oci.inference import stage2_multi_model_selection as numerical
    from oci.inference.stage2_dina import unpack_nuisances
    from oci.inference.stage2_elastic_net_selection import Stage2ElasticNetSelectionConfig

    frame = pd.DataFrame({"_oci_row_id": [0, 1, 2]})
    value = numerical._nuisances(
        frame,
        frame.iloc[:1],
        [],
        np.array([0, 1, 0]),
        np.array([0, 1, 1]),
        binary=True,
        policy=Stage2ElasticNetSelectionConfig(internal_cv_folds=2),
        seed=1,
    )
    assert np.isfinite(value["training_propensity"]).all()
    assert np.isfinite(value["validation_outcome"]).all()
    assert value["dina_nuisance_status"] == "not_estimable"
    with pytest.raises(numerical.NotEstimable, match="missing_treatment_arm"):
        unpack_nuisances(value)
