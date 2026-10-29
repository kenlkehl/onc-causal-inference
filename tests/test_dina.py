"""Scientific and integration contracts for binary DINA."""


import numpy as np
import pandas as pd
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
