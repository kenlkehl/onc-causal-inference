"""Numerical and isolation checks needed for this experimental intervention."""
import copy
import numpy as np
import pytest
import torch
from common import POLICY, assert_scope
from perturb import activation, optimize, project_cap, random_at_distance
from probes import Probe, loss


def test_activation_and_directional_gradient_match_native_and_finite_difference():
    from oci.inference.neural_cohort_witness import _torch_soft_activations
    rng = np.random.default_rng(10)
    z = torch.tensor(rng.normal(size=(9, 4, 6)), dtype=torch.float64)
    mask = torch.ones((9, 4), dtype=torch.bool)
    mask[:3, 2:] = False
    q = torch.tensor(rng.normal(size=(2, 6)), dtype=torch.float64, requires_grad=True)
    direction = torch.tensor(rng.normal(size=(2, 6)), dtype=torch.float64)
    actual = activation(z, mask, q, 0.05)
    expected = _torch_soft_activations(z, mask, q, temperature=0.05)
    torch.testing.assert_close(actual, expected)
    objective = actual.square().mean()
    gradient = torch.autograd.grad(objective, q)[0]
    step = 1e-5
    finite = (activation(z, mask, q + step * direction, 0.05).square().mean()
              - activation(z, mask, q - step * direction, 0.05).square().mean()) / (2 * step)
    assert float((gradient * direction).sum()) == pytest.approx(float(finite.detach()), rel=1e-6)


def test_spherical_constraints_and_equal_size_controls():
    rng = np.random.default_rng(8)
    original = rng.normal(size=(40, 32))
    original /= np.linalg.norm(original, axis=1, keepdims=True)
    radius = np.linspace(0.001, 0.3, 40)
    actual = project_cap(torch.tensor(original + rng.normal(size=original.shape)),
                         torch.tensor(original), torch.tensor(radius)).numpy()
    np.testing.assert_allclose(np.linalg.norm(actual, axis=1), 1, atol=1e-10)
    assert np.all(np.linalg.norm(actual - original, axis=1) <= radius + 1e-10)
    for i in range(40):
        control = random_at_distance(original[i], radius[i], i)
        assert np.linalg.norm(control - original[i]) == pytest.approx(radius[i], abs=1e-7)


def test_real_optimizer_reduces_signal_without_collapsing_variance():
    torch.set_num_threads(1)
    rng = np.random.default_rng(44)
    z = rng.normal(size=(240, 1, 8)).astype(np.float32)
    z /= np.linalg.norm(z, axis=2, keepdims=True)
    q = np.zeros((1, 8), dtype=np.float32)
    q[0, 0] = 1
    c = z[:, 0, 0].copy()
    policy = copy.deepcopy(POLICY)
    policy.update(radii=[1.2], epochs=45, learning_rate=0.05)
    chosen, _ = optimize(torch.tensor(z), torch.ones((240, 1), dtype=torch.bool), q,
                         c, np.ones(240), policy, seed=9)
    before = z[:, 0] @ q[0]
    after = z[:, 0] @ chosen[0]["vector"]
    assert abs(np.cov(after, c, ddof=0)[0, 1]) < 0.6 * abs(np.cov(before, c, ddof=0)[0, 1])
    assert chosen[0]["distance"] <= 1.2 + 1e-6
    assert 0.5 <= np.std(after) / np.std(before) <= 2.0


def test_effect_probe_beats_constant_on_independent_known_heterogeneity():
    rng = np.random.default_rng(6)
    x = rng.normal(size=(700, 1))
    t = rng.binomial(1, 0.5, len(x))
    u = t - 0.5
    v = u * (0.1 + 0.5 * x[:, 0]) + rng.normal(0, 0.03, len(x))
    y = (v > 0).astype(int)
    train, valid = np.arange(500), np.arange(500, 700)
    probe = Probe("linear", "effect", POLICY).fit(x[train], t[train], y[train], u[train], v[train])
    observed = loss("effect", probe.predict(x[valid]), t[valid], y[valid], u[valid], v[valid])
    baseline = loss("effect", np.full(200, np.dot(u[train], v[train]) / np.dot(u[train], u[train])),
                    t[valid], y[valid], u[valid], v[valid])
    assert observed < baseline / 10


def test_scope_refuses_outer_leakage_overlap_and_missing_training_rows():
    outer = {"fit_row_ids": [0, 1, 2, 3], "heldout_row_ids": [4, 5]}
    assert_scope([0, 1, 2], [3], outer)
    for train, valid in [([0, 1, 4], [3]), ([0, 1, 2], [2, 3]), ([0, 1], [3])]:
        with pytest.raises(AssertionError):
            assert_scope(train, valid, outer)


def test_llm_contract_requires_literal_training_evidence_and_missingness_policy():
    from interpret import validate_response
    packet = {"evidence": [{"evidence_id": "training_only_1", "text": "Pretreatment serum sodium was 132 mmol/L."}]}
    value = {"summary": "A documented numeric laboratory value.", "limitations": ["Role is uncertain."],
             "candidates": [{"name": "Serum sodium", "variable_type": "continuous", "categories": [],
                             "description": "Sodium before treatment", "measurement_definition": "mmol/L",
                             "extraction_instruction": "Extract the last documented pretreatment value.",
                             "missing_value_policy": "null_if_not_documented", "role_hypotheses": ["effect_modifier"],
                             "citations": [{"evidence_id": "training_only_1", "quote": "serum sodium was 132 mmol/L"}],
                             "uncertainty": "Effect modification is a hypothesis."}]}
    assert validate_response(value, packet) == value
    invalid = copy.deepcopy(value)
    invalid["candidates"][0]["citations"][0]["evidence_id"] = "outer_test_1"
    with pytest.raises(ValueError, match="Unknown evidence_id"):
        validate_response(invalid, packet)
    invalid = copy.deepcopy(value)
    invalid["candidates"][0]["citations"][0]["quote"] = "Sodium was always normal."
    with pytest.raises(ValueError, match="does not occur"):
        validate_response(invalid, packet)
    invalid = copy.deepcopy(value)
    invalid["candidates"][0]["missing_value_policy"] = "assume_normal"
    with pytest.raises(ValueError, match="null_if_not_documented"):
        validate_response(invalid, packet)
