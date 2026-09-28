"""Patient boundaries, batch evidence, and its path into clinical review."""
from dataclasses import replace
import json
import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from oci.inference import stage2_matched_batch as batch
from oci.inference.stage2_multi_model_config import MatchedBatchConfig, multi_model_config_from_mapping
from oci.inference.stage2_multi_model_selection import aggregate_evidence, _matched_batch_enabled
from oci.inference.stage2_multi_model_adjudication import build_multi_model_role_evidence
from oci.inference.stage2_role_adjudication import Stage2RoleAdjudicationConfig
from oci.inference import stage2_clinical_prompts as prompts
from oci.inference.stage2_modifier_concepts import _concept_input
from tests.test_stage2_multi_model import feature, small_policy


def policy(**changes):
    cfg = replace(MatchedBatchConfig(enabled=True, train_passes=15, validation_passes=10), **changes)
    return replace(small_policy(), multi_model=replace(small_policy().multi_model, matched_batch=cfg))


def cohort():
    rng = np.random.default_rng(421)
    n, train_n = 1200, 900
    x = rng.normal(size=(n, 2))
    t = np.tile([0, 1], n // 2)
    y = (t - .5) * 2 * x[:, 0] + rng.normal(scale=.05, size=n)
    frame = pd.DataFrame(dict(modifier=x[:, 0], noise=x[:, 1], constant=1.))
    frame.insert(0, "_oci_row_id", np.arange(n))
    defs = [feature(k) for k in ("modifier", "noise", "constant")]
    return dict(train=frame.iloc[:train_n], valid=frame.iloc[train_n:], definitions=defs,
        t=t[:train_n], y=y[:train_n], tv=t[train_n:], yv=y[train_n:],
        nuisance={"e": np.full(train_n, .5), "m": np.zeros(train_n),
                  "ev": np.full(n - train_n, .5), "mv": np.zeros(n - train_n)},
        policy=policy(), seed=9)


def test_batches_balance_references_are_training_only_and_shuffle_without_replacement():
    rng = np.random.default_rng(31)
    n = 320
    t, y, q = np.tile([0, 1], n // 2), rng.normal(size=n), rng.uniform(size=(n, 2))
    fit, hold = np.arange(240), np.arange(240, n)
    cfg = replace(MatchedBatchConfig(), max_smd=10, min_reference_arm=4)
    sampler = batch.MatchedBatches(t, y, q, fit, fit, 9, cfg)
    one, two = list(sampler.one_pass()), list(sampler.one_pass())
    rows = np.concatenate([r for r, *_ in one])
    assert len(np.unique(rows)) == len(rows)
    assert any(not np.array_equal(a[0], b[0]) for a, b in zip(one, two))
    for rows, group, delta, _ in one:
        assert t[rows].sum() == len(rows) // 2
        assert len(np.unique(sampler.groups[rows])) == 1
        assert delta == pytest.approx(y[rows][t[rows] == 1].mean() - y[rows][t[rows] == 0].mean()
                                     - sampler.references[sampler.group_ids[group]]["contrast"])
    first = batch.MatchedBatches(t, y, q, fit, hold, 9, cfg)
    changed = y.copy(); changed[hold] += 1000 * t[hold]
    second = batch.MatchedBatches(t, changed, q, fit, hold, 9, cfg)
    assert first.references == second.references
    for a, b in zip(first.one_pass(), second.one_pass()):
        np.testing.assert_array_equal(a[0], b[0])
        assert b[2] - a[2] == pytest.approx(1000)
    changed_q = q.copy(); changed_q[hold] = 1e9
    third = batch.MatchedBatches(t, y, changed_q, fit, hold, 9, cfg)
    for a, b in zip(first.cuts, third.cuts):
        np.testing.assert_array_equal(a, b)
    assert first.references == third.references
    strict = batch.MatchedBatches(t, y, q, fit, fit, 9, replace(cfg, max_smd=0))
    assert not list(strict.one_pass()) and strict.stats["imbalance_rejected"] > 0
    with pytest.raises(ValueError, match="disjoint"):
        batch.MatchedBatches(t, y, q, fit, np.arange(10, n), 9, cfg)


def test_ridge_uses_bin_baseline_and_mean_loss_penalty():
    rng = np.random.default_rng(9)
    x = rng.normal(size=(120, 3)); groups = np.repeat([0, 1], 60)
    x[:, 2] = groups  # Entirely between-bin variation supplies no within-bin evidence.
    y = 2 * x[:, 0] - x[:, 1] + 3 * groups
    model = batch.BlockRidge([np.array([0, 1]), np.array([2])], 3, 2, .05)
    model.add(x, y, groups); model.finalize()
    z, residual = x[:, :2] - model.xmean[groups, :2], y - model.ymean[groups]
    expected = np.linalg.solve(z.T @ z / len(x) + .05 * np.eye(2), z.T @ residual / len(x))
    np.testing.assert_allclose(model.beta[0], expected)
    assert not model.informative[1]
    repeated = batch.BlockRidge(model.blocks, 3, 2, .05)
    for _ in range(10): repeated.add(x, y, groups)
    repeated.finalize()
    np.testing.assert_allclose(model.predict(x, groups), repeated.predict(x, groups), atol=1e-12)


def test_signal_recovers_on_separate_patients_and_reaches_role_and_concept_prompts():
    args = cohort()
    result = batch.score_candidates(**args)
    assert result == batch.score_candidates(**args)
    rows = {r["feature_id"]: r for r in result["records"]}
    assert rows["modifier"]["score"] > .7
    assert rows["noise"]["score"] < .1
    assert rows["constant"]["status"] == "not_evaluable"
    assert result["audit"]["variants"]["filtered"]["training_batches"] < result["audit"]["variants"]["unfiltered"]["training_batches"]
    assert result["audit"]["variants"]["filtered"]["validation_batches"] == result["audit"]["variants"]["unfiltered"]["validation_batches"]
    cell = {"family": "matched_batch_contrast", "inner_fold": 1, **result}
    ids = [f["feature_id"] for f in args["definitions"]]
    summary = aggregate_evidence([cell], ids)
    assert summary["modifier"][0]["exposures"] == 1
    assert summary["modifier"][0]["batch_diagnostics"]["mean_training_patients"] <= len(args["train"])
    summary["modifier"][0]["batch_diagnostics"]["patient_ids"] = "DO_NOT_LEAK"
    evidence = build_multi_model_role_evidence(definitions=args["definitions"],
        statistical_report={"multi_model_evidence": summary, "policy": args["policy"].public_dict()},
        policy=Stage2RoleAdjudicationConfig())
    for text in (prompts.evidence_input(evidence, evidence["candidates"]),
                 _concept_input(evidence, evidence["candidates"], dict.fromkeys(ids, 1), 1)):
        assert "DO_NOT_LEAK" not in text
        assert "10% error reduction" in text and "Shuffles reuse patients" in text
        assert "Filtered and unfiltered" in text
    assert "patient_ids" not in json.dumps(evidence)


def test_sparse_bins_empty_tail_and_constant_outcomes_are_unavailable_not_negative():
    args = cohort()
    for changes in ({"min_reference_arm": 1000}, {"min_delta": 1e9}):
        result = batch.score_candidates(**{**args, "policy": policy(**changes)})
        assert all(r["selected"] is None and r["status"] == "not_evaluable" for r in result["records"])
    assert result["audit"]["variants"]["unfiltered"]["validation_batches"] > 0
    assert result["records"][0]["unfiltered_score"] > .7
    zero = batch.score_candidates(**{**args, "y": np.zeros(len(args["y"])), "yv": np.zeros(len(args["yv"]))})
    assert all(r["score"] is None for r in zero["records"])
    assert "NaN" not in json.dumps(zero, allow_nan=False)
    mismatch = {**args["nuisance"], "ev": np.where(args["tv"] == 1, .9, .5)}
    no_match = batch.score_candidates(**{**args, "nuisance": mismatch})
    assert no_match["audit"]["primary_validation_batches"] == 0
    assert all(r["score"] is None for r in no_match["records"])


def test_configuration_preserves_saved_defaults_and_rejects_ontology_options():
    assert not _matched_batch_enabled(SimpleNamespace())  # An in-memory policy from an older running process.
    old = multi_model_config_from_mapping({})
    assert "matched_batch" not in old.public_dict()
    del old.__dict__["matched_batch"]
    restored = pickle.loads(pickle.dumps(old))
    restored.validate()
    assert not restored.matched_batch.enabled
    assert "matched_batch" not in restored.public_dict()
    enabled = multi_model_config_from_mapping({"matched_batch": {"enabled": True}})
    assert multi_model_config_from_mapping(enabled.public_dict()) == enabled
    disabled_custom = replace(enabled, matched_batch=replace(enabled.matched_batch, enabled=False, train_passes=7))
    assert multi_model_config_from_mapping(disabled_custom.public_dict()) == disabled_custom
    for raw in ({"top_modifiers": 20}, {"refine_ontology": True}, {"batch_size": 7},
                {"ridge_penalty": 0}, {"max_smd": float("nan")}, {"train_passes": True}):
        with pytest.raises(ValueError):
            multi_model_config_from_mapping({"matched_batch": raw})
