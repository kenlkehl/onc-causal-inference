"""Counterfactual scales, train-only fits, and production architecture dispatch."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from oci.inference import plain_handoff_stage2_analysis as analysis
from oci.inference import stage2_effect_estimators as effects
from tests.test_stage2_multi_model import feature, small_policy
from tests.test_stage2_propensity_overlap import final_fixture


@pytest.mark.parametrize("binary", [False, True])
def test_interaction_model_learns_counterfactual_effects_on_the_outcome_scale(binary):
    rng = np.random.default_rng(533)
    n = 1600
    c, m = rng.normal(size=(2, n))
    category = rng.integers(0, 2, n)
    t = rng.binomial(1, expit(0.4 * c))
    baseline = -0.3 + 0.8 * c + 0.4 * category
    shift = 0.4 + 1.6 * m
    mu0, mu1 = (
        (expit(baseline), expit(baseline + shift)) if binary else (baseline, baseline + shift)
    )
    y = (
        rng.binomial(1, np.where(t, mu1, mu0))
        if binary
        else np.where(t, mu1, mu0) + rng.normal(0, 0.4, n)
    )
    frame = pd.DataFrame(
        {"_oci_row_id": range(n), "c": c, "m": m, "category": np.where(category, "a", "b")}
    )
    definitions = [
        feature("c"),
        feature("m"),
        {**feature("category"), "value_type": "categorical", "categories_or_unit": ["a", "b"]},
    ]
    policy = replace(small_policy(), minimum_log10_alpha=-3, max_iter=3000)
    fit = dict(
        train=frame.iloc[:1200],
        valid=frame.iloc[1200:],
        definitions=definitions,
        modifier_ids=["m"],
        treatment=t[:1200],
        outcome=y[:1200],
        binary=binary,
        policy=policy,
        seed=37,
    )
    result = effects.fit_interaction_effect(**fit)
    assert np.corrcoef(result["tau"], (mu1 - mu0)[1200:])[0, 1] > 0.95
    assert np.sqrt(np.mean((result["tau"] - (mu1 - mu0)[1200:]) ** 2)) < 0.12
    assert np.allclose(result["tau"], result["mu1"] - result["mu0"])
    if binary:
        assert np.all((result["mu0"] >= 0) & (result["mu0"] <= 1))
        assert np.all((result["mu1"] >= 0) & (result["mu1"] <= 1))
    audit = result["audit"]
    assert {
        key
        for key, block in zip(audit["coefficient_feature_ids"], audit["coefficient_blocks"])
        if block == "effect"
    } == {"m"}
    assert set(audit["main_feature_ids"]) == {"c", "m", "category"}
    assert "treatment" in audit["coefficient_blocks"]
    assert set(audit["fit_row_ids"]).isdisjoint(audit["validation_row_ids"])
    # Adding an extreme validation record cannot change fitting or other predictions.
    poison = pd.DataFrame({"_oci_row_id": [n], "c": [1e6], "m": [-1e6], "category": ["unseen"]})
    altered = effects.fit_interaction_effect(**{**fit, "valid": pd.concat([fit["valid"], poison])})
    assert np.allclose(result["tau"], altered["tau"][:-1])
    assert audit["coefficients"] == altered["audit"]["coefficients"]
    no_interactions = effects.fit_interaction_effect(**{**fit, "modifier_ids": []})
    assert "effect" not in no_interactions["audit"]["coefficient_blocks"]
    # Zero interactions on the logit scale can still yield varying risk differences.
    assert np.ptp(no_interactions["tau"]) > 0 if binary else np.ptp(no_interactions["tau"]) < 1e-12


def test_final_interaction_dispatch_overlap_resume_and_no_heldout_label_use(monkeypatch, tmp_path):
    kwargs, calls = final_fixture(monkeypatch, tmp_path)
    kwargs.update(
        estimator="linear_interactions",
        statistical_policy=small_policy(),
        min_propensity=0.1,
        max_propensity=0.9,
    )
    diagnostics = analysis.estimate_outer_fold(**kwargs)
    frame = pd.read_csv(tmp_path / "predictions.csv")
    assert not any(c[0] == "effect" for c in calls)  # A forest was not fitted.
    assert diagnostics["model_family"] == "penalized_logistic_interactions"
    assert diagnostics["effect_fit_rows"] == 12
    assert diagnostics["interaction_fit_audit"]["fit_row_ids"] == [
        i for i in range(18) if i % 6 not in (0, 4)
    ]
    assert frame.loc[frame.effect_eligible, "estimated_cate"].notna().all()
    assert frame.loc[~frame.effect_eligible, "estimated_cate"].isna().all()
    assert frame.estimated_cate_lower_95.isna().all()
    assert diagnostics["confidence_interval_95"] is not None  # AIPW ATE, not CATE inference.
    changed = kwargs["dataset"].copy()
    changed.loc[18:, "outcome"] = 1 - changed.loc[18:, "outcome"]
    analysis.estimate_outer_fold(
        **{**kwargs, "dataset": changed, "output_dir": tmp_path / "changed"}
    )
    pd.testing.assert_series_equal(
        frame.estimated_cate, pd.read_csv(tmp_path / "changed/predictions.csv").estimated_cate
    )
    monkeypatch.setattr(
        effects, "fit_interaction_effect", lambda **k: pytest.fail("cached fit recomputed")
    )
    assert analysis.estimate_outer_fold(**kwargs) == diagnostics
    # An architecture change must not reuse the interaction result.
    forest = analysis.estimate_outer_fold(**{**kwargs, "estimator": "causal_forest"})
    assert forest["model_family"] == "causal_forest_dml" and ("effect", 12) in calls
