"""M-01: unpenalized honest nuisance covariates in the penalized interaction model."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from oci.inference import stage2_effect_estimators as effects
from oci.inference.stage2_elastic_net_selection import Stage2ElasticNetSelectionConfig
from oci.inference.stage2_multi_model_config import Stage2MultiModelConfig
from tests.test_stage2_multi_model import feature, small_policy


def _confounded_null_effect(seed, n=500, p=40):
    """Many weak confounders, strong aggregate confounding, true ATE exactly 0."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, p))
    t = rng.binomial(1, expit(0.15 * x.sum(1)))
    y = rng.binomial(1, expit(-0.3 + 0.08 * x.sum(1))).astype(float)
    frame = pd.DataFrame({"_oci_row_id": range(n), **{f"x{j}": x[:, j] for j in range(p)}})
    e, mu0 = np.zeros(n), np.zeros(n)
    for fit, hold in StratifiedKFold(5, shuffle=True, random_state=seed).split(x, t):
        e[hold] = LogisticRegression(max_iter=2000).fit(x[fit], t[fit]).predict_proba(x[hold])[:, 1]
        controls = fit[t[fit] == 0]
        mu0[hold] = (
            LogisticRegression(max_iter=2000).fit(x[controls], y[controls])
            .predict_proba(x[hold])[:, 1]
        )
    return frame, [feature(f"x{j}") for j in range(p)], t, y, e, mu0


def test_unpenalized_propensity_covariate_shrinks_regularization_induced_confounding():
    policy = Stage2ElasticNetSelectionConfig(
        internal_cv_folds=3, max_iter=3000, minimum_log10_alpha=-3, maximum_log10_alpha=0,
        optimization_tolerance=1e-5, categorical_min_count=2,
        multi_model=Stage2MultiModelConfig(regularization_grid_size=8),
    )
    plain_bias, adjusted_bias = [], []
    for seed in range(200, 212):
        frame, definitions, t, y, e, mu0 = _confounded_null_effect(seed)
        kw = dict(train=frame, valid=frame, definitions=definitions, modifier_ids=[],
                  treatment=t, outcome=y, binary=True, policy=policy, seed=seed)
        plain = effects.fit_interaction_effect(**kw)
        adjusted = effects.fit_interaction_effect(**kw, adjustment=effects.nuisance_adjustment(
            train_propensity=e, valid_propensity=e, train_mu0=mu0, valid_mu0=mu0, binary=True))
        # The true average risk difference is exactly zero.
        plain_bias.append(float(plain["tau"].mean()))
        adjusted_bias.append(float(adjusted["tau"].mean()))
        audit = adjusted["audit"]["nuisance_adjustment"]
        assert audit["penalized"] is False and audit["interacted_with_treatment"] is False
        assert "regularization-induced confounding" in audit["description"]
    # Shrunken confounder main effects push confounding into T; the unpenalized
    # propensity/prognostic covariates absorb it.
    assert np.mean(plain_bias) > 0.04
    assert abs(np.mean(adjusted_bias)) < 0.5 * np.mean(plain_bias)


@pytest.mark.parametrize("binary", [False, True])
def test_adjustment_columns_are_unpenalized_main_effects_and_validation_safe(binary):
    rng = np.random.default_rng(7)
    n = 400
    c, m = rng.normal(size=(2, n))
    t = rng.binomial(1, expit(0.8 * c))
    base = 0.6 * c
    shift = 0.5 + 0.8 * m
    y = (rng.binomial(1, expit(base + t * shift)).astype(float) if binary
         else base + t * shift + rng.normal(0, 0.5, n))
    frame = pd.DataFrame({"_oci_row_id": range(n), "c": c, "m": m})
    e = expit(0.8 * c) * 0.9 + 0.05
    mu0 = expit(base) if binary else base
    adjustment = effects.nuisance_adjustment(
        train_propensity=e[:300], valid_propensity=e[300:], train_mu0=mu0[:300],
        valid_mu0=mu0[300:], binary=binary,
    )
    assert adjustment["names"] == [
        "logit_propensity", "logit_prognostic_mu0" if binary else "prognostic_mu0"]
    np.testing.assert_allclose(adjustment["train"][:, 0], effects.bounded_logit(e[:300]))
    fit = dict(
        train=frame.iloc[:300], valid=frame.iloc[300:], definitions=[feature("c"), feature("m")],
        modifier_ids=["m"], treatment=t[:300], outcome=y[:300], binary=binary,
        policy=replace(small_policy(), minimum_log10_alpha=-3, max_iter=3000), seed=5,
        adjustment=adjustment,
    )
    result = effects.fit_interaction_effect(**fit)
    audit = result["audit"]
    blocks = dict(zip(audit["coefficient_feature_ids"], audit["coefficient_blocks"]))
    assert blocks["__logit_propensity__"] == "adjustment"
    assert {"treatment", "adjustment"} <= set(audit["coefficient_blocks"])
    assert "adjustment:logit_propensity" in audit["unpenalized_groups"]
    assert np.allclose(result["tau"], result["mu1"] - result["mu0"])
    # The covariates are main effects: with no selected interaction a continuous
    # outcome model still has one constant effect.
    if not binary:
        flat = effects.fit_interaction_effect(**{**fit, "modifier_ids": []})
        assert np.ptp(flat["tau"]) < 1e-10
    # Validation covariates only shift validation predictions, never the fit.
    shifted = {**adjustment, "valid": adjustment["valid"] + 3.0}
    moved = effects.fit_interaction_effect(**{**fit, "adjustment": shifted})
    assert moved["audit"]["coefficients"] == audit["coefficients"]
    with pytest.raises(ValueError, match="align"):
        effects.fit_interaction_effect(
            **{**fit, "adjustment": {**adjustment, "train": adjustment["train"][:10]}})
