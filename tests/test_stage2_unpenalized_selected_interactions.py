"""The final interaction model leaves the test-selected modifiers unpenalized.

Under ``modifier_selection.rule: orthogonal_tests`` the selected modifiers'
interaction and main-effect columns are exempt from the tuned group elastic
net, guarded by events per parameter, with a fixed-ridge and then a fully
penalized fallback for separation. Zero modifiers, a failed guard, the
``balanced_evidence`` rule, and every validation fit keep the single tuned
penalty exactly.
"""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy.special import expit

from oci.inference import stage2_effect_estimators as effects
from tests.test_stage2_multi_model import feature, small_policy

UNPENALIZED = effects.UnpenalizedSelectedModifiers(events_per_parameter=10.0)


def _data(binary, *, n=600, seed=7):
    rng = np.random.default_rng(seed)
    c = rng.normal(size=(n, 3))
    m = rng.normal(size=n)
    level = rng.choice(["a", "b", "c"], size=n)
    t = rng.binomial(1, expit(0.4 * c[:, 0]))
    base = -0.2 + 0.5 * c.sum(axis=1) + 0.3 * m + 0.4 * (level == "b")
    shift = 0.3 + 0.8 * m - 0.6 * (level == "c")
    y = (
        rng.binomial(1, expit(base + t * shift)).astype(float)
        if binary
        else base + t * shift + rng.normal(scale=0.5, size=n)
    )
    frame = pd.DataFrame(
        {"_oci_row_id": range(n), "c0": c[:, 0], "c1": c[:, 1], "c2": c[:, 2], "m": m,
         "level": level}
    )
    definitions = [
        feature("c0"), feature("c1"), feature("c2"), feature("m"),
        {**feature("level"), "value_type": "categorical", "categories_or_unit": ["a", "b", "c"]},
    ]
    return frame, definitions, t, y


def _fit(binary, *, modifiers=("m", "level"), policy=None, frame_data=None, **kwargs):
    frame, definitions, t, y = frame_data or _data(binary)
    cut = int(0.8 * len(frame))
    return effects.fit_interaction_effect(
        train=frame.iloc[:cut], valid=frame.iloc[cut:], definitions=definitions,
        modifier_ids=list(modifiers), treatment=t[:cut], outcome=y[:cut], binary=binary,
        policy=policy or small_policy(), seed=17, **kwargs,
    )


@pytest.mark.parametrize("binary", [True])
def test_strong_penalty_leaves_an_unpenalized_glm_on_treatment_and_modifiers(binary):
    """Confounders are zeroed; T, Z and T x Z equal the unpenalized GLM fit."""
    frame, definitions, t, y = _data(binary, n=800, seed=3)
    policy = replace(
        small_policy(), minimum_log10_alpha=1, maximum_log10_alpha=2,
        max_iter=20_000, optimization_tolerance=1e-10,
    )
    result = _fit(binary, modifiers=["m"], policy=policy, frame_data=(frame, definitions, t, y),
                  unpenalized_selected=UNPENALIZED)
    audit = result["audit"]
    coefficients = dict(zip(
        [f"{block}:{key}" for key, block in
         zip(audit["coefficient_feature_ids"], audit["coefficient_blocks"])],
        audit["coefficients"],
    ))
    for key in ("main:c0", "main:c1", "main:c2", "main:level"):
        assert coefficients[key] == 0.0
    train, valid = frame.iloc[:640], frame.iloc[640:]
    design = np.column_stack((np.ones(640), t[:640], train.m, t[:640] * train.m))
    family = sm.families.Binomial() if binary else sm.families.Gaussian()
    reference = sm.GLM(y[:640], design, family=family).fit(tol=1e-12)
    arms = [
        reference.predict(np.column_stack((np.ones(160), np.full(160, arm), valid.m,
                                           arm * valid.m)))
        for arm in (0, 1)
    ]
    np.testing.assert_allclose(result["tau"], arms[1] - arms[0], atol=2e-5)
    # The same data with the single tuned penalty shrinks the interaction to zero.
    shrunk = _fit(binary, modifiers=["m"], policy=policy, frame_data=(frame, definitions, t, y))
    effect = [v for k, b, v in zip(shrunk["audit"]["coefficient_feature_ids"],
                                   shrunk["audit"]["coefficient_blocks"],
                                   shrunk["audit"]["coefficients"]) if b == "effect"]
    assert effect == [0.0]


@pytest.mark.parametrize("binary", [True])
def test_events_per_parameter_guard_falls_back_to_the_fully_penalized_model(binary):
    frame, definitions, t, y = _data(binary)
    plain = _fit(binary, frame_data=(frame, definitions, t, y))
    allowed = _fit(binary, frame_data=(frame, definitions, t, y), unpenalized_selected=UNPENALIZED)
    guard = allowed["audit"]["selected_modifier_penalty"]["guard"]
    # m: value; level: two reference contrasts. Main effect plus interaction each.
    assert guard["modifier_encoded_columns"] == {"level": 2, "m": 1}
    assert guard["unpenalized_parameters"] == 6
    assert guard["eligible_rows"] == 480
    if binary:
        events = int(y[:480].sum())
        assert (guard["events"], guard["non_events"]) == (events, 480 - events)
        assert guard["information"] == min(events, 480 - events)
        assert guard["information_basis"] == "min_events_non_events"
    else:
        assert guard["information"] == 480 and guard["information_basis"] == "eligible_rows"
    assert guard["parameter_limit"] == pytest.approx(guard["information"] / 10.0)
    # Just above the limit: the guard fails and the model is the fully penalized one.
    ratio = guard["information"] / 5.9
    blocked = _fit(binary, frame_data=(frame, definitions, t, y),
                   unpenalized_selected=effects.UnpenalizedSelectedModifiers(ratio))
    record = blocked["audit"].pop("selected_modifier_penalty")
    assert (record["path"], record["reason"]) == ("fully_penalized", "events_per_parameter_guard")
    assert record["guard"]["parameter_limit"] == pytest.approx(5.9)
    assert not record["guard"]["passed"] and record["attempts"] == []
    np.testing.assert_array_equal(blocked["tau"], plain["tau"])
    assert blocked["audit"] == plain["audit"]
    exactly = _fit(binary, frame_data=(frame, definitions, t, y),
                   unpenalized_selected=effects.UnpenalizedSelectedModifiers(
                       guard["information"] / 6.0))
    assert exactly["audit"]["selected_modifier_penalty"]["path"] == "unpenalized"
