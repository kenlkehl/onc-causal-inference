"""Calibration, power, and mechanics of the orthogonal modifier tests."""

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold

from oci.inference import stage2_modifier_tests as mt

COMPONENTS = mt.COMPONENTS
BINARY_WEIGHTS = mt.default_weights("binary")


def feature(key, **overrides):
    return {
        "feature_id": key,
        "name": key,
        "value_type": "continuous",
        "categories_or_unit": ["unitless"],
        "measurement_definition": "A measured baseline attribute.",
        "description": "A baseline measurement.",
        **overrides,
    }


DEFINITIONS = [
    feature("confounder"),
    feature("prognostic"),
    feature("modifier"),
    feature("stage", value_type="categorical", categories_or_unit=["a", "b", "c"]),
    *[feature(f"noise_{j}") for j in range(4)],
]


def cross_fitted_nuisances(x, t, y, *, binary, seed, folds=5):
    """Out-of-fold e, mu0, mu1 fitted on the true covariates."""
    e, mu0, mu1 = (np.empty(len(t)) for _ in range(3))
    for fit, hold in KFold(folds, shuffle=True, random_state=seed).split(x):
        e[hold] = LogisticRegression(C=1e6, max_iter=1000).fit(
            x[fit], t[fit]).predict_proba(x[hold])[:, 1]
        for arm, target in ((0, mu0), (1, mu1)):
            rows = fit[t[fit] == arm]
            if binary:
                model = LogisticRegression(C=1e6, max_iter=1000).fit(x[rows], y[rows])
                target[hold] = model.predict_proba(x[hold])[:, 1]
            else:
                target[hold] = LinearRegression().fit(x[rows], y[rows]).predict(x[hold])
    return e, mu0, mu1


def simulate(rng, n, scenario):
    """Candidates: confounder, prognostic, modifier, a categorical with missing values, noise."""
    c, prognostic, modifier = rng.normal(size=(3, n))
    noise = rng.normal(size=(n, 4))
    stage = rng.choice(["a", "b", "c"], size=n).astype(object)
    stage[rng.random(n) < 0.1] = None
    t = rng.binomial(1, 1 / (1 + np.exp(-0.6 * c)))
    if scenario == "binary_null":
        # No treatment effect at all: no candidate modifies it on any scale.
        y = rng.binomial(1, 1 / (1 + np.exp(-(-0.2 + 0.6 * c + prognostic))))
    elif scenario == "binary_constant_odds_ratio":
        # Constant odds ratio at low baseline risk: the risk difference rises
        # with the prognostic variable, a risk-difference-only modifier.
        y = rng.binomial(1, 1 / (1 + np.exp(-(-2.0 + 1.2 * prognostic + 1.5 * t))))
    elif scenario == "binary_multiplicative":
        y = rng.binomial(1, 1 / (1 + np.exp(-(0.5 * c + 0.8 * t * modifier))))
    elif scenario == "continuous_null":
        y = 0.6 * c + prognostic + 0.5 * t + rng.normal(size=n)
    elif scenario == "continuous_additive":
        y = 0.6 * c + prognostic + t * (0.5 + 0.3 * modifier) + rng.normal(size=n)
    else:
        raise ValueError(scenario)
    frame = pd.DataFrame({
        "confounder": c, "prognostic": prognostic, "modifier": modifier, "stage": stage,
        **{f"noise_{j}": noise[:, j] for j in range(4)},
    })
    x = np.column_stack((c, prognostic, modifier))
    return frame, x, t.astype(float), y.astype(float), not scenario.startswith("continuous")


def run_tests(rng, n, scenario, *, seed, q=0.1):
    frame, x, t, y, binary = simulate(rng, n, scenario)
    e, mu0, mu1 = cross_fitted_nuisances(x, t, y, binary=binary, seed=seed)
    records, audit = mt.candidate_tests(
        frame, DEFINITIONS, treatment=t, outcome=y, propensity=e, mu0=mu0, mu1=mu1,
        binary=binary, propensity_clip=0.02,
    )
    weights = mt.default_weights("binary" if binary else "continuous")
    records, selected = mt.combine_and_select(records, weights=weights, q=q, binary=binary)
    return {r["feature_id"]: r for r in records}, selected, audit


# ---------------------------------------------------------------------------
# Cauchy combination


def test_cauchy_combination_known_values_renormalization_and_extremes():
    equal = mt.cauchy_combination({c: 0.03 for c in COMPONENTS}, BINARY_WEIGHTS)
    assert equal["p_value"] == pytest.approx(0.03, rel=1e-9)
    assert sum(equal["weights"].values()) == pytest.approx(1.0)
    # One available component returns its own p-value with weight one.
    single = mt.cauchy_combination(
        {"r_projection": 0.2, "aipw_projection": None, "outcome_interaction": None},
        BINARY_WEIGHTS,
    )
    assert single["p_value"] == pytest.approx(0.2, rel=1e-9)
    assert single["weights"] == {"r_projection": 1.0}
    # Weights renormalize over the components with p-values (or positive weight).
    two = mt.cauchy_combination(
        {"r_projection": 0.01, "aipw_projection": None, "outcome_interaction": 0.5},
        BINARY_WEIGHTS,
    )
    assert two["weights"] == pytest.approx({"r_projection": 1 / 3, "outcome_interaction": 2 / 3})
    statistic = math.tan((0.5 - 0.01) * math.pi) / 3 + 2 * math.tan(0.0) / 3
    assert two["statistic"] == pytest.approx(statistic, rel=1e-9)
    assert two["p_value"] == pytest.approx(0.5 - math.atan(statistic) / math.pi, rel=1e-9)
    zero_weight = mt.cauchy_combination(
        {c: p for c, p in zip(COMPONENTS, (0.01, 0.9, 0.5))},
        {"r_projection": 1.0, "aipw_projection": 0.0, "outcome_interaction": 1.0},
    )
    assert zero_weight["components"] == ["r_projection", "outcome_interaction"]
    assert zero_weight["weights"] == {"r_projection": 0.5, "outcome_interaction": 0.5}
    none = mt.cauchy_combination({c: None for c in COMPONENTS}, BINARY_WEIGHTS)
    assert none["p_value"] is None and none["components"] == []
    # Extreme p-values are clamped: finite statistics and probabilities.
    for values in ((0.0, 0.5, 0.5), (1e-300, 0.9, 0.9), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0)):
        result = mt.cauchy_combination(dict(zip(COMPONENTS, values)), BINARY_WEIGHTS)
        assert math.isfinite(result["statistic"]) and 0.0 <= result["p_value"] <= 1.0
    tiny = mt.cauchy_combination(dict(zip(COMPONENTS, (0.0, 0.0, 0.0))), BINARY_WEIGHTS)
    assert tiny["p_value"] == pytest.approx(mt.P_VALUE_CLAMP, rel=1e-6)
    assert mt.cauchy_combination(dict(zip(COMPONENTS, (1.0, 1.0, 1.0))), BINARY_WEIGHTS)[
        "p_value"] == pytest.approx(1.0, abs=1e-12)
    # Monotone in each component p-value.
    grid = np.linspace(0.001, 0.999, 60)
    for name in COMPONENTS:
        values = [
            mt.cauchy_combination({**dict.fromkeys(COMPONENTS, 0.3), name: p}, BINARY_WEIGHTS)[
                "p_value"]
            for p in grid
        ]
        assert np.all(np.diff(values) > 0)


def test_hc3_wald_matches_statsmodels():
    import statsmodels.api as sm

    rng = np.random.default_rng(5)
    n = 250
    x = np.column_stack((np.ones(n), rng.normal(size=(n, 3)), rng.binomial(1, 0.3, n)))
    y = x @ [0.5, 0.2, 0.0, -0.1, 0.3] + rng.normal(size=n) * (1 + np.abs(x[:, 1]))
    p_value, details = mt.hc3_wald_test(y, x, [1, 3, 4])
    fit = sm.OLS(y, x).fit(cov_type="HC3")
    restriction = np.eye(5)[[1, 3, 4]]
    wald = fit.wald_test(restriction, use_f=False, scalar=True)
    assert p_value == pytest.approx(float(wald.pvalue), rel=1e-8)
    assert details["statistic"] == pytest.approx(float(wald.statistic), rel=1e-8)
    assert details["degrees_of_freedom"] == 3
    np.testing.assert_allclose(details["standard_errors"], fit.bse[[1, 3, 4]], rtol=1e-8)
    # A row with leverage one (a column nonzero in one row) has no HC3 test.
    spike = np.zeros(n)
    spike[0] = 1.0
    assert mt.hc3_wald_test(y, np.column_stack((x, spike)), [5])[1]["reason"] == (
        "row_with_leverage_one")
    assert mt.hc3_wald_test(y, x, [])[1]["reason"] == "no_independent_candidate_columns"


# ---------------------------------------------------------------------------
# Calibration under the null and power


@pytest.mark.parametrize("scenario", ["binary_null"])
def test_components_and_combination_are_calibrated_without_heterogeneity(scenario):
    rng = np.random.default_rng(41 if scenario == "binary_null" else 42)
    replicates = 120
    p_values = {name: [] for name in (*COMPONENTS, "combined")}
    any_selection = 0
    for replicate in range(replicates):
        records, selected, audit = run_tests(rng, 600, scenario, seed=replicate)
        assert audit["evaluable"] and audit["eligible_rows"] == 600
        any_selection += bool(selected)
        for record in records.values():
            for name in COMPONENTS:
                p_values[name].append(record["components"][name]["p_value"])
            p_values["combined"].append(record["combined_p_value"])
        assert all(not r["labels"] for r in records.values() if not r["selected"])
    pooled = len(p_values["combined"])
    for name, values in p_values.items():
        values = np.asarray(values, dtype=float)
        assert np.isfinite(values).all() and len(values) == pooled
        if name != "combined":
            # The component tests are uniform under this null.
            assert stats.kstest(values, "uniform").pvalue > 0.001, name
        # The Cauchy combination of dependent components is accurate in the
        # lower tail (it piles up near 1), which is all that selection uses.
        for alpha, tolerance in ((0.01, 0.012), (0.05, 0.025), (0.1, 0.035)):
            assert abs(np.mean(values < alpha) - alpha) < tolerance, (name, alpha)
    # BH at q = 0.1 under the global null: family-wise "any selection" <= q.
    monte_carlo_se = math.sqrt(0.1 * 0.9 / replicates)
    assert any_selection / replicates <= 0.1 + 2.5 * monte_carlo_se


def test_risk_difference_only_modifier_is_found_by_the_absolute_components():
    rng = np.random.default_rng(51)
    replicates = 16
    absolute_detected = interaction_detected = selected = 0
    for replicate in range(replicates):
        records, _, _ = run_tests(rng, 1000, "binary_constant_odds_ratio", seed=replicate)
        record = records["prognostic"]
        components = record["components"]
        absolute_detected += components["r_projection"]["passes"] or components[
            "aipw_projection"]["passes"]
        interaction_detected += components["outcome_interaction"]["passes"]
        if record["selected"]:
            selected += 1
            assert "absolute" in record["labels"]
            assert "multiplicative" not in record["labels"]
            assert "combined_only" not in record["labels"]
    assert absolute_detected >= 0.7 * replicates
    assert interaction_detected <= 0.2 * replicates
    assert selected >= 0.6 * replicates


# ---------------------------------------------------------------------------
# Selection mechanics


def record(feature_id, r, a, o, **extra):
    return {
        "feature_id": feature_id,
        "components": {
            "r_projection": {"status": "ok", "p_value": r},
            "aipw_projection": {"status": "ok", "p_value": a},
            "outcome_interaction": {"status": "ok", "p_value": o},
        },
        **extra,
    }


def test_combine_and_select_family_ordering_cap_and_labels():
    records = [
        record("strong", 1e-8, 1e-8, 1e-8),
        record("medium", 1e-4, 1e-4, 0.5),
        record("null", 0.6, 0.7, 0.8),
        record("locked", 1e-12, 1e-12, 1e-12),
        record("untestable", None, None, None),
    ]
    family = ["strong", "medium", "null", "untestable"]
    updated, selected = mt.combine_and_select(
        records, weights=BINARY_WEIGHTS, q=0.1, binary=True, family_ids=family
    )
    by_id = {r["feature_id"]: r for r in updated}
    assert selected == ["strong", "medium"]
    assert [by_id[k]["selection_rank"] for k in ("strong", "medium")] == [1, 2]
    # Records outside the family keep a combined p-value but get no q-value.
    locked = by_id["locked"]
    assert locked["combined_p_value"] < 1e-10 and locked["combined_q_value"] is None
    assert not locked["selected"] and not locked["bh_family"]
    assert all(c["q_value"] is None for c in locked["components"].values())
    # Untestable candidates do not count in the BH family size.
    untestable = by_id["untestable"]
    assert untestable["combined_p_value"] is None and untestable["combined_q_value"] is None
    expected_q = mt._benjamini_hochberg([by_id[k]["combined_p_value"] for k in family])
    assert [by_id[k]["combined_q_value"] for k in family] == expected_q
    assert by_id["strong"]["labels"] == ["absolute", "multiplicative"]
    assert by_id["medium"]["labels"] == ["absolute"]
    assert by_id["null"]["labels"] == [] and by_id["null"]["q_threshold"] == 0.1
    # Original records are not mutated.
    assert "combined_p_value" not in records[0]
    capped, kept = mt.combine_and_select(
        records, weights=BINARY_WEIGHTS, q=0.1, binary=True, family_ids=family,
        max_modifiers=1,
    )
    capped = {r["feature_id"]: r for r in capped}
    assert kept == ["strong"]
    assert capped["medium"]["excluded_by_cap"] and not capped["medium"]["selected"]
    assert capped["medium"]["selection_rank"] is None and capped["medium"]["labels"] == []
    assert not capped["strong"]["excluded_by_cap"]
    continuous, _ = mt.combine_and_select(
        records, weights=mt.default_weights("continuous"), q=0.1, binary=False,
        family_ids=family,
    )
    assert continuous[0]["labels"] == ["absolute", "additive"]
