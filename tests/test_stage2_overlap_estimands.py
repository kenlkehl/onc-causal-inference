"""Co-primary ATE/ATO estimands, the reporting-only overlap flag, and clip sensitivity."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from oci.inference import stage2_diagnostics as diagnostics
from oci.inference.plain_handoff_stage2 import plain_stage2_config_from_mapping


def _population(rng, n):
    x = rng.normal(size=n)
    e = expit(1.5 * x)
    tau = 0.5 + x ** 2
    return x, e, tau


def _predictions(rng, n=1000, *, folds=2, clip=0.02):
    x, e, tau = _population(rng, n)
    t = rng.binomial(1, e).astype(float)
    mu0, mu1 = x, x + tau
    y = np.where(t == 1, mu1, mu0) + rng.normal(size=n)
    return pd.DataFrame({
        "_oci_row_id": np.arange(n), "outer_fold": np.arange(n) % folds + 1,
        "treatment": t, "outcome": y, "propensity": e, "mu0": mu0, "mu1": mu1,
        "effect_eligible": True, "aipw_score": diagnostics.dr_scores(y, t, e, mu0, mu1, clip=clip),
        "estimated_cate": tau,
    })


def test_ato_recovers_the_overlap_weighted_truth_with_calibrated_intervals():
    _, e, tau = _population(np.random.default_rng(0), 2_000_000)
    ato_truth = float(np.sum(e * (1 - e) * tau) / np.sum(e * (1 - e)))
    ate_truth = float(np.mean(tau))
    assert ate_truth - ato_truth > 0.3  # the two populations differ
    rng = np.random.default_rng(1)
    covered, estimates, reps = 0, [], 200
    for _ in range(reps):
        frame = _predictions(rng)
        result = diagnostics.co_primary_estimands(frame, propensity_clip=0.02)["ato"]
        low, high = result["confidence_interval_95"]
        covered += low <= ato_truth <= high
        estimates.append(result["estimate"])
    assert np.mean(estimates) == pytest.approx(ato_truth, abs=0.03)
    assert 0.90 <= covered / reps <= 0.99


def test_lead_estimand_changes_only_order_and_labels():
    frame = _predictions(np.random.default_rng(2))
    ate_first = diagnostics.co_primary_estimands(frame, propensity_clip=0.02)
    ato_first = diagnostics.co_primary_estimands(frame, propensity_clip=0.02, lead_estimand="ato")
    assert ate_first["order"] == ["ate", "ato"] and ato_first["order"] == ["ato", "ate"]
    assert list(ato_first)[:5] == ["lead_estimand", "order", "description", "ato", "ate"]
    assert ate_first["ate"]["role"] == "lead co-primary estimand"
    assert ato_first["ato"]["role"] == "lead co-primary estimand"
    for name in ("ate", "ato"):
        strip = {k: v for k, v in ate_first[name].items() if k != "role"}
        assert strip == {k: v for k, v in ato_first[name].items() if k != "role"}
    assert ate_first["ate_minus_ato"] == ato_first["ate_minus_ato"]
    scores = frame["aipw_score"].to_numpy()
    assert ate_first["ate"]["estimate"] == pytest.approx(scores.mean())
    assert ate_first["ate"]["standard_error"] == pytest.approx(
        scores.std(ddof=1) / np.sqrt(len(scores)))
    assert set(ate_first["ato"]["per_fold"]) == {"1", "2"}
    assert ate_first["ato"]["population"].startswith("patients in clinical equipoise")
    with pytest.raises(ValueError, match="lead_estimand"):
        diagnostics.co_primary_estimands(frame, propensity_clip=0.02, lead_estimand="att")


def test_overlap_flag_marks_extreme_propensities_and_changes_nothing(tmp_path):
    flags = diagnostics.outside_overlap_flags(
        [0.01, 0.05, 0.5, 0.95, 0.96, np.nan], (0.05, 0.95))
    assert flags.tolist() == [True, False, False, False, True, False]
    frame = _predictions(np.random.default_rng(3))
    before = diagnostics.co_primary_estimands(frame, propensity_clip=0.02)
    flagged = frame.copy()
    flagged["outside_overlap"] = diagnostics.outside_overlap_flags(flagged["propensity"],
                                                                   (0.1, 0.9))
    after = diagnostics.co_primary_estimands(flagged, propensity_clip=0.02)
    assert before == after
    summary = diagnostics.overlap_flag_summary(flagged, (0.1, 0.9))
    e = frame["propensity"].to_numpy()
    assert summary["pooled"]["outside_overlap"] == int(((e < 0.1) | (e > 0.9)).sum())
    assert summary["pooled"]["below_lower_bound"] + summary["pooled"]["above_upper_bound"] == (
        summary["pooled"]["outside_overlap"])
    assert sum(f["outside_overlap"] for f in summary["per_fold"].values()) == (
        summary["pooled"]["outside_overlap"])
    path, compact = diagnostics.write_heterogeneity_diagnostics(
        output_dir=tmp_path, predictions=flagged, propensity_clip=0.02, seed=1, n_bootstrap=10,
        overlap_flag_bounds=(0.1, 0.9))
    full = json.loads(path.read_text())
    assert full["overlap_flags"]["bounds"] == [0.1, 0.9]
    assert full["overlap_flags"]["pooled"] == summary["pooled"]
    assert compact["overlap_flags_pooled"]["outside_overlap"] == summary["pooled"]["outside_overlap"]
    # The trimming curve and eligibility are unaffected by the flag.
    assert full["trimming_curve"]["pooled"]["curve"][0]["rows"] == len(frame)


@pytest.mark.parametrize("configured", [0.02, 0.03])
def test_clip_sensitivity_includes_the_configured_clip_equal_to_the_main_ate(configured):
    frame = _predictions(np.random.default_rng(4), clip=configured)
    result = diagnostics.clip_sensitivity(
        *(frame[c] for c in ("outcome", "treatment", "propensity", "mu0", "mu1")),
        configured_clip=configured)
    clips = [point["clip"] for point in result["points"]]
    assert clips == sorted({0.01, 0.02, 0.05, configured})
    main = next(point for point in result["points"] if point["configured"])
    assert sum(point["configured"] for point in result["points"]) == 1
    assert main["clip"] == configured
    assert main["estimate"] == pytest.approx(frame["aipw_score"].mean(), rel=0, abs=1e-12)
    assert all(point["standard_error"] > 0 for point in result["points"])
    full = diagnostics.build_heterogeneity_diagnostics(
        frame, propensity_clip=configured, seed=1, n_bootstrap=10)
    section = full["clip_sensitivity"]
    assert set(section["per_fold"]) == {"1", "2"} and section["interpretation"]
    pooled_main = next(p for p in section["pooled"]["points"] if p["configured"])
    assert pooled_main["estimate"] == pytest.approx(frame["aipw_score"].mean(), abs=1e-12)


def test_configuration_is_strict():
    raw = {"endpoint": "http://stage2.test/v1", "model": "m"}
    default = plain_stage2_config_from_mapping(raw, default_workers=1)
    assert default.lead_estimand == "ate" and default.overlap_flag_bounds == (0.05, 0.95)
    custom = plain_stage2_config_from_mapping(
        {**raw, "lead_estimand": "ato", "overlap_flag_bounds": [0.1, 0.9]}, default_workers=1)
    assert custom.lead_estimand == "ato" and custom.overlap_flag_bounds == (0.1, 0.9)
    public = custom.public_dict()
    restored = plain_stage2_config_from_mapping(
        {**raw, "lead_estimand": public["lead_estimand"],
         "overlap_flag_bounds": public["overlap_flag_bounds"]}, default_workers=1)
    assert restored.overlap_flag_bounds == custom.overlap_flag_bounds
    for bad in ({"lead_estimand": "att"}, {"lead_estimand": "ATE"},
                {"overlap_flag_bounds": [0.95, 0.05]}, {"overlap_flag_bounds": [0.0, 0.9]},
                {"overlap_flag_bounds": [0.1]}, {"overlap_flag_bounds": "0.05,0.95"},
                {"overlap_flag_bounds": [True, 0.9]}, {"overlap_flag_bounds": [0.1, float("nan")]}):
        with pytest.raises(ValueError, match="lead_estimand|overlap_flag_bounds"):
            plain_stage2_config_from_mapping({**raw, **bad}, default_workers=1)
