from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from oci.inference.nuisance_diagnostics import (
    calibration_diagnostics,
    overlap_diagnostics,
    propensity_eligibility,
    validate_propensity_bounds,
)
from oci.inference import plain_handoff_stage2_analysis as analysis
from oci.inference.plain_handoff_stage2 import plain_stage2_config_from_mapping


@pytest.mark.parametrize(
    "low,high",
    [
        (True, 0.9),
        (-0.1, 0.9),
        (0.1, 1.1),
        (0.9, 0.1),
        (0.1, 0.1),
        (float("nan"), None),
        (None, float("inf")),
    ],
)
def test_invalid_bounds(low, high):
    with pytest.raises(ValueError):
        validate_propensity_bounds(low, high)


def test_bounds_are_optional_inclusive_and_differ_from_clipping():
    p = np.array([0.0, 0.099, 0.1, 0.5, 0.9, 0.901, 1.0])
    assert propensity_eligibility(p).all()
    assert propensity_eligibility(p, 0.1, 0.9).tolist() == [
        False,
        False,
        True,
        True,
        True,
        False,
        False,
    ]
    assert propensity_eligibility(p, None, 0.9).sum() == 5
    assert propensity_eligibility(p, 0.1, None).sum() == 5
    assert overlap_diagnostics(p, 0.1, 0.9)["excluded_rows"] == 4
    with pytest.raises(ValueError):
        propensity_eligibility([np.nan])


def test_calibration_distinguishes_correct_from_overconfident_probabilities():
    # Exact frequencies at each probability level yield intercept 0, slope 1.
    p = np.repeat([0.2, 0.5, 0.8], 100)
    y = np.concatenate([np.r_[np.ones(n), np.zeros(100 - n)] for n in (20, 50, 80)])
    d = calibration_diagnostics(y, p, binary=True)
    assert d["calibration_slope"] == pytest.approx(1, abs=1e-5)
    assert d["calibration_intercept"] == pytest.approx(0, abs=1e-5)
    assert d["expected_calibration_error_10_bins"] == pytest.approx(0, abs=1e-10)
    over = calibration_diagnostics(y, np.repeat([0.01, 0.5, 0.99], 100), binary=True)
    assert over["auroc"] == d["auroc"]
    assert over["calibration_slope"] < 0.4
    assert over["brier_score"] > d["brier_score"]
    assert calibration_diagnostics([1, 1], [0.4, 0.8], binary=True)["calibration_slope"] is None
    assert calibration_diagnostics([], [], binary=True)["status"] == "no_rows"
    separated = calibration_diagnostics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], binary=True)
    assert separated["calibration_slope"] is None
    assert separated["calibration_status"] == "nonconverged_or_separated"
    continuous = calibration_diagnostics([1, 3, 5], [0, 1, 2], binary=False)
    assert continuous["calibration_intercept"] == pytest.approx(1)
    assert continuous["calibration_slope"] == pytest.approx(2)
    json.dumps(d, allow_nan=False)


def test_stage2_bound_config_roundtrip():
    c = plain_stage2_config_from_mapping(
        {
            "endpoint": "http://localhost/v1",
            "model": "test",
            "min_propensity": 0.1,
            "max_propensity": 0.9,
        },
        default_workers=1,
    )
    assert (c.min_propensity, c.max_propensity) == (0.1, 0.9)
    c2 = plain_stage2_config_from_mapping(c.public_dict(), default_workers=1)
    assert (c2.min_propensity, c2.max_propensity) == (0.1, 0.9)
    with pytest.raises(ValueError, match="stage2 level"):
        plain_stage2_config_from_mapping(
            {
                "endpoint": "http://localhost/v1",
                "model": "test",
                "statistical_selection": {"min_propensity": 0.1},
            },
            default_workers=1,
        )


def final_fixture(monkeypatch, tmp_path):
    n = 24
    dataset = pd.DataFrame(
        {
            "patient_id": range(n),
            "treatment": np.tile([0.0, 1.0], 12),
            "outcome": np.tile([0.0, 1.0, 1.0, 0.0], 6),
        }
    )
    extracted = pd.DataFrame({"_oci_row_id": range(n), "x": np.arange(n, dtype=float)})
    fitp = np.tile([0.05, 0.1, 0.5, 0.9, 0.95, 0.5], 3)
    heldp = np.array([0.05, 0.1, 0.5, 0.9, 0.95, 0.5])
    calls = []

    def crossfit(**kw):
        # Training OOF predictions plus the inner-fold ensemble's held-out ones.
        calls.append(("crossfit", len(kw["fit_ids"]), len(kw["heldout_extracted"])))
        return (
            fitp.copy(), np.full(18, 0.2), np.full(18, 0.8),
            heldp.copy(), np.full(6, 0.2), np.full(6, 0.8),
        )

    monkeypatch.setattr(analysis, "_cross_fitted_nuisance", crossfit)

    class Forest:
        def __init__(self, **kw):
            pass

        def fit(self, x, t, y, W=None):
            calls.append(("effect", len(y)))
            calls.append(("effect_y", y.copy()))

        def predict(self, x, **kw):
            calls.append(("predict", len(x)))
            return {"tau_pred": np.full(len(x), 0.4)}

        def fit_audit(self):
            return {"outcome_model_contract": {}}

    monkeypatch.setattr(analysis, "CausalForestHead", Forest)
    definitions = [
        {
            "feature_id": "x",
            "name": "x",
            "value_type": "continuous",
            "roles": ["confounder", "effect_modifier"],
            "nuisance_model_roles": ["treatment", "outcome"],
        }
    ]
    kwargs = dict(
        dataset=dataset,
        extracted_fit=extracted.iloc[:18],
        extracted_heldout=extracted.iloc[18:],
        definitions=definitions,
        nuisance_candidate_definitions=definitions,
        nuisance_candidate_fit=extracted.iloc[:18],
        nuisance_candidate_heldout=extracted.iloc[18:],
        split={"fit_row_ids": list(range(18)), "heldout_row_ids": list(range(18, 24))},
        unit_id_column="patient_id",
        treatment_column="treatment",
        outcome_column="outcome",
        outcome_type="binary",
        inner_folds=3,
        seed=3,
        propensity_clip=0.02,
        estimation_trees=20,
        output_dir=tmp_path,
    )
    return kwargs, calls


def test_final_exclusion_preserves_audit_rows_and_invalidates_cache(monkeypatch, tmp_path):
    kw, calls = final_fixture(monkeypatch, tmp_path)
    d = analysis.estimate_outer_fold(**kw, min_propensity=0.1, max_propensity=0.9)
    # External nuisances retain all training patients and score every held-out one.
    assert ("crossfit", 18, 6) in calls
    assert ("effect", 12) in calls
    assert ("predict", 4) in calls
    frame = pd.read_csv(tmp_path / "predictions.csv")
    assert len(frame) == 6
    assert frame.effect_eligible.tolist() == [False, True, True, True, False, True]
    assert frame.loc[~frame.effect_eligible, ["aipw_score", "estimated_cate"]].isna().all().all()
    assert d["ate_aipw"] == pytest.approx(frame.loc[frame.effect_eligible, "aipw_score"].mean())
    assert d["effect_estimation_rows"] == 4
    assert d["nuisance_calibration"]["heldout_treatment"]["rows"] == 6
    assert d["nuisance_calibration"]["eligible_heldout_treatment"]["rows"] == 4
    assert len(pd.read_csv(tmp_path / "nuisance_fit_predictions.csv")) == 18
    before = len(calls)
    analysis.estimate_outer_fold(**kw, min_propensity=0.1, max_propensity=0.9)
    assert len(calls) == before
    analysis.estimate_outer_fold(**kw)  # changed bounds cannot reuse trimmed estimates
    assert ("effect", 18) in calls
    assert pd.read_csv(tmp_path / "predictions.csv").effect_eligible.all()


def test_final_empty_overlap_fails_without_untrimmed_fallback(monkeypatch, tmp_path):
    kw, calls = final_fixture(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="insufficient"):
        analysis.estimate_outer_fold(**kw, min_propensity=0.11, max_propensity=0.12)
    assert not any(c[0] == "effect" for c in calls)
    assert (tmp_path / "propensity_overlap.json").exists()
    assert not (tmp_path / "complete.json").exists()


def test_bounds_change_requires_reselection_and_changes_migration_fingerprint(tmp_path):
    from tests.test_stage2_launch_preflight import saved_fixture, write
    from oci.inference import stage2_preflight, research_all_evidence_workflow as workflow

    cfg = saved_fixture(tmp_path)
    write(cfg.output_dir / "stage2" / "config.json", cfg.stage2.public_dict())
    bounded = replace(cfg, stage2=replace(cfg.stage2, min_propensity=0.1, max_propensity=0.9))
    before = workflow._stage2_reselection_policy_fingerprint(cfg)
    assert before != workflow._stage2_reselection_policy_fingerprint(bounded)
    with pytest.raises(RuntimeError, match="propensity bounds changed"):
        stage2_preflight.validate_selection_resume(bounded)
