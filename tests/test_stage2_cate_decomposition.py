"""Reporting-only CATE variance decomposition (prognosis versus selected modifiers)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit

from oci.inference import stage2_diagnostics as diagnostics

N = 800
DEFINITIONS = [
    {"feature_id": "f1", "name": "modifier_score", "value_type": "continuous",
     "categories_or_unit": ["unitless"], "roles": ["effect_modifier"]},
    {"feature_id": "f2", "name": "confounder", "value_type": "continuous",
     "categories_or_unit": ["unitless"], "roles": ["confounder"]},
]


def _policy(binary=True):
    return {"outcome_type": "binary" if binary else "continuous", "q": 0.1,
            "weights": None, "min_count": 5}


def _write_run(root: Path, cate, *, binary=True, components=None, seed=0):
    """Two folds; ``cate(prognosis, modifier, rng)`` gives each fold's CATE."""
    rng = np.random.default_rng(seed)
    frames = []
    for fold in (1, 2):
        rows = np.arange((fold - 1) * N, fold * N)
        confounder, modifier = rng.normal(size=N), rng.normal(size=N)
        mu0 = expit(-0.3 + 0.8 * confounder) if binary else 0.8 * confounder
        prognosis = logit(mu0) if binary else mu0
        treatment = rng.binomial(1, 0.5, size=N).astype(float)
        frame = pd.DataFrame({
            "_oci_row_id": rows, "outer_fold": fold, "treatment": treatment,
            "outcome": rng.binomial(1, 0.4, size=N).astype(float) if binary else rng.normal(size=N),
            "propensity": np.full(N, 0.5), "mu0": mu0, "mu1": mu0, "effect_eligible": True,
            "aipw_score": rng.normal(size=N), "estimated_cate": cate(prognosis, modifier, rng),
            "true_ite": rng.normal(size=N),
        })
        for column, function in (components or {}).items():
            frame[column] = function(prognosis, modifier, rng)
        fold_dir = root / f"outer_{fold:03d}"
        (fold_dir / "estimation").mkdir(parents=True)
        (fold_dir / "final_definitions.json").write_text(json.dumps({"features": DEFINITIONS}))
        pd.DataFrame({"_oci_row_id": rows, "modifier_score": modifier,
                      "confounder": confounder}).to_csv(
            fold_dir / "estimation" / "nuisance_candidate_heldout.csv", index=False)
        frames.append(frame)
    predictions = pd.concat(frames, ignore_index=True)
    predictions.to_csv(root / "cross_fitted_predictions.csv", index=False)
    return predictions


def _section(root, predictions, policy=_policy()):
    _, summary = diagnostics.write_heterogeneity_diagnostics(
        output_dir=root, predictions=predictions, propensity_clip=0.02, seed=1,
        n_bootstrap=10, modifier_confirmation_policy=policy,
    )
    text = (root / diagnostics.DIAGNOSTICS_FILENAME).read_text()
    assert "true_ite" not in text
    return json.loads(text)["cate_variance_decomposition"], summary


def test_pure_partition_identities():
    rng = np.random.default_rng(3)
    prognosis, modifiers = rng.normal(size=400), rng.normal(size=(400, 2))
    cate = 0.4 * prognosis + 0.3 * modifiers[:, 0] + 0.1 * rng.normal(size=400)
    result = diagnostics.cate_variance_decomposition(cate, prognosis, modifiers)
    assert result["status"] == "ok" and result["modifier_columns"] == 2
    assert result["unique_prognosis"] + result["unique_modifiers"] + result["shared"] + (
        result["unexplained"]) == pytest.approx(1.0)
    assert result["unique_prognosis"] == pytest.approx(result["r2_both"] - result["r2_modifiers"])
    assert result["shared"] == pytest.approx(
        result["r2_prognosis"] + result["r2_modifiers"] - result["r2_both"])
    none = diagnostics.cate_variance_decomposition(cate, prognosis, np.empty((400, 0)))
    assert none["r2_modifiers"] == 0.0 and none["unique_modifiers"] == pytest.approx(0.0)
    constant = diagnostics.cate_variance_decomposition(np.full(400, 0.2), prognosis, modifiers)
    assert constant["status"] == "unavailable" and constant["cate_variance"] == 0.0
    small = diagnostics.cate_variance_decomposition(cate[:4], prognosis[:4], modifiers[:4])
    assert small["status"] == "unavailable"


@pytest.mark.parametrize("binary", [True])
def test_baseline_risk_only_modifier_only_and_both(tmp_path, binary):
    cases = {
        "prognosis": lambda p, m, rng: 0.2 * p,
        "modifier": lambda p, m, rng: 0.3 * m,
        "both": lambda p, m, rng: 0.2 * p + 0.2 * m + 0.02 * rng.normal(size=len(p)),
    }
    for name, cate in cases.items():
        root = tmp_path / name
        section, summary = _section(root, _write_run(root, cate, binary=binary),
                                    policy=_policy(binary))
        assert section["status"] == "ok"
        assert section["interpretation"] and section["pooling_rule"]
        assert section["outcome_type"] == ("binary" if binary else "continuous")
        pooled = section["pooled"]["estimated_cate"]
        assert pooled["folds"] == ["1", "2"] and pooled["rows"] == 2 * N
        fold = section["per_fold"]["1"]
        assert fold["selected_modifiers"] == ["modifier_score"]
        # No missing values, so the grouped design is the centered value alone.
        assert fold["columns"]["estimated_cate"]["modifier_design_columns"] == [
            "modifier_score:value"]
        if name == "prognosis":
            assert pooled["unique_prognosis"] > 0.99 and pooled["unique_modifiers"] < 0.01
        elif name == "modifier":
            assert pooled["unique_modifiers"] > 0.99 and pooled["unique_prognosis"] < 0.01
        else:
            assert pooled["unique_prognosis"] > 0.3 and pooled["unique_modifiers"] > 0.3
            assert abs(pooled["shared"]) < 0.05 and pooled["unexplained"] < 0.05
        low, high = pooled["range"]["unique_prognosis"]
        assert low <= pooled["unique_prognosis"] <= high
        compact = summary["cate_variance_decomposition"]
        assert compact["status"] == "ok"
        assert compact["estimated_cate"]["unique_prognosis"] == pooled["unique_prognosis"]


def test_primary_heterogeneity_test_is_the_pooled_blp_beta2(tmp_path):
    predictions = _write_run(tmp_path, lambda p, m, rng: 0.2 * p + 0.2 * m)
    # psi carries the predicted heterogeneity, so beta2 > 0 is detectable.
    predictions["aipw_score"] = predictions["estimated_cate"] + np.random.default_rng(4).normal(
        scale=0.5, size=len(predictions))
    _, summary = diagnostics.write_heterogeneity_diagnostics(
        output_dir=tmp_path, predictions=predictions, propensity_clip=0.02, seed=1,
        n_bootstrap=10, modifier_confirmation_policy=_policy(),
    )
    full = json.loads((tmp_path / diagnostics.DIAGNOSTICS_FILENAME).read_text())
    primary = summary["primary_heterogeneity_test"]
    blp = full["blp"]["pooled"]
    assert primary == full["primary_heterogeneity_test"]
    assert primary["status"] == "ok" and primary["pre_specified"] is True
    assert primary["test"] == "pooled_blp_beta2_greater_than_0_one_sided"
    assert primary["beta2"] == blp["beta2_heterogeneity"]
    assert primary["p_value_one_sided"] == blp["beta2_p_value_one_sided_greater_than_0"]
    assert primary["p_value_one_sided"] < 0.01
    assert primary["secondary_tests"] == ["gates", "rate"]
    assert "primary" in primary["description"]
    gates = summary["gates_pooled"]
    assert len(gates["group_mean_estimated_cate"]) == 5
    assert gates["calibration"]["status"] == "ok"
    missing = diagnostics.primary_heterogeneity_test({"blp": {"pooled": {"status": "unavailable",
                                                                         "reason": "x"}}})
    assert missing["status"] == "unavailable" and missing["pre_specified"] is True
