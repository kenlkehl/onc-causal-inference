"""Stage 2 heterogeneity, overlap and balance diagnostics on simulated data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from oci.inference import stage2_diagnostics as diagnostics


def _simulate_fold(n: int, *, heterogeneity: float, seed: int, informative_cate: bool):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    propensity = expit(0.5 * x)
    treatment = rng.binomial(1, propensity)
    tau = 0.5 + heterogeneity * x
    mu0 = 1.0 + 0.5 * x
    mu1 = mu0 + tau
    outcome = mu0 + treatment * tau + rng.normal(size=n)
    score = diagnostics.dr_scores(outcome, treatment, propensity, mu0, mu1, clip=0.02)
    cate = tau if informative_cate else rng.normal(size=n)
    return pd.DataFrame(
        {
            "treatment": treatment,
            "outcome": outcome,
            "propensity": propensity,
            "mu0": mu0,
            "mu1": mu1,
            "aipw_score": score,
            "estimated_cate": cate,
            "x": x,
        }
    )


def _predictions(*, heterogeneity: float, informative_cate: bool, n: int = 1500, seed: int = 0):
    frames = []
    for fold in (1, 2, 3):
        frame = _simulate_fold(
            n, heterogeneity=heterogeneity, seed=seed + fold, informative_cate=informative_cate
        )
        frame.insert(0, "outer_fold", fold)
        frames.append(frame)
    predictions = pd.concat(frames, ignore_index=True)
    predictions.insert(0, "_oci_row_id", np.arange(len(predictions)))
    return predictions


def test_strong_heterogeneity_gives_calibrated_significant_blp_gates_and_rate():
    predictions = _predictions(heterogeneity=1.0, informative_cate=True)
    result = diagnostics.build_heterogeneity_diagnostics(
        predictions, propensity_clip=0.02, seed=11, n_bootstrap=200
    )

    for fold_result in result["blp"]["per_fold"].values():
        assert fold_result["status"] == "ok"
        assert fold_result["covariance"] == "HC3"
        assert 0.8 < fold_result["beta2_heterogeneity"] < 1.2
        assert fold_result["beta2_p_value_one_sided_greater_than_0"] < 1e-6
    pooled = result["blp"]["pooled"]
    assert pooled["method"] == "fixed_effect_inverse_variance"
    assert pooled["folds"] == ["1", "2", "3"]
    assert 0.9 < pooled["beta2_heterogeneity"] < 1.1
    assert pooled["beta2_p_value_one_sided_greater_than_0"] < 1e-10
    # Calibration (beta2 = 1) is not rejected for the true CATE.
    assert pooled["beta2_p_value_two_sided_equal_1"] > 0.01
    assert abs(pooled["beta1_ate"] - 0.5) < 0.15
    fold_se = [r["beta2_standard_error"] for r in result["blp"]["per_fold"].values()]
    assert pooled["beta2_standard_error"] < min(fold_se)

    gates = result["gates"]["pooled"]
    means = [group["estimate"] for group in gates["groups"]]
    assert means == sorted(means)
    assert gates["top_minus_bottom"]["estimate"] > 2.0
    assert gates["top_minus_bottom"]["p_value_two_sided"] < 1e-10
    assert gates["linear_trend_per_group"]["estimate"] > 0
    assert gates["linear_trend_per_group"]["p_value_one_sided_greater_than_0"] < 1e-10
    for fold_result in result["gates"]["per_fold"].values():
        assert fold_result["top_minus_bottom"]["estimate"] > 0
        assert [g["rows"] for g in fold_result["groups"]] == [300] * 5

    rate = result["rate"]["pooled"]
    assert rate["priority"] == "within_fold_percentile_of_estimated_cate"
    assert rate["autoc"]["estimate"] > 0
    assert rate["autoc"]["p_value_one_sided_greater_than_0"] < 1e-6
    assert rate["qini"]["estimate"] > 0
    assert rate["bootstrap_replicates"] == 200
    for fold_result in result["rate"]["per_fold"].values():
        assert fold_result["autoc"]["confidence_interval_95"][0] > 0


def test_no_heterogeneity_gives_nonsignificant_blp_slope():
    predictions = _predictions(heterogeneity=0.0, informative_cate=False, seed=100)
    result = diagnostics.build_heterogeneity_diagnostics(
        predictions, propensity_clip=0.02, seed=11, n_bootstrap=200
    )

    pooled = result["blp"]["pooled"]
    assert pooled["status"] == "ok"
    assert pooled["beta2_p_value_one_sided_greater_than_0"] > 0.05
    lower, upper = pooled["beta2_confidence_interval_95"]
    assert lower < 0 < upper
    assert abs(pooled["beta1_ate"] - 0.5) < 0.15
    for fold_result in result["blp"]["per_fold"].values():
        assert fold_result["beta2_p_value_one_sided_greater_than_0"] > 0.05
    gates = result["gates"]["pooled"]["top_minus_bottom"]
    assert gates["confidence_interval_95"][0] < 0 < gates["confidence_interval_95"][1]
    autoc = result["rate"]["pooled"]["autoc"]
    assert autoc["confidence_interval_95"][0] < 0 < autoc["confidence_interval_95"][1]


def _noise_free_overlap_example():
    propensity = np.array([0.005, 0.015, 0.03, 0.07, 0.2, 0.5, 0.8, 0.95])
    tau = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
    treatment = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    mu0 = np.zeros(8)
    mu1 = tau
    outcome = mu0 + treatment * tau
    return outcome, treatment, propensity, mu0, mu1, tau


def test_overlap_weighted_ate_on_known_example():
    outcome, treatment, propensity, mu0, mu1, tau = _noise_free_overlap_example()
    result = diagnostics.overlap_weighted_effects(outcome, treatment, propensity, mu0, mu1)

    tilt = propensity * (1 - propensity)
    assert result["augmented"]["estimate"] == pytest.approx(np.sum(tilt * tau) / np.sum(tilt))
    treated = treatment == 1
    # Controls have Y = 0, so the Hajek ATO is the (1 - e)-weighted treated mean.
    expected_hajek = np.sum((1 - propensity[treated]) * tau[treated]) / np.sum(
        1 - propensity[treated]
    )
    assert result["hajek"]["estimate"] == pytest.approx(expected_hajek)
    # Equal propensities give equal overlap weights: ESS equals the row count.
    flat = diagnostics.overlap_weighted_effects(
        outcome, treatment, np.full(8, 0.5), mu0, mu1
    )
    assert flat["effective_sample_size"]["total"] == pytest.approx(8.0)
    assert flat["effective_sample_size"]["fraction_of_rows"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# I/O layer


def _write_fold(root: Path, fold: int, rows: pd.DataFrame, fit_rows: pd.DataFrame, *,
                with_candidates: bool) -> None:
    fold_dir = root / f"outer_{fold:03d}"
    (fold_dir / "extraction" / "heldout").mkdir(parents=True)
    (fold_dir / "extraction" / "fit").mkdir(parents=True)
    (fold_dir / "estimation").mkdir(parents=True)
    retained = [
        {"name": "ecog", "feature_id": "f_ecog", "value_type": "ordinal",
         "roles": ["confounder", "effect_modifier"]},
        {"name": "ldh", "feature_id": "f_ldh", "value_type": "continuous",
         "roles": ["effect_modifier"]},
    ]
    (fold_dir / "final_definitions.json").write_text(
        json.dumps({"features": retained}), encoding="utf-8"
    )
    heldout = pd.DataFrame(
        {
            "_oci_row_id": rows["_oci_row_id"],
            "ecog": np.where(rows["x"] > 0, "2", "0"),
            "ldh": (250 + 40 * rows["x"]).round(1),
        }
    )
    heldout.to_csv(fold_dir / "extraction" / "heldout" / "harmonized.csv", index=False)
    fit = pd.DataFrame(
        {
            "_oci_row_id": fit_rows["_oci_row_id"],
            "ecog": np.where(fit_rows["x"] > 0, "2", "0"),
            "ldh": (250 + 40 * fit_rows["x"]).round(1),
        }
    )
    fit.to_csv(fold_dir / "extraction" / "fit" / "extracted.csv", index=False)
    fit_rows[["_oci_row_id", "treatment", "outcome", "propensity", "mu0", "mu1"]].to_csv(
        fold_dir / "estimation" / "nuisance_fit_predictions.csv", index=False
    )
    if with_candidates:
        candidates = [
            *retained,
            {"name": "smoking", "feature_id": "f_smoking", "value_type": "binary",
             "roles": ["confounder"]},
        ]
        (fold_dir / "estimation" / "nuisance_candidate_definitions.json").write_text(
            json.dumps({"features": candidates}), encoding="utf-8"
        )
        # Strongly treatment-associated, but absent from the propensity model.
        flip = np.random.default_rng(fold).random(len(fit_rows)) < 0.2
        smoking = np.where((fit_rows["treatment"].to_numpy() == 1) ^ flip, "yes", "no")
        fit.assign(smoking=smoking).to_csv(
            fold_dir / "estimation" / "nuisance_candidate_fit.csv", index=False
        )


def _run_directory(tmp_path: Path, *, with_candidates: bool):
    predictions = _predictions(heterogeneity=1.0, informative_cate=True, n=300, seed=40)
    for fold in (1, 2, 3):
        rows = predictions[predictions["outer_fold"] == fold]
        fit_rows = predictions[predictions["outer_fold"] != fold]
        _write_fold(tmp_path, fold, rows, fit_rows, with_candidates=with_candidates)
    # Oracle columns in the prediction table must never be consumed.
    predictions["true_ite"] = 99.0
    return predictions.drop(columns=["x"])


def test_write_heterogeneity_diagnostics_from_fold_artifacts(tmp_path: Path):
    predictions = _run_directory(tmp_path, with_candidates=False)
    path, summary = diagnostics.write_heterogeneity_diagnostics(
        output_dir=tmp_path, predictions=predictions, propensity_clip=0.02, seed=3,
        n_bootstrap=50,
    )
    assert path == tmp_path / "heterogeneity_diagnostics.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    text = path.read_text(encoding="utf-8")
    assert "true_ite" not in text
    assert payload["schema_version"] == diagnostics.SCHEMA_VERSION
    assert payload["oracle_columns_consumed"] is False
    for name in ("blp", "gates", "clan", "rate", "overlap_weighted_ate", "ipw_weights",
                 "trimming_curve", "balance"):
        assert isinstance(payload[name]["interpretation"], str) and payload[name]["interpretation"]
    for name in ("blp", "gates", "clan", "rate", "overlap_weighted_ate", "ipw_weights",
                 "trimming_curve"):
        assert set(payload[name]["per_fold"]) == {"1", "2", "3"}
        assert payload[name]["pooled"]["status"] == "ok", name

    clan = payload["clan"]["pooled"]
    ecog = {item["column"]: item for item in clan["variables"]}["ecog=2"]
    # tau = 0.5 + x and ecog = 2 when x > 0: the top quintile is all ecog 2.
    assert ecog["top_mean"] == pytest.approx(1.0)
    assert ecog["bottom_mean"] == pytest.approx(0.0)
    assert payload["clan"]["per_fold"]["1"]["variable_scope"] == (
        "retained_effect_modifiers_and_confounders"
    )

    training = payload["balance"]["training"]
    assert training["status"] == "ok"
    assert training["candidate_scope"] == "retained_features_only"
    assert training["per_fold"]["1"]["complete"] is False
    assert "nuisance_candidate_fit.csv" in (
        training["per_fold"]["1"]["sources"]["full_candidate_measurements_missing"]
    )
    assert training["pooled_summary"]["status"] == "ok"
    heldout = payload["balance"]["heldout"]
    assert heldout["status"] == "ok" and heldout["pooled"]["status"] == "ok"

    assert summary["path"] == str(path)
    assert summary["blp_pooled"]["status"] == "ok"
    assert summary["blp_pooled"]["beta2_heterogeneity"] > 0.5
    assert len(summary["trimming_curve_pooled"]) == len(diagnostics.TRIMMING_THRESHOLDS)
    assert summary["balance_training"]["candidate_scope"] == "retained_features_only"
    json.dumps(summary, allow_nan=False)
