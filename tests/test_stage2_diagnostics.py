"""Stage 2 heterogeneity, overlap and balance diagnostics on simulated data."""

from __future__ import annotations

import json
import math
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


def test_constant_cate_folds_are_unavailable_not_errors():
    predictions = _predictions(heterogeneity=0.0, informative_cate=False, n=200)
    predictions.loc[predictions["outer_fold"] == 2, "estimated_cate"] = 0.3
    result = diagnostics.build_heterogeneity_diagnostics(
        predictions, propensity_clip=0.02, seed=1, n_bootstrap=20
    )

    assert result["inputs"]["folds_with_constant_cate"] == ["2"]
    blp_fold = result["blp"]["per_fold"]["2"]
    assert blp_fold["status"] == "unavailable"
    assert "constant" in blp_fold["reason"]
    assert blp_fold["beta1_ate"] is not None
    assert result["blp"]["pooled"]["folds"] == ["1", "3"]
    assert result["gates"]["per_fold"]["2"]["status"] == "unavailable"
    assert result["rate"]["per_fold"]["2"]["status"] == "unavailable"
    assert result["gates"]["pooled"]["folds"] == ["1", "3"]
    assert result["rate"]["pooled"]["folds"] == ["1", "3"]


def test_all_constant_cate_marks_pooled_heterogeneity_unavailable():
    predictions = _predictions(heterogeneity=0.0, informative_cate=False, n=100)
    predictions["estimated_cate"] = 0.25
    result = diagnostics.build_heterogeneity_diagnostics(
        predictions, propensity_clip=0.02, seed=1, n_bootstrap=10
    )
    for name in ("blp", "gates", "rate", "clan"):
        assert result[name]["pooled"]["status"] == "unavailable"
    assert result["overlap_weighted_ate"]["pooled"]["status"] == "ok"
    json.dumps(result, allow_nan=False)


def test_rank_weighted_ate_matches_hand_computation_and_averages_ties():
    autoc, qini = diagnostics.rank_weighted_ate_point([4.0, 3.0, 2.0, 1.0], [4, 3, 2, 1])
    # TOC = cumulative means (4, 3.5, 3, 2.5) minus the mean 2.5.
    assert autoc == pytest.approx((1.5 + 1.0 + 0.5 + 0.0) / 4)
    assert qini == pytest.approx((0.25 * 1.5 + 0.5 * 1.0 + 0.75 * 0.5) / 4)
    # Reversed ranking gives the negative TOC.
    autoc_reversed, _ = diagnostics.rank_weighted_ate_point([4.0, 3.0, 2.0, 1.0], [1, 2, 3, 4])
    assert autoc_reversed == pytest.approx(-0.75)
    # Within a tie block the order is irrelevant: psi is averaged.
    tied, _ = diagnostics.rank_weighted_ate_point([4.0, 0.0, 2.0, 2.0], [2, 2, 1, 1])
    assert tied == pytest.approx(0.0)
    first = diagnostics.rank_weighted_average_treatment_effects(
        np.arange(40.0), np.arange(40.0), n_bootstrap=30, seed=5
    )
    second = diagnostics.rank_weighted_average_treatment_effects(
        np.arange(40.0), np.arange(40.0), n_bootstrap=30, seed=5
    )
    assert first == second


def test_cate_groups_are_equal_sized_and_ordered():
    cate = np.r_[np.zeros(6), np.linspace(1, 2, 14)]
    groups = diagnostics.cate_groups(cate, groups=5, seed=3)
    assert np.bincount(groups)[1:].tolist() == [4, 4, 4, 4, 4]
    for low, high in zip(range(1, 5), range(2, 6)):
        assert cate[groups == low].max() <= cate[groups == high].min()


def _noise_free_overlap_example():
    propensity = np.array([0.005, 0.015, 0.03, 0.07, 0.2, 0.5, 0.8, 0.95])
    tau = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
    treatment = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    mu0 = np.zeros(8)
    mu1 = tau
    outcome = mu0 + treatment * tau
    return outcome, treatment, propensity, mu0, mu1, tau


def test_trimming_curve_on_known_overlap_example():
    outcome, treatment, propensity, mu0, mu1, _tau = _noise_free_overlap_example()
    curve = diagnostics.aipw_trimming_curve(
        outcome, treatment, propensity, mu0, mu1, clip=0.02
    )["curve"]

    # Noise-free outcomes make psi equal tau, so each point is a mean of tau.
    assert [point["alpha"] for point in curve] == [0.0, 0.01, 0.025, 0.05, 0.1, 0.15]
    assert [point["rows"] for point in curve] == [8, 7, 6, 5, 3, 3]
    assert [point["estimate"] for point in curve] == pytest.approx([45, 50, 55, 60, 60, 60])
    assert curve[3]["propensity_range"] == [0.05, 0.95]  # inclusive bounds keep e = 0.95
    assert curve[4]["treated_rows"] == 2 and curve[4]["control_rows"] == 1


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


def test_overlap_weighted_ate_recovers_population_ato_with_poor_overlap():
    rng = np.random.default_rng(3)
    n = 200_000
    x = rng.normal(size=n)
    propensity = expit(2.5 * x)
    treatment = rng.binomial(1, propensity)
    tau = 1.0 + 2.0 * np.abs(x)
    mu0 = 0.5 * x
    mu1 = mu0 + tau
    outcome = mu0 + treatment * tau + rng.normal(size=n)
    truth = np.sum(propensity * (1 - propensity) * tau) / np.sum(propensity * (1 - propensity))
    result = diagnostics.overlap_weighted_effects(outcome, treatment, propensity, mu0, mu1)
    for key in ("hajek", "augmented"):
        estimate = result[key]
        assert abs(estimate["estimate"] - truth) < 4 * estimate["standard_error"]
        assert estimate["standard_error"] < 0.05
    # The overlap population differs from the ATE population here.
    assert truth < tau.mean() - 0.3
    weights = diagnostics.inverse_propensity_weight_summary(treatment, propensity, clip=0.02)
    assert weights["kish_effective_sample_size_fraction"] < 0.5
    assert weights["max_weight"] == pytest.approx(50.0)
    assert weights["max_weight_unclipped"] > 50.0
    assert weights["rows_clipped_low"] > 0 and weights["rows_clipped_high"] > 0


def test_inverse_propensity_weight_summary_hand_example():
    treatment = np.array([1, 1, 1, 1, 0, 0])
    propensity = np.array([0.5, 0.5, 0.5, 0.1, 0.5, 0.5])
    result = diagnostics.inverse_propensity_weight_summary(treatment, propensity, clip=0.02)
    # Treated weights 2, 2, 2, 10: ESS = 16^2 / 112; controls 2, 2: ESS = 2.
    assert result["by_arm"]["treated"]["kish_effective_sample_size"] == pytest.approx(256 / 112)
    assert result["by_arm"]["control"]["kish_effective_sample_size"] == pytest.approx(2.0)
    assert result["kish_effective_sample_size"] == pytest.approx(256 / 112 + 2.0)
    assert result["max_weight"] == pytest.approx(10.0)
    assert result["by_arm"]["treated"]["max_weight_share"] == pytest.approx(10 / 16)


def _balance_frame(n: int, seed: int, *, confounded: bool):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    true_propensity = expit(1.0 * x + 1.0 * z) if confounded else np.full(n, 0.5)
    treatment = rng.binomial(1, true_propensity)
    frame = pd.DataFrame(
        {
            "_oci_row_id": np.arange(n),
            "age_years": np.where(rng.random(n) < 0.1, None, (60 + 8 * x).round(1).astype(str)),
            "risk_band": np.where(z > 0, "high", "low"),
            "unrelated": rng.choice(["a", "b", "c"], size=n),
        }
    )
    definitions = [
        {"name": "age_years", "feature_id": "f_age", "value_type": "continuous",
         "roles": ["confounder"]},
        {"name": "risk_band", "feature_id": "f_risk", "value_type": "binary",
         "roles": ["confounder"]},
        {"name": "unrelated", "feature_id": "f_unrelated", "value_type": "categorical",
         "roles": ["effect_modifier"]},
    ]
    return frame, definitions, treatment, true_propensity, x


def test_weighted_smd_balanced_example_flags_nothing():
    frame, definitions, treatment, propensity, _x = _balance_frame(20_000, 1, confounded=False)
    encoded, meta = diagnostics.encode_measurements(frame, definitions)
    result = diagnostics.weighted_standardized_mean_differences(
        encoded, meta, treatment, propensity, clip=0.02
    )
    assert result["columns_above_threshold_unweighted"] == 0
    assert result["columns_above_threshold_weighted"] == 0
    assert result["above_threshold_weighted"] == []
    assert result["max_abs_smd_weighted"] < 0.05


def test_weighted_smd_imbalanced_example_flags_unbalanced_confounder():
    frame, definitions, treatment, true_propensity, x = _balance_frame(
        40_000, 2, confounded=True
    )
    encoded, meta = diagnostics.encode_measurements(frame, definitions)

    correct = diagnostics.weighted_standardized_mean_differences(
        encoded, meta, treatment, true_propensity, clip=0.02
    )
    by_column = {item["column"]: item for item in correct["variables"]}
    assert abs(by_column["age_years"]["smd_unweighted"]) > 0.3
    assert abs(by_column["risk_band=high"]["smd_unweighted"]) > 0.3
    assert correct["columns_above_threshold_weighted"] == 0
    assert correct["columns_above_threshold_unweighted"] >= 3

    # A propensity model that omits z balances x but leaves the risk band imbalanced.
    # E_z[expit(x + z)] is approximately expit(0.85 x) for z ~ N(0, 1).
    omitted = expit(0.85 * x)
    misspecified = diagnostics.weighted_standardized_mean_differences(
        encoded, meta, treatment, omitted, clip=0.02
    )
    assert misspecified["features_above_threshold_weighted"] == ["risk_band"]
    flagged = {item["column"] for item in misspecified["above_threshold_weighted"]}
    assert flagged == {"risk_band=high", "risk_band=low"}
    by_column = {item["column"]: item for item in misspecified["variables"]}
    assert abs(by_column["age_years"]["smd_weighted"]) < 0.1


def test_encode_measurements_levels_missingness_and_oracle_guard():
    frame = pd.DataFrame(
        {
            "_oci_row_id": [0, 1, 2, 3, 4],
            "ecog": ["0", "1", None, "1", "2"],
            "ldh": ["250", "not measured", None, "400.5", "310"],
            "histology": ["a", "b", "c", "d", "e"],
            "true_ite": ["1", "2", "3", "4", "5"],
        }
    )
    definitions = [
        {"name": "ecog", "value_type": "ordinal", "roles": ["confounder"]},
        {"name": "ldh", "value_type": "continuous", "roles": ["effect_modifier"]},
        {"name": "histology", "value_type": "categorical", "roles": ["confounder"]},
        {"name": "true_ite", "value_type": "continuous", "roles": ["effect_modifier"]},
        {"name": "absent", "value_type": "binary", "roles": ["confounder"]},
    ]
    encoded, meta = diagnostics.encode_measurements(frame, definitions, max_levels=3)
    columns = list(encoded.columns)
    assert "true_ite" not in {item["feature"] for item in meta}
    assert columns[:4] == ["ecog=1", "ecog=0", "ecog=2", "ecog=not_documented"]
    assert encoded["ecog=1"].tolist() == [0, 1, 0, 1, 0]
    assert encoded["ecog=not_documented"].tolist() == [0, 0, 1, 0, 0]
    ldh = encoded["ldh"].tolist()
    assert ldh[0] == 250 and ldh[3] == 400.5 and math.isnan(ldh[1]) and math.isnan(ldh[2])
    assert encoded["ldh=not measured"].tolist() == [0, 1, 0, 0, 0]
    assert encoded["ldh=not_documented"].tolist() == [0, 0, 1, 0, 0]
    assert [c for c in columns if c.startswith("histology")] == [
        "histology=a", "histology=b", "histology=c", "histology=other"
    ]
    assert encoded["histology=other"].tolist() == [0, 0, 0, 1, 1]
    kinds = {item["column"]: item["kind"] for item in meta}
    assert kinds["ldh"] == "continuous" and kinds["ecog=1"] == "indicator"


def test_stack_encoded_fills_unobserved_levels_with_zero_and_unmeasured_with_nan():
    first = pd.DataFrame({"risk": ["high", "high"]})
    second = pd.DataFrame({"risk": ["low", "high"], "age": ["50", "60"]})
    third = pd.DataFrame({"age": ["70", "80"]})
    definitions = [
        {"name": "risk", "value_type": "binary"},
        {"name": "age", "value_type": "continuous"},
    ]
    stacked, meta = diagnostics.stack_encoded(
        [diagnostics.encode_measurements(frame, definitions) for frame in (first, second, third)]
    )
    assert stacked["risk=low"].tolist()[:4] == [0, 0, 1, 0]
    assert np.isnan(stacked["risk=low"].tolist()[4:]).all()
    assert np.isnan(stacked["age"].tolist()[:2]).all()
    assert {item["column"] for item in meta} == {"risk=high", "risk=low", "age"}


def test_inverse_variance_pool_weights_precise_folds():
    pooled = diagnostics.inverse_variance_pool({1: (1.0, 0.1), 2: (2.0, 0.2), 3: (5.0, None)})
    weights = np.array([100.0, 25.0])
    assert pooled["estimate"] == pytest.approx((100 * 1.0 + 25 * 2.0) / weights.sum())
    assert pooled["standard_error"] == pytest.approx(1 / math.sqrt(weights.sum()))
    assert pooled["folds"] == ["1", "2"] and pooled["excluded_folds"] == ["3"]


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


def test_full_candidate_balance_flags_unadjusted_candidate(tmp_path: Path):
    predictions = _run_directory(tmp_path, with_candidates=True)
    path, summary = diagnostics.write_heterogeneity_diagnostics(
        output_dir=tmp_path, predictions=predictions, propensity_clip=0.02, seed=3,
        n_bootstrap=20,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    training = payload["balance"]["training"]
    assert training["candidate_scope"] == "all_nuisance_candidates"
    assert training["per_fold"]["2"]["complete"] is True
    # "smoking" tracks treatment but is not in the propensity model: imbalanced.
    assert "smoking" in training["pooled_summary"]["features_above_threshold_in_any_fold"]
    assert "smoking" in summary["balance_training"]["features_above_threshold_in_any_fold"]
    # Held-out candidate measurements were not persisted: retained-only fallback.
    assert payload["balance"]["heldout"]["candidate_scope"] == "retained_features_only"


def test_missing_fold_artifacts_degrade_to_unavailable(tmp_path: Path):
    predictions = _predictions(heterogeneity=1.0, informative_cate=True, n=100)
    path, summary = diagnostics.write_heterogeneity_diagnostics(
        output_dir=tmp_path, predictions=predictions, propensity_clip=0.02, seed=3,
        n_bootstrap=10,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["blp"]["pooled"]["status"] == "ok"
    assert payload["clan"]["per_fold"]["1"]["status"] == "unavailable"
    assert "final_definitions.json" in payload["clan"]["per_fold"]["1"]["reason"]
    assert payload["balance"]["training"]["status"] == "unavailable"
    assert "nuisance_fit_predictions.csv" in (
        payload["balance"]["training"]["per_fold"]["1"]["reason"]
    )
    assert summary["balance_training"]["status"] == "unavailable"


def test_unexpected_failure_is_recorded_without_raising(tmp_path: Path):
    predictions = _predictions(heterogeneity=1.0, informative_cate=True, n=50).drop(
        columns=["mu1"]
    )
    path, summary = diagnostics.write_heterogeneity_diagnostics(
        output_dir=tmp_path, predictions=predictions, propensity_clip=0.02, seed=3,
        n_bootstrap=10,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "error"
    assert "mu1" in payload["reason"]
    assert summary["status"] == "error"


def test_candidate_that_perfectly_separates_treatment_is_flagged():
    treatment = np.array([1, 1, 1, 0, 0, 0])
    frame = pd.DataFrame({"marker": ["yes", "yes", "yes", "no", "no", "no"],
                          "noise": ["1", "2", "3", "1", "2", "3"]})
    definitions = [{"name": "marker", "value_type": "binary"},
                   {"name": "noise", "value_type": "continuous"}]
    encoded, meta = diagnostics.encode_measurements(frame, definitions)
    result = diagnostics.weighted_standardized_mean_differences(
        encoded, meta, treatment, np.full(6, 0.5), clip=0.02
    )
    assert result["columns_perfectly_separating_treatment"] == ["marker=no", "marker=yes"]
    assert result["features_above_threshold_weighted"] == ["marker"]
    summary = diagnostics.summarize_fold_balance({"1": result})
    assert summary["features_perfectly_separating_treatment"] == {"marker": ["1"]}
    assert summary["features_above_threshold_in_any_fold"] == ["marker"]


def test_extraction_failures_are_not_encoded_as_not_documented():
    frame = pd.DataFrame(
        {
            "ecog": ["0", None, None, "1"],
            "ecog__oci_extraction_failed": [0, 1, 0, 0],
            "ldh": ["200", None, "300", None],
        }
    )
    definitions = [
        {"name": "ecog", "value_type": "ordinal", "extraction_failure_indicator": True},
        {"name": "ldh", "value_type": "continuous"},
    ]
    encoded, meta = diagnostics.encode_measurements(frame, definitions)
    assert encoded["ecog=not_documented"].tolist() == [0, 0, 1, 0]
    assert encoded["ecog=extraction_failed"].tolist() == [0, 1, 0, 0]
    assert encoded["ecog=0"].tolist() == [1, 0, 0, 0]
    assert "ldh=extraction_failed" not in encoded
    assert {item["level"] for item in meta if item["feature"] == "ecog"} == {
        "0", "1", "not_documented", "extraction_failed"
    }


def test_fold_failure_masks_are_merged_into_measurements(tmp_path: Path):
    predictions = _run_directory(tmp_path, with_candidates=True)
    fold_dir = tmp_path / "outer_001"
    fit = pd.read_csv(fold_dir / "estimation" / "nuisance_candidate_fit.csv", dtype=str)
    failed_rows = fit["_oci_row_id"].astype(int).iloc[:5].tolist()
    fit.loc[fit.index[:5], "ldh"] = None
    fit.to_csv(fold_dir / "estimation" / "nuisance_candidate_fit.csv", index=False)
    pd.DataFrame(
        {
            "_oci_row_id": fit["_oci_row_id"].astype(int),
            "ldh": fit["_oci_row_id"].astype(int).isin(failed_rows).astype(int),
            "ecog": 0,
        }
    ).to_csv(fold_dir / "extraction" / "failure_mask_fit.csv", index=False)

    artifacts = diagnostics.load_fold_artifacts(fold_dir)
    assert "ldh__oci_extraction_failed" in artifacts.balance_training.frame
    result = diagnostics.build_heterogeneity_diagnostics(
        predictions, propensity_clip=0.02, seed=1, n_bootstrap=10,
        fold_artifacts={1: artifacts},
    )
    variables = {
        item["column"]: item for item in result["balance"]["training"]["per_fold"]["1"]["variables"]
    }
    assert variables["ldh=extraction_failed"]["treated_rows"] + variables[
        "ldh=extraction_failed"
    ]["control_rows"] == len(fit)
    assert "ldh=not_documented" not in variables
    assert result["balance"]["training"]["per_fold"]["2"]["status"] == "unavailable"
