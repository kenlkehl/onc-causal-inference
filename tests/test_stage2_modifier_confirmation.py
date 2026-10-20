"""Outer-fold held-out confirmation of selected modifiers (reporting only)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from oci.inference import stage2_diagnostics as diagnostics
from oci.inference import stage2_modifier_tests as modifier_tests
from oci.inference.stage2_elastic_net_selection import _benjamini_hochberg

N_PER_FOLD = 1500


def _continuous(name, feature_id, roles):
    return {"feature_id": feature_id, "name": name, "value_type": "continuous",
            "categories_or_unit": ["unitless"], "roles": roles}


FOLD_DEFINITIONS = {
    1: [
        _continuous("Modifier score", "o1_f1", ["effect_modifier"]),
        _continuous("noise level", "o1_f2", ["confounder", "effect_modifier"]),
        _continuous("confounder", "o1_f3", ["confounder"]),
        _continuous("oracle_flag", "o1_f4", ["effect_modifier"]),
    ],
    2: [
        _continuous("modifier_score", "o2_f1", ["effect_modifier"]),
        {"feature_id": "o2_f2", "name": "Smoking status", "value_type": "categorical",
         "categories_or_unit": ["never", "former", "current"], "roles": ["effect_modifier"]},
        _continuous("confounder", "o2_f3", ["confounder"]),
    ],
}


def _simulate_fold(rng, fold, *, binary):
    n = N_PER_FOLD
    rows = np.arange((fold - 1) * n, fold * n)
    confounder = rng.normal(size=n)
    modifier = rng.normal(size=n)
    noise = rng.normal(size=n)
    noise[rng.random(n) < 0.1] = np.nan
    smoking = rng.choice(["never", "former", "current"], size=n)
    propensity = expit(0.6 * confounder)
    treatment = rng.binomial(1, propensity).astype(float)
    if binary:
        mu0 = expit(-0.3 + 0.6 * confounder)
        mu1 = expit(-0.3 + 0.6 * confounder + 0.4 + 1.0 * modifier)
        outcome = rng.binomial(1, np.where(treatment == 1, mu1, mu0)).astype(float)
    else:
        mu0 = confounder
        mu1 = confounder + 1.0 + 1.0 * modifier
        outcome = np.where(treatment == 1, mu1, mu0) + rng.normal(size=n)
    eligible = (propensity >= 0.1) & (propensity <= 0.9)
    aipw = diagnostics.dr_scores(outcome, treatment, propensity, mu0, mu1, clip=0.02)
    predictions = pd.DataFrame({
        "_oci_row_id": rows,
        "outer_fold": fold,
        "treatment": treatment,
        "outcome": outcome,
        "propensity": propensity,
        "mu0": mu0,
        "mu1": mu1,
        "effect_eligible": eligible,
        "aipw_score": np.where(eligible, aipw, np.nan),
        "estimated_cate": np.where(eligible, mu1 - mu0 + rng.normal(scale=0.1, size=n), np.nan),
        "true_ite": mu1 - mu0,
    })
    names = {"Modifier score" if fold == 1 else "modifier_score": modifier,
             "confounder": confounder}
    if fold == 1:
        names.update({"noise level": noise, "oracle_flag": rng.normal(size=n)})
    else:
        names["Smoking status"] = smoking
    measurements = pd.DataFrame({"_oci_row_id": rows, **names})
    return predictions, measurements


def _write_run(root: Path, *, binary: bool, seed: int = 2):
    rng = np.random.default_rng(seed)
    frames = []
    for fold, definitions in FOLD_DEFINITIONS.items():
        predictions, measurements = _simulate_fold(rng, fold, binary=binary)
        fold_dir = root / f"outer_{fold:03d}"
        (fold_dir / "estimation").mkdir(parents=True)
        (fold_dir / "selection").mkdir(parents=True)
        (fold_dir / "final_definitions.json").write_text(json.dumps({"features": definitions}))
        measurements.to_csv(fold_dir / "estimation" / "nuisance_candidate_heldout.csv",
                            index=False)
        predictions.drop(columns=["outer_fold"]).to_csv(
            fold_dir / "estimation" / "predictions.csv", index=False)
        (fold_dir / "selection" / "modifier_tests.json").write_text(json.dumps({"rule": "x"}))
        frames.append(predictions)
    predictions = pd.concat(frames, ignore_index=True)
    predictions.to_csv(root / "cross_fitted_predictions.csv", index=False)
    return predictions


def _policy(binary, **overrides):
    return {"outcome_type": "binary" if binary else "continuous", "q": 0.1,
            "weights": None, "min_count": 5, **overrides}


def _direct(root, predictions, fold, *, binary):
    """The same tests called directly on the fold's frozen inputs."""
    rows = predictions[predictions.outer_fold == fold].reset_index(drop=True)
    frame = pd.read_csv(root / f"outer_{fold:03d}" / "estimation" /
                        "nuisance_candidate_heldout.csv", dtype=str)
    frame["_oci_row_id"] = frame["_oci_row_id"].astype(int)
    frame = rows[["_oci_row_id"]].merge(frame, on="_oci_row_id")
    selected = [f for f in FOLD_DEFINITIONS[fold]
                if "effect_modifier" in f["roles"] and not f["name"].startswith("oracle")]
    records, _ = modifier_tests.candidate_tests(
        frame, selected, treatment=rows.treatment, outcome=rows.outcome,
        propensity=rows.propensity, mu0=rows.mu0, mu1=rows.mu1, binary=binary,
        propensity_clip=0.02, eligible=rows.effect_eligible, min_count=5,
    )
    return modifier_tests.combine_and_select(
        records, weights=modifier_tests.default_weights("binary" if binary else "continuous"),
        q=0.1, binary=binary, family_ids=[f["feature_id"] for f in selected],
    )


@pytest.mark.parametrize("binary", [False, True])
def test_confirmation_matches_direct_tests_and_summarizes_across_folds(tmp_path, binary):
    predictions = _write_run(tmp_path, binary=binary)
    path, summary = diagnostics.write_heterogeneity_diagnostics(
        output_dir=tmp_path, predictions=predictions, propensity_clip=0.02, seed=3,
        n_bootstrap=10, modifier_confirmation_policy=_policy(binary),
    )
    text = path.read_text()
    assert "true_ite" not in text
    section = json.loads(text)["modifier_confirmation"]
    assert section["status"] == "ok"
    assert section["interpretation"] and section["pooling_rule"]
    assert section["policy"]["weights"] == modifier_tests.default_weights(
        "binary" if binary else "continuous")
    assert section["policy"]["weights_source"] == "outcome_type_default"
    for fold in (1, 2):
        result = section["per_fold"][str(fold)]
        expected, confirmed_ids = _direct(tmp_path, predictions, fold, binary=binary)
        assert result["status"] == "ok"
        assert result["heldout_rows"] == N_PER_FOLD
        assert result["eligible_rows"] == int(
            predictions[predictions.outer_fold == fold].effect_eligible.sum())
        assert result["measurement_source"]["measurements"].endswith(
            "nuisance_candidate_heldout.csv")
        assert [m["feature_id"] for m in result["modifiers"]] == [r["feature_id"] for r in expected]
        for modifier, record in zip(result["modifiers"], expected):
            assert modifier["combined_p_value"] == pytest.approx(record["combined_p_value"])
            assert modifier["combined_q_value"] == pytest.approx(record["combined_q_value"])
            assert modifier["confirmed"] is record["selected"]
            assert modifier["labels"] == record["labels"]
            assert "selected" not in modifier and "selection_rank" not in modifier
            for name in modifier_tests.COMPONENTS:
                assert modifier["component_p_values"][name] == pytest.approx(
                    record["components"][name]["p_value"])
                assert modifier["component_status"][name] == "ok"
        assert result["confirmed_modifiers"] == len(confirmed_ids)
        by_name = {m["name"]: m for m in result["modifiers"]}
        true_name = "Modifier score" if fold == 1 else "modifier_score"
        noise_name = "noise level" if fold == 1 else "Smoking status"
        assert by_name[true_name]["confirmed"] is True
        assert by_name[noise_name]["confirmed"] is False
        assert set(by_name) == {true_name, noise_name}
        if binary:
            assert "multiplicative" in by_name[true_name]["labels"]
        else:
            assert "additive" in by_name[true_name]["labels"]
    assert section["per_fold"]["1"]["excluded_oracle_named_features"] == ["oracle_flag"]
    assert section["pooled"] == {
        "folds": ["1", "2"],
        "selected_by_fold": {"1": 2, "2": 2},
        "confirmed_by_fold": {"1": 1, "2": 1},
        "total_selected": 4,
        "total_confirmed": 2,
    }
    across = section["across_folds"]
    assert across["aggregation"] == "meinshausen_meier_buhlmann_fixed_gamma"
    assert across["folds"] == ["1", "2"]
    concepts = {tuple(m["name"] for m in c["members"]): c for c in across["concepts"]}
    # Same normalized name in both folds: one concept; the noise variables stay apart.
    assert set(concepts) == {("Modifier score", "modifier_score"), ("noise level",),
                             ("Smoking status",)}
    shared = concepts[("Modifier score", "modifier_score")]
    assert (shared["folds_selected"], shared["folds_confirmed"]) == (2, 2)
    assert shared["selected_in_folds"] == shared["confirmed_in_folds"] == ["1", "2"]
    assert shared["minimum_similarity"] == 1.0
    p_values = [section["per_fold"][fold]["modifiers"][0]["combined_p_value"]
                for fold in ("1", "2")]
    assert shared["heldout_p_values_by_fold"] == dict(zip(("1", "2"), p_values))
    assert shared["aggregated_p_value"] == pytest.approx(
        diagnostics.mmb_quantile_aggregation(p_values))
    noise = concepts[("noise level",)]
    assert noise["heldout_p_values_by_fold"]["2"] is None
    assert noise["aggregated_p_value"] == pytest.approx(min(
        1.0, 2 * np.median([noise["heldout_p_values_by_fold"]["1"], 1.0])))
    q_values = _benjamini_hochberg(
        [c["aggregated_p_value"] for c in across["concepts"]])
    assert [c["aggregated_q_value"] for c in across["concepts"]] == pytest.approx(q_values)
    assert across["concepts"][0]["concept_id"] == shared["concept_id"]
    assert shared["aggregated_supported"] is True
    # Effect sizes per fold, and pooled across the two matched folds.
    for fold in ("1", "2"):
        effects = section["per_fold"][fold]["modifiers"][0]["effects"]
        for component in ("r_projection", "aipw_projection"):
            value = effects[component]["columns"][0]
            assert value["kind"] == "value" and value["standard_error"] > 0
            low, high = value["confidence_interval_95"]
            # The simulated modifier changes the absolute effect by +1.0 per unit
            # (continuous outcome), or by a positive amount (binary).
            assert low < value["coefficient"] < high
            assert value["coefficient"] > 0
    assert shared["pooled_effects"]["status"] == "ok"
    assert shared["pooled_effects"]["columns"]["value_per_training_sd"]["r_projection"][
        "status"] == "unavailable"  # this fixture stores no training measurements
    assert "p_value_role" in section and section["p_value_role"].startswith("inferential")
    # The compact summary flows into causal_estimate.json.
    compact = summary["modifier_confirmation"]
    assert compact["status"] == "ok"
    assert (compact["total_selected"], compact["total_confirmed"]) == (4, 2)
    assert compact["confirmed_by_fold"] == {"1": 1, "2": 1}
    assert compact["concepts_supported_after_aggregation"] == [shared["concept_id"]]
    assert compact["concepts"][0]["names"] == ["Modifier score", "modifier_score"]
    assert diagnostics.summarize_diagnostics(json.loads(text))["modifier_confirmation"] == compact


def test_confirmation_cannot_change_predictions_or_selections(tmp_path):
    predictions = _write_run(tmp_path, binary=True)
    before_frame = predictions.copy()

    def snapshot():
        return {
            str(path.relative_to(tmp_path)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(tmp_path.rglob("*")) if path.is_file()
            and path.name != diagnostics.DIAGNOSTICS_FILENAME
        }

    before = snapshot()
    assert "cross_fitted_predictions.csv" in before
    assert "outer_001/selection/modifier_tests.json" in before
    assert "outer_002/final_definitions.json" in before
    for policy in (None, _policy(True)):
        diagnostics.write_heterogeneity_diagnostics(
            output_dir=tmp_path, predictions=predictions, propensity_clip=0.02, seed=3,
            n_bootstrap=10, modifier_confirmation_policy=policy,
        )
        assert snapshot() == before
    pd.testing.assert_frame_equal(predictions, before_frame)


def test_missing_inputs_are_unavailable_never_a_failure(tmp_path):
    predictions = _write_run(tmp_path, binary=False)
    # Fold 2: no candidate file, but the retained held-out file is the fallback.
    fold2 = tmp_path / "outer_002"
    candidate = pd.read_csv(fold2 / "estimation" / "nuisance_candidate_heldout.csv")
    (fold2 / "estimation" / "nuisance_candidate_heldout.csv").unlink()
    (fold2 / "extraction" / "heldout").mkdir(parents=True)
    candidate.to_csv(fold2 / "extraction" / "heldout" / "harmonized.csv", index=False)
    # Fold 3: no artifacts at all. Fold 4: no selected modifiers.
    extra = []
    for fold in (3, 4):
        rows = predictions[predictions.outer_fold == 1].copy()
        rows["_oci_row_id"] += fold * N_PER_FOLD
        rows["outer_fold"] = fold
        extra.append(rows)
    (tmp_path / "outer_004").mkdir()
    (tmp_path / "outer_004" / "final_definitions.json").write_text(json.dumps(
        {"features": [_continuous("confounder", "o4_f1", ["confounder"])]}))
    # Fold 5: selected modifiers but no held-out measurement file.
    rows = predictions[predictions.outer_fold == 1].copy()
    rows["_oci_row_id"] += 5 * N_PER_FOLD
    rows["outer_fold"] = 5
    extra.append(rows)
    (tmp_path / "outer_005").mkdir()
    (tmp_path / "outer_005" / "final_definitions.json").write_text(
        json.dumps({"features": FOLD_DEFINITIONS[1]}))
    pooled = pd.concat([predictions, *extra], ignore_index=True)
    path, summary = diagnostics.write_heterogeneity_diagnostics(
        output_dir=tmp_path, predictions=pooled, propensity_clip=0.02, seed=3,
        n_bootstrap=10, modifier_confirmation_policy=_policy(False),
    )
    section = json.loads(path.read_text())["modifier_confirmation"]
    per_fold = section["per_fold"]
    assert per_fold["2"]["status"] == "ok"
    assert per_fold["2"]["measurement_source"]["measurements"].endswith("harmonized.csv")
    assert per_fold["3"]["status"] == "unavailable"
    assert "retained definitions unavailable" in per_fold["3"]["reason"]
    assert per_fold["4"] == {**per_fold["4"], "status": "ok", "selected_modifiers": 0,
                             "confirmed_modifiers": 0, "modifiers": []}
    assert per_fold["5"]["status"] == "unavailable"
    assert "held-out measurements unavailable" in per_fold["5"]["reason"]
    assert section["status"] == "ok"
    assert section["pooled"]["folds"] == ["1", "2", "4"]
    assert section["pooled"]["selected_by_fold"]["4"] == 0
    assert summary["modifier_confirmation"]["status"] == "ok"
    assert '"status": "error"' not in json.dumps(section)


def test_without_policy_or_with_invalid_policy_the_section_is_unavailable(tmp_path):
    predictions = _write_run(tmp_path, binary=True)
    for policy, reason in (
        (None, "no modifier confirmation policy"),
        (_policy(True, outcome_type="survival"), "invalid modifier confirmation policy"),
        (_policy(True, q=0), "invalid modifier confirmation policy"),
    ):
        path, summary = diagnostics.write_heterogeneity_diagnostics(
            output_dir=tmp_path, predictions=predictions, propensity_clip=0.02, seed=3,
            n_bootstrap=10, modifier_confirmation_policy=policy,
        )
        section = json.loads(path.read_text())["modifier_confirmation"]
        assert section["status"] == "unavailable"
        assert reason in section["reason"]
        assert section["interpretation"]
        assert summary["modifier_confirmation"]["status"] == "unavailable"
        assert summary["modifier_confirmation"]["concepts"] == []


def test_configured_weights_are_used(tmp_path):
    predictions = _write_run(tmp_path, binary=True)
    weights = {"r_projection": 1.0, "aipw_projection": 0.0, "outcome_interaction": 0.0}
    path, _ = diagnostics.write_heterogeneity_diagnostics(
        output_dir=tmp_path, predictions=predictions, propensity_clip=0.02, seed=3,
        n_bootstrap=10, modifier_confirmation_policy=_policy(True, weights=weights),
    )
    section = json.loads(path.read_text())["modifier_confirmation"]
    assert section["policy"]["weights_source"] == "configured"
    for result in section["per_fold"].values():
        for modifier in result["modifiers"]:
            assert modifier["combination_weights"] == {"r_projection": 1.0}
            assert modifier["combined_p_value"] == pytest.approx(
                modifier["component_p_values"]["r_projection"])
