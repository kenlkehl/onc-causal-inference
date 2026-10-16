"""Held-out effect sizes, cross-fold concept matching, and multi-split aggregation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oci.inference import stage2_diagnostics as diagnostics
from oci.inference import stage2_modifier_tests as modifier_tests


def _feature(name, *, value_type="continuous", categories=("unitless",), description="",
             definition="", fid=None):
    return {"feature_id": fid or name, "name": name, "value_type": value_type,
            "categories_or_unit": list(categories), "description": description,
            "measurement_definition": definition, "roles": ["effect_modifier"]}


def _heldout_tests(z, feature, rng, *, n, slope=0.5, categorical_effects=None):
    """Continuous outcome with tau(X) = 1 + slope * Z (or per level); oracle nuisances."""
    x = rng.normal(size=n)
    e = 1 / (1 + np.exp(-0.5 * x))
    t = rng.binomial(1, e).astype(float)
    tau = 1.0 + (slope * z if categorical_effects is None
                 else np.array([categorical_effects[v] for v in z]))
    mu0 = x
    mu1 = x + tau
    y = np.where(t == 1, mu1, mu0) + rng.normal(size=n)
    frame = pd.DataFrame({feature["name"]: z})
    records, _ = modifier_tests.candidate_tests(
        frame, [feature], treatment=t, outcome=y, propensity=e, mu0=mu0, mu1=mu1,
        binary=False, propensity_clip=0.02, min_count=5)
    return records[0]


def test_projection_effects_recover_the_slope_with_calibrated_intervals():
    rng = np.random.default_rng(1)
    feature = _feature("score")
    covered = {"r_projection": 0, "aipw_projection": 0}
    estimates = {"r_projection": [], "aipw_projection": []}
    reps = 200
    for _ in range(reps):
        z = rng.normal(size=400)
        record = _heldout_tests(z, feature, rng, n=400)
        for component in covered:
            effect = diagnostics.projection_effects(
                record["components"][component], record["grouped_columns"], training_sd=2.0)
            column = effect["columns"][0]
            assert column["kind"] == "value" and column["column"] == "score:value"
            low, high = column["confidence_interval_95"]
            covered[component] += low <= 0.5 <= high
            estimates[component].append(column["coefficient"])
            assert column["per_training_sd"]["coefficient"] == pytest.approx(
                2.0 * column["coefficient"])
    for component in covered:
        assert np.mean(estimates[component]) == pytest.approx(0.5, abs=0.03)
        assert 0.90 <= covered[component] / reps <= 0.99


def test_categorical_effects_are_per_level_versus_the_reference():
    rng = np.random.default_rng(2)
    feature = _feature("stage", value_type="categorical", categories=("a", "b", "c"))
    z = rng.choice(["a", "b", "c"], size=3000)
    record = _heldout_tests(z, feature, rng, n=3000,
                            categorical_effects={"a": 0.0, "b": 0.4, "c": 0.0})
    effect = diagnostics.projection_effects(
        record["components"]["r_projection"], record["grouped_columns"], training_sd=None)
    by_level = {c["level"]: c for c in effect["columns"]}
    assert set(by_level) == {"b", "c"} and {c["kind"] for c in effect["columns"]} == {"level"}
    assert by_level["b"]["coefficient"] == pytest.approx(0.4, abs=0.12)
    assert abs(by_level["c"]["coefficient"]) < 0.12
    assert "per_training_sd" not in by_level["b"]
    unavailable = diagnostics.projection_effects(
        {"p_value": None, "reason": "row_with_leverage_one"}, [], training_sd=None)
    assert unavailable == {"status": "unavailable", "reason": "row_with_leverage_one"}


def _entry(fold, name, description="", definition=""):
    feature = _feature(name, description=description, definition=definition)
    return {"fold": fold, "name": name, "definition_text": diagnostics._definition_text(feature)}


def test_concept_matching_groups_synonyms_keeps_distinct_concepts_apart_one_per_fold():
    entries = [
        _entry("1", "ECOG performance status", "Baseline ECOG performance status.",
               "Extract the last pretreatment ECOG score."),
        _entry("2", "performance_status", "Pretreatment ECOG performance status score.",
               "Extract the pretreatment ECOG score."),
        _entry("3", "Performance status", "ECOG performance status before treatment.",
               "Extract the pretreatment ECOG score."),
        _entry("1", "Brain metastases", "Presence of brain metastases at baseline.",
               "Extract whether brain metastases are documented."),
        _entry("2", "Liver metastases", "Presence of liver metastases at baseline.",
               "Extract whether liver metastases are documented."),
        _entry("1", "Smoking status", "Current or former tobacco smoking.",
               "Extract whether the patient smokes."),
        _entry("2", "Serum creatinine", "Baseline serum creatinine in mg/dL.",
               "Extract the last pretreatment serum creatinine value."),
        # A second performance-status variable in fold 1 cannot join the same concept.
        _entry("1", "Performance status score", "ECOG performance status before treatment.",
               "Extract the pretreatment ECOG score."),
    ]
    concepts = diagnostics.match_concepts(entries)
    groups = [sorted(entries[k]["name"] for k in c["member_indices"]) for c in concepts]
    for concept in concepts:
        folds = [entries[k]["fold"] for k in concept["member_indices"]]
        assert len(folds) == len(set(folds))
        if len(folds) > 1:
            assert concept["minimum_similarity"] >= diagnostics.CONCEPT_SIMILARITY_THRESHOLD
    performance = next(g for g in groups if "performance_status" in g)
    assert len(performance) == 3 and "Performance status" in performance
    assert ["Brain metastases"] in groups and ["Liver metastases"] in groups
    assert ["Smoking status"] in groups and ["Serum creatinine"] in groups
    # Exactly one fold-1 performance variable is grouped with folds 2 and 3.
    assert sum(name in performance for name in (
        "ECOG performance status", "Performance status score")) == 1
    identical = diagnostics.match_concepts([_entry("1", "Age"), _entry("2", "age")])
    assert len(identical) == 1 and identical[0]["minimum_similarity"] == 1.0


def test_mmb_aggregation_and_fixed_effect_pooling_arithmetic():
    assert diagnostics.mmb_quantile_aggregation([0.01, 0.02, 0.03, None, None]) == pytest.approx(0.06)
    assert diagnostics.mmb_quantile_aggregation([0.01, None, None, None, None]) == 1.0
    assert diagnostics.mmb_quantile_aggregation([0.001] * 5) == pytest.approx(0.002)
    assert diagnostics.mmb_quantile_aggregation([0.2] * 5) == pytest.approx(0.4)
    assert diagnostics.mmb_quantile_aggregation([]) == 1.0
    pooled = diagnostics.fixed_effect_pool([(1.0, 0.1), (2.0, 0.1)])
    assert pooled["estimate"] == pytest.approx(1.5)
    assert pooled["standard_error"] == pytest.approx(0.1 / np.sqrt(2))
    assert pooled["cochran_q"] == pytest.approx(50.0)
    assert pooled["i_squared"] == pytest.approx(49 / 50)
    assert pooled["fold_range"] == [1.0, 2.0]
    homogeneous = diagnostics.fixed_effect_pool([(1.0, 0.1), (1.0, 0.2), (1.0, 0.3)])
    assert homogeneous["cochran_q"] == pytest.approx(0.0) and homogeneous["i_squared"] == 0.0
    assert diagnostics.fixed_effect_pool([(1.0, 0.1)])["status"] == "unavailable"


def _member(level_columns):
    columns = [{"kind": kind, "level": level, "coefficient": 0.3 + i * 0.1, "standard_error": 0.1,
                **({"per_training_sd": {"coefficient": 0.6, "standard_error": 0.2}}
                   if kind == "value" else {})}
               for i, (kind, level) in enumerate(level_columns)]
    return {"effects": {"r_projection": {"columns": columns},
                        "aipw_projection": {"columns": columns}}}


def test_pooled_effects_require_matching_columns():
    same = diagnostics._pooled_concept_effects(
        [_member([("level", "B")]), _member([("level", "b")])])
    assert same["status"] == "ok"
    assert same["columns"]["level=b"]["r_projection"]["estimate"] == pytest.approx(0.3)
    continuous = diagnostics._pooled_concept_effects([_member([("value", None)])] * 3)
    assert continuous["columns"]["value_per_training_sd"]["aipw_projection"]["estimate"] == (
        pytest.approx(0.6))
    different = diagnostics._pooled_concept_effects(
        [_member([("level", "b")]), _member([("level", "b"), ("level", "c")])])
    assert different["status"] == "unavailable" and "per fold only" in different["reason"]
    mixed = diagnostics._pooled_concept_effects(
        [_member([("value", None)]), _member([("level", "b")])])
    assert mixed["status"] == "unavailable"
    assert diagnostics._pooled_concept_effects([_member([("value", None)])])["status"] == (
        "unavailable")


def _write_three_fold_run(root: Path, *, seed=3, n=1500):
    """Three folds whose selected modifier is named differently in each; training SDs stored."""
    rng = np.random.default_rng(seed)
    names = {1: "Modifier score", 2: "modifier_score", 3: "Score of the modifier"}
    frames = []
    for fold, name in names.items():
        rows = np.arange((fold - 1) * n, fold * n)
        z, x = rng.normal(size=n), rng.normal(size=n)
        e = 1 / (1 + np.exp(-0.5 * x))
        t = rng.binomial(1, e).astype(float)
        mu0, mu1 = x, x + 1.0 + 0.5 * z
        y = np.where(t == 1, mu1, mu0) + rng.normal(size=n)
        frames.append(pd.DataFrame({
            "_oci_row_id": rows, "outer_fold": fold, "treatment": t, "outcome": y,
            "propensity": e, "mu0": mu0, "mu1": mu1, "effect_eligible": True,
            "aipw_score": diagnostics.dr_scores(y, t, e, mu0, mu1, clip=0.02),
            "estimated_cate": mu1 - mu0,
        }))
        fold_dir = root / f"outer_{fold:03d}"
        (fold_dir / "estimation").mkdir(parents=True)
        definitions = [_feature(name, fid=f"o{fold}", description="A modifier score.",
                                definition="Extract the score.")]
        (fold_dir / "final_definitions.json").write_text(json.dumps({"features": definitions}))
        pd.DataFrame({"_oci_row_id": rows, name: z}).to_csv(
            fold_dir / "estimation" / "nuisance_candidate_heldout.csv", index=False)
        pd.DataFrame({"_oci_row_id": rows + 10 * n, name: 2.0 * rng.normal(size=n)}).to_csv(
            fold_dir / "estimation" / "nuisance_candidate_fit.csv", index=False)
    predictions = pd.concat(frames, ignore_index=True)
    predictions.to_csv(root / "cross_fitted_predictions.csv", index=False)
    return predictions


def test_section_reports_effects_concepts_and_pooled_effects(tmp_path):
    predictions = _write_three_fold_run(tmp_path)
    snapshot = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    _, summary = diagnostics.write_heterogeneity_diagnostics(
        output_dir=tmp_path, predictions=predictions, propensity_clip=0.02, seed=1,
        n_bootstrap=10, modifier_confirmation_policy={
            "outcome_type": "continuous", "q": 0.1, "weights": None, "min_count": 5})
    assert {p: p.read_bytes() for p in snapshot} == snapshot  # nothing else changes
    section = json.loads((tmp_path / diagnostics.DIAGNOSTICS_FILENAME).read_text())[
        "modifier_confirmation"]
    for fold in ("1", "2", "3"):
        modifier = section["per_fold"][fold]["modifiers"][0]
        effects = modifier["effects"]
        assert effects["scale"] == "outcome difference per unit"
        assert effects["training_sd"] == pytest.approx(2.0, rel=0.1)
        value = effects["r_projection"]["columns"][0]
        assert value["coefficient"] == pytest.approx(0.5, abs=0.15)
        assert value["per_training_sd"]["coefficient"] == pytest.approx(
            value["coefficient"] * effects["training_sd"])
    concepts = section["across_folds"]["concepts"]
    # Folds 1 and 2 share a normalized name; fold 3's wording is matched lexically
    # only if it clears the conservative threshold.
    first = concepts[0]
    assert {"1", "2"} <= set(first["selected_in_folds"])
    pooled = first["pooled_effects"]["columns"]["value_per_training_sd"]["r_projection"]
    assert pooled["status"] == "ok" and pooled["folds"] == first["folds_selected"]
    assert pooled["estimate"] == pytest.approx(1.0, abs=0.25)  # 0.5 per unit x SD 2
    assert {"cochran_q", "i_squared", "fold_range"} <= set(pooled)
    assert first["aggregated_p_value"] == pytest.approx(diagnostics.mmb_quantile_aggregation(
        list(first["heldout_p_values_by_fold"].values())))
    assert summary["modifier_confirmation"]["concepts_supported_after_aggregation"]
