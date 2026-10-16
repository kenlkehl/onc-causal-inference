"""The reserved documented_unknown measurement state in Stage 2."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oci.inference import plain_handoff_stage2_analysis as analysis
from oci.inference import stage2_clinical_prompts as clinical_prompts
from oci.inference import stage2_diagnostics as diagnostics
from oci.inference import stage2_documented_unknown as du
from oci.inference.stage2_elastic_net_selection import _encode_design
from oci.inference.stage2_extraction_failures import extraction_failure_column
from oci.inference.stage2_statistical_selection import _encode_feature

UNKNOWN = du.DOCUMENTED_UNKNOWN


def _definition(name, value_type, categories=(), **extra):
    return {
        "feature_id": f"f_{name}",
        "name": name,
        "description": f"{name} status.",
        "value_type": value_type,
        "categories_or_unit": list(categories),
        "measurement_definition": f"Extract {name}.",
        "missing_value_rule": "Return null when undocumented.",
        **extra,
    }


DEFINITIONS = [
    _definition("egfr", "binary", ["Mutant", "Wild-type"]),
    _definition("histology", "categorical", ["Adenocarcinoma", "Squamous", "Other"]),
    _definition("ecog", "ordinal", ["0", "1", "2"]),
    _definition("albumin", "continuous", ["g/dL"]),
    _definition("free_text", "ambiguous"),
]


def _validate(values, definitions=DEFINITIONS):
    return analysis._validate_extraction(
        {"rows": [{"row_id": 0, "values": values}]}, row_ids=[0], definitions=definitions,
    )["rows"][0]["values"]


# ---------------------------------------------------------------------------
# Validators


@pytest.mark.parametrize("answer", [UNKNOWN, " Documented_Unknown ", "DOCUMENTED_UNKNOWN"])
def test_every_value_type_accepts_the_reserved_value(answer):
    values = _validate({definition["name"]: answer for definition in DEFINITIONS})
    assert values == {definition["name"]: UNKNOWN for definition in DEFINITIONS}


@pytest.mark.parametrize("malformed", [
    "documented unknown", "documented-unknown", "documented_unknown: pending",
    "Documented unknown (EGFR)",
])
def test_malformed_variants_are_rejected_for_repair(malformed):
    for definition in DEFINITIONS:
        with pytest.raises(analysis._ExtractionValueError, match="exactly as documented_unknown"):
            _validate({definition["name"]: malformed}, definitions=[definition])


def test_unknown_like_answers_map_conservatively():
    egfr, histology, ecog, albumin, _ = DEFINITIONS
    values = _validate({"egfr": "Pending", "histology": "indeterminate", "ecog": "Not assessed",
                        "albumin": "result pending", "free_text": "Equivocal"})
    assert set(values.values()) == {UNKNOWN}
    # A bare "unknown" stays a missing-value placeholder.
    assert _validate({"egfr": "unknown"}, definitions=[egfr]) == {"egfr": None}
    # A phrase containing "unknown" is not an unknown-like whole value.
    primary = _definition("primary_site", "categorical", ["Lung", "Unknown primary"])
    assert _validate({"primary_site": "Unknown primary"}, definitions=[primary]) == {
        "primary_site": "Unknown primary"}
    # A declared genuine answer is kept as a category.
    assessed = _definition("pdl1_assessed", "binary", ["Assessed", "Not assessed"])
    assert _validate({"pdl1_assessed": "not assessed"}, definitions=[assessed]) == {
        "pdl1_assessed": "Not assessed"}
    # Other undeclared phrases still raise a category error.
    with pytest.raises(analysis._ExtractionCategoryError, match="documented_unknown, or null"):
        _validate({"egfr": "exon 20"}, definitions=[egfr])


def test_category_mapping_accepts_the_reserved_value():
    items = [{"mapping_id": "category_mapping_0001", "allowed_categories": ["Mutant", "Wild-type"]}]
    for answer in (UNKNOWN, "Not assessed"):
        result = analysis._validate_category_ontology({"value": answer}, items=items)
        assert result["corrections"][0]["value"] == UNKNOWN
    with pytest.raises(ValueError, match="documented_unknown, or null"):
        analysis._validate_category_ontology({"value": "exon 20"}, items=items)


def test_definitions_drop_unknown_like_categories_only_when_the_rest_stays_valid():
    clean = analysis._validated_closed_category_values
    assert clean(value_type="binary", values=["Mutant", "Wild-type", "Unknown"],
                 source="t") == ["Mutant", "Wild-type"]
    assert clean(value_type="categorical", values=["Low", "High", "Not assessed", "Pending"],
                 source="t") == ["Low", "High"]
    # Genuine answers are kept: removing them would break the ontology.
    assert clean(value_type="binary", values=["Assessed", "Not assessed"],
                 source="t") == ["Assessed", "Not assessed"]
    assert clean(value_type="categorical", values=["Lung", "Unknown primary", "Breast"],
                 source="t") == ["Lung", "Unknown primary", "Breast"]
    # The reserved value itself can never be a category.
    with pytest.raises(ValueError, match="exactly two distinct"):
        clean(value_type="binary", values=["Present", UNKNOWN], source="t")
    assert analysis._declared_categories(
        {"value_type": "categorical", "categories_or_unit": ["a", "b", UNKNOWN]}) == ["a", "b"]


def test_explicit_features_keep_declared_unknown_like_categories():
    from oci.inference.plain_handoff_stage2 import Stage2ExplicitFeature

    feature = Stage2ExplicitFeature(
        name="egfr",
        description="EGFR mutation status.",
        value_type="categorical",
        categories_or_unit=("Mutant", "Wild-type", "Unknown"),
        measurement_definition="Extract EGFR status.",
        missing_value_rule="Return null when undocumented.",
        roles=("effect_modifier",),
    )
    assert list(feature.categories_or_unit) == ["Mutant", "Wild-type", "Unknown"]
    with pytest.raises(ValueError, match="reserved value"):
        Stage2ExplicitFeature(
            name="egfr",
            description="EGFR mutation status.",
            value_type="categorical",
            categories_or_unit=("Mutant", "Wild-type", UNKNOWN),
            measurement_definition="Extract EGFR status.",
            missing_value_rule="Return null when undocumented.",
            roles=("effect_modifier",),
        )
    locked = _definition(
        "egfr", "categorical", ["Mutant", "Wild-type", "Unknown"],
        configured_explicit_feature=True,
    )
    discovered = _definition("egfr", "categorical", ["Mutant", "Wild-type", "Unknown"])
    prepared_locked, prepared_discovered = analysis.initial_feature_modeling_definitions(
        [locked, discovered]
    )
    assert prepared_locked["categories_or_unit"] == ["Mutant", "Wild-type", "Unknown"]
    assert prepared_discovered["categories_or_unit"] == ["Mutant", "Wild-type"]
    # A declared category wins over the reserved state.
    assert _validate({"egfr": "Unknown"}, [prepared_locked]) == {"egfr": "Unknown"}
    assert _validate({"egfr": "Unknown"}, [prepared_discovered]) == {"egfr": None}


def test_prompts_describe_the_reserved_value():
    prompts = clinical_prompts.SYSTEM_PROMPTS
    phrase = "unknown, indeterminate, pending, not assessed, or equivocal"
    for slug in ("04_define_ontology", "05_extract_patient", "06_serial_chunk",
                 "07_page_observations", "08_map_categories", "09_refine_ontology",
                 "12_supervise_ontology"):
        assert "documented_unknown" in prompts[slug] and phrase in prompts[slug], slug
    assert "do not add categories such as Unknown" in prompts["04_define_ontology"]
    assert "documented_unknown" in prompts["ontology_scalar_contract"]
    messages = analysis._extraction_prompt(
        definitions=[DEFINITIONS[0]], rows=[{"row_id": 0, "text": "EGFR testing pending."}])
    assert "documented_unknown when the record states" in messages[1]["content"]
    page = analysis._page_extraction_prompt(definitions=[DEFINITIONS[0]], row={"text": "x"})
    assert "documented_unknown when the text states" in page[1]["content"]
    assert clinical_prompts.PROMPT_VERSION.startswith("stage2_clinical_prompts_v2")


# ---------------------------------------------------------------------------
# Mode resolution and harmonization


def _observation(index, value, date=None):
    return {"observation_id": f"o{index}", "value": value, "page_index": 1,
            "observation_index": index, "recorded_at": date}


def test_documented_values_take_precedence_over_unknown_observations():
    definition = {**DEFINITIONS[0], "conflict_resolution": {"strategy": "latest"}}
    observations = [_observation(1, "Mutant", "2024-01-01T00:00:00"),
                    _observation(2, UNKNOWN, "2024-02-01T00:00:00")]
    value, decision = analysis._resolve_feature_observations(
        definition=definition, observations=observations)
    assert value == "Mutant" and decision["documented_unknown_observations"] == 1
    value, decision = analysis._resolve_feature_observations(
        definition=definition, observations=[_observation(1, UNKNOWN), _observation(2, UNKNOWN)])
    assert value == UNKNOWN and decision["resolution"] == "documented_unknown_only"


def test_harmonization_passes_the_reserved_value_through():
    feature = DEFINITIONS[3]
    frame = pd.DataFrame({"albumin": [3.1, 4.2, UNKNOWN, None]})
    assert analysis._mixed_value_observations(frame, feature) is None
    mixed = pd.DataFrame({"albumin": [3.1, "<2.0", UNKNOWN]})
    observations = analysis._mixed_value_observations(mixed, feature)
    assert [row["raw_value"] for row in observations["categorical_values"]] == ["<2.0"]
    plan = {"target_representation": "continuous", "categorical_value_map": []}
    harmonized, audit = analysis._apply_one_harmonization_plan(mixed["albumin"], plan)
    assert harmonized.tolist() == [3.1, None, UNKNOWN]
    summary = analysis.feature_summaries(frame, [{**feature, "feature_id": "f"}])[0]
    assert summary["documented_unknown"] == 1
    assert summary["categorical_fallback_nonmissing"] == 0
    assert summary["recommended_modeling_strategy"] == "continuous"


# ---------------------------------------------------------------------------
# Encoders


def _frame():
    return pd.DataFrame({
        "_oci_row_id": np.arange(8),
        "egfr": ["Mutant", "Wild-type", UNKNOWN, None, "Mutant", UNKNOWN, "Wild-type", None],
        "albumin": [3.0, 4.0, UNKNOWN, None, 3.5, UNKNOWN, 4.5, None],
        extraction_failure_column("albumin"): [0, 0, 0, 1, 0, 0, 0, 0],
    })


def test_final_encoder_adds_a_separate_grouped_indicator():
    egfr = DEFINITIONS[0]
    albumin = {**DEFINITIONS[3], "extraction_failure_indicator": True}
    frame = _frame()
    encoder = analysis._FeatureEncoder([albumin]).fit(frame)
    x = encoder.transform(frame)
    # value, missing, failure, documented_unknown
    assert x.shape == (8, 4)
    np.testing.assert_array_equal(x[:, 3], [0, 0, 1, 0, 0, 1, 0, 0])
    np.testing.assert_array_equal(x[:, 1], [0, 0, 0, 0, 0, 0, 0, 1])  # failure excluded too
    np.testing.assert_array_equal(x[:, 2], [0, 0, 0, 1, 0, 0, 0, 0])
    assert x[2, 0] == pytest.approx(0.0)  # median-imputed like a missing value
    categorical = analysis._FeatureEncoder([egfr]).fit(frame).transform(frame)
    # Mutant, Wild-type, __missing__, documented_unknown
    assert categorical.shape == (8, 4)
    np.testing.assert_array_equal(categorical[:, 3], [0, 0, 1, 0, 0, 1, 0, 0])
    np.testing.assert_array_equal(categorical[:, 2], [0, 0, 0, 1, 0, 0, 0, 1])
    assert categorical[[2, 5], :3].sum() == 0
    # Without training occurrences a held-out documented_unknown is missing.
    train = frame.assign(egfr=["Mutant", "Wild-type"] * 4)
    held = analysis._FeatureEncoder([egfr]).fit(train).transform(frame)
    assert held.shape == (8, 3)
    np.testing.assert_array_equal(held[:, 2], [0, 0, 1, 1, 0, 1, 0, 1])


def test_selection_designs_group_the_indicator_with_its_parent():
    albumin = {**DEFINITIONS[3], "extraction_failure_indicator": True}
    frame = _frame()
    design = _encode_design(frame, frame, [DEFINITIONS[0], albumin], categorical_min_count=1)
    names = list(design.column_names)
    assert "egfr:documented_unknown" in names and "albumin:documented_unknown" in names
    for name, feature_id in zip(names, design.column_feature_ids):
        assert feature_id == ("f_egfr" if name.startswith("egfr:") else "f_albumin")
    assert not any("level=documented_unknown" in name for name in names)
    unknown = design.train[:, names.index("albumin:documented_unknown")]
    missing = design.train[:, names.index("albumin:missing")]
    assert len(set(np.round(unknown[[2, 5]], 9))) == 1 and unknown[2] > unknown[0]
    assert missing[2] == missing[0]  # documented_unknown is not missing
    single = _encode_feature(frame, DEFINITIONS[0])
    assert "egfr:documented_unknown" in single.main_names
    assert "egfr:documented_unknown" in single.interaction_names
    assert UNKNOWN not in single.categorical_levels
    ordinal = _encode_feature(frame.assign(egfr=["0", "1", UNKNOWN, None] * 2), DEFINITIONS[2]
                              | {"name": "egfr"})
    assert "egfr:documented_unknown" in ordinal.main_names


def test_diagnostics_encoding_reports_a_documented_unknown_indicator():
    encoded, meta = diagnostics.encode_measurements(_frame(), DEFINITIONS)
    assert "egfr=documented_unknown" in encoded and "albumin=documented_unknown" in encoded
    assert encoded["egfr=documented_unknown"].tolist() == [0, 0, 1, 0, 0, 1, 0, 0]
    assert encoded["albumin=not_documented"].tolist() == [0, 0, 0, 0, 0, 0, 0, 1]
    assert "albumin=extraction_failed" in encoded
    assert {item["level"] for item in meta if item["feature"] == "egfr"} >= {
        "Mutant", "Wild-type", UNKNOWN, "not_documented"}
    assert diagnostics._column_kind("egfr:documented_unknown")[0] == (
        "documented_unknown_indicator")


# ---------------------------------------------------------------------------
# Informativeness filter and reporting


def test_filter_counts_documented_unknown_as_an_observed_state(tmp_path):
    rows = 60
    frame = pd.DataFrame({
        "_oci_row_id": np.arange(rows),
        # 30 documented values (one level) and 30 documented unknowns: two levels.
        "egfr": ["Mutant"] * 30 + [UNKNOWN] * 30,
        # Only 10 documented values; with unknowns 40 observed rows.
        "albumin": [3.0, 4.0] * 5 + [UNKNOWN] * 30 + [None] * 20,
    })
    treatment = np.array([1, 0] * 30)
    filtered, retained, summary = analysis._filter_sparse_training_candidates(
        frame, [DEFINITIONS[0], DEFINITIONS[3]], audit_path=tmp_path / "filter.json",
        health_audit_path=tmp_path / "health.json", min_informative_patients=20,
        treatment=treatment,
    )
    assert [feature["name"] for feature in retained] == ["egfr", "albumin"]
    decisions = {d["name"]: d for d in json.loads((tmp_path / "filter.json").read_text())["features"]}
    assert decisions["egfr"]["level_counts"] == {"Mutant": 30, UNKNOWN: 30}
    assert decisions["albumin"]["observed_rows"] == 40
    assert decisions["albumin"]["documented_unknown_rows"] == 30
    assert decisions["albumin"]["not_documented_rows"] == 20
    report = summary["training_documented_unknown_report"]
    by_name = {record["name"]: record for record in report["per_feature"]}
    assert by_name["egfr"]["documented_unknown_rows"] == 30
    assert by_name["egfr"]["by_treatment_arm"]["1"]["documented_unknown_rows"] == 15
    assert "documented_unknown" in analysis.CANDIDATE_FILTER_DESCRIPTION


def test_documented_unknown_report_by_arm():
    frame = pd.DataFrame({"egfr": [UNKNOWN, UNKNOWN, "Mutant", None]})
    report = du.documented_unknown_report(frame, [DEFINITIONS[0], DEFINITIONS[3]],
                                          scope="outer_heldout", treatment=[1, 0, 1, 1])
    egfr, albumin = report["per_feature"]
    assert egfr["documented_unknown_rows"] == 2 and egfr["documented_unknown_fraction"] == 0.5
    assert egfr["by_treatment_arm"]["1"] == {
        "rows": 3, "documented_unknown_rows": 1, "documented_unknown_fraction": pytest.approx(1 / 3)}
    assert egfr["by_treatment_arm"]["0"]["documented_unknown_fraction"] == 1.0
    assert egfr["arm_difference_treated_minus_control"] == pytest.approx(1 / 3 - 1)
    assert albumin["documented_unknown_rows"] == 0
    assert report["features_with_documented_unknown"] == 1
    with pytest.raises(ValueError):
        du.documented_unknown_report(frame, [DEFINITIONS[0]], scope="x", treatment=[1])


# ---------------------------------------------------------------------------
# End to end


def _unknown_aware_completion(calls):
    from tests.test_plain_handoff_stage2 import _fake_completion, _prompt_job
    from tests.stage2_prompt_spy import prompt_inputs

    base = _fake_completion(calls)

    def complete(messages, config):
        if not messages[0]["content"].startswith("You identify clinical variables"):
            body = prompt_inputs(messages)
            if _prompt_job(body) == "extract_stage2_patient_variables":
                calls.append("extract_stage2_patient_variables")
                assert "documented_unknown" in messages[0]["content"]
                rows = []
                for patient in body["patients"]:
                    match = re.search(r"ECOG\s*([0-4])", patient["text"])
                    value = (UNKNOWN if "unknown" in patient["text"]
                             else match.group(1) if match else None)
                    rows.append({"row_id": patient["row_id"],
                                 "values": {f["name"]: value for f in body["features"]}})
                return json.dumps({"rows": rows})
        return base(messages, config)

    return complete


def test_end_to_end_documented_unknown_is_extracted_encoded_and_reported(
    tmp_path: Path, monkeypatch,
):
    from dataclasses import replace

    from tests.stage2_prompt_spy import install

    install(monkeypatch)

    import oci.inference.plain_handoff_stage2 as stage2_workflow
    from oci.inference.plain_handoff_stage2 import (
        PlainHandoffStage2Config,
        run_plain_handoff_stage2,
    )
    from oci.inference.stage2_multi_model_config import (
        ModifierCountConfig,
        ModifierSelectionConfig,
    )
    from tests.test_plain_handoff_stage2 import _bow_outcome_payload, _tfidf_topic_payload
    from tests.test_stage2_multi_model import small_policy

    handoff = tmp_path / "handoff.jsonl"
    rows = []
    for outer_fold in (1, 2):
        rows.append({"source": "text_models", "outer_fold": outer_fold, "inner_fold": None,
                     "scope": "full_outer_train",
                     "evidence": {**_bow_outcome_payload(["ECOG performance status"]),
                                  "metrics": {"handoff_fit_rows": 20,
                                              "handoff_heldout_rows": 20}}})
        rows.append({"source": "tfidf", "outer_fold": outer_fold, "inner_fold": None,
                     "scope": "full_outer_train",
                     "evidence": _tfidf_topic_payload("treatment-ecog",
                                                      ["ECOG performance status"])})
    handoff.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    texts = [
        "Pretreatment ECOG status unknown." if index % 4 == 3 else f"Pretreatment ECOG {index % 2}."
        for index in range(40)
    ]
    dataset = pd.DataFrame({
        "patient_id": [f"p{index:03d}" for index in range(40)],
        "clinical_text": texts,
        "treatment_indicator": [int(index % 4 in {1, 3}) for index in range(40)],
        "outcome_indicator": [int(index % 5 in {0, 1}) for index in range(40)],
    })
    split_path = tmp_path / "split_provenance.jsonl"
    split_rows = []
    for outer_fold, heldout in ((1, list(range(0, 20))), (2, list(range(20, 40)))):
        fit = sorted(set(range(40)) - set(heldout))
        split_rows.append({
            "outer_fold": outer_fold, "fit_row_ids": fit, "heldout_row_ids": heldout,
            "inner_splits": [
                {"inner_fold": 1, "fit_row_ids": fit[:10], "heldout_row_ids": fit[10:]},
                {"inner_fold": 2, "fit_row_ids": fit[10:], "heldout_row_ids": fit[:10]},
            ],
        })
    split_path.write_text("".join(json.dumps(row) + "\n" for row in split_rows),
                          encoding="utf-8")
    policy = small_policy()
    policy = replace(policy, multi_model=replace(
        policy.multi_model,
        modifier_selection=ModifierSelectionConfig(rule="balanced_evidence"),
        modifier_count=ModifierCountConfig(estimators=("causal_forest",)),
    ))
    calls: list[str] = []
    output = tmp_path / "stage2"
    config = PlainHandoffStage2Config(
        endpoint="http://stage2.test/v1", model="test-model", max_prompt_chars=12_000,
        workers=4, statistical_selection=policy,
        extraction_llm=stage2_workflow.Stage2ExtractionLLMConfig(
            endpoint="http://extract.test/v1", model="small-model", workers=4),
        max_review_rounds=1, estimation_trees=10, min_informative_patients=2,
        required_architectures=(), nuisance_learner="elastic_net",
    )
    result = run_plain_handoff_stage2(
        handoff_path=handoff, output_dir=output,
        clinical_question="Estimate the treatment effect.", config=config,
        completion=_unknown_aware_completion([]),
        extraction_completion=_unknown_aware_completion(calls),
        dataset=dataset, split_provenance_path=split_path, inner_folds=2, seed=7,
    )
    assert result["phase"] == "causal_estimation"
    assert "extract_stage2_patient_variables" in calls
    for fold in (1, 2):
        fold_dir = output / f"outer_00{fold}"
        fit_report = json.loads(
            (fold_dir / "extraction" / "documented_unknown_report_fit.json").read_text())
        heldout_report = json.loads(
            (fold_dir / "extraction" / "documented_unknown_report_heldout.json").read_text())
        record = fit_report["per_feature"][0]
        assert record["documented_unknown_rows"] == 5
        assert record["by_treatment_arm"]["1"]["documented_unknown_rows"] == 5
        assert record["by_treatment_arm"]["0"]["documented_unknown_rows"] == 0
        assert heldout_report["per_feature"][0]["documented_unknown_rows"] == 5
        assert heldout_report["reported_after_effect_predictions_were_frozen"] is True
        decisions = json.loads(
            (fold_dir / "extraction" / "candidate_missingness_filter.json").read_text())["features"]
        assert decisions[0]["documented_unknown_rows"] == 5
        candidates = pd.read_csv(fold_dir / "estimation" / "nuisance_candidate_fit.csv")
        assert (candidates.iloc[:, 1] == UNKNOWN).sum() == 5
        failures = json.loads(
            (fold_dir / "estimation" / "extraction_failure_diagnostics.json").read_text())
        assert failures["documented_unknown"]["heldout"]["documented_unknown_cells"] == 5
        assert failures["documented_unknown"][
            "features_with_grouped_documented_unknown_indicator"]
        # The selection design carries the grouped indicator.
        selection = (fold_dir / "selection" / "elastic_net_selection.json").read_text()
        assert ":documented_unknown" in selection
    predictions = pd.read_csv(output / "cross_fitted_predictions.csv")
    assert len(predictions) == 40 and predictions["aipw_score"].notna().all()
