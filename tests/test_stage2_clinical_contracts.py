"""Exercise clinical responses directly, without orchestration prompt spies."""
from copy import deepcopy

import pytest

from oci.inference import plain_handoff_stage2_analysis as extraction
from oci.inference import stage2_clinical_prompts as prompts
from oci.inference.stage2_modifier_concepts import _concept_validator, infer_modifier_concepts
from oci.inference.stage2_role_adjudication import Stage2RoleAdjudicationConfig
from oci.inference.stage2_value_interpretation import compile_value_interpretations


def measurement():
    return {"feature_id": "private-feature-92", "name": "serum_creatinine",
            "description": "Serum creatinine concentration.", "value_type": "continuous",
            "categories_or_unit": ["mg/dL"], "measurement_definition": "Use the latest serum creatinine.",
            "missing_value_rule": "Null when absent.",
            "conflict_resolution": {"strategy": "latest", "positive_category": None}}


def test_clinical_names_are_mapped_to_trusted_patient_and_feature_keys():
    definition = measurement()
    messages = extraction._extraction_prompt(definitions=[definition], rows=[{"row_id": 921, "text": "Creatinine 1.4 mg/dL."}])
    assert "921" not in str(messages) and "private-feature" not in str(messages)
    result = extraction._validate_extraction({"serum creatinine": 1.4}, row_ids=[921], definitions=[definition])
    assert result["rows"] == [{"row_id": 921, "values": {"serum_creatinine": 1.4}}]
    with pytest.raises(ValueError):
        extraction._validate_extraction({"different measurement": 1.4}, row_ids=[921], definitions=[definition])
    serial = extraction._validate_serial_extraction(
        {"values": {"serum creatinine": 1.4}, "decision_notes": {"serum creatinine": "Dated 2025-03-02."}},
        row_id=921, definitions=[definition])
    assert serial["rows"][0]["carry_forward_state"] == {"serum_creatinine": "Dated 2025-03-02."}


def test_occurrence_quotes_get_python_offsets_and_repeated_dates_need_context():
    text = "2025-03-02: Creatinine 1.4.\n2025-03-02: Creatinine 1.8."
    page = {"row_id": 921, "text": text, "page": {"page_index": 1, "char_start": 100, "char_end": 100+len(text), "document_chars": 100+len(text)}}
    observation = {"feature": "serum creatinine", "value": 1.8,
                   "quote": "2025-03-02: Creatinine 1.8.", "governing_date_quote": "2025-03-02"}
    result = extraction._validate_page_observations({"observations": [observation]}, page=page, definitions=[measurement()])
    row = result["rows"][0]["observations"][0]
    assert row["value"] == 1.8 and row["feature_name"] == "serum_creatinine"
    assert row["recorded_at"] == "2025-03-02"
    assert row["evidence_start"] == text.index("2025-03-02: Creatinine 1.8.")
    with pytest.raises(ValueError, match="governing date occurs repeatedly"):
        extraction._validate_page_observations({"observations": [{**observation, "quote": "Creatinine 1.8."}]}, page=page, definitions=[measurement()])


def meanings(rows, representation="categorical"):
    return {"status": "ready", "representation": representation, "reason": "Use documented numerical meanings.",
            "token_interpretations": [{"raw_text": raw, "meaning": kind, "interpretation": text} for raw, kind, text in rows]}


def observations(*phrases):
    return {"categorical_values": [{"raw_value": p, "count": 1} for p in phrases]}


def test_python_builds_exhaustive_threshold_categories_without_inventing_cutoffs():
    result = compile_value_interpretations(meanings([
        ("negative", "defined_category", "less than 1"),
        ("positive", "defined_category", "at least 50"),
        ("equivocal", "unusable", "No numerical definition is supplied.")]),
        observations=observations("negative", "positive", "equivocal"))
    assert result["canonical_categories"] == ["< 1", "[1, 50)", ">= 50"]
    assert result["categorical_value_map"][-1]["canonical_value"] is None
    assert result["numeric_bin_rules"][1] == {"lower_bound": 1., "lower_inclusive": True,
        "upper_bound": 50., "upper_inclusive": False, "canonical_value": "[1, 50)"}
    exact = compile_value_interpretations(meanings([("one point four", "exact_number", "1.4 mg/dL")], "continuous"),
                                         observations=observations("one point four"))
    assert exact["categorical_value_map"][0]["canonical_value"] == 1.4
    with pytest.raises(ValueError, match="ambiguous"):
        compile_value_interpretations(meanings([("middle", "explicit_interval", "between 1 and 50")]), observations=observations("middle"))
    with pytest.raises(ValueError, match="thresholds require categories"):
        compile_value_interpretations(meanings([("high", "explicit_interval", "at least 50")], "continuous"), observations=observations("high"))


def test_concept_recurrence_and_representatives_stay_with_supplied_evidence(tmp_path, monkeypatch):
    from oci.inference import stage2_modifier_concepts as concepts
    definitions = [{**measurement(), "feature_id": key, "name": name} for key, name in [("a", "serum_creatinine"), ("b", "creatinine_concentration")]]
    cells = [{"inner_fold": fold, "family": "univariable", "records": [
        {"feature_id": key, "role": "effect", "status": "ok", "selected": True, "score": .2}
        for key in ("a", "b")], "fit_row_ids": [10], "validation_row_ids": [11]} for fold in (1, 2)]
    from tests.test_stage2_multi_model import small_policy
    report = {"cells": cells, "policy": small_policy().public_dict(),
              "multi_model_evidence": concepts.aggregate_evidence(cells, ["a", "b"]),
              "oracle": "FORBIDDEN_ORACLE", "row_values": "FORBIDDEN_VALUES"}
    rankings = []
    def rank(**kwargs):
        rankings.append(kwargs)
        keys = [d["feature_id"] for d in kwargs["definitions"]]
        return {"ranking": [{"feature_id": key, "rationale": "Supported."} for key in keys]}
    monkeypatch.setattr(concepts, "rank_modifier_candidates", rank)
    def request(messages, validate, **kwargs):
        text = str(messages)
        assert "FORBIDDEN" not in text and "fit_row_ids" not in text
        assert "Appeared in 2 of 2 candidate lists" in text
        response = {"concepts": [{"name": "Creatinine concentration", "members": ["serum creatinine", "creatinine concentration"],
            "representatives": ["serum creatinine"], "modifier_recommendation": "retain", "rationale": "Comparable clinical measurements.", "unresolved_questions": "Redundancy is untested."}]}
        incomplete = deepcopy(response)
        incomplete["concepts"][0]["members"].pop()
        with pytest.raises(ValueError, match="every supplied"):
            validate(incomplete)
        return validate(response)
    result = infer_modifier_concepts(definitions=definitions, statistical_report=report, request_json=request,
        output_dir=tmp_path, role_policy=Stage2RoleAdjudicationConfig(), top_n=100, maximum=64,
        maximum_chars=100000, model_identity="test")
    assert len(rankings) == 3 and len(rankings[-1]["definitions"]) == 1
    assert result["concept_review"]["recurrence"] == {"a": 2, "b": 2}
    assert result["concept_review"]["confounders_changed"] is False
    assert result["ranking"][0]["feature_id"] == "a"
