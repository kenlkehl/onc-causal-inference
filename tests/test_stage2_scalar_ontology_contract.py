from __future__ import annotations

import json

import pandas as pd
import pytest

from oci.inference import plain_handoff_stage2 as workflow
from oci.inference import plain_handoff_stage2_analysis as analysis
from oci.inference.stage2_prompt_catalog import SYSTEM_PROMPTS


def feature(name="brain_sites", **kwargs):
    return {
        "feature_id": "f_" + name, "name": name,
        "description": "Anatomical sites of brain metastases.",
        "value_type": "categorical", "categories_or_unit": ["frontal", "cerebellum"],
        "measurement_definition": "List all involved anatomical regions.",
        "missing_value_rule": "Return null when undocumented.",
        "supporting_packet_ids": ["card_1"], **kwargs,
    }


def split_answer():
    return {"action": "split", "reason": "Several anatomical sites can be involved simultaneously.",
            "components": [{
                "name": name, "description": f"Metastasis in the {name}.",
                "value_type": "binary", "categories_or_unit": ["Present", "Absent"],
                "measurement_definition": f"Use the latest explicit finding of metastasis in the {name}. "
                    "Present requires positive evidence; Absent requires an explicit negative finding.",
                "missing_value_rule": "Return null if this site is not assessed.",
            } for name in ["frontal", "cerebellum"]]}


def test_ambiguous_revision_is_valid_without_invented_categories():
    original = feature("medication", value_type="ambiguous", categories_or_unit=[])
    definition = {k: original[k] for k in analysis.ONTOLOGY_DEFINITION_FIELDS}
    decision = analysis._validate_ontology_refinement(
        {"action": "revise", "reason": "No closed vocabulary is supported.",
         "definition": definition}, feature=original)
    assert decision["value_type"] == "ambiguous"
    with pytest.raises(ValueError, match="no declared categories"):
        analysis._validate_ontology_refinement(
            {**decision, "categories_or_unit": ["dose", "route"]}, feature=original)


def test_split_is_stable_atomic_and_does_not_invent_patient_values():
    original = feature()
    decision = analysis._validate_ontology_refinement(split_answer(), feature=original)
    # Cached normalized decisions must round-trip without changing component IDs.
    assert analysis._validate_ontology_refinement(decision, feature=original) == decision
    children = analysis._apply_ontology_decision(original, decision)
    assert [c["name"] for c in children] == ["brain_sites__frontal", "brain_sites__cerebellum"]
    assert all(c["supporting_packet_ids"] == ["card_1"] for c in children)
    assert all(c["ontology_parent_feature_id"] == original["feature_id"] for c in children)
    assert all(c["conflict_resolution"]["strategy"] == "latest" for c in children)
    assert all("not assessed or not mentioned" in c["missing_value_rule"] for c in children)
    assert all("explicit negative finding" in c["missing_value_rule"] for c in children)
    assert "brain_sites" not in [c["name"] for c in children]
    values = {c["clinical_label"]: "Present" for c in children}
    validated = analysis._validate_extraction(values, row_ids=[0], definitions=children)
    assert set(validated["rows"][0]["values"].values()) == {"Present"}
    with pytest.raises(analysis._ExtractionCategoryError):
        analysis._validate_extraction({"brain sites": "frontal, cerebellum"},
                                      row_ids=[0], definitions=[original])


@pytest.mark.parametrize("change", ["duplicate", "one", "locked"])
def test_invalid_splits_and_locked_features_are_rejected(change):
    original, answer = feature(), split_answer()
    if change == "duplicate":
        answer["components"][1]["name"] = answer["components"][0]["name"]
    elif change == "one":
        answer["components"] = answer["components"][:1]
    else:
        original["configured_explicit_feature"] = True
    with pytest.raises(ValueError):
        analysis._validate_ontology_refinement(answer, feature=original)


def test_semantic_check_rejects_attribute_names_as_answer_categories(tmp_path):
    original = feature("current_medications", value_type="ambiguous", categories_or_unit=[])
    bad = {k: original[k] for k in analysis.ONTOLOGY_DEFINITION_FIELDS}
    bad.update(value_type="categorical", categories_or_unit=["dose", "route"])
    calls = []

    def request(messages, validate, **kwargs):
        calls.append(messages)
        if len(calls) == 1:
            return validate({"action": "revise", "reason": "Make it categorical.", "definition": bad})
        if len(calls) == 2:
            assert messages[0]["content"] == SYSTEM_PROMPTS["ontology_scalar_contract"]
            return validate({"valid": False, "reason": "Dose and route are attributes, not medication values."})
        assert "Dose and route" in messages[-1]["content"]
        return validate({"action": "keep", "reason": "Ambiguous is allowed; no supported closed vocabulary."})

    decision = analysis._request_scalar_ontology_decision(
        messages=[{"role": "user", "content": "Review this variable."}],
        feature=original, request_json=request, output_dir=tmp_path)
    assert decision["action"] == "keep"
    assert len(calls) == 3
    assert json.loads((tmp_path / "scalar_contract_checks.json").read_text())["checks"][0]["check"]["valid"] is False


@pytest.mark.parametrize("failure_review", [False, True])
def test_both_review_paths_checkpoint_split_and_reuse_it(tmp_path, failure_review):
    original = feature()
    calls = []

    def request(messages, validate, **kwargs):
        calls.append(messages)
        if messages[0]["content"] == SYSTEM_PROMPTS["ontology_scalar_contract"]:
            return validate({"valid": True, "reason": "Independent sites become independent indicators."})
        return validate(split_answer())

    def run():
        if failure_review:
            return analysis._request_ontology_refinements(
                definitions=[original], repeated_patterns={"brain_sites": [{
                    "failure_kind": "out_of_ontology_category", "patient_count": 4,
                    "example_values": ["frontal, cerebellum"]}]},
                output_dir=tmp_path, request_json=request, workers=2)
        return analysis._request_aggregate_ontology_supervisor(
            definitions=[original], summaries=[{"feature_id": original["feature_id"]}],
            failure_summary={}, output_dir=tmp_path, cache_dir=tmp_path / "cache",
            request_json=request, workers=2)

    children, changed, report = run()
    assert changed and len(children) == 2
    assert len(calls) == 2
    assert run()[0] == children
    assert len(calls) == 2


def test_incremental_split_preserves_other_measurements_and_resumes(tmp_path):
    original = feature()
    children = analysis._apply_ontology_decision(original,
        analysis._validate_ontology_refinement(split_answer(), feature=original))
    nlr = feature("nlr", value_type="continuous", categories_or_unit=[],
                  description="NLR", measurement_definition="Extract NLR.")
    prior = pd.DataFrame({"_oci_row_id": [0, 1], "nlr": [3.2, 5.1], "brain_sites": [None, None]})
    calls = []

    def request(messages, validate, **kwargs):
        calls.append(messages)
        present = "both sites" in messages[1]["content"]
        return validate({c["clinical_label"]: "Present" if present else None for c in children})

    args = dict(dataset=pd.DataFrame({"text": ["Metastases in both sites: frontal and cerebellum.",
                                                "No brain imaging available."]}),
        row_ids=[0, 1], text_column="text", definitions=[nlr, *children],
        prior_definitions=[nlr, original], prior_extracted=prior,
        prior_failure_summary={"feature_failure_patterns": []}, output_dir=tmp_path,
        request_json=request, workers=1, max_prompt_chars=100_000, feature_batch_size=10,
        request_identity={"model": "test"}, tokenizer=None, chunk_size_tokens=10000,
        context_window_tokens=20000, max_output_tokens=2000, context_margin_tokens=1000)
    result, _ = analysis._extract_changed_features_and_merge(**args)
    assert result["nlr"].tolist() == [3.2, 5.1]
    assert "brain_sites" not in result
    assert result.loc[0, children[0]["name"]] == result.loc[0, children[1]["name"]] == "Present"
    assert pd.isna(result.loc[1, children[0]["name"]])
    assert len(calls) == 2
    resumed, _ = analysis._extract_changed_features_and_merge(**args)
    pd.testing.assert_frame_equal(result, resumed)
    assert len(calls) == 2
    encoder = analysis._FeatureEncoder([nlr, *children]).fit(result)
    assert encoder.transform(result).shape[0] == 2


def test_semantic_repair_handoff_keeps_five_off_repairs_and_good_fields(tmp_path):
    definitions = [feature(), feature("nlr", value_type="continuous", categories_or_unit=[])]
    efforts = []
    config = workflow.PlainHandoffStage2Config(endpoint="http://test/v1", model="test")

    def completion(messages, cfg):
        efforts.append(workflow._stage2_request_policy(cfg)["reasoning_effort"])
        return json.dumps({"brain sites": "frontal, cerebellum", "nlr": 3.2})

    with pytest.raises(analysis.Stage2ResponseValidationError) as caught:
        workflow._request_json(messages=[{"role": "user", "content": "Clinical record"}],
            config=config, completion=completion, request_kind="extraction",
            validate=lambda v: analysis._validate_extraction(v, row_ids=[0], definitions=definitions))
    assert efforts == ["none"] * 6 + ["high"]
    error = analysis._category_error_from_exception(caught.value)
    assert error.response["rows"][0]["values"]["nlr"] == 3.2
