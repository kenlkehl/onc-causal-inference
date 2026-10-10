"""Split components must retain their full response keys and distinct values."""
import json

import pandas as pd
import pytest

from oci.inference import plain_handoff_stage2 as workflow
from oci.inference import plain_handoff_stage2_analysis as analysis
from oci.inference import stage2_clinical_prompts as prompts


def components():
    return [{"feature_id": "f__" + name, "name": "fluid_intake__" + name,
             "clinical_label": "fluid intake: " + name,
             "description": meaning, "value_type": kind,
             "categories_or_unit": [], "measurement_definition": meaning,
             "missing_value_rule": "Null when not documented.",
             "conflict_resolution": {"strategy": "latest", "positive_category": None}}
            for name, meaning, kind in [
                ("fluid_intake_amount", "Latest documented daily fluid intake in mL.", "continuous"),
                ("fluid_intake_comparison", "Latest documented change in fluid intake.", "ambiguous")]]


def test_parent_name_is_rejected_with_complete_component_names():
    definitions = components()
    with pytest.raises(analysis._ExtractionFieldError) as caught:
        analysis._validate_extraction({"fluid intake": 1500}, row_ids=[0], definitions=definitions)
    assert set(caught.value.feature_names) == {d["name"] for d in definitions}
    assert "parent name" in str(caught.value)
    assert all(prompts.label(d) in str(caught.value) for d in definitions)
    # Even a batch with just one sibling must not guess what the parent meant.
    with pytest.raises(ValueError, match="parent name"):
        prompts.named_values({"fluid intake": "unchanged"}, definitions[:1])
    result = analysis._validate_extraction(
        dict(zip(map(prompts.label, definitions), [1500, "unchanged"])),
        row_ids=[0], definitions=definitions)
    assert result["rows"][0]["values"] == dict(zip([d["name"] for d in definitions], [1500, "unchanged"]))


def test_full_record_schema_and_repair_keep_components_separate(tmp_path):
    definitions = components()
    note = "Fluid intake is 1500 mL daily, unchanged from last visit."
    messages = analysis._extraction_prompt(definitions=definitions, rows=[{"text": note}])
    labels = list(map(prompts.label, definitions))
    assert all(json.dumps(name) in messages[-1]["content"].split("End of patient record.")[1]
               for name in labels)
    assert all(prompts.feature_text(d).splitlines()[0] == json.dumps(prompts.label(d))
               for d in definitions)
    efforts = []

    def completion(messages, config):
        schema = config.runtime_response_format["json_schema"]["schema"]
        assert schema["required"] == labels
        assert set(schema["properties"]) == set(labels)
        assert schema["additionalProperties"] is False
        efforts.append(workflow._stage2_request_policy(config)["reasoning_effort"])
        if len(efforts) == 1:
            return json.dumps({"fluid intake": 1500})
        assert "parent name" in str(messages)
        assert all(name in str(messages) for name in labels)
        return json.dumps(dict(zip(labels, [1500, "unchanged"])))

    config = workflow.PlainHandoffStage2Config(endpoint="http://test/v1", model="test")
    def request(messages, validate, **kwargs):
        return workflow._request_json(messages=messages, validate=validate, config=config,
                                      completion=completion, **kwargs)
    result = analysis._request_validated_extraction(messages=messages, row_ids=[0],
        definitions=definitions, request_json=request, ontology_audit_path=tmp_path / "category.json",
        messages_for_definitions=lambda ds: analysis._extraction_prompt(definitions=ds, rows=[{"text": note}]))
    assert efforts == ["none", "none"]
    assert result["rows"][0]["values"][definitions[0]["name"]] == 1500
    assert result["rows"][0]["values"][definitions[1]["name"]] == "unchanged"


def test_completed_split_measurements_survive_prompt_fix(tmp_path, monkeypatch):
    definitions = components()
    current = analysis._extraction_prompt
    def old_prompt(**kwargs):
        messages = current(**kwargs)
        messages[-1]["content"] = messages[-1]["content"].split("\n\nEnd of patient record.")[0]
        return messages
    monkeypatch.setattr(analysis, "_extraction_prompt", old_prompt)
    args = dict(dataset=pd.DataFrame({"text": ["Fluid intake 1500 mL daily, unchanged."]}),
        row_ids=[0], text_column="text", definitions=definitions, output_dir=tmp_path,
        workers=1, max_prompt_chars=100000, feature_batch_size=1)
    def request(messages, validate, **kwargs):
        keys = validate.response_format["json_schema"]["schema"]["required"]
        return validate({k: 1500 if k.endswith("amount") else "unchanged" for k in keys})
    first = analysis.extract_rows(request_json=request, **args)
    saved = {p: p.read_bytes() for p in tmp_path.glob("batches/*/feature_batches/*/result.json")}
    assert saved
    for name in ["complete.json", "extracted.csv", "batches/batch_00001/complete.json",
                 "batches/batch_00001/result.json"]:
        (tmp_path / name).unlink(missing_ok=True)
    monkeypatch.setattr(analysis, "_extraction_prompt", current)
    def unexpected(*args, **kwargs):
        pytest.fail("successful measurements should be reused")
    resumed = analysis.extract_rows(request_json=unexpected, **args)
    pd.testing.assert_frame_equal(first, resumed)
    assert all(p.read_bytes() == content for p, content in saved.items())


def test_serial_and_page_split_names_remain_distinct():
    definitions = components()
    labels = list(map(prompts.label, definitions))
    serial = analysis._validate_serial_extraction(
        {"values": dict(zip(labels, [1500, "unchanged"])),
         "decision_notes": {k: None for k in labels}}, row_id=0, definitions=definitions)
    assert len(serial["rows"][0]["values"]) == 2
    assert "both sides" in analysis._serial_output_instructions(definitions)
    text = "Fluid intake 1500 mL daily, unchanged."
    page = {"row_id": 0, "text": text, "page": {"page_index": 1, "char_start": 0,
            "char_end": len(text), "document_chars": len(text)}}
    result = analysis._validate_page_observations({"observations": [
        {"feature": k, "value": v, "quote": text, "governing_date_quote": None}
        for k, v in zip(labels, [1500, "unchanged"])]}, page=page, definitions=definitions)
    assert [x["feature_name"] for x in result["rows"][0]["observations"]] == [d["name"] for d in definitions]
