from dataclasses import replace
import json
from types import SimpleNamespace as NS

import pytest

from oci.inference import plain_handoff_stage2 as workflow
from oci.inference import plain_handoff_stage2_analysis as analysis


DEFINITIONS = [
    {"name": "creatinine_clearance", "value_type": "continuous", "categories_or_unit": ["mL/min"],
     "measurement_definition": "Use the latest documented clearance.", "missing_value_rule": "Null if absent.",
     "conflict_resolution": {"strategy": "latest"}},
    {"name": "drug_allergies", "value_type": "categorical", "categories_or_unit": ["None", "Present"],
     "measurement_definition": "Extract documented drug allergies.", "missing_value_rule": "Null if absent."},
]
ANSWER = {"values": {"creatinine clearance": 69.5, "drug allergies": "None"},
          "decision_notes": {"creatinine clearance": "Clearance 69.5 on 2025-01-02", "drug allergies": "NKDA documented"}}


def test_serial_prompt_reminder_follows_the_complete_record_section():
    source = "FIRST SOURCE MARKER\n" + "Long clinical text. " * 200 + "\nLAST SOURCE MARKER"
    messages = analysis._serial_extraction_prompt(
        definitions=DEFINITIONS, row_id=123, chunk_text=source,
        prior_values={d["name"]: None for d in DEFINITIONS},
        prior_feature_state={d["name"]: None for d in DEFINITIONS},
        chunk_index=1, char_start=0, char_end=len(source), document_chars=len(source),
    )
    content = messages[-1]["content"]
    assert source in content
    suffix = content.split("LAST SOURCE MARKER", 1)[1]
    assert "values and decision_notes" in suffix
    assert "creatinine clearance" in suffix and "drug allergies" in suffix
    assert "format placeholders, not suggested answers" in suffix
    assert "row_id" not in content and "carry_forward_state" not in content


def test_schema_and_validator_require_notes_without_inventing_clinical_state():
    import jsonschema
    schema = analysis._serial_response_schema(DEFINITIONS)
    jsonschema.validate(ANSWER, schema)
    validated = analysis._validate_serial_extraction(ANSWER, row_id=123, definitions=DEFINITIONS)
    assert validated["rows"][0]["values"] == {"creatinine_clearance": 69.5, "drug_allergies": "None"}
    assert validated["rows"][0]["carry_forward_state"]["creatinine_clearance"] == "Clearance 69.5 on 2025-01-02"
    for invalid in [
        {"values": ANSWER["values"]},
        {**ANSWER, "decision_notes": {}},
        {**ANSWER, "extra": "unsupported field"},
        {**ANSWER, "values": {**ANSWER["values"], "invented variable": 7}},
    ]:
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(invalid, schema)
    with pytest.raises(ValueError, match="values and decision_notes"):
        analysis._validate_serial_extraction({"values": ANSWER["values"]}, row_id=123, definitions=DEFINITIONS)


def test_serial_wire_schema_survives_repairs_and_preserves_thinking_threshold(monkeypatch, tmp_path):
    import openai
    sent = []

    class Client:
        def __init__(self, **kwargs):
            self.chat = NS(completions=NS(create=self.create))

        def create(self, **kwargs):
            sent.append(kwargs)
            # Simulate a compatible server ignoring the response-format hint.
            answer = {"wrong_wrapper": ANSWER} if len(sent) <= 6 else ANSWER
            return NS(id="test", usage=None, choices=[NS(
                finish_reason="stop", message=NS(content=json.dumps(answer)))])

        def close(self):
            pass

    monkeypatch.setattr(openai, "OpenAI", Client)
    config = workflow.PlainHandoffStage2Config(
        endpoint="http://test/v1", model="nvidia/Gemma-4-26B-A4B-NVFP4",
        thinking_after_response_repairs=5,
    )

    def request(messages, validate, *, request_kind):
        return workflow._request_json(messages=messages, validate=validate, config=config,
            completion=workflow._openai_completion, request_kind=request_kind,
            prompt_token_counter=lambda messages: 100, context_window_tokens=262144)

    result = analysis._request_validated_extraction(
        messages=[{"role": "user", "content": "2025-01-02 CrCl 69.5 mL/min. NKDA."}],
        row_ids=[123], definitions=DEFINITIONS, request_json=request,
        ontology_audit_path=tmp_path / "category_ontology_repair.json",
        messages_for_definitions=lambda definitions: [],
        validate_response=lambda value, *, definitions: analysis._validate_serial_extraction(
            value, row_id=123, definitions=definitions),
        response_schema=analysis._serial_response_schema,
        response_instructions=analysis._serial_output_instructions,
    )
    assert result["rows"][0]["values"]["creatinine_clearance"] == 69.5
    assert all(r["response_format"]["type"] == "json_schema" for r in sent)
    assert [r["extra_body"]["chat_template_kwargs"]["enable_thinking"] for r in sent] == [False] * 6 + [True]
    for r in sent[1:]:
        repair = r["messages"][-1]["content"]
        assert "values and decision_notes" in repair and "creatinine clearance" in repair
        assert "patient row" not in repair
    assert "runtime_response_format" not in replace(config, runtime_response_format=sent[0]["response_format"]).public_dict()


def test_schema_compatibility_fallback_retains_thinking_controls():
    config = workflow.PlainHandoffStage2Config(endpoint="http://test/v1", model="nvidia/Gemma-4-26B-A4B-NVFP4")
    response_format = {"type": "json_schema", "json_schema": {"name": "serial", "strict": True,
        "schema": analysis._serial_response_schema(DEFINITIONS)}}
    variants = workflow._openai_request_variants(base_kwargs={"response_format": response_format},
        request_policy=workflow._stage2_request_policy(config, "extraction"), model_family="gemma4")
    assert variants[0]["response_format"] == response_format
    assert variants[1]["response_format"] == {"type": "json_object"}
    assert variants[0]["extra_body"] == variants[1]["extra_body"]
    assert variants[1]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert "response_format" not in variants[-1]


def test_completed_serial_measurements_survive_transport_and_prompt_fix(monkeypatch, tmp_path):
    import pandas as pd
    from tests.test_plain_handoff_stage2 import _CharacterChatTokenizer

    current_prompt = analysis._serial_extraction_prompt

    def original_prompt(**kwargs):
        messages = current_prompt(**kwargs)
        messages[-1]["content"] = messages[-1]["content"].split("\n\nEnd of record section.", 1)[0]
        return messages

    monkeypatch.setattr(analysis, "_serial_extraction_prompt", original_prompt)
    source = "2025-01-02 CrCl 69.5 mL/min. NKDA. " * 60
    kwargs = dict(dataset=pd.DataFrame({"text": [source]}), row_ids=[0], text_column="text",
        definitions=DEFINITIONS, output_dir=tmp_path, workers=1, tokenizer=_CharacterChatTokenizer(),
        chunk_size_tokens=150, context_window_tokens=10000, max_output_tokens=500,
        context_margin_tokens=100, max_prompt_chars=100000, feature_batch_size=1)
    calls = []

    def first_request(messages, validate, *, request_kind):
        calls.append(messages)
        names = validate.response_format["json_schema"]["schema"]["properties"]["values"]["required"]
        return validate({key: {name: ANSWER[key][name] for name in names} for key in ANSWER})

    first = analysis.extract_rows(request_json=first_request, **kwargs)
    assert len(calls) > 1
    checkpoints = {p: p.read_bytes() for p in tmp_path.rglob("*")
                   if p.name in {"complete.json", "result.json"} and "feature_batches" in p.parts}
    # Force reconstruction from completed feature slices, as happens when a
    # patient still has other unfinished slices during checkpoint resume.
    for path in [tmp_path / "complete.json", tmp_path / "extracted.csv",
                 tmp_path / "batches/batch_00001/complete.json",
                 tmp_path / "batches/batch_00001/result.json"]:
        path.unlink(missing_ok=True)
    monkeypatch.setattr(analysis, "_serial_extraction_prompt", current_prompt)

    def unexpected_request(*args, **kwargs):
        pytest.fail("completed serial measurements must not be regenerated")

    resumed = analysis.extract_rows(request_json=unexpected_request, **kwargs)
    pd.testing.assert_frame_equal(first, resumed)
    assert all(p.read_bytes() == value for p, value in checkpoints.items())
