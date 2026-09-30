"""A bad field must not erase measurements from the rest of its batch."""
import json

import pandas as pd
import pytest

from oci.inference import plain_handoff_stage2 as workflow
from oci.inference import plain_handoff_stage2_analysis as analysis
from tests.stage2_prompt_spy import install, prompt_inputs


def definitions():
    return [{"name": name, "value_type": "continuous", "categories_or_unit": ["units"],
             "measurement_definition": "Latest documented measurement."}
            for name in ["abdomen_or_pelvis_incidental_findings", *[f"measurement_{i}" for i in range(9)]]]


def request_with(completion, *, repairs=15):
    config = workflow.PlainHandoffStage2Config(
        endpoint="http://unused.test/v1", model="test-model", max_response_repairs=repairs,
    )
    return lambda messages, validate, *, request_kind: workflow._request_json(
        messages=messages, validate=validate, request_kind=request_kind,
        config=config, completion=completion,
    )


@pytest.mark.parametrize("repairs,failed_attempts", [(15, 4), (1, 2)])
def test_reextracts_nine_fields_after_named_failure_and_checkpoints_result(tmp_path, monkeypatch, repairs, failed_attempts):
    install(monkeypatch)
    calls = []

    def completion(messages, _config):
        body = prompt_inputs(messages)
        names = [d["name"] for d in body["features"]]
        calls.append(names)
        assert body["patients"][0]["text"] == "All documented measurements."
        values = {name.replace("_", " "): 12 for name in names}
        if names[0] == "abdomen_or_pelvis_incidental_findings":
            values["abdomen or pelvic incidental findings"] = values.pop("abdomen or pelvis incidental findings")
        return json.dumps(values)

    args = dict(dataset=pd.DataFrame({"clinical_text": ["All documented measurements."]}),
                row_ids=[0], text_column="clinical_text", definitions=definitions(),
                output_dir=tmp_path, request_json=request_with(completion, repairs=repairs),
                workers=1, max_prompt_chars=100_000)
    result = analysis.extract_rows(**args)
    assert len(calls) == failed_attempts + 1
    assert list(map(len, calls)) == [10] * failed_attempts + [9]
    assert pd.isna(result.loc[0, "abdomen_or_pelvis_incidental_findings"])
    assert all(result.loc[0, f"measurement_{i}"] == 12 for i in range(9))
    audit = json.loads((tmp_path / "batches/batch_00001/field_recovery.json").read_text())
    assert audit["excluded_features"] == ["abdomen_or_pelvis_incidental_findings"]
    assert audit["status"] == "complete"
    summary = json.loads((tmp_path / "failure_summary.json").read_text())
    assert summary["structural_failure_patient_count"] == 0
    assert summary["feature_failure_patterns"][0]["feature_name"] == "abdomen_or_pelvis_incidental_findings"
    pd.testing.assert_frame_equal(result, analysis.extract_rows(**args))
    assert len(calls) == failed_attempts + 1


def test_transient_field_error_can_repair_without_exclusion(tmp_path, monkeypatch):
    install(monkeypatch)
    calls = []

    def completion(messages, _config):
        names = [d["name"] for d in prompt_inputs(messages)["features"]]
        calls.append(names)
        values = {name.replace("_", " "): 7 for name in names}
        if len(calls) <= 3:
            values.pop("measurement 0")
        return json.dumps(values)

    frame = analysis.extract_rows(dataset=pd.DataFrame({"text": ["Recorded values."]}),
        row_ids=[0], text_column="text", definitions=definitions(), output_dir=tmp_path,
        request_json=request_with(completion), workers=1, max_prompt_chars=100_000)
    assert len(calls) == 4 and all(len(names) == 10 for names in calls)
    assert frame.loc[0, "measurement_0"] == 7
    assert not list(tmp_path.rglob("field_recovery.json"))


def test_reduced_request_failure_propagates_and_preserves_checkpoints(tmp_path, monkeypatch):
    install(monkeypatch)

    def completion(messages, _config):
        names = [d["name"] for d in prompt_inputs(messages)["features"]]
        if len(names) == 9:
            raise analysis.Stage2RequestExhaustedError("server unavailable")
        return json.dumps({name.replace("_", " "): 9 for name in names[1:]})

    with pytest.raises(analysis.Stage2RequestExhaustedError, match="server unavailable"):
        analysis.extract_rows(dataset=pd.DataFrame({"text": ["Recorded values."]}),
            row_ids=[0], text_column="text", definitions=definitions(), output_dir=tmp_path,
            request_json=request_with(completion), workers=1, max_prompt_chars=100_000,
            deferred_retry_passes=0)
    assert not list(tmp_path.rglob("extraction_failure.json"))
    assert not (tmp_path / "batches/batch_00001/complete.json").exists()


@pytest.mark.parametrize("malformed_reduced_response", [False, True])
def test_serial_recovery_preserves_prior_field_and_reextracts_same_chunk(tmp_path, monkeypatch, malformed_reduced_response):
    install(monkeypatch)
    features = definitions()[:2]
    prior = {"rows": [{"row_id": 42,
        "values": {features[0]["name"]: 11, features[1]["name"]: 12},
        "carry_forward_state": {features[0]["name"]: "Earlier supported value.", features[1]["name"]: "Earlier value."}}]}
    calls = []

    def build(subset):
        return analysis._serial_extraction_prompt(definitions=subset, row_id=42,
            chunk_text="New observations: measurement 0 is 25.",
            prior_values=prior["rows"][0]["values"], prior_feature_state=prior["rows"][0]["carry_forward_state"],
            chunk_index=2, char_start=100, char_end=138, document_chars=138)

    def completion(messages, _config):
        body = prompt_inputs(messages)
        names = [d["name"] for d in body["features"]]
        calls.append(names)
        assert body["patient"]["current_chunk"] == "New observations: measurement 0 is 25."
        if len(names) == 1 and malformed_reduced_response:
            return "malformed JSON"
        values = {name.replace("_", " "): 25 for name in names}
        notes = {name.replace("_", " "): "New observation." for name in names}
        if len(names) == 2:
            notes.pop("abdomen or pelvis incidental findings")
        return json.dumps({"values": values, "decision_notes": notes})

    result = analysis._request_validated_extraction(messages=build(features), row_ids=[42],
        definitions=features, request_json=request_with(completion),
        ontology_audit_path=tmp_path / "category_ontology_repair.json",
        messages_for_definitions=build, prior_response=prior,
        validate_response=lambda value, *, definitions: analysis._validate_serial_extraction(
            value, row_id=42, definitions=definitions))
    assert list(map(len, calls)) == [2] * 4 + [1] * (16 if malformed_reduced_response else 1)
    assert result["rows"][0]["values"] == {
        features[0]["name"]: 11, features[1]["name"]: 12 if malformed_reduced_response else 25}
    assert result["rows"][0]["carry_forward_state"][features[0]["name"]] == "Earlier supported value."


def test_second_bad_field_can_be_excluded_from_reduced_batch(tmp_path, monkeypatch):
    install(monkeypatch)
    calls = []

    def completion(messages, _config):
        names = [d["name"] for d in prompt_inputs(messages)["features"]]
        calls.append(len(names))
        values = {name.replace("_", " "): 4 for name in names}
        if len(names) >= 9:
            values.pop(names[0].replace("_", " "))
        return json.dumps(values)

    frame = analysis.extract_rows(dataset=pd.DataFrame({"text": ["Recorded values."]}),
        row_ids=[0], text_column="text", definitions=definitions(), output_dir=tmp_path,
        request_json=request_with(completion), workers=1, max_prompt_chars=100_000)
    assert calls == [10] * 4 + [9] * 4 + [8]
    assert pd.isna(frame.loc[0, "measurement_0"])
    assert frame.loc[0, "measurement_1"] == 4
    summary = json.loads((tmp_path / "failure_summary.json").read_text())
    assert len(summary["feature_failure_patterns"]) == 2


@pytest.mark.parametrize("invalid_value", ["scalar", "category"])
@pytest.mark.parametrize("metadata", ["missing", "container", "incomplete"])
def test_serial_invalid_value_and_metadata_recover_prior_state(
    tmp_path, invalid_value, metadata,
):
    features = definitions()[:2]
    first, second = [feature["name"] for feature in features]
    prior_values = {first: 11, second: 12}
    if invalid_value == "category":
        features[0].update(value_type="categorical", categories_or_unit=["A", "B"])
        prior_values[first] = "A"
    prior_state = {first: "Earlier supported value.", second: "Earlier measurement."}
    prior = {"rows": [{"row_id": 42, "values": prior_values,
                       "carry_forward_state": prior_state}]}
    calls = []

    def completion(messages, config):
        # Missing serial metadata must get ordinary response repairs before
        # value repair/category mapping can use a partial response.
        assert config.runtime_request_kind == "extraction"
        calls.append(messages)
        row = {"row_id": 42, "values": {
            first: [99] if invalid_value == "scalar" else "Undeclared",
            second: 25,
        }}
        if metadata == "container":
            row["carry_forward_state"] = []
        elif metadata == "incomplete":
            row["carry_forward_state"] = {second: "New measurement."}
        return json.dumps({"rows": [row]})

    result = analysis._request_validated_extraction(
        messages=[{"role": "user", "content": "Update measurements from this chunk."}],
        row_ids=[42], definitions=features,
        request_json=request_with(completion, repairs=1),
        ontology_audit_path=tmp_path / "category_ontology_repair.json",
        messages_for_definitions=lambda _: [], prior_response=prior,
        validate_response=lambda value, *, definitions: analysis._validate_serial_extraction(
            value, row_id=42, definitions=definitions),
    )
    assert len(calls) == 2
    assert result == prior
    failure = json.loads((tmp_path / "extraction_failure.json").read_text())
    assert "carry_forward_state" in failure["validation_error"]
    assert not (tmp_path / "invalid_feature_value_repair.json").exists()
    assert not (tmp_path / "pending_category_ontology.json").exists()


def test_serial_value_recovery_retains_valid_new_values_and_normalized_metadata(tmp_path):
    features = definitions()[:2]
    first, second = [feature["name"] for feature in features]
    response = {"rows": [{"row_id": 42, "values": {first: [99], second: 25},
                          "carry_forward_state": {first: None, second: 25}}]}
    result = analysis._request_validated_extraction(
        messages=[{"role": "user", "content": "Update measurements from this chunk."}],
        row_ids=[42], definitions=features,
        request_json=request_with(lambda *_: json.dumps(response), repairs=1),
        ontology_audit_path=tmp_path / "category_ontology_repair.json",
        messages_for_definitions=lambda _: [],
        validate_response=lambda value, *, definitions: analysis._validate_serial_extraction(
            value, row_id=42, definitions=definitions),
    )
    assert result["rows"][0]["values"] == {first: None, second: 25}
    assert result["rows"][0]["carry_forward_state"] == {first: None, second: "25"}
    assert (tmp_path / "invalid_feature_value_repair.json").exists()


def test_legacy_pending_category_without_serial_metadata_reextracts(tmp_path):
    features = definitions()[:2]
    features[0].update(value_type="categorical", categories_or_unit=["A", "B"])
    first, second = [feature["name"] for feature in features]
    pending = tmp_path / "pending_category_ontology.json"
    pending.write_text(json.dumps({
        "schema_version": analysis.PENDING_CATEGORY_ONTOLOGY_SCHEMA_VERSION,
        "input_fingerprint": analysis._value_fingerprint({
            "row_ids": [42], "definitions": analysis._prompt_feature_definitions(features),
        }),
        "issues": [{"row_id": 42, "feature_name": first,
                    "prior_extracted_value": "Undeclared", "allowed_categories": ["A", "B"],
                    "definition": features[0]}],
        "response": {"rows": [{"row_id": 42, "values": {first: "Undeclared", second: 25}}]},
    }))
    calls = []
    fresh = {"rows": [{"row_id": 42, "values": {first: "A", second: 25},
                       "carry_forward_state": {first: "Documented A.", second: "New measurement."}}]}

    def completion(_messages, config):
        assert config.runtime_request_kind == "extraction"
        calls.append(config.runtime_request_kind)
        return json.dumps(fresh)

    result = analysis._request_validated_extraction(
        messages=[{"role": "user", "content": "Update measurements from this chunk."}],
        row_ids=[42], definitions=features, request_json=request_with(completion),
        ontology_audit_path=tmp_path / "category_ontology_repair.json",
        messages_for_definitions=lambda _: [],
        validate_response=lambda value, *, definitions: analysis._validate_serial_extraction(
            value, row_id=42, definitions=definitions),
    )
    assert calls == ["extraction"]
    assert result == fresh
    assert not pending.exists()
