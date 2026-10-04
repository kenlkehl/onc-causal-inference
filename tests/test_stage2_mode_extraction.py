import json

import pandas as pd
import pytest

from oci.inference import plain_handoff_stage2_analysis as analysis
from tests.stage2_prompt_spy import install, prompt_inputs
from tests.test_stage2_extraction_preparation import CharacterTokenizer, definition


class Tokenizer(CharacterTokenizer):
    def __call__(self, text, **kwargs):
        return {"input_ids": range(len(text))}


@pytest.mark.parametrize("tokenizer", [None, Tokenizer()])
def test_mixed_features_use_disjoint_prompts_and_resume_without_old_quote_failures(tmp_path, monkeypatch, tokenizer):
    install(monkeypatch)
    mode = definition()
    scalar = dict(definition(1), measurement_definition="Latest documented state.",
                  conflict_resolution={"strategy": "latest"})
    # Superseded all-variable extraction must not contaminate quality review.
    legacy = tmp_path / "pages" / "old" / "extraction_issues.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text(json.dumps({"events": [{"row_id": 0, "feature_name": "state_1",
        "failure_kind": "invalid_page_observation_provenance", "reason": "not an exact substring"}]}))
    calls = []

    def request(messages, validate, *, request_kind):
        body = prompt_inputs(messages)
        names = [d["name"] for d in body["features"]]
        calls.append((body["job"], names))
        if body["job"] == "extract_stage2_patient_variables":
            assert names == ["state_1"]
            assert "quotation" not in messages[0]["content"]
            return validate({"state 1": "blue"})
        assert names == ["state_0"]
        assert body["job"] == "extract_stage2_patient_variable_observations"
        return validate({"observations": [
            {"feature": "state 0", "value": value, "quote": quote, "governing_date_quote": None}
            for value, quote in [("red", "State: red."), ("red", "State: red."), ("blue", "State: blue.")]
        ]})

    args = dict(dataset=pd.DataFrame({"clinical_text": ["red red blue"]}), row_ids=[0],
                text_column="clinical_text", definitions=[mode, scalar], output_dir=tmp_path,
                request_json=request, workers=2, max_prompt_chars=100_000,
                tokenizer=tokenizer, context_window_tokens=100_000, max_output_tokens=1000)
    frame = analysis.extract_rows(**args)
    assert frame.loc[0, "state_0"] == "red" and frame.loc[0, "state_1"] == "blue"
    assert list(frame.columns) == ["_oci_row_id", "state_0", "state_1"]
    assert len(calls) == 2
    summary = json.loads((tmp_path / "failure_summary.json").read_text())
    assert summary["feature_failure_patterns"] == []
    pd.testing.assert_frame_equal(frame, analysis.extract_rows(**args))
    assert len(calls) == 2


def test_unmatched_repeated_quotes_keep_mode_counts_dates_and_stable_ids():
    page = {"row_id": 3, "text": "A clinical record.",
            "page": {"page_index": 2, "char_start": 100}}
    observations = [
        {"feature": "state 0", "value": value, "quote": "Reported state " + value,
         "governing_date_quote": date}
        for value, date in [("red", "2024-01-01"), ("red", "2024-01-01"),
                            ("blue", "2025-01-01")]
    ]
    result = analysis._validate_page_observations(
        {"observations": observations}, page=page, definitions=[definition()])
    rows = result["rows"][0]["observations"]
    assert len({r["observation_id"] for r in rows}) == 3
    assert all(r["source_start"] is None and r["recorded_at_source_start"] is None for r in rows)
    assert analysis._validate_page_observations(result, page=page, definitions=[definition()]) == result
    value, decision = analysis._resolve_feature_observations(definition=definition(), observations=rows)
    assert value == "red" and decision["observation_count"] == 3
    assert decision["selection_basis"] == "mode_then_reported_recorded_at"
    tied, _ = analysis._resolve_feature_observations(definition=definition(), observations=rows[1:])
    assert tied == "blue"
    for r in rows:
        r["recorded_at"] = None
    tied, decision = analysis._resolve_feature_observations(definition=definition(), observations=rows[1:])
    assert tied == "blue" and decision["selection_basis"] == "mode_then_reported_source_order"


@pytest.mark.parametrize("change", [{"quote": ""}, {"value": {"nested": "red"}}, {"feature": "unknown"}])
def test_relaxed_evidence_matching_preserves_structural_value_checks(change):
    observation = {"feature": "state 0", "value": "red", "quote": "State red", "governing_date_quote": None}
    observation.update(change)
    with pytest.raises(analysis._PageObservationValidationError):
        analysis._validate_page_observations({"observations": [observation]},
            page={"row_id": 0, "text": "red", "page": {"page_index": 1, "char_start": 0}},
            definitions=[definition()])


@pytest.mark.parametrize("date_quote", [2024, 2024.0, True, [], {}])
def test_nonstring_date_quote_retains_valid_observations_for_repair(date_quote):
    observations = [
        {"feature": "state 0", "value": "red", "quote": "State red",
         "governing_date_quote": "2024"},
        {"feature": "state 0", "value": "blue", "quote": "State blue",
         "governing_date_quote": date_quote},
    ]
    with pytest.raises(analysis._PageObservationValidationError) as caught:
        analysis._validate_page_observations(
            {"observations": observations},
            page={"row_id": 0, "text": "2024 State red. State blue.",
                  "page": {"page_index": 1, "char_start": 0}},
            definitions=[definition()],
        )
    issue, = caught.value.issues
    assert issue["observation_index"] == 2 and issue["feature_name"] == "state_0"
    assert issue["raw_observation"]["governing_date_quote"] == date_quote
    assert "quote string" in issue["reason"]
    retained, = caught.value.response["rows"][0]["observations"]
    assert retained["value"] == "red" and retained["recorded_at"] == "2024"


def test_changed_mode_contract_rejects_old_completion_checkpoint(tmp_path, monkeypatch):
    install(monkeypatch)
    calls = []
    def request(messages, validate, *, request_kind):
        calls.append(messages)
        return validate({"observations": [{"feature": "state 0", "value": "red",
            "quote": "Red", "governing_date_quote": None}]})
    args = dict(dataset=pd.DataFrame({"clinical_text": ["red"]}), row_ids=[0],
                text_column="clinical_text", definitions=[definition()], output_dir=tmp_path,
                request_json=request, workers=1, max_prompt_chars=100_000)
    analysis.extract_rows(**args)
    path = tmp_path / "pages/row_00000000/page_00001/complete.json"
    complete = json.loads(path.read_text())
    complete["schema_version"] = "stage2_single_patient_page_observations_v5_provenance_clinical_prompts_20260923"
    path.write_text(json.dumps(complete))
    analysis.extract_rows(**args)
    assert len(calls) == 2
