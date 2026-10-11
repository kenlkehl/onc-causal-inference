import json
import threading
import time
from types import SimpleNamespace as NS

import pandas as pd
import pytest

from oci.inference import plain_handoff_stage2_analysis as analysis
from oci.inference import plain_handoff_stage2 as workflow
from tests.stage2_prompt_spy import install, prompt_inputs
from tests.test_stage2_extraction_preparation import CharacterTokenizer, definition


class Tokenizer(CharacterTokenizer):
    def __call__(self, text, **kwargs):
        return {"input_ids": range(len(text))}


def test_page_prompt_reminder_and_schema_preserve_supported_occurrences():
    import jsonschema
    feature = {**definition(), "value_type": "ordinal", "categories_or_unit": ["0", "1"]}
    source = "FIRST MARKER\n" + "Clinical text. " * 100 + "\nLAST MARKER"
    messages = analysis._page_extraction_prompt(definitions=[feature], row={"text": source})
    assert source in messages[-1]["content"]
    suffix = messages[-1]["content"].split("LAST MARKER", 1)[1]
    assert "governing_date_quote" in suffix and "Omit missing-value placeholders" in suffix
    schema = analysis._page_response_schema([feature])
    observation = {"feature": "state 0", "value": 1, "quote": "Score 1", "governing_date_quote": None}
    # Numeric category labels, repeated occurrences, empty arrays, and null
    # placeholders must all be representable without trapping the decoder.
    for observations in [[], [observation, observation], [{**observation, "value": None, "quote": None}]]:
        jsonschema.validate({"observations": observations}, schema)
    for bad in [{**observation, "feature": "unknown"}, {**observation, "governing_date": None},
                {k: v for k, v in observation.items() if k != "governing_date_quote"}]:
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({"observations": [bad]}, schema)


def test_page_schema_reaches_transport_and_repairs_keep_observation_contract(monkeypatch, tmp_path):
    import openai
    sent = []
    observation = {"feature": "state 0", "value": "red", "quote": "State red.", "governing_date_quote": None}

    class Client:
        def __init__(self, **kwargs):
            self.chat = NS(completions=NS(create=self.create))

        def create(self, **kwargs):
            sent.append(kwargs)
            # A server may ignore the schema; semantic repairs still apply.
            answer = {**observation, "value": "invalid"} if len(sent) <= 6 else observation
            return NS(id="test", usage=None, choices=[NS(finish_reason="stop",
                message=NS(content=json.dumps({"observations": [answer]})))])

        def close(self):
            pass

    monkeypatch.setattr(openai, "OpenAI", Client)
    config = workflow.PlainHandoffStage2Config(endpoint="http://test/v1",
        model="nvidia/Gemma-4-26B-A4B-NVFP4", thinking_after_response_repairs=5)

    def request(messages, validate, **kwargs):
        return workflow._request_json(messages=messages, validate=validate, config=config,
            completion=workflow._openai_completion, prompt_token_counter=lambda messages: 100,
            context_window_tokens=262144, **kwargs)

    page = {"row_id": 0, "text": "State red.", "page": {"page_index": 1, "char_start": 0}}
    result = analysis._request_validated_page_observations(
        messages=analysis._page_extraction_prompt(definitions=[definition()], row=page),
        page=page, definitions=[definition()], request_json=request, audit_dir=tmp_path)
    assert result["rows"][0]["observations"][0]["value"] == "red"
    assert all(r["response_format"]["json_schema"]["schema"] == analysis._page_response_schema([definition()])
               for r in sent)
    assert [r["extra_body"]["chat_template_kwargs"]["enable_thinking"] for r in sent] == [False] * 6 + [True]
    assert all("governing_date_quote" in r["messages"][-1]["content"] for r in sent[1:])


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


def test_missing_placeholders_do_not_trigger_repairs_or_change_mode_votes(tmp_path):
    feature = {**definition(), "categories_or_unit": ["never", "daily"]}
    page = {"row_id": 0, "text": "Never. Daily. Never.", "page": {"page_index": 1, "char_start": 0}}
    observations = [
        {"feature": "state 0", "value": value, "quote": quote, "governing_date_quote": None}
        for value, quote in [(None, None), ("never", "Never."), ("unknown", "No value."),
                             ("daily", "Daily."), ("never", "Never.")]
    ]
    observations[0]["governing_date_quote"] = 2024  # Irrelevant metadata on a missing value.
    calls = []

    def request(messages, validate, *, request_kind):
        calls.append(messages)
        return validate({"observations": observations})

    result = analysis._request_validated_page_observations(messages=[], page=page,
        definitions=[feature], request_json=request, audit_dir=tmp_path)
    assert len(calls) == 1
    retained = result["rows"][0]["observations"]
    assert [x["value"] for x in retained] == ["never", "daily", "never"]
    assert analysis._validate_page_observations(result, page=page, definitions=[feature]) == result
    value, decision = analysis._resolve_feature_observations(definition=feature, observations=retained)
    assert value == "never" and decision["observation_count"] == 3
    events = [json.loads(line) for line in (tmp_path / "request_events.jsonl").read_text().splitlines()]
    assert any(e["event"] == "page_missing_observations_omitted" and e["count"] == 2 for e in events)
    assert json.loads((tmp_path / "extraction_issues.json").read_text())["events"] == []


def test_all_missing_observations_normalize_to_empty_but_declared_none_is_a_value():
    page = {"row_id": 0, "text": "No findings.", "page": {"page_index": 1, "char_start": 0}}
    missing = {"feature": "state 0", "value": None, "quote": None, "governing_date_quote": None}
    result = analysis._validate_page_observations({"observations": [missing] * 5}, page=page, definitions=[definition()])
    assert result["rows"][0]["observations"] == []
    negative = {**missing, "value": "None", "quote": "No findings."}
    feature = {**definition(), "categories_or_unit": ["None", "Present"]}
    result = analysis._validate_page_observations({"observations": [negative]}, page=page, definitions=[feature])
    assert result["rows"][0]["observations"][0]["value"] == "None"
    for bad in [{**missing, "feature": "unknown"}, {k: v for k, v in missing.items() if k != "value"}]:
        with pytest.raises(analysis._PageObservationValidationError):
            analysis._validate_page_observations({"observations": [bad]}, page=page, definitions=[feature])


@pytest.mark.parametrize("threshold", [0, 2, 5])
def test_page_errors_hand_off_after_configured_off_repairs_and_one_thinking_repair(tmp_path, threshold):
    page = {"row_id": 0, "text": "State red.", "page": {"page_index": 1, "char_start": 0}}
    policies = []

    def completion(messages, config):
        if policies:
            assert "Omit null or unavailable placeholders" in messages[-1]["content"]
            assert "invalid" in messages[-1]["content"]
        policies.append(workflow._stage2_request_policy(config)["reasoning_effort"])
        # Changing the invalid value/index must not restart the repair budget.
        valid = {"feature": "state 0", "value": "red", "quote": "State red.", "governing_date_quote": None}
        bad = {**valid, "value": "invalid " + str(len(policies))}
        return json.dumps({"observations": [valid, bad] if len(policies) % 2 else [bad, valid]})

    config = workflow.PlainHandoffStage2Config(endpoint="http://test/v1", model="test",
        thinking_after_response_repairs=threshold)

    def request(messages, validate, **kwargs):
        return workflow._request_json(messages=messages, validate=validate, config=config,
                                      completion=completion, **kwargs)

    result = analysis._request_validated_page_observations(messages=[], page=page,
        definitions=[definition()], request_json=request, audit_dir=tmp_path)
    assert policies == ["none"] * (1 + threshold) + ["high"]
    assert [o["value"] for o in result["rows"][0]["observations"]] == ["red"]
    ledger = json.loads((tmp_path / "extraction_issues.json").read_text())
    assert len(ledger["events"]) == 1 and ledger["events"][0]["failure_kind"] == "invalid_page_observation"
    assert (tmp_path / "invalid_page_observation_repair.json").exists()


@pytest.mark.parametrize("workers", [1, 2, 4])
@pytest.mark.parametrize("cached_scalar", [False, True])
def test_mixed_strategies_share_limit_and_mode_can_use_all_idle_slots(tmp_path, monkeypatch, workers, cached_scalar):
    install(monkeypatch)
    mode = definition()
    scalar = {**definition(1), "measurement_definition": "Latest documented state.",
              "conflict_resolution": {"strategy": "latest"}}
    args = dict(dataset=pd.DataFrame({"clinical_text": ["State red."] * workers}),
        row_ids=list(range(workers)), text_column="clinical_text", workers=workers,
        max_prompt_chars=100_000)
    if cached_scalar:
        analysis.extract_rows(**args, definitions=[scalar], output_dir=tmp_path / "by_strategy/values",
            request_json=lambda messages, validate, **kwargs: validate({"state 1": "red"}))
    lock = threading.Lock()
    barrier = threading.Barrier(workers)
    active = peak = mode_active = mode_peak = 0
    calls = []

    def request(messages, validate, *, request_kind):
        nonlocal active, peak, mode_active, mode_peak
        body = prompt_inputs(messages)
        is_mode = body["job"] == "extract_stage2_patient_variable_observations"
        with lock:
            calls.append(is_mode)
            active += 1
            mode_active += int(is_mode)
            peak = max(peak, active)
            mode_peak = max(mode_peak, mode_active)
        try:
            if is_mode:
                # A fixed small mode partition would deadlock here; idle scalar
                # capacity must be usable by every pending mode patient.
                barrier.wait(timeout=5)
                time.sleep(0.02)
                return validate({"observations": [{"feature": "state 0", "value": "red",
                    "quote": "State red.", "governing_date_quote": None}]})
            assert not cached_scalar, "completed scalar measurements must be reused"
            time.sleep(0.01)
            return validate({"state 1": "red"})
        finally:
            with lock:
                active -= 1
                mode_active -= int(is_mode)

    result = analysis.extract_rows(**args, definitions=[mode, scalar], output_dir=tmp_path, request_json=request)
    assert peak <= workers and mode_peak == workers
    assert calls.count(True) == workers and calls.count(False) == (0 if cached_scalar else workers)
    assert result[["state_0", "state_1"]].values.tolist() == [["red", "red"]] * workers
