from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import threading
import time
from types import SimpleNamespace as NS

import pandas as pd
import pytest

from oci.inference import plain_handoff_stage2 as workflow
from oci.inference import plain_handoff_stage2_analysis as analysis
from oci.inference import stage2_request_audit as audit


FEATURES = [{
    "feature_id": "f_ecog", "name": "ecog", "description": "Performance status",
    "value_type": "continuous", "categories_or_unit": ["score"],
    "measurement_definition": "Extract ECOG.",
    "missing_value_rule": "Return null when undocumented.",
}]


def config(**kwargs):
    return workflow.PlainHandoffStage2Config(
        endpoint="http://stage2.test/v1", model="test-model", **kwargs)


def events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.parametrize("value", [
    {"ecog": 1}, {"values": {"ecog": 1}},
    {"row_id": 7, "values": {"ecog": 1}},
    {"rows": {"row_id": 7, "values": {"ecog": 1}}},
])
def test_exact_single_patient_wrappers_preserve_measurements(value, tmp_path):
    journal = tmp_path / "events.jsonl"
    with audit.context(_audit_path=journal, patient_row_ids=[7]):
        result = analysis._validate_extraction(value, row_ids=[7], definitions=FEATURES)
    assert result == {"rows": [{"row_id": 7, "values": {"ecog": 1}}]}
    assert events(journal)[0]["event"] == "response_shape_normalized"
    assert events(journal)[0]["inferred_measurements"] == 0


@pytest.mark.parametrize("value,rows", [
    ({"values": {}}, [7]),
    ({"ecog": 1, "other": 2}, [7]),
    ({"row_id": 8, "values": {"ecog": 1}}, [7]),
    ({"ecog": 1}, [7, 8]),
    ({"values": {"ecog": [1, 2]}}, [7]),
    ({"rows": [{"row_id": 8, "values": {"ecog": 1}}], "ecog": 1}, [7]),
])
def test_wrapper_repair_cannot_invent_values_or_reassign_patient(value, rows):
    with pytest.raises((ValueError, TypeError)):
        analysis._validate_extraction(value, row_ids=rows, definitions=FEATURES)


def test_wrapper_repair_still_validates_category():
    definitions = [{**FEATURES[0], "value_type": "categorical",
                    "categories_or_unit": ["good", "poor"]}]
    with pytest.raises(analysis._ExtractionCategoryError):
        analysis._validate_extraction({"ecog": "unexpected"}, row_ids=[7], definitions=definitions)


@pytest.mark.parametrize("error", [
    "extraction response requires a rows array",
    "each extraction row requires a values object",
    "each extraction row must be an object",
    "unsupported measurement",
])
def test_all_extraction_repairs_enable_thinking_with_larger_budget_after_threshold(tmp_path, error):
    policies = []
    prompts = []

    def completion(messages, cfg):
        policies.append(workflow._stage2_request_policy(cfg))
        prompts.append(messages)
        return "{}"

    def validate(value):
        if len(policies) < 8:
            raise ValueError(error)
        return value

    path = tmp_path / "events.jsonl"
    with audit.context(_audit_path=path, patient_row_ids=[7], feature_ids=["f_ecog"]):
        workflow._request_json(
            messages=[{"role": "user", "content": "PRIVATE NOTE"}],
            config=config(extraction_max_tokens=4096, extraction_reasoning_max_tokens=32768),
            completion=completion, validate=validate, request_kind="extraction")
    # Initial response plus repairs 1–5 use the ordinary extraction policy.
    assert [p["reasoning_effort"] for p in policies] == ["none"] * 6 + ["high"] * 2
    assert [p["max_tokens"] for p in policies] == [4096] * 6 + [32768] * 2
    for prompt in prompts[1:]:
        assert prompt[0]["content"] == "PRIVATE NOTE"
        assert prompt[-2] == {"role": "assistant", "content": "{}"}
        assert f"ValueError: {error}" in prompt[-1]["content"]
    records = events(path)
    assert records[-1]["event"] == "request_validated"
    assert records[-1]["response_attempt"] == 8
    assert len({r["request_id"] for r in records}) == 1
    assert "PRIVATE NOTE" not in path.read_text()


def test_parallel_requests_have_distinct_ids_and_persist_failure(tmp_path):
    def worker(row):
        with audit.context(_audit_path=tmp_path / f"{row}.jsonl", patient_row_ids=[row]):
            with pytest.raises(analysis.Stage2RequestExhaustedError):
                workflow._request_json(
                    messages=[], config=config(), validate=dict,
                    completion=lambda *_: (_ for _ in ()).throw(
                        analysis.Stage2RequestExhaustedError("logical deadline expired")))
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(worker, [1, 2]))
    records = [events(tmp_path / f"{row}.jsonl") for row in [1, 2]]
    assert records[0][-1]["request_id"] != records[1][-1]["request_id"]
    for row, group in zip([1, 2], records):
        assert group[-1]["event"] == "request_failed"
        assert all(r["patient_row_ids"] == [row] for r in group)


def chunk(content="", reasoning="", finish=None, usage=None):
    return NS(id="server-id", usage=usage, choices=[NS(
        index=0, finish_reason=finish, delta=NS(content=content, reasoning_content=reasoning))])


class Stream:
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False

    def __iter__(self):
        yield from self.chunks

    def close(self):
        self.closed = True


def collect(stream, deadline=None):
    return audit.collect_stream(
        stream, deadline=deadline or time.monotonic() + 60,
        deadline_error=analysis.Stage2RequestExhaustedError,
        incomplete_error=workflow._RetryableStage2ResponseError)


def test_stream_records_reasoning_progress_and_final_usage(tmp_path):
    stream = Stream([
        chunk(reasoning="private reasoning"), chunk('{"ok":'), chunk("true}", finish="stop"),
        NS(id="server-id", choices=[], usage=NS(prompt_tokens=20, completion_tokens=8, total_tokens=28)),
    ])
    path = tmp_path / "events.jsonl"
    with audit.context(_audit_path=path):
        response = collect(stream)
    assert stream.closed
    assert response.choices[0].message.content == '{"ok":true}'
    assert response.choices[0].message.reasoning_content == "private reasoning"
    assert audit.usage_fields(response.usage)["completion_tokens"] == 8
    assert events(path)[0]["reasoning_chars"] == len("private reasoning")
    assert "private reasoning" not in path.read_text()


def test_incomplete_stream_cannot_be_accepted_and_logs_partial_progress(tmp_path):
    stream = Stream([chunk('{"ok":true}')])
    path = tmp_path / "events.jsonl"
    with audit.context(_audit_path=path), pytest.raises(workflow._RetryableStage2ResponseError):
        collect(stream)
    assert stream.closed
    failure = events(path)[-1]
    assert failure["event"] == "stream_failed"
    assert failure["content_chars"] == 11
    assert failure["completion_tokens"] is None


def test_stream_watchdog_closes_silent_stream_at_logical_deadline():
    class SilentStream(Stream):
        released = threading.Event()

        def __iter__(self):
            assert self.released.wait(timeout=2), "Watchdog failed to close the blocked stream"
            yield from ()

        def close(self):
            super().close()
            self.released.set()

    stream = SilentStream([])
    with pytest.raises(analysis.Stage2RequestExhaustedError, match="logical deadline"):
        collect(stream, deadline=time.monotonic() + 0.02)
    assert stream.closed


def test_stream_read_failure_is_retryable_and_preserves_partial_audit(tmp_path):
    import httpx

    class BrokenStream(Stream):
        def __iter__(self):
            yield chunk(reasoning="in flight")
            raise httpx.ReadTimeout("stream stopped responding")

    path = tmp_path / "events.jsonl"
    stream = BrokenStream([])
    with audit.context(_audit_path=path), pytest.raises(httpx.ReadTimeout) as caught:
        collect(stream)
    assert workflow._is_retryable_transport_error(caught.value)
    assert events(path)[-1]["reasoning_chars"] == len("in flight")
    assert events(path)[-1]["event"] == "stream_failed"
    assert stream.closed


@pytest.mark.parametrize("finish", ["stop", "length"])
def test_openai_streaming_keeps_cap_and_rejects_truncated_json(monkeypatch, finish):
    import openai
    requests = []
    stream = Stream([chunk('{"ok":true}', finish=finish)])

    class Client:
        def __init__(self, **kwargs):
            self.chat = NS(completions=NS(create=self.create))

        def create(self, **kwargs):
            requests.append(kwargs)
            return stream

        def close(self):
            pass

    monkeypatch.setattr(openai, "OpenAI", Client)
    cfg = config(extraction_stream=True, extraction_max_tokens=4096, runtime_request_kind="extraction")
    if finish == "length":
        with pytest.raises(workflow._Stage2OutputLengthError):
            workflow._openai_completion([], cfg)
    else:
        assert workflow._openai_completion([], cfg) == '{"ok":true}'
    assert stream.closed
    assert requests[0]["stream"] is True
    assert requests[0]["stream_options"] == {"include_usage": True}
    assert requests[0]["max_tokens"] == 4096


def extract(output, request, **kwargs):
    return analysis.extract_rows(
        dataset=pd.DataFrame({"text": ["ECOG 0", "ECOG 1"]}), row_ids=[0, 1],
        text_column="text", definitions=FEATURES, output_dir=output,
        request_json=request, workers=1, max_prompt_chars=10000, **kwargs)


def test_exhausted_patient_is_deferred_until_other_patients_finish(tmp_path):
    calls = []

    def request(messages, validate, **_kwargs):
        row = json.loads(messages[-1]["content"])["patients"][0]["row_id"]
        calls.append(row)
        if calls == [0]:
            raise analysis.Stage2RequestExhaustedError("one stalled call")
        return validate({"rows": [{"row_id": row, "values": {"ecog": row}}]})

    result = extract(tmp_path, request)
    assert calls == [0, 1, 0]
    assert result["ecog"].tolist() == [0, 1]
    assert json.loads((tmp_path / "deferred_extraction.json").read_text())["status"] == "resolved"


def test_unresolved_patient_blocks_completion_and_resume_reuses_other_patient(tmp_path):
    calls = []

    def failing(messages, validate, **_kwargs):
        row = json.loads(messages[-1]["content"])["patients"][0]["row_id"]
        calls.append(row)
        if row == 0:
            raise analysis.Stage2RequestExhaustedError("still stalled")
        return validate({"rows": [{"row_id": row, "values": {"ecog": row}}]})

    with pytest.raises(analysis.Stage2RequestExhaustedError):
        extract(tmp_path, failing)
    assert calls == [0, 1, 0]
    assert not (tmp_path / "complete.json").exists()
    assert not (tmp_path / "extracted.csv").exists()
    complete = tmp_path / "batches/batch_00002/complete.json"
    original = complete.read_bytes()
    calls.clear()

    def working(messages, validate, **_kwargs):
        row = json.loads(messages[-1]["content"])["patients"][0]["row_id"]
        calls.append(row)
        return validate({"rows": [{"row_id": row, "values": {"ecog": row}}]})

    extract(tmp_path, working, max_output_tokens=4096)
    assert calls == [0]
    assert complete.read_bytes() == original
    record = json.loads((tmp_path / "deferred_extraction.json").read_text())
    assert record["status"] == "resolved"
    assert len(record["failures"]) == 2


def test_persistent_outage_trips_circuit_breaker(tmp_path):
    def failing(*_args, **_kwargs):
        raise analysis.Stage2RequestExhaustedError("offline")
    with pytest.raises(analysis.Stage2RequestExhaustedError):
        analysis.extract_rows(
            dataset=pd.DataFrame({"text": ["ECOG 1"] * 8}), row_ids=list(range(8)),
            text_column="text", definitions=FEATURES, output_dir=tmp_path,
            request_json=failing, workers=1, max_prompt_chars=10000)
    journal = json.loads((tmp_path / "deferred_extraction.json").read_text())
    assert journal["status"] == "aborted_consecutive_failures"
    assert journal["retry_pass"] == 0
    assert len(journal["failures"]) == 3
    assert not (tmp_path / "complete.json").exists()


@pytest.mark.parametrize("field,value", [
    ("extraction_stream", "true"), ("extraction_stream", 1),
    ("extraction_deferred_retry_passes", -1), ("extraction_deferred_retry_passes", True),
    ("extraction_deferred_retry_passes", 0.5),
])
def test_invalid_reliability_settings_are_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        workflow.plain_stage2_config_from_mapping(
            {"endpoint": "http://stage2.test/v1", "model": "test-model", field: value},
            default_workers=1)


def test_reliability_settings_round_trip_and_reach_extraction():
    cfg = workflow.plain_stage2_config_from_mapping(
        {"endpoint": "http://stage2.test/v1", "model": "test-model",
         "extraction_stream": True, "extraction_deferred_retry_passes": 0,
         "extraction_max_tokens": 4096}, default_workers=1)
    assert cfg.extraction_stream is True
    assert analysis._configured_serial_extraction(cfg)["deferred_retry_passes"] == 0
    assert analysis._configured_serial_extraction(cfg)["max_output_tokens"] == 4096


@pytest.mark.parametrize("effort,cap", [("none", 4096), ("high", 32768)])
def test_reasoning_has_its_own_total_output_budget(effort, cap):
    cfg = config(extraction_max_tokens=4096, extraction_reasoning_max_tokens=32768,
                 extraction_reasoning_effort=effort)
    assert workflow._stage2_request_policy(cfg, "extraction")["max_tokens"] == cap
    assert workflow._stage2_request_policy(cfg, "interpretation")["max_tokens"] == 100000
    assert analysis._configured_serial_extraction(cfg)["max_output_tokens"] == 32768


@pytest.mark.parametrize("context_window,expected", [(131072, 32768), (20000, 18000)])
def test_reasoning_repair_uses_larger_cap_within_remaining_context(context_window, expected):
    caps = []

    def completion(_messages, cfg):
        caps.append(workflow._stage2_request_policy(cfg)["max_tokens"])
        return json.dumps({"valid": len(caps) > 1})

    def validate(value):
        if not value["valid"]:
            raise ValueError("unsupported measurement")
        return value

    workflow._request_json(
        messages=[], config=config(extraction_max_tokens=4096, extraction_reasoning_max_tokens=32768,
                                   thinking_after_response_repairs=0),
        completion=completion, validate=validate, request_kind="extraction",
        prompt_token_counter=lambda _: 1000, context_window_tokens=context_window,
        context_margin_tokens=1000)
    assert caps == [4096, expected]


@pytest.mark.parametrize("value", [True, 100, -1, "32768", 32768.5])
def test_invalid_reasoning_cap_is_rejected(value):
    with pytest.raises(ValueError, match="extraction_reasoning_max_tokens"):
        workflow.plain_stage2_config_from_mapping(
            {"endpoint": "http://stage2.test/v1", "model": "test-model",
             "extraction_reasoning_max_tokens": value}, default_workers=1)


def test_omitted_reasoning_cap_preserves_existing_budget():
    assert workflow._stage2_request_policy(
        config(extraction_max_tokens=75000, extraction_reasoning_effort="high"),
        "extraction")["max_tokens"] == 75000
