from __future__ import annotations

from tests.stage2_prompt_spy import prompt_inputs

import json
from types import SimpleNamespace as NS

import pandas as pd
import pytest

from oci.inference import plain_handoff_stage2 as workflow
from oci.inference import plain_handoff_stage2_analysis as analysis


FEATURES = [{
    "feature_id": "f_ecog", "name": "ecog", "description": "Performance status",
    "value_type": "continuous", "categories_or_unit": ["score"],
    "measurement_definition": "Extract ECOG.",
    "missing_value_rule": "Return null when undocumented.",
}]


def config(**kwargs):
    return workflow.PlainHandoffStage2Config(
        endpoint="http://stage2.test/v1", model="test-model", **kwargs)


@pytest.mark.parametrize("value,rows", [({"row_id": 8, "values": {"ecog": 1}}, [7])])
def test_wrapper_repair_cannot_invent_values_or_reassign_patient(value, rows):
    with pytest.raises((ValueError, TypeError)):
        analysis._validate_extraction(value, row_ids=rows, definitions=FEATURES)


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


@pytest.mark.parametrize("finish", ["length"])
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


def test_unresolved_patient_blocks_completion_and_resume_reuses_other_patient(tmp_path):
    calls = []

    def failing(messages, validate, **_kwargs):
        row = prompt_inputs(messages)["patients"][0]["row_id"]
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
        row = prompt_inputs(messages)["patients"][0]["row_id"]
        calls.append(row)
        return validate({"rows": [{"row_id": row, "values": {"ecog": row}}]})

    extract(tmp_path, working, max_output_tokens=4096)
    assert calls == [0]
    assert complete.read_bytes() == original
    record = json.loads((tmp_path / "deferred_extraction.json").read_text())
    assert record["status"] == "resolved"
    assert len(record["failures"]) == 2
