"""Exercise measurement recovery and isolation boundaries, without a live LLM."""
from dataclasses import replace
import json

import pandas as pd
import pytest

from oci.inference import plain_handoff_stage2 as workflow
from oci.inference import plain_handoff_stage2_analysis as analysis
from oci.inference import stage2_note_search as search


FEATURES = [{
    "name": "ecog", "value_type": "continuous", "categories_or_unit": ["score"],
    "description": "ECOG performance status", "measurement_definition": "Extract ECOG.",
    "missing_value_rule": "Null when unsupported.", "conflict_resolution": {"strategy": "latest"},
}, {
    "name": "age", "value_type": "continuous", "categories_or_unit": ["years"],
    "description": "Age", "measurement_definition": "Extract age.",
    "missing_value_rule": "Null when unsupported.", "conflict_resolution": {"strategy": "latest"},
}]


@pytest.fixture
def fake_backend(monkeypatch):
    workers = []

    class Worker:
        def __init__(self, history, limits):
            self.history, self.codes, self.closed = history, [], False
            workers.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.closed = True

        def execute(self, code):
            self.codes.append(code)
            start = self.history.find("ECOG")
            return {
                "output": "bounded search observation", "error_detail": None,
                "error": "SyntaxError" if code == "broken" else None,
                "truncated": False, "sources_truncated": False,
                "source_spans": [[start, len(self.history)]] if start >= 0 and code != "broken" else [],
            }

    monkeypatch.setattr(search, "load_backend", lambda config: (Worker, {"worker_sha256": "test"}))
    return workers


def arguments(tmp_path, request, **kwargs):
    return dict(
        dataset=pd.DataFrame({"text": ["UNREAD PREFIX " * 1000 + "ECOG 2. Age 67."]}),
        row_ids=[0], text_column="text", definitions=FEATURES,
        output_dir=tmp_path, request_json=request, workers=1, max_prompt_chars=100_000,
        note_search=search.NoteSearchConfig(enabled=True, max_review_passes=0), deferred_retry_passes=0, **kwargs,
    )


def python_action(code="search variables"):
    return {"action": "python", "code": [code], "memory": "Search the declared measurements."}


def test_batch_retrieval_values_provenance_resume_and_patient_isolation(tmp_path, monkeypatch, fake_backend):
    calls = []
    monkeypatch.setattr(analysis, "_serial_extraction_required", lambda **_: pytest.fail("No full-record planning needed"))

    def request(messages, validate, *, request_kind):
        assert request_kind == "extraction"
        text = messages[-1]["content"]
        assert "UNREAD PREFIX" not in text
        calls.append(text)
        if "Search cells remaining: 3" in text:
            assert "ECOG 2. Age 67." not in text and "ECOG 3. Age 58." not in text
            return validate(python_action())
        assert "[Document segment 1; original characters 14000:" in text
        values = {"ecog": 2, "age": 67} if "ECOG 2." in text else {"ecog": 3, "age": 58}
        return validate({"action": "final", "values": values})

    args = arguments(tmp_path, request)
    args.update(dataset=pd.DataFrame({"text": ["UNREAD PREFIX " * 1000 + "ECOG 2. Age 67.",
                                               "UNREAD PREFIX " * 1000 + "ECOG 3. Age 58."]}),
                row_ids=[1, 0], workers=2)
    first = analysis.extract_rows(**args)
    assert first["ecog"].tolist() == [3, 2]
    assert first["age"].tolist() == [58, 67]
    assert len(calls) == 4 and all(w.closed for w in fake_backend)
    for path in tmp_path.rglob("reviewed_excerpts.json"):
        item = json.loads(path.read_text())["excerpts"][0]
        assert item["text"] in {"ECOG 2. Age 67.", "ECOG 3. Age 58."}
        assert item["start"] == len("UNREAD PREFIX " * 1000)
    pd.testing.assert_frame_equal(first, analysis.extract_rows(**args))
    assert len(calls) == 4


def test_endpoint_failure_resumes_saved_search_without_new_search_request(tmp_path, fake_backend):
    calls = []
    outage = True

    def request(messages, validate, *, request_kind):
        calls.append(messages[-1]["content"])
        if "Search cells remaining: 3" in calls[-1]:
            return validate(python_action())
        if outage:
            raise analysis.Stage2RequestExhaustedError("endpoint unavailable")
        return validate({"action": "final", "values": {"ecog": 2, "age": 67}})

    args = arguments(tmp_path, request)
    with pytest.raises(analysis.Stage2RequestExhaustedError):
        analysis.extract_rows(**args)
    assert not (tmp_path / "complete.json").exists()
    outage = False
    result = analysis.extract_rows(**args)
    assert result.loc[0, "ecog"] == 2
    assert len(calls) == 3
    assert len(fake_backend) == 2 and [len(w.codes) for w in fake_backend] == [1, 1]
    assert all(w.closed for w in fake_backend)


def test_named_field_failure_recovers_remaining_measurements(tmp_path, fake_backend):
    calls = []

    def completion(messages, config):
        text = messages[1]["content"]
        calls.append(text)
        if "Search cells remaining: 3" in text:
            return json.dumps(python_action())
        # Deliberately omit one variable until OCI retries just the valid field.
        return json.dumps({"action": "final", "values": {"ecog": 2}})

    config = workflow.PlainHandoffStage2Config(endpoint="http://unused.test/v1", model="test", max_response_repairs=4)
    def request(messages, validate, *, request_kind):
        return workflow._request_json(messages=messages, config=config, completion=completion,
                                      validate=validate, request_kind=request_kind)

    result = analysis.extract_rows(**arguments(tmp_path, request))
    assert result.loc[0, "ecog"] == 2 and pd.isna(result.loc[0, "age"])
    assert len(fake_backend[0].codes) == 1
    audit = json.loads((tmp_path / "batches/batch_00001/field_recovery.json").read_text())
    assert audit["excluded_features"] == ["age"]


def test_failed_cells_are_errors_while_successful_empty_search_can_return_null(tmp_path, fake_backend):
    def broken(messages, validate, *, request_kind):
        return validate(python_action("broken"))

    with pytest.raises(analysis.Stage2InfrastructureError, match="Every note-search cell failed"):
        analysis.extract_rows(**arguments(tmp_path / "broken", broken))
    assert not (tmp_path / "broken/extracted.csv").exists()

    def empty(messages, validate, *, request_kind):
        if "Search cells remaining: 3" in messages[-1]["content"]:
            return validate(python_action())
        return validate({"action": "final", "values": {"ecog": None, "age": None}})

    args = arguments(tmp_path / "empty", empty)
    args["dataset"] = pd.DataFrame({"text": ["No relevant measurement supplied."]})
    result = analysis.extract_rows(**args)
    assert result[["ecog", "age"]].isna().all().all()


def test_method_and_budget_changes_cannot_reuse_existing_measurements(tmp_path, fake_backend):
    config = search.NoteSearchConfig(enabled=True)
    search.claim_output_method(tmp_path, config)
    for changed in [search.NoteSearchConfig(), replace(config, max_cells=4)]:
        with pytest.raises(ValueError, match="fresh extraction output directory"):
            search.claim_output_method(tmp_path, changed)
    old = tmp_path / "legacy"
    (old / "batches").mkdir(parents=True)
    with pytest.raises(ValueError, match="fresh extraction output directory"):
        search.claim_output_method(old, config)


def test_mode_variables_keep_full_record_observation_path(tmp_path, monkeypatch, fake_backend):
    from tests.stage2_prompt_spy import install, prompt_inputs
    install(monkeypatch)
    modes = [{**FEATURES[0], "conflict_resolution": {"strategy": "mode"}}]

    def request(messages, validate, *, request_kind):
        body = prompt_inputs(messages)
        assert body["patient"]["text"] == "ECOG 1. ECOG 2. ECOG 2."
        return validate({"observations": [{"feature": "ecog", "value": n,
            "quote": f"ECOG {n}", "governing_date_quote": None} for n in [1, 2, 2]]})

    args = arguments(tmp_path, request)
    args.update(dataset=pd.DataFrame({"text": ["ECOG 1. ECOG 2. ECOG 2."]}), definitions=modes)
    result = analysis.extract_rows(**args)
    assert result.loc[0, "ecog"] == 2
    assert not fake_backend


def test_config_roundtrip_routes_search_to_training_refinement_and_heldout(fake_backend):
    cfg = workflow.plain_stage2_config_from_mapping({
        "endpoint": "http://unused.test/v1", "model": "test",
        "extraction_note_search": {"enabled": True, "max_cells": 2},
    }, default_workers=1)
    assert cfg.extraction_note_search.max_cells == 2
    assert cfg.public_dict()["extraction_note_search"]["enabled"] is True
    assert analysis._configured_serial_extraction(cfg)["note_search"] == cfg.extraction_note_search
    assert "extraction_note_search" in analysis.frozen_preselection_review_policy(cfg)
    assert "note_search" not in analysis._configured_serial_extraction(replace(cfg, extraction_note_search=search.NoteSearchConfig()))
    for value in [{"enabled": "yes"}, {"retry_zero_match_missing": "yes"}, {"max_cells": 0}, {"max_scan_patterns": 0},
                  {"max_scan_patterns": True}, {"worker_memory_mb": 10}, {"typo": 1}]:
        with pytest.raises(ValueError):
            search.config_from_mapping(value)


def test_prompt_budget_fails_before_inference(tmp_path, fake_backend):
    args = arguments(tmp_path, lambda *_a, **_k: pytest.fail("Must reject before calling endpoint"))
    args["max_prompt_chars"] = 100
    with pytest.raises(analysis.Stage2InfrastructureError, match="prompt exceeds"):
        analysis.extract_rows(**args)


def test_feature_slices_and_heldout_path_use_the_selected_method(tmp_path, fake_backend):
    calls = []

    def request(messages, validate, *, request_kind):
        text = messages[-1]["content"]
        definition_text = text.split("Record length:")[0]
        name = "ecog" if "\necog\n" in definition_text else "age"
        calls.append(name)
        if "Search cells remaining: 3" in text:
            return validate(python_action())
        return validate({"action": "final", "values": {name: 2 if name == "ecog" else 67}})

    cfg = workflow.PlainHandoffStage2Config(endpoint="http://unused.test/v1", model="test",
        extraction_note_search=search.NoteSearchConfig(enabled=True, max_review_passes=0))
    frame = analysis._extract_outer_heldout_measurements(
        dataset=pd.DataFrame({"text": ["ECOG 2. Age 67."]}), heldout_ids=[0], text_column="text",
        measurement_definitions=FEATURES, output_dir=tmp_path, request_json=request, workers=1,
        max_prompt_chars=100_000, feature_batch_size=1, request_identity={"model": "test"},
        tokenizer=None, frozen_cache=None, serial_extraction=analysis._configured_serial_extraction(cfg),
    )
    assert frame[["ecog", "age"]].iloc[0].tolist() == [2, 67]
    assert calls == ["ecog", "ecog", "age", "age"]
    assert len(list(tmp_path.rglob("reviewed_excerpts.json"))) == 2


def test_answer_before_any_successful_search_is_not_scientific_missingness(tmp_path, fake_backend):
    config = workflow.PlainHandoffStage2Config(endpoint="http://unused.test/v1", model="test", max_response_repairs=1)
    def request(messages, validate, *, request_kind):
        return workflow._request_json(messages=messages, config=config,
            completion=lambda *_: json.dumps({"action": "final", "values": {"ecog": None, "age": None}}),
            validate=validate, request_kind=request_kind)
    with pytest.raises(analysis.Stage2InfrastructureError, match="No successful note search"):
        analysis.extract_rows(**arguments(tmp_path, request))
    assert not (tmp_path / "extracted.csv").exists()


def test_builtin_worker_search_state_and_isolation():
    config = search.NoteSearchConfig(enabled=True, cell_timeout_seconds=1)
    search.preflight(config)
    worker_class, identity = search.load_backend(config)
    assert len(identity["worker_sha256"]) == 64
    history = "Synthetic α record. ECOG 2. Age 67."
    with worker_class(history, config) as worker:
        observation = worker.execute("hits = scan([r'\\bECOG\\b', r'\\bAge\\b']); hits")
        assert observation["error"] is None and observation["source_spans"]
        assert worker.execute("len(hits['hits'])")["output"].strip() == "2"
        assert worker.execute("open('/etc/passwd').read()")["error"] == "PermissionError"
        assert worker.execute("import socket; socket.socket()")["error"] is not None
        with pytest.raises(RuntimeError, match="wall-time"):
            worker.execute("while True: pass")


def test_real_worker_accepts_128_patterns_and_preserves_runtime_limits():
    config = search.NoteSearchConfig(enabled=True, max_code_chars=100)
    worker_class, identity = search.load_backend(config)
    assert identity["implementation"] == "oci_builtin"
    assert search.policy_identity(config) != search.policy_identity(replace(config, max_scan_patterns=12))
    history = " ".join(f"F{i:03d}" for i in range(128))
    with worker_class(history, config) as worker:
        raw = worker.execute("result = scan([r'\\bF%03d\\b' % i for i in range(128)], context=0); print(json.dumps(result))")
        assert raw["error"] is None and raw["source_spans"]
        result = json.loads(raw["output"])
        assert result["match_count"] == 128 and len(result["hits"]) == 12
        assert result["hits"][0]["quote"] == "F000" and result["hits"][-1]["quote"] == "F127"
        assert not raw["sources_truncated"]
        assert len(raw["output"]) <= config.max_output_chars
        rejected = worker.execute("scan(['F000'] * 129)")
        assert rejected["error"] == "ValueError" and "1-128" in rejected["error_detail"]
        with pytest.raises(RuntimeError, match="code limit"):
            worker.execute(" " * 101)
