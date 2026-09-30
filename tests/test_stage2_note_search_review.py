"""Coverage, recovery, and restart behavior for bounded measurement review."""
from dataclasses import replace
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from oci.inference import stage2_note_search as search
from oci.inference import stage2_note_search_review as review
from oci.inference import plain_handoff_stage2_analysis as analysis
from oci.inference import plain_handoff_stage2 as workflow
from tests.test_stage2_note_search import FEATURES, arguments, fake_backend, python_action


def refine_args(tmp_path, request, history, values, **kw):
    return dict(row={"row_id": 7, "text": history}, definitions=FEATURES,
        provisional={"rows": [{"row_id": 7, "values": values}]}, retained=[], review_flags=[],
        directory=tmp_path / "review", plan_directory=tmp_path / "terms", request_identity={"model": "test"},
        request_json=request, config=search.NoteSearchConfig(enabled=True), max_prompt_chars=100_000,
        tokenizer=None, input_token_budget=None, **kw)


def vocabulary(validate):
    return validate({"terms": {"ecog": ["ECOG", "performance status"], "age": ["age"]}})


def test_revisit_missing_and_unseen_conflict_with_original_locations(tmp_path):
    history = "ECOG 3." + " filler" * 200 + "<new_note>ECOG 2. Age 67."
    calls = []
    def request(messages, validate, *, request_kind):
        if messages[0]["content"] == review.PLAN_PROMPT:
            assert history not in messages[-1]["content"]
            return vocabulary(validate)
        calls.append(messages)
        content = messages[-1]["content"]
        assert "Document segment 2" in content and "original characters" in content
        assert "ECOG 3." in content and "ECOG 2. Age 67." in content
        return validate({"values": {"ecog": 2, "age": 67}, "needs_more_evidence": []})
    args = refine_args(tmp_path, request, history, {"ecog": 3, "age": None})
    args["retained"] = [{"start": 0, "end": 7, "text": history[:7]}]
    result = review.refine(**args)
    assert result["rows"][0]["values"] == {"ecog": 2, "age": 67}
    assert len(calls) == 1
    audit = json.loads((tmp_path / "review/complete.json").read_text())
    assert audit["unresolved_or_coverage_limited_features"] == []
    assert audit["rounds"][0]["changed"] == ["ecog", "age"]


def test_zero_matches_get_one_shared_expansion_then_remain_missing(tmp_path):
    calls = []
    def request(messages, validate, *, request_kind):
        calls.append(messages)
        if messages[0]["content"] == review.PLAN_PROMPT:
            return vocabulary(validate)
        assert messages[0]["content"] == review.ZERO_MATCH_PROMPT
        return validate({"terms": ["ECOG", "AGE", "not documented here"]})
    def patient(i):
        args = refine_args(tmp_path, request, "No matching clinical measurement.", {"ecog": None, "age": None})
        args["directory"] = tmp_path / f"patient_{i}"
        return review.refine(**args)
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(patient, range(3)))
    assert len(calls) == 3  # One original plan and one expansion per variable across all patients.
    assert all(r["rows"][0]["values"] == {"ecog": None, "age": None} for r in results)
    for i in range(3):
        audit = json.loads((tmp_path / f"patient_{i}/complete.json").read_text())
        assert set(audit["zero_match_research"]) == {"ecog", "age"}
        assert all(x["match_count_after_research"] == 0 for x in audit["zero_match_research"].values())
        assert audit["rounds"] == []


def test_failed_review_resumes_provisional_without_repeating_initial_extraction(tmp_path, fake_backend):
    calls, outage = [], True
    def request(messages, validate, *, request_kind):
        system, text = messages[0]["content"], messages[-1]["content"]
        calls.append(system)
        if system == review.PLAN_PROMPT:
            return vocabulary(validate)
        if system == review.REVIEW_PROMPT:
            if outage:
                raise analysis.Stage2RequestExhaustedError("temporary outage")
            return validate({"values": {"age": 67}, "needs_more_evidence": []})
        if "Search cells remaining: 3" in text:
            return validate(python_action())
        return validate({"action": "final", "values": {"ecog": 2, "age": None}})
    args = arguments(tmp_path, request)
    args["note_search"] = search.NoteSearchConfig(enabled=True)
    with pytest.raises(analysis.Stage2RequestExhaustedError):
        analysis.extract_rows(**args)
    assert len(fake_backend) == 1
    outage = False
    result = analysis.extract_rows(**args)
    assert result.loc[0, "age"] == 67
    assert len(fake_backend) == 1 and len(calls) == 5


def test_per_field_coverage_and_bounded_fallback_preserve_failed_prior_value(tmp_path):
    history = ("ECOG 2. " + "x" * 1000 + " ") * 12 + "Age 67."
    calls, fallback_names = [], []
    def request(messages, validate, *, request_kind):
        if messages[0]["content"] == review.PLAN_PROMPT:
            return vocabulary(validate)
        calls.append(messages)
        if len(calls) == 1:
            assert "Age 67." in messages[-1]["content"]
            return validate({"values": {"ecog": 2, "age": 67}, "needs_more_evidence": ["ecog"]})
        return validate({"values": {"ecog": 2}, "needs_more_evidence": ["ecog"]})
    def fallback(definitions, directory):
        name = definitions[0]["name"]
        fallback_names.append(name)
        # Simulate failure in a serial chunk. The parent ledger is empty.
        analysis._write_json(directory / "extraction_issues.json", {"events": []})
        analysis._write_json(directory / "serial_chunks/chunk_00001/extraction_issues.json",
            {"events": [{"failure_kind": "structural_response_failure"}]})
        return {"rows": [{"row_id": 7, "values": {name: None}}]}
    result = review.refine(**refine_args(tmp_path, request, history, {"ecog": 2, "age": None},
                                       fallback_extract=fallback))
    assert len(calls) == 2 and fallback_names == ["ecog"]
    assert result["rows"][0]["values"] == {"ecog": 2, "age": 67}
    audit = json.loads((tmp_path / "review/complete.json").read_text())
    assert audit["failed_full_record_fallback_features"] == ["ecog"]
    assert audit["unresolved_or_coverage_limited_features"] == ["ecog"]


def test_structural_review_failure_preserves_prior_measurement(tmp_path):
    cfg = workflow.PlainHandoffStage2Config(endpoint="http://unused.test/v1", model="test", max_response_repairs=1)
    def request(messages, validate, *, request_kind):
        if messages[0]["content"] == review.PLAN_PROMPT:
            return vocabulary(validate)
        if messages[0]["content"] == review.ZERO_MATCH_PROMPT:
            return validate({"terms": []})
        return workflow._request_json(messages=messages, config=cfg, completion=lambda *_: '{}',
                                      validate=validate, request_kind=request_kind)
    args = refine_args(tmp_path, request, "ECOG 2.", {"ecog": 2, "age": None})
    args["config"] = replace(args["config"], max_review_passes=1)
    result = review.refine(**args)
    assert result["rows"][0]["values"]["ecog"] == 2
    audit = json.loads((tmp_path / "review/complete.json").read_text())
    assert audit["unresolved_or_coverage_limited_features"] == ["ecog"]


def test_zero_match_alternatives_recover_missing_value_and_resume_review(tmp_path):
    calls, outage = [], True
    history = "ECOG 2. The patient is a 67-year-old adult."
    def request(messages, validate, *, request_kind):
        system, content = messages[0]["content"], messages[-1]["content"]
        calls.append(system)
        if system == review.PLAN_PROMPT:
            return vocabulary(validate)
        if system == review.ZERO_MATCH_PROMPT:
            assert "Unsuccessful search terms\n- age" in content
            assert history not in content and "Meaning: ECOG" not in content
            # Old vocabulary and equivalent alternative spellings are removed by Python.
            return validate({"terms": ["AGE", "year old", "YEAR-OLD"]})
        if outage:
            raise analysis.Stage2RequestExhaustedError("temporary outage")
        assert "67-year-old" in content and "original characters" in content
        return validate({"values": {"age": 67}, "needs_more_evidence": []})
    args = refine_args(tmp_path, request, history, {"ecog": 2, "age": None})
    args["retained"] = [{"start": 0, "end": 7, "text": history[:7]}]
    with pytest.raises(analysis.Stage2RequestExhaustedError):
        review.refine(**args)
    assert not (tmp_path / "review/complete.json").exists()
    outage = False
    result = review.refine(**args)
    assert result["rows"][0]["values"] == {"ecog": 2, "age": 67}
    assert calls.count(review.ZERO_MATCH_PROMPT) == calls.count(review.PLAN_PROMPT) == 1
    audit = json.loads((tmp_path / "review/complete.json").read_text())
    assert audit["zero_match_research"]["age"]["alternative_terms"] == ["year old"]
    assert audit["zero_match_research"]["age"]["match_count_after_research"] == 1


def test_zero_match_expansion_validation_failure_uses_bounded_fallback(tmp_path):
    cfg = workflow.PlainHandoffStage2Config(endpoint="http://unused.test/v1", model="test", max_response_repairs=1)
    def request(messages, validate, *, request_kind):
        if messages[0]["content"] == review.PLAN_PROMPT:
            return vocabulary(validate)
        assert messages[0]["content"] == review.ZERO_MATCH_PROMPT
        return workflow._request_json(messages=messages, config=cfg, completion=lambda *_: '{}',
                                      validate=validate, request_kind=request_kind)
    def fallback(definitions, directory):
        assert [d["name"] for d in definitions] == ["age"]
        return {"rows": [{"row_id": 7, "values": {"age": 67}}]}
    history = "ECOG 2. The patient is sixty-seven years old."
    args = refine_args(tmp_path, request, history, {"ecog": 2, "age": None}, fallback_extract=fallback)
    args["retained"] = [{"start": 0, "end": len(history), "text": history}]
    result = review.refine(**args)
    assert result["rows"][0]["values"]["age"] == 67
    audit = json.loads((tmp_path / "review/complete.json").read_text())
    assert audit["zero_match_research"]["age"]["vocabulary_status"] == "invalid_response"
    assert audit["full_record_fallback_features"] == ["age"]


def test_zero_match_expansion_can_be_disabled(tmp_path):
    def request(messages, validate, *, request_kind):
        assert messages[0]["content"] == review.PLAN_PROMPT
        return vocabulary(validate)
    args = refine_args(tmp_path, request, "No measurements.", {"ecog": None, "age": None})
    args["config"] = replace(args["config"], retry_zero_match_missing=False)
    result = review.refine(**args)
    assert result["rows"][0]["values"] == {"ecog": None, "age": None}
    assert json.loads((tmp_path / "review/complete.json").read_text())["zero_match_research"] == {}


def test_literal_search_is_escaped_bounded_and_uses_string_positions():
    text = "NLR 4. renal_function. xNLRx renal-function. " * 600
    matches = review.literal_matches(text, ["NLR", "renal function", "[not regex]"])
    assert matches["match_count"] == 1800
    assert len(matches["positions"]) == 512 and matches["positions_omitted"]
    assert all(text[a:b] in {"NLR", "renal_function", "renal-function"} for a, b in matches["positions"])
    locations = review.TextLocations("α date 2099<new_note>date 1900 ECOG 2")
    start = locations.history.index("date 1900")
    span = locations.span(start, len(locations.history))
    assert span["segment_start"] == 2 and span["text"] == locations.history[start:]
    assert set(span) == {"start", "end", "text", "segment_start", "segment_end"}
