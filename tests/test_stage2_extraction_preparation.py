import json

import pandas as pd
import pytest

from oci.inference import plain_handoff_stage2_analysis as analysis
from tests.stage2_prompt_spy import install, prompt_inputs


class CharacterTokenizer:
    def __init__(self):
        self.calls = 0

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        self.calls += 1
        return range(sum(len(m["content"]) + len(m["role"]) + 2 for m in messages) + 11)


def definition(index=0):
    return {
        "name": f"state_{index}", "description": "Documented clinical state.",
        "value_type": "categorical", "categories_or_unit": ["red", "blue"],
        "measurement_definition": "Use the most frequently documented state.",
        "missing_value_rule": "Return null when undocumented.",
        "conflict_resolution": {"strategy": "mode"},
    }


def test_fitting_record_does_not_repeatedly_tokenize_near_complete_prefixes():
    batches = [[definition(i)] for i in range(270)]
    source = "clinical finding 漢字\n" * 100
    tokenizer = CharacterTokenizer()
    pages = analysis._lossless_extraction_pages(
        {"row_id": 0, "text": source}, definition_batches=batches,
        max_prompt_chars=100_000, tokenizer=tokenizer, input_token_budget=100_000,
    )
    assert len(pages) == 1 and pages[0]["text"] == source
    assert tokenizer.calls <= 2 * len(batches)


@pytest.mark.parametrize("token_limited", [False, True])
def test_prepared_page_prompts_preserve_lossless_coverage_and_every_batch_budget(token_limited):
    batches = [[definition(0)], [dict(definition(1), description="Long definition. " * 40)]]
    templates = [analysis._page_extraction_prompt_template(definitions=b) for b in batches]
    tokenizer = CharacterTokenizer()
    overhead = max(analysis.prompt_token_count(tokenizer, p) for p in templates)
    budget = overhead + 200
    char_budget = 100_000 if token_limited else budget
    token_budget = budget if token_limited else 100_000
    source = "red 漢字\n\nblue — repeated observation. " * 50
    pages = analysis._lossless_extraction_pages(
        {"row_id": 5, "text": source}, definition_batches=batches,
        max_prompt_chars=char_budget, tokenizer=tokenizer,
        input_token_budget=token_budget, prompt_templates=templates,
    )
    assert len(pages) > 1
    assert "".join(p["text"] for p in pages) == source
    for page in pages:
        meta = page["page"]
        assert source[meta["char_start"]:meta["char_end"]] == page["text"]
        for batch in batches:
            prompt = analysis._page_extraction_prompt(definitions=batch, row=page)
            assert analysis._prompt_chars(prompt) <= char_budget
            assert analysis.prompt_token_count(tokenizer, prompt) <= token_budget
    # Reusing shared prefixes across patients must not append source text to them.
    assert templates == [analysis._page_extraction_prompt_template(definitions=b) for b in batches]


def test_extraction_starts_before_later_patients_are_planned_and_reuses_checkpoints(tmp_path, monkeypatch):
    install(monkeypatch)
    original = analysis._lossless_extraction_pages
    planned = []
    requests = []

    def plan(row, **kwargs):
        if row["row_id"] > 0:
            assert requests, "The first patient must reach extraction before planning the cohort"
        planned.append(row["row_id"])
        return original(row, **kwargs)

    monkeypatch.setattr(analysis, "_lossless_extraction_pages", plan)

    def request(messages, validate, *, request_kind):
        body = prompt_inputs(messages)
        patient = body["patient"]
        requests.append(patient["row_id"])
        return validate({"observations": [
            {"feature": body["features"][0]["name"], "value": "red",
             "quote": "red", "governing_date_quote": None},
        ]})

    args = dict(dataset=pd.DataFrame({"clinical_text": ["red", "red"]}),
                row_ids=[0, 1], text_column="clinical_text", definitions=[definition()],
                output_dir=tmp_path, request_json=request, workers=1,
                max_prompt_chars=100_000, tokenizer=CharacterTokenizer(),
                context_window_tokens=100_000, max_output_tokens=1000)
    first = analysis.extract_rows(**args)
    assert planned == [0, 1]
    assert requests == [0, 1]
    assert first["state_0"].tolist() == ["red", "red"]
    completion = json.loads((tmp_path / "complete.json").read_text())
    assert completion["pages"] == 2 and completion["paged_rows"] == 2
    before = len(requests)
    second = analysis.extract_rows(**args)
    pd.testing.assert_frame_equal(first, second)
    assert len(requests) == before
    assert json.loads((tmp_path / "pages/row_00000000/planning_status.json").read_text())["status"] == "complete"


def test_partial_paged_patient_retry_preserves_mode_counts_and_completed_pages(tmp_path, monkeypatch):
    install(monkeypatch)
    feature = definition()
    prefix = analysis._prompt_chars(analysis._page_extraction_prompt_template(definitions=[feature]))
    calls = []

    def request(messages, validate, *, request_kind):
        page = prompt_inputs(messages)["patient"]
        index = page["page"]["page_index"]
        calls.append(index)
        if index == 2 and calls.count(2) == 1:
            raise analysis.Stage2RequestExhaustedError("temporary transport failure")
        value = page["text"].strip()
        return validate({"observations": [{"feature": "state 0", "value": value,
            "quote": value, "governing_date_quote": None}]})

    frame = analysis.extract_rows(
        dataset=pd.DataFrame({"clinical_text": ["red red blue"]}), row_ids=[0],
        text_column="clinical_text", definitions=[feature], output_dir=tmp_path,
        request_json=request, workers=1, max_prompt_chars=prefix + 4,
        deferred_retry_passes=1,
    )
    assert calls == [1, 2, 2, 3]
    assert frame.loc[0, "state_0"] == "red"
    deferred = json.loads((tmp_path / "deferred_extraction.json").read_text())
    assert deferred["status"] == "resolved" and deferred["unresolved"] == []
    assert json.loads((tmp_path / "complete.json").read_text())["pages"] == 3
