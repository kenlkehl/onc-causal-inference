"""Check exact prompt coverage, actual Qwen requests, and miniature responses.

This verifies saved artifacts. It makes no model calls or production changes.
"""
from __future__ import annotations

import hashlib
import json
import re
import runpy
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
MANIFEST = json.loads((OUT / "manifest.json").read_text())
OLD_CHECK = runpy.run_path(
    str(OUT.parent / "all_other_prompts_naive_review_2026-09-23/verify_and_index.py")
)["check_response"]
EXPECTED_PARAMETERS = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
    "frequency_penalty": 0.0,
    "reasoning_effort": "xhigh",
    "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True},
    "max_tokens": 65536,
}
REVIEW_FIELDS = {
    "understanding", "input_meaning", "can_complete", "questions",
    "unnecessary_context", "sample_response",
}
ROUND_ORDER = {label: n for n, label in enumerate(
    ["round1", "round2", "final", "clarified", "evidence_limits", "loss_comparison"]
)}


def read(path):
    return json.loads(path.read_text())


def rel(path):
    return str(path.relative_to(OUT))


def current_result(slug, phase, messages):
    matches = []
    for result_path in (OUT / "runs").glob(f"*/{slug}/{phase}/result.json"):
        if read(result_path.parent / "task_messages.json") == messages:
            label = result_path.parents[2].name
            matches.append((ROUND_ORDER[label], result_path))
    assert matches, f"No current {phase}: {slug}"
    _, result_path = max(matches)
    result = read(result_path)
    request = read(result_path.parent / "request.json")
    assert result["http_status"] == 200, result_path
    assert result["finish_reason"] == "stop", result_path
    assert result["model"] == request["model"] == "Inferact/Qwen3.8-Flash-Next-NVFP4"
    assert result["reasoning_characters"] > 0, result_path
    for key, value in EXPECTED_PARAMETERS.items():
        assert request[key] == value, (result_path, key)
    fingerprint = hashlib.sha256(
        json.dumps(request, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    assert result["request_sha256"] == fingerprint, result_path
    assert json.loads(result["content"]) == result["parsed"], result_path
    if phase == "execute":
        assert request["messages"] == messages
    else:
        quoted = request["messages"][-1]["content"]
        assert all(message["content"] in quoted for message in messages)
        parsed = result["parsed"]
        assert set(parsed) == REVIEW_FIELDS, result_path
        assert parsed["can_complete"] is True, result_path
        assert isinstance(parsed["questions"], list)
        assert isinstance(parsed["unnecessary_context"], list)
        assert parsed["sample_response"] is not None
    return result_path, result


def check_example(number, response):
    if number == 4:
        assert set(response) == {
            "description", "value_type", "unit", "categories",
            "measurement_definition", "missing_value_rule",
            "conflict_resolution", "caveats",
        }
        assert response["value_type"] == "continuous"
        assert response["unit"] == "mg/dL" and response["categories"] == []
        assert response["conflict_resolution"] == {
            "strategy": "latest", "positive_category": None,
        }
        for key in ("description", "measurement_definition", "missing_value_rule"):
            assert isinstance(response[key], str) and response[key].strip()
        # A direct threshold is supported by this example and must survive both
        # the definition and its missingness rule. Read the prose as well.
        assert "threshold" in response["measurement_definition"].lower()
        assert "threshold" in response["missing_value_rule"].lower()
        assert isinstance(response["caveats"], list)
    else:
        OLD_CHECK(2 if number == 1 else number, response)


def check_style(messages):
    text = "\n".join(m["content"] for m in messages)
    forbidden = [
        r"\brecord_scope\b", r"\bPython\b", r"\bcaller\b",
        r"\bpipeline\b", r"\bupstream\b", r"\bdownstream\b",
        r"\bcheckpoint\b", r"\bprovenance\b", r"\bindices\b",
        r"\bidentifiers\b", r"\brather than\b", r"\binstead of\b",
        r"\bnot\b[^.\n]{0,160}\bbut\b",
    ]
    for pattern in forbidden:
        assert not re.search(pattern, text, re.IGNORECASE), pattern


def main():
    assert len(MANIFEST) == 23
    records = []
    selected = []
    review_transcript = [
        "# Qwen's final comprehension reviews — September 23, 2026", "",
        "Each entry uses the exact current prompt in an isolated conversation. "
        "The direct example answer comes from a separate call using the task "
        "messages themselves. These miniature examples test instructions and "
        "response contracts; they do not measure clinical or causal performance.", "",
    ]
    for item in MANIFEST:
        slug = item["slug"]
        number = int(slug[:2])
        messages = read(OUT / "prompts" / f"{slug}.json")
        check_style(messages)
        review_path, review = current_result(slug, "review", messages)
        execute_path, execute = current_result(slug, "execute", messages)
        check_example(number, execute["parsed"])
        selected.extend([review, execute])
        answer = review["parsed"]
        record = {
            "prompt": slug, "title": item["title"],
            "status": "reviewed_proposal_not_integrated",
            "prompt_sha256": hashlib.sha256(
                (OUT / "prompts" / f"{slug}.json").read_bytes()
            ).hexdigest(),
            "review_result": rel(review_path),
            "execution_result": rel(execute_path),
            "can_complete": answer["can_complete"],
            "clarification_questions": answer["questions"],
            "unnecessary_context_feedback": answer["unnecessary_context"],
            "direct_example_check": "passed",
            "review_reasoning_characters": review["reasoning_characters"],
            "execution_reasoning_characters": execute["reasoning_characters"],
        }
        records.append(record)
        review_transcript += [
            f"## {slug}: {item['title']}", "",
            f"[Exact prompt]({OUT / 'prompts' / (slug + '.md')}) · "
            f"[Review request]({review_path.parent / 'request.json'}) · "
            f"[Direct execution request]({execute_path.parent / 'request.json'})", "",
            "### Comprehension review", "", "```json",
            json.dumps({k: v for k, v in answer.items() if k != "sample_response"},
                       indent=2, ensure_ascii=False),
            "```", "", "### Direct example response", "", "```json",
            json.dumps(execute["parsed"], indent=2, ensure_ascii=False),
            "```", "",
        ]
    all_results = [read(p) for p in (OUT / "runs").glob("*/*/*/result.json")]
    question_count = sum(len(r["clarification_questions"]) for r in records)
    usage = {key: sum((result.get("usage") or {}).get(key, 0) for result in selected)
             for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
    report = {
        "date": "2026-09-23", "model": "Inferact/Qwen3.8-Flash-Next-NVFP4",
        "endpoint": "http://sn4622130540:8001", "parameters": EXPECTED_PARAMETERS,
        "current_prompts_reviewed": len(records),
        "current_prompts_directly_executed": len(records),
        "current_comprehension_reviews_can_complete": len(records),
        "current_clarification_questions": question_count,
        "current_direct_example_checks_passed": len(records),
        "current_calls_with_reasoning": len(selected),
        "current_calls_truncated": 0,
        "current_selected_call_usage": usage,
        "historical_live_calls_including_revisions": len(all_results),
        "historical_http_errors": sum(r.get("http_status") != 200 for r in all_results),
        "historical_truncations": sum(r.get("finish_reason") == "length" for r in all_results),
        "limitations": "Invented miniature examples. Contract checks include known "
            "values and coverage where applicable. Role, theme, ranking, and concept "
            "judgments are inspected without requiring a preferred causal decision. "
            "No production pipeline or clinical benchmark was run in this revision.",
        "proposals": records,
    }
    (OUT / "verification.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    (OUT / "QWEN_REVIEWS_AND_RESPONSES_2026-09-23.md").write_text("\n".join(review_transcript))
    rows = ["| Prompt | Fresh review | Direct example |", "|---|---|---|"]
    for item in records:
        slug = item["prompt"]
        n_questions = len(item["clarification_questions"])
        review_link = OUT / item["review_result"]
        execution_link = OUT / item["execution_result"]
        rows.append(
            f"| [{slug}: {item['title']}]({OUT / 'prompts' / (slug + '.md')}) "
            f"| [Understood; {n_questions} questions]({review_link.parent / 'answer.txt'}) "
            f"| [Passed]({execution_link.parent / 'answer.txt'}) |"
        )
    (OUT / "PROMPT_CHECK_INDEX_2026-09-23.md").write_text("\n".join(rows) + "\n")
    report_path = OUT / "REVIEW_REPORT_2026-09-23.md"
    if report_path.exists():
        prose = report_path.read_text()
        summary = (
            f"**Final result:** Qwen reported it could complete all {len(records)} tasks, "
            f"with {question_count} remaining clarification questions. All {len(records)} "
            "direct miniature examples passed their contract and applicable known-value "
            f"checks. The {len(selected)} final-version calls returned reasoning and "
            f"completed without truncation. The record includes {len(all_results)} "
            "calls across all revisions."
        )
        prose = re.sub(r"\*\*Final result:\*\*[^\n]+", lambda _: summary, prose)
        report_path.write_text(prose)
    print(f"Verified {len(records)} current prompts, {len(selected)} isolated calls, "
          f"{question_count} clarification questions, and {len(records)} direct examples.")


if __name__ == "__main__":
    main()
