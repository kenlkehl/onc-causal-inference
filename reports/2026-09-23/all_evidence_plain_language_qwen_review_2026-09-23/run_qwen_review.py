"""Isolated Qwen recipient reviews and direct executions of saved prompts."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent
BASE = "http://sn4622130540:8001"
PARAMETERS = {
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
REVIEW_SYSTEM = """Read the supplied task as someone receiving it for the first time. Use the task messages and their example input to work out what you are being asked to do.

Explain the task and the meaning of its input in plain language. Ask any clarification questions whose answers would change your action or output. Identify any wording that requires unexplained context, or unnecessary software/process detail. Empty question and unnecessary-context lists are appropriate when the task is clear.

Then carry out the example task if you can. Return one JSON object with exactly these keys:
- understanding: a concise statement of the task.
- input_meaning: a concise statement of what the supplied information represents.
- can_complete: a boolean.
- questions: an array of clarification questions.
- unnecessary_context: an array of passages that could be removed, with a brief reason for each.
- sample_response: your answer to the example task as a JSON value, or null if a required clarification prevents an answer.

Keep the review concise. The task conversation is quoted below for this review."""


def write_json(path, value):
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def get_json(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as response:
        return json.load(response)


def metadata():
    models = get_json("/v1/models")
    ids = [row["id"] for row in models["data"]]
    assert len(ids) == 1 and "qwen3.8-flash-next" in ids[0].lower(), ids
    result = {"checked_at": datetime.now(timezone.utc).isoformat(), "endpoint": BASE, "models": models, "parameters": PARAMETERS,
              "recommendation_sources": ["https://huggingface.co/Qwen/Qwen3.8-Flash-Next#api-usage", "https://huggingface.co/Inferact/Qwen3.8-Flash-Next-NVFP4#api-usage", "https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next"],
              "output_limit_note": "65536 is this review's completion ceiling; all calls retain xhigh and thinking. Check finish_reason to detect truncation."}
    try:
        result["server_version"] = get_json("/version")
    except Exception as exc:
        result["server_version_error"] = str(exc)
    try:
        schema = get_json("/openapi.json")
        properties = schema["components"]["schemas"]["ChatCompletionRequest"]["properties"]
        result["request_schema_controls"] = {k: properties.get(k) for k in PARAMETERS}
    except Exception as exc:
        result["schema_error"] = str(exc)
    write_json(OUT / "sources" / "server_and_sampling.json", result)
    return ids[0]


def review_messages(task_messages):
    quoted = "\n\n".join(f"{m['role'].upper()} MESSAGE\n{m['content']}" for m in task_messages)
    return [{"role": "system", "content": REVIEW_SYSTEM}, {"role": "user", "content": "<task_to_review>\n" + quoted + "\n</task_to_review>\n\nGive your review using understanding, input_meaning, can_complete, questions, unnecessary_context, and sample_response. Put the example task's answer inside sample_response."}]


def run_one(path, *, phase, label, model):
    original = json.loads(path.read_text())
    messages = review_messages(original) if phase == "review" else original
    payload = {"model": model, "messages": messages, **PARAMETERS}
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    directory = OUT / "runs" / label / path.stem / phase
    directory.mkdir(parents=True, exist_ok=True)
    response_path = directory / "result.json"
    if response_path.exists():
        cached = json.loads(response_path.read_text())
        if cached.get("request_sha256") == fingerprint and cached.get("http_status") == 200:
            return path.stem, "cached", cached
        raise RuntimeError(f"Existing results differ for {directory}; use a new label.")
    write_json(directory / "task_messages.json", original)
    write_json(directory / "request.json", payload)
    began = time.monotonic()
    request = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    print(f"START {label} {phase} {path.stem}", flush=True)
    try:
        with urllib.request.urlopen(request, timeout=1200) as response:
            body = json.load(response)
            status = response.status
        choice = body["choices"][0]
        message = choice["message"]
        content = message.get("content") or ""
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        # Some server configurations place the think block in content. Keep
        # only final text in readable artifacts, recording the reasoning size.
        if "</think>" in content:
            prefix, content = content.rsplit("</think>", 1)
            reasoning += prefix
            content = content.strip()
        result = {"http_status": status, "request_sha256": fingerprint, "model": body.get("model"), "elapsed_seconds": round(time.monotonic() - began, 3), "finish_reason": choice.get("finish_reason"), "usage": body.get("usage"), "reasoning_characters": len(reasoning), "content": content}
        try:
            result["parsed"] = json.loads(content)
            if phase == "review":
                parsed = result["parsed"]
                fields = {"understanding", "input_meaning", "can_complete", "questions", "unnecessary_context", "sample_response"}
                if not isinstance(parsed, dict) or set(parsed) != fields or not isinstance(parsed.get("can_complete"), bool):
                    result["review_schema_error"] = "Expected the review object, including a boolean can_complete and sample_response."
        except (json.JSONDecodeError, TypeError) as exc:
            result["json_error"] = str(exc)
        write_json(response_path, result)
        (directory / "answer.txt").write_text(content + "\n")
        print(f"DONE {label} {phase} {path.stem} finish={result['finish_reason']} seconds={result['elapsed_seconds']} reasoning_chars={len(reasoning)}", flush=True)
        return path.stem, "done", result
    except Exception as exc:
        result = {"request_sha256": fingerprint, "elapsed_seconds": round(time.monotonic() - began, 3), "error": str(exc)}
        if isinstance(exc, urllib.error.HTTPError):
            result.update(http_status=exc.code, error_body=exc.read().decode(errors="replace"))
        write_json(response_path, result)
        print(f"ERROR {label} {phase} {path.stem}: {result}", flush=True)
        return path.stem, "error", result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["review", "execute"], required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--ids", default="")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    model = metadata()
    wanted = {int(x) for x in args.ids.split(",") if x}
    paths = [p for p in sorted((OUT / "prompts").glob("*.json")) if not wanted or int(p.stem[:2]) in wanted]
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, p, phase=args.phase, label=args.label, model=model) for p in paths]
        for future in concurrent.futures.as_completed(futures):
            slug, state, result = future.result()
            results.append({"prompt": slug, "state": state, "finish_reason": result.get("finish_reason"), "json_ok": "parsed" in result, "can_complete": result.get("parsed", {}).get("can_complete") if args.phase == "review" and isinstance(result.get("parsed"), dict) else None})
            write_json(OUT / "runs" / args.label / f"{args.phase}_summary.json", sorted(results, key=lambda x: x["prompt"]))
    print(json.dumps(sorted(results, key=lambda x: x["prompt"]), indent=2), flush=True)


if __name__ == "__main__":
    main()
