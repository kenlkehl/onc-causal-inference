"""Checkpoint exact clinical prompts and normalized responses."""
from hashlib import sha256
import json
import os
import threading
from pathlib import Path

from .stage2_prompt_catalog import PROMPT_VERSION


def fingerprint(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def request_review(directory, messages, validate, *, request_json, identity, maximum_chars=200000):
    if sum(len(m["content"]) for m in messages) > maximum_chars:
        raise ValueError("clinical review prompt exceeds max_prompt_chars; no input was truncated")
    directory = Path(directory)
    key = fingerprint({"messages": messages, "identity": identity, "version": PROMPT_VERSION})
    response_path, complete_path = directory / "response.json", directory / "complete.json"
    write_json(directory / "prompt.json", {"messages": messages, "input_fingerprint": key})
    if response_path.is_file() and complete_path.is_file():
        complete = json.loads(complete_path.read_text())
        if complete.get("input_fingerprint") == key:
            response = json.loads(response_path.read_text())
            if complete.get("response_sha256") != fingerprint(response):
                raise ValueError("clinical review checkpoint response hash does not match")
            return validate(response)
    response = validate(request_json(messages, validate, request_kind="interpretation"))
    write_json(response_path, response)
    write_json(complete_path, {"status": "complete", "input_fingerprint": key, "response_sha256": fingerprint(response)})
    return response
