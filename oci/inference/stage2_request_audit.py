"""Request correlation and progress records without patient text or responses."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps
import json
import logging
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from uuid import uuid4

LOGGER = logging.getLogger(__name__)
_CONTEXT: ContextVar[dict | None] = ContextVar("stage2_request_audit", default=None)


@contextmanager
def context(**fields):
    token = _CONTEXT.set({**(_CONTEXT.get() or {}), **fields})
    try:
        yield
    finally:
        _CONTEXT.reset(token)


def update(**fields):
    current = _CONTEXT.get()
    if current is not None:
        current.update(fields)


def request_id():
    return (_CONTEXT.get() or {}).get("request_id", "unscoped")


def event(name, **fields):
    current = _CONTEXT.get()
    if current is None:
        return
    record = {k: v for k, v in current.items() if not k.startswith("_")}
    record.update(schema_version="stage2_request_events_v1", event=name,
                  at=datetime.now(timezone.utc).isoformat(timespec="milliseconds"), **fields)
    if "_started" in current:
        record["elapsed_seconds"] = round(time.monotonic() - current["_started"], 3)
    if current.get("_audit_path"):
        path = Path(current["_audit_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        # One checkpoint belongs to one worker. Each logical request has a
        # distinct ID, including a resumed attempt appended to this journal.
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    LOGGER.info("Stage 2 request_id=%s event=%s details=%s", request_id(), name,
                json.dumps(fields, sort_keys=True, allow_nan=False))


def audited(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        if (_CONTEXT.get() or {}).get("request_id"):
            # The concurrency admission wrapper recurses into the same request.
            return function(*args, **kwargs)
        cfg = kwargs["config"]
        with context(request_id=uuid4().hex, _started=time.monotonic(),
                     request_kind=kwargs.get("request_kind", "interpretation"),
                     endpoint=cfg.endpoint, model=cfg.model):
            event("request_queued")
            try:
                result = function(*args, **kwargs)
            except Exception as error:
                event("request_failed", error_type=type(error).__name__, error=str(error)[:2000])
                raise
            event("request_validated")
            return result
    return wrapped


def usage_fields(usage):
    if usage is None:
        return {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
    return {key: (usage.get(key) if isinstance(usage, dict) else getattr(usage, key, None))
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")}


def collect_stream(stream, *, deadline, deadline_error, incomplete_error):
    """Consume final text and reasoning separately; never accept partial output.

    The HTTP client enforces read inactivity. A watchdog also closes a silent
    stream at the logical deadline, even if SSE heartbeat bytes keep arriving.
    """
    content, reasoning = [], []
    finish_reason = None
    usage = None
    provider_id = None
    started = time.monotonic()
    last_recorded = started
    last_progress = None
    progress_reported = False
    chunk_count = 0
    deadline_reached = threading.Event()

    def expire():
        deadline_reached.set()
        try:
            stream.close()
        except Exception:
            LOGGER.exception("Stage 2 could not close stream at its logical deadline")

    timer = threading.Timer(max(0.001, deadline - time.monotonic()), expire)
    timer.daemon = True
    timer.start()
    try:
        for chunk in stream:
            if deadline_reached.is_set() or time.monotonic() >= deadline:
                raise deadline_error("Stage 2 logical deadline expired during streaming")
            provider_id = getattr(chunk, "id", None) or provider_id
            usage = getattr(chunk, "usage", None) or usage
            for choice in chunk.choices:
                if getattr(choice, "index", 0) != 0:
                    continue
                delta = choice.delta
                text = getattr(delta, "content", None) or ""
                thought = next((getattr(delta, key, None) for key in
                                ("reasoning_content", "reasoningContent", "reasoning")
                                if getattr(delta, key, None)), "")
                if not isinstance(text, str) or not isinstance(thought, str):
                    raise incomplete_error("Stage 2 streamed text must contain string deltas")
                content.append(text)
                reasoning.append(thought)
                if text or thought:
                    chunk_count += 1
                    last_progress = time.monotonic()
                finish_reason = getattr(choice, "finish_reason", None) or finish_reason
            if last_progress is not None and (
                not progress_reported or time.monotonic() - last_recorded >= 30
            ):
                event("stream_progress", provider_request_id=provider_id,
                      content_chars=sum(map(len, content)), reasoning_chars=sum(map(len, reasoning)),
                      progress_chunks=chunk_count)
                last_recorded = time.monotonic()
                progress_reported = True
        if deadline_reached.is_set() or time.monotonic() >= deadline:
            raise deadline_error("Stage 2 logical deadline expired during streaming")
        if not finish_reason:
            raise incomplete_error("Stage 2 stream ended without a final finish reason")
        return SimpleNamespace(
            id=provider_id, usage=usage,
            choices=[SimpleNamespace(finish_reason=finish_reason,
                                     message=SimpleNamespace(content="".join(content),
                                                             reasoning_content="".join(reasoning)))],
        )
    except Exception as error:
        event("stream_failed", provider_request_id=provider_id,
              error_type=type(error).__name__, error=str(error)[:2000],
              content_chars=sum(map(len, content)), reasoning_chars=sum(map(len, reasoning)),
              progress_chunks=chunk_count, usage_available=usage is not None,
              last_progress_seconds_ago=(None if last_progress is None else
                                        round(time.monotonic() - last_progress, 3)),
              **usage_fields(usage))
        if deadline_reached.is_set():
            raise deadline_error("Stage 2 logical deadline expired during streaming") from error
        raise
    finally:
        timer.cancel()
        stream.close()
