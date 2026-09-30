"""OCI-owned transport for a resource-limited, isolated note-search interpreter."""
from __future__ import annotations

import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import time


class NoteSearchWorkerError(RuntimeError):
    """Infrastructure failure, rather than a missing clinical measurement."""


class NoteSearchWorker:
    def __init__(self, history, limits):
        limits.validate()
        if sys.platform != "linux":
            raise NoteSearchWorkerError("Note-search isolation requires Linux and libseccomp.")
        if not isinstance(history, str) or len(history.encode("utf-8")) > limits.max_history_bytes:
            raise ValueError("Patient record exceeds the configured history limit.")
        self.limits = limits
        self.history_length = len(history)
        self.process = subprocess.Popen(
            [sys.executable, "-I", "-S", str(Path(__file__).with_name("_note_search_child.py"))],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            cwd="/", env={}, close_fds=True, start_new_session=True, bufsize=0,
        )
        for stream in (self.process.stdin, self.process.stdout):
            os.set_blocking(stream.fileno(), False)
        try:
            ready = self._request({"history": history, "memory_mb": limits.worker_memory_mb,
                "output_chars": limits.max_output_chars, "patterns": limits.max_scan_patterns})
            if ready != {"ready": True}:
                raise NoteSearchWorkerError("Worker isolation unavailable; no generated code ran.")
        except BaseException:
            self.close()
            raise

    def _request(self, payload):
        deadline = time.monotonic() + self.limits.cell_timeout_seconds
        sending = memoryview((json.dumps(payload, ensure_ascii=True) + "\n").encode("ascii"))
        received = bytearray()
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(self.process.stdin, selectors.EVENT_WRITE)
                selector.register(self.process.stdout, selectors.EVENT_READ)
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise NoteSearchWorkerError("Note-search worker exceeded its wall-time limit.")
                    events = selector.select(remaining)
                    for key, _ in events:
                        if key.fileobj is self.process.stdin:
                            sent = os.write(key.fd, sending[:65536])
                            sending = sending[sent:]
                            if not sending:
                                selector.unregister(key.fileobj)
                        else:
                            part = os.read(key.fd, 65536)
                            if not part:
                                raise NoteSearchWorkerError("Note-search worker stopped without a result.")
                            received.extend(part)
                            if len(received) > self.limits.max_output_chars * 12 + 10000:
                                raise NoteSearchWorkerError("Note-search worker exceeded its output limit.")
                            if b"\n" in part:
                                result = json.loads(received)
                                if not isinstance(result, dict):
                                    raise ValueError("Invalid result envelope")
                                return result
        except BaseException as exc:
            self.close()
            if isinstance(exc, (OSError, ValueError)):
                raise NoteSearchWorkerError("Invalid note-search worker response or transport.") from None
            raise

    def execute(self, code):
        if not isinstance(code, str) or len(code) > self.limits.max_code_chars:
            raise NoteSearchWorkerError("Search cell exceeds the configured code limit.")
        result = self._request({"code": code})
        try:
            valid = (
                set(result) == {"output", "error", "error_detail", "truncated", "source_spans", "sources_truncated"}
                and isinstance(result["output"], str) and len(result["output"]) <= self.limits.max_output_chars
                and type(result["truncated"]) is bool and type(result["sources_truncated"]) is bool
                and isinstance(result["source_spans"], list) and len(result["source_spans"]) <= 64
                and all(isinstance(s, list) and len(s) == 2
                        and all(type(n) is int for n in s) and 0 <= s[0] < s[1] <= self.history_length
                        for s in result["source_spans"])
                and all(result[key] is None or (isinstance(result[key], str) and len(result[key]) <= limit)
                        for key, limit in (("error", 100), ("error_detail", 500)))
            )
        except (KeyError, TypeError):
            valid = False
        if not valid:
            self.close()
            raise NoteSearchWorkerError("Note-search worker returned an invalid cell result.") from None
        return result

    def close(self):
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait(timeout=5)
        for stream in (self.process.stdin, self.process.stdout):
            if stream and not stream.closed:
                stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
