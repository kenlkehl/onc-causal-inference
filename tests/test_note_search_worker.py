"""Exercise OCI's actual isolated process without an external checkout or LLM."""
import builtins
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

import pytest

from oci.inference import note_search_worker as transport
from oci.inference import stage2_note_search as search


def test_backend_never_imports_matchminer_and_has_no_checkout_setting(monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.startswith("matchminer"):
            pytest.fail("OCI must not import MatchMiner")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    config = search.NoteSearchConfig(enabled=True)
    worker, identity = search.load_backend(config)
    assert worker.__module__ == "oci.inference.note_search_worker"
    assert identity["implementation"] == "oci_builtin"
    assert "source_checkout" not in asdict(config)
    with pytest.raises(ValueError, match="Unknown extraction_note_search options"):
        search.config_from_mapping({"source_checkout": "/retired-checkout"})
    search.preflight(config)


def test_exact_scan_counts_bounds_pagination_and_unicode_provenance():
    text = "α ECOG 1. ECOG 2. Age 67. " * 600
    config = search.NoteSearchConfig(max_output_chars=2000)
    with transport.NoteSearchWorker(text, config) as worker:
        raw = worker.execute("result = scan(['ECOG'] * 128, context=8, limit=4); print(json.dumps(result))")
        result = json.loads(raw["output"])
        assert result["match_count"] == 1200 and result["omitted"]
        assert len(result["hits"]) <= 4 and len(raw["output"]) <= 2000
        assert not raw["truncated"]
        for hit in result["hits"]:
            assert text[hit["start"]:hit["end"]] == hit["quote"]
            assert [hit["start"], hit["end"]] in raw["source_spans"]
        first = json.loads(worker.execute("p=search('ECOG', context=0, limit=1); print(json.dumps(p))")["output"])
        second = json.loads(worker.execute("print(json.dumps(search('ECOG', start=p['next_start'], context=0, limit=1)))")["output"])
        assert first["has_more"] and second["hits"][0]["match_start"] > first["hits"][0]["match_start"]
        assert worker.execute("scan(['.*?'])")["error"] == "ValueError"
        assert worker.execute("scan(['ECOG'], context=2001)")["error"] == "ValueError"


def test_source_and_output_overflow_are_bounded_and_patient_processes_are_separate():
    limits = search.NoteSearchConfig(max_output_chars=1024)
    with transport.NoteSearchWorker("A" * 1000, limits) as first:
        raw = first.execute("for i in range(80): read(i, i + 1)\nprint('x' * 100000)")
        assert len(raw["source_spans"]) == 64 and raw["sources_truncated"]
        assert len(raw["output"]) == 1024 and raw["truncated"]
        first.execute("patient_secret = 'A'")
        with transport.NoteSearchWorker("B" * 1000, limits) as second:
            assert second.execute("patient_secret")["error"] == "NameError"
            assert second.execute("read(0, 1)")["source_spans"] == [[0, 1]]
    assert first.process.poll() is not None


def test_os_isolation_denies_files_network_processes_and_parent_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("OCI_SYNTHETIC_SECRET", "do-not-inherit")
    target = tmp_path / "must-not-exist"
    with transport.NoteSearchWorker("Synthetic patient", search.NoteSearchConfig()) as worker:
        assert worker.execute(f"open({str(target)!r}, 'w')")["error"] == "PermissionError"
        assert not target.exists()
        assert worker.execute("import os; os.fork()")["error"] == "PermissionError"
        assert worker.execute("os.execv('/bin/true', ['true'])")["error"] == "PermissionError"
        assert worker.execute("os.getenv('OCI_SYNTHETIC_SECRET', 'absent')")["output"].strip() == "'absent'"
        # Direct libc socket syscall also fails, independently of Python imports.
        assert worker.execute("import ctypes; ctypes.CDLL(None).socket(2, 1, 0)")["output"].strip() == "-1"


@pytest.mark.parametrize("code", ["while True: pass", "re.search('(a+)+$', 'a' * 100000 + '!')"])
def test_wall_timeout_kills_infinite_code_and_catastrophic_regex(code):
    worker = transport.NoteSearchWorker("Synthetic", search.NoteSearchConfig(cell_timeout_seconds=0.5))
    with pytest.raises(transport.NoteSearchWorkerError, match="wall-time"):
        worker.execute(code)
    assert worker.process.poll() is not None


def test_memory_exhaustion_does_not_become_a_missing_measurement():
    with transport.NoteSearchWorker("Synthetic", search.NoteSearchConfig(worker_memory_mb=128)) as worker:
        result = worker.execute("bytearray(300 * 1024 * 1024)")
        assert result["error"] == "MemoryError"
        assert result["source_spans"] == []


def test_unavailable_isolation_fails_closed_before_generated_code():
    child = Path(transport.__file__).with_name("_note_search_child.py")
    script = ("import runpy\n"
              f"module = runpy.run_path({str(child)!r})\n"
              "def unavailable(*a, **k): raise OSError('unavailable')\n"
              "module['ctypes'].CDLL = unavailable\n"
              "module['serve']()\n")
    setup = {"history": "Synthetic", "memory_mb": 128, "output_chars": 1000, "patterns": 128}
    process = subprocess.run([sys.executable, "-I", "-S", "-c", script], input=json.dumps(setup) + "\n"
                             + json.dumps({"code": "print('MUST NOT RUN')"}) + "\n",
                             text=True, capture_output=True, timeout=5, env={}, cwd="/")
    assert process.stdout == '{"ready":false}\n'


def test_malformed_worker_provenance_is_rejected_and_process_closed(monkeypatch):
    # Validation uses explicit conditions, independent of Python assertions.
    worker = transport.NoteSearchWorker("Synthetic", search.NoteSearchConfig())
    monkeypatch.setattr(worker, "_request", lambda *_: {"output": "", "error": None,
        "error_detail": None, "truncated": False, "source_spans": [[-1, 100]], "sources_truncated": False})
    with pytest.raises(transport.NoteSearchWorkerError, match="invalid cell result"):
        worker.execute("read(0, 1)")
    assert worker.process.poll() is not None
