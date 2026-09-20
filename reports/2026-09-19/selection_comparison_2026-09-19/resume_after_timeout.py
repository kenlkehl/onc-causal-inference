"""One unchanged-input resume after the original extraction worker drains."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "results"
RECOVERY = OUT / "recovery/transport_timeout_2026-09-19"
ORIGINAL_PIDS = {2151402, 2161190}
SCRIPTS = {HERE / "launch.py", HERE / "compare.py"}


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def is_worker(argv, cwd):
    for argument in argv[1:]:
        if not argument.endswith(("/launch.py", "/compare.py", "launch.py", "compare.py")):
            continue
        path = Path(argument)
        path = path if path.is_absolute() else Path(cwd) / path
        if path.resolve() in SCRIPTS:
            return True
    return False


def workers():
    found = []
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            if directory.stat().st_uid != os.getuid():
                continue
            argv = [part.decode() for part in (directory / "cmdline").read_bytes().split(b"\0") if part]
            cwd = (directory / "cwd").resolve(strict=True)
            if is_worker(argv, cwd):
                found.append({"pid": int(directory.name), "argv": argv, "cwd": str(cwd)})
        except (OSError, UnicodeError):
            continue
    return sorted(found, key=lambda item: item["pid"])


def state(phase, **details):
    value = {"updated_at": now(), "phase": phase, "supervisor_pid": os.getpid(), **details}
    temporary = RECOVERY / ".state.tmp"
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(RECOVERY / "state.json")
    with (RECOVERY / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(value) + "\n")
    print(json.dumps(value), flush=True)


def run_phase(phase, script, arguments):
    command = [sys.executable, "-u", str(HERE / script), *arguments]
    environment = dict(os.environ, MPLCONFIGDIR="/tmp/oci_selection_comparison_matplotlib",
                       OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    with (RECOVERY / "worker_output.log").open("a") as output:
        child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=output, stderr=subprocess.STDOUT)
        state(phase, child_pid=child.pid, command=command)
        result = child.wait()
    if result:
        raise RuntimeError(f"{phase} exited with status {result}; inspect worker_output.log")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args()
    if args.inspect:
        print(json.dumps(workers(), indent=2))
        return
    RECOVERY.mkdir(parents=True, exist_ok=True)
    with (RECOVERY / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt = RECOVERY / "resume_started.json"
        if receipt.exists():
            raise RuntimeError("This incident's single resume was already started; do not launch it again")
        try:
            deadline = time.monotonic() + 3 * 60 * 60
            prior = None
            while True:
                active = workers()
                if any(worker["pid"] not in ORIGINAL_PIDS for worker in active):
                    raise RuntimeError(f"A different comparison worker is active: {[w['pid'] for w in active]}")
                if not active:
                    break
                if active != prior:
                    state("waiting_for_original_worker_to_exit", workers=active)
                    prior = active
                if time.monotonic() >= deadline:
                    raise RuntimeError("Original worker has not drained within three hours; no resume launched")
                time.sleep(30)
            # A second check prevents ordinary manual restarts during the wait
            # from being mistaken for permission to launch another comparison.
            if workers():
                raise RuntimeError("Comparison worker appeared before resume; leaving it untouched")
            receipt.write_text(json.dumps({"started_at": now(), "supervisor_pid": os.getpid(),
                "reason": "one extraction logical request exhausted its 7200-second transport/repair deadline",
                "configuration_changes": [], "scientific_source_changes": [],
                "source_hash_validation": "compare.py run validates the original frozen source manifest",
                "checkpoint_policy": "reuse only checkpoints accepted by the original fingerprint checks"}, indent=2) + "\n")
            run_phase("resuming_unchanged_comparison", "compare.py", ["run"])
            run_phase("evaluating_frozen_predictions", "compare.py", ["evaluate"])
            run_phase("rendering_report", "report.py", [])
            state("report_ready_for_review", report=str(HERE / "selection_comparison_report_2026-09-19.md"))
        except Exception as error:
            state("recovery_failed", error=f"{type(error).__name__}: {error}")
            raise


if __name__ == "__main__":
    main()
