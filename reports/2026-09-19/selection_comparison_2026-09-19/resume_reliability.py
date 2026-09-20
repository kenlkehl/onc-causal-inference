"""Run the explicit reliability revision once, then evaluate and write its report."""
import fcntl
import os
import subprocess
import sys

from compare import HERE, now, sha256, write_json

REVISION = HERE / "results/revisions/reliability_v2"


def main():
    # Prevent two copies of this revision launcher from producing concurrent writes.
    with (REVISION / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (REVISION / "runner_started.json").exists():
            raise RuntimeError("This one-shot revision was already launched; inspect state before another resume")
        manifest = REVISION / "manifest.json"
        write_json(REVISION / "runner_started.json", {"pid": os.getpid(), "started_at": now(),
                                                     "manifest_sha256": sha256(manifest)})
        commands = [
            [sys.executable, "-u", str(HERE / "compare.py"), "run", "--revision", str(manifest)],
            [sys.executable, "-u", str(HERE / "compare.py"), "evaluate"],
            [sys.executable, "-u", str(HERE / "report.py")],
        ]
        for command in commands:
            child = subprocess.Popen(command)
            state = {"phase": "running", "command": command, "pid": child.pid,
                     "supervisor_pid": os.getpid(), "started_at": now()}
            write_json(REVISION / "runner_state.json", state)
            code = child.wait()
            write_json(REVISION / "runner_state.json", {**state, "phase": "completed" if code == 0 else "failed",
                                                        "returncode": code, "finished_at": now()})
            if code:
                raise SystemExit(code)
        write_json(REVISION / "runner_complete.json", {"completed_at": now()})


if __name__ == "__main__":
    main()
