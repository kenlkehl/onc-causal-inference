"""Resume the frozen experiment under a detached, exit-recording supervisor."""

import fcntl
import os
import subprocess
import sys
import traceback

from common import DATE, HERE, manifest, now, sha, write


def main():
    lock_path = HERE / "llm_runtime" / "recovery_supervisor.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest()
        base = {
            "started_at": now(),
            "supervisor_pid": os.getpid(),
            "supervisor_source_sha256": sha(__file__),
            "reason": "Prior selection, completion, and progress processes were absent; resume verified checkpoints without changing scientific settings.",
        }
        stages = (
            ("selection", HERE / f"select_parallel_{DATE}.py", "recovery_selection.log"),
            ("fit", HERE / f"fit_{DATE}.py", "fit.log"),
            ("evaluate", HERE / f"evaluate_{DATE}.py", "evaluate.log"),
        )
        monitor = None
        history = []
        current_stage = "preflight"
        try:
            with (HERE / "progress.log").open("a") as monitor_log:
                monitor = subprocess.Popen(
                    [sys.executable, "-u", str(HERE / f"progress_{DATE}.py"), "--watch"],
                    cwd=HERE,
                    stdin=subprocess.DEVNULL,
                    stdout=monitor_log,
                    stderr=subprocess.STDOUT,
                )
                for current_stage, script, log_name in stages:
                    with (HERE / log_name).open("a") as output:
                        child = subprocess.Popen(
                            [sys.executable, "-u", str(script)],
                            cwd=HERE,
                            stdin=subprocess.DEVNULL,
                            stdout=output,
                            stderr=subprocess.STDOUT,
                        )
                        write(HERE / "recovery_status.json", {
                            **base, "at": now(), "phase": current_stage,
                            "child_pid": child.pid, "monitor_pid": monitor.pid,
                            "completed_stages": history,
                        })
                        write(HERE / "finish_status.json", {
                            "at": now(), "phase": "waiting_for_selection" if current_stage == "selection" else current_stage,
                            "supervisor_pid": os.getpid(), "child_pid": child.pid,
                        })
                        returncode = child.wait()
                    history.append({"stage": current_stage, "finished_at": now(), "returncode": returncode})
                    if returncode:
                        raise RuntimeError(f"{current_stage} exited with code {returncode}; see {log_name}")
                if not (HERE / "experiment_complete.json").exists():
                    raise RuntimeError("Evaluation exited without an experiment completion record")
                write(HERE / "recovery_status.json", {
                    **base, "at": now(), "phase": "complete", "completed_stages": history,
                })
                write(HERE / "finish_status.json", {"at": now(), "phase": "complete"})
        except BaseException as exc:
            write(HERE / "recovery_status.json", {
                **base, "at": now(), "phase": "failed", "failed_stage": current_stage,
                "completed_stages": history, "error": str(exc), "traceback": traceback.format_exc(),
            })
            write(HERE / "finish_status.json", {"at": now(), "phase": "failed", "error": str(exc)})
            raise
        finally:
            if monitor is not None and monitor.poll() is None:
                monitor.terminate()
                monitor.wait(timeout=20)
            with (HERE / "progress.log").open("a") as output:
                subprocess.run(
                    [sys.executable, "-u", str(HERE / f"progress_{DATE}.py")],
                    cwd=HERE, stdin=subprocess.DEVNULL, stdout=output,
                    stderr=subprocess.STDOUT, check=False,
                )


if __name__ == "__main__":
    main()
