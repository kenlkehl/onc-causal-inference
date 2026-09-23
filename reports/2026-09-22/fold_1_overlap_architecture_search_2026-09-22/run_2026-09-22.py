"""Detached supervisor: evidence/ranking, final fits, then frozen evaluation."""

import fcntl
import importlib.util
import os
import signal
import subprocess
import sys
import time
import traceback

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/stage2-mpl")

from common import HERE, DATE, now, read, write, manifest, sha


def load(name):
    path = HERE / f"{name}_{DATE}.py"
    spec = importlib.util.spec_from_file_location(f"experiment_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    lock_path = HERE / "supervisor.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest()
        progress = load("progress")
        if (HERE / "experiment_complete.json").exists():
            progress.snapshot()
            return
        processes, logs, history = {}, [], []
        base = {"started_at": now(), "pid": os.getpid(), "process_group": os.getpgrp(),
                "source_sha256": sha(__file__)}
        stage = "starting"

        def status(phase, **extra):
            write(HERE / "supervisor_status.json", {
                **base, "at": now(), "phase": phase, "completed_stages": history,
                "children": {name: {"pid": process.pid, "process_group": process.pid,
                                     "returncode": process.poll()} for name, process in processes.items()},
                **extra,
            })

        def start(name, script):
            output = (HERE / f"{name}.log").open("a")
            logs.append(output)
            processes[name] = subprocess.Popen(
                [sys.executable, "-u", str(HERE / f"{script}_{DATE}.py")], cwd=HERE,
                stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                start_new_session=True,
            )

        def wait_for(names, phase):
            pending = set(names)
            while pending:
                for name in list(pending):
                    code = processes[name].poll()
                    if code is None:
                        continue
                    history.append({"stage": name, "at": now(), "returncode": code})
                    if code:
                        raise RuntimeError(f"{name} failed with code {code}; see {name}.log")
                    pending.remove(name)
                status(phase)
                progress.snapshot()
                if pending:
                    time.sleep(30)

        def stop(signum, frame):
            raise InterruptedError(f"Experiment stopped by signal {signum}")

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        try:
            stage = "evidence_and_selection"
            start("numerical", "numerical")
            start("selection", "select_parallel")
            wait_for(("numerical", "selection"), stage)
            if not (HERE / f"selection_frozen_{DATE}.json").exists():
                raise RuntimeError("Selection exited without a frozen selection")
            for stage in ("fit", "evaluate"):
                start(stage, stage)
                wait_for((stage,), stage)
            if not (HERE / "experiment_complete.json").exists():
                raise RuntimeError("Evaluation exited without a completion record")
            status("complete")
        except BaseException as exc:
            status("cancelled" if isinstance(exc, InterruptedError) else "failed",
                   failed_stage=stage, error=str(exc), traceback=traceback.format_exc())
            raise
        finally:
            for process in processes.values():
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=10)
            for output in logs:
                output.close()
            progress.snapshot()


if __name__ == "__main__":
    main()
