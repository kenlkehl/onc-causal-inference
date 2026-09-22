"""Complete estimation and reporting when the running selection is frozen."""

import subprocess
import sys
import time

from common import HERE, DATE, read, write, now


def main():
    deadline = time.monotonic() + 12 * 3600
    while not (HERE / f"selection_frozen_{DATE}.json").exists():
        status = read(HERE / "selection_status.json")
        if status["phase"] == "failed":
            raise RuntimeError(status["error"])
        if time.monotonic() > deadline:
            raise TimeoutError("Selection has not finished")
        time.sleep(10)
    for stage in ("fit", "evaluate"):
        with (HERE / f"{stage}.log").open("a") as log:
            subprocess.run([sys.executable, "-u", str(HERE / f"{stage}_{DATE}.py")], stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        write(HERE / "finish_status.json", {"phase": "failed", "at": now(), "error": str(exc)})
        raise
