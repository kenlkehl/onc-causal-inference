"""Continue the authorized local pilot through checked inner-validation reporting."""
import subprocess
import sys
import time
import traceback
from common import HERE, RUN, now, read, write


def progress():
    folds = {}
    for i in range(1, 6):
        path = RUN / "folds" / f"inner_{i:03d}" / "status.json"
        folds[str(i)] = read(path) if path.exists() else {"phase": "waiting"}
    llm = read(RUN / "llm_status.json") if (RUN / "llm_status.json").exists() else {"phase": "waiting", "completed": 0}
    complete = (RUN / "complete.json").exists()
    lines = ["# Stage 1 query perturbations — September 22, 2026", "", f"Updated: {now()}", "",
             "Only outer fold 1 training patients are used. All 200 outer-test patients are excluded.", "",
             f"- Numerical folds complete: {sum(x['phase'] == 'complete' for x in folds.values())}/5.",
             f"- LLM interpretations complete: {llm.get('completed', 0)}/75.",
             f"- Final evaluation: {'complete' if complete else 'waiting for numerical and LLM freezes'}.", "",
             "| Inner split | Numerical phase |", "| --- | --- |",
             *[f"| {i} | {value['phase']} |" for i, value in folds.items()], "",
             "[Protocol](PILOT_PROTOCOL_2026-09-22.md) · [Final report](REPORT_2026-09-22.md)", ""]
    (HERE / "STATUS_2026-09-22.md").write_text("\n".join(lines))


def main():
    deadline = time.monotonic() + 12 * 3600
    while not ((RUN / "numerical_frozen.json").exists() and (RUN / "llm_frozen.json").exists()):
        progress()
        for kind in ("numerical", "llm"):
            if (RUN / f"{kind}_failed.json").exists():
                raise RuntimeError(f"{kind} phase failed; inspect its saved failure artifact")
        if time.monotonic() > deadline:
            raise TimeoutError("Pilot prerequisites did not complete in 12 hours")
        time.sleep(20)
    with (RUN / "evaluation.log").open("a") as log:
        subprocess.run([sys.executable, "-u", str(HERE / "evaluate.py")], check=True, stdout=log, stderr=subprocess.STDOUT)
    progress()


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        write(RUN / "finish_failed.json", {"at": now(), "error": str(error), "traceback": traceback.format_exc()})
        raise
