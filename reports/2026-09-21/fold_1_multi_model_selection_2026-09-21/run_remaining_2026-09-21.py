"""Continue this experiment after the independently checkpointed numerical pass."""
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
print("Waiting for the frozen numerical evidence", flush=True)
while not (HERE / "numerical_frozen_2026-09-21.json").exists():
    log = (HERE / "numerical.log").read_text()
    if "Traceback (most recent call last)" in log:
        raise RuntimeError("The numerical pass failed; inspect numerical.log")
    time.sleep(10)

for phase, script in [
    ("numerical_validation", "validate_numerical_2026-09-21.py"),
    ("adjudication", "adjudicate_2026-09-21.py"),
    ("fitting", "fit_2026-09-21.py"),
    ("evaluation", "evaluate_2026-09-21.py"),
]:
    print(f"Starting {phase}", flush=True)
    with (HERE / f"{phase}.log").open("a") as output:
        subprocess.run([sys.executable, "-u", str(HERE / script)],
                       stdout=output, stderr=subprocess.STDOUT, check=True)
    print(f"Completed {phase}", flush=True)
print("Experiment complete; inspect evaluation_2026-09-21.json", flush=True)
