"""Wait for the user-started extractor, verify it, then run the experiment."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time
import urllib.request

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
OUT.mkdir(parents=True, exist_ok=True)
EXTRACTION = "http://sn4622130540:8001/v1"
PRIMARY = "http://sn4622130540:8000/v1"


def models(endpoint):
    with urllib.request.urlopen(endpoint + "/models", timeout=10) as response:
        return json.load(response)["data"]


def main():
    started = time.monotonic()
    while True:
        try:
            available = models(EXTRACTION)
            matches = [m for m in available if all(s in json.dumps(m).lower() for s in ("gemma", "26b", "a4b"))]
            if len(matches) != 1:
                raise RuntimeError(f"Expected one Gemma 4 26B A4B extractor; received {[m['id'] for m in available]}")
            extractor = matches[0]
            primary = models(PRIMARY)
            if not any(m["id"] == "gemma4-31b" for m in primary):
                raise RuntimeError("The original gemma4-31b adjudicator is not advertised")
            break
        except urllib.error.URLError as error:
            elapsed = int(time.monotonic() - started)
            status = {"phase": "waiting_for_extractor_startup", "elapsed_seconds": elapsed,
                      "endpoint": EXTRACTION, "last_error": str(error)}
            (OUT / "startup_status.json").write_text(json.dumps(status, indent=2) + "\n")
            print(json.dumps(status), flush=True)
            if elapsed >= 1800:
                raise RuntimeError("Extractor still unavailable after 30 minutes; rerun launch.py once started") from error
            time.sleep(30)
    (OUT / "verified_services.json").write_text(json.dumps({"primary": primary, "extractor": extractor}, indent=2) + "\n")
    print(json.dumps({"phase": "services_ready", "extractor_model": extractor["id"]}), flush=True)
    environment = dict(os.environ, MPLCONFIGDIR="/tmp/oci_selection_comparison_matplotlib",
                       OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    command = [sys.executable, "-u", str(HERE / "compare.py")]
    subprocess.run(command + ["prepare", "--extractor-model", extractor["id"]], env=environment, check=True)
    subprocess.run(command + ["run"], env=environment, check=True)
    subprocess.run(command + ["evaluate"], env=environment, check=True)


if __name__ == "__main__":
    main()
