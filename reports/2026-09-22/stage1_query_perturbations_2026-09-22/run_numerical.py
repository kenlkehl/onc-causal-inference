"""One fresh numerical worker per fold, avoiding repeated PyTorch interop setup.

Use this entry point for the pilot. The originally frozen fit.py scientific
implementation is unchanged; finished folds are integrity-checked and reused.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback
from common import HERE, RUN, POLICY, now, read, sha, write
from fit import fit_one


def main():
    write(RUN / "numerical_scheduler.json", {"at": now(), "entry_point": str(HERE / "run_numerical.py"),
          "source_sha256": sha(__file__), "max_workers": POLICY["fold_workers"], "max_tasks_per_child": 1,
          "reason": "PyTorch interop threads can be configured only once in each process; frozen science is unchanged."})
    with ProcessPoolExecutor(max_workers=POLICY["fold_workers"], max_tasks_per_child=1) as pool:
        for result in as_completed([pool.submit(fit_one, i) for i in range(1, 6)]):
            result.result()
    write(RUN / "numerical_frozen.json", {"at": now(), "manifest_sha256": sha(RUN / "manifest.json"),
          "scheduler": read(RUN / "numerical_scheduler.json"),
          "folds": {f"inner_{i:03d}": sha(RUN / "folds" / f"inner_{i:03d}" / "frozen.json") for i in range(1, 6)}})


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        write(RUN / "numerical_failed.json", {"at": now(), "error": str(error), "traceback": traceback.format_exc()})
        raise
