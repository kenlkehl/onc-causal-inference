"""Schedule independent ranking batches concurrently, then verify native replay.

Only two independent list comprehensions become ordered executor maps. Prompt
contents, merge order, validators, checkpoint identities, and chosen prefixes
remain native. A cache-only replay of the original function checks every result.
"""

from concurrent.futures import ThreadPoolExecutor
import importlib.util
import inspect
from pathlib import Path
import threading

from common import HERE, DATE, write, sha, now
from oci.inference import stage2_modifier_ranking as ranking

NATIVE = ranking.rank_modifier_candidates
SOURCE = inspect.getsource(NATIVE)
INITIAL = "lists = [request(batch)[:maximum] for batch in batches]"
MERGE = """lists = [
            merge(lists[i], lists[i + 1]) if i + 1 < len(lists) else lists[i]
            for i in range(0, len(lists), 2)
        ]"""
assert SOURCE.count(INITIAL) == SOURCE.count(MERGE) == 1
PARALLEL = SOURCE.replace(INITIAL, "lists = list(_parallel_map(lambda batch: request(batch)[:maximum], batches))")
PARALLEL = PARALLEL.replace(MERGE, "lists = list(_parallel_map(lambda i: merge(lists[i], lists[i + 1]) if i + 1 < len(lists) else lists[i], range(0, len(lists), 2)))")


def parallel_rank(**arguments):
    root = Path(arguments["output_dir"])
    if (root / "ranking.json").exists():
        return NATIVE(**arguments)
    workers = 3 if root.name == "ranking" else 16
    scope = root.parent.name if root.name == "ranking" else root.name
    source_request = arguments["request_json"]
    # Preserve the calling script's request-audit context in child threads.
    local = inspect.getclosurevars(source_request).nonlocals.get("state")

    def request(*args, **kwargs):
        if local is not None:
            name = f"{scope}/worker_{threading.get_ident()}"
            if getattr(local, "name", None) != name:
                local.name, local.number = name, 0
        return source_request(*args, **kwargs)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        namespace = {**ranking.__dict__, "_parallel_map": pool.map}
        exec(compile(PARALLEL, str(Path(__file__)), "exec"), namespace)
        prefetched = namespace["rank_modifier_candidates"](**{**arguments, "request_json": request})

    def miss(*args, **kwargs):
        raise RuntimeError("Parallel ranking differed from native request order or payloads")

    replayed = NATIVE(**{**arguments, "request_json": miss})
    assert prefetched == replayed
    write(root / "parallel_scheduler.json", {"at": now(), "workers": workers,
          "native_ranking_source_sha256": sha(ranking.__file__), "scheduler_source_sha256": sha(__file__),
          "scheduler_path": str(Path(__file__).resolve()), "native_replay_without_cache_misses": True,
          "changes": "Independent initial batches and merge pairs use ordered executor.map; sequential merges are unchanged."})
    return replayed


def main():
    spec = importlib.util.spec_from_file_location("selection_run", HERE / f"select_{DATE}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ranking.rank_modifier_candidates = parallel_rank
    module.main()


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        import traceback
        write(HERE / "selection_status.json", {"phase": "failed", "at": now(), "error": str(exc), "traceback": traceback.format_exc()})
        raise
