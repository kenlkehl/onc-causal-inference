"""Extend saved native fold rankings to 100 with verified checkpoint reuse."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import inspect
import logging
import os
from pathlib import Path
import shutil
import threading
import traceback

from common import HERE, PRIOR, INPUTS, N, MODEL, read, write, sha, now, verify, runtime, request
from oci.inference import stage2_modifier_ranking as ranking
from oci.inference.stage2_role_adjudication import role_adjudication_config_from_mapping

NATIVE = ranking.rank_modifier_candidates
NATIVE_SOURCE = inspect.getsource(NATIVE)
INITIAL = "lists = [request(batch)[:maximum] for batch in batches]"
MERGE = """lists = [
            merge(lists[i], lists[i + 1]) if i + 1 < len(lists) else lists[i]
            for i in range(0, len(lists), 2)
        ]"""
assert NATIVE_SOURCE.count(INITIAL) == NATIVE_SOURCE.count(MERGE) == 1
PARALLEL = NATIVE_SOURCE.replace(INITIAL, "lists = list(_parallel_map(lambda batch: request(batch)[:maximum], batches))")
PARALLEL = PARALLEL.replace(MERGE, "lists = list(_parallel_map(lambda i: merge(lists[i], lists[i + 1]) if i + 1 < len(lists) else lists[i], range(0, len(lists), 2)))")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if (HERE / "rankings_frozen.json").exists():
        verify(read(HERE / "rankings_frozen.json")["files"])
        return
    plan = read(PRIOR / "input_manifest_2026-09-22.json")
    # Numerical identity and all source hashes from the completed run must match.
    verify(plan["sources"])
    verify({str(INPUTS / "definitions.json"): plan["input_files"][str(INPUTS / "definitions.json")]})
    definitions = read(INPUTS / "definitions.json")
    previous = read(PRIOR / "count_result.json")
    role_policy = role_adjudication_config_from_mapping(plan["role_policy"])
    service = runtime()
    source_freeze = read(PRIOR / "selection_frozen_2026-09-22.json")["files"]
    files = {str(INPUTS / "definitions.json"): sha(INPUTS / "definitions.json"),
             str(PRIOR / "count_result.json"): sha(PRIOR / "count_result.json"),
             str(HERE / "PROTOCOL_2026-09-23.md"): sha(HERE / "PROTOCOL_2026-09-23.md"),
             str(Path(__file__)): sha(__file__), str(HERE / "common.py"): sha(HERE / "common.py"),
             str(Path(ranking.__file__)): sha(ranking.__file__)}
    write(HERE / "ranking_input.json", {"at": now(), "baseline_commit": "568ef7a", "n": N,
          "files": files, "model": service.model_identity, "policy": plan["policy"],
          "no_numerical_or_extraction_rerun": True, "no_outer_test_or_oracle_inputs": True})

    def one_fold(old):
        number = old["inner_fold"]
        root = HERE / "ranking_cache" / f"fold_{number:03d}"
        output = HERE / "fold_rankings" / f"fold_{number:03d}.json"
        state = HERE / "fold_rankings" / f"fold_{number:03d}_status.json"
        status = read(PRIOR / "numerical_jobs" / f"job_{number}" / "status.json")
        verify({status["report_path"]: status["report_sha256"]})
        report = read(status["report_path"])
        assert report["input_fingerprint"] == old["nested_evidence_fingerprint"]
        old_root = Path(old["ranking_path"]).parent
        copied = 0
        # Keep old checkpoints immutable. New comparisons only write into root.
        for src in sorted((old_root / "requests").glob("*/*")):
            if src.suffix != ".json":
                continue
            if str(src) not in source_freeze:
                raise ValueError(f"Unfrozen ranking checkpoint: {src}")
            verify({str(src): source_freeze[str(src)]})
            dst = root / "requests" / src.relative_to(old_root / "requests")
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
                copied += 1
            else:
                assert sha(dst) == sha(src)
        lock = threading.Lock()
        calls, completed = 0, 0

        def get_json(messages, validate, *, request_kind="interpretation"):
            nonlocal calls, completed
            with lock:
                calls += 1
                serial = calls
                write(state, {"at": now(), "phase": "extending", "fold": number,
                              "new_requests": calls, "completed_requests": completed})
            result = request(service, messages=messages, validate=validate,
                             audit_dir=root / f"live_requests/request_{serial:04d}")
            with lock:
                completed += 1
                write(state, {"at": now(), "phase": "extending", "fold": number,
                              "new_requests": calls, "completed_requests": completed})
            return result

        args = dict(definitions=definitions, statistical_report=report, request_json=get_json,
                    output_dir=root, role_policy=role_policy, maximum=N,
                    maximum_chars=plan["policy"]["multi_model"]["max_prompt_chars"], model_identity=MODEL)
        write(state, {"at": now(), "phase": "extending", "fold": number,
                      "copied_checkpoint_files": copied, "new_requests": 0, "completed_requests": 0})
        with ThreadPoolExecutor(max_workers=3) as pool:
            namespace = {**ranking.__dict__, "_parallel_map": pool.map}
            exec(compile(PARALLEL, str(Path(__file__)), "exec"), namespace)
            result = namespace["rank_modifier_candidates"](**args)

        def miss(*args, **kwargs):
            raise RuntimeError("Native replay required an uncached request")

        assert NATIVE(**{**args, "request_json": miss}) == result
        ids = [r["feature_id"] for r in result["ranking"]]
        assert len(ids) == len(set(ids)) == N
        assert ids[:len(old["ranking"])] == [r["feature_id"] for r in old["ranking"]]
        value = {**result, "inner_fold": number,
                 "original_prefix_preserved": True, "native_replay_verified": True,
                 "numerical_evidence_sha256": status["report_sha256"],
                 "numerical_evidence_path": status["report_path"],
                 "training_rows": len(old["ranking_training_row_ids"]),
                 "scoring_rows": len(old["scoring_row_ids"])}
        write(output, value)
        write(state, {"at": now(), "phase": "complete", "fold": number,
                      "new_requests": calls, "completed_requests": completed, "ranked": len(ids)})
        print({"fold": number, "phase": "complete", "new_requests": calls}, flush=True)
        return output

    write(HERE / "status.json", {"at": now(), "phase": "extending_fold_rankings", "pid": os.getpid()})
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(one_fold, old) for old in previous["folds"]]
        outputs = [future.result() for future in as_completed(futures)]
    files.update({str(p): sha(p) for p in outputs})
    write(HERE / "rankings_frozen.json", {"at": now(), "files": files, "n_per_fold": N,
          "folds": 5, "all_original_prefixes_preserved": True, "all_native_replays_verified": True})
    write(HERE / "status.json", {"at": now(), "phase": "rankings_complete", "n_per_fold": N})


if __name__ == "__main__":
    lockfile = (HERE / "rankings.lock").open("w")
    fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        main()
    except BaseException as exc:
        write(HERE / "status.json", {"at": now(), "phase": "failed", "error": str(exc),
                                    "traceback": traceback.format_exc()})
        raise
