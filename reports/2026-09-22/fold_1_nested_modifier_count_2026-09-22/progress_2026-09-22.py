"""Display experiment checkpoints; optionally refresh a local status document."""

import argparse
from functools import lru_cache
import json
import os
import time

from common import HERE, DATE, now


def read_or_empty(path):
    return json.loads(path.read_text()) if path.exists() else {}


@lru_cache(maxsize=4096)
def request_summary(path, modified_ns, size):
    # Checkpoint prompts are large and stable. Cache only their display metadata;
    # changed files get a new key and are read again.
    payload = read_or_empty(path)["payload"]
    return payload["task"], len(payload["candidates"])


def summarize_request(path):
    metadata = path.stat()
    return request_summary(path, metadata.st_mtime_ns, metadata.st_size)


def snapshot():
    numerical = [read_or_empty(p) for p in sorted((HERE / "numerical_jobs").glob("job_*/status.json"))]
    selection = read_or_empty(HERE / "selection_status.json")
    root_status = read_or_empty(HERE / "status.json")
    recovery = read_or_empty(HERE / "recovery_status.json")
    folders = sorted((HERE / "modifier_count").glob("*/fold_*"))
    rows = []
    for folder in folders:
        responses = list(folder.glob("ranking/requests/*/response.json"))
        ranking = read_or_empty(folder / "ranking/ranking.json")
        if ranking:
            candidates = len(ranking["reviewed_candidate_ids"])
        else:
            candidates = sum(count for task, count in (
                summarize_request(response.parent / "prompt.json") for response in responses
            ) if task == "rank_stage2_modifiers")
        rows.append({"fold": folder.name, "candidates_reviewed": candidates,
                     "requests_accepted": len(responses), "ranking_complete": bool(ranking),
                     "count_fits": len(list(folder.glob("size_*/seed_*.json")))})
    full = {"initial_batches": 0, "merge_requests": 0, "candidates_reviewed": 0,
            "requests_started": 0, "requests_accepted": 0, "ranking_complete": False}
    for directory in (HERE / "modifier_count").glob("*/full_training_ranking"):
        full["requests_started"] += len(list(directory.glob("requests/*/prompt.json")))
        full["ranking_complete"] |= (directory / "ranking.json").exists()
        for response in directory.glob("requests/*/response.json"):
            task, count = summarize_request(response.parent / "prompt.json")
            full["requests_accepted"] += 1
            if task == "rank_stage2_modifiers":
                full["initial_batches"] += 1
                full["candidates_reviewed"] += count
            else:
                full["merge_requests"] += 1
    report_path = HERE / f"REPORT_{DATE}.md"
    result = {"updated_at": now(), "numerical_jobs_complete": sum(r.get("phase") == "complete" for r in numerical),
              "selection_phase": selection.get("phase"), "pipeline_phase": root_status.get("phase"), "folds": rows,
              "full_training_ranking": full,
              "selection_frozen": (HERE / f"selection_frozen_{DATE}.json").exists(),
              "matched_fits_complete": len(list((HERE / "matched_residuals").glob("seed_*/metrics.json"))),
              "native_fits_complete": len(list((HERE / "production").glob("seed_*/fit_frozen.json"))),
              "predictions_frozen": (HERE / f"predictions_frozen_{DATE}.json").exists(),
              "final_report_exists": report_path.exists(),
              "complete": (HERE / "experiment_complete.json").exists() and report_path.exists()}
    failure = next((state.get("error", "See the execution log") for state in (selection, root_status, recovery)
                    if state.get("phase") == "failed" or str(state.get("phase", "")).endswith("_failed")), None)
    result["failed"] = failure is not None
    count_fits = sum(row["count_fits"] for row in rows)
    if result["complete"]:
        current = "Complete — final report available"
    elif failure:
        current = "Stopped — " + failure
    elif result["predictions_frozen"]:
        current = "In progress — evaluating held-out estimates and writing the final report"
    elif result["selection_frozen"]:
        current = "In progress — fitting the final forests"
    elif full["ranking_complete"]:
        current = "In progress — freezing the final selected features"
    elif full["requests_started"]:
        current = "In progress — building the full-training LLM ranking"
    elif count_fits == 120:
        current = "In progress — preparing the full-training LLM ranking"
    elif count_fits:
        current = "In progress — validating modifier counts"
    else:
        current = "In progress — numerical evidence and inner-fold rankings"
    result["current_stage"] = current
    if result["complete"]:
        result["final_result"] = root_status
    full_status = "Complete" if full["ranking_complete"] else (
        f"In progress: {full['candidates_reviewed']}/352 candidates initially reviewed; "
        f"{full['merge_requests']} merge requests accepted" if full["requests_started"] else "Pending")
    lines = [f"# Fold 1 nested modifier selection — {DATE}", "", f"Updated: {result['updated_at']}", "",
             f"**{current}.**", "",
             "The table of inner folds covers modifier-count validation. Final feature selection, model fitting, and evaluation are separate stages below.", "",
             "| Experiment stage | Status |", "| --- | --- |",
             f"| Numerical evidence | {result['numerical_jobs_complete']}/6 jobs complete |",
             f"| Inner-fold LLM rankings | {sum(row['ranking_complete'] for row in rows)}/5 complete |",
             f"| Modifier-count validation | {count_fits}/120 count/seed fits complete |",
             f"| Full-training LLM ranking | {full_status} |",
             f"| Final selected features frozen | {'Complete' if result['selection_frozen'] else 'Pending'} |",
             f"| Fixed-residual forest fits | {result['matched_fits_complete']}/3 complete |",
             f"| Native production forest fits | {result['native_fits_complete']}/3 complete |",
             f"| Held-out evaluation and final report | {'Complete' if result['complete'] else 'In progress' if result['predictions_frozen'] else 'Pending'} |", "",
             "| Count-validation fold | Candidates initially reviewed | LLM requests accepted | Ranking complete | Count/seed fits |",
             "| --- | ---: | ---: | --- | ---: |"]
    for row in rows:
        lines.append(f"| {row['fold']} | {row['candidates_reviewed']}/352 | {row['requests_accepted']} | {'Yes' if row['ranking_complete'] else 'No'} | {row['count_fits']}/24 |")
    lines += ["", "The original 189 confounders are retained. Modifier budgets are 0, 4, 8, 12, 16, 24, 32, and 64, with minimum mean nested R-loss as the selection rule.",
              "", "After count validation, the pipeline builds the full-training ranking, freezes selection, fits three native and three fixed-residual forests, then evaluates oracle recovery and held-out ITE performance.",
              "", f"[Protocol](PROTOCOL_{DATE}.md)", ""]
    if result["final_report_exists"]:
        lines += [f"[Final report](REPORT_{DATE}.md)", ""]
    else:
        lines += ["**Final report: not created yet.** It will be generated after final model fitting and held-out evaluation finish.", ""]
    path = HERE / f"STATUS_{DATE}.md"
    temp = path.with_suffix(f".md.{os.getpid()}.tmp")
    temp.write_text("\n".join(lines))
    temp.replace(path)
    return result


def main(watch):
    deadline = time.monotonic() + 12 * 3600
    while True:
        result = snapshot()
        print(json.dumps(result), flush=True)
        if not watch or result["complete"] or result["failed"] or time.monotonic() > deadline:
            break
        time.sleep(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    main(parser.parse_args().watch)
