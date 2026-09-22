"""Display experiment checkpoints; optionally refresh a local status document."""

import argparse
import json
from pathlib import Path
import time

from common import HERE, DATE, now


def read_or_empty(path):
    return json.loads(path.read_text()) if path.exists() else {}


def snapshot():
    numerical = [read_or_empty(p) for p in sorted((HERE / "numerical_jobs").glob("job_*/status.json"))]
    selection = read_or_empty(HERE / "selection_status.json")
    root_status = read_or_empty(HERE / "status.json")
    folders = sorted((HERE / "modifier_count").glob("*/fold_*"))
    rows = []
    for folder in folders:
        initial, merges, candidates = 0, 0, 0
        for response in folder.glob("ranking/requests/*/response.json"):
            payload = read_or_empty(response.parent / "prompt.json")["payload"]
            if payload["task"] == "rank_stage2_modifiers":
                initial += 1
                candidates += len(payload["candidates"])
            else:
                merges += 1
        rows.append({"fold": folder.name, "initial_batches": initial, "candidates_reviewed": candidates,
                     "merge_requests": merges, "ranking_complete": (folder / "ranking/ranking.json").exists(),
                     "count_fits": len(list(folder.glob("size_*/seed_*.json")))})
    result = {"updated_at": now(), "numerical_jobs_complete": sum(r.get("phase") == "complete" for r in numerical),
              "selection_phase": selection.get("phase"), "pipeline_phase": root_status.get("phase"), "folds": rows,
              "native_fits_complete": len(list((HERE / "production").glob("seed_*/fit_frozen.json"))),
              "complete": (HERE / "experiment_complete.json").exists()}
    if result["complete"]:
        result["final_result"] = root_status
    lines = [f"# Fold 1 nested modifier selection — {DATE}", "", f"Updated: {result['updated_at']}", "",
             f"- Numerical evidence jobs complete: **{result['numerical_jobs_complete']}/6**.",
             f"- Selection phase: **{result['selection_phase']}**.",
             f"- Pipeline phase: **{result['pipeline_phase']}**.",
             f"- Native production fits complete: **{result['native_fits_complete']}/3**.", "",
             "| Count-validation fold | Candidates initially reviewed | Initial batches accepted | Merge requests accepted | Ranking complete | Count/seed fits |",
             "| --- | ---: | ---: | ---: | --- | ---: |"]
    for row in rows:
        lines.append(f"| {row['fold']} | {row['candidates_reviewed']}/352 | {row['initial_batches']} | {row['merge_requests']} | {'Yes' if row['ranking_complete'] else 'No'} | {row['count_fits']}/24 |")
    if selection.get("phase") == "failed":
        lines += ["", "Selection stopped: " + str(selection.get("error"))]
    lines += ["", "The original 189 confounders are retained. Modifier budgets are 0, 4, 8, 12, 16, 24, 32, and 64, with minimum mean nested R-loss as the selection rule.",
              "", "After count validation, the pipeline builds the full-training ranking, freezes selection, fits three native and three fixed-residual forests, then evaluates oracle recovery and held-out ITE performance.",
              "", f"[Protocol](PROTOCOL_{DATE}.md) · [Final report](REPORT_{DATE}.md)",
              "", "The final report becomes available when the complete experiment finishes.", ""]
    path = HERE / f"STATUS_{DATE}.md"
    temp = path.with_suffix(".md.tmp")
    temp.write_text("\n".join(lines))
    temp.replace(path)
    return result


def main(watch):
    deadline = time.monotonic() + 12 * 3600
    while True:
        result = snapshot()
        print(json.dumps(result), flush=True)
        if not watch or result["complete"] or result["selection_phase"] == "failed" or time.monotonic() > deadline:
            break
        time.sleep(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    main(parser.parse_args().watch)
