"""Compare only frozen outputs; never make a model call or alter a selection."""

import csv
import datetime as dt
import hashlib
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "fold_1_cross_fold_modifier_concepts_2026-09-23"


def index_review(response):
    by_feature, representatives = {}, {}
    for concept in response["concepts"]:
        for key in concept["member_feature_ids"]:
            if key in by_feature:
                raise ValueError(f"Duplicate membership: {key}")
            by_feature[key] = concept
        for row in concept["representatives"]:
            representatives[row["feature_id"]] = row
    return by_feature, representatives


def main():
    freeze = json.loads((HERE / "selection_frozen.json").read_text())
    for filename, expected in freeze["files_sha256"].items():
        actual = hashlib.sha256((HERE / filename).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Frozen file changed: {filename}")
    (HERE / "comparison_started.json").write_text(json.dumps({
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "selection_frozen_utc": freeze["frozen_utc"],
        "freeze_sha256": hashlib.sha256((HERE / "selection_frozen.json").read_bytes()).hexdigest(),
    }, indent=2) + "\n")
    payload = json.loads((HERE / "blinded_input.json").read_text())["payload"]
    sol = json.loads((HERE / "concept_response.json").read_text())
    gemma = json.loads((PREVIOUS / "concept_response.json").read_text())
    sol_by_feature, sol_selected = index_review(sol)
    gemma_by_feature, gemma_selected = index_review(gemma)
    candidates = {row["feature_id"]: row for row in payload["candidates"]}
    if set(candidates) != set(sol_by_feature) or set(candidates) != set(gemma_by_feature):
        raise ValueError("Both responses must cover the same candidate union")
    rows = []
    for key, candidate in candidates.items():
        ranks = candidate["top100_ranks_by_inner_fold"]
        rows.append({"feature_id": key, "name": candidate["definition"]["name"],
                     "ranks": ranks, "folds_in_top100": sum(rank is not None for rank in ranks),
                     "sol_selected": key in sol_selected, "gemma_selected": key in gemma_selected,
                     "sol_concept": sol_by_feature[key]["name"], "sol_decision": sol_by_feature[key]["decision"],
                     "gemma_concept": gemma_by_feature[key]["name"], "gemma_decision": gemma_by_feature[key]["decision"]})
    with (HERE / "candidate_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    s, g = set(sol_selected), set(gemma_selected)
    def label(keys):
        return [{"feature_id": key, "name": candidates[key]["definition"]["name"]} for key in sorted(keys)]
    comparison = {"model": freeze["model"], "candidate_union_size": len(candidates),
                  "sol_concepts": len(sol["concepts"]), "gemma_concepts": len(gemma["concepts"]),
                  "sol_decisions": dict(Counter(row["decision"] for row in sol["concepts"])),
                  "gemma_decisions": dict(Counter(row["decision"] for row in gemma["concepts"])),
                  "sol_selected": len(s), "gemma_selected": len(g),
                  "intersection": len(s & g), "union": len(s | g),
                  "jaccard": len(s & g) / len(s | g) if s | g else 1.0,
                  "shared": label(s & g), "sol_only": label(s - g), "gemma_only": label(g - s),
                  "sol_largest_groups": [{"name": row["name"], "decision": row["decision"], "size": len(row["member_feature_ids"])} for row in sorted(sol["concepts"], key=lambda row: len(row["member_feature_ids"]), reverse=True)[:10]],
                  "gemma_largest_groups": [{"name": row["name"], "decision": row["decision"], "size": len(row["member_feature_ids"])} for row in sorted(gemma["concepts"], key=lambda row: len(row["member_feature_ids"]), reverse=True)[:10]],
                  "sol_representative_recurrence": dict(Counter(sum(rank is not None for rank in candidates[key]["top100_ranks_by_inner_fold"]) for key in s)),
                  "gemma_representative_recurrence": dict(Counter(sum(rank is not None for rank in candidates[key]["top100_ranks_by_inner_fold"]) for key in g))}
    (HERE / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
