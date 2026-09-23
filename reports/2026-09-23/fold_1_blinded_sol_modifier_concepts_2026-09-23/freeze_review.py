"""Validate and freeze the blinded review before any comparison with oracle labels."""

import csv
import datetime as dt
import hashlib
import json
from collections import Counter
from pathlib import Path

from validate_response import validator

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if (HERE / "selection_frozen.json").exists():
        raise SystemExit("Selection is already frozen; refusing to overwrite it.")
    packet = json.loads((HERE / "blinded_input.json").read_text())
    payload = packet["payload"]
    index = {record["evidence_id"]: candidate["feature_id"]
             for candidate in payload["candidates"]
             for record in candidate["effect_evidence_by_method"].values()}
    raw = json.loads((HERE / "review_response.txt").read_text())
    response = validator(payload, index)(raw)
    events = [json.loads(line) for line in (HERE / "review_events.jsonl").read_text().splitlines() if line.strip()]
    item_types = Counter(event.get("item", {}).get("type") for event in events if "item" in event)
    tool_events = [event for event in events if event.get("item", {}).get("type") not in
                   {None, "agent_message", "reasoning", "error", "plan"}]
    if tool_events:
        raise ValueError(f"Unexpected tool activity requires audit before freezing: {dict(item_types)}")
    (HERE / "concept_response.json").write_text(json.dumps(response, indent=2) + "\n")
    candidates = {candidate["feature_id"]: candidate for candidate in payload["candidates"]}
    selected = []
    for concept in response["concepts"]:
        for representative in concept["representatives"]:
            key = representative["feature_id"]
            candidate = candidates[key]
            ranks = candidate["top100_ranks_by_inner_fold"]
            selected.append({"feature_id": key, "name": candidate["definition"]["name"],
                             "concept_id": concept["concept_id"], "concept_name": concept["name"],
                             "folds_in_top100": sum(rank is not None for rank in ranks),
                             "ranks": ranks, "reason": representative["reason"]})
    (HERE / "selected_modifiers.json").write_text(json.dumps(selected, indent=2) + "\n")
    with (HERE / "selected_modifiers.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["feature_id", "name", "concept_id", "concept_name", "folds_in_top100", "ranks", "reason"])
        writer.writeheader()
        writer.writerows(selected)
    frozen_files = ["blinded_input.json", "review_prompt.txt", "response_schema.json", "input_manifest.json",
                    "isolation_probe.json", "run_review.py", "freeze_review.py", "validate_response.py",
                    "review_launch.json", "review_complete.json", "review_response.txt", "review_events.jsonl",
                    "review_stderr.log", "concept_response.json", "selected_modifiers.json", "selected_modifiers.csv"]
    freeze = {"frozen_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "model": json.loads((HERE / "review_launch.json").read_text())["model"], "reasoning_effort": "high",
              "candidates": len(candidates), "concepts": len(response["concepts"]),
              "decisions": dict(Counter(row["decision"] for row in response["concepts"])),
              "selected_modifiers": len(selected), "event_item_types": dict(item_types),
              "tool_events": len(tool_events), "oracle_or_previous_review_supplied": False,
              "files_sha256": {name: sha(HERE / name) for name in frozen_files}}
    (HERE / "selection_frozen.json").write_text(json.dumps(freeze, indent=2) + "\n")
    print(json.dumps({key: value for key, value in freeze.items() if key != "files_sha256"}, indent=2))


if __name__ == "__main__":
    main()
