"""Check the review artifacts and assemble their readable index; no model calls."""
from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
SPECS = runpy.run_path(str(OUT / "build_replacements.py"))["SPECS"]
LABELS = {"Serum creatinine", "Emphysema on imaging"}
FOLLOWUPS = {6, 10, 14, 15, 16, 17, 18, 19, 22, 23}
FINAL_RESPONSES = {5, 6, 10, 14, 15, 16, 17, 18, 19, 22, 23}


def exact_keys(value, keys):
    assert isinstance(value, dict), value
    assert set(value) == set(keys), (set(value), set(keys))


def nonempty(value):
    assert isinstance(value, str) and value.strip(), value


def check_response(number, value):
    if number == 2:
        exact_keys(value, {"candidates"})
        assert len(value["candidates"]) == 2
        for row in value["candidates"]:
            exact_keys(row, {"name", "description", "basis", "uncertainty"})
            for key in ("name", "description", "basis"):
                nonempty(row[key])
            assert isinstance(row["uncertainty"], str)
    elif number == 3:
        exact_keys(value, {"merges"})
        assert len(value["merges"]) == 1
        group = value["merges"][0]
        exact_keys(group, {"members", "canonical_label"})
        assert set(group["members"]) == {"Creatinine level", "Serum creatinine"}
        nonempty(group["canonical_label"])
    elif number == 4:
        exact_keys(value, {"description", "value_type", "categories_or_unit", "measurement_definition", "missing_value_rule", "conflict_resolution", "caveats"})
        assert value["value_type"] == "continuous"
        assert value["categories_or_unit"] == ["mg/dL"]
        assert value["conflict_resolution"] == {"strategy": "latest", "positive_category": None}
    elif number in {5, 22, 23}:
        assert value == {"Serum creatinine": 1.2, "Emphysema on imaging": "Present"}
    elif number == 6:
        exact_keys(value, {"values", "decision_notes"})
        assert value["values"] == {"Serum creatinine": 1.2, "Emphysema on imaging": "Present"}
        exact_keys(value["decision_notes"], LABELS)
        for note in value["decision_notes"].values():
            assert note is None or isinstance(note, str) and len(note) <= 2048
    elif number == 7:
        exact_keys(value, {"observations"})
        assert len(value["observations"]) == 3
        source = "2025-01-01: serum creatinine 1.0 mg/dL. 2025-01-10: serum creatinine 1.2 mg/dL. CT documents emphysema."
        observed = set()
        for row in value["observations"]:
            exact_keys(row, {"feature", "value", "quote", "governing_date_quote"})
            assert row["quote"] in source
            assert row["governing_date_quote"] is None or row["governing_date_quote"] in source
            observed.add((row["feature"], row["value"], row["governing_date_quote"]))
        assert observed == {("Serum creatinine", 1.0, "2025-01-01"), ("Serum creatinine", 1.2, "2025-01-10"), ("Emphysema on imaging", "Present", None)}
    elif number in {8, 11}:
        assert value == {"value": "Present" if number == 8 else "Below 1.0"}
    elif number in {9, 12}:
        exact_keys(value, {"action", "reason"})
        assert value["action"] == "keep"
        nonempty(value["reason"])
    elif number == 10:
        exact_keys(value, {"status", "representation", "reason", "token_interpretations"})
        assert value["status"] == "ready" and value["representation"] == "categorical"
        nonempty(value["reason"])
        tokens = {}
        for row in value["token_interpretations"]:
            exact_keys(row, {"raw_text", "meaning", "interpretation"})
            nonempty(row["interpretation"])
            assert row["raw_text"] not in tokens
            tokens[row["raw_text"]] = row["meaning"]
        assert tokens == {"<1.0": "explicit_interval", "high": "unusable"}
    elif number == 13:
        exact_keys(value, {"action", "reason", "members", "canonical_label", "category_equivalences"})
        assert value["action"] == "merge"
        assert set(value["members"]) == {"Creatinine level", "Serum creatinine"}
        assert value["category_equivalences"] == []
        nonempty(value["reason"])
    elif number in {14, 17}:
        exact_keys(value, {"confounder", "effect_modifier"})
        for row in value.values():
            exact_keys(row, {"assign", "assessment", "stability", "rationale", "evidence_comments"})
            assert isinstance(row["assign"], bool)
            assert row["assessment"] in {"supported", "plausible", "uncertain", "not_supported"}
            assert row["stability"] in {"consistent", "mixed", "insufficient"}
            assert not row["assign"] or row["assessment"] in {"supported", "plausible"}
            nonempty(row["rationale"])
            assert isinstance(row["evidence_comments"], list)
            for comment in row["evidence_comments"]:
                nonempty(comment)
        # Record decisions for inspection; do not require agreement with a
        # preferred causal judgment as the definition of prompt comprehension.
    elif number == 15:
        exact_keys(value, {"themes"})
        members = []
        for row in value["themes"]:
            exact_keys(row, {"name", "members", "interpretation", "disagreements"})
            nonempty(row["name"])
            nonempty(row["interpretation"])
            members += row["members"]
        assert set(members) == LABELS and len(members) == len(LABELS)
    elif number == 16:
        assert value == {"merges": []}
    elif number == 18:
        exact_keys(value, {"ordered_groups"})
        members = []
        for row in value["ordered_groups"]:
            exact_keys(row, {"features", "rationale"})
            nonempty(row["rationale"])
            members += row["features"]
        assert set(members) == LABELS and len(members) == len(LABELS)
    elif number == 19:
        exact_keys(value, {"preferred_feature", "rationale"})
        assert value["preferred_feature"] is None or value["preferred_feature"] in LABELS
        nonempty(value["rationale"])
    elif number == 20:
        exact_keys(value, {"interpretation", "limitations"})
        for text in value.values():
            nonempty(text)
    elif number == 21:
        exact_keys(value, {"concepts"})
        members = []
        for row in value["concepts"]:
            exact_keys(row, {"name", "members", "modifier_recommendation", "rationale", "representatives", "unresolved_questions"})
            members += row["members"]
            assert set(row["representatives"]) <= set(row["members"])
            assert row["modifier_recommendation"] in {"retain", "uncertain", "exclude"}
            if row["modifier_recommendation"] == "retain":
                assert row["representatives"]
            if row["modifier_recommendation"] == "exclude":
                assert row["representatives"] == []
            nonempty(row["rationale"])
        assert set(members) == LABELS and len(members) == len(LABELS)
    else:
        raise AssertionError(number)


def main():
    assert len(SPECS) == 22
    inventory = ROOT / "reports/2026-09-23/all_evidence_prompt_examples_2026-09-23/rendered"
    records = []
    combined = ["# All revised prompt proposals — September 23, 2026", "", "These are proposals 02–23, not production replacements. See REVIEW_REPORT_2026-09-23.md for review findings, implementation status, and policy tradeoffs.", ""]
    for spec in SPECS:
        slug = spec["slug"]
        number = int(slug[:2])
        prefix = f"{number:02d}"
        current_path = OUT / "revised" / f"{slug}.json"
        current = json.loads(current_path.read_text())
        assert current == spec["messages"], slug
        original = json.loads((OUT / "original" / f"{slug}.json").read_text())
        assert original == json.loads((inventory / f"{slug}.json").read_text())["messages"], slug
        required = [f"reviews/{prefix}_original.md", f"checks/{prefix}_revised.md", f"checks/{prefix}_response.json"]
        if number in FOLLOWUPS:
            required.append(f"checks/{prefix}_followup.md")
        if number == 5:
            required.append("checks/05_final_fresh.md")
        for rel in required:
            assert (OUT / rel).is_file(), rel
        response_file = f"checks/{prefix}_{'final_response' if number in FINAL_RESPONSES else 'response'}.json"
        response = json.loads((OUT / response_file).read_text())
        check_response(number, response)
        records.append({"prompt": slug, "status": "reviewed_proposal_not_integrated", "original_review": required[0], "fresh_replacement_check": required[1], "targeted_followup": f"checks/{prefix}_followup.md" if number in FOLLOWUPS else None, "extra_fresh_check": "checks/05_final_fresh.md" if number == 5 else None, "verified_example_response": response_file, "sha256": hashlib.sha256(current_path.read_bytes()).hexdigest()})
        combined += [(OUT / "revised" / f"{slug}.md").read_text(), "\n---\n"]

    adopted = runpy.run_path(str(ROOT / "oci/inference/stage2_discovery_prompt.py"))
    approved = (ROOT / "reports/2026-09-23/discovery_prompt_naive_review_2026-09-23/01_discover_system_v3.txt").read_text().strip()
    assert adopted["DISCOVERY_SYSTEM_PROMPT"].strip() == approved
    result = {"date": "2026-09-23", "review_model": "gpt-5.6-sol", "fresh_original_reviews": 22, "fresh_replacement_reviews": 22, "additional_fresh_final_reviews": 1, "targeted_followups": len(FOLLOWUPS), "final_example_responses_checked": 22, "discovery_adopted_verbatim": True, "discovery_version": adopted["DISCOVERY_PROMPT_VERSION"], "production_test_result": {"passed": 245, "warnings": 11, "command": "MPLCONFIGDIR=/tmp/stage2-prompt-review-mpl /home/klkehl/thisenv/bin/python -m pytest tests/test_plain_handoff_stage2.py tests/test_plain_handoff_stage2_evidence.py tests/test_stage2_runtime_regressions.py -q --disable-warnings --maxfail=3", "note": "Previously completed in this task; artifact verification does not rerun the tests."}, "limitations": "Invented miniature examples; schema and comprehension checks, not an extraction or causal-estimation benchmark. Semantic role/ranking judgments are recorded without requiring a favored answer.", "proposals": records}
    (OUT / "verification.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    (OUT / "ALL_REVISED_PROMPTS_2026-09-23.md").write_text("\n".join(combined))
    print("Verified 22 original/revised prompt pairs, final example responses, review coverage, and adopted discovery text.")


if __name__ == "__main__":
    main()
