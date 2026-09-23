"""Report a completed frozen review, using the existing oracle mapping only post hoc."""

import datetime as dt
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "fold_1_cross_fold_modifier_concepts_2026-09-23"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    freeze = json.loads((HERE / "selection_frozen.json").read_text())
    for filename, expected in freeze["files_sha256"].items():
        assert hashlib.sha256((HERE / filename).read_bytes()).hexdigest() == expected, filename
    write_json(HERE / "oracle_audit_started.json", {
        "at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "frozen_at": freeze["frozen_utc"],
        "note": "Post-freeze comparison in the parent process. No oracle feedback is sent to the reviewer.",
    })
    previous_audit = json.loads((PREVIOUS / "oracle_recovery.json").read_text())
    comparison = json.loads((HERE / "comparison.json").read_text())
    response = json.loads((HERE / "concept_response.json").read_text())
    selected = json.loads((HERE / "selected_modifiers.json").read_text())
    selected_ids = {row["feature_id"] for row in selected}
    candidates = {row["feature_id"]: row for row in json.loads((HERE / "blinded_input.json").read_text())["payload"]["candidates"]}
    concept_by_feature = {key: concept for concept in response["concepts"] for key in concept["member_feature_ids"]}
    previous_definitions = json.loads((PREVIOUS / "selected_definitions.json").read_text())["features"]
    confounders = {row["feature_id"] for row in previous_definitions if "confounder" in row["roles"]}
    assert len(confounders) == 189
    oracle = []
    for original in previous_audit["direct"]:
        keys = set(original["candidate_ids"])
        oracle.append({"oracle_variable": original["oracle_variable"], "true_role": original["true_role"],
                       "candidate_ids": original["candidate_ids"],
                       "selected_as_modifier": sorted(keys & selected_ids),
                       "gemma_selected_as_modifier": original["selected_as_modifier"],
                       "retained_as_confounder": sorted(keys & confounders),
                       "containing_concepts": [{"feature_id": key, "name": concept_by_feature[key]["name"], "decision": concept_by_feature[key]["decision"]} for key in sorted(keys & set(concept_by_feature))]})
    direct_modifiers = sum(bool(row["selected_as_modifier"]) for row in oracle if row["true_role"] == "effect_modifier")
    direct_confounders = sum(bool(row["retained_as_confounder"]) for row in oracle if row["true_role"] == "confounder")
    audit = {"at": dt.datetime.now(dt.timezone.utc).isoformat(), "mapping_sha256": previous_audit["mapping_sha256"],
             "mapping_source": str(PREVIOUS / "oracle_recovery.json"), "direct": oracle,
             "direct_oracle_modifiers_recovered": direct_modifiers, "oracle_modifier_total": 5,
             "direct_oracle_confounders_preserved": direct_confounders, "oracle_confounder_total": 5,
             "confounder_count": len(confounders), "modifier_count": len(selected_ids),
             "note": "Direct recovery uses the unchanged pre-existing identity mapping; proxies are not counted as direct recovery. No estimator was fit."}
    write_json(HERE / "oracle_recovery.json", audit)
    name = lambda key: candidates[key]["definition"]["name"]
    rows = []
    for concept in response["concepts"]:
        if concept["decision"] == "retain":
            reps = "; ".join(name(row["feature_id"]) for row in concept["representatives"])
            recurrence = "; ".join(f"{sum(rank is not None for rank in candidates[row['feature_id']]['top100_ranks_by_inner_fold'])}/5" for row in concept["representatives"])
            rows.append(f"| {concept['name']} | {reps} | {recurrence} |")
    union_q = [value for candidate in candidates.values() for value in candidate["effect_evidence_by_method"]["univariable"].get("q", []) if value is not None]
    usage = [json.loads(line)["usage"] for line in (HERE / "review_events.jsonl").read_text().splitlines() if json.loads(line).get("type") == "turn.completed"][-1]
    audit_checks = {"frozen_hashes_verified": True, "minimum_supplied_per_fold_median_univariable_q": min(union_q),
                    "first_response_passed_validator": True, "repair_calls": 0, "tool_events": freeze["tool_events"],
                    "usage": usage, "candidate_coverage": 217, "representatives_in_at_least_four_top100_lists": len(selected_ids),
                    "new_candidates_recurrence": {name(key): candidates[key]["top100_ranks_by_inner_fold"] for key in sorted(selected_ids - {row["feature_id"] for row in comparison["shared"]})}}
    write_json(HERE / "assessment_checks.json", audit_checks)
    report = """# Blinded GPT-6 Sol versus Gemma modifier review — 2026-09-23

**GPT-6 Sol produced a smaller, more granular review, but exact oracle-modifier recovery did not improve: 1/5, NLR, for both models.** Sol selected 14 measurements from 11 retained concepts, compared with Gemma's 21 measurements from 21 retained concepts. Nine measurements overlap. The reviewer had no oracle or earlier-review access.

1. **Model-access problem resolved**
   - The installed CLI was 0.154.0. The requested GPT-6 Sol failed there even with a fresh model catalog.
   - A temporary official CLI 0.156.1 succeeded on the same account. A paired retest of the old client still failed. This identifies a client-version compatibility/catalog problem; the original error overstated it as an account-support problem.
   - The actual review used **GPT-6 Sol**, high reasoning. GPT-5.6 Sol was not used. The globally installed CLI was not changed.
   - See [access diagnosis](MODEL_ACCESS_DIAGNOSIS_2026-09-23.md) for the tests and official release-note link.

2. **What the blinded reviewer received**
   - Identical synthesis instructions and identical numerical evidence to the Gemma review: 217 distinct candidates from five inner-fold top-100 lists, their definitions, fold ranks, and five families of modifier evidence from all five splits.
   - No oracle identities or values, previous review, previous selected list, parent conversation, patient rows, or held-out outcomes.
   - A separate filesystem namespace excluded the repository, source datasets, reports, and previous Codex sessions. There were zero tool calls in the review event log.
   - The first response assigned all 217 candidates exactly once and passed the same structural validator. No repair requests were needed. Selection and provenance were hashed before the post hoc comparison.
   - Codex's agent wrapper and enforced JSON response schema differ from Gemma's direct service call. This is one practical reviewer comparison, not a controlled comparison of model weights alone.

3. **Selection comparison**

   | Measure | Gemma 4 31B | GPT-6 Sol |
   |---|---:|---:|
   | Candidates reviewed | 217 | 217 |
   | Total concept groups | 30 | 58 |
   | Retained concepts | 21 | 11 |
   | Uncertain concepts | 3 | 14 |
   | Excluded concepts | 6 | 33 |
   | Selected modifier measurements | 21 | 14 |
   | Largest concept group | 99 | 11 |
   | Selected measurements appearing in at least four top-100 lists | 13/21 | 14/14 |
   | Exact oracle modifiers recovered | 1/5 | 1/5 |

   - Shared selections: NLR, white blood cell count, dyspnea severity, TTF-1, KRAS G12C variant allele frequency, MET amplification, Ki-67, prior platinum exposure, and liver metastasis presence.
   - Added by Sol: **creatinine clearance, BUN, CK5/6, emphysema, and dizziness**.
   - Omitted by Sol relative to Gemma: serum creatinine, confusion, epidural hemorrhage, glucose, hematocrit, broad lung-disease diagnosis, lymph-node dissection, pneumonia, pulmonary nodules, radiation-session count, skeletal compression fracture, and targeted-therapy use.
   - The original 189 confounders remain the fixed downstream adjustment set; this review only proposes modifier changes. All five mapped oracle confounders remain included.

4. **The 14 selected measurements**

   | Retained concept | Existing representative measurement(s) | Top-100 recurrence, matching order |
   |---|---|---|
"""
    report += "\n".join("   " + row for row in rows)
    report += """

5. **What happened to each oracle modifier**

   | Oracle variable | Directly selected by Sol? | Sol decision and proxy interpretation |
   |---|---|---|
   | NLR | Yes | Retained directly in the leukocyte-balance concept. |
   | Histology | No | The broader histology/lineage concept is retained, represented by CK5/6 and TTF-1. These are related lineage measurements, not direct recovery of the histology variable. Gemma already retained TTF-1 separately. |
   | EGFR status | No | Excluded in the EGFR/ALK/RET alteration group. |
   | Brain metastasis presence | No | Intracranial lesion burden is uncertain; no representative is selected. |
   | Hemoglobin | No | The broader CBC/toxicity group is uncertain; neither hemoglobin nor hematocrit is selected. Gemma had retained hematocrit as an anemia proxy. |

   - All five oracle modifiers were represented in the supplied candidate union. Sol's review did not recover the four missing direct measurements.
   - True confounders creatinine clearance and prior platinum exposure also appear in Sol's modifier list. Direct oracle-role recovery is a benchmark, not proof that every other selected variable is useless: proxies and baseline-risk variables can carry probability-scale treatment-effect information.

6. **What improved, and what remains limited**
   - **More granular grouping:** the largest group shrank from 99 candidates to 11. Sol separates many meaningful facets and explains why related measurements are not interchangeable. Some residual groups remain broad, including other CBC abnormalities/toxicities and other neuroimaging findings.
   - **Better handling of concrete recurrence examples:** CK5/6 ranks 29, 9, 7, 28, and 4, and BUN ranks 9, 98, 44, 7, and 8. Sol recognizes both as recurrent and retains them; Gemma's exclusion reasoning had understated their recurrence.
   - **Measurement definitions receive attention:** Sol explicitly identifies the misleading FGFR-named feature that actually specifies GFR, distinguishes PRO-CTCAE dyspnea from broader symptom coding, and distinguishes KRAS-specific from generic variant allele frequency.
   - **More explicit uncertainty:** brain-lesion and CBC concepts are left uncertain rather than claimed as established modifiers. Sol discusses testing/documentation patterns, timing ambiguity, and disagreement between model families.
   - **The underlying evidence is still weak:** the minimum supplied per-fold median univariable BH q-value across the candidate union is 0.30. This describes the supplied summaries, not every individual resample. Repeated selection and forest permutation scores do not establish a causal effect, and overlapping folds are not independent replications.
   - **No demonstrated sensitivity improvement:** exact recovery remains 1/5. The added histology proxy is counterbalanced by removal of the anemia proxy; there is no basis to claim overall proxy coverage or causal-estimation performance improved.
   - **No new outcome/effect model was fitted.** Correlation, ITE RMSE, and held-out R-loss for this 14-modifier set remain unmeasured. This experiment establishes a different, more carefully organized proposal, not better CATE estimation.

7. **Saved artifacts**
   - [Selected modifiers](selected_modifiers.csv), [all candidate comparisons](candidate_comparison.csv), and [complete Sol concept decisions and rationales](concept_response.json).
   - [Oracle audit](oracle_recovery.json), [comparison metrics](comparison.json), and [audit checks](assessment_checks.json).
   - [Frozen selection manifest](selection_frozen.json), [blinded input](blinded_input.json), [protocol](PROTOCOL_2026-09-23.md), and [model-access diagnosis](MODEL_ACCESS_DIAGNOSIS_2026-09-23.md).
"""
    (HERE / "REPORT_2026-09-23.md").write_text(report)
    outputs = ["REPORT_2026-09-23.md", "MODEL_ACCESS_DIAGNOSIS_2026-09-23.md", "comparison.json", "oracle_recovery.json", "assessment_checks.json", "candidate_comparison.csv", "selection_frozen.json"]
    complete = {"completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "state": "complete",
                "actual_model": "gpt-6-sol", "cli_version": "0.156.1", "fallback_used": False,
                "selected_modifiers": len(selected_ids), "oracle_modifiers_recovered": direct_modifiers,
                "new_estimator_fit": False, "files_sha256": {filename: hashlib.sha256((HERE / filename).read_bytes()).hexdigest() for filename in outputs}}
    write_json(HERE / "experiment_complete.json", complete)
    write_json(HERE / "status.json", {key: value for key, value in complete.items() if key != "files_sha256"})
    print(json.dumps({key: value for key, value in complete.items() if key != "files_sha256"}, indent=2))


if __name__ == "__main__":
    main()
