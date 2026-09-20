"""Post-hoc, read-only trace of saved Stage 2 feature-selection decisions.

This script does not refit models, request LLM output, or use patient oracle
values. Mappings were reviewed against the saved measurement definitions.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUN = ROOT / "artifacts/research_all_evidence/five_conf_five_mod_nsclc_full/stage2"
STEM = "five_conf_five_mod_selection_trace_2026-09-19"

NAMES = ["age", "sex", "ecog_performance_status", "creatinine_clearance", "prior_platinum_therapy",
         "histology_type", "egfr_mutation_status", "baseline_nlr", "brain_metastases_status", "baseline_hemoglobin"]
# Primary feature IDs in NAMES order. Broadening/coarsening is separately noted;
# related biomarkers or lesion counts are not counted as the original modifier.
PRIMARY = {
    1: [9, 302, 109, 95, 267, 192, 111, 228, 52, 155],
    2: [11, 29, 87, 77, 229, 163, 89, 193, 40, 127],
    3: [11, 295, 101, 109, 261, 57, 103, 225, 49, 147],
    4: [15, 362, 312, 102, 330, 239, 125, 282, 47, 175],
    5: [12, 34, 224, 101, 232, 160, 96, 194, 46, 127],
}
EXTRA_ALIASES = {1: {"sex": [140], "egfr_mutation_status": [32]},
                 2: {"sex": [116]}, 3: {"ecog_performance_status": [128, 256]}}
MAPPING_NOTES = {
    (3, "creatinine_clearance"): "Consolidation combined CrCl with eGFR in one measurement; this is a broadened renal-function feature, not an isolated CrCl measurement.",
    (5, "creatinine_clearance"): "Consolidation combined CrCl with eGFR in one measurement; this is a broadened renal-function feature, not an isolated CrCl measurement.",
    (2, "creatinine_clearance"): "Ontology supervision changed continuous clearance to ordinal kidney-function stages.",
    (4, "prior_platinum_therapy"): "Prior chemotherapy regimen is a broader proxy for platinum history and does not reproduce the oracle treatment-line categories.",
}
READ = {}
MANIFEST = {}


def read(path):
    if path not in READ:
        raw = path.read_bytes()
        MANIFEST[str(path.relative_to(ROOT))] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        READ[path] = json.loads(raw)
    return READ[path]


def compact_definition(feature):
    return {k: feature.get(k) for k in ["feature_id", "name", "description", "value_type", "categories_or_unit", "measurement_definition", "missing_value_rule", "harmonization_plan"]}


def main():
    # Freeze the analyzed saved stages before deriving oracle-role summaries.
    for fold in range(1, 6):
        p = RUN / f"outer_{fold:03d}"
        for rel in ["interpreted_candidates.json", "feature_definitions.json", "selection/input.json",
                    "selection/statistical_evidence.json", "selection/elastic_net_selection.json",
                    "selection/role_adjudication/advisory/report.json", "final_definitions.json",
                    "estimation/diagnostics.json"]:
            read(p / rel)
    config = read(RUN / "config.json")
    completion = read(RUN / "complete.json")
    oracle = read(ROOT / "synthetic_data/example_synthetic_datasets/five_confounders_five_effect_modifiers_nsclc_with_structured/metadata.json")
    oracle_by_name = {f["name"]: f for f in oracle["features"]}
    result = {"date": "2026-09-19", "run": str(RUN.relative_to(ROOT)),
              "completed_at": completion["completed_at"], "selection_mode": config["statistical_selection"]["selection_mode"],
              "optional_selection_consolidation_enabled": config["selection_consolidation"]["enabled"],
              "method": "Saved-artifact trace. X means an effect-prediction input; W means adjustment only. Excluded means absent from all final forest inputs. Direct modifier measurements are distinguished from related proxies. No model refits or patient-level accuracy assessment.",
              "folds": []}
    for fold in range(1, 6):
        p = RUN / f"outer_{fold:03d}"
        initial_doc = read(p / "feature_definitions.json")
        initial = {f["feature_id"]: f for f in initial_doc["features"]}
        inp = read(p / "selection/input.json")
        before = {f["feature_id"]: f for f in inp["definitions"]}
        final = {f["feature_id"]: f for f in read(p / "final_definitions.json")["features"]}
        stat = read(p / "selection/statistical_evidence.json")
        decisions = {f["feature_id"]: f for f in stat["decisions"]}
        advisory = read(p / "selection/role_adjudication/advisory/report.json")
        annotation = {f["feature_id"]: f for f in advisory["annotations"]}
        diagnostic = read(p / "estimation/diagnostics.json")
        candidates = {c["candidate_id"]: c for c in read(p / "interpreted_candidates.json")}
        assert set(initial) == set(before) == set(decisions)
        assert set(final) == {fid for fid, d in decisions.items() if d["retained"]}
        assert advisory["llm_may_change_selection"] is False
        assert all(final[fid]["forest_role"] == decisions[fid]["forest_role"] for fid in final)
        fresult = {
            "outer_fold": fold, "discovery_candidates": len(candidates),
            "consolidated_features": len(initial), "preselection_features": len(before), "final_features": len(final),
            "no_feature_ids_removed_before_statistical_selection": True,
            "advisory_status": advisory["status"], "advisory_can_change_selection": False,
            "forest_effect_features": diagnostic["effect_modifiers"], "forest_adjustment_features": diagnostic["pure_confounders_in_w"],
            "all_effect_input_names": [f["name"] for f in final.values() if f["forest_role"] == "X"],
            "forest_effect_fit_rows": diagnostic["effect_fit_rows"],
            "joint_inner_models": [{k: row.get(k) for k in ["inner_fold", "status", "solver_converged", "regularization_alpha", "heldout_r_loss_improvement"]} | {"selected_feature_count": len(row["selected_feature_ids"])} for row in stat["multivariable_modifier_elastic_net_screen"]["folds"]],
            "concepts": {},
        }
        for concept, number in zip(NAMES, PRIMARY[fold]):
            primary_id = f"outer_{fold:03d}_feature_{number:03d}"
            ids = [primary_id] + [f"outer_{fold:03d}_feature_{n:03d}" for n in EXTRA_ALIASES.get(fold, {}).get(concept, [])]
            assert all(fid in before for fid in ids)
            alias_records = []
            for fid in ids:
                feature = before[fid]
                origins = [cid for cid, disposition in initial_doc["candidate_dispositions"].items() if disposition.get("feature_name") == initial[fid]["name"]]
                assert origins, (fold, fid, "missing candidate lineage")
                tests = [{"inner_fold": row["inner_fold"], **{k: test.get(k) for k in ["status", "reason", "rank", "selected_top_n", "heldout_r_loss_improvement", "tested_interaction_columns"]}} for row in stat["effect_modifier_screen"]["folds"] for test in row["tests"] if test["feature_id"] == fid]
                alias_records.append({
                    "feature_id": fid, "name": feature["name"], "initial_definition": compact_definition(initial[fid]),
                    "screened_definition": compact_definition(feature), "numerical_decision": decisions[fid],
                    "candidate_wise_top_n_votes": stat["effect_modifier_screen"]["votes"][fid],
                    "candidate_wise_tests": tests, "advisory_annotation": annotation.get(fid),
                    "discovery_origins": [{"candidate_id": cid, "name": candidates[cid]["name"], "description": candidates[cid]["description"], "architecture": candidates[cid]["architecture"], "disposition": initial_doc["candidate_dispositions"][cid]} for cid in origins],
                    "joint_by_inner_fold": [{"inner_fold": row["inner_fold"], "selected": fid in row["selected_feature_ids"]} for row in stat["multivariable_modifier_elastic_net_screen"]["folds"]],
                })
            routes = {decisions[fid]["forest_role"] for fid in ids}
            route = "X" if "X" in routes else "W" if "W" in routes else "excluded"
            tasks = sorted({task for fid in ids for task in decisions[fid]["modeling_tasks"]})
            fresult["concepts"][concept] = {
                "oracle_roles": oracle_by_name[concept]["roles"], "oracle_categories": oracle_by_name[concept].get("categories"),
                "primary_feature_id": primary_id, "mapped_features": alias_records,
                "forest_route": route, "modeling_tasks": tasks,
                "mapping_note": MAPPING_NOTES.get((fold, concept)),
                "hard_exclusion_boundary": "statistical_task_selection" if route == "excluded" else None,
                "effect_input_exclusion_boundary": "joint_group_elastic_net_effect_selection" if "effect_modifier" in oracle_by_name[concept]["roles"] and route != "X" else None,
            }
        result["folds"].append(fresult)
    result["modifier_route_counts"] = dict(Counter(c["forest_route"] for f in result["folds"] for c in f["concepts"].values() if "effect_modifier" in c["oracle_roles"]))
    result["confounder_route_counts"] = dict(Counter(c["forest_route"] for f in result["folds"] for c in f["concepts"].values() if "confounder" in c["oracle_roles"]))
    result["joint_inner_models_selecting_no_features"] = sum(m["selected_feature_count"] == 0 for f in result["folds"] for m in f["joint_inner_models"])
    result["source_files"] = MANIFEST
    target = OUT / (STEM + ".json")
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    for name in NAMES:
        print(name, [f["concepts"][name]["forest_route"] for f in result["folds"]], flush=True)
    print("MODIFIERS", result["modifier_route_counts"], "CONFOUNDERS", result["confounder_route_counts"], flush=True)
    print("EMPTY_JOINT_MODELS", result["joint_inner_models_selecting_no_features"], flush=True)
    print(target, flush=True)


if __name__ == "__main__":
    main()
