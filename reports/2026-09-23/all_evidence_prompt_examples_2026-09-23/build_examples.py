"""Render current prompt templates with invented inputs; never call an LLM.

Run with the repository Python environment. Examples are documentation, not
patient data, model responses, fitted statistics, or proposed prompt changes.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
from oci.inference import plain_handoff_stage2 as discovery
from oci.inference import plain_handoff_stage2_analysis as extraction
from oci.inference import stage2_sequential_consolidation as consolidation
from oci.inference import stage2_role_adjudication as roles
from oci.inference import stage2_multi_model_adjudication as multi
from oci.inference import stage2_modifier_ranking as ranking
from oci.inference import stage2_taskwise_annotation as annotation


def nested(module, parent_name, name, bindings):
    """Execute only a nested payload builder, not its enclosing LLM workflow."""
    path = Path(module.__file__)
    tree = ast.parse(path.read_text())
    parent = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == parent_name)
    node = next(n for n in ast.walk(parent) if isinstance(n, ast.FunctionDef) and n.name == name)
    scope = dict(vars(module))
    scope.update(bindings)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope[name], {"path": str(path.relative_to(ROOT)), "line": node.lineno, "function": f"{parent_name}.{name}"}


def source(function):
    return {"path": str(Path(inspect.getsourcefile(function)).relative_to(ROOT)),
            "line": inspect.getsourcelines(function)[1], "function": function.__name__}


def messages(system, payload):
    return [{"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, sort_keys=True, ensure_ascii=False)}]


entries = []


def save(slug, title, content, origins, activation, note=""):
    destination = OUT / "rendered"
    destination.mkdir(parents=True, exist_ok=True)
    sources = [source(o) if callable(o) else o for o in origins]
    for item in sources:
        item["sha256"] = hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest()
    metadata = {"id": slug, "title": title, "activation": activation, "sources": sources,
                "input_status": "Invented illustrative inputs, not observed study data or results.",
                "note": note, "messages": content}
    (destination / f"{slug}.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    lines = [f"# {title}", "", "**Unabridged current template, with invented miniature inputs. No LLM was called.**",
             "", f"Activation: {activation}", "", note, ""]
    for item in sources:
        lines += [f"Source: [{item['function']}]({ROOT / item['path']}:{item['line']})", ""]
    for message in content:
        lines += [f"## {message['role'].capitalize()} message", ""]
        try:
            body = json.dumps(json.loads(message["content"]), indent=2, ensure_ascii=False)
            language = "json"
        except json.JSONDecodeError:
            body, language = message["content"], "text"
        lines += [f"```{language}", body, "```", ""]
    (destination / f"{slug}.md").write_text("\n".join(lines))
    entries.append({k: v for k, v in metadata.items() if k != "messages"})


creatinine = {"feature_id": "example_creatinine", "name": "serum_creatinine",
              "description": "Serum creatinine concentration.", "value_type": "continuous",
              "categories_or_unit": ["mg/dL"],
              "measurement_definition": "Latest documented pretreatment serum creatinine, in mg/dL; preserve a reported threshold if no exact number exists.",
              "missing_value_rule": "Null if unreported or unresolved.",
              "conflict_resolution": {"strategy": "latest", "positive_category": None}, "roles": []}
alias = {**copy.deepcopy(creatinine), "feature_id": "example_creatinine_alias", "name": "creatinine_level"}
emphysema = {"feature_id": "example_emphysema", "name": "emphysema",
            "description": "Documented emphysema.", "value_type": "binary",
            "categories_or_unit": ["Present", "Absent"],
            "measurement_definition": "Explicit pretreatment documentation of emphysema presence or absence.",
            "missing_value_rule": "Null if unreported; silence is not absence.",
            "conflict_resolution": {"strategy": "latest", "positive_category": None}, "roles": []}
definitions = [creatinine, emphysema]
text = "2025-01-01 pretreatment lab: serum creatinine 1.0 mg/dL. 2025-01-10 pretreatment lab: serum creatinine 1.2 mg/dL. CT documents emphysema."
packets = [{"content": {"representative_evidence": [{"text": text}]}}]
save("01_discover", "01. Discover atomic clinical features", discovery._interpretation_prompt(architecture="not_exposed", packets=packets), [discovery._interpretation_prompt], "Core pathway")
save("02_audit_unmapped", "02. Audit evidence left uncited", discovery._rejected_packet_audit_prompt(architecture="not_exposed", packets=packets), [discovery._rejected_packet_audit_prompt], "Conditional: initial discovery leaves an evidence item uncited")
save("03_merge_aliases", "03. Consolidate candidate names before extraction", discovery._global_candidate_pool_prompt(groups=[creatinine, alias, emphysema]), [discovery._global_candidate_pool_prompt], "Core pathway; repeated bounded partitions")
save("04_define_ontology", "04. Define a measurement ontology", discovery._operationalization_prompt(feature_name=creatinine["name"], supporting_evidence=[text]), [discovery._operationalization_prompt], "Core pathway; per discovered feature")
save("05_extract_patient", "05. Extract one patient's values", extraction._extraction_prompt(definitions=definitions, rows=[{"row_id": 1, "text": text}]), [extraction._extraction_prompt], "Training and heldout extraction; same template")
chunk = "2025-01-10 pretreatment lab: serum creatinine 1.2 mg/dL. CT documents emphysema."
save("06_serial_chunk", "06. Update extraction with the next record chunk", extraction._serial_extraction_prompt(definitions=definitions, row_id=1, chunk_text=chunk, prior_values={"serum_creatinine": 1.0, "emphysema": None}, prior_feature_state={"serum_creatinine": "Latest date: 2025-01-01", "emphysema": None}, chunk_index=1, char_start=len(text)-len(chunk), char_end=len(text), document_chars=len(text)), [extraction._serial_extraction_prompt], "Conditional: serial long-record extraction")
save("07_page_observations", "07. Extract source-grounded page observations", extraction._page_extraction_prompt(definitions=definitions, row={"row_id": 1, "text": text}), [extraction._page_extraction_prompt], "Alternate oversized-record/page path in shared extraction code", "Later cross-page reconciliation applies the declared conflict rule in code; it is not a separate LLM prompt.")
mapping = {"mapping_id": "category_mapping_0000", "feature_name": "emphysema", "value_type": "binary", "description": emphysema["description"], "measurement_definition": emphysema["measurement_definition"], "missing_value_rule": emphysema["missing_value_rule"], "allowed_categories": ["Present", "Absent"], "prior_extracted_value": "present on CT", "occurrence_count": 3}
save("08_map_categories", "08. Normalize an out-of-ontology categorical value", extraction._category_ontology_prompt([mapping]), [extraction._category_ontology_prompt], "Conditional: extracted categorical values violate declared categories")
failures = [{"failure_kind": "invalid_category", "reason": "Repeated returned values are outside the declared categories.", "patient_count": 3, "example_values": ["present on CT", "positive"], "allowed_categories": ["Present", "Absent"]}]
save("09_refine_ontology", "09. Refine a schema after repeated extraction failures", extraction._ontology_refinement_prompt(feature=emphysema, failure_patterns=failures), [extraction._ontology_refinement_prompt], "Conditional: repeated feature-attributable training extraction failures")
observations = extraction._mixed_value_observations(pd.DataFrame({"serum_creatinine": [0.8, 1.2, "<1.0", "high"]}), creatinine)
save("10_harmonize_values", "10. Harmonize mixed numeric and text values", extraction._harmonization_prompt(feature=creatinine, observations=observations, prior_plan=None), [extraction._harmonization_prompt], "Conditional: mixed training representations")
prior = {"target_representation": "categorical", "reason": "Illustrative representation with one explicit numeric boundary.", "canonical_categories": ["Below 1.0", "At least 1.0"], "numeric_bin_rules": [{"lower_bound": None, "lower_inclusive": False, "upper_bound": 1.0, "upper_inclusive": False, "canonical_value": "Below 1.0"}, {"lower_bound": 1.0, "lower_inclusive": True, "upper_bound": None, "upper_inclusive": False, "canonical_value": "At least 1.0"}], "unmapped_value_rule": "null"}
save("11_extend_value_map", "11. Extend a frozen map for new training text values", extraction._harmonization_delta_prompt(feature=creatinine, prior_plan=prior, new_categorical_values=[{"raw_value": "less than 1.0", "count": 2}]), [extraction._harmonization_delta_prompt], "Conditional: new training tokens during incremental extraction", "The sample threshold is invented solely to demonstrate the template, not a clinical recommendation.")
summary = extraction.feature_summaries(pd.DataFrame({"emphysema": ["Present"]*5 + ["Absent"]*3 + [None]*2}), [emphysema])[0]
save("12_supervise_ontology", "12. Supervise extraction ontology using aggregate diagnostics", extraction._aggregate_ontology_supervisor_prompt(feature=emphysema, summary=summary, failure_patterns=failures), [extraction._aggregate_ontology_supervisor_prompt], "Core training extraction/supervision loop", "Aggregate diagnostic values here are a miniature invented fixture.")
frame = pd.DataFrame({"serum_creatinine": [0.8, 1.0, 1.2, 1.4], "creatinine_level": [0.8, 1.0, 1.2, 1.4]})
step = {"schema_version": "illustrative_step", "step": 1, "pivot_feature_id": creatinine["feature_id"], "active_candidate_count": 3, "neighbor_count": 1, "equivalence_policy": {"minimum_pairwise_association": consolidation.DEFAULT_MINIMUM_PAIRWISE_ASSOCIATION}, "features": [consolidation._prompt_feature(f, frame=frame, cosine_similarity=1.0 if i == 0 else 0.98, protected=False) for i, f in enumerate([creatinine, alias])], "pairwise_associations": [{"left_feature_id": creatinine["feature_id"], "right_feature_id": alias["feature_id"], "n_pairwise_complete": 4, "evaluable": True, "association_kind": "illustrative_numeric_association", "association": 1.0, "signed_association": 1.0, "missingness_absolute_phi": None, "missingness_jaccard": None, "details": {}}]}
save("13_post_extraction_aliases", "13. Check extracted measurements for lossless alias consolidation", consolidation._decision_messages(step, max_latents_per_cluster=2), [consolidation._decision_messages], "Optional: sequential_consolidation.enabled", "Default configuration class has enabled=False; some experiment/config files explicitly enable it. 'Latents' is a schema term here: the prompt forbids inventing broader concepts.")
policy = roles.Stage2RoleAdjudicationConfig()
statistical = {"nuisance_screen": {"folds": [{"inner_fold": 1, "treatment": {"selected_feature_ids": ["example_creatinine"], "feature_group_l2_norms": {"example_creatinine": 0.12}}, "outcome": {"selected_feature_ids": ["example_creatinine"], "feature_group_l2_norms": {"example_creatinine": 0.15}}}]}}
role_evidence = roles.build_stage2_role_evidence(definitions=definitions, statistical_report=statistical, policy=policy)
role_payload = roles._role_request_payload(evidence=role_evidence, batch_index=1, batch_count=1)
save("14_default_roles", "14. Adjudicate roles in the llm_roles mode", messages(roles.ROLE_ADJUDICATION_SYSTEM_PROMPT, role_payload), [roles._role_request_payload, roles.build_stage2_role_evidence], "Alternative selection branch: llm_roles", "Unsupplied statistical results are empty/null in this small fixture, not negative study results.")
def evidence_row(family, role, supported, score):
    return {"family": family, "role": role, "exposures": 3, "evaluated": 3, "not_evaluable": 0, "supported": supported, "support_fraction": supported/3, "mean_score": score, "folds": [{"inner_fold": i+1, "evaluated": 1, "supported": int(i < supported)} for i in range(3)]}
model_report = {"policy": {"min_propensity": 0.1, "max_propensity": 0.9}, "multi_model_evidence": {
    "example_creatinine": [evidence_row("penalized_main", "treatment", 2, 0.12), evidence_row("penalized_main", "outcome", 3, 0.15), evidence_row("causal_forest", "effect", 1, 0.001)],
    "example_emphysema": [evidence_row("univariable", "effect", 2, 1.8), evidence_row("univariable_rlearner", "effect", 1, 0.002), evidence_row("causal_forest", "effect", 1, 0.001)]}}
evidence = multi.build_multi_model_role_evidence(definitions=definitions, statistical_report=model_report, policy=policy)
cards = evidence["candidates"]
themes = [{"name": "Renal function", "member_feature_ids": ["example_creatinine"], "evidence_ids": [cards[0]["modeling_evidence"][0]["evidence_id"]], "interpretation": "Illustrative association evidence.", "disagreements": "Little effect evidence in this invented example."}, {"name": "Pulmonary disease", "member_feature_ids": ["example_emphysema"], "evidence_ids": [cards[1]["modeling_evidence"][0]["evidence_id"]], "interpretation": "Illustrative heterogeneous effect evidence.", "disagreements": "Methods disagree in this invented example."}]
theme_builder, theme_source = nested(multi, "adjudicate_multi_model_roles", "theme_payload", {"evidence": evidence})
merge_builder, merge_source = nested(multi, "adjudicate_multi_model_roles", "merge_payload", {"theme_payload": theme_builder})
role_builder, role_source = nested(multi, "adjudicate_multi_model_roles", "role_payload", {"evidence": evidence, "themes": themes})
save("15_model_themes", "15. Review multi-model evidence for themes", messages(multi.SYSTEM_PROMPT, theme_builder(cards)), [theme_source], "Multi-model selection branch")
save("16_merge_themes", "16. Merge theme summaries across batches", messages(multi.SYSTEM_PROMPT, merge_builder(themes)), [merge_source], "Conditional: multi-model themes exceed the request batch limit")
save("17_model_roles", "17. Assign roles using multi-model evidence and themes", messages(multi.SYSTEM_PROMPT, role_builder(cards)), [role_source], "Multi-model selection branch")
rank_builder, rank_source = nested(ranking, "rank_modifier_candidates", "payload", {"evidence": evidence})
save("18_rank_modifiers", "18. Rank modifier candidates", messages(ranking.SYSTEM_PROMPT, rank_builder(cards)), [rank_source], "Multi-model branch with cross-validated modifier-count selection")
save("19_merge_rankings", "19. Interleave ordered modifier lists", messages(ranking.SYSTEM_PROMPT, rank_builder(cards, [["example_creatinine"], ["example_emphysema"]])), [rank_source], "Conditional: ranking spans multiple batches")
advisory = copy.deepcopy(role_payload)
advisory.update({"task": "annotate_stage2_roles_only", "prompt_version": annotation.SCHEMA_VERSION})
advisory["decision_policy"].update({"annotation_only": True, "may_change_numerical_selection": False})
annotation_path = Path(annotation.__file__)
annotation_line = next(i for i, line in enumerate(annotation_path.read_text().splitlines(), 1) if '"task": "annotate_stage2_roles_only"' in line)
save("20_advisory_roles", "20. Annotate numerically selected roles without changing selection", messages(annotation.SYSTEM_PROMPT, advisory), [roles._role_request_payload, {"path": str(annotation_path.relative_to(ROOT)), "line": annotation_line, "function": "annotation-only payload modification"}], "Alternative selection branch: independent_tasks, with LLM annotation enabled")
original = ROOT / "reports/2026-09-23/fold_1_blinded_sol_modifier_concepts_2026-09-23/blinded_input.json"
experiment = json.loads(original.read_text())
# Preserve instructions/schema/design metadata; replace all candidate data.
concept_payload = copy.deepcopy(experiment["payload"])
concept_payload["candidate_union_size"] = len(cards)
concept_payload["candidates"] = [{
    "feature_id": c["feature_id"], "definition": c["definition"],
    "top100_ranks_by_inner_fold": [20, 30, None, 25, 40],
    "effect_evidence_by_method": {"causal_forest": {
        "evidence_id": f"crossfold:{c['feature_id']}:causal_forest",
        "n": [9, 9, 9, 9, 9], "s": [5, 4, 3, 6, 4],
        "score": [0.001, 0.0005, -0.0002, 0.0015, 0.0004]}}
} for c in cards]
save("21_cross_fold_concepts", "21. Experimental global cross-fold concept review", messages(experiment["system"], concept_payload), [{"path": str(original.relative_to(ROOT)), "line": 1, "function": "saved experiment system and user payload"}], "Experimental report-level pass; not integrated into the production pathway", "Exact system, instructions, response schema, and design metadata from the saved experiment; invented two-candidate payload. The actual 217-candidate request remains in blinded_input.json.")
save("22_validation_repair", "22. Repair a response using the exact validation error", [discovery._repair_message(ValueError("missing required feature serum_creatinine in row 1"))], [discovery._repair_message], "Shared transport: schema/validation failure", "Appended to the original task context and available failed-response feedback; this is not a standalone extraction prompt.")
save("23_length_repair", "23. Repair an overlong response", [discovery._repair_message(discovery._Stage2OutputLengthError("response reached the configured completion-token limit"))], [discovery._repair_message], "Shared transport: completion length exceeded", "Preserve required records and fields; shorten redundant content. Example error text is invented.")

manifest = {"date": "2026-09-23", "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "scope": "Current prompt templates plus explicitly labeled optional, alternative, and experimental variants", "no_model_calls": True, "examples": entries}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
index = ["# Full rendered prompt examples — September 23, 2026", "",
         "Current instructions and response schemas with invented miniature inputs. No model calls were made.", "",
         f"[Readable guide]({OUT / 'PROMPT_EXAMPLES_2026-09-23.md'})", ""]
for entry in entries:
    path = OUT / "rendered" / f"{entry['id']}.md"
    index += [f"- [{entry['title']}]({path}) — {entry['activation']}"]
index += ["", "Each rendered Markdown file has a matching `.json` file containing the exact message strings and source metadata.", ""]
(OUT / "RENDERED_INDEX_2026-09-23.md").write_text("\n".join(index))
print(f"Rendered {len(entries)} examples into {OUT / 'rendered'}")
