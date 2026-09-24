"""Python-side inputs for controlled orchestration-test responders.

Prompts are now prose, and IDs are deliberately absent. Tests that exercise
checkpointing, repair routing, or batching inspect spied caller arguments to
choose their simulated replies. This helper never changes an actual prompt.
Separate contract tests exercise the new model-facing JSON responses directly.
"""
from copy import deepcopy
from functools import wraps
import inspect
import json
import threading

_LOCAL = threading.local()
_SHARED = {}


def _records():
    if not hasattr(_LOCAL, "records"):
        _LOCAL.records = {}
    return _LOCAL.records


def prompt_inputs(messages):
    text = next(m["content"] for m in messages if m["role"] == "user")
    if text in _records():
        return deepcopy(_records()[text])
    if text in _SHARED:
        return deepcopy(_SHARED[text])
    return json.loads(text)


def install(monkeypatch):
    from oci.inference import plain_handoff_stage2 as workflow
    from oci.inference import plain_handoff_stage2_analysis as analysis
    from oci.inference import stage2_clinical_prompts as clinical
    from oci.inference import stage2_multi_model_adjudication as multi
    _LOCAL.records = {}
    _SHARED.clear()

    def spy(module, name, project):
        original = getattr(module, name)
        @wraps(original)
        def wrapped(*args, **kwargs):
            bound = inspect.signature(original).bind(*args, **kwargs)
            bound.apply_defaults()
            result = original(*args, **kwargs)
            metadata = deepcopy(project(bound.arguments))
            _records()[result[1]["content"]] = metadata
            _SHARED[result[1]["content"]] = metadata
            return result
        monkeypatch.setattr(module, name, wrapped)

    spy(workflow, "_global_candidate_pool_prompt", lambda a: {
        "job": "consolidate_stage2_candidate_pool",
        "features": [workflow._candidate_pool_feature_view(g) for g in sorted(a["groups"], key=workflow._candidate_group_sort_key)],
        "configured_feature_names": list(a["configured_feature_names"]), "batch_ordering": a["batch_ordering"]})
    spy(workflow, "_operationalization_prompt", lambda a: {
        "candidate_feature_name": a["feature_name"], "supporting_evidence": list(a["supporting_evidence"])})
    spy(workflow, "_rejected_packet_audit_prompt", lambda a: {
        "job": "audit_unmapped_text_evidence_for_missed_clinical_features",
        "evidence_items": [{"item": i, "text": workflow._readable_supporting_text([p])} for i, p in enumerate(a["packets"], 1)]})
    spy(analysis, "_extraction_prompt", lambda a: {"job": "extract_stage2_patient_variables",
        "features": analysis._prompt_feature_definitions(a["definitions"]), "patients": list(a["rows"])})
    spy(analysis, "_serial_extraction_prompt", lambda a: {"job": "update_stage2_patient_variables_serially",
        "features": analysis._prompt_feature_definitions(a["definitions"]),
        "patient": {"row_id": a["row_id"], "prior_extraction": a["prior_values"], "prior_feature_state": a["prior_feature_state"],
        "current_chunk": a["chunk_text"], "chunk": {k: a[k] for k in ("chunk_index", "char_start", "char_end", "document_chars")}}})
    spy(analysis, "_page_extraction_prompt", lambda a: {"job": "extract_stage2_patient_variable_observations",
        "features": analysis._page_prompt_feature_definitions(a["definitions"]), "patient": a["row"]})
    spy(analysis, "_category_ontology_prompt", lambda a: {"job": "map_extracted_values_to_declared_category_ontology", "items": list(a["items"])})
    spy(analysis, "_ontology_refinement_prompt", lambda a: {"job": "refine_stage2_feature_ontology_from_repeated_extraction_failures",
        "feature": a["feature"], "repeated_failure_patterns": a["failure_patterns"]})
    spy(analysis, "_aggregate_ontology_supervisor_prompt", lambda a: {"job": "review_stage2_small_model_extraction_ontology",
        "feature": a["feature"], "aggregate_extraction_summary": a["summary"], "aggregate_validation_failures": a["failure_patterns"]})
    spy(analysis, "_harmonization_prompt", lambda a: {"job": "harmonize_stage2_mixed_numeric_and_categorical_values",
        "feature": a["feature"], "observed_training_representations": a["observations"], "prior_plan_to_extend_or_replace": a["prior_plan"]})
    spy(analysis, "_harmonization_delta_prompt", lambda a: {"job": "extend_stage2_harmonization_map_for_new_text_values",
        "feature": a["feature"], "frozen_harmonization_plan": a["prior_plan"], "new_observed_training_text_values": a["new_categorical_values"]})

    # Modeling responders inspect the safe evidence object before its rendering.
    evidence_input = clinical.evidence_input
    @wraps(evidence_input)
    def record_evidence(evidence, cards):
        text = evidence_input(evidence, cards)
        _records()[text] = {"candidates": list(cards), "role_evidence": {**evidence, "candidates": list(cards)}}
        return text
    monkeypatch.setattr(clinical, "evidence_input", record_evidence)
    original_messages = clinical.messages
    tasks = {"14_default_roles": "adjudicate_stage2_roles_from_all_evidence", "15_model_themes": "review_stage2_multi_model_themes",
             "17_model_roles": "adjudicate_stage2_multi_model_roles", "18_rank_modifiers": "rank_stage2_modifiers",
             "19_merge_rankings": "merge_stage2_modifier_ranking", "20_advisory_roles": "annotate_stage2_roles_only"}
    @wraps(original_messages)
    def record_messages(slug, text):
        messages = original_messages(slug, text)
        if slug in tasks:
            core = text.split("\n\nRelated clinical themes\n")[0].split("\n\nRecorded decision\n")[0]
            data = deepcopy(_records().get(core, {}))
            data["task"] = tasks[slug]
            if slug == "17_model_roles":
                data["themes"] = text.split("Related clinical themes\n")[-1]
            _records()[messages[1]["content"]] = data
        return messages
    monkeypatch.setattr(clinical, "messages", record_messages)


def role_response(*, confounder=False, modifier=False):
    return {role: {"assign": assign, "assessment": "plausible" if assign else "not_supported",
        "stability": "insufficient", "rationale": "Exploratory support." if assign else "Insufficient supporting evidence.",
        "evidence_comments": ["Compare the supplied validation evidence."]}
        for role, assign in (("confounder", confounder), ("effect_modifier", modifier))}
