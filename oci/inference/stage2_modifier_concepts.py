"""Training-only clinical concept review of per-fold modifier shortlists.

The same procedure runs inside every architecture/count training split and on
full outer-training evidence. No dataset, patient values, oracle truth, or
outer-test information enters this interface.
"""
from collections import Counter
from copy import deepcopy
from pathlib import Path

from . import stage2_clinical_prompts as clinical_prompts
from .stage2_multi_model_adjudication import build_multi_model_role_evidence
from .stage2_multi_model_selection import aggregate_evidence
from .stage2_modifier_ranking import rank_modifier_candidates
from .stage2_prompt_io import request_review
from .stage2_role_adjudication import _fingerprint, _write_json

SCHEMA_VERSION = "stage2_modifier_concepts_v1_nested_clinical_review"


def _concept_validator(cards):
    by_id = {c["feature_id"]: c for c in cards}
    names = {clinical_prompts.label(c["definition"]): c["feature_id"] for c in cards}
    def validate(response):
        if set(response) != {"concepts"} or not isinstance(response["concepts"], list):
            raise ValueError("return concepts, an array of clinical groups")
        covered, concepts = set(), []
        for concept in response["concepts"]:
            normalized = "member_feature_ids" in concept
            members = concept.get("member_feature_ids" if normalized else "members")
            reps = concept.get("representative_feature_ids" if normalized else "representatives")
            if not isinstance(members, list) or not members or not isinstance(reps, list):
                raise ValueError("each concept requires members and representatives")
            if not normalized:
                members = [clinical_prompts.resolve_label(x, names) for x in members]
                reps = [clinical_prompts.resolve_label(x, names) for x in reps]
            if (len(set(members)) != len(members) or not set(members) <= set(by_id)
                    or covered & set(members) or len(set(reps)) != len(reps) or not set(reps) <= set(members)):
                raise ValueError("each candidate belongs to one concept; representatives must be distinct members")
            recommendation = concept.get("modifier_recommendation")
            if recommendation not in {"retain", "uncertain", "exclude"}:
                raise ValueError("recommend retain, uncertain, or exclude")
            if (recommendation == "retain" and not reps) or (recommendation == "exclude" and reps):
                raise ValueError("retained concepts require representatives; excluded concepts have none")
            for key in ("name", "rationale", "unresolved_questions"):
                if not isinstance(concept.get(key), str) or (key != "unresolved_questions" and not concept[key].strip()):
                    raise ValueError(f"concept {key} must be text")
            covered.update(members)
            concepts.append({"name": concept["name"], "member_feature_ids": members,
                "representative_feature_ids": reps, "modifier_recommendation": recommendation,
                "rationale": concept["rationale"], "unresolved_questions": concept["unresolved_questions"]})
        if covered != set(by_id):
            raise ValueError("include every supplied clinical variable in one concept")
        return {"concepts": concepts}
    return validate


def _concept_input(evidence, cards, recurrence, list_count):
    parts = [clinical_prompts.study_text(evidence),
        f"The candidates below appeared in at least one of {list_count} lists. Each list contains up to the requested number of candidates ranked by modifier evidence in one overlapping patient group.",
        "Recurrence describes selection in this dataset. Measured redundancy between candidates and uncertainty intervals for score differences are unavailable."]
    for card in cards:
        definition = card["definition"]
        parts.append(clinical_prompts.label(definition) + "\n" + str(definition.get("description") or "")
                     + "\nMeasurement: " + str(definition.get("measurement_definition") or "")
                     + "\nType and unit/categories: " + str(definition.get("value_type")) + "; "
                     + "; ".join(map(str, definition.get("categories_or_unit") or [])))
        parts.append(f"Appeared in {recurrence[card['feature_id']]} of {list_count} candidate lists.")
        for row in card["modeling_evidence"]:
            if row["role"] != "effect":
                continue
            text = f"- {clinical_prompts.METHODS[row['family']]}: {row['supported']}/{row['evaluated']} usable fits supported a modifier signal; {row['not_evaluable']} unavailable."
            if row.get("mean_score") is not None:
                text += f" Mean {clinical_prompts.SCORES[row['family']]}: {row['mean_score']:.5g}."
            if row.get("median_p") is not None:
                text += f" Median p-value {row['median_p']:.5g}."
            if row.get("median_q") is not None:
                text += f" Median adjusted q-value {row['median_q']:.5g}."
            parts.append(text)
    return "\n\n".join(parts)


def infer_modifier_concepts(*, definitions, statistical_report, request_json, output_dir,
    role_policy, top_n, maximum, maximum_chars, model_identity):
    directory = Path(output_dir)
    cells = statistical_report.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("concept review requires the numerical cells for each inner fold")
    ids = [str(f["feature_id"]) for f in definitions]
    rankings = []
    for fold in sorted({int(c["inner_fold"]) for c in cells}):
        fold_cells = [c for c in cells if int(c["inner_fold"]) == fold]
        fold_report = {"policy": statistical_report["policy"],
            "study_context": clinical_prompts.study_context(statistical_report),
            "multi_model_evidence": aggregate_evidence(fold_cells, ids)}
        ranked = rank_modifier_candidates(definitions=definitions, statistical_report=fold_report,
            request_json=request_json, output_dir=directory / f"fold_{fold:03d}" / "ranking",
            role_policy=role_policy, maximum=top_n, maximum_chars=maximum_chars, model_identity=model_identity)
        rankings.append({"inner_fold": fold, **ranked})
    recurrence = Counter(key for ranked in rankings for key in {r["feature_id"] for r in ranked["ranking"]})
    evidence = build_multi_model_role_evidence(definitions=definitions, statistical_report=statistical_report, policy=role_policy)
    cards = [c for c in evidence["candidates"] if c["feature_id"] in recurrence]
    identity = {"version": SCHEMA_VERSION, "evidence": _fingerprint(evidence),
                "rankings": _fingerprint(rankings), "top_n": top_n, "model": model_identity}
    if cards:
        response = request_review(directory / "review", clinical_prompts.messages("21_cross_fold_concepts",
            _concept_input(evidence, cards, recurrence, len(rankings))), _concept_validator(cards),
            request_json=request_json, identity=identity, maximum_chars=maximum_chars)
    else:
        response = {"concepts": []}
    representatives = {key for c in response["concepts"] if c["modifier_recommendation"] in {"retain", "uncertain"}
                       for key in c["representative_feature_ids"]}
    # A concept can propose uncertain representatives for validation. No new
    # measurements or rollups are constructed, and confounder roles are untouched.
    reduced = [f for f in definitions if str(f["feature_id"]) in representatives]
    reduced_report = {"policy": statistical_report["policy"],
        "study_context": clinical_prompts.study_context(statistical_report),
        "multi_model_evidence": {key: statistical_report["multi_model_evidence"][key] for key in representatives}}
    ordered = rank_modifier_candidates(definitions=reduced, statistical_report=reduced_report,
        request_json=request_json, output_dir=directory / "representative_ranking", role_policy=role_policy,
        maximum=maximum, maximum_chars=maximum_chars, model_identity=model_identity)
    result = {**ordered, "concept_review": {"schema_version": SCHEMA_VERSION, "identity": identity,
        "top_n_per_inner_fold": top_n, "candidate_union_count": len(cards),
        "recurrence": dict(recurrence), "concepts": response["concepts"],
        "representative_feature_ids": sorted(representatives), "inner_fold_rankings": rankings,
        "representative_policy": "retain and explicitly nominated uncertain representatives enter nested validation",
        "confounders_changed": False, "oracle_or_outer_test_information_used": False}}
    _write_json(directory / "concepts.json", result)
    return result
