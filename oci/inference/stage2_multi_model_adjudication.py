"""Bounded cross-candidate theme review followed by evidence-cited role decisions.

This interface accepts definitions and aggregate evidence only. Numerical cell
artifacts, row identifiers, predictions, error messages, and source paths are
not copied into a prompt. Every candidate survives theme summarization and gets
an explicit final decision; themes never merge or change its measurements.
"""

from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path

from .stage2_multi_model_config import FAMILIES, PROMPT_VERSION, SCHEMA_VERSION
from .stage2_role_adjudication import (
    _canonical_json,
    _feature_id,
    _fingerprint,
    _prompt_safe_definition,
    _role_response_validator,
    _selected_from_adjudication,
    _write_json,
)

from . import stage2_clinical_prompts as clinical_prompts
from .stage2_prompt_io import request_review

PROMPT_VERSION = clinical_prompts.PROMPT_VERSION
SYSTEM_PROMPT = clinical_prompts.SYSTEM_PROMPTS["17_model_roles"]


def _number(value, *, integer=False):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("multi-model prompt statistics must be finite numbers")
    if integer and (value < 0 or value != int(value)):
        raise ValueError("multi-model evidence counts must be nonnegative integers")
    return int(value) if integer else float(value)


def build_multi_model_role_evidence(*, definitions, statistical_report, policy):
    """An explicit allowlist; never pass through arbitrary report fields."""
    source = statistical_report.get("multi_model_evidence") or {}
    cards = []
    ids = [_feature_id(f) for f in definitions]
    if len(set(ids)) != len(ids) or set(source) != set(ids):
        raise ValueError("multi-model summaries must cover every candidate exactly once")
    for feature in definitions:
        feature_id = _feature_id(feature)
        rows, seen = [], set()
        for raw in source[feature_id]:
            family, role = raw.get("family"), raw.get("role")
            if (
                family not in FAMILIES
                or role not in {"treatment", "outcome", "effect"}
                or (family, role) in seen
            ):
                raise ValueError("unknown or duplicate multi-model evidence family/role")
            seen.add((family, role))
            row = {
                "evidence_id": f"multi:{feature_id}:{family}:{role}",
                "family": family,
                "role": role,
            }
            for key in ("exposures", "evaluated", "not_evaluable", "supported", "q_supported"):
                row[key] = _number(raw.get(key), integer=True)
            if any(
                row[k] is None for k in ("exposures", "evaluated", "not_evaluable", "supported")
            ):
                raise ValueError("multi-model evidence is missing counts")
            if not 0 <= row["supported"] <= row["evaluated"] <= row["exposures"]:
                raise ValueError("inconsistent multi-model support counts")
            if row["not_evaluable"] != row["exposures"] - row["evaluated"]:
                raise ValueError("inconsistent multi-model evaluability counts")
            for key in (
                "support_fraction",
                "mean_score",
                "min_score",
                "max_score",
                "median_p",
                "median_q",
                "mean_model_gain",
                "mean_split_importance",
                "mean_permutation_sd",
            ):
                row[key] = _number(raw.get(key))
            row["folds"] = [
                {
                    k: _number(fold.get(k), integer=True)
                    for k in ("inner_fold", "evaluated", "supported")
                }
                for fold in raw.get("folds", [])
            ]
            rows.append(row)
        cards.append(
            {
                "feature_id": feature_id,
                "definition": _prompt_safe_definition(feature, policy=policy),
                "modeling_evidence": rows,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "study_context": clinical_prompts.study_context(statistical_report),
        "candidates": cards,
        "analysis_populations": {
            "modifier_min_propensity": _number(
                (statistical_report.get("policy") or {}).get("min_propensity")
            ),
            "modifier_max_propensity": _number(
                (statistical_report.get("policy") or {}).get("max_propensity")
            ),
            "bounds_are_inclusive": True,
            "null_bound_means_unrestricted": True,
            "modifier_evidence": "propensity_eligible_patients_using_training_only_nuisances",
            "association_evidence": "all_sampled_patients_except_joint_interaction_model_main_effects",
        },
        "evidence_boundary": {
            "aggregate_outer_training_evidence_only": True,
            "row_level_values_are_excluded": True,
            "patient_identifiers_are_excluded": True,
            "outer_heldout_rows_are_excluded": True,
            "oracle_columns_are_excluded": True,
            "dataset_paths_and_dataset_names_are_excluded": True,
        },
        "score_meaning": {
            "univariable": "minus_log10_p; support uses nominal_p; q_support recorded separately",
            "penalized_main": "nonzero_group_norm_for_prediction",
            "penalized_interactions": "nonzero_group_norm_in_joint_outcome_interaction_model",
            "orthogonal_linear": "nonzero_group_norm_in_joint_R_loss_model",
            "univariable_rlearner": "heldout_R_loss_gain_over_constant_effect",
            "predictive_forest": "heldout_prediction_loss_increase_after_group_permutation",
            "causal_forest": "heldout_R_loss_increase_after_group_permutation",
        },
        "stability_contract": {
            "missing_is_not_negative": True,
            "no_single_method_gate": True,
            "overlapping_repeats_are_not_independent": True,
            "themes_do_not_merge_measurements": True,
        },
    }


def _text(value, limit=4000, *, empty=False):
    if not isinstance(value, str) or (not empty and not value.strip()) or len(value) > limit:
        raise ValueError(f"expected text no longer than {limit} characters")
    return value.strip()


def _themes_validator(member_ids, evidence_ids, *, maximum, cards=()):
    member_ids, evidence_ids = set(member_ids), set(evidence_ids)
    labels = {clinical_prompts.label(c["definition"]): c["feature_id"] for c in cards}
    by_id = {c["feature_id"]: c for c in cards}

    def validate(response):
        themes = response.get("themes")
        if not isinstance(themes, list) or not 1 <= len(themes) <= maximum:
            raise ValueError(f"theme review requires 1 through {maximum} themes")
        cleaned, covered = [], set()
        for theme in themes:
            if "members" in theme:
                members = [clinical_prompts.resolve_label(x, labels) for x in theme["members"]]
                cites = [r["evidence_id"] for key in members for r in by_id[key]["modeling_evidence"]]
            else:
                members, cites = theme.get("member_feature_ids"), theme.get("evidence_ids")
            if (not isinstance(members, list) or not members or len(set(members)) != len(members)
                    or not set(members) <= member_ids or covered & set(members)):
                raise ValueError("each clinical variable must appear in exactly one theme")
            if not isinstance(cites, list) or not set(cites) <= evidence_ids:
                raise ValueError("theme provenance contains unknown evidence")
            covered.update(members)
            cleaned.append({"name": _text(theme.get("name"), 300), "member_feature_ids": members,
                "evidence_ids": list(dict.fromkeys(cites)), "interpretation": _text(theme.get("interpretation")),
                "disagreements": _text(theme.get("disagreements"), empty=True),
                "evidence_attachment": "Available member evidence attached by Python."})
        if covered != member_ids:
            raise ValueError("include every supplied variable in a theme")
        return {"themes": cleaned}
    return validate


def _bounded_batches(items, payload_builder, *, maximum_items, maximum_chars):
    batches, current = [], []
    def size(batch):
        payload = payload_builder(batch)
        return sum(len(m["content"]) for m in payload) if isinstance(payload, list) else len(_canonical_json(payload)) + len(SYSTEM_PROMPT)
    for item in items:
        if current and (len(current) >= maximum_items or size([*current, item]) > maximum_chars):
            batches.append(current)
            current = []
        current.append(item)
        if size(current) > maximum_chars:
            raise ValueError("one clinical review item exceeds max_prompt_chars")
    if current:
        batches.append(current)
    return batches


def _decision_validator(definitions, cards):
    base = _role_response_validator(definitions=definitions)
    available = {c["feature_id"]: c["modeling_evidence"] for c in cards}
    def validate(response):
        if set(response) == {"confounder", "effect_modifier"} and len(definitions) == 1:
            feature = definitions[0]
            response = {"summary": "Clinical role review", "decisions": [clinical_prompts.role_decision(
                response, feature, modeling_evidence=available[_feature_id(feature)])]}
        for raw in response.get("decisions", []):
            if "evidence_ids" in raw:
                allowed = {x["evidence_id"] for x in available.get(raw.get("feature_id"), [])}
                if not isinstance(raw["evidence_ids"], list) or not set(raw["evidence_ids"]) <= allowed:
                    raise ValueError("role provenance must use this variable's supplied evidence")
        cleaned = base(response)
        for row in cleaned["decisions"]:
            # These references describe the input, not model-authored citations.
            row["evidence_ids"] = [x["evidence_id"] for x in available[row["feature_id"]]]
            row["evidence_attachment"] = "Available input evidence attached by Python; consult role_assessments for the model's explanation."
        return cleaned
    return validate


def theme_text(themes, by_id):
    return "\n\n".join(t["name"] + "\nVariables: "
        + "; ".join(clinical_prompts.label(by_id[key]["definition"]) for key in t["member_feature_ids"])
        + "\nInterpretation: " + t["interpretation"] + "\nDisagreements: " + (t["disagreements"] or "None stated.") for t in themes)


def _merge_validator(themes):
    by_name = {t["name"]: t for t in themes}
    if len(by_name) != len(themes):
        raise ValueError("theme names must be distinct before requesting a merge")
    def validate(response):
        if "themes" in response:  # already normalized checkpoint
            candidates = {x for t in themes for x in t["member_feature_ids"]}
            refs = {x for t in themes for x in t["evidence_ids"]}
            return _themes_validator(candidates, refs, maximum=len(themes))(response)
        if set(response) != {"merges"} or not isinstance(response["merges"], list):
            raise ValueError("return merges, an array of overlapping clinical themes")
        used, result = set(), []
        for group in response["merges"]:
            sources = group.get("source_themes")
            if (not isinstance(sources, list) or len(sources) < 2 or len(set(sources)) != len(sources)
                    or not set(sources) <= set(by_name) or used & set(sources)):
                raise ValueError("a merge requires at least two distinct supplied themes, each used once")
            used.update(sources)
            result.append({"name": _text(group.get("name"), 300),
                "member_feature_ids": [key for source in sources for key in by_name[source]["member_feature_ids"]],
                "evidence_ids": list(dict.fromkeys(key for source in sources for key in by_name[source]["evidence_ids"])),
                "interpretation": _text(group.get("interpretation")),
                "disagreements": _text(group.get("disagreements"), empty=True)})
        result.extend(t for t in themes if t["name"] not in used)
        return {"themes": result}
    return validate


def adjudicate_multi_model_roles(*, definitions, statistical_report, request_json, output_dir, policy):
    policy.validate()
    if not policy.enabled:
        raise ValueError("multi_model requires LLM role adjudication")
    evidence = build_multi_model_role_evidence(definitions=definitions, statistical_report=statistical_report, policy=policy)
    maximum_chars = int(statistical_report["policy"]["multi_model"]["max_prompt_chars"])
    maximum_items = max(2, int(policy.max_candidates_per_request))
    directory = Path(output_dir)
    identity = {"evidence": _fingerprint(evidence), "policy": policy.public_dict(), "prompt": PROMPT_VERSION,
                "model": statistical_report.get("adjudication_model_identity"),
                "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest()}
    _write_json(directory / "evidence.json", evidence)
    cards = evidence["candidates"]
    by_id = {c["feature_id"]: c for c in cards}
    def request(name, messages, validate):
        return request_review(directory / name, messages, validate, request_json=request_json,
                              identity=identity, maximum_chars=maximum_chars)
    def theme_messages(batch):
        return clinical_prompts.messages("15_model_themes", clinical_prompts.evidence_input(evidence, batch))
    batches = _bounded_batches(cards, theme_messages, maximum_items=maximum_items, maximum_chars=maximum_chars)
    themes = []
    for index, batch in enumerate(batches):
        response = request(f"themes/initial_{index:03d}", theme_messages(batch),
            _themes_validator([c["feature_id"] for c in batch],
                {r["evidence_id"] for c in batch for r in c["modeling_evidence"]}, maximum=len(batch), cards=batch))
        themes.extend(response["themes"])
    # Summarize overlaps when useful. Context limits never force unrelated merges.
    level = 0
    while len(themes) > 1:
        counts = {name: sum(t["name"] == name for t in themes) for name in {t["name"] for t in themes}}
        themes = [{**t, "name": t["name"] + (" (" + clinical_prompts.label(by_id[t["member_feature_ids"][0]]["definition"]) + ")" if counts[t["name"]] > 1 else "")} for t in themes]
        def merge_messages(batch):
            return clinical_prompts.messages("16_merge_themes", theme_text(batch, by_id))
        batches = _bounded_batches(sorted(themes, key=lambda t: t["name"].casefold()), merge_messages,
            maximum_items=max(len(themes), maximum_items), maximum_chars=maximum_chars)
        reduced = []
        for index, batch in enumerate(batches):
            if len(batch) == 1:
                reduced.extend(batch)
            else:
                reduced.extend(request(f"themes/merge_{level:03d}_{index:03d}", merge_messages(batch), _merge_validator(batch))["themes"])
        changed = len(reduced) < len(themes)
        themes = reduced
        if not changed:
            break
        level += 1
    _write_json(directory / "themes.json", {"themes": themes, "all_candidate_ids_preserved": True})
    decisions = []
    definitions_by_id = {_feature_id(f): f for f in definitions}
    for index, card in enumerate(cards):
        relevant = [t for t in themes if card["feature_id"] in t["member_feature_ids"]]
        messages = clinical_prompts.messages("17_model_roles", clinical_prompts.evidence_input(evidence, [card])
            + "\n\nRelated clinical themes\n" + theme_text(relevant, by_id))
        response = request(f"roles/batch_{index:03d}", messages,
            _decision_validator([definitions_by_id[card["feature_id"]]], [card]))
        decisions.extend(response["decisions"])
    combined = _decision_validator(definitions, cards)({"summary": "Clinical role reviews", "decisions": decisions})
    selected = _selected_from_adjudication(definitions=definitions, adjudication=combined)
    for feature in selected:
        if feature.get("configured_explicit_feature") is not True:
            feature["selection_source"] = "multi_model_llm_adjudication"
    report = {"schema_version": SCHEMA_VERSION, "prompt_version": PROMPT_VERSION, "status": "complete",
              "evidence_fingerprint": identity["evidence"], "themes": themes, **combined,
              "retained_feature_ids": [_feature_id(f) for f in selected]}
    _write_json(directory / "response.json", report)
    _write_json(directory / "complete.json", {"status": "complete", "identity": identity,
        "candidate_count": len(definitions), "retained_count": len(selected)})
    return selected, report, evidence
