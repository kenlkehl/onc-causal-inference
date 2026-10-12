"""Bounded cross-candidate theme review followed by evidence-cited role decisions.

This interface accepts definitions and aggregate evidence only. Numerical cell
artifacts, row identifiers, predictions, error messages, and source paths are
not copied into a prompt. Every candidate survives theme summarization and gets
an explicit final decision; themes never merge or change its measurements.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
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
# This predecessor has identical evidence, prompts, and accepted normalized
# responses. Revalidate its completed reviews without repeating successful LLM
# work when installing the stricter diagnostics and theme recovery below.
_COMPATIBLE_SOURCE_HASHES = (
    "505082edb7e50381746a08fcc9c6cca01a206afdb6f79695bf8750c86b4117ac",
)


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
            if family == "matched_batch_contrast":
                batch = raw.get("batch_diagnostics") or {}
                row["batch_diagnostics"] = {
                    k: _number(batch.get(k), integer=True) for k in (
                        "filtered_evaluated", "unfiltered_evaluated", "filter_agreement_evaluated", "filter_agreement_count")}
                row["batch_diagnostics"].update({k: _number(batch.get(k)) for k in (
                    "mean_filtered_gain", "mean_unfiltered_gain", "mean_training_patients", "mean_validation_patients",
                    "mean_training_batches", "mean_validation_batches", "mean_training_proposed", "mean_training_imbalance_rejected",
                    "mean_validation_proposed", "mean_validation_imbalance_rejected")})
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
            "orthogonal_linear": "nonzero_feature_group_norm; binary DINA log odds; continuous R-loss",
            "univariable_rlearner": "heldout_DINA_likelihood_gain_binary_or_R_loss_gain_continuous",
            "predictive_forest": "heldout_prediction_loss_increase_after_group_permutation",
            "dina": "heldout_DINA_likelihood_loss_increase_after_whole_feature_group_permutation",
            "causal_forest": "heldout_R_loss_increase_after_group_permutation",
            "matched_batch_contrast": "relative_heldout_batch_deviation_MSE_gain_over_training_fitted_bin_intercepts",
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
    if cards and (len(labels) != len(cards) or set(by_id) != member_ids):
        raise ValueError("theme inputs require distinct clinical labels and complete candidate cards")
    display = {key: clinical_prompts.label(c["definition"]) for key, c in by_id.items()}

    def names(keys):
        return json.dumps([display.get(key, key) for key in sorted(keys)], ensure_ascii=False)

    roster = ("Copy each complete variable name exactly once, including all colon-separated "
        "components. Do not shorten names or combine distinct measurements. "
        "Required variable names: " + names(member_ids))

    def validate(response):
        if not isinstance(response, Mapping):
            raise ValueError("theme review requires an object containing themes")
        themes = response.get("themes")
        if not isinstance(themes, list) or not 1 <= len(themes) <= maximum:
            raise ValueError(f"theme review requires 1 through {maximum} themes")
        cleaned, counts = [], Counter()
        for theme in themes:
            if not isinstance(theme, Mapping):
                raise ValueError("each theme must be an object")
            if "members" in theme:
                if not isinstance(theme["members"], list) or not theme["members"]:
                    raise ValueError("theme members must be a nonempty list of variable names")
                try:
                    members = [clinical_prompts.resolve_label(x, labels) for x in theme["members"]]
                except ValueError as exc:
                    raise ValueError(f"{exc}. {roster}") from exc
                cites = [r["evidence_id"] for key in members for r in by_id[key]["modeling_evidence"]]
            else:
                members, cites = theme.get("member_feature_ids"), theme.get("evidence_ids")
            if (not isinstance(members, list) or not members
                    or any(not isinstance(key, str) for key in members)):
                raise ValueError("theme members must be a nonempty list of variable names")
            if not set(members) <= member_ids:
                raise ValueError("unknown clinical variables: " + names(set(members) - member_ids)
                    + ". " + roster)
            if (not isinstance(cites, list) or any(not isinstance(key, str) for key in cites)
                    or not set(cites) <= evidence_ids):
                raise ValueError("theme provenance contains unknown evidence")
            counts.update(members)
            cleaned.append({"name": _text(theme.get("name"), 300), "member_feature_ids": members,
                "evidence_ids": list(dict.fromkeys(cites)), "interpretation": _text(theme.get("interpretation")),
                "disagreements": _text(theme.get("disagreements"), empty=True),
                "evidence_attachment": "Available member evidence attached by Python."})
        missing = member_ids - counts.keys()
        duplicates = {key for key, count in counts.items() if count > 1}
        if missing or duplicates:
            raise ValueError("include every supplied variable in exactly one theme; "
                f"missing variables: {names(missing)}; repeated variables: {names(duplicates)}. " + roster)
        return {"themes": cleaned}

    validate.repair_instructions = roster
    if cards:
        validate.response_format = {"type": "json_schema", "json_schema": {
            "name": "clinical_themes", "strict": True, "schema": {
                "type": "object", "additionalProperties": False, "required": ["themes"],
                "properties": {"themes": {"type": "array", "minItems": 1, "maxItems": maximum,
                    "items": {"type": "object", "additionalProperties": False,
                        "required": ["name", "members", "interpretation", "disagreements"],
                        "properties": {
                            "name": {"type": "string", "minLength": 1, "maxLength": 300},
                            "members": {"type": "array", "minItems": 1, "maxItems": len(cards),
                                "items": {"type": "string", "enum": list(labels)}},
                            "interpretation": {"type": "string", "minLength": 1, "maxLength": 4000},
                            "disagreements": {"type": "string", "maxLength": 4000},
                        }}}}}}}
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


def adjudicate_multi_model_roles(*, definitions, statistical_report, request_json, output_dir, policy, workers=1):
    from .stage2_parallel import ordered_map
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
    compatible_identities = [{**identity, "source_sha256": value} for value in _COMPATIBLE_SOURCE_HASHES]
    _write_json(directory / "evidence.json", evidence)
    cards = evidence["candidates"]
    by_id = {c["feature_id"]: c for c in cards}
    theme_recoveries = {}

    def request(name, messages, validate, *, recover=None, strategy=None):
        # Only exhausted response validation is recoverable here. Transport,
        # checkpoint-integrity, and programming errors must still propagate.
        from .plain_handoff_stage2_analysis import Stage2ResponseValidationError
        recovery_path = directory / name / "recovery.json"
        recovery_key = _fingerprint({"messages": messages, "identity": identity})
        saved = json.loads(recovery_path.read_text()) if recover and recovery_path.exists() else {}
        if saved.get("input_fingerprint") == recovery_key:
            theme_recoveries[name] = saved
            theme_recoveries.update(saved.get("child_recoveries", {}))

        def request_or_recover(messages, validate, **kwargs):
            record = theme_recoveries.get(name)
            if record is None:
                try:
                    return request_json(messages, validate, **kwargs)
                except Stage2ResponseValidationError as exc:
                    if recover is None:
                        raise
                    record = {"input_fingerprint": recovery_key, "strategy": strategy,
                        "error": str(exc), "status": "started"}
                    _write_json(recovery_path, record)
            response = validate(recover())
            record = {**record, "status": "complete", "child_recoveries": {
                key: value for key, value in theme_recoveries.copy().items()
                if key.startswith(name + "/")}}
            _write_json(recovery_path, record)
            theme_recoveries[name] = record
            return response

        return request_review(directory / name, messages, validate, request_json=request_or_recover,
            identity=identity, compatible_identities=compatible_identities, maximum_chars=maximum_chars)
    def theme_messages(batch):
        return clinical_prompts.messages("15_model_themes", clinical_prompts.evidence_input(evidence, batch))
    batches = _bounded_batches(cards, theme_messages, maximum_items=maximum_items, maximum_chars=maximum_chars)
    def review_theme_batch(name, batch):
        def recover():
            if len(batch) == 1:
                # Grouping provides context, not a role decision. Keep a neutral
                # singleton and let the subsequent role review assess its evidence.
                card = batch[0]
                return {"themes": [{"name": clinical_prompts.label(card["definition"])[:300],
                    "member_feature_ids": [card["feature_id"]],
                    "evidence_ids": [r["evidence_id"] for r in card["modeling_evidence"]],
                    "interpretation": "Theme review was unavailable. Assess this variable using its supplied evidence.",
                    "disagreements": "No theme-level assessment is available."}]}
            middle = len(batch) // 2
            return {"themes": [*review_theme_batch(name + "/split_0", batch[:middle]),
                *review_theme_batch(name + "/split_1", batch[middle:])]}

        response = request(name, theme_messages(batch),
            _themes_validator([c["feature_id"] for c in batch],
                {r["evidence_id"] for c in batch for r in c["modeling_evidence"]}, maximum=len(batch), cards=batch),
            recover=recover, strategy="split_batch" if len(batch) > 1 else "neutral_singleton")
        return response["themes"]

    def initial_themes(item):
        index, batch = item
        return review_theme_batch(f"themes/initial_{index:03d}", batch)
    themes = [theme for group in ordered_map(initial_themes, enumerate(batches), workers=workers,
              thread_name="stage2-role-themes") for theme in group]
    # Summarize overlaps when useful. Context limits never force unrelated merges.
    level = 0
    while len(themes) > 1:
        counts = {name: sum(t["name"] == name for t in themes) for name in {t["name"] for t in themes}}
        themes = [{**t, "name": t["name"] + (" (" + clinical_prompts.label(by_id[t["member_feature_ids"][0]]["definition"]) + ")" if counts[t["name"]] > 1 else "")} for t in themes]
        def merge_messages(batch):
            return clinical_prompts.messages("16_merge_themes", theme_text(batch, by_id))
        batches = _bounded_batches(sorted(themes, key=lambda t: t["name"].casefold()), merge_messages,
            maximum_items=max(len(themes), maximum_items), maximum_chars=maximum_chars)
        def merge(item):
            index, batch = item
            if len(batch) == 1:
                return batch
            return request(f"themes/merge_{level:03d}_{index:03d}", merge_messages(batch), _merge_validator(batch),
                recover=lambda: {"themes": batch}, strategy="retain_unmerged_themes")["themes"]
        reduced = [theme for group in ordered_map(merge, enumerate(batches), workers=workers,
                   thread_name="stage2-role-theme-merge") for theme in group]
        changed = len(reduced) < len(themes)
        themes = reduced
        if not changed:
            break
        level += 1
    _write_json(directory / "themes.json", {"themes": themes, "all_candidate_ids_preserved": True})
    definitions_by_id = {_feature_id(f): f for f in definitions}
    def review(item):
        index, card = item
        relevant = [t for t in themes if card["feature_id"] in t["member_feature_ids"]]
        messages = clinical_prompts.messages("17_model_roles", clinical_prompts.evidence_input(evidence, [card])
            + "\n\nRelated clinical themes\n" + theme_text(relevant, by_id))
        response = request(f"roles/batch_{index:03d}", messages,
            _decision_validator([definitions_by_id[card["feature_id"]]], [card]))
        return response["decisions"]
    decisions = [decision for group in ordered_map(review, enumerate(cards), workers=workers,
                 thread_name="stage2-role-review") for decision in group]
    combined = _decision_validator(definitions, cards)({"summary": "Clinical role reviews", "decisions": decisions})
    selected = _selected_from_adjudication(definitions=definitions, adjudication=combined)
    for feature in selected:
        if feature.get("configured_explicit_feature") is not True:
            feature["selection_source"] = "multi_model_llm_adjudication"
    report = {"schema_version": SCHEMA_VERSION, "prompt_version": PROMPT_VERSION, "status": "complete",
              "evidence_fingerprint": identity["evidence"], "themes": themes, **combined,
              "theme_recoveries": theme_recoveries,
              "retained_feature_ids": [_feature_id(f) for f in selected]}
    _write_json(directory / "response.json", report)
    _write_json(directory / "complete.json", {"status": "complete", "identity": identity,
        "candidate_count": len(definitions), "retained_count": len(selected)})
    return selected, report, evidence
