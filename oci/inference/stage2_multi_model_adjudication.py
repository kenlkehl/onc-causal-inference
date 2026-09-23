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

SYSTEM_PROMPT = """You review pretreatment candidate measurements using modeling evidence
from one outer-training fold. First identify themes across candidates; then
reconcile their roles as confounder, effect_modifier, both, or neither.
All evidence is fallible. No individual family, p-value, or support frequency is
a hard selection gate. A credible complementary signal can justify retention
even when other methods shrink it away. Do not mistake correlated aliases or
proxies for multiple independent discoveries. Themes organize evidence; they
do not establish equivalence, merge measurements, or transfer a role to every
theme member. Preserve investigator-locked roles exactly.

Treatment prediction, outcome prognosis, and confounding are distinct. Discuss
whether a candidate could be a common cause rather than an instrument or only a
prognostic factor. Effect modification needs treatment-heterogeneity evidence;
outcome main-effect importance alone does not establish it. Univariable logistic
interactions are on the log-odds scale and unadjusted for other covariates.
Orthogonal linear models, candidate R-learners, and causal forests assess the
probability/outcome scale after elastic-net nuisance adjustment. Their targets
and biases differ. A model family is evidence, not an independent replication.
All modifier evidence uses the supplied propensity-eligible population. Treatment
and outcome association screens use all sampled training patients; main effects
from the joint interaction model instead share its restricted population.

Use exposure and evaluability denominators. Missing or nonconverged fits are not
negative votes. Repeated samples and folds overlap; support fractions are not
causal probabilities or formal stability-selection error guarantees. Raw and BH
p-values do not correct the upstream adaptive discovery process. Permutation
importance can be diluted by correlated alternatives and does not prove a
causal role. Compare fold consistency, methods, subsets, and conflicting facts.
Definitions and theme names cannot establish a role without the supplied
empirical evidence. Never invent an oracle, data-generating process, or hidden
truth. Return the requested JSON and cite only supplied evidence IDs.
""".strip()


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


def _text(value, limit):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"expected nonempty text no longer than {limit} characters")
    return value.strip()


def _themes_validator(member_ids, evidence_ids, *, maximum):
    member_ids, evidence_ids = set(member_ids), set(evidence_ids)

    def validate(response):
        themes = response.get("themes")
        if not isinstance(themes, list) or not 1 <= len(themes) <= maximum:
            raise ValueError(f"theme review requires 1 through {maximum} themes")
        cleaned, covered = [], set()
        for theme in themes:
            members = theme.get("member_feature_ids")
            cites = theme.get("evidence_ids")
            if (
                not isinstance(members, list)
                or not members
                or len(set(members)) != len(members)
                or not set(members) <= member_ids
            ):
                raise ValueError("theme contains unknown, duplicate, or missing member IDs")
            if not isinstance(cites, list) or len(cites) > 12 or not set(cites) <= evidence_ids:
                raise ValueError("theme cites unknown modeling evidence")
            covered.update(members)
            cleaned.append(
                {
                    "name": _text(theme.get("name"), 200),
                    "member_feature_ids": members,
                    "evidence_ids": list(dict.fromkeys(cites)),
                    "interpretation": _text(theme.get("interpretation"), 1600),
                    "disagreements": _text(theme.get("disagreements"), 1600),
                }
            )
        if covered != member_ids:
            raise ValueError("theme synthesis must preserve every supplied candidate")
        return {"themes": cleaned}

    return validate


def _bounded_batches(items, payload_builder, *, maximum_items, maximum_chars):
    batches, current = [], []
    for item in items:
        trial = [*current, item]
        too_large = (
            len(trial) > maximum_items
            or len(SYSTEM_PROMPT) + len(_canonical_json(payload_builder(trial))) > maximum_chars
        )
        if too_large and current:
            batches.append(current)
            current = [item]
        else:
            current = trial
        if len(SYSTEM_PROMPT) + len(_canonical_json(payload_builder(current))) > maximum_chars:
            raise ValueError(
                "one multi-model prompt item exceeds max_prompt_chars; increase the explicit budget"
            )
    if current:
        batches.append(current)
    return batches


def _request(directory, name, payload, validate, *, request_json, maximum_chars, identity):
    if len(SYSTEM_PROMPT) + len(_canonical_json(payload)) > maximum_chars:
        raise ValueError(
            "multi-model prompt exceeds max_prompt_chars; no candidate evidence was truncated"
        )
    fingerprint = _fingerprint(
        {
            "payload": payload,
            "system": SYSTEM_PROMPT,
            "identity": identity,
            "version": PROMPT_VERSION,
        }
    )
    directory = directory / name
    response_path, complete_path = directory / "response.json", directory / "complete.json"
    _write_json(
        directory / "prompt.json",
        {"system": SYSTEM_PROMPT, "payload": payload, "input_fingerprint": fingerprint},
    )
    if response_path.is_file() and complete_path.is_file():
        complete = json.loads(complete_path.read_text())
        response = json.loads(response_path.read_text())
        if complete.get("input_fingerprint") == fingerprint:
            if complete.get("response_sha256") != _fingerprint(response):
                raise ValueError("corrupt multi-model LLM response checkpoint")
            return validate(response)
    response = validate(
        request_json(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _canonical_json(payload)},
            ],
            validate,
            request_kind="interpretation",
        )
    )
    _write_json(response_path, response)
    _write_json(
        complete_path,
        {
            "status": "complete",
            "input_fingerprint": fingerprint,
            "response_sha256": _fingerprint(response),
        },
    )
    return response


def _decision_validator(definitions, cards):
    base = _role_response_validator(definitions=definitions)
    allowed = {
        card["feature_id"]: {r["evidence_id"] for r in card["modeling_evidence"]} for card in cards
    }
    roles_by_evidence = {
        r["evidence_id"]: r["role"] for card in cards for r in card["modeling_evidence"]
    }
    locked = {_feature_id(f) for f in definitions if f.get("configured_explicit_feature") is True}

    def validate(response):
        cleaned = base(response)
        raw = {row["feature_id"]: row for row in response["decisions"]}
        for row in cleaned["decisions"]:
            original = raw[row["feature_id"]]
            citations = original.get("evidence_ids")
            if not isinstance(citations, list) or not set(citations) <= allowed[row["feature_id"]]:
                raise ValueError(
                    "role decision must cite only this candidate's supplied evidence IDs"
                )
            if row["roles"] and not citations and row["feature_id"] not in locked:
                raise ValueError("retained roles require empirical evidence references")
            cited_roles = {roles_by_evidence[k] for k in citations}
            if row["feature_id"] not in locked:
                if "effect_modifier" in row["roles"] and "effect" not in cited_roles:
                    raise ValueError("modifier roles must cite treatment-heterogeneity evidence")
                if "confounder" in row["roles"] and not {"treatment", "outcome"} <= cited_roles:
                    raise ValueError(
                        "confounder roles must discuss both treatment and outcome evidence"
                    )
            stability = original.get("stability")
            if stability not in {"consistent", "mixed", "insufficient"}:
                raise ValueError(
                    "role decision requires stability: consistent, mixed, or insufficient"
                )
            row.update({"evidence_ids": list(dict.fromkeys(citations)), "stability": stability})
        return cleaned

    return validate


def adjudicate_multi_model_roles(
    *, definitions, statistical_report, request_json, output_dir, policy
):
    policy.validate()
    if not policy.enabled:
        raise ValueError("multi_model requires LLM role adjudication")
    evidence = build_multi_model_role_evidence(
        definitions=definitions, statistical_report=statistical_report, policy=policy
    )
    cfg = statistical_report["policy"]["multi_model"]
    maximum_chars = int(cfg["max_prompt_chars"])
    maximum_items = max(2, int(policy.max_candidates_per_request))
    directory = Path(output_dir)
    identity = {
        "evidence": _fingerprint(evidence),
        "policy": policy.public_dict(),
        "prompt": PROMPT_VERSION,
        "model": statistical_report.get("adjudication_model_identity"),
        "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    _write_json(directory / "evidence.json", evidence)
    cards = evidence["candidates"]

    def theme_payload(batch):
        return {
            "task": "review_stage2_multi_model_themes",
            "prompt_version": PROMPT_VERSION,
            "score_meaning": evidence["score_meaning"],
            "analysis_populations": evidence["analysis_populations"],
            "candidates": batch,
            "required_response": {
                "themes": [
                    {
                        "name": "theme",
                        "member_feature_ids": ["candidate IDs"],
                        "evidence_ids": [
                            "at most 12 representative supplied modeling evidence IDs"
                        ],
                        "interpretation": "common or complementary evidence",
                        "disagreements": "contradictions, weak signals, and proxy distinctions",
                    }
                ]
            },
            "coverage": "Cover every supplied candidate, including weak/unevaluable candidates.",
        }

    batches = _bounded_batches(
        cards, theme_payload, maximum_items=maximum_items, maximum_chars=maximum_chars
    )
    themes = []
    for index, batch in enumerate(batches):
        response = _request(
            directory,
            f"themes/initial_{index:03d}",
            theme_payload(batch),
            _themes_validator(
                [c["feature_id"] for c in batch],
                {r["evidence_id"] for c in batch for r in c["modeling_evidence"]},
                maximum=len(batch),
            ),
            request_json=request_json,
            maximum_chars=maximum_chars,
            identity=identity,
        )
        themes.extend(response["themes"])
    # A hierarchical reduction exposes cross-batch themes without a giant prompt.
    level = 0
    while len(themes) > maximum_items:

        def merge_payload(batch):
            return {
                "task": "merge_stage2_multi_model_themes",
                "themes": batch,
                "maximum_output_themes": max(1, len(batch) // 2),
                "instructions": "Preserve all candidate IDs and distinctions. Broader parent themes organize evidence; they do not imply measurement equivalence.",
                "required_response": theme_payload([])["required_response"],
            }

        batches = _bounded_batches(
            themes, merge_payload, maximum_items=maximum_items, maximum_chars=maximum_chars
        )
        if all(len(batch) == 1 for batch in batches):
            raise ValueError("theme merge cannot fit two summaries; increase max_prompt_chars")
        reduced = []
        for index, batch in enumerate(batches):
            if len(batch) == 1:
                reduced.extend(batch)
                continue
            response = _request(
                directory,
                f"themes/merge_{level:03d}_{index:03d}",
                merge_payload(batch),
                _themes_validator(
                    {k for t in batch for k in t["member_feature_ids"]},
                    {k for t in batch for k in t["evidence_ids"]},
                    maximum=max(1, len(batch) // 2),
                ),
                request_json=request_json,
                maximum_chars=maximum_chars,
                identity=identity,
            )
            reduced.extend(response["themes"])
        themes = reduced
        level += 1
    _write_json(directory / "themes.json", {"themes": themes, "all_candidate_ids_preserved": True})
    by_id = {_feature_id(f): f for f in definitions}

    def role_payload(batch):
        ids = {c["feature_id"] for c in batch}
        relevant = [t for t in themes if ids & set(t["member_feature_ids"])]
        return {
            "task": "adjudicate_stage2_multi_model_roles",
            "prompt_version": PROMPT_VERSION,
            "score_meaning": evidence["score_meaning"],
            "analysis_populations": evidence["analysis_populations"],
            "candidates": batch,
            "themes": relevant,
            "required_response": {
                "summary": "overall interpretation",
                "decisions": [
                    {
                        "feature_id": "each supplied candidate exactly once",
                        "roles": ["confounder and/or effect_modifier, or empty"],
                        "evidence_ids": [
                            "this candidate's evidence IDs; modifiers cite effect evidence; confounders cite treatment and outcome evidence"
                        ],
                        "evidence_for": ["specific facts"],
                        "evidence_against": ["specific facts"],
                        "inner_fold_consistency": "compare folds and subset exposures",
                        "cross_method_reconciliation": "reconcile disagreements",
                        "rationale": "justify roles from evidence",
                        "stability": "consistent, mixed, or insufficient",
                    }
                ],
            },
        }

    batches = _bounded_batches(
        cards,
        role_payload,
        maximum_items=int(policy.max_candidates_per_request),
        maximum_chars=maximum_chars,
    )
    decisions, summaries = [], []
    for index, batch in enumerate(batches):
        response = _request(
            directory,
            f"roles/batch_{index:03d}",
            role_payload(batch),
            _decision_validator([by_id[c["feature_id"]] for c in batch], batch),
            request_json=request_json,
            maximum_chars=maximum_chars,
            identity=identity,
        )
        decisions.extend(response["decisions"])
        summaries.append(response["summary"])
    combined = _decision_validator(definitions, cards)(
        {"summary": " ".join(summaries), "decisions": decisions}
    )
    selected = _selected_from_adjudication(definitions=definitions, adjudication=combined)
    for feature in selected:
        if feature.get("configured_explicit_feature") is not True:
            feature["selection_source"] = "multi_model_llm_adjudication"
    report = {
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "status": "complete",
        "evidence_fingerprint": identity["evidence"],
        "themes": themes,
        "decisions": combined["decisions"],
        "summary": combined["summary"],
        "retained_feature_ids": [_feature_id(f) for f in selected],
    }
    _write_json(directory / "response.json", report)
    _write_json(
        directory / "complete.json",
        {
            "status": "complete",
            "identity": identity,
            "candidate_count": len(definitions),
            "retained_count": len(selected),
        },
    )
    return selected, report, evidence
