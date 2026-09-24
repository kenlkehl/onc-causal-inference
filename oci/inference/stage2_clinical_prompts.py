"""Readable clinical inputs and Python-owned response bookkeeping for Stage 2."""
from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence

from .stage2_prompt_catalog import PROMPT_VERSION, SYSTEM_PROMPTS


def messages(slug: str, clinical_input: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPTS[slug]},
        {"role": "user", "content": clinical_input.strip()},
    ]


def label(value) -> str:
    name = value.get("name") if isinstance(value, Mapping) else value
    return str(name or "").replace("_", " ").strip()


def label_map(features) -> dict[str, str]:
    result = {}
    for feature in features:
        name = str(feature["name"])
        display = label(feature)
        if not display or display in result:
            raise ValueError("clinical variable labels must be nonempty and distinct")
        result[display] = name
    return result


def resolve_label(value, mapping):
    if not isinstance(value, str):
        raise ValueError("a clinical variable name must be text")
    if value in mapping:
        return mapping[value]
    matches = {target for name, target in mapping.items()
               if label(value).casefold() == label(name).casefold()}
    if len(matches) != 1:
        raise ValueError(f"unknown or ambiguous clinical variable {value!r}")
    return matches.pop()


def named_values(values, features):
    if not isinstance(values, Mapping):
        raise ValueError("expected values keyed by the supplied clinical variable names")
    names = label_map(features)
    result = {}
    for key, value in values.items():
        name = resolve_label(key, names)
        if name in result:
            raise ValueError(f"duplicate value for {label(name)}")
        result[name] = value
    missing = set(names.values()) - set(result)
    if missing:
        raise ValueError("missing clinical variables: " + ", ".join(label(x) for x in sorted(missing)))
    return result


def scalar(value) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def feature_text(feature, *, include_conflict=True):
    lines = [label(feature)]
    for key, title in (("description", "Meaning"), ("value_type", "Value type"),
                       ("measurement_definition", "Measurement rule"),
                       ("missing_value_rule", "Missing-value rule"),
                       ("accepted_representations", "Supported forms")):
        if feature.get(key):
            lines.append(f"- {title}: {feature[key]}")
    categories = feature.get("categories_or_unit") or []
    if categories:
        title = "Unit" if feature.get("value_type") == "continuous" else "Allowed categories"
        lines.append(f"- {title}: " + "; ".join(map(str, categories)))
    conflict = feature.get("conflict_resolution")
    if include_conflict and isinstance(conflict, Mapping):
        strategy = conflict.get("strategy", "single_or_null")
        rules = {
            "latest": "Use the latest dated observation; break equal-date ties using the last mention. If all observations are undated, use the last mention.",
            "earliest": "Use the earliest dated observation; break equal-date ties using the first mention. If all observations are undated, use the first mention.",
            "maximum": "Use the largest supported numerical measurement.",
            "minimum": "Use the smallest supported numerical measurement.",
            "mode": "Use the most frequently documented value.",
            "single_or_null": "Use the value when observations agree; use null for unresolved conflicts.",
            "any_positive": f"Use {conflict.get('positive_category')} when any observation supports that category.",
        }
        lines.append("- Choose among repeated observations using this rule: " + rules[strategy])
    return "\n".join(lines)


def definitions_text(features, *, include_conflict=True):
    label_map(features)
    return "Clinical variables\n\n" + "\n\n".join(
        feature_text(f, include_conflict=include_conflict) for f in features
    )


_PRIVATE_KEYS = {
    "feature_id", "candidate_id", "evidence_id", "evidence_ids", "row_id", "row_ids",
    "fit_row_ids", "heldout_row_ids", "patient_id", "patient_ids", "inner_fold",
    "schema_version", "prompt_version", "input_fingerprint", "source_sha256",
    "configured_explicit_feature", "configured_roles", "source_path", "dataset_path",
}


def readable(value, *, indent=0):
    """Render an already allowlisted diagnostic object without software identifiers."""
    if isinstance(value, Mapping):
        lines = []
        for key, item in value.items():
            if key in _PRIVATE_KEYS or str(key).endswith(("_ids", "_fingerprint", "_path")):
                continue
            title = str(key).replace("_", " ").capitalize()
            if isinstance(item, (Mapping, list, tuple)):
                lines.append(" " * indent + title + ":")
                lines.append(readable(item, indent=indent + 2))
            else:
                lines.append(" " * indent + f"{title}: {scalar(item)}")
        return "\n".join(lines)
    if isinstance(value, (list, tuple)):
        return "\n".join(" " * indent + "- " + readable(item, indent=indent + 2).lstrip()
                         for item in value)
    return " " * indent + scalar(value)


METHODS = {
    "univariable": "Univariable association or interaction model",
    "penalized_main": "Penalized main-effect model",
    "penalized_interactions": "Penalized outcome model with treatment interactions",
    "orthogonal_linear": "Penalized linear model of treatment/outcome residuals",
    "univariable_rlearner": "Univariable R-learner",
    "predictive_forest": "Predictive forest",
    "causal_forest": "Causal forest",
}
SCORES = {
    "univariable": "minus log10 of the association p-value",
    "penalized_main": "coefficient-group magnitude",
    "penalized_interactions": "coefficient-group magnitude",
    "orthogonal_linear": "coefficient-group magnitude",
    "univariable_rlearner": "validation R-loss improvement over a constant effect",
    "predictive_forest": "validation prediction-loss increase after shuffling the variable",
    "causal_forest": "validation R-loss increase after shuffling the variable in the same fitted forest",
}


def study_text(evidence):
    study = evidence.get("study_context") or {}
    question = study.get("clinical_question")
    lines = ["Study", "Observational comparison of treatment 1 with treatment 0."]
    if question:
        lines.append(str(question))
    binary = study.get("outcome_type", "binary") == "binary"
    lines.append("The effect is the difference in the probability of the recorded outcome under treatment 1 versus 0."
                 if binary else "The effect is the difference in the recorded outcome under treatment 1 versus 0.")
    lines.append("The clinical variables describe health before treatment. Analyses use overlapping groups of patients from this dataset.")
    population = evidence.get("analysis_populations") or {}
    low, high = population.get("modifier_min_propensity"), population.get("modifier_max_propensity")
    if low is not None or high is not None:
        lines.append(f"Modifier analyses use estimated treatment probabilities from {low if low is not None else 0} to {high if high is not None else 1}, including the bounds.")
    return "\n".join(lines)


def evidence_text(card):
    definition = dict(card.get("definition") or {})
    definition.setdefault("name", card.get("name", "Clinical variable"))
    lines = [feature_text(definition), "", "Model results"]
    rows = card.get("modeling_evidence")
    if rows is None:
        lines.append(readable({k: v for k, v in card.items() if k not in {"definition", "feature_id"}}))
        return "\n".join(lines)
    for row in rows:
        family = row["family"]
        target = {"treatment": "treatment choice", "outcome": "outcome", "effect": "treatment-effect differences"}[row["role"]]
        lines.append(f"- {METHODS[family]} — {target}: {row.get('supported', 0)} supported results among {row.get('evaluated', 0)} usable fits; {row.get('not_evaluable', 0)} fits unavailable.")
        if row.get("mean_score") is not None:
            lines.append(f"  Mean {SCORES[family]}: {row['mean_score']}; range {row.get('min_score')} to {row.get('max_score')}.")
        for key, title in (("median_p", "Median p-value"), ("median_q", "Median multiplicity-adjusted q-value"),
                           ("q_supported", "Fits meeting the adjusted q-value criterion"),
                           ("mean_model_gain", "Mean whole-model validation gain over a constant predictor/effect"),
                           ("mean_split_importance", "Mean forest split importance"),
                           ("mean_permutation_sd", "Mean spread across permutation repeats")):
            if row.get(key) is not None:
                lines.append(f"  {title}: {row[key]}.")
        if family == "univariable" and row.get("median_q") is None:
            lines.append("  Multiplicity-adjusted q-values are unavailable.")
        if row.get("folds"):
            lines.append("  Supported/usable fits across overlapping patient groups: " + "; ".join(
                f"{f.get('supported', 0)}/{f.get('evaluated', 0)}" for f in row["folds"]) + ".")
    return "\n".join(lines)


def evidence_input(evidence, cards):
    return study_text(evidence) + "\n\n" + "\n\n".join(evidence_text(c) for c in cards)


def role_decision(value, feature, *, modeling_evidence=()):
    """Validate a one-variable judgment; attach internal identity and evidence in Python."""
    if not isinstance(value, Mapping) or set(value) != {"confounder", "effect_modifier"}:
        raise ValueError("return confounder and effect_modifier judgments for this clinical variable")
    fields = {"assign", "assessment", "stability", "rationale", "evidence_comments"}
    roles, explanations, comments = [], [], []
    for role in ("confounder", "effect_modifier"):
        row = value[role]
        if not isinstance(row, Mapping) or set(row) != fields:
            raise ValueError(f"{role} requires assign, assessment, stability, rationale, and evidence_comments")
        if not isinstance(row["assign"], bool):
            raise ValueError(f"{role}.assign must be a boolean")
        if row["assessment"] not in {"supported", "plausible", "uncertain", "not_supported"}:
            raise ValueError(f"invalid {role} assessment")
        if row["stability"] not in {"consistent", "mixed", "insufficient"}:
            raise ValueError(f"invalid {role} stability")
        if row["assign"] and row["assessment"] not in {"supported", "plausible"}:
            raise ValueError(f"retaining {role} requires a supported or plausible assessment")
        if not isinstance(row["rationale"], str) or not row["rationale"].strip():
            raise ValueError(f"{role} requires a rationale")
        if not isinstance(row["evidence_comments"], list) or any(not isinstance(x, str) for x in row["evidence_comments"]):
            raise ValueError(f"{role}.evidence_comments must be an array of text")
        if row["assign"]:
            roles.append(role)
        explanations.append(f"{role}: {row['rationale']}")
        comments.extend(row["evidence_comments"])
    if feature.get("configured_explicit_feature") is True:
        roles = list(feature.get("roles") or [])
    stability = {row["stability"] for row in value.values()}
    return {
        "feature_id": str(feature.get("feature_id") or feature["name"]), "roles": roles,
        "rationale": "\n".join(explanations),
        "inner_fold_consistency": "; ".join(f"{role}: {value[role]['stability']}" for role in value),
        "cross_method_reconciliation": "\n".join(comments) or "The supplied evidence is insufficient for further comparison.",
        "evidence_for": [x for role in value if value[role]["assign"] for x in value[role]["evidence_comments"]],
        "evidence_against": [x for role in value if not value[role]["assign"] for x in value[role]["evidence_comments"]],
        "evidence_ids": [row["evidence_id"] for row in modeling_evidence],
        "evidence_attachment": "Python attached available input evidence; these are not model-authored citations.",
        "stability": next(iter(stability)) if len(stability) == 1 else "mixed",
        "role_assessments": dict(value),
    }


def study_context(report):
    raw = report.get("study_context") or {}
    return {key: str(raw[key]) for key in ("clinical_question", "outcome_type") if raw.get(key) is not None}
