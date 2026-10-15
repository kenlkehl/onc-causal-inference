"""Readable clinical inputs and Python-owned response bookkeeping for Stage 2."""
from __future__ import annotations

import json
from collections.abc import Mapping

from .stage2_prompt_catalog import (  # noqa: F401 - versions are read as module attributes
    ALIAS_MERGE_PROMPT_VERSION,
    PROMPT_VERSION,
    SYSTEM_PROMPTS,
)


def messages(slug: str, clinical_input: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPTS[slug]},
        {"role": "user", "content": clinical_input.strip()},
    ]


def label(value) -> str:
    name = (value.get("clinical_label") or value.get("name")) if isinstance(value, Mapping) else value
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
        components = [name for name in mapping
                      if ":" in name and label(value).casefold()
                      == label(name.rsplit(":", 1)[0]).casefold()]
        if components:
            raise ValueError(
                f"unknown or ambiguous clinical variable {value!r}; this is a parent name. "
                "Return each component separately using its complete name, including "
                "the text after the colon: " + json.dumps(components, ensure_ascii=False)
            )
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
    # Quote the complete label so a colon in a split component is part of its
    # name, rather than a delimiter between the variable and its definition.
    lines = [json.dumps(label(feature), ensure_ascii=False)]
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
    "orthogonal_linear": "Grouped DINA log-odds model for binary outcomes; residual linear model otherwise",
    "univariable_rlearner": "Candidate DINA model for binary outcomes; R-learner otherwise",
    "predictive_forest": "Predictive forest",
    "causal_forest": "Causal forest",
    "dina": "Grouped Gao-Hastie DINA log-odds effect model",
    "matched_batch_contrast": "Matched patient-batch contrast model",
}
SCORES = {
    "univariable": "minus log10 of the association p-value",
    "penalized_main": "coefficient-group magnitude",
    "penalized_interactions": "coefficient-group magnitude",
    "orthogonal_linear": "coefficient-group magnitude",
    "univariable_rlearner": "validation DINA likelihood gain for binary outcomes, R-loss gain otherwise, over a constant effect",
    "dina": "validation DINA loss increase after jointly shuffling all levels of the variable",
    "predictive_forest": "validation prediction-loss increase after shuffling the variable",
    "causal_forest": "validation R-loss increase after shuffling the variable in the same fitted forest",
    "matched_batch_contrast": "fractional reduction in validation batch-deviation error beyond bin intercepts",
}


def batch_evidence_text(row):
    """Explain the additional evidence in both role and cross-fold concept reviews."""
    lines = [
        "Patients were grouped by predicted treatment probability and predicted outcome risk. "
        "Within each group, random batches contained equal numbers receiving each treatment. "
        "The target was each batch's observed treated-minus-untreated outcome contrast minus its training group's contrast. "
        "This candidate's average values predicted that deviation in separate validation patients. "
        "The comparison model contained the same group intercepts. Positive gain means lower prediction error; "
        "0.10 means a 10% error reduction. This is evidence about observed contrast variation and can reflect residual confounding. "
        "It does not measure an individual patient's causal effect. Shuffles reuse patients; support counts refer to model fits."]
    d = row.get("batch_diagnostics") or {}
    for name in ("filtered", "unfiltered"):
        gain = d.get(f"mean_{name}_gain")
        if gain is not None:
            lines.append(f"Mean validation gain after {name} training: {gain:.5g} across {d.get(name + '_evaluated')} usable fits.")
    lines.append("Filtered training emphasizes unusually large deviations. Validation batches are scored without filtering on outcomes.")
    if d.get("filter_agreement_evaluated"):
        lines.append(f"Filtered and unfiltered training agreed on whether this candidate helped in "
                     f"{d['filter_agreement_count']} of {d['filter_agreement_evaluated']} comparable fits.")
    for phase in ("training", "validation"):
        if d.get(f"mean_{phase}_patients") is not None:
            lines.append(f"Mean unique {phase} patients per usable primary fit: {d[f'mean_{phase}_patients']:.5g}; "
                         f"mean batches: {d[f'mean_{phase}_batches']:.5g}.")
        if d.get(f"mean_{phase}_proposed") is not None:
            lines.append(f"Mean {phase} batches proposed per attempted fit: {d[f'mean_{phase}_proposed']:.5g}; "
                         f"rejected for nuisance imbalance: {d[f'mean_{phase}_imbalance_rejected']:.5g}.")
    return "\n".join(lines)


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


def study_context(report):
    raw = report.get("study_context") or {}
    return {key: str(raw[key]) for key in ("clinical_question", "outcome_type") if raw.get(key) is not None}
