"""Deterministic task-wise routing; semantic causal roles are not selectors.

The ``roles`` field is retained only as the existing estimator's X/W adapter.
Use ``modeling_tasks`` and ``selection_authority`` when interpreting new runs.
No patient values, oracle columns, clinical names, or LLM output enter routing.
"""
from __future__ import annotations

import copy
from numbers import Integral
from typing import Any, Mapping, Sequence

INDEPENDENT_TASKS = "independent_tasks"
LEGACY_ROLES = "llm_roles"
SCHEMA_VERSION = "stage2_independent_task_selection_v1"
TASKS = ("treatment", "outcome", "effect")


def independent_tasks_enabled(report: Mapping[str, Any]) -> bool:
    """Read the frozen scientific policy, not a model-authored role label."""
    return (report.get("policy") or {}).get("selection_mode") == INDEPENDENT_TASKS


def _key(feature: Mapping[str, Any]) -> str:
    return str(feature.get("feature_id") or feature["name"])


def route_taskwise_features(
    definitions: Sequence[Mapping[str, Any]],
    *,
    treatment_votes: Mapping[str, int],
    outcome_votes: Mapping[str, int],
    effect_votes: Mapping[str, int],
    inner_folds: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Route any-fold nonzero groups while preserving investigator overrides.

    Each task's votes come from its own regularized model. In particular,
    ``effect_votes`` must be from the joint R-loss model, not a top-N screen.
    Fold frequencies are stability diagnostics, not causal-role probabilities.
    Investigator confounder roles force both nuisance inputs; effect-only roles
    retain X eligibility without forcing either external nuisance input.
    """
    if isinstance(inner_folds, bool) or not isinstance(inner_folds, Integral) or inner_folds < 1:
        raise ValueError("inner_folds must be a positive integer")
    keys = [_key(feature) for feature in definitions]
    if len(keys) != len(set(keys)):
        raise ValueError("task-wise routing requires unique feature IDs")
    votes = dict(zip(TASKS, (treatment_votes, outcome_votes, effect_votes)))
    for task, task_votes in votes.items():
        if set(task_votes) != set(keys):
            raise ValueError(f"{task} votes must cover every candidate exactly once")
        for value in task_votes.values():
            if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value <= inner_folds:
                raise ValueError(f"invalid {task} selection vote count")
    selected: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    task_sets: dict[str, list[str]] = {task: [] for task in TASKS}
    for source in definitions:
        feature = copy.deepcopy(dict(source))
        feature_id = _key(feature)
        chosen = {task: bool(votes[task][feature_id]) for task in TASKS}
        locked = feature.get("configured_explicit_feature") is True
        if locked:
            configured_roles = list(feature.get("roles") or [])
            if not configured_roles or set(configured_roles) - {"confounder", "effect_modifier"}:
                raise ValueError("configured explicit features require supported locked roles")
            # Preserve the investigator's routing contract exactly, including
            # no accidental effect promotion for a confounder-only feature.
            chosen = {
                "treatment": "confounder" in configured_roles,
                "outcome": "confounder" in configured_roles,
                "effect": "effect_modifier" in configured_roles,
            }
        nuisance_roles = [task for task in TASKS[:2] if chosen[task]]
        roles = (["confounder"] if nuisance_roles else []) + (
            ["effect_modifier"] if chosen["effect"] else []
        )
        if locked:
            roles = list(source["roles"])
        modeling_tasks = [task for task in TASKS if chosen[task]]
        forest_role = "X" if chosen["effect"] else ("W" if nuisance_roles else "excluded")
        source_label = "investigator_locked" if locked else SCHEMA_VERSION
        decision = {
            "feature_id": feature_id,
            "name": str(feature["name"]),
            "configured_explicit_feature": locked,
            "roles": roles,
            "nuisance_model_roles": nuisance_roles,
            "modeling_tasks": modeling_tasks,
            "forest_role": forest_role,
            "retained": bool(modeling_tasks),
            "selection_source": source_label,
            "selection_authority": INDEPENDENT_TASKS,
            "roles_are_model_routing_not_causal_labels": not locked,
            "task_votes": {task: int(votes[task][feature_id]) for task in TASKS},
            "task_selection_frequency": {
                task: float(votes[task][feature_id] / inner_folds) for task in TASKS
            },
        }
        decisions.append(decision)
        for task in modeling_tasks:
            task_sets[task].append(feature_id)
        if decision["retained"]:
            feature.update({key: copy.deepcopy(decision[key]) for key in (
                "roles", "nuisance_model_roles", "modeling_tasks", "forest_role",
                "selection_source", "selection_authority", "roles_are_model_routing_not_causal_labels",
            )})
            selected.append(feature)
    routing = {
        "schema_version": SCHEMA_VERSION,
        "selection_authority": INDEPENDENT_TASKS,
        "selection_rule": "nonzero_group_in_any_inner_fold_per_task",
        "all_candidates_eligible_before_fold_local_encoding": True,
        "effect_selection_source": "joint_group_elastic_net_r_loss",
        "candidate_top_n_is_binding": False,
        "llm_roles_are_binding": False,
        "task_feature_ids": task_sets,
        "forest_x_feature_ids": [row["feature_id"] for row in decisions if row["forest_role"] == "X"],
        "forest_w_feature_ids": [row["feature_id"] for row in decisions if row["forest_role"] == "W"],
        "forest_internal_nuisances": "independently_regularized_on_X_plus_W",
        "external_aipw_nuisances": "separate_task_specific_feature_lists",
    }
    return selected, decisions, routing


def route_from_statistical_report(
    definitions: Sequence[Mapping[str, Any]], report: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Reconstruct immutable numerical selection before advisory LLM calls."""
    if not independent_tasks_enabled(report):
        raise ValueError("task-wise routing requires independent_tasks policy")
    nuisance = report["nuisance_screen"]
    effect = report["multivariable_modifier_elastic_net_screen"]
    return route_taskwise_features(
        definitions,
        treatment_votes=nuisance["treatment_votes"],
        outcome_votes=nuisance["outcome_votes"],
        effect_votes=effect["votes"],
        inner_folds=report["inner_folds"],
    )


def finalize_taskwise_report(
    definitions: Sequence[Mapping[str, Any]], report: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replace legacy provisional routing without rewriting diagnostic evidence."""
    selected, decisions, routing = route_from_statistical_report(definitions, report)
    updated = copy.deepcopy(dict(report))
    updated.update({
        "selection_method": SCHEMA_VERSION,
        "selection_authority": INDEPENDENT_TASKS,
        "taskwise_routing": routing,
        "decisions": decisions,
        "retained_feature_ids": [_key(feature) for feature in selected],
        "measurement_dependency_feature_ids": [_key(feature) for feature in selected],
    })
    nuisance_ids = sorted(set(routing["task_feature_ids"]["treatment"]) | set(routing["task_feature_ids"]["outcome"]))
    updated["nuisance_screen"].update({
        "selection_rule": "independent_any_fold_group_selection_per_task",
        "union_confounder_feature_ids": nuisance_ids,
        "union_confounder_field_is_legacy_routing_name": True,
        "union_is_used_by_both_nuisance_models": False,
    })
    updated["cross_fitted_nuisance_models"].update({
        "treatment_feature_ids": sorted(_key(feature) for feature in definitions),
        "outcome_feature_ids": sorted(_key(feature) for feature in definitions),
        "design_policy": "all_candidates_then_independent_fold_local_regularization",
        "cross_fold_screen_union_is_used": False,
    })
    updated["effect_modifier_screen"].update({
        "role": "diagnostic_only",
        "hard_selection_gate": False,
        "top_n_is_binding": False,
        "top_n_evidence_feature_ids": updated["effect_modifier_screen"].get("stable_effect_modifier_feature_ids", []),
        "stable_effect_modifier_feature_ids": sorted(routing["task_feature_ids"]["effect"]),
        "stable_effect_modifier_ids_source": "joint_r_loss_with_investigator_overrides",
    })
    updated["multivariable_modifier_elastic_net_screen"].update({
        "role": "numerical_effect_selection",
        "hard_selection_gate": True,
        "selection_rule": "nonzero_group_in_any_inner_fold",
    })
    return selected, updated
