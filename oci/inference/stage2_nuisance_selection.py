"""Training-only predictive supports for the multi-model nuisance inputs.

These are model-routing decisions, not claims that a variable is a confounder.
The modifier ranking/count search remains a separate selection procedure.
"""

import copy

SCHEMA_VERSION = "stage2_predictive_nuisance_selection_v1"
TASKS = ("treatment", "outcome", "effect")


def _route_features(definitions, votes, inner_folds):
    """Route any-fold nonzero nuisance groups while preserving investigator overrides.

    Fold frequencies are stability diagnostics, not causal-role probabilities.
    Investigator confounder roles force both nuisance inputs; effect-only roles
    retain X eligibility without forcing either external nuisance input.
    Unlocked effect routing is left to the nested modifier ranking/count search.
    """
    keys = [str(feature.get("feature_id") or feature["name"]) for feature in definitions]
    if len(keys) != len(set(keys)):
        raise ValueError("task-wise routing requires unique feature IDs")
    votes = {**votes, "effect": dict.fromkeys(keys, 0)}
    selected, decisions = [], []
    task_sets = {task: [] for task in TASKS}
    for source, feature_id in zip(definitions, keys):
        feature = copy.deepcopy(dict(source))
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
        decision = {
            "feature_id": feature_id,
            "name": str(feature["name"]),
            "configured_explicit_feature": locked,
            "roles": roles,
            "nuisance_model_roles": nuisance_roles,
            "modeling_tasks": modeling_tasks,
            "forest_role": "X" if chosen["effect"] else ("W" if nuisance_roles else "excluded"),
            "retained": bool(modeling_tasks),
            "selection_source": "investigator_locked" if locked else SCHEMA_VERSION,
            "selection_authority": "multi_model",
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
    return selected, decisions, task_sets


def select_predictive_nuisances(definitions, folds):
    """Conservatively retain each task's nonzero groups in any training fold.

    Outcome support includes the marginal outcome model and either arm-specific
    outcome model. Repeated partitions describe stability, not independent samples.
    No validation losses, semantic roles, or LLM responses determine these supports.
    """
    keys = {str(f["feature_id"]) for f in definitions}
    if len(folds) < 2:
        raise ValueError("predictive nuisance selection requires at least two training folds")
    votes = {task: dict.fromkeys(keys, 0) for task in ("treatment", "outcome")}
    for fold in folds:
        if set(fold["fit_row_ids"]) & set(fold["heldout_row_ids"]):
            raise ValueError("nuisance selection training and validation rows overlap")
        for task in votes:
            support = set(fold["task_feature_ids"][task])
            if not support <= keys:
                raise ValueError("nuisance support contains an unknown candidate")
            for key in support:
                votes[task][key] += 1
    selected, decisions, task_sets = _route_features(definitions, votes, len(folds))
    # This audit describes nuisance retention, not the modifier set that the
    # subsequent count search will choose. Do not publish provisional X/W lists
    # as though they were the final estimator's routing.
    routing = {
        "schema_version": SCHEMA_VERSION,
        "selection_authority": "multi_model",
        "selection_rule": "nonzero_group_in_any_training_fold_per_nuisance_task",
        "all_candidates_eligible_before_fold_local_encoding": True,
        "effect_selection_source": "separate_nested_modifier_ranking_and_count_search",
        "candidate_top_n_is_binding": False,
        "llm_roles_are_binding": False,
        "task_feature_ids": {task: task_sets[task] for task in ("treatment", "outcome")},
        "forest_internal_nuisances": "independently_regularized_on_X_plus_W",
        "external_aipw_nuisances": "all_candidate_cross_fitted_elastic_nets",
        "outcome_support": "union_of_marginal_and_arm_specific_outcome_supports",
        "treatment_votes": votes["treatment"],
        "outcome_votes": votes["outcome"],
        "training_folds": len(folds),
        "folds": folds,
        "cross_fold_union_used_for_scoring_nuisances": False,
        "repetitions_are_independent_samples": False,
    }
    return selected, decisions, routing
