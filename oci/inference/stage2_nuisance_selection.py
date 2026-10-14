"""Training-only predictive supports for the multi-model nuisance inputs.

These are model-routing decisions, not claims that a variable is a confounder.
The modifier ranking/count search remains a separate selection procedure.
"""

from .stage2_taskwise_policy import route_taskwise_features

SCHEMA_VERSION = "stage2_predictive_nuisance_selection_v1"


def select_predictive_nuisances(definitions, folds, *, effect_votes=None):
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
    selected, decisions, routing = route_taskwise_features(
        definitions,
        treatment_votes=votes["treatment"],
        outcome_votes=votes["outcome"],
        effect_votes=effect_votes or dict.fromkeys(keys, 0),
        inner_folds=len(folds),
    )
    for row in [*selected, *decisions]:
        row["selection_authority"] = "multi_model"
        if row.get("configured_explicit_feature") is not True:
            row["selection_source"] = SCHEMA_VERSION
    # This audit describes nuisance retention, not the modifier set that the
    # subsequent count search will choose. Do not publish provisional X/W lists
    # as though they were the final estimator's routing.
    routing["task_feature_ids"] = {
        task: routing["task_feature_ids"][task] for task in ("treatment", "outcome")
    }
    for name in ("forest_x_feature_ids", "forest_w_feature_ids"):
        routing.pop(name)
    routing.update(
        schema_version=SCHEMA_VERSION,
        selection_authority="multi_model",
        selection_rule="nonzero_group_in_any_training_fold_per_nuisance_task",
        outcome_support="union_of_marginal_and_arm_specific_outcome_supports",
        effect_selection_source=(
            "joint_orthogonal_model_support"
            if effect_votes is not None
            else "separate_nested_modifier_ranking_and_count_search"
        ),
        treatment_votes=votes["treatment"],
        outcome_votes=votes["outcome"],
        training_folds=len(folds),
        folds=folds,
        cross_fold_union_used_for_scoring_nuisances=False,
        repetitions_are_independent_samples=False,
    )
    return selected, decisions, routing
