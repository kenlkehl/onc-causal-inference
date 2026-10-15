"""Prompt-safe multi-model evidence cards for the optional concept review.

This interface accepts definitions and aggregate evidence only. Numerical cell
artifacts, row identifiers, predictions, error messages, and source paths are
not copied into a prompt.
"""

from __future__ import annotations

import math

from .stage2_multi_model_config import FAMILIES, SCHEMA_VERSION
from .stage2_role_adjudication import _feature_id, _prompt_safe_definition

from . import stage2_clinical_prompts as clinical_prompts


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
