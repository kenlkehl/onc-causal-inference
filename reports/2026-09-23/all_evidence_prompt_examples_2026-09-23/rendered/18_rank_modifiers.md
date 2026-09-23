# 18. Rank modifier candidates

**Unabridged current template, with invented miniature inputs. No LLM was called.**

Activation: Multi-model branch with cross-validated modifier-count selection



Source: [rank_modifier_candidates.payload](/data1/ken/pcori_dev/causal-dragonnet-text/oci/inference/stage2_modifier_ranking.py:132)

## System message

```text
Rank pretreatment measurements as inputs to a heterogeneous treatment
effect model, using only the supplied modeling evidence. Compare effect-signal
magnitude, held-out R-loss gains, fold consistency, evaluability, complementary
information, and redundant proxies. Prognostic importance alone is not effect
modification. Logistic interactions concern log odds; orthogonal models concern
outcome/probability differences. Methods and overlapping folds are not independent
replications; support fractions and p-values are not causal probabilities.
All modifier evidence uses the supplied propensity-eligible population; assess
its evaluability within that population, not as evidence for excluded patients.
Order every supplied candidate exactly once, including weak or unevaluable ones.
Put weak, contradictory, or redundant evidence later; do not invent support.
Cite only each candidate's supplied effect-evidence IDs. No count or hard p-value
threshold is chosen here. Confounder retention and investigator locks are handled
separately. Do not merge, rename, or alter measurements. Never infer an oracle or
hidden data-generating process. For a merge, interleave the supplied ordered
lists without changing the relative order within either list. Return JSON only.
```

## User message

```json
{
  "analysis_populations": {
    "association_evidence": "all_sampled_patients_except_joint_interaction_model_main_effects",
    "bounds_are_inclusive": true,
    "modifier_evidence": "propensity_eligible_patients_using_training_only_nuisances",
    "modifier_max_propensity": 0.9,
    "modifier_min_propensity": 0.1,
    "null_bound_means_unrestricted": true
  },
  "candidates": [
    {
      "definition": {
        "categories_or_unit": [
          "mg/dL"
        ],
        "categories_or_unit_truncated": false,
        "configured_roles": [],
        "derived_equivalent_measurement": false,
        "description": "Serum creatinine concentration.",
        "evidence_axes": [],
        "feature_id": "example_creatinine",
        "investigator_locked": false,
        "measurement_definition": "Latest documented pretreatment serum creatinine, in mg/dL; preserve a reported threshold if no exact number exists.",
        "missing_value_rule": "Null if unreported or unresolved.",
        "name": "serum_creatinine",
        "source_feature_ids": [],
        "supporting_architectures": [],
        "value_type": "continuous"
      },
      "feature_id": "example_creatinine",
      "modeling_evidence": [
        {
          "evaluated": 3,
          "evidence_id": "multi:example_creatinine:penalized_main:treatment",
          "exposures": 3,
          "family": "penalized_main",
          "folds": [
            {
              "evaluated": 1,
              "inner_fold": 1,
              "supported": 1
            },
            {
              "evaluated": 1,
              "inner_fold": 2,
              "supported": 1
            },
            {
              "evaluated": 1,
              "inner_fold": 3,
              "supported": 0
            }
          ],
          "max_score": null,
          "mean_model_gain": null,
          "mean_permutation_sd": null,
          "mean_score": 0.12,
          "mean_split_importance": null,
          "median_p": null,
          "median_q": null,
          "min_score": null,
          "not_evaluable": 0,
          "q_supported": null,
          "role": "treatment",
          "support_fraction": 0.6666666666666666,
          "supported": 2
        },
        {
          "evaluated": 3,
          "evidence_id": "multi:example_creatinine:penalized_main:outcome",
          "exposures": 3,
          "family": "penalized_main",
          "folds": [
            {
              "evaluated": 1,
              "inner_fold": 1,
              "supported": 1
            },
            {
              "evaluated": 1,
              "inner_fold": 2,
              "supported": 1
            },
            {
              "evaluated": 1,
              "inner_fold": 3,
              "supported": 1
            }
          ],
          "max_score": null,
          "mean_model_gain": null,
          "mean_permutation_sd": null,
          "mean_score": 0.15,
          "mean_split_importance": null,
          "median_p": null,
          "median_q": null,
          "min_score": null,
          "not_evaluable": 0,
          "q_supported": null,
          "role": "outcome",
          "support_fraction": 1.0,
          "supported": 3
        },
        {
          "evaluated": 3,
          "evidence_id": "multi:example_creatinine:causal_forest:effect",
          "exposures": 3,
          "family": "causal_forest",
          "folds": [
            {
              "evaluated": 1,
              "inner_fold": 1,
              "supported": 1
            },
            {
              "evaluated": 1,
              "inner_fold": 2,
              "supported": 0
            },
            {
              "evaluated": 1,
              "inner_fold": 3,
              "supported": 0
            }
          ],
          "max_score": null,
          "mean_model_gain": null,
          "mean_permutation_sd": null,
          "mean_score": 0.001,
          "mean_split_importance": null,
          "median_p": null,
          "median_q": null,
          "min_score": null,
          "not_evaluable": 0,
          "q_supported": null,
          "role": "effect",
          "support_fraction": 0.3333333333333333,
          "supported": 1
        }
      ]
    },
    {
      "definition": {
        "categories_or_unit": [
          "Present",
          "Absent"
        ],
        "categories_or_unit_truncated": false,
        "configured_roles": [],
        "derived_equivalent_measurement": false,
        "description": "Documented emphysema.",
        "evidence_axes": [],
        "feature_id": "example_emphysema",
        "investigator_locked": false,
        "measurement_definition": "Explicit pretreatment documentation of emphysema presence or absence.",
        "missing_value_rule": "Null if unreported; silence is not absence.",
        "name": "emphysema",
        "source_feature_ids": [],
        "supporting_architectures": [],
        "value_type": "binary"
      },
      "feature_id": "example_emphysema",
      "modeling_evidence": [
        {
          "evaluated": 3,
          "evidence_id": "multi:example_emphysema:univariable:effect",
          "exposures": 3,
          "family": "univariable",
          "folds": [
            {
              "evaluated": 1,
              "inner_fold": 1,
              "supported": 1
            },
            {
              "evaluated": 1,
              "inner_fold": 2,
              "supported": 1
            },
            {
              "evaluated": 1,
              "inner_fold": 3,
              "supported": 0
            }
          ],
          "max_score": null,
          "mean_model_gain": null,
          "mean_permutation_sd": null,
          "mean_score": 1.8,
          "mean_split_importance": null,
          "median_p": null,
          "median_q": null,
          "min_score": null,
          "not_evaluable": 0,
          "q_supported": null,
          "role": "effect",
          "support_fraction": 0.6666666666666666,
          "supported": 2
        },
        {
          "evaluated": 3,
          "evidence_id": "multi:example_emphysema:univariable_rlearner:effect",
          "exposures": 3,
          "family": "univariable_rlearner",
          "folds": [
            {
              "evaluated": 1,
              "inner_fold": 1,
              "supported": 1
            },
            {
              "evaluated": 1,
              "inner_fold": 2,
              "supported": 0
            },
            {
              "evaluated": 1,
              "inner_fold": 3,
              "supported": 0
            }
          ],
          "max_score": null,
          "mean_model_gain": null,
          "mean_permutation_sd": null,
          "mean_score": 0.002,
          "mean_split_importance": null,
          "median_p": null,
          "median_q": null,
          "min_score": null,
          "not_evaluable": 0,
          "q_supported": null,
          "role": "effect",
          "support_fraction": 0.3333333333333333,
          "supported": 1
        },
        {
          "evaluated": 3,
          "evidence_id": "multi:example_emphysema:causal_forest:effect",
          "exposures": 3,
          "family": "causal_forest",
          "folds": [
            {
              "evaluated": 1,
              "inner_fold": 1,
              "supported": 1
            },
            {
              "evaluated": 1,
              "inner_fold": 2,
              "supported": 0
            },
            {
              "evaluated": 1,
              "inner_fold": 3,
              "supported": 0
            }
          ],
          "max_score": null,
          "mean_model_gain": null,
          "mean_permutation_sd": null,
          "mean_score": 0.001,
          "mean_split_importance": null,
          "median_p": null,
          "median_q": null,
          "min_score": null,
          "not_evaluable": 0,
          "q_supported": null,
          "role": "effect",
          "support_fraction": 0.3333333333333333,
          "supported": 1
        }
      ]
    }
  ],
  "ordered_lists": [],
  "required_response": {
    "ranking": [
      {
        "evidence_ids": [
          "this candidate's supplied effect evidence IDs"
        ],
        "feature_id": "each supplied candidate exactly once",
        "rationale": "compare evidence, redundancy, and uncertainty"
      }
    ]
  },
  "score_meaning": {
    "causal_forest": "heldout_R_loss_increase_after_group_permutation",
    "orthogonal_linear": "nonzero_group_norm_in_joint_R_loss_model",
    "penalized_interactions": "nonzero_group_norm_in_joint_outcome_interaction_model",
    "penalized_main": "nonzero_group_norm_for_prediction",
    "predictive_forest": "heldout_prediction_loss_increase_after_group_permutation",
    "univariable": "minus_log10_p; support uses nominal_p; q_support recorded separately",
    "univariable_rlearner": "heldout_R_loss_gain_over_constant_effect"
  },
  "task": "rank_stage2_modifiers",
  "version": "stage2_modifier_ranking_v1"
}
```
