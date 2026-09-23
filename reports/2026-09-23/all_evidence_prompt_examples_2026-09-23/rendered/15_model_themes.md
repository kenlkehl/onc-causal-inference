# 15. Review multi-model evidence for themes

**Unabridged current template, with invented miniature inputs. No LLM was called.**

Activation: Multi-model selection branch



Source: [adjudicate_multi_model_roles.theme_payload](/data1/ken/pcori_dev/causal-dragonnet-text/oci/inference/stage2_multi_model_adjudication.py:350)

## System message

```text
You review pretreatment candidate measurements using modeling evidence
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
  "coverage": "Cover every supplied candidate, including weak/unevaluable candidates.",
  "prompt_version": "stage2_multi_model_themes_v1",
  "required_response": {
    "themes": [
      {
        "disagreements": "contradictions, weak signals, and proxy distinctions",
        "evidence_ids": [
          "at most 12 representative supplied modeling evidence IDs"
        ],
        "interpretation": "common or complementary evidence",
        "member_feature_ids": [
          "candidate IDs"
        ],
        "name": "theme"
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
  "task": "review_stage2_multi_model_themes"
}
```
