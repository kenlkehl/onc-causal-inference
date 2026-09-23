# 14. Adjudicate roles in the llm_roles mode

**Unabridged current template, with invented miniature inputs. No LLM was called.**

Activation: Alternative selection branch: llm_roles

Unsupplied statistical results are empty/null in this small fixture, not negative study results.

Source: [_role_request_payload](/data1/ken/pcori_dev/causal-dragonnet-text/oci/inference/stage2_role_adjudication.py:553)

Source: [build_stage2_role_evidence](/data1/ken/pcori_dev/causal-dragonnet-text/oci/inference/stage2_role_adjudication.py:362)

## System message

```text
You are the final causal-role adjudicator inside one outer training fold. Every
candidate measurement is pretreatment by a hard upstream invariant. Decide
which candidates should be retained as confounders, effect modifiers, both, or
neither using only the supplied definitions and fold-honest evidence.

Treat every statistical method as fallible evidence, not as a gate or an oracle.
For confounding, distinguish a plausible common cause of treatment and outcome
from a treatment-only predictor, an outcome-only prognostic factor, a mediator,
or an instrument. Elastic-net support for either nuisance task alone is not
sufficient. Univariable evidence can recover signals suppressed by correlated
covariates, but multiplicity, instability, and disagreement must be discussed.

For effect modification, require empirical treatment-heterogeneity evidence.
Reconcile the candidate-wise held-out R-loss comparison with the joint grouped
elastic-net interaction model. Outcome prognosis or a main-effect association
alone is not modifier evidence. Negative held-out R-loss gains and inconsistent
fold behavior count as evidence against promotion.

Definitions may inform causal interpretation, but a suggestive feature name is
not hidden truth. Never infer a data-generating process, synthetic provenance,
oracle label, or true role that is not present in the supplied evidence. You do
not have outer-heldout data. Investigator-locked roles must be preserved
exactly. Return one JSON object and cover every candidate exactly once.
```

## User message

```json
{
  "candidate_batch": {
    "batch_count": 1,
    "batch_index": 1,
    "candidate_count": 2
  },
  "decision_policy": {
    "allow_no_role": true,
    "assess_disagreement_explicitly": true,
    "assess_inner_fold_consistency_explicitly": true,
    "preserve_investigator_locked_roles_exactly": true,
    "statistical_methods_are_evidence_not_gates": true
  },
  "prompt_version": "stage2_all_evidence_role_prompt_v1",
  "required_response": {
    "decisions": [
      {
        "cross_method_reconciliation": "string",
        "evidence_against": [
          "specific supplied statistical facts"
        ],
        "evidence_for": [
          "specific supplied statistical facts"
        ],
        "feature_id": "every supplied feature ID exactly once",
        "inner_fold_consistency": "string",
        "rationale": "causal-role conclusion grounded in supplied evidence",
        "roles": [
          "zero or more of confounder, effect_modifier"
        ]
      }
    ],
    "summary": "string"
  },
  "role_evidence": {
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
        "statistical_evidence": {
          "candidate_augmented_r_learner": {
            "folds": [],
            "top_n_votes": 0
          },
          "multivariable_modifier_elastic_net": {
            "folds": [],
            "selection_votes": 0
          },
          "multivariable_nuisance_elastic_net": {
            "folds": [
              {
                "inner_fold": 1,
                "outcome_group_l2_norm": 0.15,
                "outcome_selected": true,
                "treatment_group_l2_norm": 0.12,
                "treatment_selected": true
              }
            ],
            "outcome_votes": 0,
            "treatment_votes": 0
          },
          "provisional_statistical_roles": [],
          "univariable_confounder_screen": {
            "folds": [],
            "multiplicity_adjusted_joint_support_votes": 0,
            "nominal_joint_support_votes": 0
          }
        }
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
        "statistical_evidence": {
          "candidate_augmented_r_learner": {
            "folds": [],
            "top_n_votes": 0
          },
          "multivariable_modifier_elastic_net": {
            "folds": [],
            "selection_votes": 0
          },
          "multivariable_nuisance_elastic_net": {
            "folds": [
              {
                "inner_fold": 1,
                "outcome_group_l2_norm": null,
                "outcome_selected": false,
                "treatment_group_l2_norm": null,
                "treatment_selected": false
              }
            ],
            "outcome_votes": 0,
            "treatment_votes": 0
          },
          "provisional_statistical_roles": [],
          "univariable_confounder_screen": {
            "folds": [],
            "multiplicity_adjusted_joint_support_votes": 0,
            "nominal_joint_support_votes": 0
          }
        }
      }
    ],
    "evidence_boundary": {
      "candidate_measurements_are_pre_index_treatment": true,
      "data_generation_metadata_is_excluded": true,
      "dataset_paths_and_dataset_names_are_excluded": true,
      "definition_fields_use_an_explicit_allowlist": true,
      "inner_heldout_rows_are_used_only_for_fold_honest_evaluation": true,
      "oracle_columns_are_excluded": true,
      "outer_heldout_rows_are_excluded": true,
      "patient_identifiers_are_excluded": true,
      "row_level_values_are_excluded": true,
      "statistical_evidence_uses_outer_training_rows_only": true
    },
    "methodology": {
      "all_statistics_are_evidence_not_automatic_role_labels": true,
      "confounder": [
        "multivariable grouped elastic-net treatment and marginal-outcome support",
        "candidate-wise treatment and outcome association tests, including outcome adjusted for treatment"
      ],
      "effect_modifier": [
        "candidate-augmented univariable R-learner held-out R-loss comparisons",
        "joint multivariable grouped elastic-net R-loss interaction selection"
      ]
    },
    "schema_version": "stage2_fold_honest_role_evidence_v1",
    "temporal_scope": "pre_index_treatment"
  },
  "task": "adjudicate_stage2_roles_from_all_evidence"
}
```
