# 11. Extend a frozen map for new training text values

**Unabridged current template, with invented miniature inputs. No LLM was called.**

Activation: Conditional: new training tokens during incremental extraction

The sample threshold is invented solely to demonstrate the template, not a clinical recommendation.

Source: [_harmonization_delta_prompt](/data1/ken/pcori_dev/causal-dragonnet-text/oci/inference/plain_handoff_stage2_analysis.py:4585)

## System message

```text
You extend one frozen clinical value map without revising its ontology. Return JSON only.
```

## User message

```json
{
  "feature": {
    "categories_or_unit": [
      "mg/dL"
    ],
    "description": "Serum creatinine concentration.",
    "feature_id": "example_creatinine",
    "measurement_definition": "Latest documented pretreatment serum creatinine, in mg/dL; preserve a reported threshold if no exact number exists.",
    "missing_value_rule": "Null if unreported or unresolved.",
    "name": "serum_creatinine"
  },
  "frozen_harmonization_plan": {
    "canonical_categories": [
      "Below 1.0",
      "At least 1.0"
    ],
    "numeric_bin_rules": [
      {
        "canonical_value": "Below 1.0",
        "lower_bound": null,
        "lower_inclusive": false,
        "upper_bound": 1.0,
        "upper_inclusive": false
      },
      {
        "canonical_value": "At least 1.0",
        "lower_bound": 1.0,
        "lower_inclusive": true,
        "upper_bound": null,
        "upper_inclusive": false
      }
    ],
    "reason": "Illustrative representation with one explicit numeric boundary.",
    "target_representation": "categorical",
    "unmapped_value_rule": "null"
  },
  "information_boundary": "The values come only from outer-training patients. No treatment, outcome, held-out text, or held-out values are supplied.",
  "job": "extend_stage2_harmonization_map_for_new_text_values",
  "new_observed_training_text_values": [
    {
      "count": 2,
      "raw_value": "less than 1.0"
    }
  ],
  "response_schema": {
    "categorical_value_map": [
      {
        "canonical_value": "finite number/null for continuous; frozen canonical category/null for categorical",
        "raw_value": "one exact supplied text token"
      }
    ]
  },
  "rules": [
    "Do not revise the frozen target representation, categories, or numeric bins.",
    "Return exactly one mapping for each supplied raw_value and no other raw values.",
    "Copy every raw_value exactly, including punctuation, spacing, and case.",
    "For a continuous target, use a finite number only when the exact text has an unambiguous value in the feature's stated unit.",
    "For a categorical target, use only a frozen canonical category.",
    "Map an unusable or ambiguous text token to null rather than guessing."
  ]
}
```
