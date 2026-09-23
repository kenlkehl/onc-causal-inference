# 05. Extract one patient's values

**Unabridged current template, with invented miniature inputs. No LLM was called.**

Activation: Training and heldout extraction; same template



Source: [_extraction_prompt](/data1/ken/pcori_dev/causal-dragonnet-text/oci/inference/plain_handoff_stage2_analysis.py:1720)

## System message

```text
You extract prespecified variables from supplied clinical text. Return JSON only.
```

## User message

```json
{
  "features": [
    {
      "accepted_representations": "one JSON number, or one documented categorical/threshold string when the numeric measurement is unavailable",
      "categories_or_unit": [
        "mg/dL"
      ],
      "conflict_resolution": {
        "dated_observations_precede_undated": true,
        "positive_category": null,
        "source_order_tie_breaker": "last",
        "strategy": "latest",
        "strategy_source": "explicit_ontology"
      },
      "description": "Serum creatinine concentration.",
      "measurement_definition": "Latest documented pretreatment serum creatinine, in mg/dL; preserve a reported threshold if no exact number exists.",
      "missing_value_rule": "Null if unreported or unresolved.",
      "name": "serum_creatinine",
      "value_type": "continuous"
    },
    {
      "categories_or_unit": [
        "Present",
        "Absent"
      ],
      "conflict_resolution": {
        "dated_observations_precede_undated": true,
        "positive_category": null,
        "source_order_tie_breaker": "last",
        "strategy": "latest",
        "strategy_source": "explicit_ontology"
      },
      "description": "Documented emphysema.",
      "measurement_definition": "Explicit pretreatment documentation of emphysema presence or absence.",
      "missing_value_rule": "Null if unreported; silence is not absence.",
      "name": "emphysema",
      "value_type": "binary"
    }
  ],
  "job": "extract_stage2_patient_variables",
  "patients": [
    {
      "row_id": 1,
      "text": "2025-01-01 pretreatment lab: serum creatinine 1.0 mg/dL. 2025-01-10 pretreatment lab: serum creatinine 1.2 mg/dL. CT documents emphysema."
    }
  ],
  "response": {
    "rows": [
      {
        "row_id": "one supplied integer row_id",
        "values": {
          "every supplied feature name": "scalar value or null"
        }
      }
    ]
  },
  "rules": [
    "Use only the supplied clinical text for the patient in that row.",
    "Apply the measurement definition and missing-value rule literally.",
    "Consider every explicitly supported observation for a feature before selecting its one output value.",
    "When multiple supported observations remain, apply that feature's conflict_resolution policy literally. The conflict_resolution policy governs if prose in the measurement definition is ambiguous or inconsistent about how to choose among observations.",
    "For latest or earliest conflict resolution, prefer observations with an explicit governing date or time. If none are dated, use clinical-text source order and the declared source_order_tie_breaker; do not treat the first mention, diagnosis value, or demographics value as automatically authoritative.",
    "For a binary, categorical, or ordinal feature, return one declared category exactly.",
    "Do not substitute 0/1 or true/false for a declared category unless that exact value is declared.",
    "Return one scalar value or null per feature; never return an object or array.",
    "For a continuous feature, return one JSON number whenever the record supplies the requested numeric measurement. If the record supplies only a documented categorical or threshold representation of that same measurement, return that one concise string instead of discarding it or inventing a number. From a composite such as 147/93, use only a component explicitly named by the feature; if the definition requests multiple components, return null rather than a ratio string or aggregate.",
    "Return null when the record does not support a value.",
    "Return every row and every feature exactly once."
  ]
}
```
