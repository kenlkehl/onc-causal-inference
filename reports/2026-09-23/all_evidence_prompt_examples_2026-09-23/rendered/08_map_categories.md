# 08. Normalize an out-of-ontology categorical value

**Unabridged current template, with invented miniature inputs. No LLM was called.**

Activation: Conditional: extracted categorical values violate declared categories



Source: [_category_ontology_prompt](/data1/ken/pcori_dev/causal-dragonnet-text/oci/inference/plain_handoff_stage2_analysis.py:1213)

## System message

```text
You normalize previously extracted categorical values to a closed ontology. Return JSON only.
```

## User message

```json
{
  "items": [
    {
      "allowed_categories": [
        "Present",
        "Absent"
      ],
      "description": "Documented emphysema.",
      "feature_name": "emphysema",
      "mapping_id": "category_mapping_0000",
      "measurement_definition": "Explicit pretreatment documentation of emphysema presence or absence.",
      "missing_value_rule": "Null if unreported; silence is not absence.",
      "occurrence_count": 3,
      "prior_extracted_value": "present on CT",
      "value_type": "binary"
    }
  ],
  "job": "map_extracted_values_to_declared_category_ontology",
  "response": {
    "corrections": [
      {
        "mapping_id": "one supplied mapping_id",
        "value": "one exact allowed category or null"
      }
    ]
  },
  "rules": [
    "Use only the feature definition, allowed categories, and prior extracted value.",
    "Do not perform clinical extraction and do not infer any new patient information.",
    "Map by semantic equivalence to exactly one allowed category.",
    "Return null when the prior value does not map unambiguously.",
    "Return every mapping_id exactly once and no additional mapping IDs."
  ]
}
```
