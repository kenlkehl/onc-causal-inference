# 09_refine_ontology: Clarify instructions after extraction failures

Clinical observations and numerical results below are invented examples.

## System message

```text
Review the extraction instructions for one clinical variable and decide whether they need clarification.

What you receive

The current definition and examples of attempted answers that failed its required format or categories.

How to decide

Choose keep when the failed answers are ordinary synonyms for existing categories and the clinical measurement rule is clear. Revise when the examples reveal an unclear measurement rule, unsuitable type, or missing distinction in the category definitions. A clarification can explain how to interpret equivalent wording. Use null for missing or unresolved findings. An explicit negative must describe this patient's finding to support Absent. Failed outputs are evidence about how the instructions were interpreted; the patient's actual value remains unknown from those outputs alone.

What to return

For keep, return one JSON object with action equal to keep and reason (text). For revise, return action equal to revise, reason, and definition. definition contains exactly description, value_type, categories_or_unit, measurement_definition, and missing_value_rule. value_type is continuous, binary, categorical, or ordinal. categories_or_unit contains a unit for a dimensional continuous value, an empty array for a unitless value, exactly two distinct labels for binary values, or at least two for categorical/ordinal values. Preserve the same clinical variable.
```

## User message

```text
Clinical variable: Emphysema on imaging
Description: Imaging documentation of emphysema presence or absence.
Value type: binary
Categories: Present, Absent
Measurement rule: Use an explicit positive or negative imaging statement.
Missing-value rule: Use null for unreported or unresolved findings.

Failed answers from three patients: positive; present on CT. These strings fall outside the two required category labels.
```

## Developer integration notes — excluded from the prompt

Python attaches identity and restricts revisions to the allowed definition fields, then controls re-extraction. Ordinary category synonyms can use the category-mapping path.
