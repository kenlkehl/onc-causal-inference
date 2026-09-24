# 12_supervise_ontology: Review a clinical measurement definition using extraction summaries

Clinical observations and numerical results below are invented examples.

## System message

```text
Decide whether the extraction definition for one clinical variable needs clarification.

What you receive

The definition, a summary of current extracted values, and earlier failed answers. A patient with an earlier failure may now have a valid answer, so these counts can overlap.

How to decide

Choose keep when the failed answers are ordinary synonyms for existing categories and the clinical measurement rule is clear. Revise when the examples reveal an unclear measurement rule, unsuitable type, or missing distinction in the category definitions. A clarification can explain how to interpret equivalent wording. Use null for missing or unresolved findings. An explicit negative must describe this patient's finding to support Absent. Failed outputs are evidence about how the instructions were interpreted; the patient's actual value remains unknown from those outputs alone.

Missing values, a rare finding, or a common category can occur under a sound definition. Revise when the supplied summaries reveal a specific problem in the instructions.

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

Current values from 10 patients
- Present: 5
- Absent: 3
- Missing: 2

Earlier failures affected 3 patients. Example failed answers: positive; present on CT. Their overlap with the current counts is unknown.
```

## Developer integration notes — excluded from the prompt

Python preserves the clinical target and roles, distinguishes current/historical count populations, and controls revisions and re-extraction.
