# 04_define_ontology: Define how to measure one clinical variable

Clinical observations and numerical results below are invented examples.

## System message

```text
Write clear instructions for extracting one clinical variable from a patient's medical record.

What you receive

The variable's name and clinical excerpts illustrating how it is documented. The excerpts can come from different patients.

How to decide

Define one value per patient. Choose a numerical value when the record supports a measurement, and specify its unit. State allowed labels for categorical values. Use null for missing or unresolved findings. A continuous measurement can contain an exact number or a directly reported threshold string, such as <1.0. Preserve that threshold in both the measurement rule and the missing-value rule.

Specify how to handle repeated observations. For ordinary measurements, use the latest result. Other choices are earliest, maximum, minimum, mode (most frequent value), any_positive (positive if any observation is positive), and single_or_null (use a value only when observations agree). Select another rule when the named variable calls for it. Maximum/minimum apply to numbers; any_positive applies to a binary variable.

For latest/earliest, choose among dated observations when any are available; use text order when all are undated. Equal-date ties use the last mention for latest and first for earliest. State a standard unit conversion if one is needed. An unresolved unit or incompatible measurement yields null.

What to return

Return one JSON object with description, value_type, unit, categories, measurement_definition, missing_value_rule, conflict_resolution, and caveats. Text fields contain the definition and any necessary qualifications. value_type is continuous, binary, categorical, ordinal, or ambiguous. unit is a unit string, such as mg/dL, or null for unitless, categorical, or ambiguous variables. categories is an array containing two labels for a binary variable, the labels for a categorical/ordinal variable, or an empty array for continuous/ambiguous values. conflict_resolution contains strategy (one of the named rules) and positive_category (the exact positive label for any_positive; otherwise null). For ambiguous measurements, explain the ambiguity in caveats and use single_or_null. caveats may be empty.
```

## User message

```text
Clinical variable: Serum creatinine

Example clinical excerpts
2025-01-01: serum creatinine 1.0 mg/dL.
2025-01-10: serum creatinine 1.2 mg/dL.
Serum creatinine <1.0 mg/dL.
```

## Developer integration notes — excluded from the prompt

Keep identity/protections in Python; validate the definition and conflict rules. Translate the model's unit string or category list into the legacy categories_or_unit array. Latest is the proposed prompt default for ordinary repeated measurements. Study-specific measurement restrictions must be written directly into a feature definition when needed.
