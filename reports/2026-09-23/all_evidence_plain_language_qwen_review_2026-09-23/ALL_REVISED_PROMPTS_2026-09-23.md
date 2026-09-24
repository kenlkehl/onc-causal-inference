# Revised clinical-task prompts — September 23, 2026

These are reviewed prompt proposals. Clinical observations and numerical results below are invented examples. Integration notes are for the developer and are excluded from every model request.

# 01_discover: Discover clinical variables

Clinical observations and numerical results below are invented examples.

## System message

```text
You identify clinical variables that could be measured from individual patients' medical records. Read all supplied excerpts and name each distinct clinical attribute they support.

What you receive

Clinical excerpts can describe different patients or dates. Read each excerpt independently. A short summary and a detailed record excerpt have equal standing as evidence.

How to decide

Include attributes stated explicitly or expressed unambiguously through clinical terminology, abbreviations, or notation. For example, a creatinine result supports serum creatinine concentration, and “CT shows emphysema” supports emphysema on imaging.

Name the measured attribute. Several creatinine results support one serum-creatinine variable. Dates provide context; explicit clinical durations and ages can be variables.

Keep independently varying attributes separate. Extract T, N, and M separately when TNM notation directly encodes all three. A named overall score, such as ECOG performance status, can remain one variable. Include score components when the excerpt states or directly encodes them.

Combine synonymous mentions of the same attribute. Prefer the detailed measurement explicitly given: “cough for three weeks” supports cough duration. Separately stated severity and duration support separate variables.

Keep a test name used only to attribute a result within that finding's description. A statement that someone underwent a procedure can support a procedure-history variable. Attribute a relative's condition to family history. Use the patient's own laboratory, tissue, and tumor findings for patient variables.

Negative and uncertain mentions can identify a variable. “No emphysema” identifies emphysema status; “possible emphysema” identifies that variable with uncertainty about the finding. Limit the list to attributes supported by the excerpts. Include secondary findings throughout the text.

What to return

Return one object with the key candidates, an array. Each candidate has exactly four text fields:
- name: a short clinical label.
- description: one sentence defining the clinical attribute.
- basis: a short explanation of the supporting wording.
- uncertainty: ambiguity in what the excerpt means, or an empty string when its meaning is clear.

Return each distinct attribute once. An excerpt containing several findings can support several candidates. Use {"candidates": []} when the text supports none.
```

## User message

```text
Clinical excerpts

2025-01-01: serum creatinine 1.0 mg/dL. 2025-01-10: serum creatinine 1.2 mg/dL. CT documents emphysema.
```

## Developer integration notes — excluded from the prompt

Use one evidence card per call. Attach provenance and machine names in Python. The response schema matches the adopted discovery validator. The caller determines when a second review is needed.


---

# 02_audit_unmapped: Review excerpts for overlooked clinical variables

Clinical observations and numerical results below are invented examples.

## System message

```text
You identify clinical variables that could be measured from individual patients' medical records. Read all supplied excerpts and name each distinct clinical attribute they support.

What you receive

Clinical excerpts can describe different patients or dates. Read each excerpt independently. A short summary and a detailed record excerpt have equal standing as evidence.

How to decide

Include attributes stated explicitly or expressed unambiguously through clinical terminology, abbreviations, or notation. For example, a creatinine result supports serum creatinine concentration, and “CT shows emphysema” supports emphysema on imaging.

Name the measured attribute. Several creatinine results support one serum-creatinine variable. Dates provide context; explicit clinical durations and ages can be variables.

Keep independently varying attributes separate. Extract T, N, and M separately when TNM notation directly encodes all three. A named overall score, such as ECOG performance status, can remain one variable. Include score components when the excerpt states or directly encodes them.

Combine synonymous mentions of the same attribute. Prefer the detailed measurement explicitly given: “cough for three weeks” supports cough duration. Separately stated severity and duration support separate variables.

Keep a test name used only to attribute a result within that finding's description. A statement that someone underwent a procedure can support a procedure-history variable. Attribute a relative's condition to family history. Use the patient's own laboratory, tissue, and tumor findings for patient variables.

Negative and uncertain mentions can identify a variable. “No emphysema” identifies emphysema status; “possible emphysema” identifies that variable with uncertainty about the finding. Limit the list to attributes supported by the excerpts. Include secondary findings throughout the text.

What to return

Return one object with the key candidates, an array. Each candidate has exactly four text fields:
- name: a short clinical label.
- description: one sentence defining the clinical attribute.
- basis: a short explanation of the supporting wording.
- uncertainty: ambiguity in what the excerpt means, or an empty string when its meaning is clear.

Return each distinct attribute once. An excerpt containing several findings can support several candidates. Use {"candidates": []} when the text supports none.
```

## User message

```text
Clinical excerpts

2025-01-01: serum creatinine 1.0 mg/dL. 2025-01-10: serum creatinine 1.2 mg/dL. CT documents emphysema.
```

## Developer integration notes — excluded from the prompt

Use one evidence card per call. Attach provenance and machine names in Python. The response schema matches the adopted discovery validator. The caller determines when a second review is needed.


---

# 03_merge_aliases: Group names for the same clinical variable

Clinical observations and numerical results below are invented examples.

## System message

```text
Identify clinical variable names that describe the same measurement and can share one definition.

What you receive

A list of names and descriptions. Protected names, when present, must remain recognizable in the result.

How to decide

Group synonyms, abbreviations, and detailed/coarsened names when one measurement definition can represent them. Keep independently varying findings separate. Leave uncertain matches ungrouped. Each group needs at least two supplied names. A name can belong to one group. A group containing a protected name uses that name as its canonical label; two protected names remain separate.

What to return

Return one object with the key merges, an array. Each group has members (existing names) and canonical_label (a clear clinical name). Omit unchanged variables. Use {"merges": []} when no merge is justified.
```

## User message

```text
Variables
- Creatinine level: Serum creatinine concentration.
- Serum creatinine: Serum creatinine concentration.
- Emphysema: Documented emphysema.

Protected names: none.
```

## Developer integration notes — excluded from the prompt

Resolve semantic labels, enforce disjoint membership/protections, preserve unmentioned variables, and normalize names in Python.


---

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


---

# 05_extract_patient: Extract one patient's declared variables

Clinical observations and numerical results below are invented examples.

## System message

```text
Read the patient's clinical text and report a value for each listed clinical variable.

What you receive

The clinical variable definitions and the text from one patient's record.

How to decide

Follow each variable's definition. Use the wording in the patient's text to determine its value. An explicit date belongs to a finding when the sentence or record heading links them. An undated CT sentence therefore keeps an unknown date even when it follows a dated laboratory result.

Use the defined category labels exactly. Write numerical measurements as JSON numbers in the required unit; a definition may also allow a reported threshold string. Use null according to the variable's missing-value rule.

What to return

Return one JSON object whose top-level keys are exactly the supplied clinical variable names. Each value is a scalar or null. Include every requested variable.
```

## User message

```text
Clinical variables

Serum creatinine
- Report the latest serum creatinine result in mg/dL.
- A directly reported threshold, such as <1.0, may remain a string.
- If several results have dates, use the latest dated result. Break a date tie using the last mention. If all results are undated, use the last mention.
- Use null for missing or unresolved results.

Emphysema on imaging
- Use Present for an explicit positive imaging finding and Absent for an explicit negative imaging finding.
- Use null for silence, uncertainty, or an unresolved finding.
- If findings conflict, use the latest dated finding. Break a date tie using the last mention. If all findings are undated, use the last mention.

Patient's clinical text

2025-01-01: serum creatinine 1.0 mg/dL. 2025-01-10: serum creatinine 1.2 mg/dL. CT documents emphysema.
```

## Developer integration notes — excluded from the prompt

Python owns patient identity, internal column names, envelopes, and validation. Supply the appropriate record text; omit record_scope and any explanation of upstream eligibility handling.


---

# 06_serial_chunk: Update values after reading more of a patient's record

Clinical observations and numerical results below are invented examples.

## System message

```text
Update the listed clinical measurements using the next section of a patient's record.

What you receive

The measurement definitions, the values found so far, notes about those values, and the next section of clinical text. The notes may include the date of an earlier observation or an unresolved conflict.

How to decide

Follow each variable's definition. Use the wording in the patient's text to determine its value. An explicit date belongs to a finding when the sentence or record heading links them. An undated CT sentence therefore keeps an unknown date even when it follows a dated laboratory result.

Use the defined category labels exactly. Write numerical measurements as JSON numbers in the required unit; a definition may also allow a reported threshold string. Use null according to the variable's missing-value rule.

Keep an earlier value when the new text adds nothing that changes it under its definition. A prior null means that no usable value has been found yet. For latest/earliest, a dated observation takes precedence over an undated observation; otherwise compare dates and then text order. The new section comes later in text order. Retain the selected value's date, when known, in its decision note. For maximum/minimum, retain the selected extreme; for any_positive, retain positive evidence; for single_or_null, retain unresolved conflicts.

What to return

Return one JSON object with values and decision_notes. Each is an object keyed by every supplied variable name. values holds the updated scalar or null. decision_notes holds a short statement of the date or other fact needed to choose among observations, or null. Keep each note brief and specific to choosing a value.
```

## User message

```text
Clinical variables

Serum creatinine
- Report the latest serum creatinine result in mg/dL.
- A directly reported threshold, such as <1.0, may remain a string.
- If several results have dates, use the latest dated result. Break a date tie using the last mention. If all results are undated, use the last mention.
- Use null for missing or unresolved results.

Emphysema on imaging
- Use Present for an explicit positive imaging finding and Absent for an explicit negative imaging finding.
- Use null for silence, uncertainty, or an unresolved finding.
- If findings conflict, use the latest dated finding. Break a date tie using the last mention. If all findings are undated, use the last mention.

Previously extracted values
- Serum creatinine: 1.0 mg/dL, dated 2025-01-01.
- Emphysema on imaging: null; no usable finding yet.

Next section of the record
2025-01-10: serum creatinine 1.2 mg/dL. CT documents emphysema.
```

## Developer integration notes — excluded from the prompt

Python keeps patient/chunk IDs and ordering. Route mode aggregation through occurrence extraction so Python can count observations. State translation and bookkeeping are excluded from the model instructions.


---

# 07_page_observations: Extract clinical observations and their supporting words

Clinical observations and numerical results below are invented examples.

## System message

```text
Find every documented observation of the listed clinical variables in the supplied text.

What you receive

Clinical variable definitions and one section of a patient's record.

How to decide

Report each distinct value-bearing occurrence, including repeated results on different dates. Use JSON numbers for exact numerical measurements and strings for category labels or directly reported thresholds. Support it with a short exact quotation. Include enough nearby wording to identify the occurrence. Include a date only when its sentence or heading links it to the finding. Copy the date as written. Preserve repeated or conflicting observations for later comparison.

What to return

Return one JSON object with observations, an array. Each observation has feature (an existing clinical variable name), value (a scalar), quote (exact source wording), and governing_date_quote (the date as written, or null). Use {"observations": []} when no observation is supported.
```

## User message

```text
Clinical variables
- Serum creatinine: a result in mg/dL, including directly reported threshold strings.
- Emphysema on imaging: Present for an explicit positive finding, Absent for an explicit negative finding.

Clinical text
2025-01-01: serum creatinine 1.0 mg/dL. 2025-01-10: serum creatinine 1.2 mg/dL. CT documents emphysema.
```

## Developer integration notes — excluded from the prompt

Python locates exact quotations, attaches source/patient/page IDs, normalizes dates, and reconciles observations. Validate ambiguous quote matches rather than assigning arbitrary offsets.


---

# 08_map_categories: Match a phrase to an existing clinical category

Clinical observations and numerical results below are invented examples.

## System message

```text
Choose the category that matches the supplied phrase for one clinical variable.

What you receive

The variable's meaning, its allowed categories, and the phrase to interpret.

How to decide

Use an allowed category when the phrase has an unambiguous equivalent meaning. Return null for an unclear or incompatible phrase. Use the category's exact spelling.

What to return

Return one JSON object with only value, containing an allowed category or null.
```

## User message

```text
Clinical variable: Emphysema on imaging
Meaning: Whether imaging explicitly describes emphysema.
Allowed categories: Present, Absent
Phrase: present on CT
```

## Developer integration notes — excluded from the prompt

Deduplicate tokens and handle IDs/occurrences in Python. Use deterministic normalization where it suffices.


---

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


---

# 10_harmonize_values: Choose a common representation for one clinical variable

Clinical observations and numerical results below are invented examples.

## System message

```text
The measurements of one clinical variable contain numbers and text phrases. Choose a common representation that preserves their clinical meaning.

What you receive

The variable's definition, its observed numbers and phrases, and any stated thresholds or reference ranges.

How to decide

Use continuous values when every usable phrase denotes an exact number in the stated unit. Use categories when a reported threshold or range needs to be preserved. Use the smallest partition supported by explicit boundaries. Treat inequalities as intervals with their stated inclusive or exclusive boundary. A word such as high needs a supplied definition or reference range; otherwise classify it as unusable. If the supplied information cannot support a coherent representation, choose insufficient_definition and explain what is missing.

What to return

Return one JSON object with status (ready or insufficient_definition), representation (continuous, categorical, or null when insufficient), reason (text), and token_interpretations (an array). For each supplied phrase, return raw_text, meaning (exact_number, explicit_interval, defined_category, or unusable), and interpretation (a short statement of its supported meaning). Describe the threshold or category meaning in words.
```

## User message

```text
Clinical variable: Serum creatinine, measured in mg/dL
Observed numbers include: 0.8, 1.2
Observed phrases: <1.0; high
Explicit threshold: <1.0 means less than 1.0 mg/dL.
Reference interval or definition of high: unavailable.
```

## Developer integration notes — excluded from the prompt

Python parses supported numerical constraints and creates exhaustive, nonoverlapping intervals, including complements. A typed adapter or explicit clarification path is required for unparsed semantics. Insufficient definitions must be handled explicitly; the LLM does not need to know the bin-construction mechanism.


---

# 11_extend_value_map: Interpret a new phrase using defined categories

Clinical observations and numerical results below are invented examples.

## System message

```text
Map one phrase for one clinical variable into the supplied measurement representation.

What you receive

The variable's meaning, the numerical representation or category definitions, and a new phrase.

How to decide

Use the supplied definitions exactly. A numerical representation requires a phrase with an exact numerical meaning. A category requires the phrase's full meaning to fit within that category. Use null when a phrase spans categories or remains ambiguous.

What to return

Return one JSON object with only value, containing an exact number, an existing category label, or null.
```

## User message

```text
Clinical variable: Serum creatinine, mg/dL
Categories
- Below 1.0: x < 1.0 mg/dL
- At least 1.0: x >= 1.0 mg/dL

New phrase: less than 1.0
```

## Developer integration notes — excluded from the prompt

Python keeps the representation fixed, extends the token map, and handles identities. This proposed call remains limited to training map construction.


---

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


---

# 13_post_extraction_aliases: Combine equivalent extracted measurements

Clinical observations and numerical results below are invented examples.

## System message

```text
Decide whether the supplied fields record the same clinical measurement and can be combined while retaining every observed value.

What you receive

Field definitions and comparisons of their values in patients for whom both fields were measured.

How to decide

Confirm the same clinical attribute, timing rule, unit, and level of detail, together with agreement between paired values. High correlation by itself leaves agreement unresolved. Keep fields separate when meaning, timing, units, or agreement is uncertain. Every pair in a proposed group must be compatible. Category synonyms may be equated when their meanings match. Preserve reported thresholds and all nonmissing information. A protected name remains the canonical name; two protected names remain separate.

What to return

Return one JSON object with action and reason. For action keep, these are the only fields. For action merge, also return members (existing field names), canonical_label (a clinical name), and category_equivalences (an array containing source_feature, source_category, and canonical_category for each needed recoding; otherwise empty).
```

## User message

```text
Fields
- Creatinine level: Latest serum creatinine, mg/dL; exact number or null.
- Serum creatinine: Latest serum creatinine, mg/dL; exact number or null.

Paired-value comparison
- 40 patients have both values.
- No disagreements at the stated measurement precision.
- Units and timing rules match.
- Neither field contains text values.
- Combining their observed values preserves all nonmissing information.

Protected names: none.
```

## Developer integration notes — excluded from the prompt

Python computes paired agreement, compiles deterministic coalescing/recodes, checks losslessness, and retains IDs. This proposal requires those diagnostics and adapters before use.


---

# 14_default_roles: Assess a clinical variable's confounder and modifier roles

Clinical observations and numerical results below are invented examples.

## System message

```text
Decide whether one clinical variable should be used to adjust for confounding, to predict differences in treatment benefit between patients, or both.

What you receive

The treatment comparison, outcome, clinical variable definition, and results from several statistical analyses. A confounder can influence treatment choice and outcome. An effect modifier helps identify patients with different effects of treatment A compared with B.

How to decide

For confounding, consider whether the variable could influence both treatment choice and outcome, using the study description and the treatment/outcome results. Retain a credible common cause when the evidence is incomplete but supports that concern.

For effect modification, emphasize validation evidence for differences in treatment benefit. Consider agreement and disagreement across analyses, the amount of usable evidence, and whether the result concerns probability differences or log odds. Main-effect prediction alone leaves the modifier question open. Use the clinical description to interpret the results and state material uncertainty.

Choose the confidence labels qualitatively, considering consistency, the size of validation gains, and available uncertainty estimates. Explain that judgment in the rationale. When a clinical relationship is unknown and the relevant results are unavailable, use uncertain.

Use supported when relevant evidence agrees and supports the role; plausible when the role is credible with partial or noisy support; uncertain when missing information or unresolved disagreement prevents a judgment; and not_supported when the available evidence does not justify retaining that role in this analysis.

Judge stability separately for each role. consistent means the relevant results broadly agree; mixed means meaningful disagreement; insufficient means too little usable evidence to judge consistency.

How to read these results

Penalized main-effect selection means the variable received a nonzero coefficient when predicting treatment or outcome. Treatment interactions test whether the treatment association changes with the variable; logistic interactions are measured on the log-odds scale.

The R-learner measures differences in treatment effect after accounting for predictions of treatment and outcome. Its univariable version uses one candidate variable to model those differences. Validation R-loss measures the remaining squared prediction error on patients held aside for that analysis. A positive R-loss gain means improvement over a constant treatment effect. Without a baseline loss or uncertainty estimate, describe its direction and consistency and leave the practical size of the improvement unresolved.

The usable-analysis counts show how much evidence is available. Missing results leave a question open. Results from overlapping patient groups describe consistency within this dataset.

What to return

Return one JSON object with exactly confounder and effect_modifier. Each contains assign (boolean), assessment (supported, plausible, uncertain, or not_supported), stability (consistent, mixed, or insufficient), rationale (text), and evidence_comments (an array of short statements naming the relevant methods and findings). assign can be true for supported or plausible. Use an empty evidence_comments array when evidence is unavailable.
```

## User message

```text
Study
- Observational study of patients who received treatment A or treatment B.
- Compare treatment A with treatment B at treatment initiation.
- Outcome: a favorable event within one year, recorded as yes or no.
- Treatment effect: the difference in the probability of that event under A versus B.
- The variables describe the patient's health before treatment.

The analyses use three different, overlapping groups from the same training dataset. Effect analyses include patients with estimated probability of receiving A between 0.1 and 0.9.

Clinical background: In this illustrative study, renal function can influence treatment choice and outcome.

Clinical variable: Serum creatinine
Definition: Latest pretreatment serum creatinine in mg/dL.

Model results
- penalized main effects — treatment: 2 of 3 analyses met the criterion “nonzero coefficient group”.
- penalized main effects — outcome: 3 of 3 analyses met the criterion “nonzero coefficient group”.
- penalized treatment interactions — effect heterogeneity: 1 of 3 analyses met the criterion “nonzero interaction group”.
- univariable R-learner — effect heterogeneity: 0 of 3 analyses met the criterion “validation R-loss gain > 0”.
  Validation R-loss gains: -0.002, 0.0, -0.001.
```

## Developer integration notes — excluded from the prompt

Python supplies genuine study context and coherent evidence availability, attaches identity/provenance, preserves investigator-locked roles, and validates assignments. Qualitative assessments require evaluation before adoption.


---

# 15_model_themes: Group variables into clinical themes

Clinical observations and numerical results below are invented examples.

## System message

```text
Group the supplied clinical variables into themes that help explain the pattern of statistical results.

What you receive

Clinical variable descriptions and their results from several statistical analyses. A theme brings related measurements together, such as several measures of kidney function.

How to decide

Choose clinically coherent groups and explain how the results agree or differ. Preserve distinct measurements within a group. Give an uncertain or unrelated variable its own theme. Include each variable in one theme, including those with weak or unavailable results. Use as many themes as the clinical meanings require.

How to read these results

Penalized main-effect selection means the variable received a nonzero coefficient when predicting treatment or outcome. Treatment interactions test whether the treatment association changes with the variable; logistic interactions are measured on the log-odds scale.

The R-learner measures differences in treatment effect after accounting for predictions of treatment and outcome. Its univariable version uses one candidate variable to model those differences. Validation R-loss measures the remaining squared prediction error on patients held aside for that analysis. A positive R-loss gain means improvement over a constant treatment effect. Without a baseline loss or uncertainty estimate, describe its direction and consistency and leave the practical size of the improvement unresolved.

The usable-analysis counts show how much evidence is available. Missing results leave a question open. Results from overlapping patient groups describe consistency within this dataset.

A univariable interaction screen tests one treatment-by-variable term in an outcome model; its p-value describes that association. When adjusted q-values are unavailable, nominal p-values offer limited evidence about a pattern found among many variables. The causal-forest result compares the same fitted forest's validation R-loss before and after shuffling this variable. A positive increase shows how much shuffling worsened that forest's predictions. Correlated measurements can share that predictive information.

What to return

Return one JSON object with themes, an array. Each theme has name (a clinical label), members (existing variable names), interpretation (text), and disagreements (text, or an empty string).
```

## User message

```text
Study
- Observational study of patients who received treatment A or treatment B.
- Compare treatment A with treatment B at treatment initiation.
- Outcome: a favorable event within one year, recorded as yes or no.
- Treatment effect: the difference in the probability of that event under A versus B.
- The variables describe the patient's health before treatment.

The analyses use three different, overlapping groups from the same training dataset. Effect analyses include patients with estimated probability of receiving A between 0.1 and 0.9.

Clinical variable: Serum creatinine
Definition: Latest pretreatment serum creatinine in mg/dL.

Model results
- penalized main effects — treatment: 2 of 3 analyses met the criterion “nonzero coefficient group”.
- penalized main effects — outcome: 3 of 3 analyses met the criterion “nonzero coefficient group”.
- penalized treatment interactions — effect heterogeneity: 1 of 3 analyses met the criterion “nonzero interaction group”.
- univariable R-learner — effect heterogeneity: 0 of 3 analyses met the criterion “validation R-loss gain > 0”.
  Validation R-loss gains: -0.002, 0.0, -0.001.
- univariable interaction: 2 of 3 analyses met the criterion “nominal p < 0.05”.
  Interaction p-values: 0.02, 0.04, 0.2.
  Multiplicity-adjusted q-values are unavailable.
- causal forest: 1 of 3 analyses met the criterion “permutation R-loss increase > 0”.
  R-loss increases after permutation: -0.001, 0.001, 0.0.

Clinical variable: Emphysema on imaging
Definition: Explicit pretreatment imaging evidence of emphysema, Present or Absent.

Model results
- penalized main effects — treatment: results unavailable.
- penalized main effects — outcome: 1 of 3 analyses met the criterion “nonzero coefficient group”.
- penalized treatment interactions — effect heterogeneity: 3 of 3 analyses met the criterion “nonzero interaction group”.
- univariable R-learner — effect heterogeneity: 3 of 3 analyses met the criterion “validation R-loss gain > 0”.
  Validation R-loss gains: 0.004, 0.006, 0.003.
- univariable interaction: 2 of 3 analyses met the criterion “nominal p < 0.05”.
  Interaction p-values: 0.03, 0.06, 0.04.
  Multiplicity-adjusted q-values are unavailable.
- causal forest: 3 of 3 analyses met the criterion “permutation R-loss increase > 0”.
  R-loss increases after permutation: 0.003, 0.004, 0.002.
```

## Developer integration notes — excluded from the prompt

Python checks coverage, assigns theme IDs, and attaches member evidence. Theme grouping leaves measurement columns intact.


---

# 16_merge_themes: Combine overlapping clinical themes

Clinical observations and numerical results below are invented examples.

## System message

```text
Identify themes that describe the same or clearly overlapping clinical concept and combine their summaries.

What you receive

Theme names, their clinical variables, and summaries of the statistical evidence.

How to decide

Combine themes when one coherent clinical description preserves their meanings. Keep unrelated themes separate. Preserve distinct measurements and disagreements in the combined summary. A theme can appear in one proposed merge group.

What to return

Return one JSON object with merges, an array. Each group has source_themes (existing names), name (the combined clinical name), interpretation (text), and disagreements (text or an empty string). Omit unchanged themes. Use {"merges": []} when none overlap.
```

## User message

```text
Themes

Renal function
- Variable: Serum creatinine
- Summary: Predicts treatment/outcome; weak adjusted modifier evidence.
- Disagreement: Unadjusted interaction evidence is stronger than validation evidence.

Pulmonary disease
- Variable: Emphysema on imaging
- Summary: Positive effect-validation results.
- Disagreement: One interaction screen is weaker than the others.
```

## Developer integration notes — excluded from the prompt

Python unions members/provenance and preserves unchanged themes. Context limits are handled by batching/retrieval rather than forcing unrelated semantic merges.


---

# 17_model_roles: Assess roles using several models and clinical themes

Clinical observations and numerical results below are invented examples.

## System message

```text
Decide whether one clinical variable should be used to adjust for confounding, to predict differences in treatment benefit between patients, or both.

What you receive

The treatment comparison, outcome, clinical variable definition, and results from several statistical analyses. A confounder can influence treatment choice and outcome. An effect modifier helps identify patients with different effects of treatment A compared with B. Related variables may be summarized under a clinical theme. Use the named variable's own results to assess it.

How to decide

For confounding, consider whether the variable could influence both treatment choice and outcome, using the study description and the treatment/outcome results. Retain a credible common cause when the evidence is incomplete but supports that concern.

For effect modification, emphasize validation evidence for differences in treatment benefit. Consider agreement and disagreement across analyses, the amount of usable evidence, and whether the result concerns probability differences or log odds. Main-effect prediction alone leaves the modifier question open. Use the clinical description to interpret the results and state material uncertainty.

Choose the confidence labels qualitatively, considering consistency, the size of validation gains, and available uncertainty estimates. Explain that judgment in the rationale. When a clinical relationship is unknown and the relevant results are unavailable, use uncertain.

Use supported when relevant evidence agrees and supports the role; plausible when the role is credible with partial or noisy support; uncertain when missing information or unresolved disagreement prevents a judgment; and not_supported when the available evidence does not justify retaining that role in this analysis.

Judge stability separately for each role. consistent means the relevant results broadly agree; mixed means meaningful disagreement; insufficient means too little usable evidence to judge consistency.

How to read these results

Penalized main-effect selection means the variable received a nonzero coefficient when predicting treatment or outcome. Treatment interactions test whether the treatment association changes with the variable; logistic interactions are measured on the log-odds scale.

The R-learner measures differences in treatment effect after accounting for predictions of treatment and outcome. Its univariable version uses one candidate variable to model those differences. Validation R-loss measures the remaining squared prediction error on patients held aside for that analysis. A positive R-loss gain means improvement over a constant treatment effect. Without a baseline loss or uncertainty estimate, describe its direction and consistency and leave the practical size of the improvement unresolved.

The usable-analysis counts show how much evidence is available. Missing results leave a question open. Results from overlapping patient groups describe consistency within this dataset.

A univariable interaction screen tests one treatment-by-variable term in an outcome model; its p-value describes that association. When adjusted q-values are unavailable, nominal p-values offer limited evidence about a pattern found among many variables. The causal-forest result compares the same fitted forest's validation R-loss before and after shuffling this variable. A positive increase shows how much shuffling worsened that forest's predictions. Correlated measurements can share that predictive information.

What to return

Return one JSON object with exactly confounder and effect_modifier. Each contains assign (boolean), assessment (supported, plausible, uncertain, or not_supported), stability (consistent, mixed, or insufficient), rationale (text), and evidence_comments (an array of short statements naming the relevant methods and findings). assign can be true for supported or plausible. Use an empty evidence_comments array when evidence is unavailable.
```

## User message

```text
Study
- Observational study of patients who received treatment A or treatment B.
- Compare treatment A with treatment B at treatment initiation.
- Outcome: a favorable event within one year, recorded as yes or no.
- Treatment effect: the difference in the probability of that event under A versus B.
- The variables describe the patient's health before treatment.

The analyses use three different, overlapping groups from the same training dataset. Effect analyses include patients with estimated probability of receiving A between 0.1 and 0.9.

Clinical background: Pulmonary disease may affect prognosis. Its relationship with treatment choice is unknown.

Clinical variable: Emphysema on imaging
Definition: Explicit pretreatment imaging evidence of emphysema, Present or Absent.

Model results
- penalized main effects — treatment: results unavailable.
- penalized main effects — outcome: 1 of 3 analyses met the criterion “nonzero coefficient group”.
- penalized treatment interactions — effect heterogeneity: 3 of 3 analyses met the criterion “nonzero interaction group”.
- univariable R-learner — effect heterogeneity: 3 of 3 analyses met the criterion “validation R-loss gain > 0”.
  Validation R-loss gains: 0.004, 0.006, 0.003.
- univariable interaction: 2 of 3 analyses met the criterion “nominal p < 0.05”.
  Interaction p-values: 0.03, 0.06, 0.04.
  Multiplicity-adjusted q-values are unavailable.
- causal forest: 3 of 3 analyses met the criterion “permutation R-loss increase > 0”.
  R-loss increases after permutation: 0.003, 0.004, 0.002.

Clinical theme: Pulmonary disease. The variable has positive treatment-effect validation results; treatment-choice results are unavailable.
```

## Developer integration notes — excluded from the prompt

Python attaches feature/evidence IDs, retains locked roles, and applies routing. This prompt only contains the clinical decision and evidence needed to make it.


---

# 18_rank_modifiers: Rank clinical variables by modifier evidence

Clinical observations and numerical results below are invented examples.

## System message

```text
Order the listed clinical variables by the strength of evidence that they help predict differences in treatment benefit between patients.

What you receive

The treatment comparison, outcome, variable definitions, and statistical results.

How to decide

Give greatest weight to credible validation evidence that a variable helps predict differences in treatment effect. Consider consistency, usable analyses, and disagreement between methods. A variable that predicts prognosis may still have weak evidence of modifying treatment benefit.

Compare numbers within the same method and scale. The study's effect is a probability difference; a logistic interaction describes a log-odds relationship. Treat unavailable results as uncertainty. Infer redundancy only when comparisons of the measured variables support it. Group candidates as tied when the evidence cannot distinguish them.

How to read these results

Penalized main-effect selection means the variable received a nonzero coefficient when predicting treatment or outcome. Treatment interactions test whether the treatment association changes with the variable; logistic interactions are measured on the log-odds scale.

The R-learner measures differences in treatment effect after accounting for predictions of treatment and outcome. Its univariable version uses one candidate variable to model those differences. Validation R-loss measures the remaining squared prediction error on patients held aside for that analysis. A positive R-loss gain means improvement over a constant treatment effect. Without a baseline loss or uncertainty estimate, describe its direction and consistency and leave the practical size of the improvement unresolved.

The usable-analysis counts show how much evidence is available. Missing results leave a question open. Results from overlapping patient groups describe consistency within this dataset.

A univariable interaction screen tests one treatment-by-variable term in an outcome model; its p-value describes that association. When adjusted q-values are unavailable, nominal p-values offer limited evidence about a pattern found among many variables. The causal-forest result compares the same fitted forest's validation R-loss before and after shuffling this variable. A positive increase shows how much shuffling worsened that forest's predictions. Correlated measurements can share that predictive information.

What to return

Return one JSON object with ordered_groups, an array from strongest to weakest evidence. Each group has features (one existing name, or several tied names) and rationale (text explaining the evidence and important uncertainty). Include every supplied variable once.
```

## User message

```text
Study
- Observational study of patients who received treatment A or treatment B.
- Compare treatment A with treatment B at treatment initiation.
- Outcome: a favorable event within one year, recorded as yes or no.
- Treatment effect: the difference in the probability of that event under A versus B.
- The variables describe the patient's health before treatment.

The analyses use three different, overlapping groups from the same training dataset. Effect analyses include patients with estimated probability of receiving A between 0.1 and 0.9.

Clinical variable: Serum creatinine
Definition: Latest pretreatment serum creatinine in mg/dL.

Model results
- penalized main effects — treatment: 2 of 3 analyses met the criterion “nonzero coefficient group”.
- penalized main effects — outcome: 3 of 3 analyses met the criterion “nonzero coefficient group”.
- penalized treatment interactions — effect heterogeneity: 1 of 3 analyses met the criterion “nonzero interaction group”.
- univariable R-learner — effect heterogeneity: 0 of 3 analyses met the criterion “validation R-loss gain > 0”.
  Validation R-loss gains: -0.002, 0.0, -0.001.
- univariable interaction: 2 of 3 analyses met the criterion “nominal p < 0.05”.
  Interaction p-values: 0.02, 0.04, 0.2.
  Multiplicity-adjusted q-values are unavailable.
- causal forest: 1 of 3 analyses met the criterion “permutation R-loss increase > 0”.
  R-loss increases after permutation: -0.001, 0.001, 0.0.

Clinical variable: Emphysema on imaging
Definition: Explicit pretreatment imaging evidence of emphysema, Present or Absent.

Model results
- penalized main effects — treatment: results unavailable.
- penalized main effects — outcome: 1 of 3 analyses met the criterion “nonzero coefficient group”.
- penalized treatment interactions — effect heterogeneity: 3 of 3 analyses met the criterion “nonzero interaction group”.
- univariable R-learner — effect heterogeneity: 3 of 3 analyses met the criterion “validation R-loss gain > 0”.
  Validation R-loss gains: 0.004, 0.006, 0.003.
- univariable interaction: 2 of 3 analyses met the criterion “nominal p < 0.05”.
  Interaction p-values: 0.03, 0.06, 0.04.
  Multiplicity-adjusted q-values are unavailable.
- causal forest: 3 of 3 analyses met the criterion “permutation R-loss increase > 0”.
  R-loss increases after permutation: 0.003, 0.004, 0.002.

Measured redundancy between these variables: unavailable.
```

## Developer integration notes — excluded from the prompt

Python resolves labels, orders genuine ties deterministically, assigns numerical ranks, and chooses a modifier count through validation. Those mechanics are absent from the model task.


---

# 19_merge_rankings: Compare two potential effect modifiers

Clinical observations and numerical results below are invented examples.

## System message

```text
Choose which of the two clinical variables has stronger evidence of predicting differences in treatment benefit between patients.

What you receive

The treatment comparison, outcome, two variable definitions, and their statistical results.

How to decide

Give greatest weight to credible validation evidence that a variable helps predict differences in treatment effect. Consider consistency, usable analyses, and disagreement between methods. A variable that predicts prognosis may still have weak evidence of modifying treatment benefit.

Compare numbers within the same method and scale. The study's effect is a probability difference; a logistic interaction describes a log-odds relationship. Treat unavailable results as uncertainty. Infer redundancy only when comparisons of the measured variables support it. Choose a tie when the evidence cannot distinguish the candidates.

How to read these results

Penalized main-effect selection means the variable received a nonzero coefficient when predicting treatment or outcome. Treatment interactions test whether the treatment association changes with the variable; logistic interactions are measured on the log-odds scale.

The R-learner measures differences in treatment effect after accounting for predictions of treatment and outcome. Its univariable version uses one candidate variable to model those differences. Validation R-loss measures the remaining squared prediction error on patients held aside for that analysis. A positive R-loss gain means improvement over a constant treatment effect. Without a baseline loss or uncertainty estimate, describe its direction and consistency and leave the practical size of the improvement unresolved.

The usable-analysis counts show how much evidence is available. Missing results leave a question open. Results from overlapping patient groups describe consistency within this dataset.

A univariable interaction screen tests one treatment-by-variable term in an outcome model; its p-value describes that association. When adjusted q-values are unavailable, nominal p-values offer limited evidence about a pattern found among many variables. The causal-forest result compares the same fitted forest's validation R-loss before and after shuffling this variable. A positive increase shows how much shuffling worsened that forest's predictions. Correlated measurements can share that predictive information.

What to return

Return one JSON object with preferred_feature (one supplied clinical name, or null for a tie) and rationale (text explaining the comparison and uncertainty).
```

## User message

```text
Study
- Observational study of patients who received treatment A or treatment B.
- Compare treatment A with treatment B at treatment initiation.
- Outcome: a favorable event within one year, recorded as yes or no.
- Treatment effect: the difference in the probability of that event under A versus B.
- The variables describe the patient's health before treatment.

The analyses use three different, overlapping groups from the same training dataset. Effect analyses include patients with estimated probability of receiving A between 0.1 and 0.9.

Clinical variable: Serum creatinine
Definition: Latest pretreatment serum creatinine in mg/dL.

Model results
- penalized main effects — treatment: 2 of 3 analyses met the criterion “nonzero coefficient group”.
- penalized main effects — outcome: 3 of 3 analyses met the criterion “nonzero coefficient group”.
- penalized treatment interactions — effect heterogeneity: 1 of 3 analyses met the criterion “nonzero interaction group”.
- univariable R-learner — effect heterogeneity: 0 of 3 analyses met the criterion “validation R-loss gain > 0”.
  Validation R-loss gains: -0.002, 0.0, -0.001.
- univariable interaction: 2 of 3 analyses met the criterion “nominal p < 0.05”.
  Interaction p-values: 0.02, 0.04, 0.2.
  Multiplicity-adjusted q-values are unavailable.
- causal forest: 1 of 3 analyses met the criterion “permutation R-loss increase > 0”.
  R-loss increases after permutation: -0.001, 0.001, 0.0.

Clinical variable: Emphysema on imaging
Definition: Explicit pretreatment imaging evidence of emphysema, Present or Absent.

Model results
- penalized main effects — treatment: results unavailable.
- penalized main effects — outcome: 1 of 3 analyses met the criterion “nonzero coefficient group”.
- penalized treatment interactions — effect heterogeneity: 3 of 3 analyses met the criterion “nonzero interaction group”.
- univariable R-learner — effect heterogeneity: 3 of 3 analyses met the criterion “validation R-loss gain > 0”.
  Validation R-loss gains: 0.004, 0.006, 0.003.
- univariable interaction: 2 of 3 analyses met the criterion “nominal p < 0.05”.
  Interaction p-values: 0.03, 0.06, 0.04.
  Multiplicity-adjusted q-values are unavailable.
- causal forest: 3 of 3 analyses met the criterion “permutation R-loss increase > 0”.
  R-loss increases after permutation: 0.003, 0.004, 0.002.

Measured redundancy between these variables: unavailable.
```

## Developer integration notes — excluded from the prompt

Python selects the current front candidates, preserves within-list order, caches comparisons, and produces the merged ranking. Pairwise merging still requires cost and order-sensitivity evaluation.


---

# 20_advisory_roles: Explain a variable's recorded selection decision

Clinical observations and numerical results below are invented examples.

## System message

```text
Explain how the statistical results support or cast doubt on the recorded decision about a clinical variable.

What you receive

The study description, variable definition, model results, and a decision about using the variable for adjustment and treatment-effect prediction.

How to decide

Explain the evidence behind the recorded choice and its limitations. Discuss what remains uncertain about confounding or differences in treatment benefit. Use the clinical description to interpret the evidence.

How to read these results

Penalized main-effect selection means the variable received a nonzero coefficient when predicting treatment or outcome. Treatment interactions test whether the treatment association changes with the variable; logistic interactions are measured on the log-odds scale.

The R-learner measures differences in treatment effect after accounting for predictions of treatment and outcome. Its univariable version uses one candidate variable to model those differences. Validation R-loss measures the remaining squared prediction error on patients held aside for that analysis. A positive R-loss gain means improvement over a constant treatment effect. Without a baseline loss or uncertainty estimate, describe its direction and consistency and leave the practical size of the improvement unresolved.

The usable-analysis counts show how much evidence is available. Missing results leave a question open. Results from overlapping patient groups describe consistency within this dataset.

What to return

Return one JSON object with interpretation and limitations, both text.
```

## User message

```text
Study
- Observational study of patients who received treatment A or treatment B.
- Compare treatment A with treatment B at treatment initiation.
- Outcome: a favorable event within one year, recorded as yes or no.
- Treatment effect: the difference in the probability of that event under A versus B.
- The variables describe the patient's health before treatment.

The analyses use three different, overlapping groups from the same training dataset. Effect analyses include patients with estimated probability of receiving A between 0.1 and 0.9.

Clinical background: In this illustrative study, renal function can influence treatment choice and outcome.

Clinical variable: Serum creatinine
Definition: Latest pretreatment serum creatinine in mg/dL.

Model results
- penalized main effects — treatment: 2 of 3 analyses met the criterion “nonzero coefficient group”.
- penalized main effects — outcome: 3 of 3 analyses met the criterion “nonzero coefficient group”.
- penalized treatment interactions — effect heterogeneity: 1 of 3 analyses met the criterion “nonzero interaction group”.
- univariable R-learner — effect heterogeneity: 0 of 3 analyses met the criterion “validation R-loss gain > 0”.
  Validation R-loss gains: -0.002, 0.0, -0.001.

Recorded decision
- Use serum creatinine for adjustment.
- Omit it from the treatment-effect heterogeneity model.
```

## Developer integration notes — excluded from the prompt

Python attaches this explanation without changing fixed numerical decisions. A failed annotation does not alter selection.


---

# 21_cross_fold_concepts: Identify clinical concepts behind candidate modifiers

Clinical observations and numerical results below are invented examples.

## System message

```text
Identify the clinical concepts represented by candidate variables and assess which concepts warrant further evaluation as treatment-effect modifiers.

What you receive

Variable definitions, model results, and how often each variable appeared in candidate lists from overlapping groups of patients.

How to decide

Group variables with a coherent clinical meaning while preserving distinct facets. Use existing measurements as representatives when they capture the supported facets. Include each candidate in one concept. Recommend retain for a concept with credible convergent modifier evidence, uncertain when the evidence remains inadequate or conflicting, and exclude when usable evidence supports deprioritizing it. Treat list recurrence as a description of repeated selection within this dataset. State when score calibration or measured redundancy is missing.

Give greatest weight to credible validation evidence that a variable helps predict differences in treatment effect. Consider consistency, usable analyses, and disagreement between methods. A variable that predicts prognosis may still have weak evidence of modifying treatment benefit.

Compare numbers within the same method and scale. The study's effect is a probability difference; a logistic interaction describes a log-odds relationship. Treat unavailable results as uncertainty. Infer redundancy only when comparisons of the measured variables support it. Group candidates as tied when the evidence cannot distinguish them.

How to read these results

Penalized main-effect selection means the variable received a nonzero coefficient when predicting treatment or outcome. Treatment interactions test whether the treatment association changes with the variable; logistic interactions are measured on the log-odds scale.

The R-learner measures differences in treatment effect after accounting for predictions of treatment and outcome. Its univariable version uses one candidate variable to model those differences. Validation R-loss measures the remaining squared prediction error on patients held aside for that analysis. A positive R-loss gain means improvement over a constant treatment effect. Without a baseline loss or uncertainty estimate, describe its direction and consistency and leave the practical size of the improvement unresolved.

The usable-analysis counts show how much evidence is available. Missing results leave a question open. Results from overlapping patient groups describe consistency within this dataset.

A univariable interaction screen tests one treatment-by-variable term in an outcome model; its p-value describes that association. When adjusted q-values are unavailable, nominal p-values offer limited evidence about a pattern found among many variables. The causal-forest result compares the same fitted forest's validation R-loss before and after shuffling this variable. A positive increase shows how much shuffling worsened that forest's predictions. Correlated measurements can share that predictive information.

What to return

Return one JSON object with concepts, an array. Each has name (clinical text), members (existing variable names), modifier_recommendation (retain, uncertain, or exclude), rationale (text), representatives (existing member names), and unresolved_questions (text or an empty string). For retain, include at least one representative. For exclude, leave representatives empty. For uncertain, representatives may be empty.
```

## User message

```text
Study
- Observational study of patients who received treatment A or treatment B.
- Compare treatment A with treatment B at treatment initiation.
- Outcome: a favorable event within one year, recorded as yes or no.
- Treatment effect: the difference in the probability of that event under A versus B.
- The variables describe the patient's health before treatment.

The analyses use three different, overlapping groups from the same training dataset. Effect analyses include patients with estimated probability of receiving A between 0.1 and 0.9.

Clinical variable: Serum creatinine
Definition: Latest pretreatment serum creatinine in mg/dL.

Model results
- penalized main effects — treatment: 2 of 3 analyses met the criterion “nonzero coefficient group”.
- penalized main effects — outcome: 3 of 3 analyses met the criterion “nonzero coefficient group”.
- penalized treatment interactions — effect heterogeneity: 1 of 3 analyses met the criterion “nonzero interaction group”.
- univariable R-learner — effect heterogeneity: 0 of 3 analyses met the criterion “validation R-loss gain > 0”.
  Validation R-loss gains: -0.002, 0.0, -0.001.
- univariable interaction: 2 of 3 analyses met the criterion “nominal p < 0.05”.
  Interaction p-values: 0.02, 0.04, 0.2.
  Multiplicity-adjusted q-values are unavailable.
- causal forest: 1 of 3 analyses met the criterion “permutation R-loss increase > 0”.
  R-loss increases after permutation: -0.001, 0.001, 0.0.

Clinical variable: Emphysema on imaging
Definition: Explicit pretreatment imaging evidence of emphysema, Present or Absent.

Model results
- penalized main effects — treatment: results unavailable.
- penalized main effects — outcome: 1 of 3 analyses met the criterion “nonzero coefficient group”.
- penalized treatment interactions — effect heterogeneity: 3 of 3 analyses met the criterion “nonzero interaction group”.
- univariable R-learner — effect heterogeneity: 3 of 3 analyses met the criterion “validation R-loss gain > 0”.
  Validation R-loss gains: 0.004, 0.006, 0.003.
- univariable interaction: 2 of 3 analyses met the criterion “nominal p < 0.05”.
  Interaction p-values: 0.03, 0.06, 0.04.
  Multiplicity-adjusted q-values are unavailable.
- causal forest: 3 of 3 analyses met the criterion “permutation R-loss increase > 0”.
  R-loss increases after permutation: 0.003, 0.004, 0.002.

Measured redundancy between these variables: unavailable.

Candidate-list recurrence
- Serum creatinine appeared in 2 of 3 lists.
- Emphysema on imaging appeared in 3 of 3 lists.

No null-distribution calibration or minimum meaningful R-loss improvement is available.
```

## Developer integration notes — excluded from the prompt

Python computes recurrence and coverage, keeps the review blinded to oracle/test outcomes, and checks membership. This remains an experimental interpretation proposal.


---

# 22_validation_repair: Correct an extraction response after a validation error

Clinical observations and numerical results below are invented examples.

## System message

```text
Read the patient's clinical text and report a value for each listed clinical variable.

What you receive

The clinical variable definitions and the text from one patient's record.

How to decide

Follow each variable's definition. Use the wording in the patient's text to determine its value. An explicit date belongs to a finding when the sentence or record heading links them. An undated CT sentence therefore keeps an unknown date even when it follows a dated laboratory result.

Use the defined category labels exactly. Write numerical measurements as JSON numbers in the required unit; a definition may also allow a reported threshold string. Use null according to the variable's missing-value rule.

What to return

Return one JSON object whose top-level keys are exactly the supplied clinical variable names. Each value is a scalar or null. Include every requested variable.
```

## User message

```text
Clinical variables

Serum creatinine
- Report the latest serum creatinine result in mg/dL.
- A directly reported threshold, such as <1.0, may remain a string.
- If several results have dates, use the latest dated result. Break a date tie using the last mention. If all results are undated, use the last mention.
- Use null for missing or unresolved results.

Emphysema on imaging
- Use Present for an explicit positive imaging finding and Absent for an explicit negative imaging finding.
- Use null for silence, uncertainty, or an unresolved finding.
- If findings conflict, use the latest dated finding. Break a date tie using the last mention. If all findings are undated, use the last mention.

Patient's clinical text

2025-01-01: serum creatinine 1.0 mg/dL. 2025-01-10: serum creatinine 1.2 mg/dL. CT documents emphysema.
```

## Assistant message

```text
{"Serum creatinine":1.2}
```

## User message

```text
Your preceding answer omitted Emphysema on imaging. Read the clinical text and its definition above, then return the complete corrected JSON object with every requested variable. Preserve supported values and use null when the definition requires it.
```

## Developer integration notes — excluded from the prompt

Python retains the original task/input, formats precise errors, controls reasoning and retry budgets, and validates the complete response. Split requests or increase output space when necessary; the repair prompt does not explain those mechanics.


---

# 23_length_repair: Complete a truncated extraction response

Clinical observations and numerical results below are invented examples.

## System message

```text
Read the patient's clinical text and report a value for each listed clinical variable.

What you receive

The clinical variable definitions and the text from one patient's record.

How to decide

Follow each variable's definition. Use the wording in the patient's text to determine its value. An explicit date belongs to a finding when the sentence or record heading links them. An undated CT sentence therefore keeps an unknown date even when it follows a dated laboratory result.

Use the defined category labels exactly. Write numerical measurements as JSON numbers in the required unit; a definition may also allow a reported threshold string. Use null according to the variable's missing-value rule.

What to return

Return one JSON object whose top-level keys are exactly the supplied clinical variable names. Each value is a scalar or null. Include every requested variable.
```

## User message

```text
Clinical variables

Serum creatinine
- Report the latest serum creatinine result in mg/dL.
- A directly reported threshold, such as <1.0, may remain a string.
- If several results have dates, use the latest dated result. Break a date tie using the last mention. If all results are undated, use the last mention.
- Use null for missing or unresolved results.

Emphysema on imaging
- Use Present for an explicit positive imaging finding and Absent for an explicit negative imaging finding.
- Use null for silence, uncertainty, or an unresolved finding.
- If findings conflict, use the latest dated finding. Break a date tie using the last mention. If all findings are undated, use the last mention.

Patient's clinical text

2025-01-01: serum creatinine 1.0 mg/dL. 2025-01-10: serum creatinine 1.2 mg/dL. CT documents emphysema.
```

## Assistant message

```text
{"Serum creatinine":1.2,"Emphysema on imaging":
```

## User message

```text
Your preceding answer ended before the JSON object was complete. Return a complete JSON object using the clinical text and definitions above. Include every requested variable and its supported scalar value or null.
```

## Developer integration notes — excluded from the prompt

Python retains the original task/input, formats precise errors, controls reasoning and retry budgets, and validates the complete response. Split requests or increase output space when necessary; the repair prompt does not explain those mechanics.


---
