"""Standard Stage 2 system prompts, initially reviewed with Qwen on 2026-09-23.

The definition, extraction, category-mapping, refinement and supervision
prompts describe the reserved documented_unknown answer
(:mod:`oci.inference.stage2_documented_unknown`). Since 2026-11-16 every
discovered candidate is defined or decomposed into closed-format variables with
two-part label definitions before harmonized merging
(:mod:`oci.inference.stage2_variable_ontology`); refinement and supervision no
longer offer an ambiguous (free-text) type.

Inputs, identity assignment, and validation live in the Python callers.
The review artifacts preserve the exact example conversations and model replies.
"""

PROMPT_VERSION = "stage2_clinical_prompts_v5_zero_counts_20261117"
HARMONIZED_MERGE_PROMPT_VERSION = "stage2_merge_and_harmonize_v1_20261116"
# Revision of the extraction prompts (05, 06, 07) only. Extraction checkpoints
# carry it in their schema versions, so a change re-extracts without touching
# discovery or feature definitions. The patient text comes first in each
# request, so a server's prefix cache reuses it across the patient's requests.
EXTRACTION_PROMPT_VERSION = "stage2_extraction_record_first_v1_20261119"

SYSTEM_PROMPTS = {
    '01_discover': """You identify clinical variables that could be measured from individual patients' medical records. Read all supplied excerpts and name each distinct clinical attribute they support.

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

Return each distinct attribute once. An excerpt containing several findings can support several candidates. Use {"candidates": []} when the text supports none.""",
    '02_audit_unmapped': """You identify clinical variables that could be measured from individual patients' medical records. Read all supplied excerpts and name each distinct clinical attribute they support.

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

Return each distinct attribute once. An excerpt containing several findings can support several candidates. Use {"candidates": []} when the text supports none.""",
    '03_merge_and_harmonize': """Find clinical variables in this list that measure the same thing, and give each such group one shared definition. The variables were defined separately from different clinical excerpts. None has been extracted from records yet, so each merged group will be extracted once, using its shared definition.

What you receive

A list of variables, each with an identifier (such as V07), a name, a description, its answer format (value type with a unit or allowed labels), each label's meaning and the statements that establish it, and its measurement rule. A variable marked locked has a fixed definition: a group contains at most one locked variable, and a group that contains one uses that variable's name, value type, unit, and labels unchanged in its shared definition.

How to decide

Merge variables only when they measure the same clinical attribute of the same patient: the same quantity, finding, or test; the same specimen or anatomical site; compatible timing; and answers that can share one format. Differences in wording, spelling, abbreviation, or label phrasing (for example Present/Absent versus Yes/No) are not reasons to keep variables apart.

Keep these separate: a measured value and an abnormal/normal category for it; a test result and whether the test was performed; the presence, severity, and subtype of a condition; current treatment and prior treatment history; a specific finding and a broader one that includes it; different anatomical sites; different genes or alterations. Related meaning, correlation, or a shared organ system is not enough to merge.

For each merged group, write a shared definition that every member's meaning fits:
- Use one value type and labels that preserve every distinction the members make. Give each shared label a definition with two parts, as the members do: meaning (the clinical state, or the range of values in a stated unit, that the label stands for) and established_by (the documented statements that assign the label directly, combining those of the member labels mapped to it). The extractor applies one order of precedence: a documented statement listed in established_by decides the label, and meaning is used only when the record describes the patient's state without such a statement.
- For numbers, choose one unit and state any conversion a member needs. Members recorded on different scales or instruments measure different quantities unless a stated conversion maps one onto the other: keep them apart. A grade with a few defined levels stays ordinal.
- A shared label means exactly what each member label mapped to it means. A label for a finding cannot map to a label for a more specific diagnosis or interpretation of that finding, and a label for one specific treatment, test, or finding cannot map to a label for a broader class that includes it.
- A residual member label (such as Other, unspecified, or not otherwise specified) maps only to a residual shared label of the same scope, never to a specific label or to the residual of a narrower class. A specific member label never maps to a residual shared label.
- Shared labels are mutually exclusive: never list a level together with its own sublevels, or two labels whose definitions overlap. When members record the same attribute at different levels of detail, so that a member label would fit several shared labels or none, keep those members apart.
- The shared labels come from one classification system. Members that classify the same attribute in different systems (for example a detailed system and a summary system that groups its levels) are different variables: keep them apart.
- Every shared label is the target of at least one member label. Do not add labels that no member has.
- Map every member label to exactly one shared label in category_crosswalk. A member label that fits no shared label, or two labels of one member that would collapse into one shared label, means the members are different variables: keep them apart.
- Write the measurement rule, the missing-value rule (null when not mentioned; the reserved answer documented_unknown when the record states the value is unknown, indeterminate, pending, not assessed, or equivocal), and the repeated-observation rule from the members' rules. When their repeated-observation rules differ, use latest for ordinary measurements.
- The shared definition follows the same format rules as its members: value_type is continuous, binary, categorical, or ordinal; a binary variable has exactly two labels; categories are possible answers to one question; a continuous variable has a unit (or null) and no categories.

What to return

Return one JSON object with groups, an array. Each group has members (the identifiers of at least two listed variables), reason (one sentence), definition, and category_crosswalk. definition has name, description, value_type, unit, categories, label_definitions (an object whose keys are exactly the shared labels; each value is an object with meaning and established_by; empty for a continuous variable), measurement_definition, missing_value_rule, and conflict_resolution (strategy and positive_category). category_crosswalk is an array of objects with member (an identifier), member_label, and shared_label; it is empty for continuous variables. An identifier belongs to at most one group. Omit variables that merge with nothing. Use {"groups": []} when nothing merges.""",
    '04_define_or_decompose': """Decide how to measure one candidate clinical variable from patients' medical records so that every answer has a closed format: a number in a stated unit, or exactly one label from a short fixed list.

What you receive

A candidate variable named during review of clinical text: its name, one or more descriptions, the wording that suggested it, and clinical excerpts illustrating how it is documented. The excerpts can come from different patients.

How to decide

A clean variable asks one question that has one answer per patient. The answer is a number in a stated unit, or one label from a fixed list of mutually exclusive labels that are possible answers to that question. Free text, open lists of findings, and answers that copy the record's wording are not clean answers.

Write every definition for an extractor who has not seen these excerpts and must reach the same answer from any patient's record. The extractor needs to know, for every number and every label, which documentation supports it.

First decide whether the candidate, as described, can be measured as one clean variable without losing its clinical meaning. If it can, define it (action define).

Otherwise decompose it (action decompose) into component variables that are each clean. A candidate needs decomposition when it bundles several attributes (for example the presence, severity, and location of a finding), several separately documented findings (for example the parts of a multi-part examination, or the members of a test panel), or answers whose documented wording is open-ended. Typical clean components are:
- the presence or absence of one named finding, symptom, condition, procedure, or treatment (binary);
- the result of one named test, using that test's standard result labels;
- one measured quantity with its unit (continuous);
- one grade, stage, or score with its defined levels (ordinal or categorical).

Base the components on attributes the excerpts document, at the level of detail clinicians use: one test result, one examination sign, one symptom. Do not add components the excerpts do not support. Return at most 12 components; when an open-ended collection documents more members than that, keep the clinically most important members the excerpts document.

Rules for every variable you return

- value_type is continuous, binary, categorical, or ordinal. There is no free-text or ambiguous type.
- Categories are possible answers to the variable's single question. They are not attribute names such as size, location, dose, or frequency, and never a list of co-occurring findings. A binary variable has exactly two labels. A categorical or ordinal variable has at least two labels; list ordinal labels from lowest to highest. A residual label such as Other is acceptable only alongside specific labels that cover the commonly documented answers.
- Every label needs a definition with two parts, including each label of a binary variable. meaning is the clinical state of the patient that the label stands for, written in clinical terms rather than record wording, so that an extractor can assign the label from a record that describes the patient's state without naming a level. established_by lists the documented statements that assign the label directly: a grade, score, stage, or category from a recognized system, the label's own term, and common equivalent terms. The excerpts show how some records document the variable; other records may use different words, a different scale, or only a description of the finding and its effects.
- The extractor applies one order of precedence. A documented statement listed in established_by decides the label. meaning is used only when the record describes the patient's state without such a statement. When the levels come from a recognized grading system, meaning follows that system's own definition of each level; do not borrow criteria from a different scale.
- When labels group the values of a quantity, meaning gives each label's range in a stated unit and says which label takes a value that falls on a boundary; the ranges cover every value without overlapping. When records state the quantity itself, prefer recording it as a continuous variable. Relative words such as high, low, early, or advanced are not definitions unless a stated boundary defines them.
- When records can state that the attribute is absent or does not apply to this patient (no such finding, no such treatment, no change), the labels include that answer, established by an explicit statement about this patient; when labels grade severity, it is the lowest label. Without it, a record that documents the absence has no answer. When records often describe a measured quantity only as absent or normal instead of giving a number, add a binary presence variable beside the continuous one.
- Name and describe the clinical attribute itself, not the instrument or form that recorded it in the excerpts, unless the variable is a numeric score that only that instrument produces.
- A grade with a few defined levels is ordinal, even when records write it as a number. A continuous variable measures one quantity on one scale in one unit. Results from different instruments or scales are different variables unless a stated conversion maps one onto the other; define one variable per scale.
- The labels of one variable come from one classification system. When records classify the same attribute in two systems (for example a detailed system and a summary system that groups its levels), define one variable per system.
- A binary presence variable states what documentation counts as its positive label and what explicit statement counts as its negative label.
- A continuous variable states its unit (null when unitless), the conversion for any other unit records commonly use, and which documented number to record. A count of findings, lesions, or events is 0 when the record explicitly states that none are present. It has no categories. It may accept a directly reported threshold string such as <1.0; say so in its measurement rule.
- Use null when the record does not mention the variable or its value cannot be resolved. A negative answer, such as Absent or Not detected, requires an explicit statement about this patient. Every variable also has the reserved answer documented_unknown for a record that explicitly states that the value is unknown, indeterminate, pending, not assessed, or equivocal; do not add categories such as Unknown, Indeterminate, Pending, or Not assessed. State in missing_value_rule when to answer null and when to answer documented_unknown.
- Choose one rule for repeated observations: latest (the usual choice for ordinary measurements), earliest, maximum, minimum, mode (most frequent value), any_positive (positive if any observation is positive), or single_or_null (use a value only when observations agree). Maximum and minimum apply to numbers; any_positive applies to a binary variable and names its positive label.
- Give each variable a short, readable snake_case name that states what it measures, following the pattern <test>_result or <finding>_on_examination. When you define the candidate without decomposing it, you may keep its name.

What to return

Return one JSON object with action (define or decompose), reason (one or two sentences), and variables (an array: exactly one variable for define, 2 to 12 for decompose). Each variable has name, description (one sentence), value_type, unit (a unit string, or null), categories (an array of labels; empty for a continuous variable), label_definitions (an object whose keys are exactly the variable's labels; each value is an object with meaning and established_by, both text; empty for a continuous variable), measurement_definition, missing_value_rule, and conflict_resolution (an object with strategy and positive_category: the exact positive label for any_positive, otherwise null).""",
    '05_extract_patient': """Read the patient's clinical text and report a value for each listed clinical variable.

What you receive

The text from one patient's record, followed by the clinical variable definitions.

How to decide

Follow each variable's definition. Use the wording in the patient's text to determine its value. An explicit date belongs to a finding when the sentence or record heading links them. An undated CT sentence therefore keeps an unknown date even when it follows a dated laboratory result.

Use the defined category labels exactly. When a variable gives label definitions, a documented statement that a label's definition says establishes it decides that label; use a label's meaning only when the record describes the patient's state without such a statement. Write numerical measurements as JSON numbers in the required unit; a definition may also allow a reported threshold string. Use null according to the variable's missing-value rule, and when the record does not mention the variable. Answer documented_unknown, written exactly so, when the record explicitly states that the value is unknown, indeterminate, pending, not assessed, or equivocal and documents no usable value. documented_unknown is not a category: never replace such a statement with one of the variable's categories.

What to return

Return one JSON object whose top-level keys are exactly the supplied clinical variable names. Each value is a scalar or null. Include every requested variable.""",
    '06_serial_chunk': """Update the listed clinical measurements using the next section of a patient's record.

What you receive

The next section of clinical text, followed by the measurement definitions, the values found so far, and notes about those values. The notes may include the date of an earlier observation or an unresolved conflict.

How to decide

Follow each variable's definition. Use the wording in the patient's text to determine its value. An explicit date belongs to a finding when the sentence or record heading links them. An undated CT sentence therefore keeps an unknown date even when it follows a dated laboratory result.

Use the defined category labels exactly. When a variable gives label definitions, a documented statement that a label's definition says establishes it decides that label; use a label's meaning only when the record describes the patient's state without such a statement. Write numerical measurements as JSON numbers in the required unit; a definition may also allow a reported threshold string. Use null according to the variable's missing-value rule, and when the record does not mention the variable. Answer documented_unknown, written exactly so, when the record explicitly states that the value is unknown, indeterminate, pending, not assessed, or equivocal and documents no usable value. documented_unknown is not a category: never replace such a statement with one of the variable's categories.

Keep an earlier value when the new text adds nothing that changes it under its definition. A prior null means that no usable value has been found yet. A prior documented_unknown means that the record so far only states that the value is unknown: replace it when the new section documents a usable value, and keep a usable earlier value when the new section only states that the value is unknown. For latest/earliest, a dated observation takes precedence over an undated observation; otherwise compare dates and then text order. The new section comes later in text order. Retain the selected value's date, when known, in its decision note. For maximum/minimum, retain the selected extreme; for any_positive, retain positive evidence; for single_or_null, retain unresolved conflicts.

What to return

Return one JSON object with values and decision_notes. Each is an object keyed by every supplied variable name. values holds the updated scalar or null. decision_notes holds a short statement of the date or other fact needed to choose among observations, or null. Keep each note brief and specific to choosing a value.""",
    '07_page_observations': """Find every documented occurrence of the listed clinical variables so Python can determine each variable's most frequently recorded value.

What you receive

One section of a patient's record, followed by the clinical variable definitions.

How to decide

Report each distinct value-bearing occurrence once, in the order it appears in the supplied text. Include repeated results on different dates. Use JSON numbers for numerical measurements and strings for category labels or directly reported thresholds. When a variable gives label definitions, a documented statement that a label's definition says establishes it decides that label; use a label's meaning only when the record describes the patient's state without such a statement. Report an occurrence that explicitly states that the variable's value is unknown, indeterminate, pending, not assessed, or equivocal using the value documented_unknown; it is not a category, and Python uses it only when no occurrence documents a usable value. Include a short supporting quotation with each occurrence. Keep its clinical meaning faithful; differences in punctuation, capitalization, or spacing are acceptable. Include a date only when its sentence or heading links it to the finding. Preserve repeated or conflicting observations so Python can count them and choose the most frequent value. A frequency tie is resolved using the latest reported date, then the last occurrence in text order.

If the measurement definition calls for null, omit that occurrence from the observations array. Do not create placeholder observations for missing values, unrelated findings, or mentions that lack the requested attribute. An explicitly documented negative category is a supported value and should be included. Report only occurrences present in this section; stop after the last one without cycling through the record again.

What to return

Return one JSON object with observations, an array. Each observation has feature (an existing clinical variable name), value (a scalar), quote (supporting wording from the record), and governing_date_quote (the linked date, or null). Use {"observations": []} when no observation is supported.""",
    '08_map_categories': """Choose the category that matches the supplied phrase for one clinical variable.

What you receive

The variable's meaning, its allowed categories, and the phrase to interpret.

How to decide

Use an allowed category when the phrase has an unambiguous equivalent meaning. Return documented_unknown when the phrase states that the value is unknown, indeterminate, pending, not assessed, or equivocal. Return null for an unclear or incompatible phrase. Use the category's exact spelling.

What to return

Return one JSON object with only value, containing an allowed category, documented_unknown, or null.""",
    '09_refine_ontology': """Review the extraction instructions for one clinical variable and decide whether they need clarification.

What you receive

The current definition and examples of attempted answers that failed its required format or categories.

How to decide

Choose keep when the failed answers are ordinary synonyms for existing categories and the clinical measurement rule is clear. Revise when the examples reveal an unclear measurement rule, unsuitable type, or missing distinction in the category definitions. A clarification can explain how to interpret equivalent wording. Use null for missing or unresolved findings. documented_unknown is a reserved answer available for every variable when the record explicitly states that the value is unknown, indeterminate, pending, not assessed, or equivocal; it is not a category, so do not add categories such as Unknown or Pending, and a definition need not list one. An explicit negative must describe this patient's finding to support Absent. Failed outputs are evidence about how the instructions were interpreted; the patient's actual value remains unknown from those outputs alone.

What to return

Category labels must be mutually exclusive patient answers, not attribute names (such as size, location, dose, or frequency). A single categorical field cannot return several co-occurring labels. Do not use arbitrary last-mention selection or null to conceal simultaneous findings. Split a composite into independently measurable scalar components when its definition requires several findings, for example separate presence indicators for anatomical sites. Derive components only from the supplied clinical definition or examples, preserve units and temporal scope, and require explicit evidence for both positive and negative answers; unmentioned components are null. When the evidence supports no finite decomposition, keep the current definition.

For keep, return one JSON object with action equal to keep and reason (text). For revise, return action equal to revise, reason, and definition. definition contains exactly description, value_type, categories_or_unit, label_definitions, measurement_definition, and missing_value_rule. value_type is continuous, binary, categorical, or ordinal; there is no free-text or ambiguous type. categories_or_unit contains a unit for a dimensional continuous value, an empty array for a unitless value, exactly two distinct labels for binary values, or at least two for categorical/ordinal values. label_definitions is an object whose keys are exactly the labels; each value is an object with meaning (the clinical state of the patient that the label stands for) and established_by (the documented statements that assign the label directly). It is empty for a continuous value. Preserve the same clinical measurement. For split, return action equal to split, reason, and components: an array of 2 to 20 complete definitions, each with those same six fields plus name (a distinct, readable clinical component name). Each component must request one scalar answer; replace the composite rather than retaining it alongside its components.""",
    '10_harmonize_values': """The measurements of one clinical variable contain numbers and text phrases. Choose a common representation that preserves their clinical meaning.

What you receive

The variable's definition, its observed numbers and phrases, and any stated thresholds or reference ranges.

How to decide

Use continuous values when every usable phrase denotes an exact number in the stated unit. Use categories when a reported threshold or range needs to be preserved. Use the smallest partition supported by explicit boundaries. Treat inequalities as intervals with their stated inclusive or exclusive boundary. A word such as high needs a supplied definition or reference range; otherwise classify it as unusable. If the supplied information cannot support a coherent representation, choose insufficient_definition and explain what is missing.

What to return

Return one JSON object with status (ready or insufficient_definition), representation (continuous, categorical, or null when insufficient), reason (text), and token_interpretations (an array). For each supplied phrase, return raw_text, meaning (exact_number, explicit_interval, defined_category, or unusable), and interpretation (a short statement of its supported meaning). Describe the threshold or category meaning in words.""",
    '11_extend_value_map': """Map one phrase for one clinical variable into the supplied measurement representation.

What you receive

The variable's meaning, the numerical representation or category definitions, and a new phrase.

How to decide

Use the supplied definitions exactly. A numerical representation requires a phrase with an exact numerical meaning. A category requires the phrase's full meaning to fit within that category. Use null when a phrase spans categories or remains ambiguous.

What to return

Return one JSON object with only value, containing an exact number, an existing category label, or null.""",
    '12_supervise_ontology': """Decide whether the extraction definition for one clinical variable needs clarification.

What you receive

The definition, a summary of current extracted values, and earlier failed answers. A patient with an earlier failure may now have a valid answer, so these counts can overlap.

How to decide

Choose keep when the failed answers are ordinary synonyms for existing categories and the clinical measurement rule is clear. Revise when the examples reveal an unclear measurement rule, unsuitable type, or missing distinction in the category definitions. A clarification can explain how to interpret equivalent wording. Use null for missing or unresolved findings. documented_unknown is a reserved answer available for every variable when the record explicitly states that the value is unknown, indeterminate, pending, not assessed, or equivocal; it is not a category, so do not add categories such as Unknown or Pending, and a definition need not list one. The summary counts documented_unknown answers separately from missing values. An explicit negative must describe this patient's finding to support Absent. Failed outputs are evidence about how the instructions were interpreted; the patient's actual value remains unknown from those outputs alone.

Missing values, documented_unknown answers, a rare finding, or a common category can occur under a sound definition. Revise when the supplied summaries reveal a specific problem in the instructions.

What to return

Category labels must be mutually exclusive patient answers, not attribute names (such as size, location, dose, or frequency). Never invent categories merely to satisfy the schema. A single categorical field cannot return several co-occurring labels. Do not use arbitrary last-mention selection or null to conceal simultaneous findings. Split a composite into independently measurable scalar components when its definition requires several findings, for example separate presence indicators for anatomical sites. Derive components only from the supplied clinical definition or examples, preserve units and temporal scope, and require explicit evidence for both positive and negative answers; unmentioned components are null. When the evidence supports no finite decomposition, keep the current definition. Keep existing sound scalar definitions without stylistic rewrites.

For keep, return one JSON object with action equal to keep and reason (text). For revise, return action equal to revise, reason, and definition. definition contains exactly description, value_type, categories_or_unit, label_definitions, measurement_definition, and missing_value_rule. value_type is continuous, binary, categorical, or ordinal; there is no free-text or ambiguous type. categories_or_unit contains a unit for a dimensional continuous value, an empty array for a unitless value, exactly two distinct labels for binary values, or at least two for categorical/ordinal values. label_definitions is an object whose keys are exactly the labels; each value is an object with meaning (the clinical state of the patient that the label stands for) and established_by (the documented statements that assign the label directly). It is empty for a continuous value. Preserve the same clinical measurement. For split, return action equal to split, reason, and components: an array of 2 to 20 complete definitions, each with those same six fields plus name (a distinct, readable clinical component name). Each component must request one scalar answer; replace the composite rather than retaining it alongside its components.""",
    'ontology_scalar_contract': """Check whether proposed clinical extraction definitions can measure the original variable faithfully using one scalar per component.

Reject categories that name attributes (such as size, location, dose, or frequency) instead of possible patient answers. Reject a single categorical field that asks for a list of simultaneous categories or chooses an arbitrary site just to fit one label. Splitting co-occurring findings into separate scalar indicators is valid; inventing components unsupported by the original definition is not. Each component must have a clear unit or actual answer labels and an explicit missing-value rule. documented_unknown is a reserved answer available to every component when the record states that the value is unknown; a component need not list an Unknown category. Absence of a mention does not establish a negative finding. Preserve the measurement's clinical meaning and temporal rule.

Return exactly one JSON object with valid (a boolean) and reason (text describing any concrete contradiction).""",
    '13_post_extraction_aliases': """Decide whether the supplied fields record the same clinical measurement and can be combined while retaining every observed value.

What you receive

Field definitions and comparisons of their values in patients for whom both fields were measured.

How to decide

Confirm the same clinical attribute, timing rule, unit, and level of detail, together with agreement between paired values. High correlation by itself leaves agreement unresolved. Keep fields separate when meaning, timing, units, or agreement is uncertain. Every pair in a proposed group must be compatible. Category synonyms may be equated when their meanings match. Preserve reported thresholds and all nonmissing information. A protected name remains the canonical name; two protected names remain separate.

What to return

Return one JSON object with action and reason. For action keep, these are the only fields. For action merge, also return members (existing field names), canonical_label (a clinical name), and category_equivalences (an array containing source_feature, source_category, and canonical_category for each needed recoding; otherwise empty).""",
    '21_cross_fold_concepts': """Identify the clinical concepts represented by candidate variables and assess which concepts warrant further evaluation as treatment-effect modifiers.

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

Return one JSON object with concepts, an array. Each has name (clinical text), members (existing variable names), modifier_recommendation (retain, uncertain, or exclude), rationale (text), representatives (existing member names), and unresolved_questions (text or an empty string). For retain, include at least one representative. For exclude, leave representatives empty. For uncertain, representatives may be empty.""",
}
