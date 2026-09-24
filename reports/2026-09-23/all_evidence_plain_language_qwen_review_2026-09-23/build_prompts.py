"""Plain clinical-task prompt proposals. This script makes no model calls."""
from __future__ import annotations

import copy
import json
import runpy
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
PREVIOUS = OUT.parent / "all_other_prompts_naive_review_2026-09-23"
OLD = runpy.run_path(str(PREVIOUS / "build_replacements.py"))
SPECS = []


def add(slug, title, task, inputs, guidance, output, example, integration):
    system = f"{task.strip()}\n\nWhat you receive\n\n{inputs.strip()}\n\nHow to decide\n\n{guidance.strip()}\n\nWhat to return\n\n{output.strip()}"
    SPECS.append({"slug": slug, "title": title, "messages": [{"role": "system", "content": system}, {"role": "user", "content": example.strip()}], "integration_notes": integration})


clinical_text = "2025-01-01: serum creatinine 1.0 mg/dL. 2025-01-10: serum creatinine 1.2 mg/dL. CT documents emphysema."
discovery_task = "You identify clinical variables that could be measured from individual patients' medical records. Read all supplied excerpts and name each distinct clinical attribute they support."
discovery_inputs = "Clinical excerpts can describe different patients or dates. Read each excerpt independently. A short summary and a detailed record excerpt have equal standing as evidence."
discovery_guidance = """Include attributes stated explicitly or expressed unambiguously through clinical terminology, abbreviations, or notation. For example, a creatinine result supports serum creatinine concentration, and “CT shows emphysema” supports emphysema on imaging.

Name the measured attribute. Several creatinine results support one serum-creatinine variable. Dates provide context; explicit clinical durations and ages can be variables.

Keep independently varying attributes separate. Extract T, N, and M separately when TNM notation directly encodes all three. A named overall score, such as ECOG performance status, can remain one variable. Include score components when the excerpt states or directly encodes them.

Combine synonymous mentions of the same attribute. Prefer the detailed measurement explicitly given: “cough for three weeks” supports cough duration. Separately stated severity and duration support separate variables.

Keep a test name used only to attribute a result within that finding's description. A statement that someone underwent a procedure can support a procedure-history variable. Attribute a relative's condition to family history. Use the patient's own laboratory, tissue, and tumor findings for patient variables.

Negative and uncertain mentions can identify a variable. “No emphysema” identifies emphysema status; “possible emphysema” identifies that variable with uncertainty about the finding. Limit the list to attributes supported by the excerpts. Include secondary findings throughout the text."""
discovery_output = """Return one object with the key candidates, an array. Each candidate has exactly four text fields:
- name: a short clinical label.
- description: one sentence defining the clinical attribute.
- basis: a short explanation of the supporting wording.
- uncertainty: ambiguity in what the excerpt means, or an empty string when its meaning is clear.

Return each distinct attribute once. An excerpt containing several findings can support several candidates. Use {"candidates": []} when the text supports none."""
for number, title in [(1, "Discover clinical variables"), (2, "Review excerpts for overlooked clinical variables")]:
    add(f"{number:02d}_{'discover' if number == 1 else 'audit_unmapped'}", title,
        discovery_task, discovery_inputs, discovery_guidance, discovery_output,
        f"Clinical excerpts\n\n{clinical_text}",
        "Use one evidence card per call. Attach provenance and machine names in Python. The response schema matches the adopted discovery validator. The caller determines when a second review is needed.")

add("03_merge_aliases", "Group names for the same clinical variable",
    "Identify clinical variable names that describe the same measurement and can share one definition.",
    "A list of names and descriptions. Protected names, when present, must remain recognizable in the result.",
    "Group synonyms, abbreviations, and detailed/coarsened names when one measurement definition can represent them. Keep independently varying findings separate. Leave uncertain matches ungrouped. Each group needs at least two supplied names. A name can belong to one group. A group containing a protected name uses that name as its canonical label; two protected names remain separate.",
    'Return one object with the key merges, an array. Each group has members (existing names) and canonical_label (a clear clinical name). Omit unchanged variables. Use {"merges": []} when no merge is justified.',
    "Variables\n- Creatinine level: Serum creatinine concentration.\n- Serum creatinine: Serum creatinine concentration.\n- Emphysema: Documented emphysema.\n\nProtected names: none.",
    "Resolve semantic labels, enforce disjoint membership/protections, preserve unmentioned variables, and normalize names in Python.")

add("04_define_ontology", "Define how to measure one clinical variable",
    "Write clear instructions for extracting one clinical variable from a patient's medical record.",
    "The variable's name and clinical excerpts illustrating how it is documented. The excerpts can come from different patients.",
    """Define one value per patient. Choose a numerical value when the record supports a measurement, and specify its unit. State allowed labels for categorical values. Use null for missing or unresolved findings. A continuous measurement can contain an exact number or a directly reported threshold string, such as <1.0. Preserve that threshold in both the measurement rule and the missing-value rule.

Specify how to handle repeated observations. For ordinary measurements, use the latest result. Other choices are earliest, maximum, minimum, mode (most frequent value), any_positive (positive if any observation is positive), and single_or_null (use a value only when observations agree). Select another rule when the named variable calls for it. Maximum/minimum apply to numbers; any_positive applies to a binary variable.

For latest/earliest, choose among dated observations when any are available; use text order when all are undated. Equal-date ties use the last mention for latest and first for earliest. State a standard unit conversion if one is needed. An unresolved unit or incompatible measurement yields null.""",
    "Return one JSON object with description, value_type, unit, categories, measurement_definition, missing_value_rule, conflict_resolution, and caveats. Text fields contain the definition and any necessary qualifications. value_type is continuous, binary, categorical, ordinal, or ambiguous. unit is a unit string, such as mg/dL, or null for unitless, categorical, or ambiguous variables. categories is an array containing two labels for a binary variable, the labels for a categorical/ordinal variable, or an empty array for continuous/ambiguous values. conflict_resolution contains strategy (one of the named rules) and positive_category (the exact positive label for any_positive; otherwise null). For ambiguous measurements, explain the ambiguity in caveats and use single_or_null. caveats may be empty.",
    "Clinical variable: Serum creatinine\n\nExample clinical excerpts\n2025-01-01: serum creatinine 1.0 mg/dL.\n2025-01-10: serum creatinine 1.2 mg/dL.\nSerum creatinine <1.0 mg/dL.",
    "Keep identity/protections in Python; validate the definition and conflict rules. Translate the model's unit string or category list into the legacy categories_or_unit array. Latest is the proposed prompt default for ordinary repeated measurements. Study-specific measurement restrictions must be written directly into a feature definition when needed.")

definitions = """Clinical variables

Serum creatinine
- Report the latest serum creatinine result in mg/dL.
- A directly reported threshold, such as <1.0, may remain a string.
- If several results have dates, use the latest dated result. Break a date tie using the last mention. If all results are undated, use the last mention.
- Use null for missing or unresolved results.

Emphysema on imaging
- Use Present for an explicit positive imaging finding and Absent for an explicit negative imaging finding.
- Use null for silence, uncertainty, or an unresolved finding.
- If findings conflict, use the latest dated finding. Break a date tie using the last mention. If all findings are undated, use the last mention."""
extraction_guidance = """Follow each variable's definition. Use the wording in the patient's text to determine its value. An explicit date belongs to a finding when the sentence or record heading links them. An undated CT sentence therefore keeps an unknown date even when it follows a dated laboratory result.

Use the defined category labels exactly. Write numerical measurements as JSON numbers in the required unit; a definition may also allow a reported threshold string. Use null according to the variable's missing-value rule."""
flat_output = "Return one JSON object whose top-level keys are exactly the supplied clinical variable names. Each value is a scalar or null. Include every requested variable."
add("05_extract_patient", "Extract one patient's declared variables",
    "Read the patient's clinical text and report a value for each listed clinical variable.",
    "The clinical variable definitions and the text from one patient's record.",
    extraction_guidance, flat_output,
    f"{definitions}\n\nPatient's clinical text\n\n{clinical_text}",
    "Python owns patient identity, internal column names, envelopes, and validation. Supply the appropriate record text; omit record_scope and any explanation of upstream eligibility handling.")

add("06_serial_chunk", "Update values after reading more of a patient's record",
    "Update the listed clinical measurements using the next section of a patient's record.",
    "The measurement definitions, the values found so far, notes about those values, and the next section of clinical text. The notes may include the date of an earlier observation or an unresolved conflict.",
    extraction_guidance + "\n\nKeep an earlier value when the new text adds nothing that changes it under its definition. A prior null means that no usable value has been found yet. For latest/earliest, a dated observation takes precedence over an undated observation; otherwise compare dates and then text order. The new section comes later in text order. Retain the selected value's date, when known, in its decision note. For maximum/minimum, retain the selected extreme; for any_positive, retain positive evidence; for single_or_null, retain unresolved conflicts.",
    "Return one JSON object with values and decision_notes. Each is an object keyed by every supplied variable name. values holds the updated scalar or null. decision_notes holds a short statement of the date or other fact needed to choose among observations, or null. Keep each note brief and specific to choosing a value.",
    f"{definitions}\n\nPreviously extracted values\n- Serum creatinine: 1.0 mg/dL, dated 2025-01-01.\n- Emphysema on imaging: null; no usable finding yet.\n\nNext section of the record\n2025-01-10: serum creatinine 1.2 mg/dL. CT documents emphysema.",
    "Python keeps patient/chunk IDs and ordering. Route mode aggregation through occurrence extraction so Python can count observations. State translation and bookkeeping are excluded from the model instructions.")

add("07_page_observations", "Extract clinical observations and their supporting words",
    "Find every documented observation of the listed clinical variables in the supplied text.",
    "Clinical variable definitions and one section of a patient's record.",
    "Report each distinct value-bearing occurrence, including repeated results on different dates. Use JSON numbers for exact numerical measurements and strings for category labels or directly reported thresholds. Support it with a short exact quotation. Include enough nearby wording to identify the occurrence. Include a date only when its sentence or heading links it to the finding. Copy the date as written. Preserve repeated or conflicting observations for later comparison.",
    'Return one JSON object with observations, an array. Each observation has feature (an existing clinical variable name), value (a scalar), quote (exact source wording), and governing_date_quote (the date as written, or null). Use {"observations": []} when no observation is supported.',
    f"Clinical variables\n- Serum creatinine: a result in mg/dL, including directly reported threshold strings.\n- Emphysema on imaging: Present for an explicit positive finding, Absent for an explicit negative finding.\n\nClinical text\n{clinical_text}",
    "Python locates exact quotations, attaches source/patient/page IDs, normalizes dates, and reconciles observations. Validate ambiguous quote matches rather than assigning arbitrary offsets.")

add("08_map_categories", "Match a phrase to an existing clinical category",
    "Choose the category that matches the supplied phrase for one clinical variable.",
    "The variable's meaning, its allowed categories, and the phrase to interpret.",
    "Use an allowed category when the phrase has an unambiguous equivalent meaning. Return null for an unclear or incompatible phrase. Use the category's exact spelling.",
    "Return one JSON object with only value, containing an allowed category or null.",
    "Clinical variable: Emphysema on imaging\nMeaning: Whether imaging explicitly describes emphysema.\nAllowed categories: Present, Absent\nPhrase: present on CT",
    "Deduplicate tokens and handle IDs/occurrences in Python. Use deterministic normalization where it suffices.")

refinement_output = "For keep, return one JSON object with action equal to keep and reason (text). For revise, return action equal to revise, reason, and definition. definition contains exactly description, value_type, categories_or_unit, measurement_definition, and missing_value_rule. value_type is continuous, binary, categorical, or ordinal. categories_or_unit contains a unit for a dimensional continuous value, an empty array for a unitless value, exactly two distinct labels for binary values, or at least two for categorical/ordinal values. Preserve the same clinical variable."
emphysema_definition = """Clinical variable: Emphysema on imaging
Description: Imaging documentation of emphysema presence or absence.
Value type: binary
Categories: Present, Absent
Measurement rule: Use an explicit positive or negative imaging statement.
Missing-value rule: Use null for unreported or unresolved findings."""
refinement_rules = "Choose keep when the failed answers are ordinary synonyms for existing categories and the clinical measurement rule is clear. Revise when the examples reveal an unclear measurement rule, unsuitable type, or missing distinction in the category definitions. A clarification can explain how to interpret equivalent wording. Use null for missing or unresolved findings. An explicit negative must describe this patient's finding to support Absent. Failed outputs are evidence about how the instructions were interpreted; the patient's actual value remains unknown from those outputs alone."
add("09_refine_ontology", "Clarify instructions after extraction failures",
    "Review the extraction instructions for one clinical variable and decide whether they need clarification.",
    "The current definition and examples of attempted answers that failed its required format or categories.",
    refinement_rules, refinement_output,
    f"{emphysema_definition}\n\nFailed answers from three patients: positive; present on CT. These strings fall outside the two required category labels.",
    "Python attaches identity and restricts revisions to the allowed definition fields, then controls re-extraction. Ordinary category synonyms can use the category-mapping path.")

add("10_harmonize_values", "Choose a common representation for one clinical variable",
    "The measurements of one clinical variable contain numbers and text phrases. Choose a common representation that preserves their clinical meaning.",
    "The variable's definition, its observed numbers and phrases, and any stated thresholds or reference ranges.",
    "Use continuous values when every usable phrase denotes an exact number in the stated unit. Use categories when a reported threshold or range needs to be preserved. Use the smallest partition supported by explicit boundaries. Treat inequalities as intervals with their stated inclusive or exclusive boundary. A word such as high needs a supplied definition or reference range; otherwise classify it as unusable. If the supplied information cannot support a coherent representation, choose insufficient_definition and explain what is missing.",
    "Return one JSON object with status (ready or insufficient_definition), representation (continuous, categorical, or null when insufficient), reason (text), and token_interpretations (an array). For each supplied phrase, return raw_text, meaning (exact_number, explicit_interval, defined_category, or unusable), and interpretation (a short statement of its supported meaning). Describe the threshold or category meaning in words.",
    "Clinical variable: Serum creatinine, measured in mg/dL\nObserved numbers include: 0.8, 1.2\nObserved phrases: <1.0; high\nExplicit threshold: <1.0 means less than 1.0 mg/dL.\nReference interval or definition of high: unavailable.",
    "Python parses supported numerical constraints and creates exhaustive, nonoverlapping intervals, including complements. A typed adapter or explicit clarification path is required for unparsed semantics. Insufficient definitions must be handled explicitly; the LLM does not need to know the bin-construction mechanism.")

add("11_extend_value_map", "Interpret a new phrase using defined categories",
    "Map one phrase for one clinical variable into the supplied measurement representation.",
    "The variable's meaning, the numerical representation or category definitions, and a new phrase.",
    "Use the supplied definitions exactly. A numerical representation requires a phrase with an exact numerical meaning. A category requires the phrase's full meaning to fit within that category. Use null when a phrase spans categories or remains ambiguous.",
    "Return one JSON object with only value, containing an exact number, an existing category label, or null.",
    "Clinical variable: Serum creatinine, mg/dL\nCategories\n- Below 1.0: x < 1.0 mg/dL\n- At least 1.0: x >= 1.0 mg/dL\n\nNew phrase: less than 1.0",
    "Python keeps the representation fixed, extends the token map, and handles identities. This proposed call remains limited to training map construction.")

add("12_supervise_ontology", "Review a clinical measurement definition using extraction summaries",
    "Decide whether the extraction definition for one clinical variable needs clarification.",
    "The definition, a summary of current extracted values, and earlier failed answers. A patient with an earlier failure may now have a valid answer, so these counts can overlap.",
    refinement_rules + "\n\nMissing values, a rare finding, or a common category can occur under a sound definition. Revise when the supplied summaries reveal a specific problem in the instructions.",
    refinement_output,
    f"{emphysema_definition}\n\nCurrent values from 10 patients\n- Present: 5\n- Absent: 3\n- Missing: 2\n\nEarlier failures affected 3 patients. Example failed answers: positive; present on CT. Their overlap with the current counts is unknown.",
    "Python preserves the clinical target and roles, distinguishes current/historical count populations, and controls revisions and re-extraction.")

add("13_post_extraction_aliases", "Combine equivalent extracted measurements",
    "Decide whether the supplied fields record the same clinical measurement and can be combined while retaining every observed value.",
    "Field definitions and comparisons of their values in patients for whom both fields were measured.",
    "Confirm the same clinical attribute, timing rule, unit, and level of detail, together with agreement between paired values. High correlation by itself leaves agreement unresolved. Keep fields separate when meaning, timing, units, or agreement is uncertain. Every pair in a proposed group must be compatible. Category synonyms may be equated when their meanings match. Preserve reported thresholds and all nonmissing information. A protected name remains the canonical name; two protected names remain separate.",
    "Return one JSON object with action and reason. For action keep, these are the only fields. For action merge, also return members (existing field names), canonical_label (a clinical name), and category_equivalences (an array containing source_feature, source_category, and canonical_category for each needed recoding; otherwise empty).",
    "Fields\n- Creatinine level: Latest serum creatinine, mg/dL; exact number or null.\n- Serum creatinine: Latest serum creatinine, mg/dL; exact number or null.\n\nPaired-value comparison\n- 40 patients have both values.\n- No disagreements at the stated measurement precision.\n- Units and timing rules match.\n- Neither field contains text values.\n- Combining their observed values preserves all nonmissing information.\n\nProtected names: none.",
    "Python computes paired agreement, compiles deterministic coalescing/recodes, checks losslessness, and retains IDs. This proposal requires those diagnostics and adapters before use.")

study_text = """Study
- Observational study of patients who received treatment A or treatment B.
- Compare treatment A with treatment B at treatment initiation.
- Outcome: a favorable event within one year, recorded as yes or no.
- Treatment effect: the difference in the probability of that event under A versus B.
- The variables describe the patient's health before treatment.

The analyses use three different, overlapping groups from the same training dataset. Effect analyses include patients with estimated probability of receiving A between 0.1 and 0.9."""


def evidence_text(candidate):
    lines = [f"Clinical variable: {candidate['feature']}", f"Definition: {candidate['definition'].replace('eligible ', '')}", "", "Model results"]
    for row in candidate["evidence"]:
        method = row["method"] + (f" — {row['target']}" if row.get("target") else "")
        if row["status"] == "unavailable":
            lines.append(f"- {method}: results unavailable.")
            continue
        lines.append(f"- {method}: {row['positive_signal_splits']} of {row['evaluated_splits']} analyses met the criterion “{row['criterion']}”.")
        for key, label in [("validation_R_loss_gains", "Validation R-loss gains"), ("interaction_p_values", "Interaction p-values"), ("permutation_R_loss_increases", "R-loss increases after permutation")]:
            if key in row:
                lines.append(f"  {label}: {', '.join(map(str, row[key]))}.")
        if row.get("multiplicity_adjusted_q_values", "absent") is None:
            lines.append("  Multiplicity-adjusted q-values are unavailable.")
    return "\n".join(lines)


default_glossary = """How to read these results

Penalized main-effect selection means the variable received a nonzero coefficient when predicting treatment or outcome. Treatment interactions test whether the treatment association changes with the variable; logistic interactions are measured on the log-odds scale.

The R-learner measures differences in treatment effect after accounting for predictions of treatment and outcome. Its univariable version uses one candidate variable to model those differences. Validation R-loss measures the remaining squared prediction error on patients held aside for that analysis. A positive R-loss gain means improvement over a constant treatment effect. Without a baseline loss or uncertainty estimate, describe its direction and consistency and leave the practical size of the improvement unresolved.

The usable-analysis counts show how much evidence is available. Missing results leave a question open. Results from overlapping patient groups describe consistency within this dataset."""
forest_glossary = "\n\nA univariable interaction screen tests one treatment-by-variable term in an outcome model; its p-value describes that association. When adjusted q-values are unavailable, nominal p-values offer limited evidence about a pattern found among many variables. The causal-forest result compares the same fitted forest's validation R-loss before and after shuffling this variable. A positive increase shows how much shuffling worsened that forest's predictions. Correlated measurements can share that predictive information."
role_task = "Decide whether one clinical variable should be used to adjust for confounding, to predict differences in treatment benefit between patients, or both."
role_inputs = "The treatment comparison, outcome, clinical variable definition, and results from several statistical analyses. A confounder can influence treatment choice and outcome. An effect modifier helps identify patients with different effects of treatment A compared with B."
role_rules = """For confounding, consider whether the variable could influence both treatment choice and outcome, using the study description and the treatment/outcome results. Retain a credible common cause when the evidence is incomplete but supports that concern.

For effect modification, emphasize validation evidence for differences in treatment benefit. Consider agreement and disagreement across analyses, the amount of usable evidence, and whether the result concerns probability differences or log odds. Main-effect prediction alone leaves the modifier question open. Use the clinical description to interpret the results and state material uncertainty.

Choose the confidence labels qualitatively, considering consistency, the size of validation gains, and available uncertainty estimates. Explain that judgment in the rationale. When a clinical relationship is unknown and the relevant results are unavailable, use uncertain.

Use supported when relevant evidence agrees and supports the role; plausible when the role is credible with partial or noisy support; uncertain when missing information or unresolved disagreement prevents a judgment; and not_supported when the available evidence does not justify retaining that role in this analysis.

Judge stability separately for each role. consistent means the relevant results broadly agree; mixed means meaningful disagreement; insufficient means too little usable evidence to judge consistency."""
role_output = "Return one JSON object with exactly confounder and effect_modifier. Each contains assign (boolean), assessment (supported, plausible, uncertain, or not_supported), stability (consistent, mixed, or insufficient), rationale (text), and evidence_comments (an array of short statements naming the relevant methods and findings). assign can be true for supported or plausible. Use an empty evidence_comments array when evidence is unavailable."
add("14_default_roles", "Assess a clinical variable's confounder and modifier roles",
    role_task, role_inputs, role_rules + "\n\n" + default_glossary, role_output,
    f"{study_text}\n\nClinical background: In this illustrative study, renal function can influence treatment choice and outcome.\n\n{evidence_text(OLD['creatinine_default'])}",
    "Python supplies genuine study context and coherent evidence availability, attaches identity/provenance, preserves investigator-locked roles, and validates assignments. Qualitative assessments require evaluation before adoption.")

add("15_model_themes", "Group variables into clinical themes",
    "Group the supplied clinical variables into themes that help explain the pattern of statistical results.",
    "Clinical variable descriptions and their results from several statistical analyses. A theme brings related measurements together, such as several measures of kidney function.",
    "Choose clinically coherent groups and explain how the results agree or differ. Preserve distinct measurements within a group. Give an uncertain or unrelated variable its own theme. Include each variable in one theme, including those with weak or unavailable results. Use as many themes as the clinical meanings require.\n\n" + default_glossary + forest_glossary,
    "Return one JSON object with themes, an array. Each theme has name (a clinical label), members (existing variable names), interpretation (text), and disagreements (text, or an empty string).",
    f"{study_text}\n\n{evidence_text(OLD['creatinine_multi'])}\n\n{evidence_text(OLD['emphysema_multi'])}",
    "Python checks coverage, assigns theme IDs, and attaches member evidence. Theme grouping leaves measurement columns intact.")

add("16_merge_themes", "Combine overlapping clinical themes",
    "Identify themes that describe the same or clearly overlapping clinical concept and combine their summaries.",
    "Theme names, their clinical variables, and summaries of the statistical evidence.",
    "Combine themes when one coherent clinical description preserves their meanings. Keep unrelated themes separate. Preserve distinct measurements and disagreements in the combined summary. A theme can appear in one proposed merge group.",
    'Return one JSON object with merges, an array. Each group has source_themes (existing names), name (the combined clinical name), interpretation (text), and disagreements (text or an empty string). Omit unchanged themes. Use {"merges": []} when none overlap.',
    "Themes\n\nRenal function\n- Variable: Serum creatinine\n- Summary: Predicts treatment/outcome; weak adjusted modifier evidence.\n- Disagreement: Unadjusted interaction evidence is stronger than validation evidence.\n\nPulmonary disease\n- Variable: Emphysema on imaging\n- Summary: Positive effect-validation results.\n- Disagreement: One interaction screen is weaker than the others.",
    "Python unions members/provenance and preserves unchanged themes. Context limits are handled by batching/retrieval rather than forcing unrelated semantic merges.")

add("17_model_roles", "Assess roles using several models and clinical themes",
    role_task, role_inputs + " Related variables may be summarized under a clinical theme. Use the named variable's own results to assess it.",
    role_rules + "\n\n" + default_glossary + forest_glossary, role_output,
    f"{study_text}\n\nClinical background: Pulmonary disease may affect prognosis. Its relationship with treatment choice is unknown.\n\n{evidence_text(OLD['emphysema_multi'])}\n\nClinical theme: Pulmonary disease. The variable has positive treatment-effect validation results; treatment-choice results are unavailable.",
    "Python attaches feature/evidence IDs, retains locked roles, and applies routing. This prompt only contains the clinical decision and evidence needed to make it.")

ranking_rules = """Give greatest weight to credible validation evidence that a variable helps predict differences in treatment effect. Consider consistency, usable analyses, and disagreement between methods. A variable that predicts prognosis may still have weak evidence of modifying treatment benefit.

Compare numbers within the same method and scale. The study's effect is a probability difference; a logistic interaction describes a log-odds relationship. Treat unavailable results as uncertainty. Infer redundancy only when comparisons of the measured variables support it. Group candidates as tied when the evidence cannot distinguish them.\n\n""" + default_glossary + forest_glossary
two_candidates = f"{study_text}\n\n{evidence_text(OLD['creatinine_multi'])}\n\n{evidence_text(OLD['emphysema_multi'])}\n\nMeasured redundancy between these variables: unavailable."
add("18_rank_modifiers", "Rank clinical variables by modifier evidence",
    "Order the listed clinical variables by the strength of evidence that they help predict differences in treatment benefit between patients.",
    "The treatment comparison, outcome, variable definitions, and statistical results.",
    ranking_rules,
    "Return one JSON object with ordered_groups, an array from strongest to weakest evidence. Each group has features (one existing name, or several tied names) and rationale (text explaining the evidence and important uncertainty). Include every supplied variable once.",
    two_candidates,
    "Python resolves labels, orders genuine ties deterministically, assigns numerical ranks, and chooses a modifier count through validation. Those mechanics are absent from the model task.")

add("19_merge_rankings", "Compare two potential effect modifiers",
    "Choose which of the two clinical variables has stronger evidence of predicting differences in treatment benefit between patients.",
    "The treatment comparison, outcome, two variable definitions, and their statistical results.",
    ranking_rules.replace("Group candidates as tied when the evidence cannot distinguish them.", "Choose a tie when the evidence cannot distinguish the candidates."),
    "Return one JSON object with preferred_feature (one supplied clinical name, or null for a tie) and rationale (text explaining the comparison and uncertainty).",
    two_candidates,
    "Python selects the current front candidates, preserves within-list order, caches comparisons, and produces the merged ranking. Pairwise merging still requires cost and order-sensitivity evaluation.")

add("20_advisory_roles", "Explain a variable's recorded selection decision",
    "Explain how the statistical results support or cast doubt on the recorded decision about a clinical variable.",
    "The study description, variable definition, model results, and a decision about using the variable for adjustment and treatment-effect prediction.",
    "Explain the evidence behind the recorded choice and its limitations. Discuss what remains uncertain about confounding or differences in treatment benefit. Use the clinical description to interpret the evidence.\n\n" + default_glossary,
    "Return one JSON object with interpretation and limitations, both text.",
    f"{study_text}\n\nClinical background: In this illustrative study, renal function can influence treatment choice and outcome.\n\n{evidence_text(OLD['creatinine_default'])}\n\nRecorded decision\n- Use serum creatinine for adjustment.\n- Omit it from the treatment-effect heterogeneity model.",
    "Python attaches this explanation without changing fixed numerical decisions. A failed annotation does not alter selection.")

add("21_cross_fold_concepts", "Identify clinical concepts behind candidate modifiers",
    "Identify the clinical concepts represented by candidate variables and assess which concepts warrant further evaluation as treatment-effect modifiers.",
    "Variable definitions, model results, and how often each variable appeared in candidate lists from overlapping groups of patients.",
    "Group variables with a coherent clinical meaning while preserving distinct facets. Use existing measurements as representatives when they capture the supported facets. Include each candidate in one concept. Recommend retain for a concept with credible convergent modifier evidence, uncertain when the evidence remains inadequate or conflicting, and exclude when usable evidence supports deprioritizing it. Treat list recurrence as a description of repeated selection within this dataset. State when score calibration or measured redundancy is missing.\n\n" + ranking_rules,
    "Return one JSON object with concepts, an array. Each has name (clinical text), members (existing variable names), modifier_recommendation (retain, uncertain, or exclude), rationale (text), representatives (existing member names), and unresolved_questions (text or an empty string). For retain, include at least one representative. For exclude, leave representatives empty. For uncertain, representatives may be empty.",
    two_candidates + "\n\nCandidate-list recurrence\n- Serum creatinine appeared in 2 of 3 lists.\n- Emphysema on imaging appeared in 3 of 3 lists.\n\nNo null-distribution calibration or minimum meaningful R-loss improvement is available.",
    "Python computes recurrence and coverage, keeps the review blinded to oracle/test outcomes, and checks membership. This remains an experimental interpretation proposal.")

base_extract = copy.deepcopy(next(s for s in SPECS if s["slug"] == "05_extract_patient"))
for slug, title, failed, repair in [
    ("22_validation_repair", "Correct an extraction response after a validation error", '{"Serum creatinine":1.2}', "Your preceding answer omitted Emphysema on imaging. Read the clinical text and its definition above, then return the complete corrected JSON object with every requested variable. Preserve supported values and use null when the definition requires it."),
    ("23_length_repair", "Complete a truncated extraction response", '{"Serum creatinine":1.2,"Emphysema on imaging":', "Your preceding answer ended before the JSON object was complete. Return a complete JSON object using the clinical text and definitions above. Include every requested variable and its supported scalar value or null."),
]:
    SPECS.append({"slug": slug, "title": title, "messages": copy.deepcopy(base_extract["messages"]) + [{"role": "assistant", "content": failed}, {"role": "user", "content": repair}], "integration_notes": "Python retains the original task/input, formats precise errors, controls reasoning and retry budgets, and validates the complete response. Split requests or increase output space when necessary; the repair prompt does not explain those mechanics."})


def write_all():
    combined = ["# Revised clinical-task prompts — September 23, 2026", "", "These are reviewed prompt proposals. Clinical observations and numerical results below are invented examples. Integration notes are for the developer and are excluded from every model request.", ""]
    for spec in SPECS:
        slug = spec["slug"]
        (OUT / "prompts" / f"{slug}.json").write_text(json.dumps(spec["messages"], ensure_ascii=False, indent=2) + "\n")
        lines = [f"# {slug}: {spec['title']}", "", "Clinical observations and numerical results below are invented examples.", ""]
        for message in spec["messages"]:
            lines += [f"## {message['role'].title()} message", "", "```text", message["content"], "```", ""]
        lines += ["## Developer integration notes — excluded from the prompt", "", spec["integration_notes"], ""]
        rendered = "\n".join(lines)
        (OUT / "prompts" / f"{slug}.md").write_text(rendered)
        combined += [rendered, "\n---\n"]
    (OUT / "ALL_REVISED_PROMPTS_2026-09-23.md").write_text("\n".join(combined))
    (OUT / "manifest.json").write_text(json.dumps([{k: v for k, v in s.items() if k != "messages"} for s in SPECS], ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {len(SPECS)} clinical-task prompt proposals.")


if __name__ == "__main__":
    write_all()
