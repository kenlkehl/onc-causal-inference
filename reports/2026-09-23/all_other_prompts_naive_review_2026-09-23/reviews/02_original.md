# Naive review of request 02: `audit_unmapped`

## Restatement of the task

The request asks me to re-examine every string in every supplied evidence item and identify all missed patient-level clinical features supported by the text. I should favor recall and enumerate distinct atomic variables, including secondary details outside the apparent main topic. Each candidate should be a reusable clinical measurement or attribute rather than an umbrella, composite, inventory, input artifact, or analysis artifact.

For every candidate, I should return:

- a nonempty snake_case feature name identifying the exact extraction target;
- a description of exactly one atomic patient-level variable with a coherent value domain;
- an evidence rationale explaining how the cited text supports the feature and whether it is explicit or inferred;
- caveats covering limitations, ambiguity, or competing clinical explanations; and
- one or more valid evidence item numbers in `supporting_items`.

The response must be JSON only, with a top-level `candidates` list. If the evidence supports no valid candidate, that list should be empty. The request says not to choose value types, units, categories, or extraction ontologies in this step.

## Restatement of the supplied input

There is one evidence item, numbered `1`, whose `text` array contains one string. The string reports two pretreatment serum creatinine lab observations on different dates, one at 1.0 mg/dL and one at 1.2 mg/dL, and says that CT documents emphysema.

The job is labeled `audit_unmapped_text_evidence_for_missed_clinical_features`. The accompanying task statement says this text was not cited by an initial review and should now be audited for atomic patient-level features that may have been missed.

## Restatement of the requested output

The output is a JSON object with a `candidates` array. Each array element follows the five-field candidate structure shown under `response`: `name`, `description`, `evidence_rationale`, `caveats`, and `supporting_items`. Candidates should represent reusable feature definitions, not the particular observed values or dates embedded in their names.

## Do I fully understand what to do?

I understand the overall audit objective, the evidence to inspect, the candidate-level schema, and most inclusion and exclusion rules. I do not fully understand how to apply several rules to this example because the request simultaneously requires a feature to have one value per patient and tells me to use longitudinal evidence that contains two values for the same measurement. It is also unclear whether dates and the `pretreatment` label are merely context for the creatinine observations or are themselves eligible patient-level features. Those decisions could change the candidate list.

## Clarification questions that could change the output

1. How should the requirement that a candidate have “one value per patient” apply to a longitudinal measurement such as serum creatinine when the same patient has two documented values at different dates? Is the intended candidate the reusable measurement concept despite repeated observations, or must the candidate encode a defined timepoint or summary to yield one patient-level value?

2. Are measurement dates eligible patient-level clinical variables, or should dates always be treated only as provenance or temporal context? The instruction explicitly says to inspect timepoint labels and use longitudinal information, but it also excludes documentation artifacts and requires a reusable variable with one value per patient.

3. Is `pretreatment` intended to support a separate patient-level clinical attribute, such as treatment status or phase, or should it only qualify the lab observations? If it can support a feature, what exact patient-level construct is intended when the treatment itself is not identified?

4. Should an imaging statement such as “CT documents emphysema” support only the clinical condition, or can it also support an imaging-modality or imaging-performed feature? This affects whether modality/procedure information is considered a clinical patient attribute or a documentation artifact.

5. Does “return explicitly documented clinical features and narrower latent clinical features reasonably implied by the text” permit both an explicit candidate and a distinct inferred candidate arising from the same phrase when the inferred construct is clinically narrower, or should only one be returned to avoid overlapping targets?

6. What exact JSON schema is mandatory beyond the illustrated `response` object? In particular, must every candidate contain all five illustrated fields even when there are no meaningful caveats, and are additional fields forbidden?

7. Should candidate ordering follow first appearance in the evidence, clinical importance, or some other deterministic convention? Different ordering would not change content but would change the exact serialized output.

## Contradictions and tensions in the instructions

1. The rule requiring “one value per patient” conflicts with the example's repeated serum creatinine values and with the instruction to use longitudinal information. A conventional lab feature can have multiple values per patient unless a timepoint, aggregation, or observation-level key is specified, but choosing any of those is constrained by the ban on instance-specific names and ad hoc aggregation.

2. The instruction says to inspect headers and timepoint labels separately and to use longitudinal information, while also excluding documentation artifacts and requiring patient-level clinical variables. It does not establish whether a calendar date or `pretreatment` label crosses the boundary from context into an eligible feature.

3. The response template asks for a description containing “one coherent value domain,” while a later rule says not to choose a value type, unit, categories, or extraction ontology. A coherent domain can be described conceptually without selecting a concrete type, but the boundary between those requirements is not precisely defined.

4. The request favors exhaustive recall and includes reasonably implied latent features, while also requiring direct or unambiguous support and forbidding invented components. There is no stated threshold for when a reasonable implication becomes sufficiently unambiguous, so borderline inferred candidates may vary between reviewers.

5. The instruction to use the narrowest supported construct could lead to an inferred narrower condition or subtype, while the instruction not to invent unstated components can require staying with the broader explicit term. The example does not specify how to decide that boundary.

## Identifier, index, and format bookkeeping software could perform

- Parse the outer message JSON and the JSON-encoded user content, rejecting malformed input before semantic review.
- Verify that each `evidence_items` entry has a unique item identifier and that every element of `text` is a string.
- Track every text-array position under each evidence item so coverage checks can confirm that no supplied string was skipped.
- Validate that every `supporting_items` value refers to an existing evidence item number and contains no duplicate item numbers.
- Enforce a top-level JSON object with a `candidates` array and validate the required candidate fields and their data types.
- Check that each candidate name is nonempty snake_case.
- Detect duplicate candidate names and exact duplicate candidate objects while preserving distinct atomic variables.
- Maintain stable candidate ordering according to a specified convention once that convention is clarified.
- Ensure the response contains valid JSON only, with no Markdown fences, commentary, trailing text, placeholders, or non-JSON values.
- Check that `supporting_items` is an array of item identifiers rather than text-array indices, dates, or candidate indices.
- Associate multiple observations of the same apparent measurement with one reusable candidate for deduplication purposes, while leaving the unresolved semantic decision about longitudinal cardinality to the reviewer.
- Flag names that appear to contain instance-specific dates, observed values, units, patient identifiers, or exemplar identifiers.
- Flag candidate descriptions or rationales that are empty, and require a defined representation for caveats when none are present.
