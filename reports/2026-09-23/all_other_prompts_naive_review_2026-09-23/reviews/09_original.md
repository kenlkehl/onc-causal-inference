# Naive review of `09_refine_ontology.json`

## Restatement of the request

The request asks an LLM to examine one existing clinical extraction feature ontology and aggregate failure diagnostics from outer-training patients, then decide whether to keep the ontology or revise it. The refinement must remain limited to the supplied feature and must still describe exactly one reusable patient-level scalar measurement. The model must return JSON only.

## Supplied input

The feature is `example_emphysema`, named `emphysema`. It is currently binary, with the exact categories `Present` and `Absent`. Its description is “Documented emphysema.” Its measurement definition calls for explicit pretreatment documentation of emphysema presence or absence. Its missing-value rule says to return null when emphysema is unreported and not to interpret silence as absence.

The only reported repeated failure pattern is `invalid_category` across three patients. The prior model outputs were `present on CT` and `positive`, while the allowed categories were `Present` and `Absent`. The prompt explicitly says these examples are failed model outputs rather than verified patient facts. No held-out patient text, treatment, or outcome is supplied.

## Requested output

The output is a single JSON object with:

- `action`: `keep` or `revise`;
- `reason`: an explanation of why the ontology is retained or changed; and
- for a revision, `value_type`, `categories_or_unit`, `description`, `measurement_definition`, and `missing_value_rule`.

Any revision must preserve one scalar feature, cannot rename or otherwise alter the feature inventory or causal roles, and must satisfy the category-count constraints for the selected value type.

## Can this be executed without consequential hidden assumptions?

Not fully. The substantive evidence strongly suggests a distinction between a bad ontology and a failure to canonicalize synonymous outputs: `present on CT` and `positive` both appear semantically compatible with `Present`, while the existing ontology already has a clinically coherent binary domain and a clear rule against treating silence as absence. However, the prompt does not state whether source phrases such as these are supposed to be normalized to the declared category label by the extractor, by a downstream validator, or by wording added to the ontology. That policy changes whether the justified action is `keep` or `revise`.

The JSON shape for `keep` is also underspecified. The response template lists all ontology fields, but labels them “required for revise” without saying whether a `keep` response should omit them, repeat the existing values, or emit nulls. Choosing among these would be a consequential format assumption if a strict parser consumes the result.

## Concrete clarification questions

1. Should semantically equivalent source/output phrases such as `present on CT` and `positive` be mapped to the exact canonical category `Present`? If so, is that mapping expected to be expressed in a revised `measurement_definition`, or is it an extractor/validator behavior that should leave the ontology unchanged?
2. Is category validation exact and case-sensitive, so that every extracted value must be literally `Present` or `Absent`?
3. For `action: "keep"`, should the response omit `value_type`, `categories_or_unit`, `description`, `measurement_definition`, and `missing_value_rule`; repeat the current ontology fields; or include them as null values?
4. If `action: "revise"` is chosen only to clarify canonicalization, may the categories and value type remain unchanged while only the descriptive rules are revised? The wording implies this is allowed, but an explicit answer would remove doubt.
5. Does “explicit ... absence” include common negative assertions such as “no emphysema” and “without evidence of emphysema,” while absence of any mention remains null? Confirming this would make the mapping rule reproducible.

## Contradictions and tensions

There is no direct contradiction in the clinical rules. The main tension is that the failure is labeled `invalid_category`, yet the examples are ordinary positive synonyms for an already allowed category. The instruction says to revise only for a correctable ontology mismatch and to keep when the ontology is already appropriate, but it does not assign responsibility for canonicalizing synonyms. Both `keep` and a wording-only `revise` can therefore be defended under different unstated normalization policies.

The response template shows `categories_or_unit` as a one-element instructional placeholder (`["required for revise; empty only for unitless continuous"]`), while another rule requires exactly two distinct categories for a binary revision. A human can recognize the former as schema guidance rather than a literal allowed value, but this is fragile machine-facing notation and should be represented as an actual schema or separated from the example payload.

The description “Documented emphysema” can be read as a presence-only concept, while the declared measurement includes both explicitly documented presence and explicitly documented absence. This is not logically inconsistent, but it creates avoidable ambiguity about what the scalar represents.

## Identifier, index, format, and bookkeeping checks suitable for Python

Python can perform deterministic checks without deciding the clinical policy:

- parse both the outer message file and the JSON string embedded in the user message;
- verify that there is exactly one feature object and that `feature_id` is present and unchanged across any before/after record;
- validate `action` against `{keep, revise}` and `value_type` against `{binary, categorical, continuous, ordinal}`;
- enforce conditional required fields for `revise` and whatever explicit field policy is selected for `keep`;
- ensure a binary revision has exactly two nonempty, distinct scalar categories, and categorical or ordinal revisions have at least two;
- ensure `categories_or_unit` is an empty list only for a unitless continuous measurement, once the intended representation of units is specified;
- detect duplicate categories after trimming and, if desired by policy, case normalization;
- verify that the response is one JSON object with no prose or Markdown surrounding it;
- check that no forbidden feature-level keys imply renaming, adding, dropping, merging, splitting, or causal-role changes;
- confirm that counts such as `patient_count` are nonnegative integers and that every failure pattern has a declared kind, reason, allowed-category list, and example-value list;
- compare revised and original fields so a claimed `revise` changes at least one permitted ontology field, while preserving the feature identifier and name;
- optionally flag output examples that are not exact members of `allowed_categories`, while leaving semantic synonym mapping to the clarified policy rather than guessing it.

There are no array indices or cross-record identifier references in this instance that require reconciliation beyond confirming the single feature identity and the integrity of the failure-pattern list.
