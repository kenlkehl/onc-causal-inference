# Naive review of request 08: map categories

## Restatement of the task

The request asks the model to normalize a previously extracted categorical value to a closed category ontology. It must use only the supplied feature definition, allowed categories, and prior extracted value; it must not perform new clinical extraction or infer new patient facts. For each supplied `mapping_id`, it must return exactly one correction whose `value` is either one exact allowed category or `null` when no unambiguous mapping exists. The response must be JSON only.

## Input

There is one mapping item:

- `mapping_id`: `category_mapping_0000`
- feature: `emphysema`
- description: documented emphysema
- measurement definition: explicit pretreatment documentation of emphysema presence or absence
- allowed categories: `Present`, `Absent`
- prior extracted value: `present on CT`
- missing-value policy: return null if unreported; silence is not absence
- value type: binary
- occurrence count: 3

## Expected output

The expected response is a JSON object with a `corrections` array. That array must contain exactly one object for `category_mapping_0000`. Its `value` must be exactly `Present`, `Absent`, or `null`, subject to the semantic-mapping rule. No extra mapping IDs or surrounding prose are allowed.

## Can this be executed without consequential hidden assumptions?

Yes. The prior extracted value `present on CT` explicitly states presence and maps directly and unambiguously to the exact allowed category `Present`. This does not require extracting a new fact or treating silence as absence. The anatomical/imaging qualifier `on CT` does not conflict with the category; it supplies context for the already extracted presence statement.

No consequential policy choice is missing for this specific input. The measurement definition mentions pretreatment documentation, but the task explicitly limits the operation to mapping the already extracted value and forbids new clinical extraction. Therefore, the mapper should not independently re-evaluate timing from unavailable source text.

## Output-changing clarification questions

None are required for this item. The supplied value has one clear semantic equivalent in the declared ontology.

If this prompt is intended as a reusable interface rather than only for this concrete item, one non-blocking schema clarification would improve consistency: should the top-level output be exactly `{"corrections": [...]}`? The supplied `response` example strongly implies that shape, but labels its contents with descriptive placeholders rather than presenting a fully typed schema. This does not change the result for the present request.

## Contradictions or ambiguities

There is no substantive contradiction in the instructions.

Minor interface observations:

- The system says to normalize to a closed ontology, and the item provides that ontology explicitly as `Present` and `Absent`; these agree.
- The rule permitting `null` does not make the ontology open-ended. It represents an unmappable or ambiguous prior value rather than a third category.
- `occurrence_count` is supplied but no rule says it affects category selection. Using it to alter the mapping would violate the instruction to use only the feature definition, allowed categories, and prior extracted value.
- The item supplies `missing_value_rule`, yet the first rule says to use only the feature definition, allowed categories, and prior extracted value. For this input the distinction is immaterial because the prior value is reported and unambiguous. For a reusable prompt, this wording could be tightened because the null behavior also depends on the stated rules and missing-value policy.
- The response template uses prose placeholders (`one supplied mapping_id`, `one exact allowed category or null`) rather than literal values. They are readily understood as placeholders, but a formal JSON schema would remove that small ambiguity.

## Identifier, index, and format bookkeeping suitable for Python

A deterministic validator could:

1. Parse the outer conversation JSON and then parse the user message's JSON string.
2. Collect all input `mapping_id` values and verify that each is unique.
3. Verify that the output is valid JSON with no surrounding prose.
4. Verify that the top-level object contains the expected `corrections` array.
5. Check that every supplied `mapping_id` appears exactly once in `corrections`, with no omissions, duplicates, or additional IDs.
6. Check that each correction contains the required fields `mapping_id` and `value`.
7. Check that each non-null `value` exactly matches one of that item's `allowed_categories`, including capitalization and spelling.
8. Check that `null` is encoded as JSON `null`, not the string `"null"`.
9. Preserve the identifier exactly as `category_mapping_0000`; its numeric-looking suffix should not be parsed, renumbered, or reformatted.
10. Optionally preserve input item order in the output for deterministic serialization, although order is not stated to carry semantic meaning.

No array index is supplied or required. The stable join key is `mapping_id`, not position or `occurrence_count`.
