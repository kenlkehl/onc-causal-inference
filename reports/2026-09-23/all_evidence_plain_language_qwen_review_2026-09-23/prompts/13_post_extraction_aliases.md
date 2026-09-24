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
