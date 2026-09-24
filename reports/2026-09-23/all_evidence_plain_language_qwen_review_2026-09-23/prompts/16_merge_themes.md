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
