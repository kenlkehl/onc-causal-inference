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
