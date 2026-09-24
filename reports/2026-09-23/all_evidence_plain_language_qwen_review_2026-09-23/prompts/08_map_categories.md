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
