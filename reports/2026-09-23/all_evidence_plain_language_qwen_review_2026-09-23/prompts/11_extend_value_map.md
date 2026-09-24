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
