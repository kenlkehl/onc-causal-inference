# 02_audit_unmapped: Review excerpts for overlooked clinical variables

Clinical observations and numerical results below are invented examples.

## System message

```text
You identify clinical variables that could be measured from individual patients' medical records. Read all supplied excerpts and name each distinct clinical attribute they support.

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

Return each distinct attribute once. An excerpt containing several findings can support several candidates. Use {"candidates": []} when the text supports none.
```

## User message

```text
Clinical excerpts

2025-01-01: serum creatinine 1.0 mg/dL. 2025-01-10: serum creatinine 1.2 mg/dL. CT documents emphysema.
```

## Developer integration notes — excluded from the prompt

Use one evidence card per call. Attach provenance and machine names in Python. The response schema matches the adopted discovery validator. The caller determines when a second review is needed.
