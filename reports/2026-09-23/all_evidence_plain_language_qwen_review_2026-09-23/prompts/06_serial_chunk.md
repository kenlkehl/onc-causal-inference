# 06_serial_chunk: Update values after reading more of a patient's record

Clinical observations and numerical results below are invented examples.

## System message

```text
Update the listed clinical measurements using the next section of a patient's record.

What you receive

The measurement definitions, the values found so far, notes about those values, and the next section of clinical text. The notes may include the date of an earlier observation or an unresolved conflict.

How to decide

Follow each variable's definition. Use the wording in the patient's text to determine its value. An explicit date belongs to a finding when the sentence or record heading links them. An undated CT sentence therefore keeps an unknown date even when it follows a dated laboratory result.

Use the defined category labels exactly. Write numerical measurements as JSON numbers in the required unit; a definition may also allow a reported threshold string. Use null according to the variable's missing-value rule.

Keep an earlier value when the new text adds nothing that changes it under its definition. A prior null means that no usable value has been found yet. For latest/earliest, a dated observation takes precedence over an undated observation; otherwise compare dates and then text order. The new section comes later in text order. Retain the selected value's date, when known, in its decision note. For maximum/minimum, retain the selected extreme; for any_positive, retain positive evidence; for single_or_null, retain unresolved conflicts.

What to return

Return one JSON object with values and decision_notes. Each is an object keyed by every supplied variable name. values holds the updated scalar or null. decision_notes holds a short statement of the date or other fact needed to choose among observations, or null. Keep each note brief and specific to choosing a value.
```

## User message

```text
Clinical variables

Serum creatinine
- Report the latest serum creatinine result in mg/dL.
- A directly reported threshold, such as <1.0, may remain a string.
- If several results have dates, use the latest dated result. Break a date tie using the last mention. If all results are undated, use the last mention.
- Use null for missing or unresolved results.

Emphysema on imaging
- Use Present for an explicit positive imaging finding and Absent for an explicit negative imaging finding.
- Use null for silence, uncertainty, or an unresolved finding.
- If findings conflict, use the latest dated finding. Break a date tie using the last mention. If all findings are undated, use the last mention.

Previously extracted values
- Serum creatinine: 1.0 mg/dL, dated 2025-01-01.
- Emphysema on imaging: null; no usable finding yet.

Next section of the record
2025-01-10: serum creatinine 1.2 mg/dL. CT documents emphysema.
```

## Developer integration notes — excluded from the prompt

Python keeps patient/chunk IDs and ordering. Route mode aggregation through occurrence extraction so Python can count observations. State translation and bookkeeping are excluded from the model instructions.
