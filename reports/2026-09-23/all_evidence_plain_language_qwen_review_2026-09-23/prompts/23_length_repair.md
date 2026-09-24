# 23_length_repair: Complete a truncated extraction response

Clinical observations and numerical results below are invented examples.

## System message

```text
Read the patient's clinical text and report a value for each listed clinical variable.

What you receive

The clinical variable definitions and the text from one patient's record.

How to decide

Follow each variable's definition. Use the wording in the patient's text to determine its value. An explicit date belongs to a finding when the sentence or record heading links them. An undated CT sentence therefore keeps an unknown date even when it follows a dated laboratory result.

Use the defined category labels exactly. Write numerical measurements as JSON numbers in the required unit; a definition may also allow a reported threshold string. Use null according to the variable's missing-value rule.

What to return

Return one JSON object whose top-level keys are exactly the supplied clinical variable names. Each value is a scalar or null. Include every requested variable.
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

Patient's clinical text

2025-01-01: serum creatinine 1.0 mg/dL. 2025-01-10: serum creatinine 1.2 mg/dL. CT documents emphysema.
```

## Assistant message

```text
{"Serum creatinine":1.2,"Emphysema on imaging":
```

## User message

```text
Your preceding answer ended before the JSON object was complete. Return a complete JSON object using the clinical text and definitions above. Include every requested variable and its supported scalar value or null.
```

## Developer integration notes — excluded from the prompt

Python retains the original task/input, formats precise errors, controls reasoning and retry budgets, and validates the complete response. Split requests or increase output space when necessary; the repair prompt does not explain those mechanics.
