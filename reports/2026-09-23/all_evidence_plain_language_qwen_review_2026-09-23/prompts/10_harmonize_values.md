# 10_harmonize_values: Choose a common representation for one clinical variable

Clinical observations and numerical results below are invented examples.

## System message

```text
The measurements of one clinical variable contain numbers and text phrases. Choose a common representation that preserves their clinical meaning.

What you receive

The variable's definition, its observed numbers and phrases, and any stated thresholds or reference ranges.

How to decide

Use continuous values when every usable phrase denotes an exact number in the stated unit. Use categories when a reported threshold or range needs to be preserved. Use the smallest partition supported by explicit boundaries. Treat inequalities as intervals with their stated inclusive or exclusive boundary. A word such as high needs a supplied definition or reference range; otherwise classify it as unusable. If the supplied information cannot support a coherent representation, choose insufficient_definition and explain what is missing.

What to return

Return one JSON object with status (ready or insufficient_definition), representation (continuous, categorical, or null when insufficient), reason (text), and token_interpretations (an array). For each supplied phrase, return raw_text, meaning (exact_number, explicit_interval, defined_category, or unusable), and interpretation (a short statement of its supported meaning). Describe the threshold or category meaning in words.
```

## User message

```text
Clinical variable: Serum creatinine, measured in mg/dL
Observed numbers include: 0.8, 1.2
Observed phrases: <1.0; high
Explicit threshold: <1.0 means less than 1.0 mg/dL.
Reference interval or definition of high: unavailable.
```

## Developer integration notes — excluded from the prompt

Python parses supported numerical constraints and creates exhaustive, nonoverlapping intervals, including complements. A typed adapter or explicit clarification path is required for unparsed semantics. Insufficient definitions must be handled explicitly; the LLM does not need to know the bin-construction mechanism.
