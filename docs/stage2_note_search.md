# Optional Stage 2 extraction through a Python note-search REPL

This experimental path keeps one patient's full text in a local, isolated Python
worker. The extraction model searches it with Python and reads bounded excerpts
before returning measurements for the current feature batch. It may reduce input
tokens for long records, but it adds search requests and can miss evidence. OCI
has not established a general speed or accuracy advantage for this path.

## Enable it

Add the following settings under `stage2` in a run configuration:

```json
{
  "extraction_note_search": {
    "enabled": true,
    "max_cells": 3
  },
  "extraction_feature_batch_size": 10,
  "extraction_reasoning_effort": "none"
}
```

OCI includes its own note-search worker. No MatchMiner package or checkout is
needed. The subprocess uses Python's standard library plus Linux's
`libseccomp.so.2` to enforce a default-deny syscall policy. Filesystem opening,
network access, process creation and program execution are unavailable to generated
code. Memory, output and wall-time limits also apply. Isolation must succeed
before any generated Python runs; there is no unrestricted fallback. The standard
full-record extraction path does not require libseccomp.

Remove the obsolete external-checkout field from older run configurations;
unknown options are rejected. Saved measurements record hashes of OCI's transport
and worker, and the extraction schema version changed for this backend. Previous external-worker measurements
require a fresh output directory; they are not silently combined with new results.

Use the existing Stage 2 configuration entry points, including
`scripts/run_stage2_from_artifacts.py`; there is no separate inference server or
new endpoint configuration. The feature-batch limit remains configurable through
`extraction_feature_batch_size`. Larger batches increase the definitions, search
coverage, and output required per call; 500 fields per request have not been
validated for this path.

For one question at a time, set `extraction_feature_batch_size: 1`. Each
patient/variable pair then has its own search, provisional answer, and focused
review; the model sees only that variable's definition. The setting above uses
ten variables per batch. The September 29 smoke comparisons used fifty and do
not establish performance for single-variable extraction.

Use a fresh Stage 2 measurement output for a comparison. Existing compatible
evidence/discovery/definition artifacts can be retained or copied separately.
Changing the extraction method, search limits, or worker code in a directory
containing measured values raises an explanatory error. It never deletes or
silently adopts those measurements. Enabling this option does not resume a
paused experiment by itself.

## How a batch runs

1. OCI groups the declared definitions using its existing feature-batch limit.
   Each worker receives only one patient's clinical text. Treatment, outcomes,
   oracle values, and other patients are absent from the search input.
2. The first model request contains the definitions and record length. It asks
   for a Python cell using `scan`, `search`, or `read`. The complete record stays
   local. A cell can search several measurements together, and Python variables
   persist across cells within this patient/feature batch.
3. The parent reconstructs retrieved source spans from its original text and
   supplies those excerpts, bounded output, and a short factual notebook on the
   next turn. Every excerpt, starting with the first search results, includes
   its original character range (end exclusive) and document-segment label.
   Segments come from `<new_note>` or form-feed separators. Offsets locate text;
   they imply no chronology. Python adds these labels. The model does not copy
   quotations or manage citations.
4. The model can search again for a specific gap or return a value map using
   clinical variable names. The normal short path is one search request and one
   provisional extraction request. The default permits at most three executed search
   cells; after that, the next request must return measurements.
5. OCI validates the same scalar values, allowed categories, threshold forms,
   missingness rules, and repeated-observation rules as its standard extraction
   path. Existing field-specific repair can remove a persistently invalid field
   and recover the remaining measurements from the same retrieved evidence.
6. With review enabled (the default within this optional path), a shared,
   patient-free vocabulary request generates up to eight literal search terms
   for each declared variable. Python caches that vocabulary across patients
   and searches each patient's whole string. It tracks which matching positions
   have already been supplied for each variable.
   When a provisional value is missing and the search finds zero matches,
   a second vocabulary request for that one variable proposes alternative
   wording using its definition and the unsuccessful terms. Python removes
   equivalent/repeated terms and searches the whole record again. This happens
   once per missing zero-match variable. Alternative vocabulary is shared across
   patients; values and patient passages are never shared. New passages enter
   the focused review below.
7. Missing values with matching passages, previously unseen matches, and
   model-flagged conflicts receive focused review. Each variable gets its own
   excerpt allowance. Requests contain only affected definitions, provisional
   values, review reasons, and labeled passages, including previously seen
   matches when available. At most two review passes run by default. Context
   widens on the second pass. If the expanded vocabulary also finds no evidence,
   the value remains missing. Zero matches do not support a negative finding.
   An invalid alternative-vocabulary response is audited and can use the
   existing bounded full-record fallback; an endpoint outage stops for recovery.
8. Remaining conflicts or incomplete retrieval coverage can invoke full-record
   extraction for at most five individual fields per patient/feature batch.
   This uses the existing lossless serial extraction path when the record
   exceeds the prompt budget. Failed review/fallback validation preserves the
   previous validated value and records the failure. An endpoint outage stops
   the batch for recovery. Unresolved fields beyond the budget remain audited.
9. Patient/feature results enter the same ontology supervision, missingness
   filter, estimand refinement, statistical selection, and estimation stages.
   The chosen method also applies to alternative representations and held-out
  measurements.

The follow-up adds no date parsing, date selection, or classification of evidence
as old. Existing declared measurement/conflict rules still apply. Literal search
terms improve coverage only for the vocabulary supplied; they cannot establish
that all clinically relevant evidence has been found. Set `max_review_passes: 0`
to compare the provisional search path without these follow-up steps.

The loop uses OCI's extraction request callback, so endpoint model detection,
publisher sampling profiles, reasoning settings, request admission/concurrency,
transport retries, streaming, and response repair remain in effect. Reasoning is
off by default; repeated validation failures may enable it under the existing
repair policy. No new worker pool bypasses the configured endpoint concurrency.
The search-cell cap is distinct from HTTP calls: transport retries, response
repairs, category mapping, and recovery of remaining fields can add calls under
their existing bounded budgets.

Mode-based variables retain the standard full-record observation path because
their most-frequent-value rule requires occurrence counts. Other conflict rules
still apply, but a searched subset may omit a relevant latest value, extreme, or
contradictory observation. Search results are therefore labeled
`scope: searched_excerpts`; they do not establish exhaustive clinical review.

## Limits and failures

| Setting | Default | Meaning |
|---|---:|---|
| `max_cells` | 3 | Maximum executed Python cells per patient/feature batch |
| `max_scan_patterns` | 128 | Patterns accepted by one `scan` call; Python searches all patterns and deduplicates exact matching spans |
| `max_review_passes` | 2 | Focused review passes after provisional extraction; 0 disables review and fallback |
| `retry_zero_match_missing` | true | One alternative-wording search when a provisional value is missing and the original vocabulary finds zero matches; requires review enabled |
| `review_features_per_request` | 10 | Maximum variables per focused review request |
| `review_hits_per_feature` | 4 | Matching positions sampled per variable per review |
| `review_context_chars` | 360 | Context on each side of a match; doubles on pass two, subject to the excerpt budget |
| `max_full_record_fallback_features` | 5 | Maximum individual fields sent through full-record fallback per patient/feature batch |
| `max_output_chars` | 12,000 | Worker output limit per cell |
| `max_evidence_chars` | 32,000 | Retained first-pass text budget and total excerpt budget per focused review request |
| `max_memory_chars` | 6,000 | Model's factual notebook limit |
| `max_code_chars` | 6,000 | Python source limit per cell |
| `max_history_bytes` | 32,000,000 | Maximum UTF-8 input record size |
| `cell_timeout_seconds` | 5 | Wall-time limit per Python cell |
| `worker_memory_mb` | 512 | Worker address-space limit |

The existing extraction character and token budgets also apply, preserving the
output reservation. Focused review groups split further to fit these budgets.
Oversized single-variable state fails explicitly rather than truncating the
patient record and claiming a complete search. Excerpts omitted by configured
limits are flagged. The entire source text is not tokenized merely to plan a
search call.

Increasing the pattern limit does not increase the returned-hit, source-text,
output, memory, or execution-time budgets. `scan` still returns at most its
`limit` argument (2–20 hits), sampled across the record. `match_count` counts
distinct matched character spans exactly, including when several patterns match
the same span. Source-span overflow and omitted
excerpts remain flagged. A larger pattern allowance alone does not guarantee
that every requested clinical variable has adequate evidence. Dedicated literal
searches count all matches and retain at most 512 positions per variable (first
256 and last 256); omitted positions remain flagged and eligible for fallback.

Unavailable isolation, worker timeouts, and a search with no
successful cells propagate as errors. They do not imply absent documentation or
become an all-null patient. A successful search that finds no supporting evidence
can return null. As with standard extraction, failed final-value validation may
produce audited conservative nulls after the configured repairs.

Each active patient/feature batch has a worker containing that patient's full
text. Memory use grows with concurrent workers. There is currently no shared
patient-text search cache across different feature batches. Vocabulary plans
are shared for identical definitions and request identities, including when
patients run concurrently.

## Checkpoints and evaluation

Normal result/completion files remain in the extraction tree. A scalar batch
also has a `note_search/<input fingerprint>/` directory containing:

- `cells/cell_NNN.json`: completed search actions, used to rebuild Python state
  locally after a restart without requesting those search plans again;
- `reviewed_excerpts.json`: code-derived original text spans supplied as context,
  with an omission flag; these are retrieved context, not individually verified
  supporting citations;
- `provisional.json`: validated initial measurements, retrieved spans, and review
  flags; a resumed review does not repeat initial extraction;
- `review/pass_*/group_*/complete.json`: completed focused reviews with input
  fingerprints, previous values, results, reasons, and supplied passages;
- `review/full_record/*/review_complete.json`: individual fallback results and
  failure status;
- `review/complete.json`: final values, changes by pass, coverage counts,
  initial/alternative terms and match counts for zero-match retries, fallback
  fields, and unresolved/coverage-limited fields;
- `status.json`: method/scope, executed and successful cells, logical requests
  during this invocation, elapsed time, retained text size, and failure type.

Shared vocabulary lives under `note_search_plans/` at the extraction root.
Alternative vocabulary checkpoints live in its `zero_match/` subtree. A restart
reuses completed vocabulary requests before resuming the patient-specific review.
`measurement_method.json` records method, limits, prompt identity, and worker hashes. Ordinary
request-event logs retain endpoint requests and reported usage, including
repairs. Checkpoints containing excerpts or search notebooks are patient-bearing
artifacts and belong beside the other extraction artifacts.

Before a production switch, compare matched patients and frozen definitions
against full-record extraction. Measure factual accuracy, missingness, repeated-
observation conflicts, input/output tokens, repairs, elapsed time, and sustained
throughput. Faster selected-excerpt reading alone does not establish equivalent
measurement sensitivity.

The deterministic integration tests use synthetic records and scripted model
responses. Native worker tests exercise isolation, timeouts, output/memory bounds,
source provenance and absence of any external-checkout requirement:

```bash
python -m pytest -q tests/test_note_search_worker.py \
  tests/test_stage2_note_search.py tests/test_stage2_note_search_review.py
```
