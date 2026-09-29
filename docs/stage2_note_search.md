# Optional Stage 2 extraction through a Python note-search REPL

This experimental path keeps one patient's full text in a local, isolated Python
worker. The extraction model searches it with Python and reads bounded excerpts
before returning measurements for the current feature batch. It may reduce input
tokens for long records, but it adds search requests and can miss evidence. OCI
does not yet have a measured speed or accuracy advantage for this path.

## Enable it

Add the following settings under `stage2` in a run configuration:

```json
{
  "extraction_note_search": {
    "enabled": true,
    "source_checkout": "/path/to/matchminer-ai-inference-kehl",
    "max_cells": 3
  },
  "extraction_feature_batch_size": 10,
  "extraction_reasoning_effort": "none"
}
```

The source checkout must contain
`src/matchminer_ai/patients/_note_repl.py` and `_note_repl_worker.py`. The checkout
used during integration is
`/ksg/kehl_mm_data/mmai/v23/matchminer-ai-inference-kehl`. OCI loads that worker
unchanged, records hashes of both backend files, and uses its existing isolation.
It does not copy or redistribute MatchMiner-AI's implementation. That external
repository carries its own CC-BY-NC-ND-4.0 license.

Alternatively, install the MatchMiner-AI checkout containing the note-search
worker in OCI's Python environment and omit `source_checkout`. Linux and
`libseccomp.so.2` are required. Worker isolation must succeed before any generated
Python code runs; there is no unrestricted execution fallback. The standard
extraction path does not require this dependency.

Use the existing Stage 2 configuration entry points, including
`scripts/run_stage2_from_artifacts.py`; there is no separate inference server or
new endpoint configuration. The feature-batch limit remains configurable through
`extraction_feature_batch_size`. Larger batches increase the definitions, search
coverage, and output required per call; 500 fields per request have not been
validated for this path.

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
   next turn. It does not ask the model to copy quotations or manage citations.
4. The model can search again for a specific gap or return a value map using
   clinical variable names. The normal short path is one search request and one
   final extraction request. The default permits at most three executed search
   cells; after that, the next request must return measurements.
5. OCI validates the same scalar values, allowed categories, threshold forms,
   missingness rules, and repeated-observation rules as its standard extraction
   path. Existing field-specific repair can remove a persistently invalid field
   and recover the remaining measurements from the same retrieved evidence.
6. Patient/feature results enter the same ontology supervision, missingness
   filter, estimand refinement, statistical selection, and estimation stages.
   The chosen method also applies to alternative representations and held-out
   measurements.

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
| `max_output_chars` | 12,000 | Worker output limit per cell |
| `max_evidence_chars` | 32,000 | Total retained source text supplied on subsequent turns |
| `max_memory_chars` | 6,000 | Model's factual notebook limit |
| `max_code_chars` | 6,000 | Python source limit per cell |
| `max_history_bytes` | 32,000,000 | Maximum UTF-8 input record size |
| `cell_timeout_seconds` | 5 | Wall-time limit per Python cell |
| `worker_memory_mb` | 512 | Worker address-space limit |

The existing extraction character and token budgets also apply, preserving the
output reservation. Oversized state fails explicitly rather than truncating the
patient record and claiming a complete search. Excerpts omitted by configured
limits are flagged. The entire source text is not tokenized merely to plan a
search call.

Missing dependencies, failed isolation, worker timeouts, and a search with no
successful cells propagate as errors. They do not imply absent documentation or
become an all-null patient. A successful search that finds no supporting evidence
can return null. As with standard extraction, failed final-value validation may
produce audited conservative nulls after the configured repairs.

Each active patient/feature batch has a worker containing that patient's full
text. Memory use grows with concurrent workers. There is currently no shared
search cache across different feature batches.

## Checkpoints and evaluation

Normal result/completion files remain in the extraction tree. A scalar batch
also has a `note_search/<input fingerprint>/` directory containing:

- `cells/cell_NNN.json`: completed search actions, used to rebuild Python state
  locally after a restart without requesting those search plans again;
- `reviewed_excerpts.json`: code-derived original text spans supplied as context,
  with an omission flag; these are retrieved context, not individually verified
  supporting citations;
- `status.json`: method/scope, executed and successful cells, logical requests
  during this invocation, elapsed time, retained text size, and failure type.

`measurement_method.json` records method, limits, and worker hashes. Ordinary
request-event logs retain endpoint requests and reported usage, including
repairs. Checkpoints containing excerpts or search notebooks are patient-bearing
artifacts and belong beside the other extraction artifacts.

Before a production switch, compare matched patients and frozen definitions
against full-record extraction. Measure factual accuracy, missingness, repeated-
observation conflicts, input/output tokens, repairs, elapsed time, and sustained
throughput. Faster selected-excerpt reading alone does not establish equivalent
measurement sensitivity.

The deterministic integration tests use synthetic records and scripted model
responses. The external worker test can be enabled with:

```bash
OCI_TEST_NOTE_SEARCH_CHECKOUT=/path/to/matchminer-ai-inference-kehl \
  python -m pytest -q tests/test_stage2_note_search.py
```
