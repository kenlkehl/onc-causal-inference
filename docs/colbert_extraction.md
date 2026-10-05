# ColBERT feature extraction

New all-evidence runs and standalone explicit-feature extractors default to
`colbert`. OCI implements the encoder, token chunking, exact MaxSim retrieval,
cache, and extraction integration itself. The reference MatchMiner repository
is not imported, installed, or required. Existing Torch, Transformers,
Hugging Face Hub, and safetensors dependencies are sufficient.

The entire prepared text of each patient is chunked with the retrieval model's
tokenizer and embedded locally. Each feature's measurement definition becomes
a query. ColBERT scores a chunk by summing each query token's maximum cosine
similarity to any document token. The top chunks for each feature are combined
in original source order; overlapping spans are merged to avoid double counting.
The answering LLM receives these verbatim excerpts with their source offsets.
Retrieval never searches another patient's text or uses outcomes, treatment
assignments, causal roles, or held-out measurements to fit an index.

Stage 2 retains its existing endpoint pool, per-server admission limits,
validation/repair, deferred retries, ontology refinement, scalar reconciliation,
and quotation checks for mode variables. Patient/feature tasks run concurrently.
Oversized retrieved contexts use the existing lossless paging/serial machinery;
the standalone extractor raises if an explicit character bound is exceeded.
No selected evidence is silently truncated. Counts, modes, earliest/latest
values, and missingness describe the retrieved evidence; retrieval can miss
relevant evidence elsewhere. Choose `full_record` when exhaustive review is needed.

## Configuration

For the production workflow, put these fields in `stage2`:

```json
{
  "extraction_context_strategy": "colbert",
  "colbert": {
    "model_name": "lightonai/GTE-ModernColBERT-v1",
    "revision": null,
    "devices": ["auto"],
    "workers_per_device": 1,
    "cache_dir": ".oci_cache/colbert",
    "chunk_size": 64,
    "chunk_overlap": 0,
    "query_length": 512,
    "batch_size": 32,
    "score_batch_size": 32,
    "top_k": 20
  }
}
```

The same `colbert` object and `extraction_context_strategy` field are supported in
`explicit_features` for standalone/agentic workflows. Python callers can pass
`colbert=ColBERTConfig(...)` to `VLLMFeatureExtractor` or
`extract_explicit_features`. Compatible checkpoints contain a Transformer and
a trained `1_Dense` identity projection, with ColBERT query/document markers,
punctuation masks, and query expansion settings. Other checkpoint layouts fail
explicitly. Pin a Hub `revision` for reproducible runs. Oversized feature queries
raise rather than silently truncating; increase `query_length` within the
checkpoint's position capacity.

`auto` uses all visible CUDA devices, or CPU when CUDA is unavailable. Explicit
devices, such as `["cuda:0", "cuda:1"]`, use logical indices after
`CUDA_VISIBLE_DEVICES`. A shared pool keeps `workers_per_device` encoder workers
per device (default 1) for a retriever configuration across concurrent folds and
patient tasks. Each worker has its own encoder, serial task queue, and CUDA stream,
so multiple workers on a GPU can overlap work. Embedding batches and MaxSim scoring
execute on those workers independently of the LLM servers. Each worker also has
its own query cache and a patient-vector cache bounded to 16 patients and 128 MiB
of combined host/GPU data. The content-addressed disk cache is shared; concurrent
workers still encode a new patient index only once. Increasing the worker count
reuses existing indexes and extraction checkpoints. CPU devices support the same
worker-count setting, without CUDA streams.

Additional workers use more model/workspace memory and compete with the answering
model for GPU compute. Benchmark throughput when increasing the count; extra
workers do not guarantee a proportional speedup. Reserve sufficient GPU memory
for the encoders alongside managed vLLM servers, select dedicated GPUs, or set
`["cpu"]`. In separate OCI processes,
assign disjoint device lists to avoid loading duplicate replicas on a GPU.

On this RTX PRO 6000 run, a 2026-10-05 probe replayed 40 saved patient/feature
retrievals on GPU 7 while the pipeline continued. Scores, rankings, and rendered
excerpts matched the saved audits exactly with 1, 2, 4, and 8 workers. The first
worker's process footprint includes CUDA/library overhead shared by subsequent
workers; each additional warmed worker added about 668 MiB (0.65 GiB).

| Workers on one GPU | Total probe VRAM (GiB) | Retrievals/s with uncached queries |
| --- | ---: | ---: |
| 1 | 1.92 | 94 |
| 2 | 2.58 | 126 |
| 4 | 3.88 | 108 |
| 8 | 6.49 | 92 |

These are retrieval-only measurements with warm patient indexes, three timed
repeats of 160 requests, and the existing pipeline sharing that GPU. Eight
workers fit comfortably, but two were fastest when queries needed encoding.
With query vectors also cached, one worker was fastest (about 1,032 retrievals/s).
Whole-pipeline speed must be measured separately; this probe does not establish
an end-to-end speedup.

## All-in-one launchers

The RTX PRO 6000 x8 preset uses Gemma 4 26B-A4B for both Stage 2 roles.
When the primary and extraction model IDs match and both pools are managed,
the launcher starts one shared pool over the union of their configured GPUs.
Extraction and interpretation share the same request router; no model reload
or four/four fallback occurs. Different explicitly configured model IDs retain
the existing alternating-model workflow.

Extraction starts with `enable_thinking=false`. The first five validation repairs
include the previous response and the concrete error, with thinking still off.
If those repairs fail, the sixth repair and later repairs enable thinking. The
default `thinking_after_response_repairs` is 5. Category-mapping repairs also begin
without thinking; ordinary aggregate interpretation keeps its separate policy.

To resume extraction with a new interpretation model while reusing completed
definitions, set `frozen_feature_definition_model` to the original definition
model ID. This preserves their original input fingerprints and records their
provenance separately from the current serving model. Every outer fold must
already have matching completed definitions; missing or incompatible upstream
checkpoints fail rather than regenerate under the old model ID. Remove existing
extraction-and-later artifacts before changing a saved run's serving identity.

Every `run_*.sh` in the repository root delegates to the shared launcher and
defaults to ColBERT. For example:

```bash
STAGE2_COLBERT_DEVICES=cuda:0,cuda:1 \
STAGE2_COLBERT_WORKERS_PER_DEVICE=4 \
STAGE2_COLBERT_CACHE_DIR=/persistent/oci-colbert-cache \
STAGE2_COLBERT_TOP_K=20 ./run_one_conf_one_mod.sh
```

Supported overrides are `STAGE2_EXTRACTION_CONTEXT_STRATEGY` and
`STAGE2_COLBERT_MODEL`, `REVISION`, `DEVICES`, `WORKERS_PER_DEVICE`, `CACHE_DIR`,
`CHUNK_SIZE`, `CHUNK_OVERLAP`, `QUERY_LENGTH`, `BATCH_SIZE`, and `TOP_K` (each with the
`STAGE2_COLBERT_` prefix). The Python CLI exposes matching
`--stage2-colbert-*` options and `--stage2-extraction-context-strategy`.
Advanced settings can also be supplied with `--set stage2.colbert.KEY=VALUE`.

Explicit saved-run launches preserve their saved settings. Device, worker-count,
batch-size, and cache-location overrides are permitted on resume. Switching retrieval
models, top K, geometry, or measurement methods requires fresh measurement
outputs; checkpoints refuse to mix incompatible results. For an older full-record
run whose saved config omits the selector, explicitly add
`"extraction_context_strategy": "full_record"` before resuming. The legacy
standalone strategies `tail`, `contract_lexical_rag`, and `complete_paged_v1`
remain available. An explicitly enabled `extraction_note_search` retains its
existing route.

## Reusable disk cache and provenance

The cache is shared across cohorts, folds, feature definitions, and answering
LLMs. Patient entries are addressed by the exact source-text hash, model artifact
checksum, encoder version, and encoding geometry. Every text is indexed in full,
but each retrieval searches only that exact text's entry. Changing the question,
top K, answering server, batch size, or device reuses document vectors. Changing
source text, checkpoint weights/tokenizer, or chunk geometry creates a new entry.

Files under `cache_dir/patients` contain source chunks and token vectors. They are
owner-readable/writable, locked across processes, written by atomic rename,
checksummed, and loaded with pickle disabled. Corrupt entries are rebuilt from
the source. Concurrent requests for the same text encode it only once. The cache
directory is relative to the working directory unless an absolute path is given;
root launchers use the repository's `.oci_cache/colbert` by default.

Each Stage 2 patient/feature task saves `retrieval.json`, including its cohort
row ID, cache key/path, model signature, query strings, ranked scores, exact
original source offsets, and the context actually supplied to extraction.
Nested page/quotation offsets refer to that rendered context; the retrieval
audit maps its excerpts back to the original prepared record. These are patient
data artifacts and belong in the same controlled storage as other extraction
outputs. `.oci_cache/` is ignored by Git.
