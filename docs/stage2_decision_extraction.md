# Stage 2 decision extraction with Plumb

Stage 2 can measure features with `crh225/plumb-4b` instead of generating JSON
with the extraction LLM. Enable `stage2.decision_extraction.enabled`. The
`extraction_llm` configuration still names the extraction model, server, and
concurrency; the primary LLM continues to define/review ontologies and perform
the existing scientific reviews. The default backend for existing configurations
is unchanged.

## Root single-run launchers

For two self-managed models that fit together, enable
`stage2.vllm_co_resident_all_gpus: true` (CLI
`--stage2-vllm-co-resident-all-gpus 1`, launcher environment
`STAGE2_VLLM_CO_RESIDENT_ALL_GPUS=1`). Both models then stay resident on the union
of their configured GPU lists throughout Stage 2. Each model retains its
tensor-parallel width; the union must divide evenly by both widths. Replica
counts and disjoint HTTP/rendezvous ports are derived from that union. This
works with any dataset, fold count, or compatible number of selected GPUs and
honors logical indices after `CUDA_VISIBLE_DEVICES`.

Set explicit `--gpu-memory-utilization` arguments for both pools. Their fractions
must total at most 0.90, reserving room for retrieval and CUDA contexts. For
example, Gemma at `0.50` and Plumb at `0.28` use the same allocation on every
selected GPU. ColBERT retains its separately configured devices and workers.
The default allocation remains available when this mode is disabled.

Independent feature requests for ontology revision, mixed-value harmonization,
and role adjudication run concurrently within each fold using `stage2.workers`.
All folds share the primary model's global request limit and load-aware replica
router. Per-server admission also respects the configured vLLM sequence limit.
Refinement rounds and dependent theme-merging rounds remain ordered; feature
outputs and reports retain their original input order. Revision evidence samples
are collected in one bounded scan shared by the reviewed features, preserving
the existing sampling order and character budgets. Worker counts and serving
allocation do not enter prompt or measurement identities, and existing feature
checkpoints remain reusable after a restart.

Fresh runs through `run_one_conf_one_mod.sh`, `run_five_conf_five_mod.sh`, and
their H100/RTX PRO 6000 variants default to full-record LLM extraction with ten
variables per request. Set `STAGE2_DECISION_EXTRACTION=1` to opt into Plumb.
Plumb uses ColBERT excerpts, one feature per prompt, the 3000-token cap, three
numeric narrowing passes, and independent ±5% verification described below.
The primary model still defines and reviews ontologies. For an explicit Plumb run:

```bash
STAGE2_DECISION_EXTRACTION=1 ./run_five_conf_five_mod.sh /path/to/plumb-run
```

With Plumb selected, the base one/five scripts expect an external primary server
on port 8010 and a Plumb classification server on port 8020. Configure
`STAGE2_EXTRACTION_ENDPOINT` and `STAGE2_EXTRACTION_MODEL` to match an existing
classifier, or use the managed extraction `STAGE2_EXTRACTION_VLLM_*` settings.
The external-server example below uses port 8134 and served name `plumb-4b`:

```bash
STAGE2_EXTRACTION_ENDPOINT=http://127.0.0.1:8134/v1 \
STAGE2_EXTRACTION_MODEL=plumb-4b \
./run_five_conf_five_mod.sh /path/to/plumb-run
```

The eight-GPU presets first run Gemma 4 26B replicas across all eight GPUs for
feature discovery and initial ontology preparation. Numeric bounds and ambiguous
types are prepared concurrently from feature contracts, with a checkpoint per
feature. All folds finish this phase before patient extraction starts. Gemma on
GPU 0 stays loaded while the temporary Gemma replicas on GPUs 1–7 stop and Plumb
starts, with tensor parallelism of one. H100 uses seven Plumb replicas on GPUs
1–7 and the Red Hat AI FP8-dynamic Gemma checkpoint on GPU 0. RTX PRO 6000 uses
eight Plumb replicas on GPUs 0–7; its NVIDIA NVFP4 Gemma server shares GPU 0
with Plumb and uses 50% GPU-memory allocation. Completed preparation checkpoints
are reused on restart; once all are complete, the run starts directly with this
configured allocation. Extraction allows 128 concurrent requests across the replicas;
each Plumb scheduler allows up to eight sequences within its token budget.
Plumb uses a 3072-token server window, bf16, eager mode,
and 28% GPU-memory allocation per replica; the primary model retains its
existing long-context preset. Override `STAGE2_EXTRACTION_VLLM_EXTRA_ARGS_JSON`
to change Plumb's serving resources. The backend supplies the readout conversion
arguments automatically. Managed decision launches sync both the `local-llm`
and `decision-extraction` extras unless `OCI_PYTHON` is supplied.
Model, GPU allocation, and worker environment overrides remain available.

Each retrieval worker reuses validated patient indexes and their padded MaxSim
document batches in a bounded memory cache: at most 16 patients and 128 MiB of
combined host vectors and GPU buffers per worker. A changed or replaced index
file invalidates its memory entry and goes through the existing checksum and
source-span validation. The scoring operations, padding masks, and ranking order
remain the same. Prompt packing tokenizes candidate prefixes in one batch, keeps
the original first-over-budget stopping rule, and sends the chosen token IDs
directly to Plumb without tokenizing the final prompt again.

Feature-query embeddings share one host-memory cache across all retrieval
workers and outer folds using that pool. The cache is scoped to the pool's
encoder/checkpoint signature and exact feature query text; patient text never
enters it. Concurrent misses for the same query encode it once and share the
result. Its LRU vector budget defaults to 1 GiB; configure
`stage2.colbert.query_cache_max_bytes`, `STAGE2_COLBERT_QUERY_CACHE_MAX_BYTES`,
or `--stage2-colbert-query-cache-max-bytes` (bytes; `0` disables retention).
Keys and array bookkeeping add a small amount of host memory above this budget.
The setting is excluded from encoding and measurement identities, so completed
extractions and patient indexes remain compatible. Aggregate cache counters
(hits, misses, coalesced requests, encodings, evictions, and retained bytes) are
logged once per minute during retrieval.

Set `stage2.decision_preparation_workers` (or
`STAGE2_DECISION_PREPARATION_WORKERS`,
`--stage2-decision-preparation-workers`) to a positive integer to prepare prompts
across CPU cores. For example, `STAGE2_DECISION_PREPARATION_WORKERS=16` creates
one pool of 16 processes shared by all outer folds. The default `0` prepares
prompts in the main process. Each child loads only the configured tokenizer,
with CUDA disabled and one native thread; source rendering, chat formatting,
and tokenization run in that child. The processes use the `spawn` start method
and shut down when the Stage 2 driver exits. Classifier routing and its global
request limit remain in the parent. This setting does not change prompt tokens,
retrieval rankings, measurement identities, or checkpoint compatibility.

Explicit saved-run launches (`OCI_RUN_CONFIG` with `STAGE2_ONLY=1`, including
preflight and reselection) preserve the saved backend and model configuration.
The new defaults apply to fresh configuration construction; choose a fresh
output directory when switching an existing run to Plumb. Other JSON-based
entry points keep decision extraction opt-in.

## Run configuration

[`research_all_evidence_plumb.json`](../example_configs/research_all_evidence_plumb.json)
is a full workflow example. Set its dataset, output directory, and primary LLM
endpoint for your study. It starts a separate managed Plumb server on GPU 1,
uses CPU ColBERT retrieval, and leaves the primary endpoint externally managed.
The example's 0.28 GPU-memory allocation was tested alongside the existing worker
on this machine's A6000. GPU numbers follow `CUDA_VISIBLE_DEVICES`.

```bash
python scripts/run_all_evidence.py --config example_configs/research_all_evidence_plumb.json
```

Use a **fresh output directory** when switching extraction backends, retrieval
settings, or decision policy. Existing generative-extraction measurements are
not interchangeable with decision measurements. To use an existing classifier,
replace `extraction_llm.vllm` with an `endpoint` such as
`http://127.0.0.1:8134/v1`, and set `model` to that server's advertised model ID.
Multiple equivalent endpoints use current in-flight and shared load, capacity,
and round-robin ties. Decision dispatch ignores historical response latency,
so slow startup requests cannot leave healthy replicas permanently idle.
Transport failures still trigger bounded cooldown and a single recovery probe.

The tested runtime is vLLM **0.30.0**; no upgrade to the machine's active workers
was necessary. The optional dependency extra `.[decision-extraction]` declares
that minimum. Install into a dedicated serving environment if needed. Plumb's
pinned revision is `24f7bf77e7ee258a2d158c61ea2dce2b60321010`. Local weights from
the feasibility work are at `artifacts/plumb_feasibility/model`; the safetensors
file there links to the Hugging Face cache. Set `decision_extraction.tokenizer_name`
to that directory and `tokenizer_revision` to `""` to use its local tokenizer.

## Why vLLM classification works here

Plumb has a causal language-model output head, not a separately trained classifier.
vLLM's conversion copies the output-weight rows for tokens A through P into a
16-output readout, applies it to the **last prompt position**, and returns raw
logits. For this checkpoint, the output weights are tied to the input embeddings.
No token is generated. The client takes only the logits for the offered options
and computes `softmax(logits / 2.07)`, matching the original Plumb/JevK5 readout.

Managed servers automatically receive the following conversion arguments. An
external server must be launched with the same readout:

```bash
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
  --model artifacts/plumb_feasibility/model --served-model-name plumb-4b \
  --host 127.0.0.1 --port 8134 \
  --runner pooling --convert classify \
  --hf-overrides '{"classifier_from_token":["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P"],"method":"no_post_processing","num_labels":16}' \
  --pooler-config '{"pooling_type":"LAST","use_activation":false}' \
  --max-model-len 3072 --gpu-memory-utilization 0.28 --enforce-eager \
  --dtype bfloat16 --max-num-seqs 8 --max-num-batched-tokens 4096
```

Requests go to `/classify` with pretokenized input, `use_activation=false`, and
`add_special_tokens=false`. vLLM calls the response array `probs`, even when it
contains raw logits. The client checks its length, finite values, model ID,
prompt-token count, and absence of generated tokens; invalid responses or
transport failures stop extraction rather than becoming clinical missingness.
Transport retries are bounded. Generative repair/fallback is disabled for this
backend. Initial feature preparation uses the union of the configured managed
GPU allocations. Extraction and later ontology revisions use their configured
resident split; the generative backend's completion-triggered model swap does
not apply. External-server configurations retain their supplied endpoints.

The original 24 choice, noul, and score fixtures were run through this conversion:
all top answers agreed with JevK5, and the largest probability difference was
0.003671. Every response reported zero completion tokens. The comparison is
saved in `artifacts/plumb_feasibility/vllm_probe/comparison.json`.

## Feature ontologies and retrieval

Every prompt contains exactly one feature. Binary, categorical, and ordinal
features use their existing `categories_or_unit` ontology. Each decision offers
at most 14 category choices plus separate **not documented** and **none of the above**
options. Larger ontologies first select a group of categories, then select within
that group, recursively if needed. All original values and their declared order
are preserved. Rejection within an already selected group is recorded as
`category_selection_failed` and does not trigger ontology expansion.
Binary features require exactly two declared categories. Category order
is preserved; numeric bins are ascending; exit options come last. These models
are sensitive to option ordering, so the order is recorded and frozen.

Continuous features need one canonical unit and an inclusive initial domain.
For saved numeric features without a listed unit, ontology preparation proposes
a canonical unit from the feature's measurement contract, using `unitless` for
dimensionless counts, ratios, and scores. The original frozen definitions remain
unchanged:

```json
{
  "name": "pretreatment_weight",
  "description": "Pretreatment body weight",
  "value_type": "continuous",
  "categories_or_unit": ["kg"],
  "measurement_definition": "The most recent explicitly documented weight before treatment, converted to kg.",
  "missing_value_rule": "Missing if unavailable or ambiguous.",
  "decision_ontology": {"minimum": 0, "maximum": 256},
  "roles": ["confounder"]
}
```

If numeric bounds are absent, the primary LLM proposes and validates them from
the feature contract **before viewing patient measurements**. Ambiguous discovered
features are similarly converted to a closed ontology; investigator-supplied
features must have a concrete type. Ontologies with unsupported types, empty or
duplicate categories, invalid units, or invalid bounds fail explicitly.

The existing ColBERT retriever ranks chunks from that patient's record for that
feature. Whole chunks are included in rank order and rendered in source order.
The complete Plumb chat prompt, including its template and answer prefix, must
fit `max_prompt_tokens` (default and maximum: 3000). Lower-ranked chunks that
do not fit are omitted and logged; neither chunks nor prompts are silently
truncated. If even one chunk cannot fit with the feature contract, extraction
fails. A missing decision means unavailable in **retrieved evidence**, not proof
that the full record lacks it. Longitudinal conflict rules are included, but
counts and modes are limited to the retrieved observations.

## Numeric narrowing and verification

Defaults are six numeric bins plus two exits, for **eight choices per pass**,
and three passes. Each pass subdivides the chosen interval. Bins are half-open,
except for the inclusive global upper endpoint. “Outside these ranges” remains
available after the first pass, allowing rejection of an earlier wrong branch.
The midpoint of the final bin is the candidate estimate. Nominal interval width
is `(maximum - minimum) / 6**3`; more passes increase nominal resolution but can
also introduce errors.

A fourth, independent prompt asks whether the actual documented value falls
within the numeric interval equivalent to:

```
abs(candidate - documented) <= 0.05 * abs(documented)
```

For a positive candidate, that interval is `[candidate / 1.05, candidate / 0.95]`;
negative endpoints are sorted. The client computes these bounds, so Plumb only
has to make an interval decision. Units, time scope, and conflict rules still
apply. The default requires a true decision with probability at least **0.8**.
A rejected candidate becomes missing; its estimate, interval, and full decision
history remain available. Exact zero requires an exact zero estimate under this
relative-error definition, which finite midpoint bins can fail to produce.

Relevant settings:

| Setting | Default | Meaning |
| --- | ---: | --- |
| `numeric_bins` | 6 | Numeric intervals per pass; allowed 2–6, plus two exits |
| `numeric_passes` | 3 | Refinement passes; allowed 1–8 |
| `verification_relative_tolerance` | 0.05 | Relative tolerance against the documented value |
| `verification_min_probability` | 0.8 | Minimum true probability to release an estimate |
| `none_above_fraction` | 0.2 | Fraction of training patients required to trigger ontology review |
| `none_above_min_patients` | 3 | Minimum distinct training patients required as well |
| `max_prompt_tokens` | 3000 | Complete decision prompt limit |

`stage2.max_ontology_refinement_rounds` bounds review/re-extraction rounds.
The generative extractor's feature-batch size and output-token budgets do not
control decision requests.

## Ontology review and frozen held-out extraction

Review triggers when both the NOTA count and fraction thresholds are reached
for a feature. The denominator is all patients in that outer training fold.
Ordinary missingness and failed numeric verification do not trigger expansion.
The primary LLM receives the feature, aggregate failure counts, and at most
three complete retrieved evidence examples from failed **training** measurements,
bounded by its prompt limit. Row IDs, treatment/outcome columns, and held-out
rows are not supplied. Evidence excerpts may naturally mention treatments;
they are used only to revise measurement categories or numeric bounds.

The LLM may keep the ontology, replace categorical levels, or revise numeric
bounds. It cannot change feature identity, units, time scope, or missing/conflict
rules through this review. Supplied explicit-feature ontologies retain the
pipeline's existing immutability rule; unresolved NOTA is reported for manual
review. Only changed features are re-extracted. Prepared/revised definitions
are frozen before held-out measurement. The unrelated estimand-driven ontology
search and runtime model-continuation override are currently incompatible with
this backend and rejected by configuration validation.

Checkpoints fingerprint the source text, feature contract (including numeric
bounds), retrieval/model identity, and decision settings. They record full
prompts, ordered choices, raw logits, probabilities, source offsets, omitted
chunks, candidate intervals, and verification results. Fold failure summaries
separate NOTA, ordinary missingness, and verification failure.

## Integration evaluation, 2026-10-04

`scripts/plumb_stage2_test.py` runs the actual ColBERT → vLLM path on the existing
76-case synthetic continuous fixture, keeping gold values outside prompts.
Use `--output` for a fresh result directory when changing settings.

```bash
HF_HUB_OFFLINE=1 artifacts/plumb_feasibility/venv/bin/python \
  scripts/plumb_stage2_test.py \
  --output artifacts/plumb_feasibility/stage2_decision_strict_results
```

With the final 0.8 verification threshold, the fresh run accepted **50/68 numeric
cases**, and all 50 were within 5% of gold. Six missing/ambiguous and two
out-of-domain cases returned missing. Across all 76 cases there were 50 accepted,
9 NOTA, 11 verification rejections, and 6 not-documented decisions. Five rejected
numeric candidates were actually within tolerance. Maximum prompt length was
699 tokens; elapsed time was about 39 seconds with cached weights, CPU retrieval,
and four request workers. These are short fixtures, not a long-record benchmark.

| Conversion | Gold | Candidate | Final result |
| --- | ---: | ---: | --- |
| 27 mm → cm | 2.7 | 15.519 | Rejected |
| 72,500 g → kg | 72.5 | 70.519 | Accepted, 2.73% error |
| 180 lb → kg | 81.647 | 170.074 | Rejected |

An exploratory 0.5-threshold run accepted 59 estimates, including **four outside
5%**. The stronger threshold was chosen after inspecting those results, so the
zero false acceptances in the final run are **not independent validation or a
guarantee**. Verification uses the same model and can share its mistakes. Unit
conversion, small signed values, and zero remain weak points. Concurrent bf16
inference also produced small score differences between runs. Evaluate on
study-specific labeled records before relying on this as a production substitute.

Raw outputs and summaries are under
`artifacts/plumb_feasibility/stage2_decision_results` (0.5) and
`artifacts/plumb_feasibility/stage2_decision_strict_results` (0.8).

## Serving concurrency probe, 2026-10-04

`scripts/plumb_throughput_benchmark.py` replays saved decision prompts against
one managed classifier, then stops that server. The probe used GPU 1 of this
machine (an A6000 shared with its existing worker), vLLM 0.30.0, bf16 eager
serving, and 28% GPU-memory allocation. The 44 distinct prompts came from the
two-patient five-confounder/five-modifier evaluation and contained 2016–2457
tokens. Each trial replayed them four times (176 requests), after warmup.
Prefix caching was disabled to prevent replayed inputs from inflating throughput.

| Server sequence limit | Token batch budget | Requests in flight | Decisions/second | Median request latency |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 4096 | 8 | 4.70 | 1.56 s |
| 8 | 4096 | 32 | 4.61 | 7.04 s |
| 64 | 16384 | 8 | 4.73 | 1.68 s |
| 64 | 16384 | 32 | 4.72 | 6.15 s |
| 64 | 16384 | 64 | 4.71 | 13.84 s |
| 64 | 16384 | 128 | 4.71 | 20.20 s |

A repeat of concurrency 32 on the larger server preset measured 4.71 decisions/s.
All 1232 timed requests completed successfully. Top decisions agreed with the
original audits on 99.4–100% of requests, depending on trial; this is a numerical
consistency check, not an accuracy evaluation. Requests in flight include queued
requests; the sequence and token limits bound actual GPU batches.

Higher concurrency was feasible, but neither a deeper queue nor the larger
token budget improved throughput materially on this workload. These are
**serving-only** measurements: retrieval, prompt packing/tokenization, and
checkpoint writes were excluded. They do not establish an end-to-end optimum
or the optimum on H100/RTX PRO 6000. The eight-GPU launchers allow 128 requests
across seven Plumb replicas on H100 or eight on RTX PRO 6000, with eight sequences
and a default 4096-token batch per server.
That is a global request limit, including queued work, not 128 GPU sequences
per replica or a claim of measured throughput improvement.

To repeat the larger-batch probe using local weights and the saved audits:

```bash
HF_HUB_OFFLINE=1 artifacts/plumb_feasibility/venv/bin/python \
  scripts/plumb_throughput_benchmark.py \
  --gpu 1 --max-num-seqs 64 --max-num-batched-tokens 16384 \
  --concurrency 8,32,64,128,32 --repeats 4 \
  --output artifacts/plumb_feasibility/throughput_new_run
```

The GPU index follows `CUDA_VISIBLE_DEVICES`. Choose an available GPU and port;
the output directory must be new. Use `--audits` and `--model` to supply an
existing extraction's decision directory and its matching local model/tokenizer.
Raw results, server commands/logs, and prompt hashes are saved under
`artifacts/plumb_feasibility/throughput_baseline` and
`artifacts/plumb_feasibility/throughput_16k` for the measurements above.

## Preparation optimization check, 2026-10-05

A paired check used 30 saved decision audits from the five outer folds on the
RTX PRO 6000 run. Cached document scoring returned identical retrieval results,
and batched prompt packing sent identical token IDs, including the original
first-over-budget stopping behavior. With warm query vectors and patient indexes,
median retrieval time fell from 14.4 ms to 0.69 ms. Median prompt-packing time fell
from 26.7 ms to 7.53 ms. These measurements exclude classifier serving and checkpoint
writes. Results and prompt hashes are recorded in the run's restart metadata under
`plumb_preparation_throughput_20261005T015228Z/preparation_benchmark.json`.

The saved run resumed with an 8192-token Plumb batch, retaining eight sequences,
the 3072-token context, 28% Plumb memory allocation, and the Gemma server on GPU 0
at 50%. Completed extraction and upstream checkpoints were preserved.
An initial 60-second live sample measured 59.3 classifier requests/s across the
eight replicas, versus 37.6 requests/s in the prior sample, a 58% increase.
All five folds were running without recoveries. This is an early end-to-end
observation as the feature mix progresses, rather than a controlled serving-only
comparison. The samples and summary are recorded in the same restart metadata.
