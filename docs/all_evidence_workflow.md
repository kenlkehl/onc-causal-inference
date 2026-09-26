# All-evidence workflow

The researcher workflow is one command:

```bash
uv run python scripts/run_all_evidence.py \
  --config example_configs/research_all_evidence.json
```

It reads one cohort and writes everything to one output directory. With no
Stage 2 transport configured it runs Stage 1 through the plain handoff. With an
external endpoint or a pipeline-managed vLLM pool it continues through
fold-scoped feature definitions, two-endpoint patient extraction and ontology
supervision, fold-local statistical selection, and cross-fitted causal-forest
estimation. Interruption and resume
are automatic: run the same command again.

This is the repository's only all-evidence orchestration path. It uses ordinary
files and completion markers rather than a separate source snapshot, immutable
request, checkpoint-adoption system, trust policy, artifact-authentication
layer, or content-hash protocol.

The former `MultiModelForestRunner` and its TF-IDF-topic agentic Stage 2 were
retired. `multi_model_forest` remains the name of the Stage 1 scientific
configuration, not a second runnable workflow. The `oci run` command is
reserved for the retained explicit-feature workflows; configurations using
`multi_model_forest` stop with a migration message and must use this workflow.

## Configuration

Copy
[`example_configs/research_all_evidence.json`](../example_configs/research_all_evidence.json)
and edit it. It keeps the settings a researcher normally changes in one place:

```json
{
  "dataset": "/data/cohort.parquet",
  "output_dir": "/results/my_stage1_run",
  "columns": {
    "unit_id": "patient_id",
    "text": "clinical_text",
    "treatment": "treatment_indicator",
    "outcome": "outcome_indicator"
  },
  "science": {
    "clinical_question": "Which pretreatment text features confound or modify treatment effect?",
    "outcome_type": "binary",
    "outer_folds": 5,
    "inner_folds": 5,
    "seed": 42,
    "stage1": {},
    "neural_queries": {}
  },
  "models": {
    "htr": "/models/bert-tiny",
    "embeddings": "/models/qwen3-embedding-8b"
  },
  "stage2": {
    "endpoint": "",
    "model": "",
    "extraction_llm": {
      "endpoint": "",
      "model": ""
    }
  },
  "run": {
    "devices": ["cuda:0", "cuda:1"],
    "workers": 16,
    "components": [
      "embedding_cache",
      "tfidf",
      "text_models",
      "neural_queries",
      "handoff"
    ]
  }
}
```

Paths in the file are relative to the config file. `science.stage1` accepts a
nested override of any setting in the established all-evidence Stage 1 model
template. `science.neural_queries` does the same for neural-query settings.
The fully expanded settings actually used are written to the output directory
as `resolved_stage1_model_config.json` and
`resolved_neural_query_config.json`. Secret-valued API-key fields are redacted
in these readable copies but remain available to the in-memory model config.

JSON and YAML config files are accepted. YAML requires PyYAML in the current
environment.

### Architecture selection

By default, Stage 1 retains the historical behavior: it runs every architecture
enabled by the resolved model configuration. To make an architecture subset a
first-class scientific choice, set `science.stage1_architectures` to a JSON list
or pass a comma-separated CLI value:

```bash
uv run python scripts/run_all_evidence.py \
  --config my_run.json \
  --architectures embedding_whole_cohort,tfidf_semantic_retrieval_contrasts
```

The registry resolves private prerequisites such as the embedding cache and
whole-cohort contrast computation, but only selected architecture envelopes are
written to a targeted handoff and admitted to Stage 2. A saved explicit
selection cannot be changed while resuming. Omit the setting to resume an older
run unchanged, or choose a fresh output directory for a different subset.

## Full, Stage-1-only, and Stage-2-only runs

Stage 2 reads `handoff/evidence.jsonl` and completes the remainder of the
analysis. It first uses the `all_evidence_fusion` scientific projections to
compile exact-deduplicated, fold-local semantic evidence cards. It reuses the
Stage 1 embedding cache by memory map where compatible and retains card/member/
raw-path lineage for audit. Within each outer fold, the primary LLM sees every
compiled card and lists every mentioned or implied pretreatment patient-level
clinical feature. One card may support many candidates. There is no ColBERT
interaction filter, evidence-community graph, retrieval stage, causal-role
filter, or feature-count cap.

The primary model then performs merge-only alias consolidation and defines one
extraction ontology per consolidated candidate. A user-supplied small LLM
extracts all candidate values from outer-training clinical text, one patient
per prompt. It may use a different endpoint or another model on the same
endpoint as the primary model. The primary model receives only aggregate extraction
values and validation failures to review and ontologize the small model's
output; it receives no patient text, treatment or outcome values, causal-role
evidence, performance metrics, or p-values. It may revise only the same
candidate's extraction schema, and a revision triggers small-model re-extraction.
Only candidates whose prompt-facing schema changed are re-extracted; unchanged
raw columns are reused and merged with the refreshed columns. Extraction remains
one patient per prompt, with the existing per-patient feature batching.

Once ontologies are frozen, an optional second pass consolidates equivalent
extracted measurements. Inner-fold grouped elastic nets, candidate-wise tests,
and R-loss models then produce selection evidence. The default `llm_roles`
mode uses primary-model adjudication; the opt-in `independent_tasks` mode uses
separate numerical treatment, outcome, and effect selections with optional
advisory annotations. The opt-in `multi_model` mode combines several model
families, clinical theme review, and nested selection of modifier count and
final estimator. P/q values contribute evidence.
Explicit investigator variables retain their configured ontologies and roles
regardless of selection evidence. Only the retained variables are
extracted from outer-held-out text. The final heterogeneous-effect model is an
honest causal forest by default. Multi-model architecture search can select a
penalized outcome model with treatment interactions. Outer-held-out nuisance
predictions supply cross-fitted AIPW scores. Stage 2 is enabled by specifying `stage2.endpoint` or
`stage2.vllm`; dataset-backed execution additionally requires a separate
`stage2.extraction_llm` model configuration.

The only supported compiler is `semantic_cluster_cards_v2`. It checks each
outer fold for every architecture in the frozen Stage 1 selection (or the
resolved legacy enable flags) before making an interpretation request. Missing evidence fails
with a readable Stage 1 rerun instruction. This is an in-process set comparison,
not an artifact-authentication, byte-attestation, or deployment-gate system.
The former `raw_packets_v1` compatibility option is intentionally unsupported
because it combined scientifically distinct architectures.

For initial feature discovery, Python sends one compiled card per request,
containing only its readable representative texts. Card and packet IDs,
evidence kind, detail objects, truncation flags, axes, polarity, semantic
grouping, architectures, scores, support counts, folds, and other provenance
stay outside the model prompt. The self-contained prose instructions explain
the purpose, what the excerpts represent, and which choices belong downstream.
The model returns ordinary clinical names, descriptions, textual bases, and
material uncertainties. Python validates that response, normalizes names,
preserves distinct candidates when normalized names collide, and attaches the
card's provenance. There are no model-authored IDs or item indices. Discovery does not
choose causal roles, value types, units, categories, or extraction ontologies.
All returned candidates continue to lossless, iterative consolidation; the
later one-feature ontology request sees the canonical name and its directly
associated readable supporting text. Cards with no candidates still enter the
recall audit, which uses the same clinical response contract on one card at a
time. All 23 reviewed prompt types are now standard: discovery, aliases,
measurement definitions, patient/chunk/occurrence extraction, category mapping,
ontology review, harmonization, role/theme review, modifier ordering, concept
review, and repairs. Their source is `stage2_prompt_catalog.py`; repair wording
is assembled with the actual validation error. Python assigns identifiers,
source offsets, provenance, ranking positions, and merge bookkeeping. Clinical
names remain in responses so measurements and judgments can be matched.
The prompt version and exact request content participate in checkpoint identity. Use a fresh Stage 2 output directory to regenerate definitions
under the new discovery contract; existing completed definitions are preserved.

An external endpoint configuration is:

```json
{
  "stage2": {
    "endpoint": "http://127.0.0.1:8010/v1",
    "model": "Qwen/Qwen3.8-27B",
    "workers": 32,
    "extraction_llm": {
      "endpoint": "http://127.0.0.1:8020/v1",
      "model": "small-extractor",
      "api_key": "EMPTY",
      "workers": 32
    },
    "request_timeout": 7200,
    "request_attempt_timeout": 900,
    "transport_max_attempts": 6,
    "max_tokens": 100000,
    "extraction_max_tokens": 4096,
    "extraction_reasoning_max_tokens": 32768,
    "extraction_stream": true,
    "extraction_deferred_retry_passes": 1,
    "max_response_repairs": 15,
    "thinking_after_response_repairs": 5,
    "repetition_penalty": null,
    "interpretation_reasoning_effort": "auto",
    "extraction_reasoning_effort": "auto",
    "evidence_compiler": "semantic_cluster_cards_v2",
    "evidence_max_cards_per_fold": 400,
    "evidence_max_exemplars_per_card": 4,
    "evidence_max_exemplar_chars": 2400,
    "operationalization_max_prompt_chars": 640000,
    "consolidation_batch_size": 20,
    "consolidation_alphabetical_rounds": 5,
    "consolidation_max_rounds": 55,
    "consolidation_policy": {
      "strategy": "mixed",
      "semantic_fraction": 0.6,
      "random_fraction": 0.0,
      "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
      "embedding_device": "cpu",
      "early_stop_min_rounds": 3,
      "early_stop_patience": 2,
      "early_stop_min_reduction": 0.005
    },
    "extraction_feature_batch_size": 10,
    "extraction_chunk_size_tokens": 50000,
    "extraction_context_window_tokens": 131072,
    "extraction_context_margin_tokens": 1024,
    "max_review_rounds": 2,
    "ontology_refinement_min_failure_patients": 3,
    "max_ontology_refinement_rounds": 2,
    "input_temporal_scope": "pre_index_treatment",
    "selection_consolidation": {
      "enabled": true,
      "neighbor_count": 10,
      "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
      "embedding_device": "cpu",
      "max_latents_per_cluster": 2,
      "minimum_pairwise_association": 0.85
    },
    "statistical_selection": {
      "l1_ratio": 0.8,
      "nuisance_selection_rule": "any_inner_fold_union",
      "modifier_selection_rule": "any_inner_fold_union",
      "internal_cv_folds": 3,
      "regularization_grid_size": 16,
      "one_standard_error_rule": true,
      "nuisance_prediction_one_standard_error_rule": false,
      "modifier_one_standard_error_rule": false,
      "univariable_confounder_p_value_threshold": 0.05,
      "univariable_confounder_q_value_threshold": 0.10,
      "modifier_top_n_per_inner_fold": 10,
      "modifier_ridge_alpha": 10.0,
      "modifier_continuous_winsor_quantile": 0.005
    },
    "role_adjudication": {
      "enabled": true,
      "max_candidates_per_request": 20
    },
    "estimation_trees": 200,
    "explicit_features": [
      {
        "name": "age_at_treatment_decision",
        "description": "Age at the pretreatment decision point.",
        "value_type": "continuous",
        "categories_or_unit": ["years"],
        "measurement_definition": "Extract age in years at the treatment-decision point.",
        "missing_value_rule": "Return null when age cannot be determined from the pretreatment record.",
        "roles": ["confounder"]
      }
    ]
  }
}
```

The example above and `example_configs/research_all_evidence.json` explicitly
enable `stage2.selection_consolidation.enabled`. The **omitted-setting default
is false**, including fresh runs through the standard synthetic shell launchers.
Saved-run launches preserve the supplied policy. This is the second,
post-extraction pass; it does not control discovery-time merge-only
consolidation. Set it explicitly in the run configuration, or use
`--set stage2.selection_consolidation.enabled=true` on the Python entry point
for a new run. Check `stage2/config.json` for a saved run's resolved policy;
do not infer it from the example or from the presence of a consolidation report.

The exhaustive interpretation output is written to
`outer_NNN/interpreted_candidates.json`. Every candidate then enters
`outer_NNN/consolidation/`; the merge checkpoints and
`feature_definitions.json` retain exact provenance and origin dispositions.
There is no intermediate selected-candidate registry.

`stage2.explicit_features` is optional. Each entry is a complete, investigator-
specified variable definition, not merely a requested name. It requires
`name`, `description`, `value_type`, `categories_or_unit`,
`measurement_definition`, `missing_value_rule`, and one or more `roles` from
`confounder` and `effect_modifier`; `"both"` expands to both roles. Closed ontologies are
validated just like model-authored definitions: binary variables need exactly
two values, and categorical or ordinal variables need at least two. A
continuous feature may use an empty list when it has no meaningful unit.
`stability_summary` and `caveats` are optional. For compatibility with the
standalone explicit-feature vocabulary, `type`, `categories`, and `unit` are
accepted aliases; the ontology fields may also be placed inside an `ontology`
object while `name` and `roles` remain alongside it.

Configured features enter the iterative alias-consolidation batches in every
outer fold. In every round they are hard invariants: they cannot be excluded,
two distinct configured feature names cannot be merged, and a merge containing
one must use its exact configured name as the output. When Stage 1 discovers an
alias, Python retains one consolidated feature, attaches the discovered packet
and architecture provenance, keeps the configured name and roles, and uses the
supplied ontology without making the one-feature ontology request. They still
undergo training-fold extraction, but ontology supervision and statistical
selection must keep them without revising the supplied ontology or roles. If a required feature cannot
be extracted well enough for the workflow's health checks, the run fails
visibly rather than silently dropping or redefining it.

`stage2.model` is optional. If it is empty or omitted, Stage 2 queries the
OpenAI-compatible `/models` endpoint once at startup and uses the result when
exactly one model ID is advertised. If the server advertises multiple model
IDs, set `stage2.model` explicitly to avoid an ambiguous selection.
The same rule applies independently to an external
`stage2.extraction_llm.model`. Pipeline-managed pools require an explicit model
because the pipeline must launch it. The extraction model configuration must be
supplied for dataset-backed Stage 2, but its external endpoint may be the same
as the primary endpoint; its API key may be set in the nested config or
`OCI_STAGE2_EXTRACTION_API_KEY`.

### Pipeline-managed vLLM replicas

`stage2.vllm` tells the pipeline to own the orchestrator vLLM lifecycle, while
`stage2.extraction_llm.vllm` independently does the same for the extractor. A
managed role requires its model and must omit its corresponding endpoint. The
workflow environment must include the `local-llm` optional dependency group.
For example:

```json
{
  "stage2": {
    "model": "Qwen/Qwen3.8-27B",
    "workers": 32,
    "vllm_rapid_switch_seconds": 900,
    "extraction_llm": {
      "model": "LiquidAI/LFM2.5-2.6B",
      "workers": 64,
      "vllm": {
        "gpus": ["cuda:2", "cuda:3"],
        "gpus_per_server": 1,
        "base_port": 8110,
        "download_dir": "/models/huggingface",
        "extra_args": ["--gpu-memory-utilization", "0.80"]
      }
    },
    "vllm": {
      "gpus": ["cuda:0", "cuda:1"],
      "gpus_per_server": 2,
      "base_port": 8010,
      "download_dir": "/models/huggingface",
      "startup_timeout": 7200,
      "extra_args": [
        "--gpu-memory-utilization", "0.90",
        "--max-model-len", "65536"
      ]
    }
  }
}
```

The pipeline interprets both GPU lists as logical CUDA indices. When its parent
environment has `CUDA_VISIBLE_DEVICES=4,6`, for example, managed logical GPUs 0
and 1 are mapped to physical selections 4 and 6 for the child processes. The
list for each dedicated role is divided into equal groups in supplied order and
the tensor-parallel size is `gpus_per_server`. The replica count is derived
from the list length; an explicit `server_count` is optional but must agree.
Uneven division, duplicate devices within one pool, or more servers than GPUs
is rejected before launch.

When both roles are pipeline-managed, their lifecycle initially alternates.
Interpretation and feature definition first run to completion for every outer
fold with only the orchestrator model loaded across the ordered union of both
roles' GPU lists. Stage 2 then checkpoints and unloads it before loading the
extractor across the same union. At an ontology-supervision barrier it
checkpoints extraction, unloads the extractor, and reloads the orchestrator; it
switches back whenever re-extraction or held-out extraction is required. For
each role, if the union divides evenly by its configured tensor-parallel width,
Stage 2 preserves that width and adds replicas. Otherwise it launches one
tensor-parallel server across the complete union, which requires that model to
support the resulting tensor-parallel size. Added replicas use consecutive
ports after the highest configured port for that role.

Stage 2 tracks the monotonic elapsed time between those transitions. When two
consecutive switches are less than `stage2.vllm_rapid_switch_seconds` apart
(900 seconds by default), it treats the workload as rapidly alternating and
keeps both models resident from then on. The orchestrator and extractor use
their exact separately configured GPU lists, tensor-parallel widths, replica
counts, and ports in this concurrent fallback. A value of `0` disables the
fallback. The phase file records the last switch time, elapsed interval,
cutoff, allocation mode, and configured GPU allocations. A
feature-definition-only run never starts the extractor. Resumes use the
persisted managed-model phase together with ordinary request checkpoints; they
retain a previously selected concurrent split, while an older run with
extraction artifacts safely resumes with the extractor first.

The runner checks every requested port before launch, starts all replicas with
separate `CUDA_VISIBLE_DEVICES` values, and waits for each `/v1/models` API to
advertise a model. `stage2.workers` remains the total primary-model request concurrency; a
thread-safe round-robin router spreads those requests over the ready endpoints.
A retry advances to the next endpoint, so one transiently unhealthy replica
does not receive every retry of the same request. Extraction requests use an
independent round-robin router and `stage2.extraction_llm.workers` ceiling.
Server logs and redacted manifests containing commands, PIDs, endpoints, GPU
assignments, and exit codes are stored in
`stage2/vllm_servers/orchestrator_all_gpus/`,
`stage2/vllm_servers/extractor_all_gpus/`,
`stage2/vllm_servers/orchestrator/`, and
`stage2/vllm_servers/extractor/`, as applicable. All managed process groups are
terminated when Stage 2 succeeds, fails, or is interrupted. Either role can
independently remain endpoint-backed instead; all-GPU staging applies only when
both roles are managed by this pipeline.

The managed vLLM fields are:

- `server_count`: number of server replicas. If omitted, it defaults to one per
  supplied GPU unless `gpus_per_server` is supplied.
- `gpus`: required list or comma-separated string of logical CUDA indices.
- `gpus_per_server`: tensor-parallel width. When supplied, the replica count is
  derived as `len(gpus) / gpus_per_server`; an explicit `server_count` must agree.
- `host`, `base_port`, or `ports`: bind address and either consecutive or exact
  ports. Defaults are `127.0.0.1`, orchestrator ports beginning at 8010, and
  extractor ports beginning at 8110.
- `internal_port_base`: start of disjoint vLLM engine rendezvous ranges. It
  defaults to 20000 for the orchestrator and 40000 for the extractor.
- `startup_timeout`, `startup_poll_interval`, and `shutdown_timeout`: lifecycle
  timing controls in seconds.
- `download_dir`, `reasoning_parser`, `language_model_only`, and
  `default_chat_template_kwargs`: named vLLM serve options.
- `extra_args`: a list of remaining raw vLLM CLI tokens, such as
  `["--gpu-memory-utilization", "0.90"]`. Orchestration-owned and named options
  cannot be duplicated here.

The cross-pool `stage2.vllm_rapid_switch_seconds` setting is top-level because
it controls the lifecycle relationship between the two managed pools rather
than either individual vLLM server.

Unless explicitly overridden, any model name containing `gemma` uses
`reasoning_parser: "gemma4"` and `language_model_only: true`; it does not set a
server-wide chat-template thinking default. Any model name containing `qwen`
uses `reasoning_parser: "qwen3"` and `language_model_only: true`.

Stage 2 selects reasoning per Chat Completions request. Evidence interpretation
and audit, consolidation, operationalization, category mapping, aggregate
ontology supervision, and ontology refinement go to the primary model with
`reasoning_effort: "auto"` by default. The extraction model also uses `auto`.
After `/models` identifies the backing model, Qwen3.8 Flash Next resolves to
`xhigh` for both roles; other models retain high interpretation and initially
disabled extraction reasoning. Explicit settings override these defaults. Primary-model
requests receive the configured `max_tokens` ceiling (100,000 by default), while
patient extraction receives `extraction_max_tokens` (75,000 when omitted;
the recommended example uses 4,096 for ten-feature non-thinking requests).
`extraction_reasoning_max_tokens` supplies a separate total allowance for reasoning
plus final JSON when thinking is enabled, including repair escalation: 32,768 in
the example, or the shared `extraction_max_tokens` ceiling when omitted/null. These
permit long responses but do not request minimum lengths; each model still stops
at EOS as soon as its JSON is complete. A repair request dynamically lowers the
ceiling when necessary to keep its prompt, output allowance, and safety margin
inside the extraction model's context window.
Stage 2 uses [publisher sampling profiles](stage2_sampling.md), with explicit
configuration overrides taking precedence.
Stage 2 first verifies each live endpoint's selected model through `/models`.
It recognizes Qwen 3 (including 3.8), Gemma 4, and LFM 2.5 model IDs and sends
their chat-template thinking switch. Flash Next also receives `xhigh` and
`preserve_thinking`; rejected reasoning or sampling controls cause a visible
failure. Other families retain the portable prompt fallback and progressively
more standard request fields when an endpoint rejects an extension. Responses are accepted whether
reasoning is separated into `reasoning_content` or remains inline in Qwen/LFM
`<think>` blocks or Gemma thought channels. The configured fields are
`interpretation_reasoning_effort` and `extraction_reasoning_effort`.
Qwen 3.8 translates an explicit `high` policy to its accepted wire value
`xhigh`. Explicit thinking-off requests use the template switch and omit the
enabled-only effort enum. Effective policies and publisher sources appear in
`model_identity.json`; request events record the controls sent on each attempt.

A complete logical request is bounded by `request_timeout` (7200 seconds by
default), including transport retries and response-repair turns. Individual
HTTP calls use `request_attempt_timeout` (900 seconds by default) for transport
timeouts. With `extraction_stream: true`, this bounds waiting for the next network
read, so active streaming can continue longer than 900 seconds, up to the shared
logical deadline. Streaming is opt-in (false when omitted) for endpoint
compatibility; the recommended example enables it. Retryable transport failures
receive at most `transport_max_attempts` (6 by default) per response attempt,
including compatibility negotiation. A completed response that fails JSON parsing or schema validation receives up
to `max_response_repairs` validator-guided retries (15 by default). Every retry
includes the concrete validation error. Repairs through
`thinking_after_response_repairs` (5 by default) retain the normal request
policy; repairs after that threshold force `reasoning_effort` to at least
`high`, enabling thinking for the managed vLLM reasoning parsers. This includes
structural extraction errors such as a missing `rows` array, row object, or
`values` object. Reasoning-enabled extraction repairs use
`extraction_reasoning_max_tokens` when configured, within the available context
and the same logical request deadline.

For a single patient, an exact, complete feature-value map, a single `values`
object, or a `rows` object instead of an array can be wrapped into the canonical
response. No values are inferred; supplied row IDs and all values still pass the
normal validation. Partial or ambiguous wrappers are rejected.

Extraction checkpoints append `request_events.jsonl` records with a logical
request ID, patient/feature identifiers, retry counters, reasoning mode, output
cap, provider request ID, timing, finish reason, and reported token usage.
Streaming records the first progress and then progress every 30 seconds when
chunks arrive, including content/reasoning character counts without their text.
Missing usage is recorded as unavailable. An incomplete stream or a token-limit
finish cannot become a successful measurement checkpoint.

`extraction_deferred_retry_passes` defaults to 1. A patient or page task whose
logical request is exhausted is recorded in `deferred_extraction.json`; other
independent tasks continue before a bounded retry pass reuses its completed
feature/chunk checkpoints. Remaining failures abort extraction and prevent
fitting. Consecutive exhausted tasks (at least three, or the configured worker
count if larger) abort early during a persistent outage. Set the option to 0 for
the former immediate-abort behavior. Configuration/programming errors still
abort immediately.

Python first coalesces only exact normalized-name duplicates; this is identity
bookkeeping and makes no semantic decision between distinct names. It then
uses a reproducible mixture of nonoverlapping batches of
`consolidation_batch_size` candidates (20 by default). Every candidate appears
once per round. Approximately 60% of batches use semantic neighbors and 40% use
alphabetical neighborhoods. Semantic
retrieval embeds candidate names and descriptions with Qwen3-Embedding-0.6B;
the LLM reviews the clinical descriptions for exact duplicate variables whose
names differ only in spelling, synonymous wording, or abbreviation. Each merge
retains an existing member name. Creatinine-clearance aliases can merge;
creatinine clearance and estimated GFR remain separate. Different scales,
methods, timing, and levels of specificity remain separate. Uncertain matches
pass through unchanged.
Each semantic pivot retrieves its closest still-unassigned candidates. Pivots
and lexical boundaries rotate each round. Batches run concurrently without
overlapping merge directives. Saved embeddings are reused across rounds and
restarts; new/changed candidate descriptions are embedded when needed.

After at least three mixed rounds, two consecutive successful rounds with
less than 0.5% candidate-count reduction stop consolidation. The reduction is
`(input_groups - output_groups) / input_groups`, including exact-name coalescing.
Validation-fallback rounds break the low-yield streak. A single batch containing
the whole pool can stop immediately after a successful no-merge review. The hard
cap is `consolidation_max_rounds` (55). Controls live in
`stage2.consolidation_policy`: `semantic_fraction` (the remaining fraction is
alphabetical), `embedding_model`, `embedding_device`,
`early_stop_min_rounds`, `early_stop_patience`, and `early_stop_min_reduction`.
The compatibility field `random_fraction` must be zero. The default strategy is
`mixed`; `strategy: "legacy"` reproduces the historical
grouping of five alphabetical rounds followed by seeded shuffles, governed by
`consolidation_alphabetical_rounds`. Mixed results use
`consolidation/candidate_pool_consolidation_mixed/`; legacy artifacts are retained.
The process also stops when the pool is empty or only configured features remain. Identical
canonical output names produced by independent batches are coalesced exactly
while retaining all provenance and candidate descriptions.

The clinical question is deliberately absent from every consolidation batch.
The model returns `merge_directives`, each with an `inputs` list of exact names
from that batch in `members` and one `canonical_label`. Python translates the
clinical names to its internal merge directives. Iterative consolidation is
strictly merge-only: every supplied feature survives each round either
unchanged or as a member of one merged alias family. The response contract has
no exclusion list, and a response that supplies `exclude_feature_names` is
invalid and falls back losslessly if repairs do not remove it. Each batch sees
no candidate or group IDs and does not restate unchanged features. Python
validates that every supplied name exists, maps names back to internal groups,
unions merged provenance, and passes every unmentioned name through unchanged.
Original candidate descriptions are carried through every round so later
prompts do not lose semantic evidence.
The response normalizer recognizes only unique names and descriptions actually
supplied in that batch; this permits a copied description to resolve back to its
exact feature without fuzzy or domain-specific matching. It restores a reused
canonical output omitted from its own input family and ignores a degenerate
one-feature no-op. If a batch is still structurally invalid after bounded
repairs, Python writes `fallback.json`, passes every member of that batch
through unchanged, and records the fallback in the round and root completion
summaries. This lossless fallback also preserves every configured feature.
There is no fuzzy blocker, neighbor selection, or pairwise LLM request during
discovery-time alias consolidation. Python
leaves every discovered causal role empty until fold-local all-evidence adjudication;
investigator-configured features alone carry supplied roles at this point.

Every group remaining after consolidation is operationalized for extraction.
Each ontology request contains only the
canonical feature name, a deduplicated flat list of readable supporting text,
the ontology instructions, and the response contract. Python extracts only
`representative_evidence.text`
from the compiled packets directly cited during exhaustive discovery. Packet boundaries, evidence
kind, truncation flags, details, the outer-fold number, clinical question,
candidate or group IDs, discovery value type, evidence axes, semantic grouping,
architecture names, scores, support counts, and candidate summaries are not
sent. The model chooses binary, categorical, continuous, ordinal, or ambiguous
at this point from the feature name and readable evidence, then supplies allowed
values or a unit and the extraction and missingness rules. Python validates the
closed-ontology shape but does not hard-code domain-specific features,
categories, or units. There is no LLM ranking, diversity selection, or
feature-count cap between alias grouping and operationalization. A group that
contains a configured explicit feature skips this request and uses its supplied
ontology instead.

Operationalization requests have an independent
`operationalization_max_prompt_chars` allowance (640,000 characters by
default). Python reserves repair headroom and deterministically packs whole
supporting excerpts under the remaining initial-prompt budget. The checkpoint
records counts and character totals for available, included, omitted, and
truncated evidence together with a fingerprint of the full available evidence
list, while the final feature retains complete packet provenance. If the model
still returns an invalid ontology after all bounded repairs, Stage 2 writes
`fallback.json` and uses a conservative `ambiguous` ontology for training-fold
extraction and aggregate supervision rather than aborting the fold.

The API key may be set as `stage2.api_key` or in `OCI_STAGE2_API_KEY`. Other
operational controls include `request_timeout` (7200 seconds by default),
`request_attempt_timeout` (900 seconds by default),
`transport_max_attempts` (6 by default),
`transport_retry_backoff`, `max_response_repairs`,
`thinking_after_response_repairs`, `max_tokens`, `extraction_max_tokens`,
`extraction_reasoning_max_tokens`,
`extraction_stream`, `extraction_deferred_retry_passes`,
`max_prompt_chars`,
`consolidation_max_prompt_chars`,
`operationalization_max_prompt_chars`,
`consolidation_batch_size`, `consolidation_alphabetical_rounds`,
`consolidation_max_rounds`, `consolidation_policy`,
`extraction_max_prompt_chars`, `extraction_feature_batch_size`,
`extraction_chunk_size_tokens`, `extraction_context_window_tokens`,
`extraction_context_margin_tokens`,
`vllm_rapid_switch_seconds`,
`evidence_compiler`, `evidence_max_cards_per_fold`,
`evidence_max_exemplars_per_card`, `evidence_max_exemplar_chars`,
`max_review_rounds`,
`ontology_refinement_min_failure_patients`,
`max_ontology_refinement_rounds`, `input_temporal_scope`,
`statistical_selection`, `estimation_trees`,
`propensity_clip`, `min_nonmissing_fraction`, `max_dominant_fraction`,
`temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`,
`frequency_penalty`, `repetition_penalty`, `interpretation_reasoning_effort`,
and `extraction_reasoning_effort`. The nested `extraction_llm` object controls its
endpoint or managed `vllm` pool, model, API key, and workers. A configured primary endpoint or managed
vLLM pool makes the default mode `full`. The modes can always be made explicit:

Each candidate-consolidation batch uses the independent
`consolidation_max_prompt_chars` limit (640,000 characters by default), each
one-feature ontology request uses `operationalization_max_prompt_chars`
(640,000 by default), and patient-variable extraction uses
`extraction_max_prompt_chars` (also 640,000 by default). Each extraction prompt
contains one patient and at most `extraction_feature_batch_size` frozen feature
definitions (10 by default); Stage 2 checkpoints and merges the feature batches.
`max_prompt_chars`
continues to bound interpretation and ontology-supervision planning. These character limits
are safety/planning guards, not claims about the model's token context. Primary
completions send `max_tokens` as a 100,000-token output ceiling; patient
extraction sends `extraction_max_tokens` for non-thinking requests (4,096 in the
recommended example; 75,000 when omitted) and `extraction_reasoning_max_tokens`
when thinking is enabled (32,768 in the example). Neither is a
generation target or minimum. A response that reaches its ceiling enters Stage
2's bounded repair or fallback path. The extraction-only ceiling is transport
policy and does not invalidate completed feature-definition checkpoints when it
changes, so an interrupted extraction can resume under a safer ceiling (down to
4,096 tokens for non-thinking extraction). Completed feature/patient measurements
remain reusable. Incomplete serial-chunk fingerprints include the output
reservation and may require re-extraction if that reservation changes; the normal
fingerprint check remains enforced. The chunk planner reserves the larger of the
two configured extraction ceilings so a reasoning repair has sufficient room.
Extraction always isolates one patient and never sends more than the configured
feature batch. Long records are read in ordered, lossless contiguous chunks of
at most `extraction_chunk_size_tokens` (50,000 by default), preferring nearby
note, paragraph, line, sentence, or word boundaries. Each chunk receives the
validated cumulative scalar extraction from all earlier chunks and returns the
updated structured state. The planner uses the extraction model's own chat
tokenizer and reduces the source chunk when definitions or prior state need
more of the 131,072-token context. Per-chunk inputs, results, and completion
markers are checkpointed, and fingerprints include the prior state, so restarts
continue at the first unfinished compatible chunk without dropping source text.
The exact extraction tokenizer must be present locally under the configured
model ID, either in the managed vLLM download directory or Hugging Face cache.

Cross-page reconciliation is local and deterministic; it does not make another
LLM request. Each frozen ontology carries a conflict strategy (`latest`,
`earliest`, `maximum`, `minimum`, `mode`, `any_positive`, or
`single_or_null`). Verified dates take precedence for temporal strategies and
absolute source order is the documented fallback. Stage 2 writes every
observation, the selected observation ID, policy, and selection basis to the
patient's `reconciliation/decisions.json`. Historical feature definitions
without the structured field receive an explicit, audited compatibility rule
derived from their measurement definition.

Clinical
text remains Unicode instead of expanding into token-heavy ASCII escape
sequences. Independent outer folds execute concurrently. `stage2.workers`
controls combined primary-model concurrency, including consolidation and
supervisor fan-outs; `stage2.extraction_llm.workers` independently controls
small-model patient extraction without combining patients.
Stage 2 sequential candidate consolidation and all-evidence role selection occur
inside each independent outer fold;
those checkpoints are reusable independently of endpoint URLs.

```bash
# Run/resume Stage 1 and stop after the handoff.
uv run python scripts/run_all_evidence.py --config run.json --stage1-only

# Skip Stage 1 and run/resume Stage 2 from the existing handoff.
uv run python scripts/run_all_evidence.py --config run.json --stage2-only

# Run both when run.mode was explicitly set to stage1 in the file.
uv run python scripts/run_all_evidence.py \
  --config run.json --set run.mode=full
```

The same choice can be stored as `run.mode: "full"`, `"stage1"`, or
`"stage2"`. `--stage1-only` and `--stage2-only` override it. Endpoint and model
values can also be supplied as `--stage2-endpoint` and `--stage2-model`.
The small model has `--stage2-extraction-endpoint`,
`--stage2-extraction-model`, `--stage2-extraction-api-key`, and
`--stage2-extraction-workers`. Configure grouped selection under
`stage2.statistical_selection` (or with a `--set` override). Configure the
preceding equivalence-only alias pass under `stage2.selection_consolidation`.
The two output
ceilings use `--stage2-max-tokens` and `--stage2-extraction-max-tokens`.
Managed mode additionally has `--stage2-vllm-servers`,
`--stage2-vllm-gpus`, `--stage2-vllm-rapid-switch-seconds`,
`--stage2-vllm-base-port`,
`--stage2-vllm-download-dir`, `--stage2-vllm-reasoning-parser`,
`--stage2-vllm-language-model-only` (and its `--no-` form),
`--stage2-vllm-default-chat-template-kwargs`, and repeatable
`--stage2-vllm-extra-arg` options.

## Arguments instead of a config

The common settings can be supplied directly:

```bash
uv run python scripts/run_all_evidence.py \
  --dataset /data/cohort.parquet \
  --output-dir /results/my_stage1_run \
  --unit-id-column patient_id \
  --text-column clinical_text \
  --treatment-column treatment_indicator \
  --outcome-column outcome_indicator \
  --outcome-type binary \
  --outer-folds 5 \
  --inner-folds 5 \
  --seed 42 \
  --devices cuda:0,cuda:1 \
  --workers 16 \
  --htr-model /models/bert-tiny \
  --embedding-model /models/qwen3-embedding-8b \
  --stage2-endpoint http://127.0.0.1:8010/v1 \
  --stage2-model Qwen/Qwen3.8-27B \
  --stage2-extraction-endpoint http://127.0.0.1:8020/v1 \
  --stage2-extraction-model small-extractor \
  --stage2-max-tokens 100000 \
  --stage2-extraction-max-tokens 75000 \
  --stage2-extraction-chunk-size-tokens 50000 \
  --stage2-review-rounds 2 \
  --stage2-estimation-trees 200 \
  --set stage2.statistical_selection.l1_ratio=0.8
```

Any less-common nested value can be set without adding another CLI flag:

```bash
uv run python scripts/run_all_evidence.py \
  --config my_run.json \
  --set science.stage1.architecture.multi_model_forest.tfidf_topic.topic_count=50 \
  --set science.neural_queries.query_epochs=80
```

Command-line values override the config file.

## Clinical modifier concepts and architecture selection

In `multi_model` mode, set
`stage2.statistical_selection.multi_model.modifier_count.concept_review=true`
to review the concepts represented by the top candidates from each inner fold.
`concept_top_n_per_fold` defaults to 100. Python assembles each union and its
recurrence counts. The LLM groups clinically related measurements, explains the
evidence and uncertainty, and nominates existing representative measurements.
Retained representatives and explicitly nominated uncertain representatives
enter the subsequent modifier-count search. Confounder roles are preserved.

This review is repeated inside each count-validation training partition using
numerical evidence computed within that partition. Each final estimator/count
combination is scored on the same held-out overlap-eligible patients using
R-loss. The default candidates are an honest causal forest and a penalized
linear outcome model with treatment interactions (logistic for binary outcomes).
The final review uses all outer-training evidence after the search selects an
architecture and count budget. Available representatives can be fewer than the
budget; the audit records the actual counts. Investigator-locked roles remain
fixed. No oracle values enter these steps.

Concept review is disabled when omitted. It adds repeated numerical fits and
LLM reviews; each is checkpointed. Rankings are constructed from clinical
ordered groups and pairwise comparisons of leading candidates. Python owns
positions, deterministic ties, and list merging. The candidate definitions and
measurement ontology are still learned on the outer-training set, so the inner
R-loss comparison is conditional on that upstream adaptation. Outer-held-out
evaluation remains the final test of the whole workflow.

A fresh run against preserved Stage 1 files can use
`scripts/run_stage2_from_artifacts.py CONFIG.json`. The launcher projects the
dataset to the four declared patient/text/treatment/outcome columns, records
source hashes and endpoint identities, checks tokenizer availability, and keeps
status and logs alongside the dated run configuration. `--preflight` performs
those checks without starting discovery or fitting.

## Parallel execution

Top-level components run in order. The embedding cache must exist before its
consumers run, and TF-IDF writes the shared outer and inner split definitions
used by the other evidence families. Within those boundaries, the runner uses
the available CPUs and GPUs as follows.

| Component | Unit of parallel work | Concurrency control |
|---|---|---|
| `embedding_cache` | A contiguous shard of the complete chunk corpus | One encoder worker per entry in `run.devices` when multiple CUDA devices are configured |
| `tfidf` | An outer/full or exact-inner context | Separate CPU processes, bounded by `run.workers` and the number of contexts; native numerical threads are limited to one per process |
| `text_models` | An outer/full or exact-inner context, with independent BoW folds inside it | One fixed process lane per configured CUDA device; `run.workers` is divided among active lanes and bounds their combined BoW fold threads |
| `neural_queries` | An outer/full or exact-inner context | One fixed process lane per configured CUDA device; CPU-only runs use at most `run.workers` lanes |
| `handoff` | None | The completed JSONL files are combined serially |
| `stage2` | Independent outer folds; deterministic pair chunks within each inner fold | Outer folds execute concurrently; primary calls are bounded by `stage2.workers`, small-model extraction calls by `stage2.extraction_llm.workers`, and pair chunks from every fold share one loky pool capped by `stage2.workers`; role passes remain ordered |

A fixed CUDA lane processes its assigned contexts serially on one GPU. This
provides device affinity and prevents a process queue from placing two
simultaneous contexts on the same GPU. The runner creates at most one lane per
configured CUDA device, bounded by the number of unfinished contexts, and
assigns the largest remaining context to the least-loaded lane. Each
neural-query context performs its inner-fold fits, final query-bank fits, and
evidence retrieval on its assigned device. Thus the principal neural-query
parallelism is across independent contexts.

The CPU budget is shared rather than repeated across the text-model lanes. For
`W` requested CPU workers and `L` active lanes, each lane receives at least
`floor(W / L)` workers, and the remainder is assigned one worker at a time.
BoW nuisance and effect cross-fitting use that lane allocation to execute
independent folds concurrently. They use threads so the enclosing process lane
does not create a nested process pool or copy the context dataset. The three
full-context feature-importance fits are also concurrent. Native linear-algebra
and estimator thread pools remain limited to one thread per fit. If `W < L`, one
controller worker per active lane is the unavoidable minimum.

The context directories remain the scheduling unit, while text-model work is
recoverable within a context. BoW nuisance/effect/matched-pair folds, HTR
nuisance/effect/pair folds, full-view feature-importance fits, embedding contrasts,
the assembled feature bundle, and the final handoff row publish atomic
checkpoints as they finish. Neural checkpoints contain fold predictions and
evidence, not epoch, optimizer, or model state. Each lane writes the context's
`complete.json` only after its final artifacts are durable. Rerunning the same
command restores compatible substeps and redistributes only unfinished contexts.

Resume validation is intentionally lightweight. Checkpoint markers cover the
schema and implementation version, relevant scientific configuration, seed,
ordered fit/held-out row IDs, dataset path/size/mtime, and upstream marker IDs.
They validate referenced files by presence and recorded size; they do not hash
checkpoint bytes or rescan the dataset. A per-context advisory lock prevents two
restarts from fitting the same context concurrently, and
`checkpoint_progress.json` shows running, completed, reused, failed, or invalid
substeps.

## Output and resume

There is one directory to inspect and preserve:

```text
my_stage1_run/
  run_config.json
  resolved_stage1_model_config.json
  resolved_neural_query_config.json
  progress.json
  logs/
    workflow.log
  components/
    embedding_cache/
      cache/...
      complete.json
    tfidf/
      predictions.parquet
      split_provenance.jsonl
      evidence.jsonl
      stage1_tfidf_topics/contexts/...
      complete.json
    text_models/
      outer_001_full/
        evidence.json
        checkpoints/v1/
          checkpoint_progress.json
          context.lock
          units/...
        worker_artifacts/...
        complete.json
      outer_001_inner_001/...
      evidence.jsonl
      complete.json
    neural_queries/
      outer_001_full/...
      outer_001_inner_001/...
      evidence.jsonl
      complete.json
  stage1_architectures/              # explicit selection or evaluator backfill
    manifest.json
    <architecture>/evidence.jsonl
  handoff/
    evidence.jsonl
    index.json
    complete.json
  evaluations/
    stage1/
      evaluation_manifest.json
      metrics.jsonl
      comparison.csv
      summary.json
      architectures/<architecture>/metrics.jsonl
  stage2/
    config.json
    evidence_compilation/
      packets.jsonl
      summary.json
      compile_complete.json
      outer_001/
        cards.jsonl
        members.jsonl
        lineage.jsonl
    outer_001/
      input_packets.jsonl
      interpretations/...
      interpreted_candidates.json
      consolidation/...
      feature_definitions.json
      definitions_complete.json
      ontology_supervision/
        supervisor_cache/<fingerprint-prefix>/<fingerprint>/...
        round_001/
          definitions_before_extraction.json
          extraction/
            batches/...
            failure_summary.json
            extracted.csv
            complete.json
          failure_ontology_refinement/
            round_001/
              feature_.../
              extraction/...
              result.json
              complete.json
            result.json
            complete.json
          aggregate_extraction_summary.json
          supervisor/
            feature_0001/...
            result.json
            complete.json
          complete.json
        convergence.json
      preselection/                 # present after guarded reselection
        input.json
        complete.json
      selection/
        candidate_consolidation/
          input.json
          steps/...
          registry.json
          report.json
          complete.json
        statistical_evidence.json
        role_adjudication/
          evidence.json
          prompt.json
          batches/batch_NNN/{prompt.json,response.json,complete.json}
          response.json
          complete.json
        elastic_net_selection.json
        selected_definitions.json
        measurement_definitions.json
        selected_latent_states.json       # selected latents and recursive ancestors
      final_definitions.json
      extraction/
        all_candidates_fit/...
        fit/extracted.csv
        heldout/
          batches/...
          harmonized.csv
          complete.json
        extracted_features.csv
      estimation/
        predictions.csv
        diagnostics.json
        complete.json
      complete.json
    features_by_outer_fold.jsonl
    cross_fitted_predictions.csv
    posthoc_predictions_with_oracle_ite.csv
    posthoc_oracle_ite_metrics.json
    causal_estimate.json
    summary.json
    reselection_state.json          # present after guarded reselection
    reselection_archives/...
    complete.json
```

`progress.json` is the first place to look while a run is active. The model
outputs are under `components/<name>/`. The plain Stage 2 boundary is
`handoff/evidence.jsonl`; `handoff/index.json` explains the source files and
references the original per-family JSONL files under `components/` without
adding separate per-family copies under `handoff/`. Python consumers can stream
the combined rows with:

```python
from oci.inference.research_all_evidence_workflow import iter_stage1_handoff

for evidence_context in iter_stage1_handoff("/results/my_stage1_run"):
    ...
```

Completion has one intentionally simple rule:

- if `components/<name>/complete.json` exists, that component is reused and its
  progress status is `already_complete`
  (`handoff/complete.json` and a causal-estimation `stage2/complete.json` serve
  the same purpose for those components);
- if it does not exist, the component runs in the existing directory;
- text-model and neural-query contexts use the same rule inside their context
  directories.

A causal-estimation marker is not accepted as final when descendants contain a
legacy request-exhaustion fallback. The runner preserves the old leaf files as
`superseded_infrastructure_*`, retries only those affected extraction slices,
and then rebuilds their dependent summaries and estimates.

An interrupted component's partial files are left in place. There is no
`--resume` flag. Rerun the same command. Stage 2 skips completed interpretation
and consolidation leaves, extraction batches, ontology-supervision leaves, and
completed fold estimates.
The compiled packet plan is cached under `stage2/evidence_compilation/`; on a
restart the runner hashes the handoff and normally loads this small cache rather
than reparsing the raw Stage 1 evidence. Single-card interpretation requests
are skipped only when their input fingerprint (compiled card, model identity,
and discovery version) matches.
It writes an outer-fold completion marker only after held-out estimation, and
writes the final top-level marker only after the cross-fitted estimates have
been assembled.

### Reselect a completed Stage 2 run

Use `--stage2-reselect` to replace selection and its downstream estimates while
reusing the completed interpretation, consolidation, ontology supervision, and
all-candidate outer-training extraction:

```bash
uv run python scripts/run_all_evidence.py \
  --config /path/to/completed_run/run_config.json \
  --stage2-only \
  --stage2-reselect
```

This is a guarded migration, not a broad `--rerun stage2`. Before moving any
result it verifies every fold's feature-definition fingerprint, completed
selector input, post-ontology definitions, training-matrix columns and row IDs,
source text, treatment and outcome values, inner splits, review policy, and
primary/extraction model IDs. Saved configurations from the retired p-value
selector are accepted only in this mode; its obsolete selection settings are
ignored while the current `statistical_selection` defaults or overrides are
applied. The current `univariable_confounder_p_value_threshold` and
`univariable_confounder_q_value_threshold` fields remain active as descriptive
evidence thresholds, not inclusion gates.

The prior selection, selected-feature extraction, estimation, and root results
are moved without deletion to `stage2/reselection_archives/`. Each fold receives
a fingerprinted `preselection/` snapshot that directly loads the verified
post-ontology definitions, harmonized training matrix, and a manifest for its
archived raw held-out measurements. Consequently a cache miss cannot silently
trigger interpretation or outer-training extraction. Interrupted preparation
resumes through `stage2/reselection_state.json`.

The primary and extraction model IDs must be the same as in the completed run;
transport endpoints and worker counts may change. Group-elastic-net selection
runs from scratch. For each selected original feature, raw held-out values are
reused when the archived column has matching row, source-text, model, frame, and
measurement-definition fingerprints. Only missing or definition-incompatible
measurements are sent to the extraction LLM. Causal estimation then runs again.
`--stage2-reselect` cannot be combined with `--rerun`.

Raw-packet interpretation checkpoints do not match semantic-card inputs and are
therefore rerun automatically. If a prior run already produced
`feature_definitions.json`, the runner fails closed instead of silently mixing
those definitions with a new evidence plan or a changed explicit-feature
configuration; preserve that directory for audit and select a fresh Stage 2
output directory for the new definition inputs. The Stage 1 handoff itself does
not need to be regenerated when only `stage2.explicit_features` changes.

The supervision boundary is deliberately fold-honest. Every value supplied to
the aggregate ontology supervisor comes from outer-training extraction. The
primary model sees per-feature counts and aggregate values plus validation
failure patterns, but no patient text, treatment, outcome, causal-role evidence,
model performance, or p-values. Its response is bounded to `keep` or a same-
feature ontology revision. It cannot add, drop, split, merge, rename, or assign
a role. Stage 2 re-extracts only revised features, merges them into the cached
raw training matrix, and repeats for at most
`max_review_rounds`; `ontology_supervision/convergence.json` records whether the
latest aggregate ontology was stable.

Before supervised evidence construction, `selection/candidate_consolidation/`
records the optional sequential, outer-training-only pass. When disabled, its
report records that status and the candidate set passes through unchanged.
When enabled, the loop visits the original candidate
order once. At each still-active pivot, the configured embedding model retrieves
the nearest active neighbors (ten by default), Python calculates Spearman,
bias-corrected Cramer's V, or correlation-ratio evidence as appropriate, and the
primary model chooses no replacement or one or more disjoint structured latents.
Every source pair in a replacement must have an evaluable association of at
least `minimum_pairwise_association` (0.85 by default), but this is necessary,
not sufficient. The sources must be aliases for the same attribute, entity,
time scope, granularity, and scale. Broader/narrower concepts, component/total
relationships, and merely correlated measurements must remain separate. An
accepted canonical measurement must consume the pivot and preserve source value
type, units or category cardinality, and missingness using only coalescing or a
lossless synonymous-category recode. Nominal categories may use a canonical
union when every source category is retained injectively; binary and ordinal
scales retain full cardinality. Continuous coalescing treats values that violate
a continuous ontology as missing and falls through to the next valid alias.
Whenever multiple valid sources are nonmissing on one training row, their
numeric values or canonical recoded categories must agree; conflicting overlaps
reject the replacement instead of being resolved by source order. It
immediately replaces its aliases in
the active pool, so later pivots retrieve the canonical measurement instead.
Treatment, outcome, causal roles, and outer-heldout rows are absent from every
consolidation request.

### Numerical evidence and selection modes

For the opt-in `multi_model` mode, see [Stage 2 selection from multiple
models](stage2_multi_model.md). It combines repeated penalized models,
univariable screens, R-learners, and predictive/causal forests with cross-candidate
LLM theme review and final role adjudication. The following describes the
existing `llm_roles` and `independent_tasks` numerical components.

Final selection is written to the historically named
`selection/elastic_net_selection.json`; its standalone numerical component is
also written to `selection/statistical_evidence.json`. Every inner-training partition fits a
logistic group elastic net for treatment and a separate group elastic net for
the marginal outcome. Ordered measurements use one standardized numerical
score. A nominal factor's standardized contrasts and missingness indicator are
penalized as one group, so its selection does not depend on one surviving dummy
coefficient. Feature groups earn treatment and outcome votes separately. A
single vote in either task places the feature in a provisional confounder
union. In parallel, candidate-wise omnibus models test association with
treatment, marginal outcome, and outcome adjusted for treatment. Raw p-values,
within-fold Benjamini-Hochberg q-values, and cross-fold support counts are
evidence only, never hard gates. The
`univariable_confounder_p_value_threshold` (default 0.05) and
`univariable_confounder_q_value_threshold` (default 0.10) create nominal and
multiplicity-adjusted joint-support flags for treatment and treatment-adjusted
outcome association. They do not retain or discard a feature automatically.
These tests concern adaptively discovered candidates; the within-fold q-values
do not adjust for the complete upstream discovery and selection procedure.

In `llm_roles`, the propensity and marginal-outcome grouped elastic nets used
to construct modifier evidence both use the provisional common union. In
`independent_tasks`, each inner fold instead fits both nuisances over all
candidates, with separate regularization choices and no cross-fold screen-union
restriction.
For binary targets, the report records inner-heldout and pooled out-of-fold
AUROC alongside log loss.

Those elastic nets generate one inner-heldout propensity and outcome prediction
for every outer-training row. Within each inner fold, candidate-specific grouped
elastic-net calibrations augment both nuisances using nested cross-fitting. A
ridge-stabilized R-learner compares constant and candidate-varying treatment-
effect models on untouched inner-heldout rows. All estimable interaction
contrasts for a categorical candidate enter together and receive one held-out
R-loss score. The ten largest gains per inner fold are selected by default,
without a positive-gain gate. A second modifier view jointly fits all candidate
treatment-interaction groups in one grouped-elastic-net R-loss model per fold.
Its coefficient supports and held-out whole-model R-loss gains are retained.
The joint supports are evidence in `llm_roles` and the binding effect-selection
criterion in `independent_tasks`. Nuisance screens use the one-standard-error
rule by default; nuisance prediction and joint modifier fitting default to
minimum-CV-loss choices.

`stage2.statistical_selection.selection_mode` controls final selection:

| Mode | Binding inclusion and routing | LLM role |
| --- | --- | --- |
| `llm_roles` (default when omitted) | Final adjudicated roles; if adjudication is disabled, the provisional nuisance union and candidate-wise top-N modifier union | Reconciles all aggregate evidence and assigns confounder, effect modifier, both, or neither |
| `independent_tasks` | Separate treatment, outcome, and joint-R-loss effect supports; a nonzero group in any inner fold suffices for its task | Optional advisory annotations cannot add, remove, or reroute features |
| `multi_model` | Final roles from seven evidence families across training resamples and forest feature subsets; no individual screen is binding | Required theme reconciliation across candidates, then evidence-cited role decisions |

In `llm_roles`, the primary LLM receives bounded slices of one allowlisted
aggregate role-evidence artifact and assigns final roles. A slice contains at most
`role_adjudication.max_candidates_per_request` candidates (20 by default) and
retains their global votes, ranks, and fold evidence. The adjudicator must
reconcile multivariable and univariable confounder evidence, candidate-wise and
joint modifier evidence, method disagreement, and fold consistency. Its
interface has no dataset argument and excludes row values, identifiers,
outer-heldout information, oracle fields, dataset paths or names, and
data-generation metadata. Investigator-locked roles remain exact. Failure does
not fall back silently to the provisional unions. Explicitly setting
`stage2.role_adjudication.enabled: false` uses those provisional roles instead.

In `independent_tasks`, effect selection requires neither a nuisance-task vote
nor a candidate-wise top-N rank. P/q values and candidate-wise ranks remain
diagnostics. Annotation transport/validation failures do not veto numerical
estimation. Use `selection_authority`, `modeling_tasks`, and `taskwise_routing`
to interpret saved decisions; the legacy `final_role_assignment` string can
still name the LLM branch without making annotations binding. The external
propensity model uses treatment-task features; external outcome models use
outcome-task features **plus every selected effect modifier**. Forest `X`
contains effect features, `W` contains the treatment/outcome union excluding
`X`, and both internal forest nuisance models regularize over `X + W`.
See [independent task selection](stage2_independent_tasks.md) for routing,
investigator overrides, diagnostics, and saved-run commands.

Explicit investigator features retain exactly their configured roles. The
outer-heldout partition is inaccessible during selection and receives only the
selected original measurement dependencies afterward. Fitted latent states are
then applied in creation order, including recursive latent ancestors.

The nuisance votes, cross-fitted elastic-net diagnostics, candidate-specific
held-out R-loss comparisons, grouped categorical interactions, and final
decisions are checkpointed under `selection/`.
Changing only endpoint URLs does not invalidate completed scientific
checkpoints. `model_identity.json` records the model IDs actually advertised at
startup. Changing the primary model raises an error before interpretation
checkpoints are reused. The extraction model may change after
extraction-and-later outer-fold artifacts are removed; completed interpretation,
consolidation, and feature-definition checkpoints do not depend on the extractor
identity and are retained.

Continuous definitions accept either a numeric scalar or a documented
categorical/threshold scalar when no exact number is reported. The extraction
summary records numeric and fallback-category distributions before modeling.

When a continuous extraction contains both numeric and categorical values, a
separate LLM request receives only the feature ontology and aggregated
outer-training values. It chooses one validated continuous mapping or an
exhaustive categorical binning plan. That plan is frozen in the feature
definition and applied deterministically to final training and held-out values;
held-out values and outcomes are never sent back for harmonization. Later
outer-training tokens extend only the frozen value map rather than regenerating
the full ontology. Mapping bookkeeping defects are normalized conservatively and
recorded, with missing or conflicting mappings assigned null. If bounded repairs
still fail, the feature-level `fallback.json` records the validation error and
the fold continues. A safe prior plan is retained with unresolved new tokens
mapped to null; without a prior plan, raw mixed values remain available to the
`continuous_with_categorical_fallback` encoder. The round-level
`harmonization.json` reports all normalized maps and validation fallbacks; final
fold definitions, fold completion records, and the root summary propagate the
same fallback audit trail.

Every single-patient extraction writes `extraction_issues.json`, including
feature-attributable invalid scalar/type values and values outside a declared
closed ontology. The extraction directory aggregates those events by feature,
failure kind, and distinct patient in `failure_summary.json`. A generic malformed
response is counted separately as a structural transport/format failure and
cannot trigger an ontology change. When the same attributable pattern occurs in
at least `ontology_refinement_min_failure_patients` outer-training patients (3
by default), a one-feature ontology-refinement component receives the frozen
definition, aggregate count, and a bounded list of failed model outputs—but no
patient text, treatment, outcome, or held-out information. It may keep the
definition or revise only its description, value type, categories or unit,
measurement rule, and missingness rule. The feature name, identity, roles, and
support remain fixed. The revised features are then re-extracted across the
training patients and merged with cached unchanged-feature columns before
monitoring runs again, for at most `max_ontology_refinement_rounds` revisions (2 by default),
before aggregate ontology supervision proceeds. Investigator-configured explicit ontologies
are immutable: their repeated failures are audited, but no refinement request is
made. The held-out extraction begins only after
`final_definitions.json` has been written. The reported average treatment effect
is the mean of the held-out AIPW scores across outer folds; its standard error
is the empirical standard error of those scores. `estimated_cate` in the
prediction file comes from an honest `CausalForestDML` fit on the outer-training
rows. Effect modifiers form its heterogeneity matrix and pure confounders form
its controls; dual-role variables are represented once in the heterogeneity
matrix. A constant heterogeneity design is used when no modifier survives, so
the final model remains a causal forest. Fold diagnostics include its fit audit
and held-out effect confidence intervals. The supported nuisance path is
elastic-net-only; the obsolete strict random-forest runtime-config contract is
retired. The fit audit records every fitted cross-fit nuisance clone, including
its effective CV folds, selected regularization, and optimizer iteration-limit
status.

If the dataset supplies `true_ite_prob`, Stage 2 hashes and freezes
`cross_fitted_predictions.csv` before joining that oracle column. The final
`causal_estimate.json` includes overall Pearson and Spearman correlations
between `estimated_cate` and oracle ITE. Detailed overall and per-fold
correlations, MAE, RMSE, ATE bias, and ITE dispersion are written to
`posthoc_oracle_ite_metrics.json`; the joined rows are stored separately in
`posthoc_predictions_with_oracle_ite.csv`. Without an oracle column, the same
metrics file records `available: false`.

Stage 1 itself can be evaluated architecture by architecture after the handoff
has frozen its evidence:

```bash
uv run oci-evaluate-stage1 \
  --run-dir /results/my_stage1_run \
  --metadata /data/synthetic/metadata.json \
  --architectures all
```

The evaluator hashes all consumed evidence and row-score sidecars before it
loads oracle-bearing data. It never fits, ranks, or selects Stage 1 models. The
per-architecture files contain native metrics appropriate to each evidence
representation plus a common recovery view; `comparison.csv` is the compact
cross-architecture summary. A completed legacy handoff is backfilled into the
same additive `stage1_architectures/` contract without refitting.

To intentionally rerun one component, use:

```bash
uv run python scripts/run_all_evidence.py \
  --config my_run.json \
  --rerun tfidf
```

This removes that component's completion markers; it does not delete model
outputs. Contexts are then recomputed in their existing directories.

To print status without starting work:

```bash
uv run python scripts/run_all_evidence.py \
  --config my_run.json \
  --status
```

The workflow does not rely on one global config-identity gate. Stage 2 validates
scientific fingerprints at individual checkpoints, and incompatible changes
can require recomputation or raise a compatibility error. A coarse progress
marker alone does not establish that saved outputs match a changed config.
Use a new output directory for a scientifically different run, the guarded
`--stage2-reselect` path for supported downstream changes, or an explicit rerun
of the affected components when appropriate.

## Components

- `embedding_cache` encodes reusable text chunks.
- `tfidf` runs first and writes the plainly readable fold definitions used by
  the remaining evidence families, plus topic and residual n-gram evidence.
- `text_models` runs the established BoW, HTR, matched-pair, whole-cohort
  embedding, cluster-local embedding, and lexical-retrieval evidence models.
- `neural_queries` fits and saves each outer/full or exact-inner context
  independently.
- `handoff` gathers the completed evidence into the stable Stage 2 input path.
- `stage2` exhaustively interprets semantic cards, supervises small-model
  extraction ontologies, builds statistical evidence, applies the configured
  LLM-role or independent-task selection policy, and writes held-out
  causal-forest effects and AIPW scores before aggregating the outer folds.

The Stage 1 scientific model implementations are reused. The plain Stage 2 path
replaces the former production control plane with readable directories,
auditable regression screens, an honest causal forest, and cross-fitted AIPW
scores. These choices are recorded in the fold diagnostics and final estimate
rather than hidden in an authenticated deployment specification.
