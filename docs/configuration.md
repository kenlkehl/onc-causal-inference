# Configuration and operations

This page is the reference for configuring, launching, resuming, and inspecting
an all-evidence workflow run. It covers the configuration file, the workflow
command line, the synthetic launcher `run_synthetic.sh`, request and recovery
behavior, sampling, parallelism, the output directory, and Stage 2-only
launches. LLM serving (external endpoints and managed vLLM pools) is described
in [Serving](serving.md); what the pipeline does with these settings is in
[How OCI works](how_it_works.md) and [Stage 2 selection](stage2_selection.md).
Terms such as [handoff](glossary.md#handoff),
[interpretation LLM and extraction LLM](glossary.md#interpretation-llm-and-extraction-llm),
and [fresh run and saved run](glossary.md#fresh-run-and-saved-run) are defined
in the [glossary](glossary.md).

## Ways to configure a run

The workflow is `scripts/run_all_evidence.py` (equivalently,
`python -m oci.inference.research_all_evidence_workflow`). Settings come from
three places, applied in this order:

1. A JSON or YAML configuration file passed with `--config` (YAML requires
   PyYAML). Relative paths in the file are resolved against the file's
   directory.
2. Dedicated command-line flags, such as `--dataset` or `--architectures`
   (see [Command-line flags](#command-line-flags)).
3. `--set KEY=VALUE` overrides, applied in the order given. `KEY` is a dotted
   path into the configuration, for example `stage2.workers` or
   `science.stage1.architecture.htr_max_chunks`. `VALUE` is parsed as JSON when
   possible (`--set stage2.min_propensity=0.1`,
   `--set 'stage2.statistical_selection.multi_model.modifier_count.estimators=["dina"]'`)
   and is otherwise used as a string.

Unknown keys anywhere under `stage2` are errors: a misspelled Stage 2 setting
stops the run instead of silently falling back to a default.

At startup the workflow writes the resolved settings to
`OUTPUT_DIR/run_config.json`, with API keys replaced by `<redacted>`. That file
can be passed back to `--config` to resume or relaunch the run.

Start from `example_configs/research_all_evidence_multi_model.json`, which
configures both stages. `example_configs/research_all_evidence.json` leaves
`stage2.endpoint` and `stage2.model` empty, so it runs Stage 1 only until an
interpretation LLM is configured.

## Configuration file reference

A configuration file has these top-level blocks:

```json
{
  "dataset": "data/cohort.parquet",
  "output_dir": "results/cohort_run",
  "columns": {},
  "science": {},
  "models": {},
  "stage2": {},
  "run": {}
}
```

### Top level

| Key | Default | Meaning |
|---|---|---|
| `dataset` | required | Parquet or CSV file with one row per patient |
| `output_dir` | required | Run directory; it holds all results and is the resume checkpoint |
| `stage1_template` | `example_configs/all_evidence_stage1_model.json` | Stage 1 text-model template (see [Stage 1 template settings](#stage-1-template-settings)) |
| `neural_query_template` | `example_configs/all_evidence_neural_query_model.json` | Template for the `neural_query_moments` architecture |

### `columns`

| Key | Default | Meaning |
|---|---|---|
| `unit_id` | `patient_id` | Unique, nonmissing patient identifier |
| `text` | `clinical_text` | Pretreatment clinical text |
| `treatment` | `treatment_indicator` | Binary treatment, 0 or 1 |
| `outcome` | `outcome_indicator` | Outcome, binary or continuous as declared by `science.outcome_type` |

All supplied text is treated as pretreatment; the workflow adds no date-based
screening. Columns whose names start with `true_`, `oracle_`, or `ground_truth`
are treated as [oracle columns](glossary.md#oracle-columns) and are never used
for modeling.

### `science`

| Key | Default | Meaning |
|---|---|---|
| `clinical_question` | `"Identify text-derived confounders and treatment-effect modifiers."` | Study question given to the Stage 1 and Stage 2 language models |
| `outcome_type` | `binary` | `binary` or `continuous` |
| `outer_folds` | 5 | Outer cross-fitting folds ([outer and inner folds](glossary.md#outer-and-inner-folds)) |
| `inner_folds` | 5 | Inner folds within each outer-training set |
| `seed` | 42 | Seed for fold splits and models |
| `stage1_architectures` | omitted | Comma-separated string or list of Stage 1 architectures, or `all` (all ten; rejected for continuous outcomes, where `matched_pair_uplift` is unavailable). Omitted runs the architectures enabled by the Stage 1 template: eight with the bundled template (seven for continuous outcomes, which skip `matched_pair_uplift`). See [Stage 1 architectures](stage1_architectures.md) |
| `htr_enabled` | `true` | `false` disables all HTR modeling, including matched-pair HTR (same as `--disable-htr`) |
| `stage1` | `{}` | Overrides merged into the Stage 1 template |
| `neural_queries` | `{}` | Overrides merged into the neural-query template |

An architecture selection is recorded in `stage1_architectures/manifest.json`
and cannot change when a run resumes; use a different output directory to
change it.

### `models`

| Key | Default | Meaning |
|---|---|---|
| `htr` | `prajjwal1/bert-tiny` | Chunk encoder for the [HTR](glossary.md#htr) architectures (Hugging Face ID or local path) |
| `embeddings` | `Qwen/Qwen3-Embedding-8B` | Embedding model for the embedding architectures and the neural-query architecture |

### `run`

| Key | Default | Meaning |
|---|---|---|
| `devices` | `["cpu"]` | Stage 1 devices, as a list or comma-separated string such as `cuda:0,cuda:1` |
| `workers` | 1 | Stage 1 CPU budget; also the default for `stage2.workers` (capped at 4) |
| `components` | all five Stage 1 components | Subset of `embedding_cache`, `tfidf`, `text_models`, `neural_queries`, `handoff`. With an architecture selection, components the selection does not need are skipped |
| `mode` | `full` when Stage 2 is configured, otherwise `stage1` | `full`, `stage1`, or `stage2` (the flags `--stage1-only` and `--stage2-only` set this) |
| `log_level` | `INFO` | Logging level for `logs/workflow.log` |

### `stage2`

Stage 2 is enabled when `stage2.endpoint`, `stage2.model`, or `stage2.vllm` is
set, and it also requires `stage2.extraction_llm`. The tables below group the
keys; an omitted key takes the default shown.

**Interpretation LLM and transport**

| Key | Default | Meaning |
|---|---|---|
| `endpoint` | empty | OpenAI-compatible base URL of the interpretation LLM. Mutually exclusive with `vllm` |
| `model` | empty | Served model ID. With `endpoint`, it may be omitted when the server advertises exactly one model. Required with `vllm` |
| `api_key` | `EMPTY` | API key; when absent from the configuration, `OCI_STAGE2_API_KEY` is used |
| `vllm` | none | Managed vLLM pool for the interpretation LLM ([Serving](serving.md#managed-vllm-pools)) |
| `vllm_rapid_switch_seconds` | 900 | Managed pools only: when two managed models switch again within this interval, both stay resident on their configured GPUs for the rest of the run; 0 always alternates |
| `workers` | `min(4, run.workers)` | Maximum concurrent interpretation requests across all outer folds |
| `frozen_feature_definition_model` | empty | Reuse completed feature definitions produced by this model ID while `model` serves the remaining requests. Definitions that are missing are not regenerated |
| `extraction_llm` | none | Extraction LLM settings (next table) |

**`stage2.extraction_llm`**

Configure exactly one of `endpoint`, `endpoints`, or `vllm`.

| Key | Default | Meaning |
|---|---|---|
| `endpoint` | empty | OpenAI-compatible base URL of the extraction LLM |
| `endpoints` | none | List of equivalent prestarted servers, each `{"endpoint": URL, "max_concurrency": N}`, used with load-aware routing |
| `vllm` | none | Managed vLLM pool for the extraction LLM ([Serving](serving.md#managed-vllm-pools)) |
| `model` | empty | Served model ID; may be omitted for an external server that advertises exactly one model. Required with `vllm` |
| `api_key` | `EMPTY` | API key; when absent, `OCI_STAGE2_EXTRACTION_API_KEY` is used |
| `workers` | `min(4, run.workers)`, or the sum of `max_concurrency` with `endpoints` | Maximum concurrent extraction requests |
| `runtime_endpoint`, `runtime_model`, `runtime_api_key` | empty, empty, `EMPTY` | Continuation route: send remaining extraction requests to this server and model while checkpoints keep the configured model as their identity. Set endpoint and model together; `OCI_STAGE2_EXTRACTION_RUNTIME_API_KEY` supplies a missing key |

**Requests, repairs, and recovery** (see [Request and recovery defaults](#request-and-recovery-defaults))

| Key | Default | Meaning |
|---|---|---|
| `request_timeout` | 14400 | Seconds allowed for one logical request, including transport retries and repairs |
| `request_attempt_timeout` | 3600 | Seconds allowed for one HTTP attempt |
| `transport_max_attempts` | 6 | HTTP attempts per response attempt |
| `transport_retry_backoff` | 2.0 | Base of the exponential transport backoff, in seconds |
| `outer_fold_recovery_attempts` | 2 | Times an outer fold is resumed from its checkpoints after a request exhausts its budget |
| `outer_fold_recovery_backoff` | 60.0 | Seconds before the first fold recovery; multiplied by the attempt number |
| `max_response_repairs` | 15 | Validator-guided repair turns for an invalid response |
| `thinking_after_response_repairs` | 5 | Repairs after this many turn reasoning on (at least `high`) |
| `interpretation_reasoning_effort` | `auto` | `auto`, `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, or `max` |
| `extraction_reasoning_effort` | `none` | Same values; `none` disables thinking for initial extraction requests |
| `max_tokens` | 100000 | Output ceiling for interpretation requests (minimum 100000); not a target length |
| `extraction_max_tokens` | 75000 | Output ceiling for extraction requests without thinking (minimum 4096) |
| `extraction_reasoning_max_tokens` | null | Output ceiling for extraction requests with thinking; null uses `extraction_max_tokens` |
| `extraction_stream` | `false` | Stream extraction responses and record per-request events |
| `extraction_deferred_retry_passes` | 1 | Passes that retry patients whose requests exhausted their budget, after the other patients finish; 0 fails immediately |
| `runtime_disable_extraction` | `false` | Keep the configured extraction model as checkpoint identity but never launch or call it; any extraction request fails. For reselection runs whose extractions are complete |

**Evidence, discovery, and extraction definitions**

| Key | Default | Meaning |
|---|---|---|
| `evidence_max_cards_per_fold` | 400 | Maximum [evidence cards](glossary.md#evidence-card) per outer fold (minimum 16) |
| `evidence_max_exemplars_per_card` | 4 | Text excerpts per card |
| `evidence_max_exemplar_chars` | 2400 | Characters per excerpt (minimum 256) |
| `max_prompt_chars` | 100000 | Character limit for interpretation prompts |
| `consolidation_batch_size` | 20 | Candidates per duplicate-merging request (minimum 2) |
| `consolidation_max_rounds` | 55 | Maximum duplicate-merging rounds |
| `consolidation_max_prompt_chars` | 640000 | Character limit for duplicate-merging prompts |
| `consolidation_policy` | see below | How merging batches are formed and when merging stops |
| `operationalization_max_prompt_chars` | 640000 | Character limit for prompts that write [extraction definitions](glossary.md#extraction-definition-ontology) |
| `max_review_rounds` | 2 | Rounds of aggregate extraction review per outer fold |
| `ontology_refinement_min_failure_patients` | 3 | A recurring extraction failure triggers definition refinement when it affects at least this many patients (minimum 2) |
| `max_ontology_refinement_rounds` | 2 | Refinement rounds per review round; 0 disables refinement |
| `required_architectures`, `included_architectures` | the eight default architectures; all | Architectures whose evidence must be present, and architectures whose evidence is compiled. The workflow sets both from the frozen Stage 1 selection; they matter only for `scripts/run_stage2_from_artifacts.py`. Accept a list, a comma-separated string, or `all` |
| `explicit_features` | `[]` | Investigator-specified variables (see [Explicit features](#explicit-features)) |

`stage2.consolidation_policy`:

| Key | Default | Meaning |
|---|---|---|
| `semantic_fraction` | 0.6 | Fraction of merging batches built from embedding neighbors; the rest are alphabetical |
| `random_fraction` | 0.0 | Must be 0 |
| `embedding_model`, `embedding_device` | `Qwen/Qwen3-Embedding-0.6B`, `cpu` | Sentence embedding model used to form semantic batches |
| `early_stop_min_rounds` | 3 | Rounds always run before early stopping is considered |
| `early_stop_patience` | 2 | Consecutive low-yield rounds that stop merging |
| `early_stop_min_reduction` | 0.005 | Minimum fractional reduction in candidates for a round to count as productive |

**Extraction capacity**

| Key | Default | Meaning |
|---|---|---|
| `extraction_context_window_tokens` | 131072 | Context window assumed for the extraction LLM. Set it to the server's real window ([Serving](serving.md#context-budgeting)) |
| `extraction_context_margin_tokens` | 1024 | Tokens held back for tokenizer differences |
| `extraction_chunk_size_tokens` | 50000 | Maximum source-text tokens per extraction request; longer records are processed in ordered chunks |
| `extraction_feature_batch_size` | 10 | Variables per extraction request |
| `extraction_max_prompt_chars` | 640000 | Character limit for extraction prompts |

The window must exceed the larger output ceiling plus the margin. With the
defaults the input budget per extraction request is
131072 − 75000 − 1024 = 55048 tokens. Each request contains one patient's text.

**Selection and estimation**

| Key | Default | Meaning |
|---|---|---|
| `min_informative_patients` | 20 | A candidate is kept only if at least this many outer-training patients have an observed value; categorical candidates also need two levels (counting "not documented") with at least this many patients each, and continuous candidates need non-identical values |
| `max_extraction_failure_fraction` | 0.10 | Candidates whose training [extraction failures](glossary.md#extraction-failure) exceed this fraction of patients are dropped |
| `min_propensity`, `max_propensity` | null | Inclusive [propensity eligibility bounds](glossary.md#propensity-eligibility-and-clipping) for effect estimation; either may be set alone. Set them here, not under `statistical_selection` |
| `propensity_clip` | 0.02 | Clipping of AIPW denominators to `[c, 1 − c]`; excludes no one. Under the `orthogonal_tests` modifier rule it also clips the AIPW projection test, so it is part of the selection identity |
| `estimation_trees` | 200 | Trees in the final causal forest and, under `balanced_evidence`, in the forests used for modifier-budget selection (minimum 10). Under `orthogonal_tests` selection does not depend on it, so a change refits only the final estimation |
| `modeling_workers` | 1 | CPU processes shared by numerical selection, modifier testing, and final estimation across all outer folds (see [Parallelism](#parallelism)) |
| `estimand_ontology` | disabled | Outcome-informed search over alternative definitions ([Optional extensions](optional_extensions.md#estimand-informed-ontology-search)) |
| `selection_consolidation` | disabled | Merging of equivalent extracted variables before selection ([Optional extensions](optional_extensions.md#pre-selection-measurement-consolidation)) |
| `statistical_selection` | see below | Variable selection and final-model choice |
| `role_adjudication` | see below | Truncation limits for variable definitions shown in clinical-review prompts |

**Sampling** (see [Sampling profiles](#sampling-profiles))

| Key | Default | Meaning |
|---|---|---|
| `temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`, `frequency_penalty`, `repetition_penalty` | null | Null uses the model's sampling profile. A number overrides the profile for both LLM roles |

### `stage2.statistical_selection`

These settings control the deterministic selection procedure described in
[Stage 2 selection](stage2_selection.md). `selection_mode`, if present, must be
`multi_model`, the only selection mode.

| Key | Default | Meaning |
|---|---|---|
| `l1_ratio` | 0.8 | Elastic-net mixing for nuisance and joint fits |
| `internal_cv_folds` | 3 | Folds for penalty cross-validation and, under `balanced_evidence`, for nested modifier-budget splits (minimum 2) |
| `minimum_log10_alpha`, `maximum_log10_alpha` | −5.0, −1.0 | Bounds of the elastic-net penalty grid |
| `coefficient_tolerance` | 1e-7 | Group coefficient norm above which a variable counts as selected by a nuisance model |
| `optimization_tolerance` | 1e-6 | Solver tolerance |
| `categorical_min_count` | 5 | Categorical levels with fewer patients are pooled; modifier-test indicator columns with fewer patients on either side are dropped |
| `max_iter` | 5000 | Solver iteration limit |
| `modifier_one_standard_error_rule` | `false` | Use the one-standard-error rule when choosing the DINA penalty |
| `modifier_ridge_alpha` | 10.0 | Ridge penalty for single-candidate effect fits |
| `modifier_continuous_winsor_quantile` | 0.005 | Winsorizing quantile for continuous candidates in DINA designs |
| `multi_model` | see below | Evidence-ensemble settings |

`stage2.statistical_selection.multi_model`:

| Key | Default | Meaning |
|---|---|---|
| `repeats` | 3 | Resamples per evidence family; the first uses all rows |
| `row_fraction` | 0.8 | Treatment-stratified subsample fraction for later resamples |
| `l1_ratios` | `[0.2, 0.8, 1.0]` | Elastic-net mixing values cycled across resamples |
| `regularization_grid_size` | 8 | Penalty grid size (minimum 3) |
| `forest_trees` | 800 | Trees in evidence forests |
| `forest_min_samples_leaf` | 10 | Minimum leaf size in evidence forests |
| `feature_subset_size` | 32 | Candidates per random subset for joint evidence families |
| `permutation_repeats` | 10 | Shuffles for permutation importance |
| `nominal_p_threshold` | 0.05 | Nominal p-value support threshold in the evidence families |
| `q_threshold` | 0.1 | [BH q-value](glossary.md#bh-q-value) support threshold in the evidence families and the `balanced_evidence` interaction utility (the modifier tests use `modifier_selection.q`) |
| `interaction_ranking_weight` | 0.5 | Weight of the adjusted-interaction score in modifier ranking (0 to 1) |
| `max_prompt_chars` | 100000 | Character limit for concept-review prompts |
| `diagnostic_families` | `[]` | Optional extra families recorded in `statistical_evidence.json` only: `penalized_main`, `predictive_forest` |
| `matched_batch` | disabled | Matched-batch contrast evidence ([Optional extensions](optional_extensions.md#matched-batch-contrast-evidence)) |
| `stability` | see below | Repeated partitions and grouping of equivalent measurements |
| `modifier_selection` | see below | Modifier rule and final-estimator policy |
| `modifier_count` | see below | Modifier-budget and final-estimator search of the `balanced_evidence` rule |

`multi_model.stability`:

| Key | Default | Meaning |
|---|---|---|
| `partition_repeats` | 3 | The original inner split plus this many minus one additional partitions |
| `subset_partitions` | 4 | Random candidate-subset partitions for joint families |
| `group_equivalent_measurements` | `true` | Group near-identical measurements when counting evidence |
| `group_min_observations`, `group_min_overlap`, `group_min_association`, `group_min_name_similarity` | 30, 0.8, 0.95, 0.8 | Thresholds for treating two measurements as equivalent |

`multi_model.modifier_selection` ([Stage 2 selection](stage2_selection.md#modifier-selection-by-orthogonal-tests)):

| Key | Default | Meaning |
|---|---|---|
| `rule` | `orthogonal_tests` | `orthogonal_tests`: Benjamini–Hochberg selection on Cauchy-combined component tests. `balanced_evidence`: the nested ranking and budget search configured by `modifier_count` |
| `q` | 0.1 | Benjamini–Hochberg level for selection and for the outer-fold confirmation, in (0, 1] |
| `weights` | null | Component weights for the Cauchy combination: null, or an object with exactly the keys `r_projection`, `aipw_projection`, and `outcome_interaction`, each finite and nonnegative, with a positive sum. Null uses 0.25, 0.25, 0.5 for binary outcomes and 1/3 each for continuous outcomes. Also used by the confirmation |
| `max_modifiers` | 64 | Maximum number of discovered modifiers; null for no cap. Investigator-specified modifiers are added on top |
| `estimator_policy` | null | `orthogonal_tests` only: null or `ensemble` averages every final estimator for the outcome type; `causal_forest`, `linear_interactions`, or `dina` (binary outcomes only) fits one estimator. Must be null under `balanced_evidence`, which chooses its estimator by nested validation |

Unknown keys are errors. `modifier_count.concept_review: true` requires
`rule: "balanced_evidence"`. The rule and its settings are part of the
selection identity, so changing them after selection has run requires
[guarded reselection](#guarded-reselection); the multi-model evidence itself is
reused.

`multi_model.modifier_count` (used only when `modifier_selection.rule` is
`balanced_evidence`):

| Key | Default | Meaning |
|---|---|---|
| `candidate_counts` | `[0, 4, 8, 12, 16, 24, 32]` | [Modifier budgets](glossary.md#modifier-budget) to compare; sorted, starting at 0. The full ranked shortlist is always compared as well |
| `max_ranked_modifiers` | 64 | Cap on the ranked shortlist |
| `selection_rule` | `balanced_evidence` | The only count rule |
| `relative_r_loss_tolerance` | 1e-4 | Relative R-loss tolerance in the selection rule |
| `forest_seeds` | 3 | Causal-forest seeds averaged per evaluation |
| `estimators` | `["causal_forest", "linear_interactions", "dina"]` | Final effect models to compare; `dina` is dropped for continuous outcomes |
| `concept_review` | `false` | LLM concept review of modifier shortlists ([Optional extensions](optional_extensions.md#modifier-concept-review)); requires `modifier_selection.rule: "balanced_evidence"` |
| `concept_top_n_per_fold` | 100 | Top-ranked candidates per evidence fold given to concept review |

### `stage2.role_adjudication`

| Key | Default | Meaning |
|---|---|---|
| `max_description_chars` | 1200 | Description characters per variable in review prompts |
| `max_measurement_definition_chars` | 1600 | Measurement-rule characters per variable |
| `max_categories_per_feature` | 40 | Categories shown per variable |

Roles are assigned by the statistical selection procedure; these limits apply
to the variable summaries in the concept-review prompt. They are part of the
selection checkpoint identity, so changing them on a completed run requires
[reselection](#guarded-reselection).

### Explicit features

`stage2.explicit_features` lists variables the investigator wants measured,
either as a list or as `{"enabled": true, "features": [...]}`. Each entry is a
complete extraction definition:

| Field | Meaning |
|---|---|
| `name` | Variable name (normalized to lowercase snake case) |
| `description` | What the variable means |
| `value_type` | `binary`, `categorical`, `ordinal`, `continuous`, or `ambiguous` |
| `categories_or_unit` | Allowed categories, or the unit for a continuous variable (aliases: `categories`, `unit`) |
| `measurement_definition` | How to measure it from the text |
| `missing_value_rule` | When to record it as missing |
| `roles` | `confounder`, `effect_modifier`, or both (`both` is accepted) |
| `conflict_resolution` | Optional rule for repeated or conflicting mentions |
| `stability_summary`, `caveats` | Optional notes |

Explicit features are extracted like discovered candidates, but their
definitions are never revised by the review steps and their roles are fixed:
a `confounder` enters the nuisance models and an `effect_modifier` enters the
effect model regardless of the selection evidence. Explicit modifiers are
always included: they stay outside the Benjamini–Hochberg family of the
orthogonal tests and do not count toward the `balanced_evidence` modifier
budget.

### Stage 1 template settings

Stage 1 model settings live in the template file and are overridden through
`science.stage1`, for example
`--set science.stage1.architecture.multi_model_forest.embedding_contrast.max_chunks=1024`.
Commonly changed keys, relative to `science.stage1.architecture`:

| Key | Template value | Meaning |
|---|---|---|
| `htr_max_chunks` | 512 | Maximum HTR chunks per record |
| `multi_model_forest.embedding_contrast.max_chunks` | 512 | Maximum embedding chunks per record |
| `multi_model_forest.embedding_contrast.retrieval_tfidf_enabled` | `false` | Add `tfidf_semantic_retrieval_contrasts` to the default selection |
| `multi_model_forest.tfidf_topic.orphan_ngram_enabled` | `false` | Add `tfidf_orphan_ngrams` to the default selection |
| `multi_model_forest.embedding_contrast.include_cluster_contrast_vectors` | `true` | Produce `embedding_clustered` |
| `multi_model_forest.bow_discovery_enabled`, `htr_evidence_enabled`, `matched_pair_uplift_enabled` | `true` | Enable the BoW, HTR, and matched-pair architectures |

Chunk-based models stop with a capacity error rather than truncating a long
record; raise the relevant `max_chunks` value when that happens. The workflow
fills in the dataset, columns, folds, seed, devices, HTR and embedding models,
and the embedding cache location itself. The resolved templates are written to
`resolved_stage1_model_config.json` and `resolved_neural_query_config.json`.

## Command-line flags

`python scripts/run_all_evidence.py [FLAGS]`. Flags override the configuration
file; `--set` can change any other value.

**Inputs and study design**

| Flag | Configuration key | Meaning |
|---|---|---|
| `--config PATH` | — | JSON or YAML configuration, or a saved `OUTPUT_DIR/run_config.json` |
| `--dataset PATH` | `dataset` | Input Parquet or CSV dataset |
| `--output-dir PATH` | `output_dir` | Run directory and resume checkpoint |
| `--unit-id-column NAME` | `columns.unit_id` | Patient identifier column (default `patient_id`) |
| `--text-column NAME` | `columns.text` | Clinical text column (default `clinical_text`) |
| `--treatment-column NAME` | `columns.treatment` | Binary treatment column (default `treatment_indicator`) |
| `--outcome-column NAME` | `columns.outcome` | Outcome column (default `outcome_indicator`) |
| `--outcome-type {binary,continuous}` | `science.outcome_type` | Outcome scale (default `binary`) |
| `--clinical-question TEXT` | `science.clinical_question` | Study question given to the language models |
| `--outer-folds N` | `science.outer_folds` | Outer folds (default 5) |
| `--inner-folds N` | `science.inner_folds` | Inner folds (default 5) |
| `--seed N` | `science.seed` | Seed (default 42) |

**Stage 1**

| Flag | Configuration key | Meaning |
|---|---|---|
| `--devices LIST` | `run.devices` | Comma-separated Stage 1 devices, such as `cuda:0,cuda:1` (default `cpu`) |
| `--workers N` | `run.workers` | Stage 1 CPU budget; also the default Stage 2 request concurrency (default 1) |
| `--architectures LIST` | `science.stage1_architectures` | Comma-separated Stage 1 architectures, or `all` for all ten (binary outcomes only). Frozen on resume |
| `--htr-model ID` | `models.htr` | HTR chunk encoder |
| `--disable-htr` | `science.htr_enabled=false` | Disable all HTR modeling, keeping the other evidence families |
| `--embedding-model ID` | `models.embeddings` | Embedding model for the embedding and neural-query architectures |

**Run mode** (`--stage1-only` and `--stage2-only` are mutually exclusive)

| Flag | Meaning |
|---|---|
| `--stage1-only` | Run or resume Stage 1 through the handoff, then stop |
| `--stage2-only` | Skip Stage 1 and run or resume Stage 2 from the saved handoff, usually with `--config OUTPUT_DIR/run_config.json` |
| `--stage2-reselect` | With `--stage2-only`: archive a completed Stage 2 selection and everything after it, then rerun selection from the frozen definitions and outer-training extractions ([Guarded reselection](#guarded-reselection)) |
| `--preflight-only` | With `--stage2-only`: validate the saved run and the Stage 2 reuse plan without writing files, starting servers, or fitting models; prints a JSON report |
| `--rerun COMPONENT` | Remove a component's completion markers (not its outputs) so it runs again; repeatable. One of `embedding_cache`, `tfidf`, `text_models`, `neural_queries`, `handoff`, `stage2` |
| `--status` | Print `progress.json` and exit |

**Stage 2 serving** (the two endpoint flags are mutually exclusive)

| Flag | Meaning |
|---|---|
| `--stage2-extraction-endpoint URL` | Extraction LLM base URL; replaces any `stage2.extraction_llm.endpoints` list from `--config` |
| `--stage2-extraction-endpoints JSON` | JSON array of prestarted extraction servers, each `{"endpoint": URL, "max_concurrency": N}`; replaces any `stage2.extraction_llm.endpoint` from `--config` |
| `--stage2-vllm-extra-arg TOKEN` | One extra vLLM command-line token for the managed interpretation pool; repeat for each token, and write `--stage2-vllm-extra-arg=--flag` for tokens that start with `--`. Replaces `stage2.vllm.extra_args` |
| `--stage2-extraction-vllm-extra-arg TOKEN` | The same for the managed extraction pool (`stage2.extraction_llm.vllm.extra_args`) |

**Overrides**

| Flag | Meaning |
|---|---|
| `--set KEY=VALUE` | Override any configuration value; repeatable and applied in order |

Every other Stage 2 setting is passed with `--set`, for example
`--set stage2.endpoint=http://127.0.0.1:8010/v1 --set stage2.workers=16`.

`--status` still needs the run's identity, so pass the saved configuration:
`python scripts/run_all_evidence.py --config OUTPUT_DIR/run_config.json --status`.

## The synthetic launcher (`run_synthetic.sh`)

`run_synthetic.sh` runs or resumes the full workflow on one of the two bundled
synthetic cohorts ([Synthetic data](synthetic_data.md#bundled-cohorts)):

```bash
[PRESET=NAME] [VARIABLE=VALUE ...] ./run_synthetic.sh COHORT [OUTPUT_DIR] [WORKFLOW_OPTION ...]
```

| Argument | Meaning |
|---|---|
| `COHORT` | `one_conf_one_mod` (one confounder, one effect modifier) or `five_conf_five_mod` (five of each) |
| `OUTPUT_DIR` | Default `artifacts/research_all_evidence/COHORT_nsclc_full` for the `external` preset and `COHORT_nsclc_PRESET_full` otherwise, with `_no_htr` appended when `DISABLE_HTR=1`. Rerun the same command to resume |
| `WORKFLOW_OPTION` | Passed to the workflow after the launcher's own options, so a later `--set` wins, for example `--set stage2.consolidation_max_rounds=12` |

The launcher synchronizes `.venv` from the lockfile with `uv` (adding the
`local-llm` extra when a managed vLLM pool is configured), detects GPUs, sets
`OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and
`NUMEXPR_NUM_THREADS` to 1 unless they are already set, and starts the
workflow. GPU detection needs at least one CUDA GPU visible to PyTorch. It
passes the bundled cohort's columns, a fixed clinical question, 5 × 5 folds,
seed 42, the default HTR and embedding models, and these Stage 2 definition
settings: `consolidation_batch_size=20`, `consolidation_max_rounds=55`,
`operationalization_max_prompt_chars=640000`,
`ontology_refinement_min_failure_patients=3`, and
`max_ontology_refinement_rounds=2`.

### Presets

`PRESET` selects serving and hardware defaults from `scripts/presets/NAME.env`.
Presets apply to every launch except Stage 2-only relaunches
(`STAGE2_ONLY=1`), and any variable you set explicitly keeps its value. Models, GPU layout, and ports for each preset are listed in
[Serving](serving.md#hardware-presets).

| Preset | Serving | Defaults it sets |
|---|---|---|
| `external` (default) | Prestarted OpenAI-compatible servers: interpretation at `http://127.0.0.1:8010/v1`, extraction at `http://127.0.0.1:8020/v1` | Model IDs only |
| `h100x8` | Managed vLLM on eight 80 GB H100 GPUs; the two models alternate across all GPUs | 32 workers per role, 1800 s attempt timeout, 6000 s request timeout, 128,000-token extraction window with a 4,096-token margin |
| `rtxpro6000x8` | Managed vLLM on eight 96 GB RTX PRO 6000 GPUs; one model serves both roles from one pool | As `h100x8`, plus `STAGE2_MODELING_WORKERS=128` and a 262,144-token extraction window |

Except in Stage 2-only relaunches, the `one_conf_one_mod` cohort also defaults to
`STAGE2_WORKERS=4`, `STAGE2_EXTRACTION_WORKERS=4`,
`STAGE2_REQUEST_ATTEMPT_TIMEOUT=1800`, `STAGE2_REQUEST_TIMEOUT=6000`,
`STAGE2_EXTRACTION_CONTEXT_WINDOW_TOKENS=128000`, and
`STAGE2_EXTRACTION_CONTEXT_MARGIN_TOKENS=4096`, sized for one external
reasoning server with a 128K-token window. These apply after the preset and
before explicit variables. `five_conf_five_mod` has no cohort defaults.

### Environment variables

Explicit values override preset and cohort defaults. The launcher rejects any
other `STAGE2_*` variable; pass other Stage 2 settings as
`--set stage2.KEY=VALUE`.

| Variable | Meaning |
|---|---|
| `PRESET` | Preset name (default `external`) |
| `GPU_COUNT` | Number of eligible Stage 1 GPUs to select (default: all eligible) |
| `PHYSICAL_GPUS` | Explicit host GPU indices; exported as `CUDA_VISIBLE_DEVICES` |
| `MIN_FREE_GPU_GB` | Free memory a GPU needs to be eligible (default 0) |
| `STAGE1_WORKERS` | Stage 1 CPU workers (default: sized from CPUs, folds, and GPU memory) |
| `STAGE1_ARCHITECTURES` | Comma-separated Stage 1 architectures, or `all` (default: the template's eight) |
| `DISABLE_HTR=1` | Skip all HTR modeling |
| `STAGE2_ENDPOINT` | Interpretation LLM URL; set it empty to run Stage 1 only |
| `STAGE2_MODEL`, `STAGE2_EXTRACTION_MODEL` | Served model IDs |
| `STAGE2_EXTRACTION_ENDPOINT` | Extraction LLM URL |
| `STAGE2_EXTRACTION_ENDPOINTS` | JSON array of `{"endpoint": URL, "max_concurrency": N}` |
| `STAGE2_VLLM_SERVERS`, `STAGE2_VLLM_GPUS_PER_SERVER` | Either one makes OCI manage the interpretation pool; `STAGE2_VLLM_SERVERS=0` turns management off |
| `STAGE2_VLLM_GPUS`, `STAGE2_VLLM_BASE_PORT`, `STAGE2_VLLM_INTERNAL_PORT_BASE`, `STAGE2_VLLM_DOWNLOAD_DIR`, `STAGE2_VLLM_EXTRA_ARGS_JSON` | Managed interpretation-pool settings; the last is a JSON array of vLLM tokens |
| `STAGE2_VLLM_RAPID_SWITCH_SECONDS` | `stage2.vllm_rapid_switch_seconds` |
| `STAGE2_EXTRACTION_VLLM_*` | The same pool settings for the extraction model (`SERVERS`, `GPUS`, `GPUS_PER_SERVER`, `BASE_PORT`, `INTERNAL_PORT_BASE`, `DOWNLOAD_DIR`, `EXTRA_ARGS_JSON`) |
| `STAGE2_WORKERS`, `STAGE2_EXTRACTION_WORKERS` | Request concurrency per role (default 32 each when no preset or cohort default applies) |
| `STAGE2_MODELING_WORKERS` | `stage2.modeling_workers` |
| `STAGE2_REQUEST_TIMEOUT`, `STAGE2_REQUEST_ATTEMPT_TIMEOUT` | Seconds |
| `STAGE2_EXTRACTION_CONTEXT_WINDOW_TOKENS`, `STAGE2_EXTRACTION_CONTEXT_MARGIN_TOKENS` | Extraction context budget (not used by Stage 2-only relaunches) |
| `OCI_PYTHON` | Use this interpreter instead of synchronizing `.venv` |

### Relaunching Stage 2 of a saved run

```bash
STAGE2_ONLY=1 ./run_synthetic.sh COHORT [OUTPUT_DIR]
STAGE2_ONLY=1 OCI_RUN_CONFIG=/path/to/run_config.json ./run_synthetic.sh COHORT
```

With `STAGE2_ONLY=1` the saved `run_config.json` supplies every setting;
presets, cohort defaults, and the launcher's Stage 2 definition settings are
not applied, and dependencies are not synchronized (`OCI_PYTHON` or the
existing `.venv` is used). A read-only preflight always runs first.

| Variable | Meaning |
|---|---|
| `STAGE2_ONLY=1` | Relaunch Stage 2 from the saved handoff |
| `OCI_RUN_CONFIG` | Path to the saved configuration (instead of `OUTPUT_DIR/run_config.json`); do not also pass `OUTPUT_DIR` |
| `OCI_PREFLIGHT_ONLY=1` | Stop after the preflight report |
| `STAGE2_RESELECT=1` | Request [guarded reselection](#guarded-reselection) |

Only the serving and concurrency variables apply in this mode, not the
extraction context budgets. `STAGE2_RESELECT`, `OCI_PREFLIGHT_ONLY`, and
`OCI_RUN_CONFIG` require `STAGE2_ONLY=1`.

## Request and recovery defaults

| Behavior | Default |
|---|---|
| Logical request budget, including transport retries and repairs | 14,400 s (`request_timeout`) |
| Timeout per HTTP attempt | 3,600 s (`request_attempt_timeout`) |
| HTTP attempts per response attempt | 6 (`transport_max_attempts`); retries wait 2, 4, 8, 16, 32 s |
| Validator-guided repairs of an invalid response | 15 (`max_response_repairs`; the example configuration uses 10) |
| Repairs that keep the request's normal reasoning setting | the first 5 (`thinking_after_response_repairs`); later repairs use at least `high` |
| Repairs of one extraction field before it is retried in a smaller request | 3 |
| Deferred extraction retry passes | 1 (`extraction_deferred_retry_passes`) |
| Outer-fold recoveries after a request exhausts its budget | 2, after 60 s and then 120 s (`outer_fold_recovery_attempts`, `outer_fold_recovery_backoff`) |

Retryable transport failures are connection errors, timeouts, HTTP 408, 409,
429, and 5xx responses, and empty responses; transport retries resend the same
request. A response that fails JSON parsing or validation, stops at the output
limit, or contains only reasoning is repaired: the next turn includes the
previous answer and the validation error. Time spent waiting for a local
request slot does not count against the request budget; time queued on the
server does.

When a request exhausts its budget, its outer fold stops, waits, and resumes
from its checkpoints while other folds continue; `recovery_status.json` and
`recovery_events.jsonl` in the fold directory record each attempt. Extraction
cells that still fail validation after every repair are stored as missing and
flagged as [extraction failures](glossary.md#extraction-failure), and the
failure rates are reported per fold. Transport failures never become missing
values.

## Sampling profiles

After resolving each served model through the endpoint's `/models` list,
Stage 2 looks up a checked-in sampling profile for the model family (using the
backing model ID when the server advertises one, so an alias still resolves).
The interpretation and extraction LLMs resolve separately. The Gemma 4 and
Qwen 3.8 Flash Next profiles follow the publishers' recommendations as
verified on 2026-09-23; the Qwen 3 and LFM 2.5 profiles on 2026-09-12.

| Model family | Mode | Temperature | Top-p | Top-k | Presence penalty | Repetition penalty |
|---|---|---|---|---|---|---|
| Gemma 4 | either | 1.0 | 0.95 | 64 | 0 | 1.0 |
| Qwen 3 | thinking | 0.6 | 0.95 | 20 | 0 | 1.0 |
| Qwen 3 | non-thinking | 0.7 | 0.8 | 20 | 0 | 1.0 |
| Qwen 3.5 | thinking | 1.0 | 0.95 | 20 | 1.5 | 1.0 |
| Qwen 3.6 and 3.8 (including Flash Next) | thinking | 1.0 | 0.95 | 20 | 0 | 1.0 |
| Qwen 3.5, 3.6, and 3.8 | non-thinking | 0.7 | 0.8 | 20 | 1.5 | 1.0 |
| LFM 2.5 2.6B | either | 0.1 | 1.0 | 50 | 0 | 1.1 |
| LFM 2.5 1.2B | either | 0.1 | 1.0 | 50 | 0 | 1.05 |
| Any other model | — | server defaults | | | | |

Sources: [Gemma 4](https://huggingface.co/google/gemma-4-31B-it#best-practices),
[Qwen 3](https://huggingface.co/Qwen/Qwen3-32B#best-practices),
[Qwen 3.5](https://huggingface.co/Qwen/Qwen3.5-27B#best-practices),
[Qwen 3.6](https://huggingface.co/Qwen/Qwen3.6-27B#best-practices),
[Qwen 3.8](https://huggingface.co/Qwen/Qwen3.8-27B#best-practices),
[Qwen 3.8 Flash Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next#best-practices),
[LFM 2.5 2.6B](https://huggingface.co/LiquidAI/LFM2.5-2.6B),
[LFM 2.5 1.2B](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct).

- Recognized profiles also send frequency penalty 0 and min-p 0, so a server's
  `generation_config.json` cannot add a different penalty. Where a publisher
  gives no value, these neutral values are the pipeline's choice.
- Unrecognized model families receive no sampling fields and use the server's
  defaults.
- A non-null `stage2.temperature`, `top_p`, `top_k`, `min_p`,
  `presence_penalty`, `frequency_penalty`, or `repetition_penalty` overrides
  the profile for both roles, including an explicit temperature of 0.
- Thinking-dependent values are recomputed when a repair turns thinking on.
- `top_k`, `min_p`, and `repetition_penalty` are sent in `extra_body`. If a
  server rejects them, the request is retried without them and the server's
  defaults apply.
- Reasoning: `interpretation_reasoning_effort: auto` resolves to `xhigh` for
  Qwen 3.8 Flash Next and `high` for every other model.
  `extraction_reasoning_effort: none` sends initial extraction requests with
  thinking off and the non-thinking profile; `auto` for extraction resolves to
  `none`, except `xhigh` for Flash Next.
- The resolved interpretation request policy (model, reasoning, output
  ceiling, and sampling) is part of the feature-definition checkpoint identity,
  so changing sampling settings on a saved run invalidates its completed
  definitions. `stage2/model_identity.json` records the resolved policies and
  profile provenance, and extraction request events record the controls sent
  on each attempt.

## Parallelism

Top-level components run in order, because later components reuse the
artifacts and fold definitions of earlier ones. Parallelism happens inside
components.

| Setting | Controls |
|---|---|
| `run.devices` | Stage 1 GPU parallelism. Each CUDA device gets one worker process that keeps that device, and whole discovery contexts (one outer fold, or one inner fold within it) are assigned to these processes, largest first, to the least-loaded one. The embedding cache divides its chunks across the devices |
| `run.workers` | Stage 1 CPU budget. TF-IDF runs up to this many contexts as separate processes with one numerical thread each. During `text_models` the budget is divided as evenly as possible across the active GPU processes, with at least one worker each |
| `stage2.workers` | Concurrent interpretation requests across all outer folds |
| `stage2.extraction_llm.workers` | Concurrent extraction requests across all outer folds |
| `stage2.modeling_workers` | One shared pool of CPU processes for numerical selection evidence, modifier-test nuisance fits, modifier-budget evaluations, and final nuisance and interaction-model fits across all outer folds; 1 runs these fits serially within each fold. Final causal forests use a share of the same budget as threads |

Outer folds run concurrently in Stage 2; review rounds within a fold run in
order. Changing any worker count preserves compatible checkpoints, seeds, and
validation partitions.

Set `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and
`NUMEXPR_NUM_THREADS` to 1 when starting Python directly on machines with many
cores (`run_synthetic.sh` does this). Letting each concurrent fit create its
own native thread pool can exhaust OpenBLAS thread memory and terminate the
process.

## Output directory

The output directory is both the result location and the resume checkpoint.
The tree below shows a full run with an explicit architecture selection; each
outer fold `k` appears as `outer_00k`. Conditions are noted in brackets.

```text
OUTPUT_DIR/
├── run_config.json                      resolved settings (keys redacted); reusable with --config
├── resolved_stage1_model_config.json    resolved Stage 1 template
├── resolved_neural_query_config.json    resolved neural-query template
├── progress.json                        run and component status (--status)
├── logs/workflow.log                    full log, including Stage 2 request events
├── stage1_architectures/                [explicit selection] manifest.json and one evidence.jsonl per architecture
├── components/
│   ├── embedding_cache/                 chunk embedding cache
│   ├── tfidf/                           evidence.jsonl, split_provenance.jsonl (fold plan shared by all stages), contexts
│   ├── text_models/                     evidence.jsonl, one directory per discovery context with checkpoints
│   └── neural_queries/                  evidence.jsonl, one directory per discovery context
├── handoff/                             evidence.jsonl (the handoff), index.json, complete.json
└── stage2/
    ├── config.json                      Stage 2 settings
    ├── model_identity.json              served model identities and request policies; checked on resume
    ├── evidence_compilation/            evidence cards per outer fold
    ├── vllm_servers/                    [managed vLLM] pool manifests, server logs, model_phase.json
    ├── outer_00k/
    │   ├── input_packets.jsonl          this fold's evidence cards
    │   ├── interpretations/             one checkpointed interpretation per card
    │   ├── interpreted_candidates.json  candidate variables proposed from the cards
    │   ├── consolidation/               duplicate merging and extraction-definition writing
    │   ├── feature_definitions.json     frozen candidate definitions
    │   ├── ontology_supervision/        extraction review rounds, definition refinement, convergence.json
    │   ├── estimand_ontology/           [estimand_ontology.enabled]
    │   ├── extraction/                  outer-training and held-out extractions, failure masks and reports
    │   ├── selection/                   consolidation, multi-model evidence, modifier tests or budget search, selected definitions
    │   ├── final_definitions.json       final variables, roles, modifier evidence, and final estimator
    │   ├── estimation/                  nuisance fits and audits, overlap, predictions.csv, diagnostics.json
    │   ├── preselection/                [after reselection] frozen selection inputs
    │   ├── recovery_status.json, recovery_events.jsonl
    │   └── complete.json
    ├── features_by_outer_fold.jsonl     per-fold variables, roles, and estimation summary
    ├── cross_fitted_predictions.csv     one held-out prediction per patient
    ├── causal_estimate.json             AIPW ATE with 95% CI, overlap, calibration, heterogeneity summary
    ├── heterogeneity_diagnostics.json   BLP, GATES, CLAN, RATE, overlap-weighted ATE, trimming curve, balance, modifier confirmation
    ├── posthoc_oracle_ite_metrics.json  oracle comparison (records "unavailable" without oracle columns)
    ├── posthoc_predictions_with_oracle_ite.csv  [oracle columns present]
    ├── summary.json
    ├── reselection_state.json, reselection_archives/   [after reselection]
    └── complete.json
```

Without an explicit architecture selection, `stage1_architectures/` is not
written and the handoff streams the component evidence directly.
`OUTPUT_DIR/evaluations/stage1/` is written only by `oci-evaluate-stage1`
([Synthetic data](synthetic_data.md#oracle-safe-evaluation)).

### Files to read first

| File | Content |
|---|---|
| `stage2/causal_estimate.json` | Cross-fitted AIPW ATE, standard error, and 95% CI; overlap and nuisance calibration summaries; mean estimated CATE; oracle correlations when available |
| `stage2/cross_fitted_predictions.csv` | One row per patient from its outer held-out fold: `estimated_cate` (a risk difference for binary outcomes), `aipw_score`, `propensity`, `mu0`, `mu1`, `effect_eligible`, `outer_fold`, plus per-estimator CATE columns under the ensemble (see [Modifier-selection and ensemble outputs](#modifier-selection-and-ensemble-outputs)) |
| `stage2/heterogeneity_diagnostics.json` | Whether the CATE predictions track real heterogeneity ([BLP, GATES, CLAN, RATE](glossary.md#blp-gates-clan-and-rate)); see [Stage 2 selection](stage2_selection.md) |
| `stage2/features_by_outer_fold.jsonl` and `stage2/outer_00k/final_definitions.json` | Which variables were used in each fold, their roles and definitions, and the chosen estimator |
| `stage2/outer_00k/selection/modifier_tests.json` | Every candidate's modifier tests and the selection (`orthogonal_tests`) |
| `stage2/outer_00k/selection/modifier_count/selection.json` | Modifier selection report: selected modifiers and estimator policy, or the `balanced_evidence` budget and estimator comparison |
| `stage2/outer_00k/extraction/failure_report_fit.json`, `failure_report_heldout.json` | Extraction failure rates by variable and treatment arm |
| `stage2/outer_00k/estimation/diagnostics.json`, `propensity_overlap.json`, `nuisance_fit_audits/` | Fold-level ATE, calibration, overlap, and the fitted penalty and iteration count of each nuisance model |

### Modifier-selection and ensemble outputs

`stage2/outer_00k/selection/modifier_tests.json` (rule `orthogonal_tests`)
records `q_threshold`, the resolved `weights` and `weights_source`,
`max_modifiers`, `propensity_clip`, the propensity bounds,
`sparse_indicator_min_count`, the testing-nuisance source and recalibration
audit, `summary` counts, `selected_feature_ids`, `discovered_feature_ids`,
`locked_modifier_ids`, an `interpretation` string per field, and one record per
candidate in `candidates`:

| Field | Meaning |
|---|---|
| `feature_id`, `name` | The candidate |
| `n_rows`, `n_eligible` | Outer-training patients and the propensity-eligible patients tested |
| `grouped_columns`, `dropped_sparse_columns` | Tested design columns and dropped rare indicator columns |
| `components` | Per component (`r_projection`, `aipw_projection`, `outcome_interaction`): `p_value`, `statistic`, `degrees_of_freedom`, `q_value`, `passes`, and `status`/`reason` when not evaluable |
| `combined_p_value`, `combined_statistic`, `combination_weights` | Cauchy combination and the weights renormalized over the available components |
| `combined_q_value`, `bh_family` | Benjamini–Hochberg q-value; `bh_family` is false for investigator-locked variables, which get no q-value |
| `selected`, `selection_rank`, `selection_reason`, `excluded_by_cap` | Selection outcome |
| `labels`, `extreme_propensity_signal` | Scale labels of a selected modifier and the concordance flag |
| `nuisance_source`, `q_threshold`, `investigator_locked` | Provenance |

`selection/modifier_count/<fingerprint>/testing_nuisances.csv` holds each
training patient's testing nuisances (`propensity`, `mu0`, `mu1`,
`marginal_outcome`, `inner_fold`, `effect_eligible`); they equal the training
columns of `estimation/nuisance_fit_predictions.csv`. The inner-fold models
behind them are stored in `selection/nuisance_models/<fingerprint>/` (joblib
files and a `manifest.json` with file hashes and the input, code, and library
fingerprint); the final estimation reuses verified models for the held-out
patients and refits any that fail verification, recording
`nuisance_model_store` and `model_sources` in `estimation/diagnostics.json`.
`selection/statistical_evidence.json` lists the effect evidence the rule does
not compute under `skipped_effect_evidence`. Each effect modifier in
`final_definitions.json` carries a `modifier_evidence` record with its rule,
combined and component p- and q-values, labels, flag, rank, and lock status.

Under the estimator ensemble, `estimation/predictions.csv` and
`cross_fitted_predictions.csv` add `estimated_cate_causal_forest`,
`estimated_cate_linear_interactions`, `estimated_cate_dina` (when DINA was
fitted), and the forest's interval in `estimated_cate_causal_forest_lower_95`,
`estimated_cate_causal_forest_upper_95`, and
`estimated_cate_causal_forest_standard_error`. `estimated_cate` is the
unweighted mean of the fitted components, and `estimated_cate_lower_95`,
`estimated_cate_upper_95`, and `estimated_cate_standard_error` are empty.
`estimation/diagnostics.json` records `estimator_policy`, an `ensemble` section
(component and fitted estimators, unavailable components with reasons, and each
component's audit), and `mean_estimated_cate_by_estimator`.

`stage2/heterogeneity_diagnostics.json` contains `modifier_confirmation`: the
confirmation policy, per-fold results for each selected modifier (`confirmed`,
held-out combined and component p- and q-values, labels), pooled counts of
selected and confirmed modifiers, and the cross-fold name summary with Stouffer
combinations. `stage2/causal_estimate.json` repeats a compact summary under
`heterogeneity_diagnostics.modifier_confirmation`.

## Checkpoints and resuming

Rerunning the same command resumes a run. Each component writes
`complete.json` in its directory when it finishes, and completed components are
skipped. Inside components, every Stage 1 discovery context and every expensive
Stage 2 step (each interpretation, merging batch, extraction batch, review
round, selection evidence cell, and estimation fit) is checkpointed, so an
interrupted component re-enters at its first incomplete step.

- `--status` prints `progress.json`, which names the current component and
  any error.
- `--rerun COMPONENT` removes that component's completion markers so it runs
  again; outputs stay in place. For `text_models`, fitted sub-steps whose
  inputs are unchanged are reused from each context's `checkpoints/`. For
  `tfidf`, the per-context markers are also removed, so every context is
  refit.
- Stage 1 completion markers do not detect every changed scientific setting.
  Use a different output directory for a scientifically different run. When a
  Stage 1 text producer changes, rerun `text_models` and `handoff`, and move
  the previous `stage2/` directory aside before running Stage 2 again; Stage 2
  rejects feature definitions whose evidence fingerprint does not match the handoff.

Stage 2 checks its saved state on every start:

| Change on a saved run | Result |
|---|---|
| Different interpretation model ID | Refused; use a different output directory |
| Different extraction model ID | Allowed only when no extraction-dependent checkpoints exist |
| Endpoint URLs, ports, worker counts, timeouts | Allowed |
| Interpretation sampling, reasoning, or `max_tokens` | Completed feature definitions do not match the requested policy; the run stops |
| Extraction context window, margin, chunk size, or output ceilings | Affected extraction chunks are extracted again |
| `min_propensity`, `max_propensity`, `statistical_selection`, or, after selection has run, `propensity_clip` (under `orthogonal_tests`) or `estimation_trees` (under `balanced_evidence`) | Refused; use [guarded reselection](#guarded-reselection) |
| `estimation_trees` under `orthogonal_tests` | Allowed; selection and its stored nuisance models are reused and only the final estimation is refitted |

`run_config.json` stores every API key as `<redacted>`. When relaunching from it
against servers that require a key, set the `OCI_STAGE2_*_API_KEY` environment
variables (the placeholder counts as no key) or pass the keys, for example
`--set stage2.api_key=KEY --set stage2.extraction_llm.api_key=KEY`. Do not run
two workflow processes against the same output directory.

## Stage 2-only launches

To run Stage 2 against a saved run's frozen Stage 1 handoff, without rerunning
Stage 1:

```bash
python scripts/run_all_evidence.py \
  --config OUTPUT_DIR/run_config.json \
  --stage2-only \
  --set stage2.endpoint=http://127.0.0.1:8010/v1 \
  --stage2-extraction-endpoint http://127.0.0.1:8020/v1
```

Stage 2 requires the frozen split file
`components/tfidf/split_provenance.jsonl` and never recreates folds, because a
different split could place patients that Stage 1 trained on into Stage 2
held-out sets. A Stage 2-only launch does not rewrite `run_config.json`.

### Preflight

`--stage2-only --preflight-only` validates a saved run without writing files,
contacting endpoints, starting servers, or fitting models, and prints a JSON
report. It checks that:

- the saved Stage 1 settings match the requested ones;
- the handoff and every Stage 1 component are complete and their artifacts
  exist;
- the dataset's content and row order match Stage 1, and unit IDs are unique;
- the saved outer and inner splits partition the data exactly and agree with
  the TF-IDF evidence;
- the embedding cache matches the dataset;
- saved selection inputs and frozen snapshots agree with the splits;
- the saved model IDs match the requested ones;
- the reuse plan is valid (with `--stage2-reselect`, the reselection
  compatibility checks run without moving anything).

The report states whether Stage 2 will start, resume, or reselect.

### Guarded reselection

`--stage2-only --stage2-reselect` reruns variable selection and estimation of a
completed Stage 2 run without repeating interpretation, extraction review, or
outer-training extraction. Use it after changing selection settings such as
`statistical_selection`, the propensity bounds, `estimation_trees` under
`balanced_evidence`, or `propensity_clip` under `orthogonal_tests`.

```bash
python scripts/run_all_evidence.py \
  --config OUTPUT_DIR/run_config.json \
  --stage2-only --stage2-reselect \
  --set stage2.min_propensity=0.1 --set stage2.max_propensity=0.9
```

1. Every outer fold is verified before anything moves: the completed
   selection inputs, definitions, training matrix, splits, and treatment and
   outcome data must match their recorded fingerprints. The interpretation and
   extraction model IDs must equal the completed run's; endpoints and worker
   counts may change.
2. Each fold's verified post-review definitions and training matrix are frozen
   under `outer_00k/preselection/`, together with a manifest of the held-out
   measurements already extracted.
3. Selection and everything after it (per fold: `selection/`,
   `final_definitions.json`, `estimation/`, `extraction/fit`,
   `extraction/heldout`, `extraction/extracted_features.csv`, the health files,
   and `ontology_supervision/convergence.json`; at the Stage 2 root: the
   predictions, estimates, diagnostics, and summaries) are moved to
   `stage2/reselection_archives/reselection_TIMESTAMP/`.
4. Stage 2 then reruns selection and estimation. Held-out measurements whose
   definitions are unchanged are reused; only missing or changed ones are
   extracted.

`stage2/reselection_state.json` records progress. An interrupted preparation
resumes when the same command is repeated; a reselection prepared under
different settings must be finished with those settings or its archive
restored first. After a reselection completes, an ordinary resume does not
repeat it.

## Stage 2 in a separate directory from preserved Stage 1 artifacts

`scripts/run_stage2_from_artifacts.py` starts Stage 2 in a separate directory
from a preserved handoff and split file, for example to compare Stage 2
settings on the same Stage 1 evidence. It uses its own flat configuration
schema, and its `output_dir` is the Stage 2 directory itself rather than a
workflow output root.

```json
{
  "dataset": "/data/cohort.parquet",
  "handoff_path": "/results/source_run/handoff/evidence.jsonl",
  "split_provenance_path": "/results/source_run/components/tfidf/split_provenance.jsonl",
  "output_dir": "/results/stage2_comparison",
  "report_dir": "/results/stage2_comparison_report",
  "clinical_question": "Which pretreatment characteristics confound treatment selection or modify treatment effect?",
  "unit_id_column": "patient_id",
  "text_column": "clinical_text",
  "treatment_column": "treatment_indicator",
  "outcome_column": "outcome_indicator",
  "outcome_type": "binary",
  "inner_folds": 5,
  "seed": 42,
  "stage2": {
    "endpoint": "http://127.0.0.1:8010/v1",
    "workers": 4,
    "extraction_llm": {"endpoint": "http://127.0.0.1:8020/v1", "workers": 4},
    "min_propensity": 0.1,
    "max_propensity": 0.9
  }
}
```

| Key | Meaning |
|---|---|
| `dataset`, `handoff_path`, `split_provenance_path` | Inputs from the source run; use the source run's clinical question, columns, inner-fold count, and seed |
| `output_dir` | Stage 2 output directory |
| `report_dir` | Launcher status files: `status.json`, `preflight.json` (input hashes, git commit, model identity), `result.json`, and a `run.lock` that prevents concurrent launches |
| `stage2` | A `stage2` block with the same keys as the workflow configuration. The default for `workers` is computed from 8 |
| `concurrency_control_path` | Optional JSON file of live request limits (below) |
| `tokenizer_cache_dir` | Optional cache directory for the extraction tokenizer |

```bash
python scripts/run_stage2_from_artifacts.py artifact_run.json --preflight
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python scripts/run_stage2_from_artifacts.py artifact_run.json
```

`--preflight` loads the dataset's four declared columns, hashes the inputs,
checks that the extraction tokenizer loads, resolves the served models, and
stops. The handoff itself is hashed when Stage 2 starts. The launcher does not
start managed vLLM pools, so configure external endpoints. Logs go to standard
error.

The optional `concurrency_control_path` names a JSON file such as
`{"interpretation": 4, "extraction": 4, "total": 6}`. The launcher rereads it
about once per second while running. Each role's limit cannot exceed its
configured worker count, and `total` cannot exceed their sum. A limit of 0
stops admitting requests while in-flight requests finish; an invalid edit keeps the
last valid limits. When both roles share one server, `total` bounds their
combined load.

For `required_architectures`, the launcher's default is the eight default
architectures. When the source run used a different selection, set
`stage2.required_architectures` (and, to restrict compilation,
`stage2.included_architectures`) to match its handoff.
