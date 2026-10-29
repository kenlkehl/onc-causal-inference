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

For supplied variables, choose one of two fresh-run entry points:

| Input | Mode and example | Entry point |
|---|---|---|
| Pretreatment text plus complete extraction ontologies and roles | [`explicit_only`](#explicit-only-workflow), [`research_explicit_only.json`](../example_configs/research_explicit_only.json) | LLM extraction, then downstream Stage 2 |
| Pretreatment covariates already in table columns, with types and roles | [`tabular`](#tabular-workflow), [`research_tabular.json`](../example_configs/research_tabular.json) | Numerical learners and diagnostics; no LLM |

Both skip Stage 1 and freeze the supplied definitions and roles.
`--stage2-only` by itself resumes the selected input mode from a saved
configuration; a fresh all-evidence Stage 2 run still needs a handoff.

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
| `notes` | omitted | Dated-notes input: build the patient text from a table of dated notes (see [`notes`](#notes)) |
| `tabular` | omitted | Declared measured covariates for [tabular mode](#tabular-workflow) |
| `stage1_template` | `example_configs/all_evidence_stage1_model.json` | Stage 1 text-model template (see [Stage 1 template settings](#stage-1-template-settings)) |
| `neural_query_template` | `example_configs/all_evidence_neural_query_model.json` | Template for the `neural_query_moments` architecture |

### `columns`

| Key | Default | Meaning |
|---|---|---|
| `unit_id` | `patient_id` | Unique, nonmissing patient identifier |
| `text` | `clinical_text` | Pretreatment clinical text; unused and not required in tabular mode. With [`notes`](#notes), the name of the text column that OCI derives; `dataset` must not contain it |
| `treatment` | `treatment_indicator` | Binary treatment, 0 or 1 |
| `outcome` | `outcome_indicator` | Outcome, binary or continuous as declared by `science.outcome_type` |

All supplied text is treated as pretreatment. A single text column is not
screened by date; to have OCI keep only notes dated before each patient's index
date, supply dated notes through the [`notes`](#notes) block. Columns whose
names start with `true_`, `oracle_`, or `ground_truth` are treated as
[oracle columns](glossary.md#oracle-columns) and are never used for modeling.

### `notes`

Optional. Without it, the patient text is the `columns.text` column of
`dataset`. With it, OCI builds that column from a long-format notes table,
keeping only notes dated before each patient's
[index date](glossary.md#index-date); [Pretreatment text](../README.md#pretreatment-text)
explains why. `dataset` then holds one row per patient (identifier, index date,
treatment, outcome, and any other columns) and no text column.

```json
"notes": {
  "path": "notes.parquet",
  "columns": {"unit_id": "patient_id", "date": "note_date", "text": "note_text", "note_type": null},
  "index_date_column": "index_date",
  "include_index_date": false,
  "lookback_days": null,
  "note_header": "relative_days"
}
```

| Key | Default | Meaning |
|---|---|---|
| `path` | required | CSV or Parquet notes table with one row per note; a relative path is resolved like `dataset` |
| `columns.unit_id` | the value of `columns.unit_id` | Patient identifier column of the notes table. Values must equal the cohort's identifiers, including their type |
| `columns.date` | `note_date` | Note date column: a date, date-time, or date text (ISO 8601, such as `YYYY-MM-DD`, is safest) |
| `columns.text` | `note_text` | Note text column |
| `columns.note_type` | null | Optional note type column. When set, each header ends with the note's type, for example `· Progress note` |
| `index_date_column` | required | Column of `dataset` holding each patient's index date: the treatment decision or treatment start. Every patient needs a parseable date |
| `include_index_date` | `false` | Also keep notes dated on the index day. By default only notes dated strictly before the index day are kept, because notes written that day often record the treatment decision or the first administration |
| `lookback_days` | null (all history) | When a positive integer, drop notes dated more than this many days before the index date |
| `note_header` | `relative_days` | `relative_days` starts each note with a header such as `[Note 3 of 7 · 42 days before index date]`; `none` adds no header |

Unknown keys under `notes` and `notes.columns` are errors, and `--set
notes.KEY=VALUE` changes any of them. Only the configured note columns are read
from the notes table, so its other columns, including any oracle columns, never
reach the analysis.

**The derived text column.** `columns.text` names the column that OCI creates.
If `dataset` already has a column of that name, the run stops instead of
choosing between them; drop or rename that column, or set `columns.text` to an
unused name.

**Synthetic dated notes.** The synthetic generator's post-treatment option
writes `notes.parquet` in the default column format above and an `index_date`
column in its `dataset.parquet`. That dataset also has `clinical_text`, so set
`columns.text` to another name; see
[Synthetic data](synthetic_data.md#post-treatment-dated-notes).

**How the text is built.** Dates are compared by calendar day. Values with a
time zone are converted to UTC before the day is taken, so a late-evening note
written west of UTC can fall on the following day; values without a time zone
are used as given, so supply local dates when the day boundary matters. Each
note is counted in the first category that applies:

| Category | Meaning |
|---|---|
| `unmatched_patient` | The note's identifier is not in `dataset` |
| `undated` | The note date is missing or unparseable, so the note cannot be verified as pretreatment |
| `after_index_date` | Dated after the index day |
| `on_index_date` | Dated on the index day, unless `include_index_date` is true |
| `before_lookback` | Dated more than `lookback_days` before the index date |
| `empty_text` | Otherwise eligible, but the text is empty or only whitespace |
| `retained` | Kept |

Each patient's retained notes are ordered by date (notes on the same day keep
their order in the notes table), stripped of surrounding whitespace, and joined
with a blank line between notes, each after its header unless `note_header` is
`none`. Headers state days relative to the index date and never a calendar
date, so absolute dates reach the text, and the prompts, only where the note
bodies contain them. A patient without retained notes gets empty text, which
the extraction LLM receives as a record that documents nothing; the run logs a
warning with the number of such patients.

The run stops when a cohort identifier is missing or repeated, an index date is
missing or unparseable, a configured column is missing, `dataset` already
contains the derived text column, or no note identifier matches a cohort
identifier.

**Files.** OCI writes these files at the first Stage 1 start, and every step
reads `input/analysis_cohort.parquet` in place of `dataset`:

| File | Contents |
|---|---|
| `input/analysis_cohort.parquet` | Every column of `dataset` in its row order, plus the derived text column, and nothing else |
| `input/pretreatment_notes_manifest.json` | Schema `pretreatment_notes_manifest_v1`: the resolved paths and SHA-256 hashes of `dataset` and the notes table, the `notes` settings, the note counts per category, the number of patients without retained notes, the minimum, median, and maximum retained notes and text characters per patient, the SHA-256 of `analysis_cohort.parquet`, and the creation time |
| `input/pretreatment_notes_audit.csv` | One row per cohort patient: `unit_id`, the patient's note count in each category except `unmatched_patient`, `retained_characters` (the length of the derived text, headers included), `earliest_retained_days_before_index`, and `latest_retained_days_before_index` |

The audit has no treatment or outcome columns. It is for investigators; no
model or LLM step reads it.

**Resuming.** Every later start, including `--stage2-only`, `--stage2-reselect`,
and `--preflight-only`, hashes `dataset` and the notes table again and compares
them, the `notes` settings, `columns.unit_id`, `columns.text`, and the hash of
`analysis_cohort.parquet` with the manifest. When everything matches, the
derived file is reused without being rewritten, which keeps Stage 1 checkpoints
valid because they identify the analysis file by path, size, and modification
time. Any difference stops the run and names the mismatch; analyze changed
notes or settings in a different output directory. A directory that holds a
dated-notes manifest cannot be resumed without its `notes` block, a `notes`
block cannot be added to a directory that already holds Stage 1 or Stage 2
output, and Stage 2-only launches never build the derived file. `--status`
reads only `progress.json`.

`run_config.json` keeps `dataset` (the cohort file) and the resolved `notes`
block, so `--config OUTPUT_DIR/run_config.json` reproduces both. The resolved
Stage 1 configuration (`dataset_path`) and the handoff index (`dataset`) name
`input/analysis_cohort.parquet`. `oci-evaluate-stage1` reads oracle columns
from `dataset`, whose rows match the analysis cohort one to one.

### `science`

| Key | Default | Meaning |
|---|---|---|
| `clinical_question` | `"Identify text-derived confounders and treatment-effect modifiers."` | Study question given to the Stage 1 and Stage 2 language models |
| `outcome_type` | `binary` | `binary` or `continuous` |
| `outer_folds` | 5 | Outer cross-fitting folds ([outer and inner folds](glossary.md#outer-and-inner-folds)) |
| `inner_folds` | 5 | Inner folds within each outer-training set |
| `seed` | 42 | Seed for fold splits and models |
| `stage1_architectures` | omitted | Comma-separated string or list of Stage 1 architectures, or `all` (all ten). Omitted runs the architectures enabled by the Stage 1 template: eight with the bundled template, for binary and continuous outcomes. See [Stage 1 architectures](stage1_architectures.md) |
| `htr_enabled` | `true` | `false` disables all HTR modeling and with it the `htr_neural` architecture (same as `--disable-htr`) |
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
| `contexts_per_gpu` | 1 | Maximum concurrent text-model/HTR and neural-query discovery contexts per GPU, each in a separate process. Extra lanes are capped by `workers`, and the CPU budget is divided across lanes. Safe to change on resume |
| `components` | all five Stage 1 components | Subset of `embedding_cache`, `tfidf`, `text_models`, `neural_queries`, `handoff`. With an architecture selection, components the selection does not need are skipped |
| `mode` | `full` when Stage 2 is configured, otherwise `stage1` | `full`, `stage1`, `stage2`, `explicit_only`, or `tabular` (set by the corresponding `--stage1-only`, `--stage2-only`, `--explicit-only`, or `--tabular-only` flag) |
| `log_level` | `INFO` | Logging level for `logs/workflow.log` |

### `stage2`

For text workflows, Stage 2 is enabled when `stage2.endpoint`, `stage2.model`,
or `stage2.vllm` is set, and it also requires `stage2.extraction_llm`.
`run.mode=tabular` enables the numerical part of Stage 2 without any of those
LLM settings. The tables below group the keys; an omitted key takes the default
shown.

**Interpretation LLM and transport**

| Key | Default | Meaning |
|---|---|---|
| `endpoint` | empty | OpenAI-compatible base URL of the interpretation LLM. Mutually exclusive with `vllm` |
| `model` | empty | Served model ID. With `endpoint`, it may be omitted when the server advertises exactly one model. Required with `vllm` |
| `api_key` | `EMPTY` | API key; when absent from the configuration, `OCI_STAGE2_API_KEY` is used |
| `vllm` | none | Managed vLLM pool for the interpretation LLM ([Serving](serving.md#managed-vllm-pools)) |
| `vllm_rapid_switch_seconds` | 900 | Managed pools only: when two managed models switch again within this interval, both stay resident on their configured GPUs for the rest of the run; 0 always alternates |
| `workers` | `min(4, run.workers)` | Maximum concurrent interpretation requests across all outer folds; in tabular mode, concurrent outer folds |
| `frozen_feature_definition_model` | empty | Reuse completed feature definitions produced by this model ID while `model` serves the remaining requests. Definitions that are missing are not regenerated |
| `extraction_llm` | none | Extraction LLM settings (next table) |

**`stage2.extraction_llm`**

Configure exactly one of `endpoint`, `endpoints`, or `vllm`.

| Key | Default | Meaning |
|---|---|---|
| `endpoint` | empty | OpenAI-compatible base URL of the extraction LLM |
| `endpoints` | none | List of prestarted servers, each `{"endpoint": URL, "max_concurrency": N}`, used with load-aware routing; optional `model` selects that server's served model ID |
| `allow_model_variants` | `false` | Explicitly allow different backing models and served IDs in an external extraction pool. They must share a recognized model family and compatible tokenization; every actual model is audited. The served-ID set must stay unchanged once extraction-dependent checkpoints exist |
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
| `max_review_rounds` | 4 | Maximum aggregate extraction-review rounds per outer fold; stops early when definitions are unchanged and no validation fallbacks remain |
| `ontology_refinement_min_failure_patients` | 3 | A recurring extraction failure triggers definition refinement when it affects at least this many patients (minimum 2) |
| `max_ontology_refinement_rounds` | 2 | Refinement rounds per review round; 0 disables refinement |
| `required_architectures`, `included_architectures` | the eight default architectures; all | Architectures whose evidence must be present, and architectures whose evidence is compiled. The workflow sets both from the frozen Stage 1 selection; they matter only for `scripts/run_stage2_from_artifacts.py`. Accept a list, a comma-separated string, or `all` |
| `explicit_features` | `[]` | Investigator-specified variables (see [Explicit features](#explicit-features)) |
| `feature_source` | `all_evidence` | `all_evidence` adds supplied features to discovery. `explicit_only` starts at extraction; `tabular` starts at numerical modeling. Set automatically by the corresponding run mode or flag |

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
| `min_informative_patients` | 20 | A candidate is kept only if at least this many outer-training patients have an observed value ([documented unknown](glossary.md#documented-unknown) counts as observed); categorical candidates also need two levels (counting "not documented" and documented unknown) with at least this many patients each, and continuous candidates need non-identical observed states |
| `max_extraction_failure_fraction` | 0.10 | Candidates whose training [extraction failures](glossary.md#extraction-failure) exceed this fraction of patients are dropped |
| `min_propensity`, `max_propensity` | null | Inclusive [propensity eligibility bounds](glossary.md#propensity-eligibility-and-clipping) for effect estimation; either may be set alone. Set them here, not under `statistical_selection` |
| `propensity_clip` | 0.02 | Clipping of AIPW denominators to `[c, 1 − c]`; excludes no one. Under the `orthogonal_tests` modifier rule it also clips the AIPW projection test, so it is part of the selection identity |
| `lead_estimand` | `"ate"` | Which co-primary estimand the study pre-specifies to lead with: `"ate"` (clipped AIPW ATE over eligible patients) or `"ato"` (overlap-weighted ATE). Changes only the order and role labels in `causal_estimate.json`, never the computation; not part of the selection or fold estimation identity (pooled outputs are rewritten whenever Stage 2 runs; on a completed run use `--rerun stage2`) |
| `overlap_flag_bounds` | `[0.05, 0.95]` | Bounds of the reporting-only `outside_overlap` flag in `cross_fitted_predictions.csv` (two numbers, `0 < low < high < 1`). Never removes patients or changes estimates; trimming is `min_propensity`/`max_propensity` |
| `sensitivity` | see below | Reporting-only sensitivity to unmeasured confounding |
| `nuisance_learner` | `"super_learner"` | Learner for the final-estimation nuisances `e`, `μ0`, `μ1` (shared with the modifier tests): `"super_learner"` or `"elastic_net"` ([Nuisance learner](stage2_selection.md#nuisance-learner)). The selection-stage role models and the causal forest's internal nuisances are elastic nets either way. Part of the estimation identity and, under `orthogonal_tests`, of the selection identity |
| `super_learner` | see below | Super-learner settings, used when `nuisance_learner` is `"super_learner"` |
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

### `stage2.super_learner`

| Key | Default | Meaning |
|---|---|---|
| `cv_folds` | 5 | Internal cross-validation folds that set the stack weights (at least 2; reduced automatically for a small arm or rare class) |
| `learners` | `["elastic_net", "gradient_boosting", "random_forest"]` | Base learners, any nonempty subset in this order; a single learner is used alone |

Unknown keys are errors. The learner settings are fixed (see
[Nuisance learner](stage2_selection.md#nuisance-learner)). Each nuisance fit's
audit in `estimation/nuisance_fit_audits/` holds its stack weights, per-learner
and stack internal-CV losses, and the base learners' audits, and
`estimation/diagnostics.json` summarizes them under `nuisance_learner`.

### `stage2.sensitivity`

| Key | Default | Meaning |
|---|---|---|
| `benchmark_groups` | 5 | Discovered variables per outer fold benchmarked by refitting the elastic-net nuisances without each of them ([Benchmarking](stage2_selection.md#benchmarking-against-discovered-variables)); an integer from 0 to 50, where 0 skips the refits. E-values and omitted-variable-bias bounds are always reported |

Unknown keys are errors. The analyses are reporting only: no selection or
estimation step reads them, and the key is not part of the selection or fold
estimation identity. The benchmark refits run after a fold's final estimation
with their own checkpoints, so changing `benchmark_groups` refits only the
benchmarks the next time Stage 2 runs (on a completed run, use
`--rerun stage2`).

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
| `unpenalized_selected_interactions` | null | `orthogonal_tests` only: null or `true` exempts the selected modifiers' interaction and main-effect columns from the final interaction model's tuned penalty, subject to the events-per-parameter guard and the separation fallback ([Selected-modifier penalty](stage2_selection.md#selected-modifier-penalty)); `false` keeps one penalty for every main effect and interaction. Must be null or `false` under `balanced_evidence` |
| `unpenalized_events_per_parameter` | null | Guard ratio for the exemption, a positive number: it is used only when 2 × (encoded columns of the selected modifiers) is at most min(events, non-events) / ratio for binary outcomes, or effect-eligible training rows / ratio for continuous outcomes. Null uses 10 for binary and 15 for continuous outcomes. Must be null unless the exemption applies |

Unknown keys are errors. `modifier_count.concept_review: true` requires
`rule: "balanced_evidence"`. The rule and its settings are part of the
selection identity, so changing them after selection has run requires
[guarded reselection](#guarded-reselection); the multi-model evidence itself is
reused. The exceptions are `unpenalized_selected_interactions` and
`unpenalized_events_per_parameter`, which only the final interaction model
reads: changing them refits the final estimation and reuses the selection.

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

### Explicit-only workflow

`--explicit-only` or `run.mode=explicit_only` starts a fresh analysis using
only `stage2.explicit_features`. It needs no Stage 1 models, templates,
architecture selection, evidence handoff, or previous run. It still extracts
values from the configured text column (or the existing `notes` input); it
does not ingest pre-extracted covariate columns.

Start from
[`example_configs/research_explicit_only.json`](../example_configs/research_explicit_only.json).
Configure the cohort, treatment/outcome columns, clinical question, primary
and extraction endpoints, and a nonempty list of complete feature definitions.
Each variable must declare its causal roles. The usual Stage 2 machinery
handles extraction, nuisance fitting, effect estimation, held-out predictions,
and diagnostics. Definitions and declared roles retain the protections
described above; `selection_consolidation.protect_explicit_features` must stay
`true`. No additional variables are discovered.

Both `science.outer_folds` and `science.inner_folds` must be at least two.
The first run creates nested splits with the same joint treatment/outcome
stratification and deterministic fallback used by the all-evidence workflow.
`stage2/explicit_inputs.json` stores the complete split plan, ordered cohort
fingerprint, question, seed, column settings, and supplied ontologies.
Resuming checks and reuses this file, including when Stage 2 is already complete.
Changing these inputs requires a fresh output directory. A missing or altered
split file after Stage 2 has started is an error. All-evidence and explicit-only
runs must use separate output directories.

```bash
# Fresh analysis, or resume the same analysis
uv run python scripts/run_all_evidence.py --config study.json --explicit-only

# Read-only validation; on a fresh run the split plan is previewed in memory
# Specify both model IDs in the config for preflight (no endpoint lookup).
uv run python scripts/run_all_evidence.py --config study.json --explicit-only --preflight-only

# Resume from the resolved configuration; explicit-only mode is preserved
uv run python scripts/run_all_evidence.py --config OUTPUT_DIR/run_config.json --stage2-only

# Reselect with changed downstream settings, retaining definitions and measurements
uv run python scripts/run_all_evidence.py --config OUTPUT_DIR/run_config.json --stage2-only --stage2-reselect
```

The output uses the same `stage2/outer_*`, `cross_fitted_predictions.csv`,
`causal_estimate.json`, and diagnostic artifacts as the all-evidence workflow.
The run also saves `run_config.json`, `progress.json`, and workflow logs.

### Tabular workflow

`--tabular-only` or `run.mode=tabular` runs the shared Stage 2 numerical
pipeline directly on measured covariates. It needs no Stage 1 artifacts,
text column, LLM, model downloads, or extraction ontology. Start from
[`example_configs/research_tabular.json`](../example_configs/research_tabular.json)
and edit the dataset path and column declarations.

The cohort must be a CSV or Parquet table with one row per patient, unique
nonmissing IDs, a complete 0/1 treatment with both arms present, and a complete
finite outcome. Set `science.outcome_type` to `binary` (0/1) or `continuous`.
Covariates must describe the pretreatment state; this mode does not derive
covariates or screen their measurement dates.

Declare the measured columns in a top-level `tabular` block:

```json
{
  "dataset": "cohort.csv",
  "output_dir": "results/tabular",
  "columns": {"unit_id": "patient_id", "treatment": "treated", "outcome": "outcome"},
  "science": {"outcome_type": "binary", "outer_folds": 5, "inner_folds": 5, "seed": 42},
  "run": {"mode": "tabular"},
  "tabular": {
    "features": [
      {"column": "age", "value_type": "continuous", "roles": ["confounder"]},
      {"column": "ecog", "value_type": "ordinal", "categories": [0, 1, 2, 3, 4],
       "roles": ["confounder", "effect_modifier"]},
      {"column": "biomarker_positive", "value_type": "binary", "roles": ["effect_modifier"]}
    ]
  }
}
```

| Feature field | Requirement |
|---|---|
| `column` | Exact source column name. It must differ from the ID, treatment, outcome, and other covariates. Output names are normalized to lowercase snake case and must remain distinct |
| `value_type` | `continuous`, `binary`, `categorical`, or `ordinal` |
| `roles` | Nonempty list containing `confounder`, `effect_modifier`, or both. A single string and the alias `both` are also accepted |
| `categories` | Exactly two for binary (defaults to `[0, 1]`); at least two for categorical/ordinal; omitted for continuous |
| `description` | Optional human-readable description |

Only declared covariates enter modeling. Oracle-prefixed columns (`true_`,
`oracle_`, `ground_truth`) cannot be declared as covariates; other unlisted
columns are ignored by the learners. An oracle ITE column may still be used
by the existing post hoc evaluation.

**Encoding and missingness.** Continuous covariates must contain numbers or
missing values; nonnumeric values and infinities are errors. Numeric category
codes are normalized consistently (`1` and numeric `1.0` both become `"1"`),
and surrounding whitespace in string labels is stripped. Every observed
nonmissing category must be declared in advance. Binary, categorical, and
ordinal covariates use category indicators; ordinal codes are not implicitly
treated as a linear score. Declare a numeric variable `continuous` to request
that representation. Missing covariates are allowed: continuous medians and
scales are fitted on training rows, with a missingness indicator; categorical
variables have a missing level. Use null/NaN for missingness. The internal
labels `__missing__` and `documented_unknown` cannot be declared as categories.
There are no extraction-failure states in this mode.

**What runs.** Roles stay fixed in every outer fold. OCI runs no candidate
discovery, statistical variable selection, semantic consolidation, or ontology
revision. The complete declared covariate catalog feeds the shared treatment
and outcome nuisance learners, including variables declared only as effect
modifiers. Confounder roles define adjustment terms in the effect models;
modifier roles define the effect features. The default nuisance learner is
the existing Super Learner. The default final estimator is an equal-weight
mean of fitted component CATE predictions: causal forest and interaction
model, plus DINA for binary outcomes. A component that cannot be estimated
is recorded in the fold diagnostics and omitted from the mean.

Outer held-out predictions, inner cross-fitting, AIPW ATE/ATO, overlap,
heterogeneity diagnostics, and sensitivity reporting use the same code as
the text workflow. Both fold counts must be at least two. Shared numerical
settings remain under `stage2`: for example `nuisance_learner`,
`super_learner`, `modeling_workers`, `estimation_trees`, propensity
bounds, and the interaction-model settings. To request a single final
estimator, set
`stage2.statistical_selection.multi_model.modifier_selection.estimator_policy`
to `causal_forest`, `linear_interactions`, or `dina` (binary only), instead of
the default `ensemble`. Keep `modifier_selection.rule=orthogonal_tests`;
in this mode it supplies the final estimator policy and penalty conventions
but runs no selection tests. `balanced_evidence` is not supported because it
selects modifiers and a model. LLM settings, `stage2.explicit_features`,
`notes`, and Stage 1 architecture selections are rejected.

```bash
# Fresh analysis or resume; the mode can also be set in the config
uv run python scripts/run_all_evidence.py --config study.json --tabular-only

# Read-only input validation and split preview; no servers are contacted
uv run python scripts/run_all_evidence.py --config study.json --tabular-only --preflight-only

# Resume from the saved resolved configuration
uv run python scripts/run_all_evidence.py --config OUTPUT_DIR/run_config.json --stage2-only
```

**Resume and outputs.** The first run saves the ordered cohort and feature
identity plus nested splits in `stage2/tabular_inputs.json`. Every subsequent
run validates that file, including completed runs. Changing measured values,
row order, selected columns, types, categories, roles, question, folds, or seed
requires a fresh output directory. Different input modes cannot share a run
directory. Compatible estimation checkpoints are reused; changing numerical
model settings refits affected estimates without resampling. There is no
`--stage2-reselect` operation for fixed tabular roles: rerun normally.

`stage2/measurements.csv` records normalized supplied values.
`stage2/outer_NNN/final_definitions.json` maps model names back to source
columns and records fixed roles; `measurements/fit.csv` and
`measurements/heldout.csv` record each fold's inputs. The shared outputs are
`stage2/cross_fitted_predictions.csv`, `causal_estimate.json`,
`heterogeneity_diagnostics.json`, and per-fold `estimation/` artifacts.
`run_config.json`, `progress.json`, and workflow logs are also written.
No handoff, extraction cache, or variable-selection directory is created.

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
| `multi_model_forest.bow_effect_ridge_alpha_grid` | `[0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]` | `bow_r_loss`: effect ridge penalties (`ridge_alpha`; DINA uses `ridge_alpha / n`) chosen by the pooled held-out effect loss over the effect folds; `[]` uses each view's fixed `ridge_alpha` ([details](stage1_architectures.md#2-sparse-residual-effect-associations-bow_r_loss)) |
| `multi_model_forest.matched_pair_propensity_caliper` | 0.05 | `matched_pair_uplift`: absolute caliper on the cross-fitted propensity |
| `multi_model_forest.matched_pair_prognostic_caliper` | `null` | `matched_pair_uplift`: caliper on the cross-fitted prognostic score `μ̂0`; `null` means 0.05 on the probability scale for binary outcomes and 0.2 standard deviations of `μ̂0` over the training patients for continuous outcomes. A binary value must be in (0, 1] |
| `multi_model_forest.matched_pair_l1_ratio` | 0.5 | `matched_pair_uplift`: elastic-net mixing of the pair model and of the continuous-outcome `μ̂0` model (1 is the lasso) |
| `multi_model_forest.matched_pair_cv_folds` | 5 | `matched_pair_uplift`: folds over pairs for the penalty choice and the nested held-out validation |
| `multi_model_forest.matched_pair_min_discordant_pairs`, `matched_pair_min_pairs` | 30, 30 | `matched_pair_uplift`: minimum discordant pairs (binary) or matched pairs (continuous), never below 4 (the nested cross-fit needs two pairs per fold); below it a context emits no matched-pair evidence and records why |
| `multi_model_forest.matched_pair_max_iter` | 5000 | `matched_pair_uplift`: solver iterations per penalty value |

Chunk-based models stop with a capacity error rather than truncating a long
record; raise the relevant `max_chunks` value when that happens. The workflow
fills in the dataset, columns, folds, seed, devices, HTR and embedding models,
and the embedding cache location itself. The resolved templates are written to
`resolved_stage1_model_config.json` and `resolved_neural_query_config.json`.

HTR optimization and loading settings are relative to `science.stage1.training`:

| Key | Default | Meaning |
|---|---|---|
| `batch_size` | 8 | Patients per nuisance-training and prediction batch |
| `effect_batch_size` | 8 | Patients per effect-training batch; `null` inherits `batch_size` |
| `dataloader_workers` | 2 | CPU preprocessing processes per HTR trainer; `0` loads synchronously |
| `dataloader_prefetch_factor` | 2 | Batches prepared ahead per preprocessing worker |

Stage 1 caps preprocessing workers at the context's CPU allocation minus one
CPU for training. For example, an allocation of three CPUs permits two loader
workers; one CPU keeps loading synchronous. Workers start fresh interpreters
(`spawn`, or a compatible `loky` context inside the workflow process pool),
persist across epochs, and prepare CPU tensors pinned for CUDA transfer. The training
process owns the model and GPU. Loss averages are copied to CPU once per epoch;
batch progress messages report progress and learning rate without reading losses.
Changing batch size changes the number of optimizer steps per epoch. Explicit
settings in existing configurations still override these defaults.

### Neural-query template settings

`neural_query_moments` settings live in the neural-query template and are
overridden through `science.neural_queries`, for example
`--set science.neural_queries.evidence_ngram_top_k=50`. Commonly changed keys:

| Key | Template value | Meaning |
|---|---|---|
| `treatment_query_count`, `outcome_query_count`, `effect_query_count` | 5, 5, 5 | Queries per bank |
| `query_inner_folds` | `science.inner_folds` | Inner folds that produce candidate queries |
| `temperature` | 0.05 | Temperature of the log-mean-exp patient activation |
| `evidence_top_patients` | 10 | Witness patients per query, chosen by their best-matching chunk |
| `evidence_retrieval_top_k` | 1 | Witness chunks per witness patient |
| `evidence_ngram_top_k` | 30 | Phrases kept per query: the positive-contrast terms with the largest contrast between the witness chunks and the other training patients' best-matching chunks; `null` keeps every positive-contrast term |
| `evidence_top_ngrams` | `null` | Ceiling on the phrases kept per query; a query that would keep more stops the run with a capacity error rather than dropping terms |

Queries are fitted, scored, and retrieved on chunks with at least 12 letters
or digits, the rule the embedding architectures use for retrieval
([details](stage1_architectures.md#10-learned-neural-queries-neural_query_moments)).

## Command-line flags

`python scripts/run_all_evidence.py [FLAGS]`. Flags override the configuration
file; `--set` can change any other value.

**Inputs and study design**

| Flag | Configuration key | Meaning |
|---|---|---|
| `--config PATH` | — | JSON or YAML configuration, or a saved `OUTPUT_DIR/run_config.json` |
| `--dataset PATH` | `dataset` | Input Parquet or CSV dataset (with [`notes`](#notes), the cohort file) |
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
| `--contexts-per-gpu N` | `run.contexts_per_gpu` | Concurrent discovery contexts per GPU (default 1); separate processes preserve independent training state and seeds |
| `--architectures LIST` | `science.stage1_architectures` | Comma-separated Stage 1 architectures, or `all` for all ten. Frozen on resume |
| `--htr-model ID` | `models.htr` | HTR chunk encoder |
| `--disable-htr` | `science.htr_enabled=false` | Disable all HTR modeling, keeping the other evidence families |
| `--embedding-model ID` | `models.embeddings` | Embedding model for the embedding and neural-query architectures |

**Run mode** (`--stage1-only`, `--stage2-only`, `--explicit-only`, and `--tabular-only` are mutually exclusive)

| Flag | Meaning |
|---|---|
| `--stage1-only` | Run or resume Stage 1 through the handoff, then stop |
| `--stage2-only` | Skip Stage 1 and run or resume Stage 2 from the saved handoff, usually with `--config OUTPUT_DIR/run_config.json` |
| `--explicit-only` | Extract `stage2.explicit_features` and run downstream analysis without Stage 1; creates and persists its own nested splits |
| `--tabular-only` | Read `tabular.features` from measured table columns and run the numerical pipeline without Stage 1 or any LLM |
| `--stage2-reselect` | With Stage 2-only or explicit-only mode: archive a completed Stage 2 selection and everything after it, then rerun selection from the frozen definitions and outer-training extractions ([Guarded reselection](#guarded-reselection)) |
| `--preflight-only` | With Stage 2-only, explicit-only, or tabular mode: validate inputs and the Stage 2 reuse plan without writing files, starting servers, or fitting models; prints a JSON report |
| `--rerun COMPONENT` | Remove a component's completion markers (not its outputs or identity-checked checkpoints) so it runs again ([details](#checkpoints-and-resuming)); repeatable. One of `embedding_cache`, `tfidf`, `text_models`, `neural_queries`, `handoff`, `stage2` |
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

The standalone HTR script (`scripts/run_standalone_htr.py`) has its own flags
and settings; see [Standalone HTR](standalone_htr.md#settings).

## The synthetic launcher (`run_synthetic.sh`)

`run_synthetic.sh` runs or resumes the full workflow on one of the two bundled
synthetic cohorts ([Synthetic data](synthetic_data.md#bundled-cohorts)):

```bash
[PRESET=NAME] [VARIABLE=VALUE ...] ./run_synthetic.sh COHORT [OUTPUT_DIR] [WORKFLOW_OPTION ...]
[PRESET=NAME] [VARIABLE=VALUE ...] ./run_synthetic.sh CONFIG.json [WORKFLOW_OPTION ...]
```

| Argument | Meaning |
|---|---|
| `COHORT` | `one_conf_one_mod` (one confounder, one effect modifier) or `five_conf_five_mod` (five of each) |
| `CONFIG.json` | A workflow configuration file (`.json`, `.yaml`, or `.yml`), such as a benchmark cell's `runs/VARIANT/config.json`. It supplies the dataset, output directory, columns, science, and Stage 2 settings; the launcher adds only the preset's serving settings, the detected `--devices` and `--workers`, and `STAGE1_ARCHITECTURES` or `DISABLE_HTR=1` when set. The cohort defaults and the bundled cohorts' Stage 2 definition settings below do not apply, and there is no `OUTPUT_DIR` argument; `STAGE2_ONLY=1` is refused (relaunch through `OCI_RUN_CONFIG`) |
| `OUTPUT_DIR` | Default `artifacts/research_all_evidence/COHORT_nsclc_full` for the `external` preset and `COHORT_nsclc_PRESET_full` otherwise, with `_no_htr` appended when `DISABLE_HTR=1`. Rerun the same command to resume |
| `WORKFLOW_OPTION` | Passed to the workflow after the launcher's own options, so a later `--set` wins, for example `--set stage2.consolidation_max_rounds=12` |

The launcher synchronizes `.venv` from the lockfile with `uv` (adding the
`local-llm` extra when a managed vLLM pool is configured), raises the soft
open-file limit to the hard limit (each concurrent request holds a socket),
detects GPUs, sets
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
| `dgx_station_gb300` | Managed vLLM on a DGX Station's GB300 GPU; one model serves both roles from one server | `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `PHYSICAL_GPUS=1`, `MIN_TOTAL_GPU_GB=150`, 32 workers per role, `STAGE2_MODELING_WORKERS=64`, the `h100x8` timeouts, and a 262,144-token extraction window |

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
| `MIN_TOTAL_GPU_GB` | Total memory a GPU needs to be eligible (default 0) |
| `STAGE1_WORKERS` | Stage 1 CPU workers (default: sized from CPUs, folds, and GPU memory) |
| `STAGE1_CONTEXTS_PER_GPU` | Concurrent text-model/HTR and neural-query contexts per GPU (default: workflow configuration, or 1; `dgx_station_gb300` sets 4). Preserved with benchmark `--llm-url`; explicit environment values override the preset |
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
verified on 2026-09-23, the Qwen 3 and LFM 2.5 profiles on 2026-09-12, and the
Nemotron 3.5 and Beam 1 profiles on 2026-10-16.

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
| Nemotron 3.5 | either | 1.0 | 0.95 | 0 (off) | 0 | 1.0 |
| Beam 1 | any reasoning level | 0.7 | 0.99 | 0 (off) | 0 | 1.0 |
| Any other model | — | server defaults | | | | |

Sources: [Gemma 4](https://huggingface.co/google/gemma-4-31B-it#best-practices),
[Qwen 3](https://huggingface.co/Qwen/Qwen3-32B#best-practices),
[Qwen 3.5](https://huggingface.co/Qwen/Qwen3.5-27B#best-practices),
[Qwen 3.6](https://huggingface.co/Qwen/Qwen3.6-27B#best-practices),
[Qwen 3.8](https://huggingface.co/Qwen/Qwen3.8-27B#best-practices),
[Qwen 3.8 Flash Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next#best-practices),
[LFM 2.5 2.6B](https://huggingface.co/LiquidAI/LFM2.5-2.6B),
[LFM 2.5 1.2B](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct),
[Nemotron 3.5](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16)
(its text setting; the card's vision setting, temperature 0.6, does not apply
because Stage 2 sends text only),
[Beam 1](https://huggingface.co/reflection/Beam-1.0-501B-A23B) (its
`generation_config.json`, which matches the settings of the model card's
evaluations; the card gives no other recommendation).

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
  `none`, except `xhigh` for Flash Next. Beam 1 cannot turn reasoning off and
  receives `none` as `minimal`.
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
├── input/                               [notes] analysis_cohort.parquet, pretreatment_notes_manifest.json, pretreatment_notes_audit.csv
├── stage1_architectures/                [explicit selection] manifest.json and one evidence.jsonl per architecture
├── components/
│   ├── embedding_cache/                 chunk embedding cache, timing.jsonl
│   ├── tfidf/                           evidence.jsonl, split_provenance.jsonl (fold plan shared by all stages), contexts, timing.jsonl
│   ├── text_models/                     evidence.jsonl, timing.jsonl, one directory per discovery context with checkpoints
│   └── neural_queries/                  evidence.jsonl, timing.jsonl, one directory per discovery context
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
    │   ├── estimation/                  nuisance fits and audits, overlap, predictions.csv, diagnostics.json, sensitivity_benchmarks/
    │   ├── preselection/                [after reselection] frozen selection inputs
    │   ├── recovery_status.json, recovery_events.jsonl
    │   └── complete.json
    ├── features_by_outer_fold.jsonl     per-fold variables, roles, and estimation summary
    ├── cross_fitted_predictions.csv     one held-out prediction per patient
    ├── causal_estimate.json             co-primary ATE and ATO, sensitivity summary, overlap, calibration, heterogeneity summary
    ├── heterogeneity_diagnostics.json   BLP (primary test), GATES, CLAN, RATE, overlap, balance, modifier confirmation, CATE variance decomposition, unmeasured-confounding sensitivity
    ├── posthoc_oracle_ite_metrics.json  oracle comparison (records "unavailable" without oracle columns)
    ├── posthoc_predictions_with_oracle_ite.csv  [oracle columns present]
    ├── summary.json
    ├── reselection_state.json, reselection_archives/   [after reselection]
    └── complete.json
```

Without an explicit architecture selection, `stage1_architectures/` is not
written and the handoff streams the component evidence directly. Each
component's `timing.jsonl` (with one `timing.json` per context directory)
records reporting-only wall time per architecture; see
[Cost per architecture](stage1_architectures.md#cost-per-architecture).
`OUTPUT_DIR/evaluation/stage1/` is written only by `oci-evaluate-stage1`
([Synthetic data](synthetic_data.md#oracle-safe-evaluation); its files are
listed in [Stage 1 architectures](stage1_architectures.md#outputs)).
`OUTPUT_DIR/evaluation/attribution/` is written only by `oci-evaluate-attribution`
([Stage 1 architectures](stage1_architectures.md#attribution-of-stage-2-variables)).

### Files to read first

| File | Content |
|---|---|
| `stage2/causal_estimate.json` | `co_primary_estimands`: the clipped AIPW ATE and the ATO, per fold and pooled, each with its standard error, 95% CI, target population, method, and role (`lead_estimand` and `order` set which leads), plus `ate_minus_ato`; `unmeasured_confounding_sensitivity` (pooled E-values with the risk ratio and its interval, robustness values RV and RVa for the ATE and the ATO, and the strongest benchmark); `overlap_flags` counts; the top-level AIPW `ate`, standard error, and 95% CI; overlap and nuisance calibration summaries; mean estimated CATE; oracle correlations when available |
| `stage2/cross_fitted_predictions.csv` | One row per patient from its outer held-out fold: `estimated_cate` (a risk difference for binary outcomes), `aipw_score`, `propensity`, `mu0`, `mu1`, `effect_eligible`, `outside_overlap` (reporting only: `propensity` outside `stage2.overlap_flag_bounds`), `outer_fold`, plus per-estimator CATE columns under the ensemble (see [Modifier-selection and ensemble outputs](#modifier-selection-and-ensemble-outputs)) |
| `stage2/heterogeneity_diagnostics.json` | Whether the CATE predictions track real heterogeneity ([BLP, GATES, CLAN, RATE](glossary.md#blp-gates-clan-and-rate)); see [Stage 2 selection](stage2_selection.md) |
| `stage2/features_by_outer_fold.jsonl` and `stage2/outer_00k/final_definitions.json` | Which variables were used in each fold, their roles and definitions, and the chosen estimator |
| `stage2/outer_00k/selection/modifier_tests.json` | Every candidate's modifier tests and the selection (`orthogonal_tests`) |
| `stage2/outer_00k/selection/modifier_count/selection.json` | Modifier selection report: selected modifiers and estimator policy, or the `balanced_evidence` budget and estimator comparison |
| `stage2/outer_00k/extraction/failure_report_fit.json`, `failure_report_heldout.json` | Extraction failure rates by variable and treatment arm |
| `stage2/outer_00k/extraction/documented_unknown_report_fit.json`, `documented_unknown_report_heldout.json` | [Documented-unknown](glossary.md#documented-unknown) counts per variable: `documented_unknown_rows`, `documented_unknown_fraction`, and `by_treatment_arm` (rows, count, fraction) with `arm_difference_treated_minus_control`; the held-out file also lists `encoded_as_missing_without_training_occurrences`. The same reports appear under `documented_unknown` in `estimation/extraction_failure_diagnostics.json`, and `candidate_missingness_filter.json` records `documented_unknown_rows` per candidate |
| `stage2/outer_00k/estimation/diagnostics.json`, `propensity_overlap.json`, `nuisance_fit_audits/` | Fold-level ATE, calibration, overlap, and the fitted penalty and iteration count of each nuisance model |
| `stage2/outer_00k/estimation/sensitivity_benchmarks/benchmarks.json` | Reporting only: the long elastic-net reference, the benchmarked variables with their importance, selected roles, and gains (`cf_y`, `cf_d`, `rho`, short and long `sigma2`, `nu2`, and ATE), the importance and learner descriptions, and `status` (`ok`, `disabled`, `unavailable`, or `error`); fit checkpoints are under `fits/` |

### Modifier-selection and ensemble outputs

`stage2/outer_00k/selection/modifier_tests.json` (rule `orthogonal_tests`)
records `q_threshold`, the resolved `weights` and `weights_source`,
`max_modifiers`, `propensity_clip`, the propensity bounds,
`sparse_indicator_min_count`, the testing-nuisance source and calibration
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
component's audit), and `mean_estimated_cate_by_estimator`. Under
`orthogonal_tests` the interaction model's audit
(`interaction_fit_audit.selected_modifier_penalty`) records the
selected-modifier penalty: the settings, the guard computation and decision,
each attempted fit, and the path used (`unpenalized`, `separation_ridge`, or
`fully_penalized` with its reason); its architecture diagnostics, and so each
ensemble component's audit, repeat a compact `selected_modifier_penalty`
summary.

`stage2/heterogeneity_diagnostics.json` contains `modifier_confirmation`: the
confirmation policy and `p_value_role` (inferential); per-fold results for each
selected modifier (`confirmed`, held-out combined and component p- and
q-values, labels, `definition_text`, and `effects`: per-column `r_projection`
and `aipw_projection` coefficients with `standard_error`,
`confidence_interval_95` and `p_value_two_sided`, `kind` and `level`, and
`per_training_sd` for continuous values with the fold's `training_sd`); pooled
counts of selected and confirmed modifiers; and `across_folds`: the similarity
method and threshold, MMB `gamma`, and one entry per concept with its
`members`, `pairwise_similarities`, `selected_in_folds`, `confirmed_in_folds`,
`heldout_p_values_by_fold`, `aggregated_p_value`, `aggregated_q_value`,
`aggregated_supported`, and `pooled_effects` (fixed-effect estimate, standard
error, interval, `cochran_q`, `i_squared`, `fold_range`, or the reason effects
are per fold only). `modifier_tests.json` records `p_value_role` (screening). `stage2/causal_estimate.json` repeats a compact summary under
`heterogeneity_diagnostics.modifier_confirmation`.

The same file contains `cate_variance_decomposition` (`columns`: `estimated_cate`
and any ensemble component columns; per fold and per column `rows`,
`modifier_columns`, `cate_variance`, `r2_prognosis`, `r2_modifiers`, `r2_both`,
`unique_prognosis`, `unique_modifiers`, `shared`, `unexplained`, the modifier
design columns; `pooled` row-weighted means with each value's fold `range`),
`gates.*.calibration` (slope, standard error, one-sided p-value for slope > 0,
two-sided p-value for slope = 1), and `primary_heterogeneity_test` (the pooled BLP
`beta2`, its standard error, interval, and one-sided p-value, labeled
pre-specified). It also contains `clip_sensitivity` (per fold and pooled `points`
with `clip`, `configured`, `rows`, `estimate`, `standard_error`, and
`confidence_interval_95`) and `overlap_flags` (`bounds` and per-fold and pooled
`rows`, `outside_overlap`, `fraction`, `below_lower_bound`, `above_upper_bound`).
`unmeasured_confounding_sensitivity` has `outcome_type`, `interpretation`
strings, `references`, and three parts. `evalues` gives, for `ate` and `ato`
(`pooled` and `per_fold`), `risk_ratio`, `risk_ratio_confidence_interval_95`,
`evalue_point`, `evalue_confidence_limit`, and `scale` (`causal_risk_ratio`
with the arm means and `log_risk_ratio_standard_error`, or
`approximate_risk_ratio_from_standardized_mean_difference` with
`standardized_mean_difference`, its standard error, and `outcome_sd`).
`omitted_variable_bias` gives, for `ate` and `ato`, `estimate`,
`standard_error`, `sigma2`, `nu2`, `robustness_value`,
`robustness_value_confidence`, and `contour` (for each `cf_y`, `cf_d` in the
`grid`: `max_bias`, `lower`, `upper`, their standard errors,
`lower_confidence_bound`, `upper_confidence_bound`, and
`confidence_bounds_exclude_zero`), with `rho`, the critical value, and the Riesz
representers. `benchmarks` gives `per_fold` the fold's benchmark file contents
with `fold_ate_bounds` and `pooled_ate_bounds` (at `rho_1` and, when
identified, `benchmark_rho`) for each variable, and `pooled.rows`, sorted by
the implied bias at `rho = 1`, each with a plain-language `statement`.
`causal_estimate.json` repeats `primary_heterogeneity_test`, the
pooled decomposition per column, and the pooled GATES calibration and group mean
predicted CATEs under `heterogeneity_diagnostics`.

## Checkpoints and resuming

Rerunning the same command resumes a run. Each component writes
`complete.json` in its directory when it finishes, and completed components are
skipped. Inside components, every Stage 1 discovery context and every expensive
Stage 2 step (each interpretation, merging batch, extraction batch, review
round, selection evidence cell, and estimation fit) is checkpointed, so an
interrupted component re-enters at its first incomplete step.

- `--status` prints `progress.json`, which names the current component and
  any error.
- `--rerun COMPONENT` removes the completion markers that let a component be
  skipped, so it runs again. Outputs, and checkpoints that verify their own
  identity, stay in place:
  - `text_models` and `neural_queries`: the component's and each context's
    `complete.json` are removed, so every context runs again. A text-model
    context restores each fitted sub-step from its `checkpoints/` when the
    sub-step's identity still matches and refits the others. The identity
    covers the resolved Stage 1 model settings, the dataset file's path, size,
    and modification time, the context's rows, and the checkpoint version.
    Neural-query contexts have no finer checkpoints and are refit. To refit
    every text-model context, also remove the contexts' `checkpoints/`
    directories.
  - `tfidf`: every context's markers are also removed, so every context is
    refit.
  - `embedding_cache`, `handoff`, and `stage2`: only the component's
    `complete.json` is removed. The chunk cache is reused while its embedding
    model, chunking settings, dataset path, and row count still match, and
    each Stage 2 step is revalidated against its checkpoint, as on any resume.
- With [`notes`](#notes), `input/analysis_cohort.parquet` is built once and
  verified against its manifest on every start; a changed notes table, cohort
  file, or `notes` setting stops the run.
- Stage 1 completion markers do not detect every changed scientific setting.
  Use a different output directory for a scientifically different run. When a
  Stage 1 text producer changes, rerun `text_models` and `handoff`, and move
  the previous `stage2/` directory aside before running Stage 2 again; Stage 2
  rejects feature definitions whose evidence fingerprint does not match the handoff.

Text-based Stage 2 checks its saved state on every start:

| Change on a saved run | Result |
|---|---|
| Different interpretation model ID | Refused; use a different output directory |
| Different extraction model ID | Allowed only when no extraction-dependent checkpoints exist |
| Endpoint URLs, ports, worker counts, timeouts | Allowed |
| Interpretation sampling, reasoning, or `max_tokens` | Completed feature definitions do not match the requested policy; the run stops |
| Extraction context window, margin, chunk size, or output ceilings | Affected extraction chunks are extracted again |
| `min_propensity`, `max_propensity`, `statistical_selection` (other than the two selected-modifier penalty keys), or, after selection has run, `propensity_clip`, `nuisance_learner`, or `super_learner` (under `orthogonal_tests`) or `estimation_trees` (under `balanced_evidence`) | Refused; use [guarded reselection](#guarded-reselection) |
| `nuisance_learner` or `super_learner` under `balanced_evidence` | Allowed; only the final estimation is refitted |
| `estimation_trees` under `orthogonal_tests` | Allowed; selection and its stored nuisance models are reused and only the final estimation is refitted |
| `modifier_selection.unpenalized_selected_interactions` or `unpenalized_events_per_parameter` | Allowed; selection and its stored nuisance models are reused and only the final estimation is refitted |

For [tabular mode](#tabular-workflow), ordinary reruns revalidate the
estimation checkpoints even after completion. Numerical model changes refit
affected estimates directly; input, feature-schema, or split changes require
a fresh output directory. There is no selection checkpoint to archive.

`run_config.json` stores every API key as `<redacted>`. When relaunching from it
against servers that require a key, set the `OCI_STAGE2_*_API_KEY` environment
variables (the placeholder counts as no key) or pass the keys, for example
`--set stage2.api_key=KEY --set stage2.extraction_llm.api_key=KEY`. Do not run
two workflow processes against the same output directory.

### Checkpoint relocation

Stage 1 text-model checkpoints normally include the resolved dataset path,
size, modification time, and embedding-cache directory. To keep those exact
identities and deterministic seeds after a verified copy, a migration helper
can place `stage1_checkpoint_relocation.json` in the new output directory:

```json
{
  "schema_version": "stage1_checkpoint_relocation_v1",
  "dataset_path": "/new/location/dataset.parquet",
  "dataset_sha256": "<SHA-256 of the verified original dataset>",
  "original_dataset_identity": {
    "path": "/original/location/dataset.parquet",
    "size": 12345,
    "mtime_ns": 1700000000123456789
  },
  "original_output_dir": "/original/location/output",
  "original_applied_config_fingerprint": "<from the original context checkpoint>"
}
```

Copy the full run, retain its split provenance and checkpoint payloads, and set
`dataset` and `output_dir` in its launch configuration to the new physical paths.
The workflow verifies the copied dataset's checksum and all normalized Stage 1
settings before restoring original paths **only in checkpoint identity metadata**.
It hashes the dataset once per worker and again if filesystem metadata changes;
ordinary runs without this manifest retain their existing metadata-only checks.
A mismatch stops the context. The original directories need not exist, and the
new filesystem need not preserve nanosecond timestamps. Row and checkpoint
version checks remain in force.

This manifest covers text-model checkpoints. The embedding chunk cache has a
separate path-derived key: a migration helper must validate its settings and
re-key its copied directory and metadata for the new dataset path without
changing the numerical payloads. It does not relocate dated-notes manifests or
Stage 2 checkpoints. Do not use relocation to adopt changed data or scientific
settings; those require a fresh run.

## Stage 2-only launches

For fresh runs without a handoff, use [explicit-only extraction](#explicit-only-workflow)
or [tabular mode](#tabular-workflow). Their saved configurations preserve the
input mode when resumed with `--stage2-only`.

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

A run made with `--stage1-only` saves `stage2: null`. Supply its Stage 2
settings, including both model IDs, with `--set` (for example
`--set stage2.model=MODEL --set stage2.extraction_llm.model=MODEL` beside the
endpoints above), or launch from a configuration file that holds the run's
Stage 1 settings and a `stage2` block.

### Preflight

`--stage2-only --preflight-only` validates a saved run without writing files,
contacting endpoints, starting servers, or fitting models, and prints a JSON
report. It checks that:

- the saved Stage 1 settings match the requested ones (the Stage 2 settings
  come from the request, so a `--stage1-only` run validates the same way);
- the handoff and every Stage 1 component are complete and their artifacts
  exist;
- with [`notes`](#notes), the cohort file, the notes table, the `notes`
  settings, and `input/analysis_cohort.parquet` match the dated-notes manifest
  (the report then names the derived file as `dataset` and the cohort file as
  `source_dataset`);
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
| `dataset`, `handoff_path`, `split_provenance_path` | Inputs from the source run; use the source run's clinical question, columns, inner-fold count, and seed. For a source run with [`notes`](#notes), `dataset` is its `OUTPUT_DIR/input/analysis_cohort.parquet`, not the cohort file |
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

`scripts/ablate_stage1_architectures.py prepare` writes launcher
configurations of this form, one per Stage 1 architecture, each with the
run's handoff minus that architecture's evidence; see
[Leave-one-architecture-out ablation](stage1_architectures.md#leave-one-architecture-out-ablation).
