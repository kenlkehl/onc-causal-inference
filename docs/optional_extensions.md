# Optional extensions

This page describes features that are not part of the default Stage 2
procedure, and the standalone `oci run` workflows for investigator-specified
variables. Each entry covers what the feature does, when to use it, how to
configure it, and its caveats. The default procedure is described in
[How OCI works](how_it_works.md) and [Stage 2 selection](stage2_selection.md);
terms are defined in the [glossary](glossary.md).

| Feature | Configuration | Default | In the multi-model example configuration | Extra LLM requests | Uses treatment or outcome |
|---|---|---|---|---|---|
| [Estimand-informed ontology search](#estimand-informed-ontology-search) | `stage2.estimand_ontology.enabled` | off | off | yes, plus extraction | yes, to choose definitions |
| [Modifier-concept review](#modifier-concept-review) | `stage2.statistical_selection.multi_model.modifier_count.concept_review`, with `modifier_selection.rule` set to `balanced_evidence` | off | off | yes | aggregate statistics only |
| [Matched-batch contrast evidence](#matched-batch-contrast-evidence) | `stage2.statistical_selection.multi_model.matched_batch.enabled` | off | on | no | yes, in reported evidence only |
| [Pre-selection measurement consolidation](#pre-selection-measurement-consolidation) | `stage2.selection_consolidation.enabled` | off | on | yes | no |
| [Standalone `oci run` workflows](#standalone-explicit-feature-workflows-oci-run) | separate command and configuration | — | — | — | — |

All four Stage 2 features run within each outer fold on outer-training patients
only; outer held-out patients and oracle columns are never used. Within a fold
they run in this order: extraction and the informativeness filter, estimand
search, selection consolidation, multi-model evidence (including matched-batch
evidence), and modifier selection (including concept review). Turning
any of them on or off changes the selection checkpoint identity, so on a
completed run use [guarded reselection](configuration.md#guarded-reselection).

## Estimand-informed ontology search

**What it does.** For candidate variables whose measurement definition could
plausibly be improved, the interpretation LLM proposes alternative definitions,
and the alternatives are kept only if they measurably improve adjustment or
effect modeling on inner folds. For example, a generic "performance status"
variable might gain an alternative that records a named scale within a time
window.

1. Reference models are fitted on each inner fold using all original
   candidates: cross-fitted ridge [nuisance models](glossary.md#nuisance-models)
   (propensity, outcome, and, for binary outcomes, arm-specific outcome
   models) and an effect probe on propensity-eligible patients. The effect
   probe is a ridge-penalized [DINA](glossary.md#dina) model for binary
   outcomes and a ridge R-learner for continuous outcomes. The propensity
   bounds are `stage2.min_propensity` and `stage2.max_propensity`, or 0.1 and
   0.9 when those are unset.
2. Each candidate gets a confounder score (product of its propensity and
   outcome coefficient norms) and a modifier score (its effect coefficient
   norm). The top `max_features_per_role` candidates for each role that are
   observed in at least `minimum_nonmissing_fraction` of patients form the
   shortlist. Investigator-specified features are excluded.
3. For each shortlisted candidate, the interpretation LLM receives the study
   question, the current definition, and aggregate summaries of its extracted
   values, and proposes up to `max_alternatives_per_feature` complete
   alternative definitions. The prompt states that treatment and outcome are
   unavailable, and no patient-level data are included.
4. All alternatives are extracted for the outer-training patients.
5. Each alternative is scored on the same inner folds against fixed reference
   nuisances. As a confounder, it must improve outcome prediction without
   worsening propensity loss by more than `maximum_propensity_loss_increase`
   or reducing the overlap fraction by more than
   `maximum_overlap_fraction_drop`. As a modifier, it must reduce the effect
   probe's validation loss (DINA loss for binary outcomes, R-loss for
   continuous outcomes). In both cases the mean relative gain must exceed both
   `minimum_relative_gain` and `paired_se_multiplier` standard errors, and the
   alternative must win in at least `minimum_winning_fold_fraction` of inner
   folds.
6. At most one winning alternative per role is added for each original
   candidate; the original is kept. Added alternatives pass through the
   informativeness filter, consolidation, and selection like any other
   candidate, and no role is assigned at this step.

**Outcome information.** Treatment and outcome data drive both the shortlist
and the acceptance decision. The LLM never sees them, but the definitions that
survive are chosen partly because they improve outcome-based losses on the
outer-training patients. This is a supervised, data-adaptive step. The inner
folds used here are also used later for selection evidence and, under
`balanced_evidence`, the modifier-budget comparison, so those inner-fold
comparisons are optimistic for
runs that use the search; only the outer held-out predictions evaluate the
whole procedure honestly.

**When to use it.** When candidate definitions are coarse or ambiguous (time
window, named scale versus generic wording, numeric versus categorized value)
and a better-specified definition could plausibly improve adjustment or the
detection of heterogeneity.

**Configuration** (`stage2.estimand_ontology`):

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `false` | Run the search |
| `max_features_per_role` | 8 | Shortlist size per role |
| `max_alternatives_per_feature` | 2 | Alternatives proposed per shortlisted candidate |
| `minimum_nonmissing_fraction` | 0.2 | Minimum observed fraction for originals and alternatives |
| `minimum_relative_gain` | 0.005 | Minimum mean relative loss reduction |
| `paired_se_multiplier` | 1.0 | Mean gain must also exceed this many standard errors |
| `minimum_winning_fold_fraction` | 0.6 | Fraction of inner folds in which the alternative must improve |
| `maximum_propensity_loss_increase` | 0.01 | Allowed relative increase in propensity loss for a confounder alternative |
| `maximum_overlap_fraction_drop` | 0.05 | Allowed drop in the propensity-eligible fraction |
| `ridge_penalty` | 10.0 | Ridge penalty for the reference and probe models |
| `nuisance_crossfit_folds` | 3 | Cross-fitting folds for reference nuisances |
| `minimum_training_rows` | 20 | Minimum training rows per inner fold |
| `minimum_overlap_rows` | 10 | Minimum propensity-eligible rows for the effect probe |
| `max_prompt_chars` | 40000 | Character limit per proposal prompt; longer prompts skip the candidate |

**Caveats.**

- Cost per outer fold: up to `2 × max_features_per_role` proposal requests
  (16 by default), extraction of up to `max_alternatives_per_feature` times the
  shortlist size alternative definitions for every outer-training patient, and
  held-out extraction of accepted alternatives.
- An invalid proposal after all repairs keeps the original definition.
- It cannot be combined with `stage2.runtime_disable_extraction`.
- Outputs per fold: `estimand_ontology/report.json` (shortlist, proposals,
  scores, and decisions), `estimand_ontology/runs/` (checkpoints and the
  alternative extractions), `extraction/estimand_candidate_missingness_filter.json`,
  and `extraction/estimand_candidates_fit/`. The report is also summarized in
  `selection/elastic_net_selection.json` and `final_definitions.json`.

## Modifier-concept review

**What it does.** Before the modifier-budget comparison, the interpretation LLM
groups the top-ranked effect-modifier candidates into clinical concepts and
nominates representative measurements, so that several redundant measurements
of one concept do not use up the modifier budget.

Concept review works only with the `balanced_evidence` modifier rule, whose
nested per-fold rankings it groups. Enabling it requires
`multi_model.modifier_selection.rule: "balanced_evidence"`; with the default
`orthogonal_tests` rule the configuration is rejected.

1. For each evidence fold (inner fold × stability partition), Python ranks the
   modifier candidates and takes the top `concept_top_n_per_fold`, then counts
   how often each candidate recurs across those lists.
2. One request sends the union to the LLM. For each candidate the prompt
   contains its definition (truncated by the `stage2.role_adjudication`
   limits), its recurrence count, and aggregate effect-evidence summaries
   (counts of supporting and evaluated analyses, mean score, median p- and
   q-values per evidence family, and matched-batch results when enabled),
   together with the study question, outcome type, and propensity bounds. It
   contains no patient-level values, identifiers, held-out data, or oracle
   columns.
3. The LLM assigns every candidate to exactly one concept, labels each concept
   `retain`, `uncertain`, or `exclude`, and names representatives for retained
   and uncertain concepts.
4. The representatives form the modifier pool, re-ranked deterministically.
   Candidates that were not in any top list, were not chosen as
   representatives, or belong to excluded concepts cannot become effect
   modifiers.

The review runs inside each modifier-budget validation split, using evidence
computed within that split, and once more on all outer-training evidence for
the final choice.

**When to use it.** When many correlated or redundant modifier candidates reach
the shortlist and one representative per clinical concept is preferable.

**Configuration** (under `stage2.statistical_selection.multi_model`):

| Key | Default | Meaning |
|---|---|---|
| `modifier_count.concept_review` | `false` | Run the review |
| `modifier_count.concept_top_n_per_fold` | 100 | Top candidates per evidence fold given to the review |
| `max_prompt_chars` | 100000 | Character limit for the review prompt; longer prompts fail |

**Caveats.**

- It changes only the effect-modifier pool. Nuisance (confounder) routing and
  investigator-specified modifiers are unchanged, and it cannot create
  measurements.
- Cost: one request per inner fold (the modifier-budget validation splits)
  plus one for the final choice, so six per outer fold with five inner folds,
  plus repairs. A response that remains invalid after all repairs stops the
  fold.
- With concept review on, fewer representatives than a requested budget may be
  available; the selection report records the counts actually used.
- Outputs: `selection/modifier_count/<fingerprint>/.../concepts/` (prompt,
  response, and `concepts.json`) and the `concept_review` fields of
  `selection/modifier_count/selection.json`.

## Matched-batch contrast evidence

**What it does.** Adds an evidence family, `matched_batch_contrast`, that asks
whether a candidate predicts how much the observed treatment contrast in small
matched batches of patients deviates from the contrast expected in their
nuisance stratum.

1. Patients eligible under the propensity bounds are binned by quantiles of
   their cross-fitted propensity and outcome predictions
   (`bins_per_nuisance` bins each). A bin needs at least `min_reference_arm`
   patients per arm; its reference contrast is the difference in mean outcome
   between arms.
2. Repeatedly (`train_passes`), batches of `batch_size / 2` treated and
   `batch_size / 2` comparator patients are drawn within bins. Batches whose
   largest standardized mean difference in the two nuisance predictions
   exceeds `max_smd` are discarded. The target is the batch contrast minus the
   bin reference. A batch is a tail batch when its absolute deviation is at
   least `min_delta` and at least `z_threshold` times its approximate standard
   error.
3. For each candidate, a ridge model with bin intercepts predicts the target
   from batch-mean candidate values, fitted on all batches and on tail batches
   only (`filter_training_batches` chooses which is primary).
4. On validation patients, batches are drawn the same way
   (`validation_passes`) without tail filtering, and the candidate's score is
   the relative reduction in squared error compared with bin intercepts alone.

**How it is used.** Matched-batch results are recorded in
`selection/statistical_evidence.json` and are included in the
[modifier-concept review](#modifier-concept-review) prompt. The deterministic modifier
ranking, nuisance selection, and modifier-budget rule do not read this family,
so without concept review it is reported for audit and does not change the
selected variables.

**When to use it.** As an additional, model-light view of effect heterogeneity,
mainly together with concept review or for auditing selection.

**Configuration** (`stage2.statistical_selection.multi_model.matched_batch`):

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `false` | Compute the family |
| `batch_size` | 8 | Patients per batch (even, at least 4) |
| `bins_per_nuisance` | 2 | Quantile bins per nuisance prediction |
| `min_reference_arm` | 8 | Minimum patients per arm in a bin (at least half the batch size) |
| `max_smd` | 0.25 | Maximum standardized mean difference of nuisance predictions within a batch |
| `min_delta` | 0.10 | Minimum absolute deviation of a tail batch, in outcome units |
| `z_threshold` | 1.0 | Minimum deviation of a tail batch in standard errors |
| `filter_training_batches` | `true` | Use the tail-batch model as the primary score |
| `train_passes` | 100 | Training batch passes |
| `validation_passes` | 50 | Validation batch passes |
| `ridge_penalty` | 0.01 | Ridge penalty |

**Caveats.** It uses CPU only and makes no LLM requests, but it is computed in
every evidence resample and, under `balanced_evidence`, again within each
modifier-budget validation split.
Enabling it changes the evidence checkpoint identity, so all multi-model
evidence is recomputed.

**Diagnostic families.** `stage2.statistical_selection.multi_model.diagnostic_families`
(default `[]`) can add `penalized_main` and `predictive_forest`. These produce
treatment and outcome prediction evidence that is written to
`statistical_evidence.json` only; no selection step reads them.

## Pre-selection measurement consolidation

**What it does.** After extraction and before selection, this step
(`stage2.selection_consolidation`) merges
extracted variables that measure the same thing (for example, two fields that
both record ECOG performance status), so that selection evidence is not split
between duplicates. Merged variables are represented as derived variables
(called latent variables in the outputs) computed from the originals.

1. The candidates are visited once in order. For each candidate that is still
   active and not protected, the `neighbor_count` most similar active
   candidates are retrieved by cosine similarity of sentence embeddings of
   their names, descriptions, and measurement rules.
2. Pairwise associations among those candidates are computed on
   outer-training values: absolute Spearman correlation between continuous
   variables, bias-corrected Cramér's V between categorical variables, and the
   correlation ratio for mixed pairs, plus missingness overlap and paired
   agreement.
3. One interpretation-LLM request receives the definitions and pairwise
   comparisons and answers `keep` or `merge`. A merge defines the derived
   variable as either the first available source value (coalesce) or a
   lossless recoding of source categories.
4. A merge is accepted only if every check passes:
   - at least two unprotected sources from the current cluster, including the
     candidate being visited;
   - all sources share one value type, and continuous sources share one unit
     and are coalesced, never recoded;
   - categorical coalescing requires identical categories (and order, for
     ordinal variables); a recoding maps every source category exactly once
     and reaches every output category;
   - every source pair has at least `minimum_pairwise_complete_rows` jointly
     observed patients and association of at least
     `minimum_pairwise_association`;
   - sources do not conflict where both are observed;
   - the derived variable is observed in at least `latent_min_coverage` of
     patients and is not constant.
   A response that fails these checks is repaired; after repeated identical
   errors or exhausted repairs the step keeps the variables unchanged.
5. Merged sources leave the active set (their extracted values are kept), and
   the derived variable joins the pool for later comparisons. Held-out values
   of derived variables are computed from the original measurements.

**When to use it.** When extraction produces aliases or duplicates of the same
measurement. The example configuration enables it.

**Configuration** (`stage2.selection_consolidation`):

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `false` | Run the step |
| `neighbor_count` | 10 | Similar candidates compared with each visited candidate |
| `embedding_model`, `embedding_device` | `Qwen/Qwen3-Embedding-0.6B`, `cpu` | Sentence embedding model for retrieval (`auto` device allowed) |
| `max_latents_per_cluster` | 2 | Maximum derived variables per merge decision |
| `minimum_pairwise_complete_rows` | 10 | Minimum jointly observed patients per source pair |
| `minimum_pairwise_association` | 0.85 | Minimum association per source pair |
| `categorical_rare_level_min_count` | 5 | Rare categorical levels are pooled when computing associations |
| `latent_min_coverage` | 0.05 | Minimum observed fraction of a derived variable |
| `protect_explicit_features` | `true` | Never merge investigator-specified features |
| `max_prompt_chars` | 100000 | Character limit per decision prompt; longer prompts fail |
| `runtime_llm_endpoint`, `runtime_llm_model`, `runtime_llm_api_key` | empty, empty, `EMPTY` | Optional separate server for these requests; set endpoint and model together. Not part of checkpoint identity |

**Caveats.**

- Treatment and outcome are never used.
- Cost: up to one sequential request per candidate per outer fold, plus
  repairs, and loading the embedding model.
- Outputs per fold: `selection/candidate_consolidation/` (`report.json`,
  `registry.json` of derived variables, and one `steps/step_NNNN_*` directory
  per decision), `selection/nuisance_candidates.json`, and
  `selection/selected_latent_states.json`.

## Standalone explicit-feature workflows (`oci run`)

`oci run` is a separate command for analyses in which the variables are
specified by the investigator or proposed by an LLM agent and then measured
with an LLM, followed by a causal forest. It does not use the Stage 1
architectures or the Stage 2 pipeline above.

```bash
oci init --output oci_config.json          # write a configuration with every default
oci run --config example_configs/agentic_explicit_feature_forest_config.json
```

### Model types

Set `applied_inference.architecture.model_type`:

| Model type | Where the variables come from |
|---|---|
| `explicit_feature_forest` | The investigator lists them in `applied_inference.explicit_features.features` (requires `enabled: true` and at least one feature). The LLM extracts them and a forest is fit per outer split |
| `agentic_explicit_feature_forest` | An LLM agent proposes up to `broad_candidate_count` candidates from training-fold notes in each outer fold. The union is extracted once; then, for up to `max_iterations` rounds, candidates are screened, the agent proposes additions, removals, and role changes, and a change is accepted only if inner-fold R-loss improves by at least `min_r_loss_improvement`, treatment and outcome AUROC drop by no more than the allowed amounts, and enough folds improve. Listed explicit features seed the search when `explicit_features.enabled` is true |
| `agentic_attention_variable_forest` | Per outer split, a cross-fitted HTR nuisance network is trained; the agent proposes confounders from its attention evidence, candidates are extracted and filtered for coverage and association, and the same is repeated for effect modifiers using an effect network trained with the R-loss. Listed explicit features are always included as locked features |
| `multi_model_agentic_forest` | Per outer split, evidence from BoW/TF-IDF views, HTR attention, and embedding contrasts is summarized for the agent, which proposes variables; after extraction, a coverage filter, an extracted-feature review (on by default), an optional parsimony review, and a signal review run before the final forest. Listed explicit features and any `prespecified_*` features are included |

Every type fits EconML's `CausalForestDML` per outer split with settings from
`applied_inference.architecture.explicit_feature_forest` (defaults: 200 trees,
minimum leaf size 10, `max_features: "sqrt"`, honest splitting, and
confidence intervals). Confounder-role variables go to the forest's W,
effect-modifier-role variables go to X, and a variable with both roles goes to
both ([causal forest X versus W](glossary.md#causal-forest-x-versus-w)). With
no modifiers, X is a constant and the forest estimates one effect per fold.
The forest's internal nuisance models are cross-validated elastic nets.

A configuration that omits `model_type` defaults to `multi_model_forest`,
which `oci run` rejects at run time because that model belongs to the
all-evidence workflow; always set `model_type`. Unknown keys inside
`applied_inference` and its blocks are errors.

Main settings blocks (all under `applied_inference.architecture` except
`explicit_features`; `oci init` writes every key with its default):

| Block | Used by | Notable defaults |
|---|---|---|
| `explicit_feature_forest` | all | `n_estimators` 200, `min_samples_leaf` 10, `honest` true, `inference` true |
| `agentic_feature_search` | agentic types (agent endpoint for all three) | `outer_folds` 5, `inner_folds` 3, `max_iterations` 3, `broad_candidate_count` 80, `min_r_loss_improvement` 0.01, `agent_server_url` `http://localhost:8000/v1`, `agent_temperature` 0, `agent_max_tokens` 25000. `clinical_text_examples_per_prompt: null` sends every non-empty training-fold note in full |
| `agentic_attention_variable_forest` | `agentic_attention_variable_forest` | 5 nuisance and 5 effect folds, 30 epochs each, `min_extraction_coverage` 0.10, `effect_objective` `auto` (DINA for binary outcomes) |
| `multi_model_agentic_forest` | `multi_model_agentic_forest` | `candidate_proposals_per_fold` 60, `extracted_feature_review_enabled` true, `parsimony_review_enabled` false, `fail_on_extraction_truncation` true |
| `explicit_features` (under `applied_inference`) | all | Feature list and extraction settings (below) |

For `agentic_explicit_feature_forest`, outer folds come from
`agentic_feature_search.outer_folds` and the experiment seed; `cv_folds` and
the split column do not affect its splits.

### Feature specifications

Each entry of `explicit_features.features` (or of the command-line feature
options) is a JSON object:

| Field | Required | Meaning |
|---|---|---|
| `name` | yes | Variable name; becomes columns `explicit_feat_<name>` and `explicit_feat_<name>_missing` |
| `type` | yes | `categorical` or `continuous` |
| `categories` | for categorical | Allowed values; the first is the reference level |
| `description` | no | Shown to the extraction LLM |
| `roles` | yes | `confounder`, `effect_modifier`, or both |
| `value_aliases` | no | For categorical variables, `{category: [alias, ...]}` |
| `temporal_rule`, `aggregation_rule` | no | Instructions used by `complete_paged_v1` extraction |

The list may also be given as `{"features": [...]}` or as
`{"confounders": [...], "effect_modifiers": [...]}`, where the section adds the
role. Categorical variables are encoded as k − 1 indicators plus a missing
flag; values that match no category or alias are treated as missing, never as
the reference level. Continuous variables are standardized with training-fold
statistics plus a missing flag.

### Extraction strategies

`explicit_features.extraction_context_strategy` decides what part of each note
the extraction LLM sees:

| Strategy | Behavior | Constraints |
|---|---|---|
| `full_record` (default) | The whole note | If `extraction_max_text_length` is set and a note is longer, the run stops; nothing is truncated. The default length limit is none |
| `complete_paged_v1` | Lossless paging: the note is split into pages whose cores cover every character exactly once, with surrounding context. Each page returns a status (positive, negative, missing, ambiguous) with exact character-offset citations that are validated against the page, and page results are reconciled in fan-in steps that must account for every page | Requires `complete_page_core_chars`, `complete_page_context_chars`, `complete_page_max_chars`, and `complete_reconciliation_fan_in` (at least 2); `extraction_max_text_length` must be unset or equal to `complete_page_max_chars`. One variable per request. Server modes only. Any status other than positive is stored as missing |
| `contract_lexical_rag` | Deterministic lexical retrieval of note passages that match the feature names, descriptions, categories, and aliases, kept verbatim and in source order; falls back to the end of the note when nothing matches | Requires `extraction_max_text_length` of at least 256 |
| `tail` | The last `extraction_max_text_length` characters | Truncates earlier text silently; use only when that is acceptable |

`multi_model_agentic_forest` stops if any note exceeds
`extraction_max_text_length` while `fail_on_extraction_truncation` is true (the
default), so with that model type use `full_record`, or `complete_paged_v1`
with no length limit.

Other extraction settings: `extraction_grouping_strategy` (`clinical_domain`
or `packed`), `max_variables_per_extraction_request` (1 to 10, default 10),
`extraction_batch_size` (32), `extraction_max_retries` (3),
`extraction_request_timeout` (900 s), `extraction_temperature` (0),
`extraction_max_tokens` (25000), and `source_text_temporally_valid_by_design`
(`false`; set `true` when the text is pretreatment by construction, which
changes the missing-value instructions).

**LLM backend.** `vllm_mode` is `server` (default; use an OpenAI-compatible
server at `vllm_server_url`, which may be a list used in rotation),
`start_server` (start a vLLM server subprocess), or `python_api` (in-process
vLLM). `vllm_model_name` set to `auto` or empty discovers the model from
`/v1/models` in server mode. The agent uses `agentic_feature_search.agent_*`
settings for its endpoint. For Google Vertex endpoints, install the
`extraction` extra, set `OCI_STAGE2_ENDPOINT_TRANSPORT=google_vertex`, and use
`api_key: "GOOGLE_ADC"`; `api_key: "env:VARIABLE"` reads a key from the
environment.

**Caching.** With `cache_enabled` (default), extraction results are cached
under `<cache_dir or the model type's artifact directory>/.oci_cache/`, keyed
by the dataset path and a hash of the feature definitions, prompts, model,
endpoints, and extraction settings. Per-patient results are keyed by the note
text, so they are reused across folds. Roles are not part of the key.

### Data and splits

The dataset is Parquet or CSV with `applied_inference.text_column`,
`treatment_column`, and `outcome_column` (defaults `clinical_text`,
`treatment_indicator`, `outcome_indicator`) and `outcome_type` `binary` or
`continuous`. With `cv_folds` above 1, outer folds are stratified on treatment
and outcome. With `cv_folds` of 1 or less, the `split_column` (default `split`)
supplies `train`/`val` rows for fitting and `test` rows for evaluation; a
configuration without an honest held-out split stops unless
`explicit_feature_forest.allow_non_honest_outer_split` is true. Columns whose
names start with `true_`, `oracle_`, or `ground_truth` are removed before any
model sees the data.

### Command-line options

| Option | Applies to | Meaning |
|---|---|---|
| `--config`, `-c` | all | Configuration file (required) |
| `--device`, `--workers`, `--gpu-ids`, `--output-dir` | all | Override the configuration's device, worker count, GPU IDs, and output directory |
| `--alpha-propensity` | all | Weight of the propensity loss in HTR training |
| `--verbose`, `-v` | all | Debug logging |
| `--multi-model-features-json`, `--multi-model-confounder`, `--multi-model-effect-modifier`, `--multi-model-feature-discovery-methods` | `multi_model_agentic_forest` | Prespecified features (JSON) and evidence sources (`bow`, `htr`, `embedding_contrast`, or `all`) |
| `--r-stage-min-propensity`, `--r-stage-max-propensity`, `--inner-fold-parallelism`, `--outer-fold-parallelism`, `--agent-candidate-parallelism`, `--effect-objective` | `agentic_attention_variable_forest` | Effect-stage propensity bounds, parallelism, and effect objective (`auto`, `dina`, `squared_r_loss`, `pseudo_outcome_mse`; binary outcomes accept `auto` or `dina`) |

A model-specific option used with another model type is an error.

### Outputs

`oci run` writes into its output directory:

| Path | Content |
|---|---|
| `config.json`, `summary.json` | Resolved configuration and run summary |
| `applied_inference/predictions.parquet` | One row per patient: input columns (including the text), extracted features, `pred_ite_prob` (the forest's CATE on the outcome scale), `pred_dr_score` (per-patient AIPW score), `pred_ite_lower`/`pred_ite_upper`/`pred_ite_stderr`, `pred_y0_prob`/`pred_y1_prob` derived from the CATE, outer fold, and selected features |
| `applied_inference/ate_summary.json` | Cross-fitted AIPW ATE: the mean of held-out `pred_dr_score`, with standard error, normal 95% CI, `n`, mean predicted CATE, and whether the outer split was honest |
| `applied_inference/posthoc_oracle_metrics.json` | Only when the dataset has oracle columns: computed after predictions are frozen and hashed; ITE error and correlation (against `true_ite`, or `true_ite_prob` for binary outcomes), ATE bias, propensity agreement when `true_treatment_prob` is present, and per-fold metrics |
| Per-type directory under `applied_inference/` | Search logs, agent decisions, evidence, fit audits, and checkpoints of the agentic types |

`pred_propensity_prob` and `pred_outcome_prob` in the agentic types' predictions
come from separate diagnostic random forests, not from the causal forest's
nuisance models. `ate_summary.json` is not written when the predictions have no
`pred_dr_score` (the `neural_only` mode of `agentic_attention_variable_forest`).
