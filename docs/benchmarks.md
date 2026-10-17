# Benchmarks

The benchmark harness measures how well OCI recovers known treatment effects
and data-generating roles across a set of synthetic scenarios. It builds the
scenario cohorts, writes one workflow configuration per cohort, and compares
the finished runs with the [oracle columns](glossary.md#oracle-columns). These
three commands never start a workflow run or an LLM request: you launch the
runs yourself with the printed commands, or `run_benchmark.sh` runs every
step.

```bash
python scripts/benchmark.py materialize --scenarios S0,S1 --source SOURCE_DIR --out BENCH
python scripts/benchmark.py configs --base CONFIG.json --out BENCH
python scripts/benchmark.py aggregate --out BENCH
```

The code is `oci/evaluation/benchmark.py` (registry, seeds, commands, and
metrics) and `oci/evaluation/benchmark_ceiling.py` (the oracle-covariate
ceiling). `run_benchmark.sh` runs every step on one machine, including the LLM
text and the runs ([End-to-end driver](#end-to-end-driver)).

## Contents

- [Design](#design)
- [Scenario registry](#scenario-registry)
- [Source corpora](#source-corpora)
- [Seeds and the confirmatory rule](#seeds-and-the-confirmatory-rule)
- [Commands](#commands)
- [End-to-end driver](#end-to-end-driver)
- [Metrics](#metrics)
- [Outputs](#outputs)
- [Oracle safety](#oracle-safety)
- [Caveats](#caveats)

## Design

Generated notes depend only on patient characteristics and never mention
treatment or outcome. One text corpus can therefore serve many treatment and
outcome equations: a *re-equation* keeps every note and recomputes treatment,
outcome, and the true effects in Python from a scenario's data-generating
process (DGP). Factors that need different text, such as which characteristics
the note writer is shown, are switches at generation time and need a generated
corpus.

A *cell* is one scenario, one sample size, and one seed. Each cell gets its own
cohort and one workflow run per run variant (two for S6, one otherwise):

```text
BENCH/
├── <scenario>/n<size>/seed<k>/
│   ├── data/                      # the cell's cohort (dataset.parquet, metadata.json, ...)
│   ├── materialize.json           # how the cohort was made, with input and output hashes
│   ├── runs/<variant>/config.json # the workflow configuration
│   ├── runs/<variant>/output/     # the workflow output directory (you run it)
│   └── oracle_ceiling/            # the oracle-covariate refits (written by aggregate)
├── <scenario>/generation/         # generator settings and corpus for S5 and S6
├── configs.json                   # every configuration written, with its command
└── aggregate/                     # cells.csv, scenarios.csv, summary.json, summary.md
```

## Scenario registry

`benchmarks/scenarios.json` lists a baseline and six factor families. Each
scenario changes one factor family relative to S0; the roles count source
features, and features without a role still appear in the text.

| Scenario | Factor | DGP relative to S0 | Sizes | Corpus |
|---|---|---|---|---|
| S0 | Baseline: confounders and effect modifiers only | 3 confounders, 3 effect modifiers, no modifier main effects, linear terms | 1,000, 5,000, 10,000 | Re-equation source, at least 10,000 patients |
| S1 | Instruments and prognostic-only variables | adds 2 instruments (treatment equation only) and 2 prognostic variables (outcome equation only) | 1,000, 5,000 | Re-equation source, at least 5,000 patients and 10 features |
| S2 | Dual-role variables and modifier main effects | adds 2 variables that are both confounders and effect modifiers; every effect modifier gets an outcome main effect | 1,000, 5,000 | Re-equation source, at least 5,000 patients |
| S3 | Nonlinear h(M) | `nonlinearity: "threshold"` for continuous features in the outcome main terms and the treatment interaction terms | 1,000, 5,000 | Re-equation source, at least 5,000 patients |
| S4a, S4b | One unmeasured confounder U | one latent standard-normal U in both equations, at `unmeasured_strength` 0.5 (S4a) and 1.0 (S4b); U never appears in the text | 1,000, 5,000, 10,000 | Re-equation source, at least 10,000 patients |
| S5 | MNAR documentation | the S0 DGP, re-equated on a corpus whose note writer saw each characteristic with a value-dependent probability | 1,000, 5,000 | Generated corpus, at least 5,000 patients |
| S6 | Post-treatment leak | the generator's own DGP; notes dated after the index date describe a treatment-caused event | 1,000, 5,000 | Generated corpus, at least 5,000 patients |

Every re-equation DGP pins all of its settings (role counts with random seeded
assignment, logit standard deviations of 2.0, a treatment effect of 0.2,
modifier strength 2.0, target rates of 0.5, outcome noise 1.0), so a change in
the generator's defaults cannot change a scenario. The keys follow the
generator's scenario JSON, described in [Synthetic data](synthetic_data.md).
`--scenarios` accepts scenario ids or families: `S4` selects S4a and S4b.

S6 has two run variants on the same cohort:

| Variant | Text input |
|---|---|
| `text_column` | `columns.text = clinical_text_with_post_treatment`, so the post-index notes reach the analysis |
| `dated_notes` | [dated-notes input](configuration.md#notes) from the cohort's `notes.parquet` with `index_date_column: index_date`; notes dated on or after the index date are dropped, and the derived text column is `dated_notes_text` |

## Source corpora

**Re-equation sources (S0 to S4).** A source directory holds `dataset.parquet`,
`metadata.json`, and `generation_config.json`, has a `true_<variable>` column for
every metadata feature, and has no post-treatment notes (re-equation refuses
both kinds of source). A cell larger than the source is skipped with a message,
so a small source can serve the small cells.

The bundled
`synthetic_data/example_synthetic_datasets/five_confounders_five_effect_modifiers_nsclc_with_structured/`
cohort (1,000 patients, ten features with `true_<variable>` columns) serves the
1,000-patient cells. For 5,000 and 10,000 patients, expand it with the saved DGP
(see [Expanding a cohort from a saved DGP](synthetic_data.md#expanding-a-cohort-from-a-saved-dgp)),
which writes LLM text for the additional patients. In the bundled cohort,
`prior_platinum_therapy` never takes the level "First-line": the variable's
definition spells that level with a non-breaking hyphen (U+2011), its
distribution spells it with an ASCII hyphen, and the cohort's patients were
drawn without matching the two. The generator in this repository matches the
spellings, so patients of an expansion do take the level (with the
definition's U+2011 spelling), and the bundled cohort and its expansions differ
in that variable's levels. Re-equated cells inherit the levels of their source.

**Generated corpora (S5 and S6).** Without `--generated`, `materialize` writes
the scenario's generator settings to `BENCH/<scenario>/generation/generator_config.json`
and prints two commands: the generator run (`python -m synthetic_data.cli --config
... --api-url URL --model MODEL --output-dir BENCH/<scenario>/generation/corpus`)
and the `materialize --generated` call that uses its output. The settings
generate 5,000 patients: S5 sets `documentation_rate` 0.7 with the `mnar`
mechanism (`documentation_mnar_slope` 1.0), so whether the note writer sees a
characteristic depends on its value; S6 generates six features (three
confounders and three effect modifiers) and sets `post_treatment_notes` with
rate 1.0. Edit `benchmarks/scenarios.json` to change them. The registry validates
generation settings before any corpus is generated. With `--generated DIR`, S5 cells are re-equated from
the corpus with the S0 DGP, and S6 cells are seeded subsamples without
replacement of its patients and their rows of `notes.parquet`, because
re-equation refuses corpora with post-treatment notes; S6 keeps the generated
treatment and outcome. `--generated` applies to one generated-corpus scenario
per call.

## Seeds and the confirmatory rule

`benchmarks/seeds.json` lists three development seeds (101, 102, 103) and three
confirmatory seeds (7001, 7002, 7003). A cell's seed drives the re-equation (the
subsample and the redraw of treatment and outcome) and is the run's
`science.seed` (folds and models).

**Tune only on development seeds.** Choose settings, compare variants, and
debug with development seeds; use the confirmatory seeds once, for the
version you report.

- Every command uses the development seeds by default. A confirmatory seed,
  passed with `--seeds 7001` or `--seeds confirmatory`, is refused unless
  `--confirmatory` is given; `--confirmatory` alone selects the confirmatory
  seeds.
- `configs --confirmatory` is the confirmatory use. Before writing anything it
  reads `benchmarks/confirmatory_ledger.jsonl` (`--ledger` names another file)
  and refuses a scenario that already has an entry for the same git commit,
  unless `--force` is given. It then appends one entry per scenario: the time,
  the scenario, the git commit and whether tracked files differed from it, a
  configuration hash (SHA-256 over every cell, variant, and configuration-file
  hash of the scenario's confirmatory runs), the seeds, the cells, the base
  configuration and its hash, the benchmark directory, and whether `--force`
  was used.
- `aggregate` reports development cells, and confirmatory cells only with
  `--confirmatory`. The two seed sets are summarized separately.

## Commands

All three commands take `--out BENCH`, `--scenarios`, `--sizes`, `--seeds`,
`--confirmatory`, `--registry` (default `benchmarks/scenarios.json`), and
`--seeds-file` (default `benchmarks/seeds.json`).

**`materialize`** writes `BENCH/<scenario>/n<size>/seed<k>/data/` for every
selected cell.

| Option | Meaning |
|---|---|
| `--source DIR` | Re-equation source for S0 to S4 |
| `--generated DIR` | Generated corpus for one S5- or S6-type scenario |

Re-equation goes through the generator's
`synthetic_data.reequation.reequation_cohort(source_dir, scenario, size=,
seed=, output_dir=)`, whose output directory holds `dataset.parquet`,
`metadata.json`, `reequation_manifest.json`, and `generation_config.json`. The
harness records the source files' SHA-256 hashes, the DGP and its hash, the
size, the seed, and the output hashes in `materialize.json`. Repeating the
command leaves a matching cell untouched; a cell made differently, or whose
files changed, stops the command.

**`configs --base CONFIG.json`** writes `runs/<variant>/config.json` for every
materialized cell and prints one command per run:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/run_all_evidence.py --config BENCH/S0/n1000/seed101/runs/default/config.json
```

Each configuration is the base configuration (JSON or YAML, nested like
`example_configs/research_all_evidence_multi_model.json`, or a run's
`run_config.json`) with `dataset` set to the cell's `data/dataset.parquet`,
`output_dir` to `runs/<variant>/output`, `science.seed` to the cell's seed,
`science.outcome_type` to the cohort's outcome type, and the variant's
`columns` and `notes` settings. A `notes` block in the base is dropped. Relative
template and model paths in the base are resolved against its directory. A
configuration whose run has started is never replaced by a different one.
`BENCH/configs.json` lists every configuration, its hash, its output directory,
and its command.

**`aggregate`** reads every expected cell and run variant. A run is finished
when `stage2/complete.json` reports the causal-estimation phase and
`cross_fitted_predictions.csv`, `causal_estimate.json`, and
`heterogeneity_diagnostics.json` exist; every other cell is listed as pending
with a reason (`not_materialized`, `no_config`, `not_started`, the
`progress.json` status and component, or the Stage 2 phase reached).

| Option | Default | Meaning |
|---|---|---|
| `--recovery-r2` | 0.5 | Held-out R² at which a variable recovers an oracle concept |
| `--no-ceiling` | off | Skip the oracle-covariate ceiling refits |
| `--ceiling-workers N` | 1 | CPU processes for the ceiling refits |

## End-to-end driver

`run_benchmark.sh` (`oci/evaluation/benchmark_driver.py`) runs the benchmark
on one machine, named by a [hardware preset](serving.md#hardware-presets)
(`external`, `h100x8`, `rtxpro6000x8`, or `dgx_station_gb300`):

```bash
./run_benchmark.sh MACHINE BENCH [--scenarios ...] [--sizes ...] [--seeds ...] [OPTION ...]
./run_benchmark.sh rtxpro6000x8 /data/bench --scenarios S0 --sizes 1000 --seeds 101
./run_benchmark.sh dgx_station_gb300 /data/bench --plan
```

It takes the selection options of the harness commands (`--scenarios`,
`--sizes`, `--seeds`, `--confirmatory`, `--force`, `--registry`,
`--seeds-file`, `--ledger`) and runs these steps in order. Rerunning the same
command resumes: finished steps and runs are skipped, and an interrupted
generator or run continues from its checkpoints.

| Step | What it does |
|---|---|
| `generate` | Writes the LLM text the selected cells need: an expansion of the re-equation source with its saved DGP when a selected S0 to S4 cell is larger than the source, to `BENCH/corpora/<source>_n<size>/`, and the corpus of each selected S5- or S6-type scenario, to `BENCH/<scenario>/generation/corpus/` |
| `materialize` | One cohort per cell, from the source (or its expansion) and the corpora |
| `configs` | One workflow configuration per cell from `--base` (default `benchmarks/base_config.json`: the bundled cohorts' columns, clinical question, 5 × 5 folds, HTR and embedding models, and Stage 2 definition settings, as `run_synthetic.sh` uses them) |
| `runs` | Each unfinished run, one at a time, smallest cells first, as `PRESET=MACHINE ./run_synthetic.sh BENCH/.../config.json`; the launcher adds the preset's serving settings and the detected devices and workers. A failed run is reported and the next one starts (`--stop-on-failure` stops instead) |
| `aggregate` | The aggregate report in `BENCH/aggregate/` (`--no-ceiling`, `--ceiling-workers`) |

`--through STEP` stops after a step, for example `--through materialize` to
build the cohorts on a generation machine and run them elsewhere. `--plan`
prints the plan (models, corpora, cells, and every run with its status) and
writes nothing.

**The re-equation source.** `--source` (default: the bundled five-confounder
cohort) serves the S0 to S4 cells. When the largest selected cell is larger
than the source, the driver expands the source to that size with the saved
DGP; `--source-size N` expands to N patients up front. Every S0 to S4 cell of
a benchmark directory comes from one source, recorded in
`BENCH/driver/state.json`, so a later selection that needs a larger expansion
is refused: start a new directory, or give the largest size with
`--source-size` the first time.

**Corpus generation.** The generator uses `GENERATOR_MODEL` (default
`openai/gpt-oss-120b`, the model that wrote the bundled cohorts). On a managed
preset the driver starts a vLLM pool for it on the GPUs that `run_synthetic.sh`
would select (one server per GPU; ports from 8210, internal ports from 40000;
`--gpu-memory-utilization 0.90 --max-model-len 131072`) and stops it before
the first run. The generator spreads patients across the replicas with 32
concurrent patients per server. `GENERATOR_ENDPOINT` (comma-separated URLs of
running OpenAI-compatible servers) replaces the pool and is the default for
the `external` preset (`http://127.0.0.1:8000/v1`).

| Variable | Default | Meaning |
|---|---|---|
| `GENERATOR_MODEL` | `openai/gpt-oss-120b` | Model that writes the notes |
| `GENERATOR_ENDPOINT` | none (`http://127.0.0.1:8000/v1` for `external`) | Running servers instead of a managed pool |
| `GENERATOR_API_KEY` | empty | Key for `GENERATOR_ENDPOINT` |
| `GENERATOR_WORKERS` | 32 per server | Concurrent patients |
| `GENERATOR_VLLM_GPUS` | the detected GPUs | Logical GPUs of the pool |
| `GENERATOR_VLLM_GPUS_PER_SERVER` | 1 | Tensor-parallel width |
| `GENERATOR_VLLM_BASE_PORT`, `GENERATOR_VLLM_INTERNAL_PORT_BASE` | 8210, 40000 | Public and internal ports |
| `GENERATOR_VLLM_EXTRA_ARGS_JSON` | `["--gpu-memory-utilization","0.90","--max-model-len","131072"]` | vLLM arguments |

The runs honor every `run_synthetic.sh` variable (`STAGE2_MODEL`,
`STAGE2_ENDPOINT`, `STAGE2_WORKERS`, ...), and `OCI_PYTHON` selects the
interpreter instead of synchronizing `.venv`. The synchronization installs vLLM
(the `local-llm` extra) only when OCI starts its own servers, that is, for a
managed preset without `--llm-url`.

**Running vLLM servers (`--llm-url`).** `--llm-url URL[,URL...]` names running
OpenAI-compatible servers (for example `http://localhost:8000/v1`) that serve
one model. They replace the preset's managed servers for both Stage 2 roles and
for the generator, and the preset keeps only its hardware settings
(`PHYSICAL_GPUS`, `GPU_COUNT`, `MIN_FREE_GPU_GB`, `MIN_TOTAL_GPU_GB`,
`CUDA_DEVICE_ORDER`, `STAGE1_WORKERS`, `STAGE2_MODELING_WORKERS`, and the
request timeouts); each run is launched with `PRESET=external`. The driver
reads the model ID and `max_model_len` from each server's `/models` list
(every server must report the same model):

| Setting | Value |
|---|---|
| Model, both roles | The served model ID |
| Interpretation | The first server, with a quarter of `--llm-concurrency` (default 32) requests at a time |
| Extraction | Every server, with `--llm-concurrency` requests at a time each, less the interpretation share on the first |
| Extraction window | The smallest `max_model_len`, with a 4,096-token margin (the interpretation role reads the window from the server itself) |
| Generator | The same servers and model, `--llm-concurrency` patients per server (at most 1,000, the connection pool of the generator's client for each server), and an output ceiling of at most 50,000 tokens that leaves 16,384 tokens of the window for the prompt; `GENERATOR_ENDPOINT` names other servers instead |

Set `--llm-concurrency` to the number of requests a server can run at once,
which its KV cache limits, rather than to its `--max-num-seqs`: vLLM queues
requests beyond what fits, and a queued request spends its time limit waiting
(Stage 2 allows 1,800 seconds per attempt). Raise it while vLLM's log shows
few waiting requests and rising throughput. Every generator request (with or
without `--llm-url`) allows 1,800 seconds instead of the generator's default
300, for the same reason; this changes only routing, so an interrupted
generation still resumes. Both launchers raise the soft open-file limit to the
hard limit, because every concurrent request holds a socket.

Explicit `STAGE2_WORKERS`, `STAGE2_EXTRACTION_CONTEXT_WINDOW_TOKENS`, and
timeout variables still win; `STAGE2_ENDPOINT`, `STAGE2_MODEL`, the extraction
endpoint and model variables, and `STAGE2_VLLM_*` variables contradict
`--llm-url` and are refused. Text generated by another model than
`openai/gpt-oss-120b` differs from the bundled cohorts' text, so an expansion
or corpus written by it is not interchangeable with one written by that model.

**Outputs.** `BENCH/driver/` holds `invocations.jsonl` (one line per
invocation: time, machine, arguments, git state, and models), `state.json`,
the generator configurations (`generation/`), the generator servers' logs and
manifest (`generation_servers/`), and one log per generator run and workflow
run (`logs/`). Each run also writes its own `logs/workflow.log` and
`progress.json` in its output directory.

**Confirmatory runs.** With `--confirmatory`, the first invocation writes the
configurations and their ledger entries; a rerun finds every configuration in
place and resumes the runs without a second ledger entry.

## Metrics

All metrics use the outer held-out rows of the run's cross-fitted predictions,
so every patient enters once.

**ATE.** The co-primary AIPW estimate and its 95% CI against the true ATE over
the same rows: the mean true ITE (`true_ite`, or `true_ite_prob` for binary
cohorts without it) over the effect-eligible held-out rows. Per cell: the bias
(estimate minus truth) and whether the CI covers the truth. Across seeds: the
mean bias, the RMSE (square root of the mean squared bias), and the coverage
rate.

**ATO.** The augmented overlap-weighted estimate and its CI against the true
overlap-weighted effect Σ e(1−e)τ / Σ e(1−e), with e the **true** propensity
(`true_treatment_prob`). OCI weights patients by its estimated propensity ê, so
its estimand differs from this truth and the comparison is approximate; the
cells table also reports Σ ê(1−ê)τ / Σ ê(1−ê), the effect under OCI's own
weights, and its coverage.

**CATE.** The RMSE and Pearson and Spearman correlations of `estimated_cate`
with the true ITE; the constant-ATE baseline, which is the RMSE of predicting the run's
ATE estimate for every patient; and the relative improvement, 1 − RMSE /
baseline RMSE, which is positive when the CATEs beat the constant.

**GATES coverage.** The harness rebuilds each fold's
[GATES](glossary.md#blp-gates-clan-and-rate) groups (within-fold quintiles of
`estimated_cate`) with the diagnostics' own grouping function and seed, checks
that each rebuilt group reproduces the stored group mean, and reports the share
of GATES 95% intervals that contain the true mean ITE of the group's patients,
per fold and pooled.

**Oracle-covariate ceiling.** The same ATE, ATO, CATE, and GATES metrics for a
refit of the run's final estimation on the true covariates. The refit calls
Stage 2's own final-estimation function (`estimate_outer_fold`, the function
each outer fold calls after selection) with no LLM, and gives it:

- the run's frozen outer and inner folds from `components/tfidf/split_provenance.jsonl`;
- the run's fold seeds (`science.seed` + 100000 × outer fold, as in Stage 2) and
  inner-fold count, from `run_config.json`;
- its nuisance learner, statistical policy, propensity clip and bounds, and
  forest size, from `stage2/config.json`;
- each fold's recorded final estimator (the
  [estimator ensemble](stage2_selection.md#estimator-ensemble) by default);
- in place of the extracted variables, each patient's `true_<variable>` values.

The nuisance-candidate catalog holds every metadata feature with a role
(confounder, effect modifier, instrument, or prognostic); unassigned features,
proxies, and unmeasured confounders are left out. The selected variables carry
the roles a perfect selector would give them: `confounder` for true
confounders and prognostic-only variables, `effect_modifier` for true effect
modifiers (both for dual-role variables). Instruments enter the nuisance models
only. The ceiling shows what the estimation step achieves when discovery,
documentation, extraction, and selection are perfect.

**Role-wise selection.** From the attribution report's recovery by held-out R²
(see [Attribution and ablation](stage1_architectures.md#attribution-and-ablation)):

| Metric | Definition |
|---|---|
| Modifier recall | Share of true effect modifiers, dual-role ones included, that some selected modifier recovers |
| Modifier precision | Share of selected modifiers that recover a true effect modifier |
| Confounder recall | Share of true confounders recovered by a variable in the nuisance set |
| Instruments, prognostic-only variables | Counts recovered by the nuisance set and by the selected modifiers |
| U proxies | The largest held-out R² of each unmeasured confounder `true_<u>` on a selected variable, and whether any reaches the threshold |

Each is computed per outer fold and pooled as the sum of the numerators over the
sum of the denominators.

**Unmeasured confounding (S4).** The run's E-values, robustness value RV, and
RVa (see [Sensitivity to unmeasured confounding](stage2_selection.md#sensitivity-to-unmeasured-confounding))
beside U's true strength. U's strength comes from the oracle columns through
the DoubleML benchmark formulas already used for Stage 2 benchmarking
(`stage2_sensitivity.benchmark_gains`), with U as the omitted variable. The long
model is the true propensity and true potential outcomes, which include U. The
short model integrates U out given the measured covariates: P(D = 1 | X) and the
observational outcome means E[Y | D, X], by Gauss–Hermite quadrature, using U's
coefficients from `metadata.json` (or, when the metadata does not state them,
least squares of logit e and of the untreated outcome predictor on U and the
true covariates). This gives cf_y (U's partial R² with the outcome), cf_d (its
share of the Riesz representer's variation), and rho. The harness then
computes the run's omitted-variable-bias bound at (cf_y, cf_d) with ρ = 1, and
at U's own ρ, and reports whether the bound and its confidence bounds contain
the true ATE; the same check is reported for the oracle-covariate ceiling.

**Post-treatment leak (S6).** For the text-column and dated-notes variants of
each cell: the ATE bias, the share of held-out patients whose estimated
propensity lies outside [0.05, 0.95], and whether any selected variable
recovers a `true_post_treatment_*` column at the recovery threshold, with the
text-column minus dated-notes differences.

## Outputs

`aggregate` writes `BENCH/aggregate/`:

| File | Content |
|---|---|
| `cells.csv` | One row per cell and run variant: status and pending reason, then every metric above for the run and the ceiling |
| `scenarios.csv` | One row per scenario, variant, size, and seed set: cell counts, mean bias, RMSE across seeds, coverage rates, and the means of the other metrics over finished cells |
| `summary.json` | The registry and seed-file hashes, the metric definitions and caveats, every cell's full metrics (per-fold selection detail, GATES groups, sensitivity elements, and the SHA-256 of every run artifact read), the scenario table, the S6 comparison, and the pending and failed cells |
| `summary.md` | Readable tables of the scenarios, the S4 sensitivity checks, the S6 comparison, and the pending cells |

A run whose artifacts cannot be evaluated is reported with status `error` and
the reason; the other cells are still aggregated.

## Oracle safety

For each finished run, `aggregate` reads every run artifact it uses (the run
configuration, `stage2/config.json`, the cross-fitted predictions, the causal
estimate, the heterogeneity diagnostics, the split file, the final definitions,
and the held-out extraction) once through the attribution report's hashing
reader, records each file's SHA-256, and closes the reader before any oracle
column is loaded. It checks that the oracle dataset's treatment and outcome
match the run's predictions row by row and that the split file matches the
held-out rows. It writes nothing inside a run directory: the ceiling's refits
go to the cell's `oracle_ceiling/` directory, and the reports to
`BENCH/aggregate/`.

## Caveats

- The true ATE and group means are sample quantities over the held-out rows;
  the AIPW intervals target the population effect, so coverage of the sample
  truth is approximate.
- The ATO truth uses the true propensity while OCI weights by its estimate (see
  [Metrics](#metrics)).
- The oracle-covariate ceiling assumes perfect documentation, extraction, and
  selection; it is not an upper bound in every sample.
- Recovery means a held-out R² at or above the threshold; a variable can
  recover a concept without naming it, and a correlated variable can recover
  more than one concept.
- U's strength comes from the data-generating equations; the run's bound also
  absorbs measurement error in the text-derived confounders, which U's strength
  does not include.
- S6 cells at the corpus size (5,000 patients) contain the whole corpus for
  every seed, so their replicates share one data draw and differ only in folds
  and model seeds.
- Every run uses LLM text extraction, so runs on the same cohort and seed can
  still differ.
