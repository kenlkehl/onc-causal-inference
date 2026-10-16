# Synthetic data

OCI includes a generator for synthetic oncology cohorts with a known causal
structure, and two bundled 1,000-patient cohorts made with it. Because the true
treatment effect of every synthetic patient is known, these cohorts let you
check whether the workflow recovers the confounders and effect modifiers that
generated the data, and how close its estimated effects are to the truth. Terms
such as [confounder](glossary.md#confounder),
[effect modifier](glossary.md#effect-modifier), and
[oracle columns](glossary.md#oracle-columns) are defined in the
[glossary](glossary.md).

## How a cohort is generated

The generator (`python -m synthetic_data.cli`) asks an LLM, served through any
OpenAI-compatible chat-completions endpoint, to design a data-generating process
(DGP) for a clinical question and then to write each patient's record. All
probabilities and outcomes are computed in Python from that DGP, so the truth
is exact.

1. **Variables.** The LLM proposes patient characteristics for the clinical
   question. Each is continuous or categorical and carries one or both roles:
   `confounder` or `effect_modifier`. When you request role counts, the
   generator keeps the first proposed variables that satisfy them.
2. **Equations.** The LLM writes a treatment equation and an outcome equation
   over the confounders. Continuous terms are standardized
   (`(x − mean) / sd`), and categorical terms are 0/1 indicators for every level
   except the first listed level, which is the reference. The LLM's main
   coefficients are scaled by `main_coefficient_scale` (0.3) and its
   interactions by `interaction_coefficient_scale` (0.1). Python then adds:
   - the treatment main effect, `--treatment-effect`;
   - a treatment × term interaction for every effect-modifier term, with a
     coefficient drawn uniformly from (−1, 1) and scaled by 0.1;
   - all pairwise confounder interactions in the treatment equation, drawn the
     same way.
3. **Distributions.** The LLM supplies category proportions or a mean and
   standard deviation for each variable.
4. **Scaling.** Coefficients are rescaled so that each linear predictor has
   standard deviation `--target-logit-std` (estimated from 10,000 Monte Carlo
   draws). The treatment main effect is not rescaled.
5. **Calibration.** Intercepts are solved numerically so that the mean
   treatment probability equals `--target-treatment-rate` and the mean outcome
   probability under the comparator equals `--target-control-outcome-rate`.

For each patient, Python samples the characteristics (categorical levels from
the proportions; continuous values from a moment-matched log-normal when the
mean is positive, otherwise a normal distribution), computes the propensity
`e = σ(treatment predictor)`, and draws the treatment. A binary outcome is drawn
as Bernoulli(σ(outcome predictor)); a continuous outcome is the outcome
predictor plus normal noise with standard deviation `--outcome-noise-std`. The
true individual effect is the difference between the two potential outcomes:
`σ(lp₁) − σ(lp₀)` for binary outcomes (a risk difference) and `lp₁ − lp₀` for
continuous outcomes. If no variable has the effect-modifier role, the true
effect is the same for every patient.

**Text.** The LLM writes a timeline of 15 to 30 tagged clinical events that ends
before the treatment decision, given the patient's true characteristics. The
timeline never mentions the treatment or the outcome. Narrative events
(clinical notes, imaging, pathology, and NGS reports) are each expanded into a
full document. With `--structured-data`, encounter, hospitalization, laboratory,
and patient-reported-outcome events are rendered as fixed-format text blocks.
All documents are joined in date order with the note separator to form
`clinical_text`. A configurable fraction of drug mentions is replaced by brand
names or abbreviations (`--drug-perturbation-prob`).

## Running the generator

A binary-outcome cohort with one confounder and one effect modifier, including
structured-data text:

```bash
python -m synthetic_data.cli \
  --api-url http://127.0.0.1:8000/v1 \
  --model MODEL \
  --clinical-question "Compare vinorelbine to gemcitabine for advanced/metastatic NSCLC." \
  --dataset-size 1000 \
  --num-confounders 1 \
  --num-effect-modifiers 1 \
  --structured-data \
  --output-dir ./synthetic_output/one_conf_one_mod
```

A continuous-outcome cohort with five confounders and three effect modifiers:

```bash
python -m synthetic_data.cli \
  --api-url http://127.0.0.1:8000/v1 \
  --model MODEL \
  --outcome-type continuous --outcome-noise-std 1.0 \
  --num-confounders 5 --num-effect-modifiers 3 \
  --dataset-size 5000 \
  --output-dir ./synthetic_output/continuous_example
```

`MODEL` is the model ID served at the endpoint. Output directories named
`synthetic_output*` are ignored by Git.

### Command-line options

**Scenario and causal structure**

| Option | Default | Meaning |
|---|---|---|
| `--config PATH` | none | JSON generation configuration, such as a saved `generation_config.json` (see [Configuration files](#configuration-files)) |
| `--clinical-question TEXT` | a metastatic ER+ breast cancer comparison | Treatment comparison the cohort represents |
| `--dataset-size N` | 500 | Number of patients |
| `--num-features N` | LLM chooses 8–12 | Total variables. When omitted but a role count is given, it is the sum of the role counts |
| `--num-confounders N` | none | Variables with the confounder role |
| `--num-effect-modifiers N` | none | Variables with the effect-modifier role. A variable with both roles counts toward both |
| `--treatment-effect X` | 0.20 | Treatment main-effect coefficient: log odds for binary outcomes, outcome units for continuous outcomes. It is not the ATE, because effect-modifier interactions also contribute |
| `--target-treatment-rate P` | 0.5 | Mean treatment probability |
| `--target-control-outcome-rate P` | 0.5 | Mean outcome probability under the comparator |
| `--target-logit-std S` | 2.0 | Standard deviation of each linear predictor; smaller values push propensities toward 0.5 |
| `--enforce-positivity` | off | Clip every patient's propensity to `[--min-treatment-rate, --max-treatment-rate]` |
| `--min-treatment-rate P`, `--max-treatment-rate P` | 0.1, 0.9 | Clipping bounds used with `--enforce-positivity` |
| `--outcome-type` | `binary` | `binary` or `continuous` |
| `--outcome-noise-std S` | 1.0 | Noise standard deviation for continuous outcomes |
| `--seed N` | 42 | Random seed |

**Text**

| Option | Default | Meaning |
|---|---|---|
| `--min-events N`, `--max-events N` | 15, 30 | Events per patient timeline (minimum at least 5) |
| `--note-separator TEXT` | `"\n\n<new_note>\n\n"` | String placed between documents in `clinical_text` |
| `--drug-perturbation-prob P` | 0.3 | Probability of replacing a drug name with a brand name or abbreviation |
| `--structured-data` | off | Add encounter, hospitalization, laboratory, and patient-reported-outcome blocks |
| `--no-encounters`, `--no-labs`, `--no-hospitalizations`, `--no-pros` | off | Omit one structured-data block type |

**LLM endpoint and execution**

| Option | Default | Meaning |
|---|---|---|
| `--api-url URL` | `http://localhost:8000/v1` | OpenAI-compatible base URL |
| `--api-key KEY` | empty | API key, if the server requires one |
| `--model ID` | `openai/gpt-oss-120b` | Served model ID |
| `--max-tokens N` | 50000 | Output ceiling per request |
| `--num-workers N` | 40 | Concurrent patient threads |
| `--output-dir DIR` | `./synthetic_output` | Output directory |
| `--verbose`, `-v` | off | Debug logging |

**Saved DGP**

| Option | Default | Meaning |
|---|---|---|
| `--dgp-metadata PATH` | none | `metadata.json` of an existing cohort; see [Expanding a cohort from a saved DGP](#expanding-a-cohort-from-a-saved-dgp) |

**Optional extraction check.** After generation, the generator can ask the same
endpoint to extract each true variable back from `clinical_text`, as a check of
how recoverable the variables are from the text.

| Option | Default | Meaning |
|---|---|---|
| `--extract-features` (alias `--extract-confounders`), `--no-extract-features` | off | Run the extraction check |
| `--extraction-max-text-tokens N` | 100000 | Keep only the last N tokens of each record |
| `--extraction-max-completion-tokens N` | 5000 | Output ceiling per extraction request |
| `--extraction-batch-size N` | 10 | Progress-reporting batch size; requests are sent one at a time |
| `--reasoning-marker TEXT` | none | Marker that separates reasoning from the answer in responses |

The extraction check loads the tokenizer for `--model` from Hugging Face, so
`--model` must be a Hugging Face ID or a local path. It uses the command-line
endpoint, model, and key even when `--config` is given.

### Configuration files

`--config` loads a JSON file with the same fields as `generation_config.json`.
A command-line option overrides the file only when its value differs from the
option's default; `--output-dir` always comes from the command line. A few
settings exist only in the file:

| Field | Default | Meaning |
|---|---|---|
| `main_coefficient_scale` | 0.3 | Scale for the LLM's main-effect coefficients |
| `interaction_coefficient_scale` | 0.1 | Scale for all interaction coefficients |
| `continuous_covariate_distribution` | `positive_lognormal` | `positive_lognormal` (log-normal when the mean is positive, otherwise normal) or `normal`. A configuration file without this key loads as `normal` |
| `note_types_to_expand` | clinical note, imaging, pathology, NGS reports | Event types expanded into full documents |
| `structured_data.pro_instruments` | `EORTC_QLQ_C30`, `PRO_CTCAE` | Patient-reported-outcome instruments |
| `llm.timeout` | 300 | Request timeout in seconds |
| `llm.api_base_urls` | none | List of equivalent endpoints used in rotation |

## Output files and columns

A generation run writes these files to the output directory:

| File | Content |
|---|---|
| `generation_config.json` | Resolved generation settings, written before generation starts |
| `dgp_design.json` | The DGP designed for this run and the settings it depends on; a resumed run reuses it instead of designing a new one |
| `checkpoint.json` | Completed patients, used to resume an interrupted run |
| `dataset.parquet` | One row per patient |
| `metadata.json` | The DGP: variables, roles, equations, distributions, and summary statistics of the generated cohort |
| `dataset_with_extraction.parquet`, `dataset_with_extraction.metrics.json` | Present when the extraction check runs |

`generation_config.json` and `metadata.json` never store the API key.

Columns of `dataset.parquet`:

| Column | Content |
|---|---|
| `patient_id` | Integer identifier, 0 to N − 1 |
| `clinical_text` | The patient's documents, joined in date order |
| `treatment_indicator` | 0 or 1 |
| `outcome_indicator` | 0 or 1 for binary outcomes; a real number for continuous outcomes |
| `num_notes` | Number of documents in `clinical_text` |
| `patient_prompt`, `event_timeline` | The true characteristics as text, and the generated timeline |
| `true_treatment_prob` | True propensity `e(X)` |
| `true_y0`, `true_y1` | True expected outcome under comparator and under treatment (probabilities for binary outcomes) |
| `true_ite` | `true_y1 − true_y0`, on the outcome scale |
| `true_outcome_prob`, `true_y0_prob`, `true_y1_prob`, `true_ite_prob` | Binary outcomes only: the outcome probability at the observed treatment, and probability-scale copies of the three columns above |
| `true_<variable>` | True value of each generated variable |

There is no `split` column. The workflow assigns folds itself.

`patient_prompt` and `event_timeline` contain the true characteristics in text
form but are not oracle-prefixed. Do not point `--text-column` at either of
them.

Rerunning the same command resumes an interrupted generation: the DGP is read
from `dgp_design.json` and completed patients from `checkpoint.json`. The run
stops with an error if the generation settings changed (anything other than
`output_dir`, `dataset_size`, or LLM connection settings) or if a checkpointed
patient no longer matches its re-sampled characteristics; use an empty output
directory for different settings. `--seed` fixes each patient's sampled
characteristics for a given DGP regardless of `--num-workers`; the LLM-designed
DGP and the generated text are not seeded.

## Expanding a cohort from a saved DGP

To generate more patients from an existing cohort's causal process, pass its
configuration and metadata:

```bash
python -m synthetic_data.cli \
  --config ./synthetic_data/example_synthetic_datasets/five_confounders_five_effect_modifiers_nsclc_with_structured/generation_config.json \
  --dgp-metadata ./synthetic_data/example_synthetic_datasets/five_confounders_five_effect_modifiers_nsclc_with_structured/metadata.json \
  --api-url http://127.0.0.1:8000/v1 \
  --model MODEL \
  --dataset-size 10000 \
  --num-workers 32 \
  --output-dir ./synthetic_output/five_conf_five_mod_10000
```

The variables, distributions, equations, and calibrated intercepts are loaded
from `metadata.json` and kept exactly; nothing is recalibrated. The run samples
a separate cohort (`patient_id` 0 to N − 1) and writes text for it; it
does not append to the original rows. Every setting in the source
configuration except `dataset_size`, `output_dir`, `llm`, and `seed` must match
the configuration you pass, which is why `--config` should point to the source
cohort's `generation_config.json`. The model and endpoint can differ from the
ones that generated the source.

The bundled cohorts were generated with normal draws for continuous variables
(some laboratory values are negative), and their `generation_config.json` files
set `"continuous_covariate_distribution": "normal"`, so expansions reproduce
those distributions. A saved configuration without the key is treated as
`normal`.

Saved-DGP expansion is designed for long runs:

- Each timeline, document, and completed patient is checkpointed under
  `patients/` and `requests/`; rerunning the same command resumes. The worker
  count and endpoint may change between attempts; any other change to the
  settings, the source metadata, or the generator code makes the run refuse to
  resume in that directory.
- Each request is tried three times. Truncated responses are rejected rather
  than stored as partial records, and the run stops after ten patients fail.
- An exclusive lock (`.generation.lock`) prevents two processes from writing
  the same directory.
- In addition to `dataset.parquet` and `metadata.json` (which records the
  source DGP and a `generation_status`), the directory holds
  `generation_manifest.json`, `patient_scaffold.parquet`,
  `scaffold_sha256.json`, and `progress.json`. The API key is not stored.

Cohorts with 10,000 or more patients placed under
`synthetic_data/example_synthetic_datasets/` should use a directory name
containing `_10000`, which Git ignores.

## Bundled cohorts

Both bundled cohorts are in `synthetic_data/example_synthetic_datasets/`. They
compare vinorelbine with gemcitabine for advanced or metastatic NSCLC, have
binary outcomes, include structured-data text, and were generated with seed 42.

| | One confounder, one modifier | Five confounders, five modifiers |
|---|---|---|
| Directory | `one_confounder_one_effect_modifier_nsclc_with_structured/` | `five_confounders_five_effect_modifiers_nsclc_with_structured/` |
| `run_synthetic.sh` name | `one_conf_one_mod` | `five_conf_five_mod` |
| Patients | 1,000 | 1,000 |
| Confounders | age | age, sex, ECOG performance status, creatinine clearance, prior platinum therapy |
| Effect modifiers | PD-L1 expression (<1%, 1–49%, ≥50%) | histology, EGFR mutation status, baseline neutrophil-to-lymphocyte ratio, brain metastases, baseline hemoglobin |
| Treated fraction | 0.487 | 0.521 |
| Outcome rate | 0.544 | 0.449 |
| Mean (SD) of `true_ite_prob` | 0.063 (0.163) | −0.074 (0.181) |
| Empty `clinical_text` | 1 patient | 5 patients |

Records average about 52,000 characters (maximum about 130,000) and 19
documents per patient.

Each directory contains `dataset.parquet`, `generation_config.json`,
`metadata.json`, and the extraction-check files
`dataset_with_extraction.parquet` and `dataset_with_extraction.metrics.json`.
The bundled `dataset.parquet` files carry the probability-scale oracle columns
(`true_treatment_prob`, `true_outcome_prob`, `true_y0_prob`, `true_y1_prob`,
`true_ite_prob`) and `true_<variable>` columns, but not `true_y0`, `true_y1`, or
`true_ite`. The evaluators therefore use `true_ite_prob` for these cohorts. In
`dataset_with_extraction.parquet`, the `true_<variable>` columns are parsed from
`patient_prompt`, so continuous values are rounded.

To run the full workflow on a bundled cohort, see the
[README](../README.md) and the `run_synthetic.sh` section of
[Configuration](configuration.md#the-synthetic-launcher-run_syntheticsh).

## Oracle-safe evaluation

Oracle columns are any columns whose names start with `true_`, `oracle_`, or
`ground_truth`. The workflow drops them before Stage 1, and Stage 2 refuses
prediction files that contain them. They are joined to results only after the
predictions are written and hashed.

**Stage 2.** When the dataset contains an oracle effect column, Stage 2
evaluates its frozen per-patient predictions (`estimated_cate` in
`stage2/cross_fitted_predictions.csv`) after all modeling is complete. It uses
`true_ite` when present, otherwise `true_ite_prob` for binary outcomes; a
continuous-outcome dataset with only `true_ite_prob` is reported as not
evaluable. It writes:

- `stage2/posthoc_oracle_ite_metrics.json`: overall and per-fold Pearson and
  Spearman correlations, mean absolute error, root mean squared error, ATE bias,
  and standard deviations of true and estimated effects;
- `stage2/posthoc_predictions_with_oracle_ite.csv`: the predictions joined to
  the oracle column;
- the two correlations, copied into `stage2/causal_estimate.json`.

For datasets without oracle columns the metrics file records that evaluation is
unavailable.

**Stage 1.** `oci-evaluate-stage1` compares each frozen Stage 1 architecture
with the oracle columns. It hashes and loads all Stage 1 evidence, score,
split, and timing files before it reads the dataset, and it never fits or
selects a model.

```bash
oci-evaluate-stage1 --run-dir OUTPUT_DIR --architectures all
```

| Option | Default | Meaning |
|---|---|---|
| `--run-dir DIR` | required | Workflow output directory (must contain `run_config.json`) |
| `--dataset PATH` | `dataset` from `run_config.json` | Dataset with oracle columns |
| `--metadata PATH` | `metadata.json` beside the dataset, if present | DGP metadata, used for variable names |
| `--architectures LIST` | every architecture in the run | Comma-separated names, or `all` |
| `--output-dir DIR` | `OUTPUT_DIR/evaluation/stage1` | Where results are written |
| `--null-draws N` | 200 | Random-chunk null draws per architecture and context for lexical recall; 0 disables the null |

It reads `stage1_architectures/` when present; otherwise it builds that view
from the handoff and writes it into the run directory. On outer held-out rows
it reports, for each architecture, how strongly its scores associate with each
`true_<variable>` (absolute Spearman correlation or one-versus-rest AUC), how
strongly its effect scores correlate with `true_ite` (or `true_ite_prob` for
binary outcomes), nuisance discrimination, matching coverage where relevant,
and a normalized R-loss gain. The embedding architectures are scored on the
directions handed to Stage 2. It also reports lexical recall of each true
variable's name in the evidence, compared in each discovery context with
random text windows of the same number and length drawn from that context's
training rows, inner-fold stability, and the Stage 1 wall time of each
architecture. Association metrics take the best value over an architecture's
score columns and ignore sign. The outputs are `metrics.jsonl`,
`architectures/<architecture>/metrics.jsonl`, `comparison.csv`,
`lexical_recovery_null.csv`, `timing.csv`, `summary.json`, and
`evaluation_manifest.json` (input hashes and the oracle columns used); see
[Stage 1 architectures](stage1_architectures.md#outputs) for their contents.
Rows are aligned by position, so evaluate with the same dataset file the run
used. See
[Stage 1 architectures](stage1_architectures.md) for what each architecture
produces.

**Attribution and ablation.** `oci-evaluate-attribution --oracle` hashes the
frozen Stage 2 artifacts it reads before it loads the oracle columns, then
measures which `true_<variable>` columns the selected Stage 2 variables
recover by held-out R² and credits each recovery to the Stage 1 architectures
behind the variable. `scripts/ablate_stage1_architectures.py compare --oracle`
reports the same recovery for each leave-one-architecture-out rerun. See
[Stage 1 architectures](stage1_architectures.md#attribution-and-ablation).

**Benchmarks.** `scripts/benchmark.py` re-equates a source cohort into one
cohort per benchmark scenario, size, and seed, writes the workflow
configurations, and compares the finished runs with these oracle columns under
the same hash-first rule; see [Benchmarks](benchmarks.md), which also covers
using the bundled five-confounder cohort as a re-equation source.

**Standalone `oci run`.** The explicit-feature workflows write
`applied_inference/posthoc_oracle_metrics.json` under the same rules; see
[Optional extensions](optional_extensions.md#outputs).
