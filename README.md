# Oncology Causal Inference (OCI)

OCI is a research pipeline for estimating how the effect of a treatment varies
between patients when the relevant patient characteristics are recorded only in
unstructured clinical text, with a focus on oncology. It finds candidate clinical
characteristics in the notes, turns them into explicit patient-level
measurements, and uses those measurements to estimate per-patient treatment
effects and an overall treatment effect.

You can also supply the variables yourself: use [explicit-only extraction](#extract-supplied-variables-without-stage-1)
for variables still in text, or [tabular mode](#analyze-pre-structured-tabular-data-without-an-llm)
for columns that are already measured.

The motivating question is: **for a patient with these documented
characteristics, which of two treatments is expected to produce the better
outcome, and by how much?**

## Key ideas

- A **confounder** is a pretreatment characteristic that influences both which
  treatment a patient receives and the outcome. Comparisons must adjust for it.
- An **effect modifier** is a pretreatment characteristic that changes how much a
  patient benefits from treatment. A variable can be both.
- The **average treatment effect (ATE)** is the expected outcome difference if
  everyone received the treatment versus the comparator. The **conditional
  average treatment effect (CATE)** is the same difference for patients with
  given characteristics; OCI reports one estimated CATE per patient. For binary
  outcomes both are risk differences.

Other terms are defined in the [glossary](docs/glossary.md).

## How it works

1. **Stage 1: find evidence in text.** Up to ten independent text models
   (lexical, neural, embedding, and topic models) point to passages associated
   with treatment, outcome, or variation in the treatment effect. Eight run by
   default. Their evidence is frozen in a handoff.
2. **Stage 2: define and measure variables.** An interpretation LLM reads the
   evidence, names candidate clinical variables, and writes a measurement rule
   for each. An extraction LLM measures every variable from every patient's
   record. Each cell is a value, not documented, documented as unknown (the
   record states that the value is unknown, pending, not assessed, or
   equivocal), or an extraction failure, and the three non-value states are
   encoded and reported separately. Definitions are refined, values harmonized,
   and uninformative candidates removed.
3. **Stage 2: select and estimate.** Predictive models choose which variables
   to adjust for. Effect modifiers are selected by pre-specified tests: each
   candidate gets an R-learner projection test and an AIPW projection test of
   the risk-difference effect and an adjusted interaction test on the
   odds-ratio scale, combined by a Cauchy combination, with Benjamini–Hochberg
   control across candidates. A causal forest, a penalized interaction model
   (which penalizes the confounders but not the selected modifiers' terms),
   and a DINA model (Gao and Hastie's difference-in-natural-parameters model,
   which for binary outcomes estimates a conditional log odds ratio) are all
   fitted, and their average predicts effects for held-out patients. A doubly
   robust (AIPW) estimator gives the ATE, and each fold's selected modifiers
   are re-tested on its held-out patients for reporting. The alternative
   `balanced_evidence` rule instead chooses the number of modifiers and one
   model by nested validation.

Everything runs within outer cross-validation folds, so each patient's
prediction comes from a pipeline that never saw that patient's outcome. All
supplied text is assumed to be pretreatment (see
[Pretreatment text](#pretreatment-text)). Variable roles are modeling
decisions, not proof of causal structure; causal interpretation still requires
no unmeasured confounding, adequate overlap, and accurate measurement. See
[How OCI works](docs/how_it_works.md) for the full walkthrough.

## Installation

Python 3.12 or 3.13 and [uv](https://docs.astral.sh/uv/) are required:

```bash
git clone https://github.com/kenlkehl/onc-causal-inference.git
cd onc-causal-inference
uv sync --frozen
```

Stage 1 is designed for NVIDIA CUDA GPUs. Text-based Stage 2 talks to OpenAI-compatible LLM
servers, which can be servers you already run or vLLM pools that OCI starts and
stops. For OCI-managed vLLM servers, install the `local-llm` extra and configure
the CUDA libraries:

```bash
uv sync --frozen --extra local-llm
.venv/bin/python scripts/configure_local_cuda.py
```

The host must provide a compatible NVIDIA driver. `pip install -e .` (with extras
from [pyproject.toml](pyproject.toml)) also works. See [Serving](docs/serving.md)
for hardware setup, model caching, ports, and troubleshooting.
Tabular mode runs the numerical learners on CPUs and needs no LLM server or
`local-llm` extra.

## Run your own cohort

The input is a CSV or Parquet file with one row per patient (or treatment
decision):

| Column (default name) | Content |
|---|---|
| `patient_id` | Unique identifier |
| `clinical_text` | The patient's pretreatment clinical record, or built by OCI from dated notes ([Pretreatment text](#pretreatment-text)) |
| `treatment_indicator` | Treatment, coded 0/1 |
| `outcome_indicator` | Outcome, binary 0/1 or continuous |

Copy the example configuration **inside `example_configs/`**, because relative
paths resolve against the configuration file's directory:

```bash
cp example_configs/research_all_evidence_multi_model.json example_configs/my_run.json
```

Edit the dataset and output paths, column names, `science.clinical_question`,
`science.outcome_type`, and `run.devices`. Point `stage2.endpoint` (interpretation
LLM) and `stage2.extraction_llm.endpoint` (extraction LLM) at your servers,
including the `/v1` suffix; both may be the same server. Leave the model IDs empty
to use the single model a server advertises. Multiple extraction servers can
use capacity-weighted routing; explicitly selected model variants are supported
with `stage2.extraction_llm.allow_model_variants` and per-server model IDs
([Serving](docs/serving.md#several-extraction-servers)). Then run:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
uv run python scripts/run_all_evidence.py --config example_configs/my_run.json
```

The example sets propensity eligibility bounds of 0.1 to 0.9 and enables two
optional Stage 2 steps (pre-selection measurement consolidation and
matched-batch evidence); review these before adapting it. Useful options:

| Option | Effect |
|---|---|
| `--status` | Print the run's `progress.json` and exit |
| `--stage1-only` | Run Stage 1 through the handoff and stop |
| `--architectures bow_nuisance,tfidf_topics` | Run only the named Stage 1 architectures (`all` runs all ten); the choice is fixed for the output directory |
| `--set KEY=VALUE` | Override any configuration value, for example `--set stage2.workers=16` |

All 29 flags and every configuration key are listed in
[Configuration](docs/configuration.md).

### Pretreatment text

OCI assumes that every word of the patient text was documented before the
treatment decision. It cannot verify this for a single text column, and text
written later breaks the analysis in two ways:

- **Post-treatment information biases the estimates.** Response, toxicity, or
  anything else the treatment caused is a consequence of treatment, not a
  reason for it. Adjusting for it, or letting it define subgroups, biases the
  ATE and the CATEs.
- **Direct treatment proxies break overlap.** An order, a consent, or a plan
  that names the regimen predicts treatment almost perfectly, so propensities
  approach 0 or 1 and few patients remain comparable.

In practice:

- Include only notes dated before the index date (the treatment decision or
  treatment start).
- Decide deliberately about notes written on the index day. They often record
  the decision itself or the first administration, so OCI leaves them out by
  default.
- Review the retained notes anyway. Notes dated before the index date can still
  contain plan statements ("will start carboplatin and pemetrexed") or
  copied-forward text that names the intended treatment, and date filtering
  does not remove them.

When each note has a date, OCI can apply the date rule itself. Supply a notes
table with one row per note and a `notes` block; `dataset` then holds one row
per patient with an index date column and no text column, and `columns.text`
names the text column that OCI builds:

```json
"dataset": "cohort.csv",
"columns": {"unit_id": "patient_id", "text": "clinical_text", "treatment": "treatment_indicator", "outcome": "outcome_indicator"},
"notes": {
  "path": "notes.parquet",
  "columns": {"unit_id": "patient_id", "date": "note_date", "text": "note_text", "note_type": null},
  "index_date_column": "index_date",
  "include_index_date": false
}
```

OCI keeps each patient's notes dated strictly before the index day, orders
them by date, starts each with a header such as
`[Note 3 of 7 · 42 days before index date]` (never a calendar date), and writes
the cohort with the derived text to `OUTPUT_DIR/input/analysis_cohort.parquet`.
Next to it, `pretreatment_notes_manifest.json` counts the notes kept and
dropped, by reason, and `pretreatment_notes_audit.csv` gives the counts per
patient. Every step of the run reads the derived file. A resumed run reuses it
unchanged and stops if the notes table, the cohort file, or the `notes` settings
differ from the ones it was built from. All keys and rules are in
[Configuration](docs/configuration.md#notes).

## Run a bundled synthetic cohort

`run_synthetic.sh` runs both stages on one of the two bundled synthetic
non-small-cell lung cancer cohorts (1,000 patients each, with known effects):

```bash
./run_synthetic.sh one_conf_one_mod                    # one confounder, one modifier
./run_synthetic.sh five_conf_five_mod /path/to/output  # five of each, explicit output directory
```

`PRESET` selects the serving setup:

| Preset | Stage 2 servers |
|---|---|
| `external` (default) | Servers you already run: interpretation LLM at `http://127.0.0.1:8010/v1`, extraction LLM at `http://127.0.0.1:8020/v1` |
| `h100x8` | OCI starts and stops vLLM pools on eight 80 GB H100 GPUs |
| `rtxpro6000x8` | OCI starts and stops vLLM pools on eight 96 GB RTX PRO 6000 GPUs |
| `dgx_station_gb300` | OCI starts and stops one vLLM server on a DGX Station's GB300 GPU (physical GPU 1 in PCI order) |

```bash
PRESET=rtxpro6000x8 ./run_synthetic.sh five_conf_five_mod /path/to/output
```

Environment variables override preset values, for example `STAGE2_ENDPOINT`,
`STAGE2_EXTRACTION_ENDPOINT`, `STAGE2_WORKERS`, `STAGE1_ARCHITECTURES`, and
`DISABLE_HTR=1`; extra arguments after the output directory pass through to the
workflow. Without an output directory, results go to
`artifacts/research_all_evidence/<cohort>_nsclc_full` (or
`<cohort>_nsclc_<preset>_full` for a managed preset). See
[Configuration](docs/configuration.md#the-synthetic-launcher-run_syntheticsh) and
[Serving](docs/serving.md#hardware-presets).
`./run_synthetic.sh CONFIG.json` instead runs the cohort of a workflow
configuration file with the preset's serving settings and the detected
devices.

Synthetic notes never mention treatment or outcome, so one corpus serves many
data-generating processes. `python -m synthetic_data.cli reequation` recomputes
treatment and outcome over a cohort's true characteristics for a scenario
(instruments, prognostic-only variables, dual roles, modifier main effects,
nonlinear terms, unmeasured confounders) without an LLM, copying the text
unchanged; it refuses cohorts with post-treatment notes. Factors that change the
text are generator options: correlated covariates, MCAR or MNAR documentation,
post-treatment dated notes, and near-duplicate proxies. See
[Synthetic data](docs/synthetic_data.md#re-equation).

## Choose where to enter the workflow

| Mode | What you supply | What OCI runs |
|---|---|---|
| Full all-evidence | Pretreatment text, treatment, outcome | Stage 1 discovery, LLM definition/extraction, variable selection, estimation |
| `explicit_only` / `--explicit-only` | Text plus variables, extraction ontologies, and causal roles | LLM extraction of the supplied variables, downstream estimation and diagnostics |
| `tabular` / `--tabular-only` | Measured covariate columns, types, and causal roles | Nuisance learning, the effect-estimator ensemble, and diagnostics |

Both supplied-variable paths create their own nested folds and fix the declared
roles. Use a separate output directory for each analysis.

### Extract supplied variables without Stage 1

Use `--explicit-only` (or `"run": {"mode": "explicit_only"}`) to extract
investigator-specified variables and run the existing Stage 2 estimation and
diagnostics without an evidence handoff or Stage 1 models:

```bash
uv run python scripts/run_all_evidence.py --config example_configs/research_explicit_only.json
```

Set the cohort and model endpoints, and supply complete ontologies in
`stage2.explicit_features`. Their definitions and causal roles remain fixed.
The first run saves nested folds and the input identity; subsequent runs reuse
them. See [explicit-only configuration](docs/configuration.md#explicit-only-workflow)
for preflight, resume, and the example schema.

### Analyze pre-structured tabular data without an LLM

Use `--tabular-only` (or `"run": {"mode": "tabular"}`) with a CSV or Parquet
table containing patient ID, binary treatment, outcome, and measured
pretreatment covariates:

```bash
uv run python scripts/run_all_evidence.py --config example_configs/research_tabular.json
```

Edit the example's dataset path, column names, and `tabular.features`. Each
feature declares its column, value type, categories when applicable, and
`confounder` and/or `effect_modifier` roles. No text column, extraction
ontology, model endpoint, or LLM is needed.

This path skips discovery, extraction, and variable selection. It retains
inner cross-fitted nuisance learning (Super Learner by default), outer held-out
effect predictions, the final estimator ensemble, ATE/ATO estimation, and
the existing diagnostics. Rerunning validates the frozen inputs and reuses
compatible fits; changing numerical model settings refits affected estimates.
See [tabular configuration](docs/configuration.md#tabular-workflow) for the
schema, missing values, preflight, and outputs.

### Estimate effects with HTR models alone

`scripts/run_standalone_htr.py` fits HTR models per outer fold, without an LLM,
handoff, or Stage 2, and writes cross-fitted held-out `tau`, `e`, and `m`, an
oracle-free summary, and (with `--evaluate`, on synthetic data) an oracle
comparison. The default method, `two_stage`, runs Stage 1's `htr_neural`
training unchanged: cross-fitted HTR nuisance networks, then HTR effect
networks (DINA for a binary outcome). `--method joint` instead trains one
network with propensity, outcome, and effect heads:

```bash
uv run python scripts/run_standalone_htr.py run --dataset DATA.parquet --out OUT \
  --seed 101 --devices cuda:0,cuda:1 --jobs-per-device 3 --evaluate
```

See [Standalone HTR](docs/standalone_htr.md).

## Resume, inspect, or rerun selection

The output directory is the checkpoint. **Rerunning the same command resumes an
interrupted run**; completed work is reused. The directory holds the resolved
settings (`run_config.json`), `progress.json`, and `logs/workflow.log`.

A verified relocation manifest can preserve Stage 1 text-model checkpoint
identities and seeds when a working copy moves to another filesystem. See
[checkpoint relocation](docs/configuration.md#checkpoint-relocation); ordinary
path changes alone do not preserve those identities.

```bash
# Relaunch Stage 2 from the run's resolved settings
uv run python scripts/run_all_evidence.py --config /path/to/output/run_config.json --stage2-only

# Check what Stage 2 would reuse, without writing anything
uv run python scripts/run_all_evidence.py --config /path/to/output/run_config.json --stage2-only --preflight-only

# Recompute selection and estimation after a completed Stage 2, reusing extractions
uv run python scripts/run_all_evidence.py --config /path/to/output/run_config.json --stage2-only --stage2-reselect
```

For tabular runs, use the first two commands. Roles are fixed, so
`--stage2-reselect` is not applicable; an ordinary rerun checks estimation
checkpoints even after completion.

Reselection archives the previous selection and estimates under
`stage2/reselection_archives/` and reuses the frozen definitions and extracted
measurements; it repeats no Stage 1 modeling. Under the default
`orthogonal_tests` modifier rule, changing `stage2.estimation_trees` or the
interaction model's selected-modifier penalty settings
(`modifier_selection.unpenalized_selected_interactions` and
`unpenalized_events_per_parameter`) needs no reselection: the selection and its
stored nuisance models are reused and only the final estimation is refitted. For the synthetic launcher, use
`STAGE2_ONLY=1` (with `STAGE2_RESELECT=1` or `OCI_PREFLIGHT_ONLY=1` as needed).
Changing data, definitions, or scientific settings can invalidate checkpoints;
use a separate output directory for a different study. Details are in
[Configuration](docs/configuration.md#checkpoints-and-resuming).

## Outputs and how to read them

| Path in the output directory | Contents |
|---|---|
| `input/pretreatment_notes_manifest.json`, `input/pretreatment_notes_audit.csv` | With dated notes: notes kept and dropped by reason, overall and per patient |
| `handoff/evidence.jsonl` | Frozen Stage 1 evidence passed to Stage 2 |
| `stage2/explicit_inputs.json` or `stage2/tabular_inputs.json` | Supplied-variable modes: frozen input identity and nested splits |
| `stage2/measurements.csv` | Tabular mode: normalized supplied covariate values |
| `stage2/outer_NNN/final_definitions.json` | Each outer fold's retained variables, roles, definitions, each modifier's test evidence, and the final estimator |
| `stage2/outer_NNN/selection/modifier_tests.json` | Every candidate's modifier tests, combined p- and q-values, selection, and labels |
| `stage2/outer_NNN/selection/modifier_count/selection.json` | Summary of the modifier selection and the estimator policy |
| `stage2/outer_NNN/estimation/diagnostics.json` | Model fit, nuisance calibration, overlap, and fold ATE |
| `stage2/cross_fitted_predictions.csv` | One held-out prediction per patient |
| `stage2/causal_estimate.json` | Co-primary ATE and ATO (per fold and pooled), their E-values and robustness values, overlap-flag counts, and a diagnostics summary |
| `stage2/heterogeneity_diagnostics.json` | Heterogeneity, overlap, and balance diagnostics, the outer-fold confirmation of the selected modifiers, and the sensitivity to unmeasured confounding |
| `stage2/posthoc_oracle_ite_metrics.json` | Comparison with true effects (synthetic data only) |

**Per-patient columns** in `cross_fitted_predictions.csv`:

- `estimated_cate` is the patient's predicted treatment effect (a risk
  difference for binary outcomes): by default the unweighted mean of the causal
  forest, the penalized interaction model, and, for binary outcomes, DINA.
  Their individual predictions are in `estimated_cate_causal_forest`,
  `estimated_cate_linear_interactions`, and `estimated_cate_dina`.
- `estimated_cate_standard_error`, `estimated_cate_lower_95`, and
  `estimated_cate_upper_95` are empty under the default ensemble, because the
  forest's standard error does not describe the average; the forest's own
  interval is in `estimated_cate_causal_forest_lower_95`,
  `estimated_cate_causal_forest_upper_95`, and
  `estimated_cate_causal_forest_standard_error`. When a single causal forest is
  the final model, the forest fills the three main interval columns. No
  per-patient interval is reported for the ensemble, because no available
  interval accounts for variable discovery, extraction, and selection; the GATES
  group effects and the per-modifier held-out effect sizes below are the honest
  uncertainty statements.
- `propensity`, `mu0`, and `mu1` are the nuisance predictions: the probability of
  treatment and the expected outcome under comparator and under treatment, from
  separate cross-fitted nuisance models, by default super learners that stack an
  elastic net, gradient boosting, and a random forest. For held-out patients they
  average the inner-fold models after calibrating each model on its own
  validation patients (the propensity and, for binary outcomes, `mu0` and
  `mu1`). They are not the effect
  model's predictions, so `mu1 − mu0` need not equal `estimated_cate`.
- `aipw_score` is the patient's augmented inverse-probability-weighted score,
  built from the observed outcome and the nuisance predictions.
- `effect_eligible` is false for patients outside the propensity bounds, if set;
  those patients have no `estimated_cate` or `aipw_score`.
- `outside_overlap` is true when the patient's final held-out `propensity` lies
  outside `stage2.overlap_flag_bounds` (default 0.05 to 0.95). It is a reporting
  flag only: it never removes patients or changes any estimate.

**Co-primary estimands.** `causal_estimate.json` reports two treatment effects
side by side under `co_primary_estimands`, per fold and pooled, each with a
standard error, a 95% interval, and a plain-language description of its target
population:

- **ATE** (`ate`): the clipped AIPW average treatment effect over all
  effect-eligible patients, the mean of `aipw_score` with standard error
  `sd(aipw_score)/√n`.
- **ATO** (`ato`): the overlap-weighted average treatment effect, which weights
  each patient by `e(1 − e)` and so targets patients in clinical equipoise, who
  could plausibly have received either treatment.

Choose the lead estimand before looking at results and record it in
`stage2.lead_estimand` (`"ate"` by default, or `"ato"`); it changes only the
order and labels, never the computation. The ATE answers "what if everyone
eligible were treated"; the ATO answers it for the patients in whom both
treatments are used and is more stable when propensities are extreme. A large
gap between them (`ate_minus_ato`) means the effect differs between those
populations or the ATE leans on patients with extreme propensities; with the
trimming curve and the clip sensitivity it is a secondary sensitivity
analysis. The ATE is also repeated at the top level as `ate`. It is a separate
estimate from `mean_estimated_cate`, the mean of the model's predictions: an
effect model can predict nearly constant effects while the AIPW ATE is clearly
nonzero, or the reverse.

**Sensitivity to unmeasured confounding** (`unmeasured_confounding_sensitivity`
in `heterogeneity_diagnostics.json`, summarized next to the co-primary
estimands in `causal_estimate.json`) asks how strong a confounder missing from
the extracted variables would have to be to change the conclusion, for both
the ATE and the ATO, per fold and pooled. It is reporting only.

- **E-value**: the minimum risk-ratio association that an unmeasured
  confounder would need with both treatment and outcome, beyond the measured
  variables, to explain the estimate away (VanderWeele and Ding 2017), with the
  E-value of the confidence limit closest to the null (1 when the interval
  includes it). For binary outcomes it is computed for the causal risk ratio
  that corresponds to the reported risk difference; for continuous outcomes it
  uses an approximate conversion of the standardized mean difference.
- **Robustness value** (`robustness_value`, and `robustness_value_confidence`
  for the 95% confidence bound): the share of the outcome's residual variation
  and of the treatment-weight variation (as partial R²) that confounders would
  need to explain to reduce the effect to zero, from omitted-variable-bias
  bounds that assume the most adversarial correlation and are therefore
  conservative (Chernozhukov, Cinelli, Newey, Sharma and Syrgkanis; Cinelli and
  Hazlett 2020). A table of bounds over a grid of confounding strengths is
  included.
- **Benchmarks**: each fold refits its nuisance models (as elastic nets)
  without each of its five most important discovered variables
  (`stage2.sensitivity.benchmark_groups`; 0 disables) and reports where a
  confounder as strong as that variable would move the ATE bounds. An
  unmeasured confounder may be weaker or stronger than any measured one.

Measurement error in extracted confounders acts like partial unmeasured
confounding, which these analyses also cover. Formulas, interpretation, and the
full references are in
[Sensitivity to unmeasured confounding](docs/stage2_selection.md#sensitivity-to-unmeasured-confounding).

**Modifier confirmation** (`modifier_confirmation` in
`heterogeneity_diagnostics.json`, summarized in `causal_estimate.json`)
re-tests each fold's selected modifiers on that fold's held-out patients and
reports how many were confirmed (held-out q-value at most `q`, 0.10 by
default), with each modifier's held-out effect size (the change in absolute
benefit per unit, per level, or per training SD, with 95% intervals). Across
folds, variables are grouped into concepts by name and definition similarity;
each concept's held-out p-values are aggregated across folds
(Meinshausen–Meier–Bühlmann) with Benjamini–Hochberg across concepts, and
matched effects are pooled with a heterogeneity statistic. The training-row
p-values in `modifier_tests.json` are screening values, optimistic because the
candidates were found in the same patients; the held-out confirmation is the
inferential result. Confirmation is reporting only and never changes selection
or estimation.

**Heterogeneity diagnostics** ask whether the CATE predictions capture real
differences, using only held-out data:

- **BLP `beta2`**: the slope of the AIPW scores on the predicted CATEs. Near 1
  means well-scaled predictions; near 0 means no detectable heterogeneity; a
  small one-sided p-value supports real heterogeneity.
- **GATES**: the average effect in each fifth of patients ranked by predicted
  CATE, next to the group's mean predicted CATE, with a calibration slope of
  the group effects on those means. An increasing pattern supports the ranking.
- **RATE (AUTOC, Qini)**: how much better than average the top-ranked patients
  do. Positive values with intervals above 0 indicate useful prioritization.
- **ATO**: the effect among patients who could plausibly receive either
  treatment. A large gap from the ATE signals dependence on extreme propensities.
- **Trimming curve**: the ATE after excluding patients with propensities near 0
  or 1. A flat curve is reassuring.
- **Balance**: after propensity weighting, each variable's mean should be similar
  in both arms; a weighted standardized mean difference above 0.1 flags a
  variable the propensity model does not balance.
- **CATE variance decomposition**: how much of the predicted CATE spread a linear
  projection attributes to baseline risk (the prognostic score), to the selected
  modifiers, to both, or to neither. Descriptive, not causal.

The pooled BLP `beta2` with its one-sided p-value is the pre-specified primary
heterogeneity test (`primary_heterogeneity_test` in `causal_estimate.json`);
GATES and RATE are secondary, and CLAN, the variance decomposition, and the
modifier confirmation are descriptive. No diagnostic is used for selection.
Per-fold results use about 200 held-out patients each and are low-powered, and
all results rest on a single random outer split (repeated splitting would
require rerunning the pipeline).

Read these with nuisance calibration, overlap, and fold-to-fold variability. See
[Stage 2 selection](docs/stage2_selection.md#heterogeneity-and-overlap-diagnostics)
for definitions and caveats. For synthetic data, oracle comparisons are computed
only after all predictions are frozen; oracle truth never guides selection.

To evaluate each frozen Stage 1 architecture against synthetic truth without
refitting:

```bash
uv run oci-evaluate-stage1 --run-dir /path/to/output --architectures all
```

It scores the embedding directions handed to Stage 2, compares lexical recall
of each true variable with a random-chunk null (`--null-draws`, 200 by
default), reports each architecture's Stage 1 wall time, and writes JSON and
CSV results to `/path/to/output/evaluation/stage1/`; see
[Stage 1 architectures](docs/stage1_architectures.md#oracle-safe-evaluation).

To credit each Stage 2 variable to the Stage 1 architectures whose evidence
supported it (any-source and sole-source counts per fold), and, for synthetic
cohorts, to measure which true variables the selected variables recover by
held-out R², run `oci-evaluate-attribution`. To see what Stage 2 loses without
an architecture, `scripts/ablate_stage1_architectures.py prepare` writes one
leave-one-architecture-out handoff and launcher configuration per architecture
(it starts no LLM requests), and `compare` reads the finished reruns:

```bash
uv run oci-evaluate-attribution --run-dir /path/to/output --oracle
python scripts/ablate_stage1_architectures.py prepare --run-dir /path/to/output --out /path/to/ablations
python scripts/ablate_stage1_architectures.py compare --run-dir /path/to/output --ablation-dir /path/to/ablations
```

See [Stage 1 architectures](docs/stage1_architectures.md#attribution-and-ablation).

To benchmark OCI across synthetic scenarios (instruments and prognostic-only
variables, dual-role variables, nonlinear effects, an unmeasured confounder,
MNAR documentation, and post-treatment notes), `scripts/benchmark.py` builds one
cohort per scenario, sample size, and seed from `benchmarks/scenarios.json`,
writes one workflow configuration per cohort (it launches no run), and compares
the finished runs with the oracle columns: ATE and ATO coverage, CATE RMSE
against a constant-ATE baseline, GATES coverage, an oracle-covariate ceiling
refitted through Stage 2's own final estimation, role-wise selection, and the
sensitivity and leak checks. Tune only on development seeds; confirmatory seeds
need `--confirmatory` and are logged in `benchmarks/confirmatory_ledger.jsonl`:

```bash
python scripts/benchmark.py materialize --scenarios S0,S1 --source /path/to/source_cohort --out /path/to/bench
python scripts/benchmark.py configs --base /path/to/config.json --out /path/to/bench
python scripts/benchmark.py aggregate --out /path/to/bench
```

`run_benchmark.sh` runs all of it on one machine, named by a hardware preset:
it writes the LLM text the selected cells need (an expansion of the source
cohort for cells larger than it, and the S5 and S6 corpora) on a vLLM pool it
starts and stops, builds the cohorts and configurations, runs each cell in
turn, and aggregates. Rerun the same command to resume:

```bash
./run_benchmark.sh rtxpro6000x8 /path/to/bench --scenarios S0 --sizes 1000 --seeds 101
./run_benchmark.sh dgx_station_gb300 /path/to/bench --plan   # what would run, and what is done
./run_benchmark.sh dgx_station_gb300 /path/to/bench --llm-url http://localhost:8000/v1  # a running vLLM server
```

See [Benchmarks](docs/benchmarks.md).

## Documentation

| Guide | Scope |
|---|---|
| [Glossary](docs/glossary.md) | Terms used throughout the documentation |
| [How OCI works](docs/how_it_works.md) | End-to-end walkthrough of both stages |
| [Stage 1 architectures](docs/stage1_architectures.md) | The ten evidence architectures, selection rules, Stage 1 evaluation, and attribution and ablation of Stage 2 results |
| [Stage 2 selection](docs/stage2_selection.md) | Evidence, ranking, model choice, nuisances, DINA, diagnostics, and limitations |
| [Configuration](docs/configuration.md) | Configuration keys, command-line flags, launcher, outputs, resume, and reselection |
| [Explicit-only extraction](docs/configuration.md#explicit-only-workflow) | Supply variables and ontologies, extract them with an LLM, and run Stage 2 without discovery |
| [Tabular analysis](docs/configuration.md#tabular-workflow) | Supply measured columns and run the ensemble and diagnostics without an LLM |
| [Standalone HTR](docs/standalone_htr.md) | Treatment effects from HTR models alone, per outer fold: Stage 1's cross-fitted HTR ensembles, or one joint network |
| [Serving](docs/serving.md) | LLM endpoints, managed vLLM pools, and hardware presets |
| [Optional extensions](docs/optional_extensions.md) | Optional Stage 2 steps and the standalone `oci run` workflows |
| [Synthetic data](docs/synthetic_data.md) | The synthetic cohort generator, causal roles, generation-time factors, re-equation, bundled cohorts, and oracle columns |
| [Benchmarks](docs/benchmarks.md) | The scenario registry, seeds and the confirmatory rule, the benchmark commands, metrics, and caveats |
| [R-learner weighting](docs/design_notes/r_learner_weighting.md) | Design note on weighted and unweighted effect objectives |

## Citation

```bibtex
@software{oci2024,
  author = {Kehl, Ken},
  title = {Oncology Causal Inference: Treatment Effect Estimation from Clinical Text},
  year = {2024},
  url = {https://github.com/kenlkehl/onc-causal-inference}
}
```

## License

OCI is distributed under the [MIT License](LICENSE).
