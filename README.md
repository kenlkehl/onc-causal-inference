# Oncology Causal Inference (OCI)

OCI is a research pipeline for estimating how the effect of a treatment varies
between patients when the relevant patient characteristics are recorded only in
unstructured clinical text, with a focus on oncology. It finds candidate clinical
characteristics in the notes, turns them into explicit patient-level
measurements, and uses those measurements to estimate per-patient treatment
effects and an overall treatment effect.

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
   default (seven for continuous outcomes). Their evidence is frozen in a
   handoff.
2. **Stage 2: define and measure variables.** An interpretation LLM reads the
   evidence, names candidate clinical variables, and writes a measurement rule
   for each. An extraction LLM measures every variable from every patient's
   record. Definitions are refined, values harmonized, and uninformative
   candidates removed.
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
supplied text is assumed to be pretreatment. Variable roles are modeling
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

Stage 1 is designed for NVIDIA CUDA GPUs. Stage 2 talks to OpenAI-compatible LLM
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

## Run your own cohort

The input is a CSV or Parquet file with one row per patient (or treatment
decision):

| Column (default name) | Content |
|---|---|
| `patient_id` | Unique identifier |
| `clinical_text` | The patient's pretreatment clinical record |
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
to use the single model a server advertises. Then run:

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

## Resume, inspect, or rerun selection

The output directory is the checkpoint. **Rerunning the same command resumes an
interrupted run**; completed work is reused. The directory holds the resolved
settings (`run_config.json`), `progress.json`, and `logs/workflow.log`.

```bash
# Relaunch Stage 2 from the run's resolved settings
uv run python scripts/run_all_evidence.py --config /path/to/output/run_config.json --stage2-only

# Check what Stage 2 would reuse, without writing anything
uv run python scripts/run_all_evidence.py --config /path/to/output/run_config.json --stage2-only --preflight-only

# Recompute selection and estimation after a completed Stage 2, reusing extractions
uv run python scripts/run_all_evidence.py --config /path/to/output/run_config.json --stage2-only --stage2-reselect
```

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
| `handoff/evidence.jsonl` | Frozen Stage 1 evidence passed to Stage 2 |
| `stage2/outer_NNN/final_definitions.json` | Each outer fold's retained variables, roles, definitions, each modifier's test evidence, and the final estimator |
| `stage2/outer_NNN/selection/modifier_tests.json` | Every candidate's modifier tests, combined p- and q-values, selection, and labels |
| `stage2/outer_NNN/selection/modifier_count/selection.json` | Summary of the modifier selection and the estimator policy |
| `stage2/outer_NNN/estimation/diagnostics.json` | Model fit, nuisance calibration, overlap, and fold ATE |
| `stage2/cross_fitted_predictions.csv` | One held-out prediction per patient |
| `stage2/causal_estimate.json` | Pooled ATE with standard error and a diagnostics summary |
| `stage2/heterogeneity_diagnostics.json` | Heterogeneity, overlap, and balance diagnostics, and the outer-fold confirmation of the selected modifiers |
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

**The ATE** in `causal_estimate.json` (`ate`) is the mean of `aipw_score`, with
standard error `sd(aipw_score)/√n`. It is a separate estimate from
`mean_estimated_cate`, the mean of the model's predictions: an effect model can
predict nearly constant effects while the AIPW ATE is clearly nonzero, or the
reverse.

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

## Documentation

| Guide | Scope |
|---|---|
| [Glossary](docs/glossary.md) | Terms used throughout the documentation |
| [How OCI works](docs/how_it_works.md) | End-to-end walkthrough of both stages |
| [Stage 1 architectures](docs/stage1_architectures.md) | The ten evidence architectures, selection rules, and Stage 1 evaluation |
| [Stage 2 selection](docs/stage2_selection.md) | Evidence, ranking, model choice, nuisances, DINA, diagnostics, and limitations |
| [Configuration](docs/configuration.md) | Configuration keys, command-line flags, launcher, outputs, resume, and reselection |
| [Serving](docs/serving.md) | LLM endpoints, managed vLLM pools, and hardware presets |
| [Optional extensions](docs/optional_extensions.md) | Optional Stage 2 steps and the standalone `oci run` workflows |
| [Synthetic data](docs/synthetic_data.md) | The synthetic cohort generator, bundled cohorts, and oracle columns |
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
