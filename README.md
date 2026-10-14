# Oncology Causal Inference (OCI)

OCI is a research pipeline for estimating how treatment effects vary between
patients using unstructured electronic health record text, with a focus on
oncology. It discovers clinical characteristics in notes, turns them into
explicit patient-level measurements, and uses those measurements to estimate
conditional average treatment effects (CATEs), often called individualized
treatment-effect predictions, alongside an average treatment effect (ATE).

The motivating question is: **which of two treatments is expected to produce a
better outcome for a patient with these clinical characteristics?**

## How the pipeline works

1. **Stage 1 — discover evidence.** Complementary lexical, neural, embedding,
   topic, and matched-patient models identify potentially relevant text within
   training folds. Eight evidence architectures run by default; two additional
   lexical architectures are optional.
2. **Stage 2 — define and measure variables.** An LLM interprets evidence cards,
   consolidates duplicate variable names, defines measurement rules, and extracts
   values from each patient's record. Validation, ontology repair, and value
   harmonization precede statistical selection.
3. **Stage 2 — select and estimate.** The `multi_model` path builds statistical
   evidence, selects treatment/outcome nuisance inputs by predictive regularization,
   and ranks modifiers programmatically. Nested validation selects the modifier count and final
   estimator. Frozen definitions and fitted models produce outer-held-out CATE
   predictions; separate nuisance models supply AIPW estimates of the ATE.

```mermaid
flowchart LR
    A["Clinical text + treatment + outcome"] --> B["Stage 1 evidence"]
    B --> C["Clinical variables and extraction"]
    C --> D["Statistical evidence and role review"]
    D --> E["Modifier ranking and model selection"]
    E --> F["Held-out effects and diagnostics"]
```

All supplied clinical text is assumed to be pretreatment. A **confounder** is
used for treatment-comparison adjustment; an **effect modifier** allows the
estimated effect to vary across patients. A variable can serve both roles.
Automated role assignments do not establish causal identification: interpretation
still depends on the study's confounding, overlap, consistency, and measurement
assumptions. Oracle variables in synthetic datasets are reserved for post-hoc
evaluation after predictions are frozen.

For a detailed walkthrough, read [How Stage 2 works](how_stage2_works.md).
The [technical reference](docs/all_evidence_reference.md#the-ten-stage-1-evidence-architectures)
describes the individual Stage 1 architectures.

## Installation

Python 3.12 and 3.13 are supported. From a fresh checkout of `main`:

```bash
git clone https://github.com/kenlkehl/onc-causal-inference.git
cd onc-causal-inference
uv sync --frozen
```

The full Stage 1 workflow uses NVIDIA CUDA GPUs. Stage 2 can use external
OpenAI-compatible servers or pipeline-managed vLLM. For managed local servers:

```bash
uv sync --frozen --extra local-llm
.venv/bin/python scripts/configure_local_cuda.py
```

The host must supply a compatible NVIDIA driver and required system libraries.
See the [managed-server setup](docs/all_evidence_reference.md#pipeline-managed-vllm-server-pools)
for hardware-specific installation, model caching, ports, and troubleshooting.
Editable installation is also supported with `pip install -e .` and the
appropriate optional extras from [pyproject.toml](pyproject.toml).

## Run your own cohort

Use [`scripts/run_all_evidence.py`](scripts/run_all_evidence.py) for the complete
Stage 1 → Stage 2 workflow. Input may be CSV or Parquet, with one row per patient
or treatment decision and four configurable columns:

| Column | Content |
| --- | --- |
| `patient_id` | Stable unit identifier |
| `clinical_text` | Prepared pretreatment clinical record |
| `treatment_indicator` | Binary treatment, coded 0/1 |
| `outcome_indicator` | Binary or continuous observed outcome |

Copy the multi-model example **inside `example_configs/`**, because relative
paths resolve against the configuration file's directory:

```bash
cp example_configs/research_all_evidence_multi_model.json example_configs/my_run.json
```

Edit the dataset and output paths, column names, clinical question, outcome type,
fold settings, and `run.devices`. Configure both `stage2.endpoint` and
`stage2.extraction_llm.endpoint`, including `/v1`. Both roles may share one
server. Leave model IDs empty to discover a single advertised model, or set
their exact served IDs. Match request concurrency and context/output limits to
the servers you provide.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
uv run python scripts/run_all_evidence.py --config example_configs/my_run.json
```

This example enables `multi_model` selection, matched-batch evidence,
post-extraction measurement consolidation, and propensity eligibility bounds
of 0.1–0.9. Estimand-informed ontology search and modifier-concept review are
optional and disabled in this example. Inspect its settings before adapting it
to a study; the bundled synthetic launchers use a simpler preset described below.

Use `--stage1-only` to stop at the evidence handoff. Use `--architectures
bow_nuisance,tfidf_topics`, for example, to select a Stage 1 subset, or
`--architectures all` for all ten supported architectures. Preserve the
architecture selection when resuming. See the [command guide](docs/all_evidence_quickstart.md)
for additional examples.

## Run a bundled synthetic cohort

These launchers run the existing bundled datasets through both stages; they do
not generate new synthetic patients. Pass an output directory to keep each run
separate, and repeat the same command to resume compatible checkpoints.

| Environment | Launcher |
| --- | --- |
| External LLM servers; one confounder, one modifier | [`run_one_conf_one_mod.sh`](run_one_conf_one_mod.sh) |
| External LLM servers; five confounders, five modifiers | [`run_five_conf_five_mod.sh`](run_five_conf_five_mod.sh) |
| Eight H100s; managed servers, one confounder/one modifier | [`run_one_conf_one_mod_h100x8.sh`](run_one_conf_one_mod_h100x8.sh) |
| Eight RTX PRO 6000 Blackwell GPUs; managed servers, one confounder/one modifier | [`run_one_conf_one_mod_rtxpro6000x8.sh`](run_one_conf_one_mod_rtxpro6000x8.sh) |
| Eight RTX PRO 6000 Blackwell GPUs; managed servers, five confounders/five modifiers | [`run_five_conf_five_mod_rtxpro6000x8.sh`](run_five_conf_five_mod_rtxpro6000x8.sh) |

For the RTX PRO 6000 × 8 setup:

```bash
./run_five_conf_five_mod_rtxpro6000x8.sh /persistent/results/five_conf_five_mod
```

This preset uses `nvidia/Gemma-4-26B-A4B-NVFP4` for both Stage 2 roles, sharing
one server per GPU across all eight GPUs. Fresh launches use a shared
128-worker CPU modeling budget. The one-confounder RTX launcher uses the same
hardware preset. Stage 1 finishes before the Stage 2 LLM servers start.

For servers you already run, override both endpoints and model IDs:

```bash
STAGE2_ENDPOINT=http://127.0.0.1:8010/v1 \
STAGE2_MODEL=YOUR_PRIMARY_SERVED_MODEL_ID \
STAGE2_EXTRACTION_ENDPOINT=http://127.0.0.1:8020/v1 \
STAGE2_EXTRACTION_MODEL=YOUR_EXTRACTION_SERVED_MODEL_ID \
STAGE2_WORKERS=4 STAGE2_EXTRACTION_WORKERS=4 \
./run_five_conf_five_mod.sh /persistent/results/my_run
```

Fresh synthetic launches select `multi_model`. Matched-batch evidence,
post-extraction consolidation, estimand-informed ontology search, and
modifier-concept review remain opt-in for these launchers. Explicit saved-run
launches retain their saved scientific and serving settings.

GPU selection, environment overrides, replica pools, and model-switching behavior
are documented in the [technical reference](docs/all_evidence_reference.md#bundled-synthetic-wrappers).
Set `OCI_PYTHON=/path/to/python` to reuse an installed environment without
dependency synchronization.

## Resume, inspect, or rerun selection

The output directory contains the resolved `run_config.json`, checkpoints,
`progress.json`, and `logs/workflow.log`. Inspect status without starting work:

```bash
uv run python scripts/run_all_evidence.py \
  --config /path/to/output/run_config.json --status
```

Resume Stage 2 from its saved configuration:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
uv run python scripts/run_all_evidence.py \
  --config /path/to/output/run_config.json --stage2-only
```

To intentionally recompute selection and estimation after a **completed** Stage 2
run, add `--stage2-reselect`. This guarded operation archives prior downstream
results, preserves validated training measurements and definitions, and reuses
compatible held-out measurements. It does not repeat Stage 1 modeling.
Use `--preflight-only` with the intended Stage 2 command to validate reuse before
starting work.

Runtime changes such as worker counts can preserve compatible checkpoints.
Changes to data, definitions, objectives, models, or other scientific settings
can invalidate them. Use a new output directory for a different study and the
[resume/reselection guide](docs/all_evidence_reference.md#output-and-interruption-recovery)
for controlled reuse of existing artifacts.

## Results

The all-evidence runner writes these primary outputs:

| Path under the run directory | Contents |
| --- | --- |
| `handoff/evidence.jsonl` | Frozen Stage 1 evidence passed to Stage 2 |
| `stage2/outer_*/final_definitions.json` | Each fold's retained variables and roles |
| `stage2/outer_*/selection/modifier_count/` | Rankings, model/count comparisons, and selection audits |
| `stage2/outer_*/estimation/diagnostics.json` | Chosen architecture, fit diagnostics, and fold estimates |
| `stage2/cross_fitted_predictions.csv` | Combined outer-held-out predictions |
| `stage2/causal_estimate.json` | Aggregate causal-estimation summary |
| `stage2/posthoc_oracle_ite_metrics.json` | By-fold and pooled oracle comparisons, when truth is available |

`estimated_cate` is the chosen effect model's prediction. The common `mu0` and
`mu1` columns are separate AIPW nuisance predictions, so their difference need
not equal `estimated_cate`. The AIPW ATE and the mean predicted CATE are also
distinct outputs. Individual CATE intervals are available for the forest path;
they are not currently computed for the interaction or DINA estimators.

Assess nuisance calibration, overlap, validation loss, measurement quality, and
fold variability together. For synthetic data, oracle ITE correlations are
computed only after all outer predictions are frozen; oracle truth does not
guide model selection.

## Technical details and optional paths

**DINA is part of `main`.** Binary-outcome effect discovery and orthogonal
modifier models use the Gao–Hastie Bernoulli DINA objective. In `multi_model`,
final architecture search compares causal forests, penalized interaction
outcome models, and DINA using common probability-scale validation R-loss.
DINA models a conditional log odds ratio and converts it to a risk difference
for CATE output. Continuous outcomes retain their corresponding non-DINA paths.
See the [DINA guide](docs/gao_hastie_dina.md) for equations and interpretation.

| Setting | Current behavior |
| --- | --- |
| Extraction | Full-record LLM extraction; up to 10 variables from one patient per request; lossless chunks for long records |
| Extraction reasoning | Initially off; error-directed repairs retain that setting through the configured five-repair threshold, then enable thinking on the same model |
| Modifier ranking | Equal default weight for nuisance-adjusted interaction tests and combined predictive evidence; optional clinical-concept review is separate |
| Modifier count | Interaction support chooses among budgets within a paired validation R-loss guardrail (`balanced_evidence`) |
| Nuisance selection | Separate predictive treatment/outcome supports; no LLM retention decisions in `multi_model` |
| Treatment main effect | Unpenalized in Stage 2 interaction models and continuous R-learners; DINA also leaves its constant treatment effect unpenalized |
| Selection mode | Fresh synthetic launchers and the multi-model example use `multi_model`; an omitted core setting still means `llm_roles` |
| CPU modeling | `stage2.modeling_workers` is shared across folds; core default 1, fresh RTX × 8 preset 128 |

Optional extraction backends include [ColBERT retrieval](docs/colbert_extraction.md),
[Plumb decision extraction](docs/stage2_decision_extraction.md), and
[note search](docs/stage2_note_search.md). For root launchers, ColBERT is selected
with `STAGE2_EXTRACTION_CONTEXT_STRATEGY=colbert`; Plumb is selected with
`STAGE2_DECISION_EXTRACTION=1`. These are alternatives to the default LLM path.

Other optional workflows include [independent numerical task selection](docs/stage2_independent_tasks.md),
[estimand-informed ontology refinement](docs/stage2_estimand_ontology.md), and
[standalone explicit-feature models](docs/all_evidence_reference.md#standalone-explicit-feature-workflows).
The standalone models use `oci run`; the all-evidence pipeline uses
`scripts/run_all_evidence.py`.

## Synthetic data and evaluation

The [`synthetic_data/`](synthetic_data/) package generates clinical narratives
and optional structured events with known treatment/outcome equations,
confounders, modifiers, and treatment effects.

To generate more patients from an existing dataset's **same saved causal
process**, point the generator at its configuration and metadata and an
OpenAI-compatible note-generation server:

```bash
uv run python -m synthetic_data.cli \
  --config /data/original/generation_config.json \
  --dgp-metadata /data/original/metadata.json \
  --dataset-size 10000 \
  --model nvidia/Gemma-4-31B-IT-NVFP4 \
  --api-url http://localhost:8000/v1 \
  --num-workers 32 \
  --output-dir /data/expanded_cohort
```

This preserves feature distributions and calibrated equations while generating
new patients and notes. Successful requests and patients are checkpointed;
repeat the command to resume. `progress.json` reports completion counts.

To evaluate saved Stage 1 evidence against synthetic truth without refitting:

```bash
uv run oci-evaluate-stage1 \
  --run-dir /results/my_run \
  --metadata /data/synthetic/metadata.json \
  --architectures all
```

Generation options, endpoint replication, and architecture-specific evaluation
are covered in the [synthetic-data reference](docs/all_evidence_reference.md#synthetic-data-and-stage-1-architecture-evaluation).

## Documentation

| Guide | Scope |
| --- | --- |
| [How Stage 2 works](how_stage2_works.md) | Detailed step-by-step walkthrough with the 1,000-patient example |
| [Quickstart](docs/all_evidence_quickstart.md) | Commands and configuration examples |
| [Workflow guide](docs/all_evidence_workflow.md) | Orchestration, artifact reuse, and request handling |
| [Technical reference](docs/all_evidence_reference.md) | Stage 1 architectures, managed GPU setups, extraction, and artifact layouts |
| [Stage 2 multi-model selection](docs/stage2_multi_model.md) | Evidence stability, ranking, and nested architecture/count selection |
| [DINA](docs/gao_hastie_dina.md) | Binary objectives, effect scales, and limitations |
| [LLM sampling](docs/stage2_sampling.md) | Model-specific sampling and reasoning controls |

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
