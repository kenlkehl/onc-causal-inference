# Longitudinal causal model

The model estimates the study-defined binary outcome risk for the same patient under
control and treatment, along with predictive uncertainty and propensity scores. It
conditions all estimates on an observational cohort from the same study. Training
episodes supply the known risks under both arms (and treatment propensities) for query
patients. Inference takes observed treatment and outcome for context patients and only
pretreatment (or index-day) histories for query patients.

## Patient and study encoders

Each note passes through the pretrained LiquidAI LFM2.5-230M-Base backbone.
The last non-padding token in its final hidden layer supplies a 1,024-dimensional
note vector. The implementation uses the published base checkpoint at revision
`9d2be5519834990d30996f878b6771cccbd24f2c`, right-pads notes within a batch,
and computes gradients through all its layers. Activation checkpointing trades
computation for memory. `--note-batch-size` controls the number of notes per LFM
forward pass (and can be overridden at `evaluate` and `predict` time). Notes longer
than the backbone's configured capacity raise `SemanticTruncationError`.
[LiquidAI model card](https://huggingface.co/LiquidAI/LFM2.5-230M-Base)

Each lab panel encodes canonical CBC/BMP/CMP components with fixed reference scales,
an `asinh` transform, and presence indicators. Additional real-EHR panels, components,
or alternative units are encoded through open-vocabulary hashed embeddings rather than
rejected. Vital signs use fixed `asinh` numeric projections and presence indicators.
Encounters combine mean-pooled learned embeddings of setting, exact ICD-10-CM and CPT
codes, and 3-character clinical category prefixes across 4,096 hash buckets, plus a
projected `log1p(code_count)` term so norms remain stable on real EHR encounters with
many codes.

Each submitted PRO form encodes the 11 canonical adult PRO-CTCAE v1.0 items via a
presence indicator and four cumulative ordinal thresholds, while additional PRO-CTCAE
items or other instruments (such as EORTC QLQ-C30) map into open-vocabulary hashed
embeddings.

Five modality-specific transformers summarize the five event streams using a learned
clock embedding of `log1p(-day_from_index)`, `log1p(elapsed_days)`, and a first-event
indicator (accepting events on or before the index day, `day_from_index <= 0`, with
`delta_years` optional for real EHR tables). Clinical notes produce `--note-slots`
(default 4) learned concept tokens so high-dimensional chart findings are not crushed
into a single vector, while labs, encounters, vitals, and PROs produce one summary
token each. A cross-modal timeline transformer also attends jointly over all
chronologically aligned events across all five modalities to produce a visit-aligned
fused token (default `event_width=128`, `event_layers=2`).

The resulting patient tokens (`note_slots + 5` slots, projected to `study_width=192`)
combine with a dedicated treatment/outcome token that carries treatment status,
standardized observed outcome (for context), smoothed empirical arm log-odds, and
`log1p(arm_count)`. Four two-way CausalPFN study layers alternate:

1. **Within-patient slot attention** across each patient's modality/concept tokens and
   its treatment/outcome token.
2. **Across-patient sample attention** per slot, where only context patients supply
   keys and values so query predictions are strictly independent of other queries and
   permutation-invariant in context order.

Inference computes context representations once and reuses them across both arms,
propensity queries, and all query batches.

## Binary outcome, propensity, and loss

For arm $t \in \{0, 1\}$, let $n_t$ be the number of context patients in that arm and
$k_t$ the number of observed events. Define $p_t = (k_t + 0.5)/(n_t + 1)$ and
$s_t = \sqrt{p_t(1-p_t)}$. Context tokens observe $(y - p_t)/s_t$ alongside
$\text{logit}(p_t)$ and $\log(1 + n_t)$.

The outcome head predicts a 129-bin histogram over centered log-odds offsets
$\delta = \text{logit}(\text{risk}) - \text{logit}(p_t)$ on a non-uniform
$\sinh$-warped grid spanning $[-6, +6]$ logits (concentrating resolution in
$[-2.5, +2.5]$ logits while covering extreme risks). During training, the objective
combines:

1. Local-width Gaussian-smoothed histogram cross-entropy on both oracle arm risks
   (`true_y0`, `true_y1`).
2. Direct CATE mean-squared error on $(\hat{p}_1 - \hat{p}_0) - (p_1 - p_0)$ weighted
   by `--cate-loss-weight` (default 2.0).
3. Propensity binary cross-entropy from a treatment-masked query head against
   `true_treatment_prob` weighted by `--propensity-loss-weight` (default 0.5).

During inference, the histogram yields both the posterior mean and standard deviation
(`risk_control`, `risk_treated`, `risk_control_sd`, `risk_treated_sd`), the
`risk_difference`, the estimated `propensity_score`, and an `outside_overlap` flag
when propensity falls outside $[0.05, 0.95]$.

## Studies used for training

A training episode has three levels.

1. **Design.** A frozen LLM-authored design lists the cancer population, treatment
   comparison, binary endpoint and time horizon from its seed question, plus candidate
   variables and permitted causal roles. The bundled Nemotron bank contains 5,000
   designs spanning 100 questions and 20 cancers (4,000 train, 1,000 held-out, including
   three cancers absent from training). Colliding chart statements within a design are
   deduplicated automatically at study-draw time. [Generation record and replay instructions](nemotron_design_bank.md)
2. **Study.** Each step samples a training design uniformly, draws permitted roles
   (capping wide designs to a sparse active subset of at most 16 causal variables while
   keeping all remaining candidates in the record as distractors, and preventing
   all-unused draws), samples linear/threshold/quadratic terms plus bounded pairwise
   treatment and outcome interactions, and calibrates linear-predictor SDs and
   intercepts by Monte Carlo so treatment propensities and event rates stay well-behaved
   across widths 3–100.
3. **Patient.** Python samples disjoint context and query patients under the same study
   equations, coupling overlapping chart findings to simulated physiology, correlating
   other findings with patient burden, and varying note framing and positive-only
   charting (`--augment-notes`) so the encoder adapts to real-EHR documentation patterns.

The checkpoint writes the design bank once to `design_bank.json.gz` beside
`checkpoint.pt` (verified by SHA-256) rather than duplicating it inside every
checkpoint.

## Train, resume, and evaluate

From the repository root, with the project dependencies installed, run:

```bash
# Phase 1: train downstream cross-modal and CausalPFN layers across large contexts (N=128..512)
python -m oci.longitudinal.cli train --run-dir artifacts/longitudinal_run \
  --steps 1000 --grad-accum-steps 8 --min-context-size 128 --context-size 512 \
  --query-size 32 --freeze-note-backbone --num-workers 8 --lr-scheduler cosine \
  --warmup-steps 100 --eval-every 200 --device cuda

# Phase 2: unfreeze the note LLM with partial-gradient context encoding
python -m oci.longitudinal.cli resume --run-dir artifacts/longitudinal_run \
  --steps 2500 --no-freeze-note-backbone --note-grad-patients 64 --num-workers 8 \
  --device cuda

python -m oci.longitudinal.cli evaluate --run-dir artifacts/longitudinal_run \
  --studies 1000 --context-size 512 --query-size 32 --num-workers 8 --seed 42 --device cuda
```

To scale to large contexts (`N = 256–1,024+`) without backpropagating through `LFM2.5-230M`
on thousands of context notes per step:

- `--freeze-note-backbone` runs the note LLM in fast `no_grad` mode (4× larger note
  microbatches, no activation checkpointing) while training all downstream modality and
  study layers; `resume --no-freeze-note-backbone` unfreezes it cleanly with the saved
  optimizer groups intact.
- `--note-grad-patients K` backpropagates through the note LLM on all query patients
  plus `K` randomly sampled context patients per episode while encoding the remaining
  context notes under `no_grad` (all downstream modality and study layers still receive
  full gradients across every patient).
- `--num-workers W` runs CPU study simulation in a spawned background process pool,
  overlapping episode generation with GPU compute in exact deterministic order.

AdamW starts the note encoder at `--note-lr 1e-5` (default `--backbone-weight-decay 0.0`)
and the downstream layers at `--lr 3e-4` (`--weight-decay 0.01` on 2D+ weight matrices
only; biases, LayerNorms, embeddings, and summary tokens receive zero weight decay).
`--precision auto` enables `bfloat16` autocast on supported CUDA GPUs while keeping
master weights and losses in `float32`. `--lr-scheduler` supports `constant`, `linear`,
and `cosine` with optional `--warmup-steps`. Every step appends structured JSON metrics
to `metrics.jsonl` in `--run-dir`; `--eval-every` runs periodic held-out scoring and
writes `best_checkpoint.pt`.

Evaluation reports `risk_rmse`, `risk_brier`, `propensity_rmse`, `cate_rmse`,
`ate_rmse`, `ate_bias`, `cate_pearson`, `cate_spearman`,
`mean_within_study_cate_pearson`, three baselines (`arm_mean_cate_rmse`,
`zero_cate_rmse`, and an observational structured-feature Ridge T-learner
`linear_t_learner_cate_rmse`), stratification `by_width` and `by_generalization`
(`unseen_cancer` vs `seen_cancer_new_question`), and per-design RMSE. Pass
`--eval-cohort PATH` (repeatable) to evaluate on materialized Parquet cohorts such as
LLM-rendered studies.

## Predict

```bash
python -m oci.longitudinal.cli predict --run-dir artifacts/longitudinal_run \
  --context labeled_patients.parquet --queries new_patients.parquet \
  --output predictions.parquet --note-batch-size 32
```

Both input files use `patient_id` and sequence columns for `notes`, `labs`,
`encounters`, `vitals`, and optional `pros`. Events may occur on or before the index
day (`day_from_index <= 0`), and `delta_years` is optional. Context also requires
`treatment_indicator` and `outcome_indicator`. Patient IDs must be unique within each
file and disjoint across files; context and query files may use different Arrow ID
types. Output columns are `patient_id`, `risk_control`, `risk_treated`,
`risk_difference`, `risk_control_sd`, `risk_treated_sd`, `propensity_score`, and
`outside_overlap`. If either arm has fewer than five context patients, `predict` warns
on `stderr`.
