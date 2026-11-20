# Longitudinal causal model

The model estimates the study-defined binary outcome risk for the same patient under
control and treatment, along with predictive uncertainty and propensity scores. It
conditions all estimates on an observational cohort from the same study. Training
episodes supply the known risks under both arms (and treatment propensities) for query
patients. Inference takes observed treatment and outcome for context patients and only
pretreatment (or index-day) histories for query patients.

## Patient and study encoders

Each note passes through a pretrained Hugging Face text encoder — by default
**TinyBERT** (`huawei-noah/TinyBERT_General_4L_312D`, 4 layers, 312 hidden dim,
14.5M parameters), enabling studies up to 5,000 context patients on a single H200.
Pass `--note-model` to swap in another bidirectional encoder (e.g., `prajjwal1/bert-tiny`
or `emilyalsentzer/Bio_ClinicalBERT`) or a causal decoder (e.g.,
`LiquidAI/LFM2.5-230M-Base`, pinned at revision
`9d2be5519834990d30996f878b6771cccbd24f2c`). Rather than crushing an entire note into a
single mean-pooled vector, a trainable token pooler extracts `--note-slots` (default 8)
**multi-query attention-pooled concept vectors** alongside **masked max-pooled** (preserving
rare single-sentence findings regardless of note length) and **token-weighted mean-pooled**
representations over the backbone's contextualized token hidden states. Notes longer than
the backbone's `max_position_embeddings` (512 for TinyBERT) are split into lossless
non-overlapping windows and aggregated via exact cross-window log-sum-exp attention, global
max, and token-weighted mean rather than truncated; empty notes raise
`SemanticTruncationError`. Activation checkpointing trades computation for memory, and
`--note-batch-size` (default 64; overridable at `evaluate` and `predict` time) controls the
microbatch size.

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
`delta_years` optional for real EHR tables). Clinical notes combine their temporal
`EventTransformer` slots with cross-note max-pooled attention and token representations so
findings documented at an earlier visit are never diluted across a long chart. Labs,
vitals, PROs, and encounters each pair their `EventTransformer` summary with an explicit
per-stream longitudinal trajectory summary (`latest`, `mean`, `asinh` OLS slope per year,
observation flags, recency, and utilization counts) so clinical trends such as creatinine
slope and annualized weight loss are preserved directly. A cross-modal timeline transformer
also attends jointly over all chronologically aligned events across all five modalities to
produce a visit-aligned fused token (default `event_width=128`, `event_layers=2`).

To avoid the zero-gradient cohort-mean plateau of pure attention PFNs, `CausalStudyModel`
uses a **hybrid differentiable in-context orthogonalized R-learner plus residual two-way
CausalPFN head**:

1. **Differentiable in-context LOO-CV R-learner.** Patient slots project into dedicated
   note and structured subspaces (`--solver-dim`, default 64) standardized over the context
   cohort. Optionally, `--lexical-dim K` (default 0; e.g. `128`) appends the top-$K$
   unsupervised study-adaptive SVD components of each patient's hashed pretreatment note
   1–3grams (fitted strictly on the context cohort in `float32` and weighted by a learned
   scalar) so wide note designs and unseen cancer vocabularies do not bottleneck on fixed
   slot capacity. Inside `condition()`, exact closed-form Cholesky leave-one-out
   cross-validated Ridge fits estimate propensity $\hat{e}(x)$ and marginal outcome
   $\hat{m}(x)$ (strictly in `float32`), then fit the orthogonalized Robinson R-learner
   $(Y - \hat{m}^{(-i)}(X)) \sim (T - \hat{e}^{(-i)}(X)) \cdot [1, \phi(X)]$ for
   $\hat{\tau}(x)$ with separate learnable penalty grids. Query patients receive point
   estimates $(\hat{y}_0, \hat{y}_1, \hat{e})$ together with exact Bayesian query leverages
   $(h_m, h_\tau, h_e)$, which parameterize the centered log-odds prior over the 129-bin
   sinh histogram and send coherent patient-level gradients back into the encoder and note
   backbone from step 1.
2. **Residual two-way CausalPFN study layers.** Two alternating within-patient slot
   attention and context-only cross-patient sample attention layers (`study_width=192`,
   `study_layers=2`) add a learned residual refinement to the outcome histogram and
   propensity logits while preserving strict query independence and permutation invariance
   in context order.

Inference computes context solver fits and attention memories once and reuses them across
both arms, propensity queries, and all query batches.

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
# Stage 1: escape the cohort-mean plateau with sharp small/moderate contexts and meta-batch=32
python -m oci.longitudinal.cli train --run-dir artifacts/longitudinal_run \
  --steps 600 --grad-accum-steps 32 --min-context-size 64 --context-size 256 \
  --query-size 32 --note-batch-size 256 --note-grad-patients 64 --num-workers 16 \
  --lr 6e-4 --note-lr 3e-5 --lr-scheduler cosine --warmup-steps 30 \
  --eval-every 50 --eval-studies 32 --device cuda

# Stage 2: scale context size to N=500..5,000 once cross-patient attention heads are sharp
python -m oci.longitudinal.cli resume --run-dir artifacts/longitudinal_run \
  --steps 1200 --min-context-size 500 --context-size 5000 --query-size 64 \
  --grad-accum-steps 8 --note-grad-patients 128 --lr 2e-4 --note-lr 1e-5 \
  --eval-every 100 --eval-studies 32 --num-workers 16 --device cuda

python -m oci.longitudinal.cli evaluate --run-dir artifacts/longitudinal_run \
  --studies 1000 --context-size 5000 --query-size 64 --note-batch-size 512 \
  --num-workers 16 --seed 42 --device cuda
```

`resume` accepts curriculum overrides (`--min-context-size`, `--context-size`,
`--query-size`, `--grad-accum-steps`, `--lr`, `--note-lr`, `--eval-every`,
`--eval-studies`, `--freeze-note-backbone` / `--no-freeze-note-backbone`,
`--note-grad-patients`, and `--num-workers`). To scale to large contexts
(`N = 500–5,000`) without backpropagating through the note backbone on tens of
thousands of context notes per step:

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
`zero_cate_rmse`, and an observational LOO-CV structured-feature Ridge T-learner
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
