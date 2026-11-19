# Longitudinal causal model

The model estimates the study-defined binary outcome risk for the same patient under
control and treatment. It conditions both estimates on an observational cohort
from the same study. Training examples supply the known risks under both arms
for query patients. Inference takes observed treatment and outcome for context
patients and only pretreatment histories for query patients.

## Patient and study encoders

Each note passes through the pretrained LiquidAI LFM2.5-230M-Base backbone.
The last non-padding token in its final hidden layer supplies a 1,024-dimensional
note vector. The implementation uses the published base checkpoint at revision
`9d2be5519834990d30996f878b6771cccbd24f2c`, right-pads notes within a batch,
and computes gradients through all its layers. Activation checkpointing trades
computation for memory. `--note-batch-size` controls the number of notes per LFM
forward pass. Notes longer than the backbone's configured capacity raise an
error. [LiquidAI model card](https://huggingface.co/LiquidAI/LFM2.5-230M-Base)

Each lab panel becomes one event with its panel type, scaled numeric components,
and measurement-presence indicators. Vital signs have their own numeric
projection and presence indicators. Both use fixed input scales and an asinh
transform. An encounter combines learned embeddings of its setting, ICD-10-CM
codes, and CPT codes. Codes use 4,096 deterministic hash buckets with separate
namespaces for each system. Collisions are possible.

Each submitted PRO-CTCAE form becomes one event. For each of the form's
11 symptom–attribute pairs, a learned projection reads a response-present
indicator and four ordinal thresholds. For example, fatigue interference of
zero sets only the presence indicator. A score of two also sets the first two
thresholds. An unanswered item leaves all five indicators clear.

Five independently initialized transformers summarize the five event streams.
Each adds a learned embedding of log-days before the index decision, log-days
since the previous event of that modality (both use log1p), and a first-event
indicator. Each returns its learned summary token and masks padding. Empty
streams have a learned representation. The default is two layers, four heads,
and a 64-value summary per modality.

The five summaries are concatenated into a 320-value patient vector. Appending
the treatment indicator gives 321 inputs to the study projection. Observed
context patients also receive an embedding of their standardized outcome.
The four-layer, 128-wide study transformer attends across patients. At every
layer, only context patients supply keys and values, so predictions for a
query are independent of the other queries. Inference computes the context
representations once and reuses them for both arms and all query batches.

## Binary outcome and loss

For arm t, let n be the number of context patients in that arm and k the number
of observed events. Define p = (k + 0.5)/(n + 1) and s = sqrt(p(1-p)).
The model reads each observed binary outcome as (y-p)/s. The half-count keeps
the calculation finite for single-class or absent arms. An absent arm starts
with p = 0.5 and has no direct observational evidence in that study.

The causal head predicts a histogram over the standardized log-odds residual
z = s × (logit(risk) − logit(p)). The factor s follows from locally converting
a one-standard-deviation change on the Bernoulli probability scale to log odds.
Only probabilities enter the logit calculation. During training, both oracle
risks supervise their respective arm queries through Gaussian-smoothed histogram
cross-entropy. The default has 129 bins spanning −6 to +6, one bin spacing of
smoothing, and boundary clipping for extreme targets. During inference, each
bin center is converted back to risk with sigmoid(logit(p) + z/s). The output
is the probability-weighted mean of those risks for each arm and their difference.

## Studies used for training

A training episode has three levels.

1. **Design.** A frozen LLM-authored design lists the cancer population, treatment comparison, binary endpoint and time horizon from its seed question, plus candidate variables and their allowed causal roles. Each clinical question and all its design variants belong to one split. The bundled bank includes two designs for each of eight questions. Six questions are available for training. A colorectal cancer question and the kidney cancer population are held out.
2. **Study.** At each optimizer step Python samples a training design uniformly and draws a fresh treatment-assignment equation and outcome equation. Each candidate takes one of its design's allowed roles. The draw also varies coefficient sizes and signs, baseline event and treatment rates via their intercepts, treatment interactions, and linear, threshold or quadratic outcome terms. An unused candidate still appears in patient records. Some draws can have no effect modifiers.
3. **Patient.** Python samples disjoint context and query patients under the same study equations. The two oracle risks for each query supervise the causal head. At inference the model infers the relationships from observed context patients. Its inputs contain the five histories and treatment indicator. Question text, roles, equations and oracle values are excluded from the patient encoder.

The standard training command uses the bundled bank offline. Pass `--design-bank path/to/bank.json` to train on a larger bank. The checkpoint embeds the complete bank so resume and evaluation use the same authored designs. Evaluation cycles through the held-out designs, draws new equations and patients for each episode, and reports pooled and per-design risk-difference RMSE. Set `--studies` at least as large as the number of held-out designs to cover all of them.

Training logs include the design ID, DGP seed and patient seed. Pass these to the longitudinal simulator to materialize exactly that study. The first context-size rows formed the context. To reproduce the original six-variable prior in a new run, use `--mixed-prior`. Existing checkpoints retain the prior they were trained with. [Design generation and study replay](synthetic_data.md#generating-a-bank-of-study-designs)

## Train, resume, and evaluate

From the repository root, with the project dependencies installed, run:

```bash
python -m oci.longitudinal.cli train --run-dir artifacts/longitudinal_run \
  --steps 1000 --context-size 32 --query-size 8 --device cuda
python -m oci.longitudinal.cli resume --run-dir artifacts/longitudinal_run \
  --steps 2000 --device cuda
python -m oci.longitudinal.cli evaluate --run-dir artifacts/longitudinal_run \
  --studies 32 --context-size 32 --query-size 8 --device cuda
```

The step counts are examples. The base encoder and new layers use separate
AdamW learning rates (defaults 0.00001 and 0.0003). The periodic checkpoint
includes the backbone, other model weights, configuration, optimizer state,
and completed step count. The tokenizer lives beside it. Resume and inference
load these local files. Evaluation reports arm-risk RMSE, risk-difference RMSE
and the context-only empirical arm-rate baseline on the probability scale.

For already materialized studies, including studies with LLM-rendered notes,
repeat `--train-cohort PATH` when starting a run. Each step samples one of
these Parquet files uniformly and samples distinct patients without replacement
for the context and queries. Supply `true_y0` and `true_y1` as conditional
probabilities. For a held-out assessment with this option, use training cohorts
independent of the evaluation studies. Keep the supplied files at the same
paths and with the same contents when resuming.

## Predict

```bash
python -m oci.longitudinal.cli predict --run-dir artifacts/longitudinal_run \
  --context labeled_patients.parquet --queries new_patients.parquet \
  --output predictions.parquet
```

Both input files use the simulator's patient ID and sequence columns for
notes, labs, encounters, vitals, and PROs. For a patient without surveys,
use an empty `pros` list. Training and inference also accept older Parquets
without that column and treat their PRO history as empty.
Version 1 checkpoints load with their four trained encoders and original study
prior for resume, evaluation and prediction. Start a new run to train the fifth encoder.
Context also requires `treatment_indicator` and `outcome_indicator`. Patient IDs
must be unique and disjoint across files. Output columns are `patient_id`,
`risk_control`, `risk_treated`, and `risk_difference` (treated minus control).
For a single cohort, omit `--queries` and supply `--context-size N --seed S` to
draw N context patients and predict for the remainder. `--batch-size` controls
patient encoding and query batches. The selected context still incurs quadratic
attention cost in its number of patients.
