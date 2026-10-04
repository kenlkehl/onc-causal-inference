# Plumb-4B extraction feasibility smoke test

This standalone experiment evaluates `crh225/plumb-4b` on short, hand-written
questions before considering a Stage 2 extraction backend. It does not change
the pipeline. The clinical examples are invented; no patient data is used.

See [the adaptive continuous-value follow-up](plumb_continuous_feasibility.md) for three-pass numeric binning experiments.

## Results: 2026-10-04

All 24 original questions had the expected highest-probability answer: **8/8
choice, 8/8 noul, and 8/8 score**. All eight reversed-choice variants also kept
the expected label. All distributions were valid, and the three repeated
answers for each variant were identical.

- Median decision latency: **53.5 ms**; 95th percentile: **61.2 ms** across the
  72 measured original-question calls. Inputs were 116–183 tokens including
  the complete prompt. Model loading and warm-up are excluded.
- GPU: physical GPU 1, NVIDIA RTX A6000, BF16, batch size one, shared with an
  existing vLLM worker. PyTorch peak allocated memory during measured inference
  was **7.87 GiB**, with **8.13 GiB** reserved. These exclude other processes.
- Eager execution, no CUDA graphs. Flash linear attention was installed;
  `causal_conv1d` was absent, so that component used the correct reference
  PyTorch implementation. This is a working baseline, not a maximally optimized
  speed benchmark or a comparison against the existing extractor.
- Loading took 13.6 seconds; warm-up of the original input shapes took 34.1
  seconds. No generated tokens were used.
- Ordinal expected-score mean absolute error was **0.0648 levels** at T=2.07.
  Retempering the same probabilities to T=1.2 reduced it to **0.00690** on these
  eight examples; this does not demonstrate better calibration generally.

**Order sensitivity:** the parcel example retained `wrong_address` as its top
answer after reversing the choices, but its probability fell from **0.712 to
0.462**. A confidence threshold could therefore change whether the same item
is accepted. Stable selected labels do not establish stable probabilities.

The synthetic clinical questions correctly handled former smoking, unrecorded
smoking status, explicitly absent brain metastases, family history that does
not establish the patient's own status, and pretreatment versus follow-up
ECOG values. These results support a next experiment on fixed categorical and
ordinal extraction tasks. They are too small and simple to justify replacing
the Stage 2 extractor without a representative evaluation.

### All original questions

`P(top)` is the model's probability for the selected answer. For noul, the API
returns P(true); the full raw output retains it. For score, the returned value
is the expectation over zero-based levels, alongside the modal level.

| Case | Expected | Selected | P(top) | Returned score |
|---|---|---|---:|---:|
| `choice_delivery` | wrong_address | wrong_address | 0.7116 | — |
| `choice_routing` | billing | billing | 0.9645 | — |
| `choice_smoking_former` | former | former | 0.9208 | — |
| `choice_smoking_missing` | not_documented | not_documented | 0.9283 | — |
| `choice_explicit_absence` | absent | absent | 0.9015 | — |
| `choice_family_history` | not_documented | not_documented | 0.8975 | — |
| `choice_temporal_scope` | 1 | 1 | 0.8979 | — |
| `choice_unit_conversion` | medium | medium | 0.9338 | — |
| `noul_conjunction_true` | true | true | 0.9344 | — |
| `noul_conjunction_false` | false | false | 0.8863 | — |
| `noul_negation` | false | false | 0.9306 | — |
| `noul_exception` | true | true | 0.9447 | — |
| `noul_clinical_positive` | true | true | 0.9380 | — |
| `noul_clinical_negative` | false | false | 0.8979 | — |
| `noul_missingness_gate` | false | false | 0.9180 | — |
| `noul_explicit_measurement` | true | true | 0.9380 | — |
| `score_severity_none` | 0 | 0 | 0.9643 | 0.0545 |
| `score_severity_minor` | 1 | 1 | 0.9492 | 1.0171 |
| `score_severity_major` | 2 | 2 | 0.9197 | 1.9600 |
| `score_severity_critical` | 3 | 3 | 0.9445 | 2.9190 |
| `score_rubric_count` | 2 | 2 | 0.8580 | 1.8728 |
| `score_pain_absent` | 0 | 0 | 0.9505 | 0.0946 |
| `score_pain_moderate` | 2 | 2 | 0.9411 | 1.9681 |
| `score_ecog_temporal` | 1 | 1 | 0.9382 | 1.0723 |

### Saved evidence

- `experiments/plumb_feasibility/questions.json`: every complete question and
  expected answer, written before inference.
- `artifacts/plumb_feasibility/results/results.jsonl`: all 32 question variants,
  full distributions, expected answers, returned values, and per-repeat timings.
- `artifacts/plumb_feasibility/results/summary.json`: aggregate results,
  settings, versions, fixture hash, and provenance.
- `artifacts/plumb_feasibility/results/run.log`: full execution output.
- `artifacts/plumb_feasibility/weight_verification.json`: confirmation that the
  8,411,558,400-byte weights match Hugging Face's published SHA-256,
  `89e119ea07f4c5b4b6715560c7de6694ec3b777dfd0e62351da0b33986d2e1e1`.

## Protocol

- 24 questions: eight `choice`, eight `noul`, and eight `score`.
- Expected answers frozen in `experiments/plumb_feasibility/questions.json`
  before running inference; gold labels are never included in prompts.
- Eight additional choice questions repeat the same evidence with option order
  reversed to detect sensitivity to answer position.
- Three timed repetitions per question, after warm-up of the input shapes.
- Original JevK5 v0.2.0 runtime, BF16, batch size one, one forward pass per
  decision, zero generated tokens. No sampling or chain of thought.
- The model's temperature, 2.07, is used for all three question types.
- The optional v5.2 `noul-commit` transformation is disabled: reported
  probabilities retain the model's uncertainty.
- Score results include both the most probable level and the runtime's actual
  returned score (the expected zero-based level). A separate sensitivity
  calculation retempers the same distribution to the model card's score
  temperature of 1.2; this is not an additional inference or a fitted parameter.
- Model files, exact revisions, raw outputs, dependency versions, and weight
  checksum verification are kept under `artifacts/plumb_feasibility/`.

This is a small convenience sample, not a random dataset sample, a clinical
validation, or evidence about probability calibration. Latency is measured
on an A6000 shared with an existing vLLM worker.

## Reproduction

Run from the repository root. The experiment uses its own environment rather
than altering project dependencies. The initial setup is:

```bash
mkdir -p artifacts/plumb_feasibility/vendor
git clone https://github.com/allebee/jevk5.git artifacts/plumb_feasibility/vendor/jevk5
git -C artifacts/plumb_feasibility/vendor/jevk5 checkout 85238d7be5527370c43206fe54cd752eb3134c1b
uv venv --python 3.13 artifacts/plumb_feasibility/venv
uv pip install --python artifacts/plumb_feasibility/venv/bin/python \
  'torch==2.13.0' 'transformers==5.17.0' 'tokenizers==0.23.2' \
  'accelerate==1.14.0' 'huggingface-hub==1.33.0' \
  'safetensors==0.8.0' einops 'flash-linear-attention==0.5.2' \
  artifacts/plumb_feasibility/vendor/jevk5
artifacts/plumb_feasibility/venv/bin/hf download crh225/plumb-4b \
  --revision 24f7bf77e7ee258a2d158c61ea2dce2b60321010 \
  --local-dir artifacts/plumb_feasibility/model
```

Exact installed versions from the completed run are saved in
`artifacts/plumb_feasibility/results/environment.txt`. Use that version record
when reproducing the measured environment. Model provenance is also captured in
`artifacts/plumb_feasibility/model_revision.json`.

On this machine, slow project-filesystem writes made a full new installation
impractical. The experiment's `venv` is a link to
`/tmp/plumb-feasibility-20261004`, containing its own Transformers, tokenizers,
and JevK5, with the host's existing `/home/klkehl/thisenv` libraries inherited
read-only through a `.pth` file. Neither the project environment nor the host
environment was changed. The setup commands above create a self-contained
replacement if the temporary environment is cleared. The weights are stored
in the local Hugging Face cache and linked from the experiment's model folder;
the exact location is recorded in `model_storage.json`.

```bash
artifacts/plumb_feasibility/venv/bin/python scripts/plumb_smoke_test.py --gpu 1
```

The script sets `CUDA_VISIBLE_DEVICES` before importing PyTorch, pins the entire
model to that GPU, fails when less than 12 GiB is free, and performs offline
inference using the downloaded weights. It does not stop other GPU processes.
Use `--graphs --output artifacts/plumb_feasibility/results_graphs` to measure
the runtime's CUDA graph path separately; eager execution is the default.

## Implications for Stage 2

The current measurement ontology supports binary, categorical, ordinal, and
continuous features. Candidate discovery and ontology definition are separate
from patient-level extraction (see `lessons/stage2_pipeline_overview.md`).

Plumb's closed option set naturally maps to categorical and binary extraction.
For fields with missing values, a `choice` question with explicit present,
absent, and not-documented options is safer to evaluate than treating every
negative binary answer as clinical absence. The runtime supports at most 16
options, including any missingness category.

For ordinal features, the highest-probability level and the expected score are
different measurements. Preserve the ordered category mapping and use the
former when the ontology requires a discrete category. The `score` interface
does not extract arbitrary continuous values such as age, weight, or a lab
result; it computes an expectation over supplied ordinal levels. Continuous
fields would need a separate extraction method or an explicitly evaluated
change in measurement definition.

The decision output also lacks the exact evidence spans and citation offsets
used by the repository's complete-note extraction contracts. Long-note paging,
conflicting evidence, frozen fold-specific ontologies, provenance, and
missingness must be evaluated before integration. These short questions do not
establish that Plumb can replace all Stage 2 extraction or its generative
candidate-discovery and ontology-definition steps.

Plumb remains a fine-tuned Qwen-family language model internally. The relevant
change is a single-pass option readout instead of autoregressive text/JSON
generation.

## Sources

- [Plumb model card](https://huggingface.co/crh225/plumb-4b)
- [Pinned model files](https://huggingface.co/crh225/plumb-4b/tree/24f7bf77e7ee258a2d158c61ea2dce2b60321010)
- [Pinned JevK5 runtime](https://github.com/allebee/jevk5/blob/85238d7be5527370c43206fe54cd752eb3134c1b/jevk5/runtime.py)
- [Plumb v5.2 output transformations](https://github.com/crh225/plumb/blob/0e6d558d3ea3002c300ce121db4d3e8df6ad792a/serve/plumb_server.py)
