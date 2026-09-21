# All-evidence quickstart

For Stage 2 selection across multiple model families and repeated subsets, use
`example_configs/research_all_evidence_multi_model.json` and read the
[multi-model design and configuration guide](stage2_multi_model.md). This is an
explicit mode; the existing example and omitted-selector behavior are unchanged.

1. Copy the example config:

   ```bash
   cp example_configs/research_all_evidence.json my_all_evidence_run.json
   ```

2. Set `dataset`, `output_dir`, the four column names, model locations, devices,
   and the fold/seed parameters.

3. Run:

   ```bash
   uv run python scripts/run_all_evidence.py --config my_all_evidence_run.json
   ```

4. Watch progress:

   ```bash
   uv run python scripts/run_all_evidence.py \
     --config my_all_evidence_run.json --status
   tail -f /path/to/output/logs/workflow.log
   ```

5. If the process stops, run step 3 again. Completed components are reused and
   reported as `already_complete`.

Use `--stage1-only` to stop at the handoff or `--stage2-only` to consume an
existing handoff. Setting `stage2.endpoint` or `stage2.vllm` in the config makes
an unflagged invocation run both phases. `stage2.model` may be omitted when an
external endpoint's `/models` API advertises exactly one model ID; it is required
when the pipeline launches vLLM itself. Dataset-backed Stage 2 always requires a
`stage2.extraction_llm` configuration. It may point to a different endpoint or
the same multi-model endpoint; its model is auto-discovered only when that
endpoint advertises exactly one model.

To run a scientific subset of Stage 1, add (for example)
`--architectures bow_nuisance,tfidf_topics`. Private prerequisites are resolved
automatically and Stage 2 receives only those selected lanes. Keep the same
selection when resuming; changing it requires a fresh output directory. Omit
the option for legacy all-enabled behavior.

Stage 2 does not stop at variable definitions. For each outer fold it exhaustively
lists clinical features from every semantic evidence card, performs merge-only
consolidation, uses a separate small model to extract all candidates on training
records, and lets the primary model review only aggregate extraction ontologies.
An optional equivalence-only pass consolidates extracted aliases before grouped
elastic nets and R-loss models generate selection evidence. The default
`llm_roles` mode uses primary-model adjudication; `independent_tasks` selects
treatment, outcome, and effect inputs numerically. A causal forest is then fit
and evaluated on outer-held-out records. The common controls are:

```json
{
  "stage2": {
    "endpoint": "http://127.0.0.1:8010/v1",
    "model": "Qwen/Qwen3.8-27B",
    "workers": 32,
    "extraction_llm": {
      "endpoint": "http://127.0.0.1:8020/v1",
      "model": "small-extractor",
      "workers": 32
    },
    "request_timeout": 7200,
    "request_attempt_timeout": 900,
    "transport_max_attempts": 6,
    "max_tokens": 100000,
    "extraction_max_tokens": 4096,
    "extraction_reasoning_max_tokens": 32768,
    "extraction_stream": true,
    "extraction_deferred_retry_passes": 1,
    "max_response_repairs": 15,
    "thinking_after_response_repairs": 5,
    "repetition_penalty": null,
    "interpretation_reasoning_effort": "high",
    "extraction_reasoning_effort": "none",
    "evidence_compiler": "semantic_cluster_cards_v2",
    "evidence_max_cards_per_fold": 400,
    "extraction_feature_batch_size": 10,
    "extraction_chunk_size_tokens": 50000,
    "extraction_context_window_tokens": 131072,
    "extraction_context_margin_tokens": 1024,
    "max_review_rounds": 2,
    "input_temporal_scope": "pre_index_treatment",
    "selection_consolidation": {
      "enabled": true,
      "neighbor_count": 10,
      "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
      "embedding_device": "cpu",
      "max_latents_per_cluster": 2,
      "minimum_pairwise_association": 0.85
    },
    "statistical_selection": {
      "l1_ratio": 0.8,
      "nuisance_selection_rule": "any_inner_fold_union",
      "modifier_selection_rule": "any_inner_fold_union",
      "one_standard_error_rule": true,
      "nuisance_prediction_one_standard_error_rule": false,
      "modifier_one_standard_error_rule": false,
      "univariable_confounder_p_value_threshold": 0.05,
      "univariable_confounder_q_value_threshold": 0.10,
      "modifier_top_n_per_inner_fold": 10,
      "modifier_ridge_alpha": 10.0,
      "modifier_continuous_winsor_quantile": 0.005
    },
    "role_adjudication": {
      "enabled": true,
      "max_candidates_per_request": 20
    },
    "estimation_trees": 200,
    "explicit_features": []
  }
}
```

The example above and `example_configs/research_all_evidence.json` explicitly
enable the second consolidation pass. When
`stage2.selection_consolidation.enabled` is omitted, it defaults to **false**;
fresh runs through the standard synthetic shell launchers also leave it off.
Saved-run launches preserve their configured policy. The earlier discovery-time
alias merge is separate and is not controlled by this switch. To enable the
second pass in a new configured run, set it to `true` or add
`--set stage2.selection_consolidation.enabled=true` to the Python entry point.

Interpretation, consolidation, operationalization, category mapping, and
aggregate ontology-review requests go to the primary model with
`reasoning_effort: "high"`. One-patient value extraction alone goes to the
configured `extraction_llm` model with `reasoning_effort: "none"`. The two
models may use different endpoints or the same multi-model endpoint.
`max_tokens` is the primary model's 100,000-token output ceiling;
`extraction_max_tokens` is the patient extractor's ceiling (4,096 in this example;
75,000 when omitted for compatibility). `extraction_reasoning_max_tokens` gives
reasoning-enabled extraction its own total allowance for reasoning plus final
JSON (32,768 in this example; when omitted or null it shares `extraction_max_tokens`). Neither
asks nor forces a model to generate that many tokens, and normal EOS stopping
applies. The extraction ceiling may be lowered to 4,096 tokens to bound a
runaway non-thinking response without invalidating completed feature or patient
checkpoints. Reasoning repairs use their larger ceiling, reduced only when the
remaining context requires it. Incomplete serial chunks are reused only if their
exact planning fingerprint still matches; a changed output reservation can require
re-extraction of those chunks. Long
patient records are processed serially in lossless source chunks
of at most 50,000 tokens, carrying the validated structured extraction into the
next chunk. The planner shrinks chunks as needed to preserve the model context,
and checkpoints every chunk for restart.
Stage 2 uses [publisher sampling profiles](stage2_sampling.md); omitted or null
sampling settings select the model defaults.
Stage 2 probes `/models`, recognizes Qwen 3 (including 3.8), Gemma 4, and LFM
2.5 IDs, and sends family-appropriate per-request thinking controls. It accepts
either server-parsed reasoning fields or inline reasoning delimiters.
For Qwen 3.8, configured `high` is sent as `reasoning_effort: "xhigh"`;
thinking-off extraction requests omit that enabled-only wire enum.
The selected IDs are persisted in `stage2/model_identity.json`: endpoint URL
changes may resume, but changing either running model ID raises an error.
Each logical request has a 7,200-second budget. Transport failures receive up to
6 attempts per response attempt by default, with a 900-second transport timeout.
With streaming enabled, that timeout measures network read inactivity; active
generation remains bounded by the logical deadline and output cap.
Invalid completed responses receive up to 15 validator-guided repair retries.
The first five repairs retain the request's normal reasoning policy; repairs
6–15 force `reasoning_effort` to at least `high`, including structural extraction
errors such as a missing `rows` array. These reasoning repairs use the larger
`extraction_reasoning_max_tokens` allowance when configured. Unambiguous complete
single-patient wrappers are normalized and validated without inventing values.

The example enables streaming progress and per-checkpoint `request_events.jsonl`
logs, including request IDs, retry counts, timing, finish reasons, and available
token usage. An exhausted patient/page task is deferred while other tasks finish,
then retried once (`extraction_deferred_retry_passes: 1`, also the default).
`deferred_extraction.json` records failures and recovery. Unresolved tasks still
prevent fitting; repeated consecutive failures abort early. Set the retry-pass
option to 0 for immediate abort, or `extraction_stream: false` for a server that
does not support streaming. Streaming defaults to false when omitted.

`stage2.explicit_features` may contain investigator-specified feature
definitions. Each entry must include its complete extraction ontology and
causal roles; see the complete workflow guide for the schema. Configured
features join Stage 2 alias consolidation in every outer fold, so an
automatically discovered alias does not create a second variable. They remain
fixed and required regardless of evidence strength. Configure either role or
both; their ontologies and roles cannot be changed by the models.

Independent outer folds run concurrently. Primary request concurrency is bounded
by `stage2.workers`, and patient extraction by
`stage2.extraction_llm.workers`. Each extraction request contains exactly one
patient's text; this isolation is invariant. Before supervised selection, the
optional `stage2.selection_consolidation` pass walks the extracted candidates,
retrieves the ten nearest currently active features by default, calculates
mixed-type association evidence on outer-training rows, and asks the primary
model whether to leave them separate or replace disjoint subsets with canonical
versions of the same measurement. Replacement requires every source pair to
meet `minimum_pairwise_association` (0.85 by default), but high association is
only a necessary condition: broader/narrower concepts and merely related
variables must remain separate. Accepted aliases immediately replace their
sources in later retrievals. Lossless nominal-category unions are allowed, and
continuous coalescing skips malformed nonnumeric values in favor of the next
valid alias; the original extraction dependencies remain recorded.

Separate treatment and marginal-outcome group elastic nets run in the inner
folds. Candidate-wise tests add raw p-values and within-fold Benjamini-Hochberg
q-values for treatment, outcome, and treatment-adjusted outcome associations.
The `univariable_confounder_p_value_threshold` and
`univariable_confounder_q_value_threshold` settings label evidence; they do not
directly retain or discard candidates. Candidate-specific calibrations feed
held-out R-loss comparisons, and a joint grouped elastic net fits candidate
interactions together. Binary nuisance reports include AUROC and log loss.

`stage2.statistical_selection.selection_mode` determines how this evidence is
used:

- `llm_roles` (default): a treatment/outcome any-fold union and the candidate-wise
  top ten R-loss gains per fold give provisional roles. Top-N membership does
  not require a positive gain. The primary LLM reconciles all evidence and
  assigns final roles; disabling role adjudication uses the provisional roles.
- `independent_tasks`: nonzero groups in any inner fold determine separate
  treatment, outcome, and joint-R-loss effect supports. Residual nuisance fits
  use all candidates within each fold. P/q values and top-N rankings are
  diagnostic; optional LLM annotations cannot alter the selections.

The LLM evidence bundle excludes row values, identifiers, outer-heldout
information, oracle fields, dataset identity, and generation metadata. These
adaptive inner-fold tests are exploratory, not confirmatory causal tests. See
[independent task selection](stage2_independent_tasks.md) for estimator routing:
external outcome models include outcome-selected features plus all selected
effect modifiers.
Consolidation receives neither treatment nor outcome and is not a role-selection
screen. Outer-heldout rows remain inaccessible until selection is frozen;
selected latent states are then applied to their held-out measurement dependencies.

For pipeline-managed orchestrator and extractor roles, replace both endpoints
in the example above with nested vLLM configurations. Their GPU lists define the
allowed union used by each alternately loaded model. Here the orchestrator's
tensor-parallel width is two, while the extractor's width is one:

```json
{
  "stage2": {
    "model": "Qwen/Qwen3.8-27B",
    "workers": 32,
    "vllm_rapid_switch_seconds": 900,
    "extraction_llm": {
      "model": "LiquidAI/LFM2.5-2.6B",
      "workers": 64,
      "vllm": {
        "gpus": [2, 3],
        "gpus_per_server": 1,
        "base_port": 8110,
        "extra_args": ["--gpu-memory-utilization", "0.80"]
      }
    },
    "vllm": {
      "gpus": [0, 1],
      "gpus_per_server": 2,
      "base_port": 8010,
      "download_dir": "/models/huggingface",
      "extra_args": ["--gpu-memory-utilization", "0.90"]
    }
  }
}
```

When both roles are managed, Stage 2 initially starts only the orchestrator
model and gives it the ordered union of the orchestrator and extractor GPU
lists. It completes every fold's interpretation and feature definitions, then
alternates the extractor and orchestrator over that union as checkpoints require
each model. If two switches occur less than
`vllm_rapid_switch_seconds` apart (15 minutes by default), Stage 2 instead keeps
both servers resident on their original configured GPU allocations for the
rest of the run and on resume. Set the cutoff to `0` to always alternate. A
feature-definition-only run never loads the extractor. `gpus_per_server` sets
each role's configured tensor-parallel width and derives its replica count; an
explicit `server_count` must agree. Gemma defaults to the `gemma4` reasoning
parser, Qwen to `qwen3`, and both default to language-model-only mode. See the
complete workflow guide for all-GPU pool derivation, GPU partition rules, and
all managed-server settings.

Stage 2 compiles the raw handoff into fold-local, provenance-preserving semantic
cards under `stage2/evidence_compilation/`, and candidate discovery reads all of
them. There is no ColBERT interaction filter, evidence-community graph,
candidate reranking, or feature-count cap. Discovery-time consolidation may only
merge aliases, so every unmerged candidate proceeds to extraction. The distinct
post-extraction selection-consolidation pass may replace empirically populated
aliases with a canonical, information-preserving measurement before fold-local
univariable/elastic-net evidence construction and the configured selection mode.
LLM role adjudication or advisory annotation uses bounded candidate batches
(20 per request by default),
while preserving each candidate's global fold votes and ranks. Each completed
request is saved beneath the relevant outer-fold directory, so the same command
resumes after interruption without repeating it.

To apply the all-evidence role selector to a completed legacy run without repeating
interpretation or all-candidate training extraction:

```bash
uv run python scripts/run_all_evidence.py \
  --config /path/to/completed_run/run_config.json \
  --stage2-only --stage2-reselect
```

The command verifies all reusable inputs before archiving the previous selector
and downstream results under `stage2/reselection_archives/`. It redoes nuisance
and top-N interaction selection, then reuses archived held-out measurements
whose row, text, model, frame, and definition fingerprints still match. Only
newly required or incompatible components are extracted. Estimation is then rerun.
Keep the original primary and extraction model IDs; endpoints may change.

The Stage 2 input is always:

```text
/path/to/output/handoff/evidence.jsonl
```

The final estimate and row-level cross-fitted results are:

```text
/path/to/output/stage2/causal_estimate.json
/path/to/output/stage2/cross_fitted_predictions.csv
```

For synthetic data with known truth, evaluate the frozen Stage 1 lanes in their
own right:

```bash
uv run oci-evaluate-stage1 \
  --run-dir /path/to/output \
  --metadata /path/to/metadata.json \
  --architectures all
```

See the [complete workflow guide](all_evidence_workflow.md) for
the config schema, Stage 2 endpoint contract, output layout, direct CLI
arguments, and component reruns.
