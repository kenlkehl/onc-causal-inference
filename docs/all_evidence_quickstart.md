# All-evidence quickstart

Run these commands from the repository root after `uv sync --frozen`.
The complete Stage 1 workflow needs CUDA GPUs. Stage 2 can use external
OpenAI-compatible servers or pipeline-managed vLLM.

## Start the multi-model workflow

Copy the multi-model example **inside `example_configs/`**. Relative dataset and
template paths resolve against the configuration file's directory.

```bash
cp example_configs/research_all_evidence_multi_model.json example_configs/my_run.json
```

Edit `dataset`, `output_dir`, the four column names, `run.devices`, the clinical
question, and fold/seed settings. Configure both `stage2.endpoint` and
`stage2.extraction_llm.endpoint`, including `/v1`. Both roles can use the same
server. Leave their model IDs empty to discover the single advertised model, or
set the exact served IDs. Size both worker counts for available server capacity.

The multi-model example sets `run.mode: "full"` and enables:

- Repeated model evidence, including matched-batch contrast evidence.
- Post-extraction consolidation of equivalent measurements.
- Modifier screening at estimated propensities from 0.1 through 0.9.
- Joint modifier-count and causal-forest/penalized-interaction model search.

Estimand-informed ontology refinement and modifier-concept review are optional.
Enable both with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
uv run python scripts/run_all_evidence.py \
  --config example_configs/my_run.json \
  --set stage2.estimand_ontology.enabled=true \
  --set stage2.statistical_selection.multi_model.modifier_count.concept_review=true
```

Omit those two overrides to use the example's policies unchanged. An omitted
selector still defaults to `llm_roles`; the base
[`research_all_evidence.json`](../example_configs/research_all_evidence.json)
does not enable the multi-model selector.

Watch progress:

```bash
uv run python scripts/run_all_evidence.py \
  --config example_configs/my_run.json --status
tail -f /path/to/output/logs/workflow.log
```

Use `--stage1-only` to stop at the handoff. For configs without an explicit
`run.mode`, a primary endpoint or managed primary model implies a full run;
an explicit `run.mode: "stage1"` keeps it Stage 1 only.

To limit Stage 1 to selected evidence architectures, add
`--architectures bow_nuisance,tfidf_topics`, for example. Private prerequisites
are resolved automatically. Preserve that selection on resume; use a fresh
output directory for a different scientific configuration.

## What Stage 2 runs

For each outer fold, Stage 2:

1. Compiles the raw Stage 1 handoff into up to 400 semantic evidence cards by
   default, with source lineage. Discovery reads every card and proposes atomic
   clinical variables. The card limit is not a feature limit.
2. Consolidates exact aliases in semantic and alphabetical neighborhoods.
   Related but distinct measurements remain separate. The default ceiling is
   55 rounds; after at least three rounds, two consecutive successful rounds
   with less than 0.5% reduction stop the pass.
3. Defines each variable and extracts outer-training patients, normally ten
   variables from one patient per request. Long records are read in lossless
   serial chunks with validated prior values carried forward. Aggregate review
   can revise definitions and trigger targeted re-extraction.
4. Drops candidates **more than 95% missing** across outer-training patients.
   Exactly 95% missing is retained. The filter applies to investigator-specified
   variables too; raw measurements and exclusion audits are preserved.
5. When enabled, tests alternative definitions for both confounder and modifier
   uses with inner-validation nuisance losses and R-loss. Supported alternatives
   augment the original measurements and must pass the coverage filter.
6. When enabled, consolidates extracted aliases using semantic similarity,
   observed association, and an outcome-blind LLM equivalence review.
7. Builds numerical selection evidence using the configured mode.
   `multi_model` combines univariable screens, penalized main-effect and
   interaction models, orthogonal linear models, candidate R-learners,
   predictive forests, causal forests, and optional matched-batch contrasts.
   Repeated patient samples and forest feature subsets support stability review.
8. In `multi_model`, the LLM reviews themes and provisional roles. Optional
   concept review examines the top candidates from each inner fold and selects
   existing representative measurements. Nested R-loss validation then chooses
   a modifier budget and final estimator.
9. Freezes selection, extracts held-out measurement dependencies, and predicts
   CATEs with the selected forest or penalized interaction model. Separate
   elastic-net nuisances supply cross-fitted AIPW scores for the reported ATE.

The older `llm_roles` mode combines grouped elastic-net and candidate-wise R-loss
evidence with LLM role adjudication. `independent_tasks` selects treatment,
outcome, and effect supports numerically, with optional advisory LLM annotations.
Those paths retain the forest final estimator. P/q thresholds describe evidence;
they are not hard candidate inclusion gates.

The outer-training catalog and upstream ontology work remain fixed during
nested model search. Inner scores are conditional on that adaptation;
outer-held-out evaluation assesses the fitted workflow. Oracle variables never
guide selection.

See [multi-model selection](stage2_multi_model.md),
[estimand refinement](stage2_estimand_ontology.md), and
[independent task selection](stage2_independent_tasks.md) for details.

## Model settings and extraction recovery

Both reasoning policies default to `auto`. After probing `/models`, Stage 2
selects the resolved model family's publisher sampling profile. Qwen 3.8 Flash
Next, including the Inferact NVFP4 checkpoint, defaults to **xhigh reasoning for
both interpretation and extraction**. Other recognized families default to
high interpretation and initially non-thinking extraction. Explicit overrides
take precedence. See [sampling profiles](stage2_sampling.md).

The checked-in example uses an extraction output allowance of 4,096 tokens for
non-thinking calls and 32,768 for reasoning-enabled calls. These are total
response budgets, including reasoning. For larger reasoning allowances, set
both `stage2.extraction_max_tokens` and
`stage2.extraction_reasoning_max_tokens` appropriately; the core extraction
default when omitted is 75,000. The planner accounts for the actual context
window and shrinks serial text chunks to reserve output space.

Core request/recovery defaults are:

| Setting | Default |
| --- | --- |
| Logical request budget, including retries | 14,400 seconds |
| HTTP attempt timeout | 3,600 seconds |
| Transport attempts | 6 |
| Validator-guided response repairs | 15; checked-in examples explicitly use 10 |
| Repairs retaining the request's normal reasoning policy | First 5; subsequent repairs use at least high |
| Repeated field-specific failures before omitting only that field | 3 |
| Deferred extraction retry passes | 1 |
| Outer-fold transport/deadline recoveries | 2, with 60- then 120-second backoff |

Local waiting for a request slot is excluded from the logical budget;
server queueing is included. With streaming, the HTTP read timeout measures
network inactivity. Repair prompts include the validation error, including
structural errors. Field-local recovery retries the remaining fields, retaining
a valid prior serial-chunk value for the failed field when available, otherwise
null. Transport failures do not silently become missing measurements.

Only mode-aggregated variables require supporting observation quotations.
Scalar extraction does not require them, and exact source-string matching is
not required. Typed values and occurrence structure are still validated.

The examples enable streaming and write `request_events.jsonl`; streaming is
off if omitted. `deferred_extraction.json` records deferred tasks, and
`recovery_status.json`/`recovery_events.jsonl` record fold-level recovery.
Unresolved failures prevent fitting.

`stage2.workers` and `stage2.extraction_llm.workers` bound primary and extraction
requests across concurrent outer folds. Every extraction call still contains
one patient's text. The persisted `stage2/model_identity.json` requires the
same primary and extraction model IDs on resume; endpoint addresses can change.

## Resume, reselect, or use a saved handoff

Rerun the original command after interruption to reuse compatible checkpoints.
For a Stage 2-only resume, use the saved resolved configuration:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
uv run python scripts/run_all_evidence.py \
  --config /path/to/output/run_config.json --stage2-only
```

Do not substitute a fresh example config for a saved run: that can change its
scientific policies. Worker counts, addresses, and timeouts are runtime controls;
model identity, prompts, sampling policy, and selection settings have
checkpoint guards.

To reselect a **completed** run while retaining compatible interpretation and
training measurements, use:

```bash
uv run python scripts/run_all_evidence.py \
  --config /path/to/completed_run/run_config.json \
  --stage2-only --stage2-reselect
```

Supply deliberate selector changes through `--set`. The runner validates
reusable inputs, archives the previous selector/downstream results under
`stage2/reselection_archives/`, and reruns selection and estimation. Compatible
held-out measurements are reused; newly required definitions are extracted.

The main runner consumes `/path/to/output/handoff/evidence.jsonl`. The separate
`scripts/run_stage2_from_artifacts.py` launcher can use an existing raw handoff
and split-provenance file with a new Stage 2 output directory:

```bash
/path/to/python scripts/run_stage2_from_artifacts.py /path/to/artifact_run.json --preflight
/path/to/python scripts/run_stage2_from_artifacts.py /path/to/artifact_run.json
```

That launcher uses a **different, flat configuration schema**, not the main
runner's saved `run_config.json`. It explicitly specifies `dataset`,
`handoff_path`, `split_provenance_path`, `output_dir`, `report_dir`, clinical
question, column names, fold/seed settings, and a `stage2` object. Prefer
absolute paths. An optional `concurrency_control_path` supports live per-role
and total request limits within configured worker ceilings. See the
[complete workflow guide](all_evidence_workflow.md#saved-artifact-launcher-and-live-request-limits).
Preflight checks inputs and model services without starting LLM inference.

## Other launch options and outputs

The bundled `run_one_conf_one_mod.sh` and `run_five_conf_five_mod.sh` wrappers
retain their older scientific defaults. They preset Gemma IDs and external
endpoints on ports 8010 and 8020; override both served IDs when changing models.
Use the Python/configuration path above for `multi_model`; the wrappers'
`STAGE2_SELECTION_MODE` shortcut accepts only `llm_roles` and `independent_tasks`.

Managed vLLM needs the `local-llm` extra, model IDs, and per-role `vllm`
configurations in place of external endpoints. When both roles are managed,
they initially alternate over the union of their GPU allocations. Rapid
switching can trigger concurrent operation on the separate allocations. See
the [README's managed-server instructions](../README.md#pipeline-managed-vllm-server-pools).

Final outputs from the main runner include:

```text
/path/to/output/stage2/causal_estimate.json
/path/to/output/stage2/cross_fitted_predictions.csv
```

Fold directories contain candidate definitions, missingness audits, optional
ontology/consolidation artifacts, numerical evidence, selected roles, model
search results, and measurement checkpoints.

For synthetic data with known truth, evaluate frozen Stage 1 lanes separately:

```bash
uv run oci-evaluate-stage1 \
  --run-dir /path/to/output \
  --metadata /path/to/metadata.json \
  --architectures all
```

Legacy neural-query evidence using whole patient histories must first be
regenerated from saved queries and chunk caches under the current query-ranked
retrieval contract. See the [complete workflow guide](all_evidence_workflow.md)
for configuration, output schemas, and component reruns.
