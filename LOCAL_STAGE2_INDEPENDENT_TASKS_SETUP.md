## September 18: calibration logging and overlap-bounded restart

The five-confounder/five-modifier launch config now sets `stage2.min_propensity=0.10`
and `stage2.max_propensity=0.90`. Both bounds apply to selection R-loss and final
effect modeling, with cross-fitted training eligibility and held-out eligibility
computed without held-out outcomes. New diagnostics preserve per-patient nuisance
predictions and calibration metrics. See `docs/stage2_independent_tasks.md` for
the population definition, artifact names and inference limitations.

The completed untrimmed run has NOT been restarted or altered. Read-only preflight
passed and is saved in `artifacts/local_stage2_independent_setup/five_conf_five_mod_overlap_preflight.json`.
To restart when ready:

```bash
./artifacts/local_stage2_independent_setup/five_conf_five_mod.sh restart
```

This archives current selection/estimation outputs under the active tree's
`stage2/reselection_archives/`, reuses frozen measurements and leaves
`stage2_pre_roles_refactor` unchanged. The original launch config is saved as
`artifacts/local_stage2_independent_setup/five_conf_five_mod_untrimmed.json`.
The one-confounder launch config has not changed. The earlier setup notes below
predate this bounded-restart configuration.

# Independent Stage 2 setup — September 18, 2026

Integrated on the already checked-out `codex/stage2-independent-task-selection` branch.
Base and verified PR #3 head: `aca18285902d53a207a5fb490b235b88c3cb0114`.
Fetched `origin/main` (`62a1218f7ba5dbdae36a48bba3ae91f47d485414`) and
`origin/pr-3-independent-tasks`; the PR implementation was already present.
The configured remote is `https://github.com/kenlkehl/causal-dragonnet-text`.
No branch switch, commit, push, cohort analysis, model-server restart, dataset
regeneration, Stage 1 training, fold resampling, or handoff regeneration occurred.
The integration consists of uncommitted working-tree changes.

## Ready-to-run commands

Run these from `/data1/ken/pcori_dev/causal-dragonnet-text`. The private scripts resolve all paths and set the
verified interpreter, independent-task mode, and cohort-specific preserved config.
Both preflight commands were executed successfully during setup.

```bash
./artifacts/local_stage2_independent_setup/one_conf_one_mod.sh preflight
./artifacts/local_stage2_independent_setup/five_conf_five_mod.sh preflight
```

First real launches (not executed during setup):

```bash
./artifacts/local_stage2_independent_setup/one_conf_one_mod.sh start
# Run the second cohort after the first finishes; both use the same servers/GPU.
./artifacts/local_stage2_independent_setup/five_conf_five_mod.sh start
```

Each first launch validates the archive, independently copies the intact
`stage2_pre_roles_refactor` tree into the currently absent `stage2/`, and then
uses guarded reselection. The original source archive remains untouched.
Old active selection and estimates move into `stage2/reselection_archives/`.
Compatible all-candidate training measurements are reused. Newly selected or
incompatible held-out measurements may still require extraction.

Normal resume after migration preparation:

```bash
./artifacts/local_stage2_independent_setup/one_conf_one_mod.sh resume
./artifacts/local_stage2_independent_setup/five_conf_five_mod.sh resume
```

If copying finished but migration did not start, or `reselection_state.json`
says `preparing`, use `resume-preparation` instead of `resume`. This sets
`STAGE2_RESELECT=1` without trying to restore another copy. A `prepared` state
uses ordinary `resume`; successful completion finalizes the migration even when
the resume invocation omitted the migration flag. A repeated ordinary resume
does not create another archive. An interrupted copy never publishes a partial
`stage2/`; retry `start` and retain any `.stage2-restore-*` sibling until the
successful copy has been verified.

Do not launch two processes against the same cohort root. `start` refuses an
existing `stage2/` instead of merging or overwriting directories.

## Actual cohorts and settings

| Item | One confounder / one modifier | Five confounders / five modifiers |
| --- | --- | --- |
| Run root | `/data1/ken/pcori_dev/causal-dragonnet-text/artifacts/research_all_evidence/one_conf_one_mod_nsclc_full` | `/data1/ken/pcori_dev/causal-dragonnet-text/artifacts/research_all_evidence/five_conf_five_mod_nsclc_full` |
| Original dataset | `/data1/ken/pcori_dev/causal-dragonnet-text/synthetic_data/example_synthetic_datasets/one_confounder_one_effect_modifier_nsclc_with_structured/dataset.parquet` | `/data1/ken/pcori_dev/causal-dragonnet-text/synthetic_data/example_synthetic_datasets/five_confounders_five_effect_modifiers_nsclc_with_structured/dataset.parquet` |
| New private config | `/data1/ken/pcori_dev/causal-dragonnet-text/artifacts/local_stage2_independent_setup/one_conf_one_mod.json` | `/data1/ken/pcori_dev/causal-dragonnet-text/artifacts/local_stage2_independent_setup/five_conf_five_mod.json` |
| Reuse case | Completed legacy Stage 2, manually archived | Completed legacy Stage 2, manually archived |
| Source archive | `stage2_pre_roles_refactor` under this run root | `stage2_pre_roles_refactor` under this run root |
| Patients / evidence rows | 1,000 / 90 | 1,000 / 90 |
| Seed / outer / inner folds | 42 / 5 / 5 | 42 / 5 / 5 |
| HTR | Enabled | Enabled |
| Stage 2 primary workers | 4 | 32 |
| Stage 2 extraction workers | 32 | 32 |
| Attempt / total timeout | 1,800 / 6,000 seconds | 900 / 7,200 seconds |
| Extraction context / margin | 128,000 / 4,096 tokens | 131,072 / 1,024 tokens |

Both cohorts use `patient_id`, `clinical_text`, `treatment_indicator`, and
`outcome_indicator`, with binary outcomes. Their original absolute dataset paths
are retained. Content and order fingerprints match the frozen TF-IDF manifest;
all five outer and five inner splits match the frozen evidence and archived
selection inputs. No oracle fields entered configuration or validation decisions.

Both use the ten legacy enabled Stage 1 architectures: `bow_nuisance`,
`bow_r_loss`, `matched_pair_uplift`, `htr_neural`, `embedding_whole_cohort`,
`embedding_clustered`, `tfidf_semantic_retrieval_contrasts`, `tfidf_topics`,
`tfidf_orphan_ngrams`, and `neural_query_moments`. Stage 1 model identifiers remain
`prajjwal1/bert-tiny` and `Qwen/Qwen3-Embedding-8B`. The handoff references existing
component evidence; embedding arrays, offsets, metadata, components, and split
provenance are present.

The older `five_conf_five_mod_nsclc_full_okish_8-31-26` tree is not used. Its saved
output identity points to the current five-cohort root and it is an older run.
The current roots' `progress.json` files still describe their archived baseline
completion; this is not evidence that the new independent-task run has executed.

## Configuration provenance and runtime

Each private config combines that cohort's root `run_config.json` Stage 1 fields
with its completed archive's actual `config.json` Stage 2 fields. This matters:
the root configs contained stale Stage 2 model defaults. The sole scientific
change is `stage2.statistical_selection.selection_mode = independent_tasks`.
Existing investigator features, consolidation, ontology, sampling, and role
settings are preserved. Role interpretation remains enabled as advisory;
selection consolidation remains disabled as in the archived runs.
Saved `<redacted>` API-key placeholders were replaced by `EMPTY` for these
verified local unauthenticated servers. No secrets are in tracked source files.

Interpreter: `/home/klkehl/thisenv/bin/python` (Python 3.13.15), verified to import
`oci` from this checkout. `.venv/bin/python` is a broken symlink and was left
unchanged. Existing package versions are recorded in
`artifacts/local_stage2_independent_setup/runtime_versions.json`.

Both use external serving:

- Primary: `http://localhost:8000/v1`, selected ID `gemma4-31b`, actual model root
  `nvidia/Gemma-4-31B-IT-NVFP4`.
- Extraction: `http://localhost:8001/v1`, selected ID and root
  `google/gemma-4-E4B-it`.

Read-only `/v1/models` requests matched both archived identities. Server worker
PIDs 2617551 and 2643847 were observed on physical GPU 1, NVIDIA GB300,
UUID `GPU-b9b14780-9af1-c0ab-46e5-fa6faeef64d9`. The model servers were not altered.
Serving observations are point-in-time; the pipeline checks model identity again
at launch. Sequential execution is recommended because the endpoints/GPU are shared.

## Integration and verification

Changed/new source files:

- `run_one_conf_one_mod.sh`, `run_five_conf_five_mod.sh`
- `scripts/run_synthetic_all_evidence.sh`, `scripts/launch_saved_stage2.py`
- `oci/inference/research_all_evidence_workflow.py`, `oci/inference/stage2_preflight.py`
- `tests/test_stage2_launch_preflight.py`
- `docs/stage2_independent_tasks.md`

The shared launcher implements `STAGE2_SELECTION_MODE`, `STAGE2_ONLY`,
`STAGE2_RESELECT`, `OCI_RUN_CONFIG`, and `OCI_PREFLIGHT_ONLY`. It additionally
supports `OCI_STAGE2_SOURCE` for these already-renamed archives. Saved-run mode
uses no wrapper science defaults, performs no dependency synchronization, and
preserves external or managed serving. Unsupported saved-run overrides and
inconsistent controls fail before expensive work. The ordinary no-option
launcher path retains its defaults.

Read-only preflight checks handoff completion and source content, dataset/order
fingerprints, exact split partitions, saved scientific settings, selection
identity, model IDs, cache presence, and migration compatibility. No checkpoint
hashes or completion markers were rewritten. The workflow now refuses a stale
completed legacy selection when independent-task mode is requested. Normal
resume also correctly finalizes an existing prepared migration.

Executed validation:

- Initial focused selector/launcher/workflow suite: 68 passed.
- Added integration tests: 27 cases (included in the final full suite).
- Final full suite: **759 passed, 1 skipped, 320 warnings** in 62.05 seconds.
  The skip is the existing unpatched NMF compatibility test, which skips when
  the installed sklearn/NumPy combination requires the tested compatibility path.
- Individual `bash -n` checks of both wrappers, shared launcher, and private
  command scripts passed. `git diff --check` passed.
- Source distribution and wheel built successfully with the real dependencies,
  using a clean temporary copy of the final working source. The direct checkout
  build was stopped because setuptools traversed the large artifact tree.
- Test/build utilities (`pytest`, `build`, `black`, `wheel` and their tool
  dependencies) were installed only under `/tmp/oci-stage2-test-tools`; the
  existing scientific/serving environment was not upgraded.
- Both final archived-source preflights passed (including all five folds).
  Logs: `one_preflight.json`, `five_preflight.json`, `full_suite.log`, and
  `build.log` under `artifacts/local_stage2_independent_setup/`.

## Preservation

Stage 1 byte-hash verification: **PASS — 295,302 files, 39,552,464,737 bytes; no differences**.
The after-integration comparison checked SHA-256, size, mtime, resolved paths,
and added/missing files. Evidence: `artifacts/local_stage2_independent_setup/stage1_verification.log`.

The preservation manifest is `artifacts/local_stage2_independent_setup/stage1_sha256.json`.
It records SHA-256, size, mtime, and resolved path for both input datasets,
root saved/resolved configs, complete handoff trees, and all persistent files
under both Stage 1 component trees (including caches and split provenance).
Stage 2 archives and shared downloaded model weights are outside this Stage 1
manifest. Original run/Stage 2 configs, model identities, progress, and baseline
result summaries are copied under `artifacts/local_stage2_independent_setup/baseline/`;
`baseline/metadata.json` timestamps the setup and records revisions.

After the eventual cohort executions, verify Stage 1 again:

```bash
/home/klkehl/thisenv/bin/python artifacts/local_stage2_independent_setup/protect_stage1.py verify
```

## Direct core CLI and inspection

After an active Stage 2 copy has been restored by the first wrapper launch, the
core migration/recovery commands are:

```bash
/home/klkehl/thisenv/bin/python scripts/run_all_evidence.py \
  --config /data1/ken/pcori_dev/causal-dragonnet-text/artifacts/local_stage2_independent_setup/one_conf_one_mod.json \
  --stage2-only --stage2-reselect
/home/klkehl/thisenv/bin/python scripts/run_all_evidence.py \
  --config /data1/ken/pcori_dev/causal-dragonnet-text/artifacts/local_stage2_independent_setup/five_conf_five_mod.json \
  --stage2-only --stage2-reselect
```

For normal resume, remove `--stage2-reselect`. To validate without executing, add
`--preflight-only`. Before the initial restore, core preflight can validate each
archive by also passing `--stage2-source` with the corresponding absolute
`stage2_pre_roles_refactor` path; that core option is read-only.

```bash
./artifacts/local_stage2_independent_setup/one_conf_one_mod.sh status
./artifacts/local_stage2_independent_setup/five_conf_five_mod.sh status
tail -f artifacts/research_all_evidence/one_conf_one_mod_nsclc_full/logs/workflow.log
tail -f artifacts/research_all_evidence/five_conf_five_mod_nsclc_full/logs/workflow.log
```

Once complete, inspect each fold's `selection/elastic_net_selection.json` for
`selection_authority = independent_tasks`, task-specific feature IDs, and forest
X/W routing. Do not use the legacy `final_role_assignment` field as authority.
Check advisory annotation metadata and per-fold estimation diagnostics. Require
`stage2/causal_estimate.json`, `stage2/cross_fitted_predictions.csv`, and exactly
one held-out prediction for each of the 1,000 eligible `_oci_row_id` values.
Only after predictions are frozen should baseline causal/oracle summaries be
compared. Passing integration tests establishes readiness, not improved causal performance.

## Rollback

With the cohort workflow stopped, preserve the entire new active `stage2/` tree
under a new unused name. Copy the entire pristine `stage2_pre_roles_refactor`
back into the missing `stage2/` destination using an independent file copy
(`shutil.copytree` without symlink/hardlink sharing). Do not selectively remove
completion markers or mix old estimates with new selection metadata.

The matching baseline configurations are:

- `/data1/ken/pcori_dev/causal-dragonnet-text/artifacts/local_stage2_independent_setup/baseline/one_conf_one_mod/effective_run_config.json`
- `/data1/ken/pcori_dev/causal-dragonnet-text/artifacts/local_stage2_independent_setup/baseline/five_conf_five_mod/effective_run_config.json`

Use the corresponding baseline config with `STAGE2_ONLY=1` and the cohort
wrapper, omitting `STAGE2_SELECTION_MODE`, `STAGE2_RESELECT`, and
`OCI_STAGE2_SOURCE`. The independent-task convenience scripts always select the
new mode, so do not use them to resume a restored baseline.

No scientific or launch-readiness blockers remain. Stage 1 preservation was
verified successfully after integration and both cohort preflights. Full EHR extraction/estimation was intentionally
not run during this setup task.
