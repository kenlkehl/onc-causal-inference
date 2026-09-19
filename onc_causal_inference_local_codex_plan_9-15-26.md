# Local Codex handoff: independent Stage 2 selection

**Repository:** `kenlkehl/onc-causal-inference`  
**Prepared:** September 15, 2026  
**Goal:** Fetch PR #3, finish launcher integration, and prepare both existing synthetic-cohort scripts to run the new selector while reusing their existing local Stage 1 artifacts.

## Start here: instructions for the local Codex agent

Work in my local `onc-causal-inference` repository. Fetch and inspect PR #3, which adds `stage2.statistical_selection.selection_mode = "independent_tasks"`. Integrate it without losing local changes. Set up `run_one_conf_one_mod.sh` and `run_five_conf_five_mod.sh` so I can run this mode against the existing Stage 1 outputs for the corresponding cohorts on this machine.

**Stage 1 reuse is mandatory.** Do not regenerate either dataset, retrain Stage 1 models, rebuild embedding caches, resample folds, or regenerate the handoff. Preserve prior Stage 2 results. Reuse completed Stage 2 measurement/extraction artifacts as well when the existing guarded reselection mechanism verifies compatibility; that additional reuse is desirable, not a reason to compromise validation.

Discover local run directories, saved configurations, Python environments, and model settings before asking me for paths. Complete the integration, local tests, and non-mutating preflight. Leave exact launch and resume commands for me. Do not start the full cohort runs, merge to remote `main`, or push local changes unless I separately request that. Do not change the scientific method beyond the PR and the integration needed here.

## 1. Verified starting point and important limitations

At document preparation, PR #3 is open and a draft. Its verified head is `aca18285902d53a207a5fb490b235b88c3cb0114`, on `codex/stage2-independent-task-selection`, based on `62a1218f7ba5dbdae36a48bba3ae91f47d485414`. Recheck the live PR and local history; do not assume `main` includes the change. [S1]

The new mode uses independently regularized treatment, marginal-outcome, and joint R-loss interaction models. Effect-selected features enter forest `X`; nuisance-selected features not already in `X` enter `W`. External AIPW nuisance feature lists remain task-specific. Forest-internal nuisances independently regularize on `X + W`; the PR does not impose separate hard column masks inside that forest. LLM roles are advisory in the new mode. Existing investigator overrides remain locked. [S2]

**The PR alone does not finish this shell-script setup.** At the pinned revision, both cohort scripts call `scripts/run_synthetic_all_evidence.sh`, whose interface accepts a dataset path, an output name, and an optional output directory. It does not yet expose the new selection-mode or guarded-reselection switches. Simply exporting an arbitrary selection-mode variable will have no effect until it is wired through. [S3, S4]

The shared launcher auto-detects endpoint-backed Stage 2-only operation when the selected output root contains both `handoff/evidence.jsonl` and `handoff/complete.json`. That shortcut is conditional on external-endpoint operation and no managed model pool. Add an explicit Stage 2-only contract that works for both external and managed serving; do not rely solely on auto-detection. [S4]

The PR reports 20 focused tests, not a full repository or complete EHR/EconML run. Treat it as an implementation to validate locally, not an established performance improvement. The guarded reselection path was not end-to-end validated with this change. [S1, S2]

## 2. Inventory and protect the local state

### 2.1 Repository and runtime

Start with read-only inspection:

```bash
pwd
git status --short --branch
git remote -v
git rev-parse HEAD
git worktree list
```

Identify any active process writing to either run directory. Do not switch its code, alter its environment, or migrate its checkpoints while it is active. Do not kill a process automatically. Avoid destructive reset/clean commands and unrequested dependency upgrades. Preserve tracked and untracked local work; use a separate integration branch or worktree when necessary.

Read `CLAUDE.md` and any applicable local repository instructions. Inspect `pyproject.toml`, the lockfile, and the interpreter already used for successful runs. The launchers support `OCI_PYTHON` to use an existing executable and skip automatic dependency synchronization. Verify that this interpreter imports the intended updated checkout, not a different editable installation. [S4]

A separate worktree changes the paths of bundled datasets. For checkpoint reuse, retain the **original absolute dataset paths** from each saved run rather than silently substituting a byte-identical file in a new checkout. Preserve path/size/mtime identities wherever the checkpoint validators use them. [S5]

### 2.2 Identify the two actual run roots

Inspect local launcher invocations, configuration files, and run metadata in relevant project/storage directories. These are search hints, not established local paths:

```text
artifacts/research_all_evidence/one_conf_one_mod_nsclc_full
artifacts/research_all_evidence/five_conf_five_mod_nsclc_full
```

The launchers also allow custom output paths, and `DISABLE_HTR=1` changes the default name by adding `_no_htr`. Do not assume the default or newest directory is the correct run. [S3, S4]

For each cohort, record the following in an untracked local setup report:

| Item | What to establish |
|---|---|
| Identity | Actual run root, cohort dataset path, row count, row-ID/order identity, treatment/outcome/text columns |
| Science | Seed; outer/inner folds and saved splits; HTR setting; selected Stage 1 architectures; model identifiers |
| Stage 1 | Completion of the handoff and all referenced components; availability of embedding caches and provenance |
| Stage 2 | Not started, partial, completed legacy selection, completed independent selection, or interrupted reselection |
| Runtime | Actual primary/extractor model IDs, external versus managed serving, endpoints, GPU allocation, interpreter |

Inspect `run_config.json`, `resolved_stage1_model_config.json`, `resolved_neural_query_config.json`, `handoff/index.json`, completion markers, split provenance, and any Stage 2 configuration/model-identity records. Resolve file references to their existing local targets. The combined handoff is not necessarily a self-contained substitute for the component/cache directories. [S5]

### 2.3 Preservation manifest

Before any live migration, save a timestamped metadata snapshot outside the active Stage 2 tree. Include copies of existing run/Stage 2 configurations, progress metadata, model identity, and baseline result summaries. Keep secrets out of version control and redact them from logs/reports.

Create a content-hash manifest of each cohort's handoff, split/provenance files, and persistent Stage 1 component/cache artifacts. Include sizes and resolved paths; list anything not hashed rather than claiming complete verification. Never hardlink or symlink mutable Stage 2 outputs between comparisons. Record the baseline revision and intended new revision.

**Do not create an empty new run root and execute a full workflow.** That would not reuse Stage 1. The preferred route here is the existing run root, explicit Stage 2-only execution, and guarded archival of old Stage 2 results when needed.

## 3. Fetch and integrate the update safely

After verifying that `origin` is the intended repository, fetch without changing the working tree:

```bash
git fetch origin main
git fetch origin \
  pull/3/head:refs/remotes/origin/pr-3-independent-tasks

git rev-parse refs/remotes/origin/pr-3-independent-tasks
git show --stat refs/remotes/origin/pr-3-independent-tasks
```

Confirm that the verified commit is present:

```bash
EXPECTED=aca18285902d53a207a5fb490b235b88c3cb0114
git merge-base --is-ancestor "$EXPECTED" \
  refs/remotes/origin/pr-3-independent-tasks
```

If the fetched head differs, inspect and record the difference. A missing expected commit is a review condition, not permission to force anything. Preserve newer local fixes when integrating; do not replace a newer local branch with the PR's older base. Work on a local integration branch, and resolve conflicts with tests. Do not automatically change remote `main`.

Review these areas in the actual fetched diff and source: configuration serialization/fingerprints, the independent-task selector, advisory role handling, the Stage 2 orchestration/reselection path, and the new focused tests. Confirm that default `llm_roles` behavior remains available.

## 4. Finish integration with both existing scripts

### 4.1 Add explicit controls to the shared launcher

Implement and document the following environment-variable interface, preferably in the shared helper rather than duplicating behavior in both wrappers. **These are requested additions, not claims that the pinned scripts already support them.**

| Proposed control | Required behavior |
|---|---|
| `STAGE2_SELECTION_MODE` | Pass `--set stage2.statistical_selection.selection_mode=independent_tasks`; validate against supported modes; preserve legacy behavior when omitted |
| `STAGE2_ONLY=1` | Pass `--stage2-only`; require a complete, compatible existing handoff; never fall back to Stage 1 |
| `STAGE2_RESELECT=1` | Pass `--stage2-reselect` with Stage 2-only execution; require the guarded migration; never combine with `--rerun` |
| `OCI_RUN_CONFIG` | Path to a preserved per-cohort configuration; verify cohort/output identity; prevent hard-coded launcher defaults from overriding saved scientific settings |
| `OCI_PREFLIGHT_ONLY=1` | Resolve and validate the intended command/configuration without writing checkpoints, starting servers, running extraction, or beginning selection |

Equivalent existing local controls may be reused, but document their final names and commands. Preserve no-option launcher behavior. Reject invalid booleans/modes and inconsistent combinations, including a reuse-only request with Stage 2 disabled.

Separate **pipeline phase** from **hardware needs**. Stage 2-only must suppress Stage 1 execution even when a managed vLLM pool needs GPU discovery/allocation. Conversely, an external-endpoint-only run should not require local Stage 1 GPU allocation. Preserve existing serving configuration rather than forcing a switch between serving modes.

### 4.2 Preserve saved configuration instead of reconstructing it

Create a private, per-cohort configuration copy from the actual completed run. Do not replace it with an example config. The required scientific change is only:

```json
{
  "stage2": {
    "statistical_selection": {
      "selection_mode": "independent_tasks"
    }
  }
}
```

Preserve investigator features, selection-consolidation settings, extraction ontologies, model IDs, and other existing scientific settings. Keep existing role interpretation enabled as advisory unless explicitly configured otherwise. Preserve both cohorts' distinct runtime settings where appropriate; do not copy one cohort's config wholesale onto the other.

The saved config's dataset path, output root, columns, folds, seed, and architecture selection must win over shared-launcher defaults in reuse mode. Explicit runtime overrides may change endpoints/workers within supported rules. Validate that the selected wrapper corresponds to the saved cohort. A mode switch must not silently resume a completed legacy result as though it were an independent-task result.

### 4.3 Fail closed before expensive work

Preflight must check handoff completion and referenced paths, dataset/split compatibility, current selection authority, original model identities where extraction reuse is requested, and output destinations. It must print a redacted summary of the resolved mode and planned reuse/archival actions. Never fabricate a completion marker, relax a fingerprint check, or edit stored hashes to make a run appear compatible.

## 5. Choose the correct reuse path for each cohort

### Case A: Stage 1 complete; Stage 2 has not started

Run Stage 2-only in independent-task mode at the existing run root. Existing Stage 1 evidence/cache/provenance is consumed without recomputation. Stage 2 discovery and extraction will run because those outputs do not yet exist. Do not pass reselection just to reuse Stage 1.

### Case B: A compatible legacy Stage 2 run is complete

Prefer the repository's **guarded Stage 2 reselection**. It validates reusable inputs before archiving prior selection and downstream results under `stage2/reselection_archives/`, creates frozen preselection snapshots, reuses verified all-candidate training extraction, and re-estimates effects. Compatible held-out measurements can be reused; newly selected or incompatible measurements still require extraction. [S5]

The existing core command supports:

```bash
"$OCI_PYTHON" scripts/run_all_evidence.py \
  --config "$PRESERVED_COHORT_CONFIG" \
  --stage2-only --stage2-reselect \
  --set stage2.statistical_selection.selection_mode=independent_tasks
```

Verify locally that migration accepts this downstream selection-mode change while still rejecting changes to upstream definitions, data, splits, or measurement identity. Add regression coverage if needed. This is an integration requirement, not permission to weaken the reuse checks.

Retain the original primary/extraction model IDs for this path. Endpoints may change under the existing contract. Do not substitute the launchers' current model defaults for the IDs stored by the actual run. [S5]

### Case C: Stage 2 is partial or incompatible with guarded reselection

Do not claim complete extraction reuse. Inspect whether ordinary resume or an already-started guarded migration can safely finish from its state. If not, prepare a preservation-first fresh-Stage-2 fallback: snapshot/archive the entire prior Stage 2 tree and associated metadata, keep all Stage 1 directories untouched, and start only Stage 2 in a clean writable destination.

Use an existing supported output-isolation mechanism or a small explicit, tested archival helper; do not delete selected completion markers piecemeal. Record exactly which Stage 2 work would repeat and why. Do not perform the live archival or expensive fallback run during this setup-only task. An upstream incompatibility that cannot be safely resolved is a blocker to report, not a reason to regenerate Stage 1.

### Case D: Independent-task results or a reselection migration already exist

Use the existing state and resume semantics. Do not archive a completed independent run again simply because the same setup command was repeated. Test repeated invocation and interrupted preparation with fixtures. Keep `STAGE2_RESELECT` command-scoped for the initial migration; after preparation is complete, the normal resume command should omit it. If preparation itself was interrupted, follow the verified `reselection_state.json` recovery path. [S5]

## 6. Validate locally before declaring the setup ready

Run tests in the real checkout with the project's actual dependencies, not substituted helper implementations. Start with syntax checks and the PR's focused tests:

```bash
bash -n run_one_conf_one_mod.sh
bash -n run_five_conf_five_mod.sh
bash -n scripts/run_synthetic_all_evidence.sh
"$OCI_PYTHON" -m pytest -q tests/test_stage2_independent_tasks.py
git diff --check
```

Discover and run the existing launcher, Stage 2 configuration, reselection, cache/fingerprint, and estimator-routing tests. Run the full suite and package build when the local environment supports them; record actual passes, failures, skips, and missing prerequisites. Do not label unexecuted tests as passed.

Add integration tests for the following behaviors:

| Test area | Required evidence |
|---|---|
| Both launchers | New mode reaches the resolved statistical-selection policy; correct dataset and existing output root are used |
| No Stage 1 execution | Stub Stage 1 runners to fail if called; a reuse-only launch never calls them or rebuilds the handoff |
| Safe refusal | Missing handoff files, cohort mismatch, altered splits, invalid mode, or incompatible reuse inputs fail before model fitting |
| Legacy preservation | Omitted mode retains legacy behavior; existing completed legacy selection is not reported as a new result |
| Guarded migration | Prior Stage 2 results are archived, compatible training measurements reused, and upstream identities protected |
| Interruption/resume | Partial preparation resumes correctly; repeated normal resume does not create another migration |
| Serving | External-endpoint and managed-pool paths both respect Stage 2-only mode without changing model identities |
| Numeric authority | LLM annotations cannot change task routing; expected `X`/`W`, no-modifier fallback, and investigator overrides remain intact |

Use temporary fixtures to exercise archival and reselection before touching live outputs. A bounded selector/estimator smoke test may be run independently, with mocked LLM calls where suitable. Do not alter live folds or datasets to create a smoke test, and do not start full EHR extraction as a test.

After setup and preflight, compare the protected Stage 1 manifest against its baseline. Provide the same verification command for the user to run after the eventual cohort executions. Correct configuration propagation and successful tests establish readiness, not improved causal performance.

## 7. Leave ready-to-run commands for both cohorts

Resolve all placeholders below into real local values. Put machine-specific configurations/credentials in untracked files and provide separate cohort-specific launch snippets. The user should not have to guess which existing run directory to pass.

**The shell examples below become valid only after Section 4's controls are implemented and tested.** `ONE_CONFIG` and `FIVE_CONFIG` must point to the preserved, independently updated configurations; `ONE_RUN` and `FIVE_RUN` must be the corresponding existing run roots.

First provide non-mutating preflight commands. Set `ONE_RESELECT` and `FIVE_RESELECT` to `1` only for the initial compatible legacy migration, otherwise `0`:

```bash
OCI_RUN_CONFIG="$ONE_CONFIG" \
OCI_PREFLIGHT_ONLY=1 STAGE2_RESELECT="$ONE_RESELECT" \
STAGE2_ONLY=1 STAGE2_SELECTION_MODE=independent_tasks \
  ./run_one_conf_one_mod.sh "$ONE_RUN"

OCI_RUN_CONFIG="$FIVE_CONFIG" \
OCI_PREFLIGHT_ONLY=1 STAGE2_RESELECT="$FIVE_RESELECT" \
STAGE2_ONLY=1 STAGE2_SELECTION_MODE=independent_tasks \
  ./run_five_conf_five_mod.sh "$FIVE_RUN"
```

For a cohort with a compatible completed **legacy Stage 2** run, provide the initial guarded migration/run command:

```bash
OCI_RUN_CONFIG="$ONE_CONFIG" \
STAGE2_ONLY=1 STAGE2_RESELECT=1 \
STAGE2_SELECTION_MODE=independent_tasks \
  ./run_one_conf_one_mod.sh "$ONE_RUN"

OCI_RUN_CONFIG="$FIVE_CONFIG" \
STAGE2_ONLY=1 STAGE2_RESELECT=1 \
STAGE2_SELECTION_MODE=independent_tasks \
  ./run_five_conf_five_mod.sh "$FIVE_RUN"
```

For Stage 1-only cohorts, or ordinary resume **after reselection preparation is complete**, omit the migration flag:

```bash
OCI_RUN_CONFIG="$ONE_CONFIG" \
STAGE2_ONLY=1 STAGE2_SELECTION_MODE=independent_tasks \
  ./run_one_conf_one_mod.sh "$ONE_RUN"

OCI_RUN_CONFIG="$FIVE_CONFIG" \
STAGE2_ONLY=1 STAGE2_SELECTION_MODE=independent_tasks \
  ./run_five_conf_five_mod.sh "$FIVE_RUN"
```

Classify each real cohort independently; they may need different first-launch commands. Supply the required `OCI_PYTHON` and serving environment from the verified local configuration, not invented model IDs or ports. Recommend sequential execution when both cohorts share the same GPUs or managed endpoints. Do not launch them during setup.

Include a direct core-CLI fallback using each resolved config, plus log/progress inspection and rollback instructions. Rollback must restore an intact archived Stage 2 snapshot with its matching baseline configuration; never mix restored estimates with new independent-task selection metadata.

## 8. Completion criteria and local handoff report

The final local report must give: the integrated Git revision and changed files; both resolved cohort paths/configs; each cohort's reuse case and first/resume commands; the original model IDs and serving setup; the tests actually run; preflight results; the Stage 1 before/after checksum result; and any unresolved blockers. Save it as an untracked file such as `LOCAL_STAGE2_INDEPENDENT_TASKS_SETUP.md`.

After the user runs the cohorts, inspect each fold's `selection/elastic_net_selection.json` for `selection_authority = "independent_tasks"`, task-wise feature IDs, and forest `X`/`W` routing. With advisory interpretation enabled, check annotation-only metadata. Do not infer authority from the older `final_role_assignment` field, which retains a legacy branch label. [S2]

Expected final outputs include `stage2/cross_fitted_predictions.csv`, `stage2/causal_estimate.json`, per-fold estimation diagnostics, and optional post-hoc oracle metrics when available. Confirm one held-out prediction per eligible patient and no Stage 1 changes. Compare results with the archived baseline only after predictions are frozen; never use oracle columns to select features or tune this setup. [S5]

**Done means both scripts are ready to run the new selector using the existing Stage 1 evidence, with old results preserved and all limitations documented. It does not mean the full cohort analyses have been run.**

## Sources checked for this handoff

The implementation facts above were checked against PR #3 and the pinned source revision. The local controls and tests requested in this plan are proposed integration work, not existing features. Local paths and checkpoint compatibility remain for the local agent to establish.

[S1] PR #3 metadata and validation scope: <https://github.com/kenlkehl/onc-causal-inference/pull/3>

[S2] Independent-task mode documentation: <https://github.com/kenlkehl/onc-causal-inference/blob/aca18285902d53a207a5fb490b235b88c3cb0114/docs/stage2_independent_tasks.md>

[S3] Cohort launchers: <https://github.com/kenlkehl/onc-causal-inference/blob/aca18285902d53a207a5fb490b235b88c3cb0114/run_one_conf_one_mod.sh> and <https://github.com/kenlkehl/onc-causal-inference/blob/aca18285902d53a207a5fb490b235b88c3cb0114/run_five_conf_five_mod.sh>

[S4] Shared launcher, including automatic phase selection and interpreter handling: <https://github.com/kenlkehl/onc-causal-inference/blob/aca18285902d53a207a5fb490b235b88c3cb0114/scripts/run_synthetic_all_evidence.sh>

[S5] Workflow guide, especially “Output and resume” and “Reselect a completed Stage 2 run”: <https://github.com/kenlkehl/onc-causal-inference/blob/aca18285902d53a207a5fb490b235b88c3cb0114/docs/all_evidence_workflow.md>
