# Independent Stage 2 task selection

This opt-in comparison replaces the LLM's inclusion/routing decisions with
three independently regularized numerical tasks. Stage 1 evidence generation,
candidate discovery, merge-only consolidation, ontologies, extraction, and the
honest causal-forest estimator remain unchanged. No clinical variables or
synthetic dataset identities are hard-coded.

## Enable the comparison

In a copy of your existing all-evidence configuration, add:

```json
{
  "stage2": {
    "statistical_selection": {
      "selection_mode": "independent_tasks"
    },
    "role_adjudication": {
      "enabled": true
    }
  }
}
```

This is a fragment to merge with your existing configuration, not a replacement
for its dataset, endpoints, science settings, or extraction configuration. Run
it with the existing entry point:

```bash
python scripts/run_all_evidence.py --config path/to/independent_tasks.json
```

Use a separate output directory for the first comparison, retaining the old run
unchanged. The new mode is included in the scientific policy fingerprint. Do
not manually copy old selection/estimation completion markers into the new run.
For reuse of completed extraction, use the repository's guarded Stage 2
reselection workflow, not unguarded checkpoint edits. That workflow was not
end-to-end exercised as part of the focused validation for this change.

Omitting `selection_mode` keeps the existing `llm_roles` behavior. The default
value is omitted from the serialized scientific policy, preserving old policy
fingerprints. Set `role_adjudication.enabled` to `false` to skip advisory LLM
calls altogether; the new mode's numerical selections are the same either way.

## Numerical procedure

Within each outer-training partition:

1. Fit the existing grouped elastic-net treatment and marginal-outcome screens
   on all candidate measurements within each inner-training fold. Each task
   selects its own regularization strength using its own cross-validation loss.
2. Construct nuisance residuals using all candidate measurements, with separately
   regularized nuisance fits contained in each inner fold. Do not use a selected
   feature union aggregated from other folds to restrict these nuisance fits.
3. Fit the existing **joint grouped elastic-net R-loss interaction model** to
   all candidate groups. Missingness interactions and categorical interactions
   lacking existing minimum treatment-arm support remain excluded.
4. For each task separately, retain groups with a nonzero coefficient in any
   inner fold. Treatment and outcome support are not prerequisites for effect
   selection. Candidate-wise top-N R-loss rankings and univariable p-values
   remain diagnostics and do not select features in this mode.

The three tasks use the existing penalty family and configuration; they fit
separate coefficients and select separate lambdas. The shared `l1_ratio` and
alpha grid are not newly made task-specific. This first comparison does not
implement outcome-adaptive propensity penalties, nonlinear interaction bases,
or new instrument-exclusion rules.

## Estimator routing

`modeling_tasks` is a subset of `treatment`, `outcome`, and `effect`.
`nuisance_model_roles` preserves separate treatment/outcome selections for the
existing external AIPW nuisance fits. Outcome-screen support is retained even
without treatment support or an LLM confounder label.

- Forest **X**: effect-selected features.
- Forest **W**: treatment/outcome-selected union, excluding features already in X.
- No effect-selected feature: preserve the existing constant-X fallback.
- No task support: exclude the candidate from final extraction/modeling.

EconML's internal treatment and outcome nuisance models independently regularize
on **X plus W**; they do not apply hard task-specific column masks. The external
AIPW nuisance models use the separate task-specific feature lists. Thus this is
not a claim that a treatment-only column is absent from every outcome-model
input: inside the forest it is available but independently regularized.

For compatibility with the existing estimator, `roles: ["confounder"]` encodes
membership in the nuisance union, and `effect_modifier` encodes eligibility for
X. These are **model-routing labels, not inferred causal truth**. See
`roles_are_model_routing_not_causal_labels` and `selection_authority` in the
output. Explicit investigator features preserve their configured roles exactly;
confounder-only overrides are not automatically promoted to X. Force inclusion
means retaining the input, not forcing its estimated coefficient to be nonzero.

## LLM annotations and audit

When enabled, role interpretation uses the existing allowlisted definitions and
aggregate statistical evidence, with a new annotation-only prompt. It cannot
add, remove, or reroute a feature. Model-authored responses are stored under
`role_adjudication/advisory/`; numerical decisions remain in the final
`decisions` field, while LLM interpretations are in `annotations`.

Annotation transport or response-validation failures are recorded and do not
veto numerical estimation. Filesystem/audit-writing failures remain visible.
Advisory caches have a distinct version and never overwrite or adopt binding
LLM-role checkpoints.

The authoritative new fields are:

- `selection_authority: "independent_tasks"`
- `taskwise_routing.task_feature_ids`
- `taskwise_routing.forest_x_feature_ids` and `forest_w_feature_ids`
- each decision's `modeling_tasks`, `task_votes`, and `forest_role`
- `role_adjudication.mode: "annotation_only"` when annotations are enabled

The older orchestrator's `final_role_assignment` field is retained for legacy
compatibility and still names the executed role-processing branch. In this mode
it **must not be used to infer selection authority**. Likewise, historical
`confounder` field names in nuisance-union reports are routing aliases. The new
explicit authority and task fields avoid treating those names as causal labels.

## Validation and remaining scope

Focused tests cover pure modifiers, prognosis-only and treatment-only inputs,
overlapping task support, discarded noise, investigator overrides, no-modifier
fallback routing, invalid votes, unchanged default policy serialization, LLM
veto/promotion isolation, advisory cache isolation, annotation failure behavior,
allowlisted prompts, legacy binding adjudication, a small binary-data numerical
smoke test, and invariance to changes in outer-heldout outcomes/oracle columns.

The numerical smoke test checks execution and invariants, **not improved causal
accuracy**. No five-confounder/five-modifier EHR benchmark, GPU/LLM extraction
run, or complete EconML end-to-end workflow was run for this change. The local
focused harness used the source-equivalent baseline univariable helper subset;
those unchanged helpers are present in full in the repository. The full project
test suite and package build were not run in that limited environment.

This remains a feature-selection ablation with any-fold unions, not fully soft
end-to-end estimation. Outer-fold protection does not turn adaptively discovered
candidate-wise p-values into independent confirmatory tests. A common linear
interaction basis can still miss nonlinear modifiers, and predictive selection
does not establish a sufficient confounding-adjustment set.

## Launch against a saved synthetic cohort

The two synthetic wrappers support configuration-preserving Stage 2 launches:

```bash
OCI_PYTHON=/path/to/existing/python \
OCI_RUN_CONFIG=/private/cohort-independent.json \
STAGE2_ONLY=1 STAGE2_SELECTION_MODE=independent_tasks \
OCI_PREFLIGHT_ONLY=1 ./run_one_conf_one_mod.sh /existing/cohort-run
```

Use `run_five_conf_five_mod.sh` for the other cohort. The config must retain the
original absolute dataset and output paths. Start from that cohort's
`run_config.json`; if Stage 2 has subsequently been resumed with different
settings, use its actual `stage2/config.json` for the `stage2` section. Change
only `stage2.statistical_selection.selection_mode` for this comparison. Supply
credentials separately when saved configs contain `<redacted>` values.

| Control | Behavior |
| --- | --- |
| `STAGE2_SELECTION_MODE` | `llm_roles` or `independent_tasks`; omission retains the saved policy, or the legacy default on the ordinary launcher path. |
| `STAGE2_ONLY=1` | Reuse a completed handoff; never execute Stage 1 or resample splits. |
| `OCI_RUN_CONFIG` | Preserved per-cohort config; requires `STAGE2_ONLY=1`. If omitted in Stage 2-only mode, read the output root's `run_config.json`. |
| `STAGE2_RESELECT=1` | Require Stage 2-only guarded migration of completed selection and downstream estimates. |
| `OCI_PREFLIGHT_ONLY=1` | Validate without writes, dependency synchronization, endpoint calls, server startup, extraction, or fitting; requires Stage 2-only mode. |
| `OCI_STAGE2_SOURCE` | Optional intact Stage 2 archive for the first guarded launch. Preflight validates it in place. A real launch copies it independently into the missing `stage2/` directory before migration. Requires reselection; refuses an existing destination. |

Saved-run launches never apply the wrappers' model, ontology, fold, seed, HTR,
or worker defaults. Both external endpoints and saved managed vLLM layouts are
supported. Managed serving uses the configured GPUs without enabling Stage 1.
`OCI_PYTHON` must be an existing environment; if omitted, the existing repository
`.venv/bin/python` is used without synchronization.

Explicit runtime overrides are supported for `STAGE2_ENDPOINT`, `STAGE2_MODEL`,
`STAGE2_EXTRACTION_ENDPOINT`, `STAGE2_EXTRACTION_MODEL`, `STAGE2_WORKERS`,
`STAGE2_EXTRACTION_WORKERS`, `STAGE2_REQUEST_TIMEOUT`, and
`STAGE2_REQUEST_ATTEMPT_TIMEOUT`. The primary and extraction `STAGE2_*VLLM_`
controls support `SERVERS`, `GPUS`, `GPUS_PER_SERVER`, `BASE_PORT`,
`INTERNAL_PORT_BASE`, and `DOWNLOAD_DIR`. Model identity validation still applies;
these controls cannot substitute a different model when reusing measurements.
Other `STAGE2_` overrides fail visibly on this path; edit the preserved config
explicitly instead. A blank `STAGE2_ENDPOINT` is the legacy Stage 2-disable
request and is rejected with Stage 2-only execution. To preserve a managed pool,
omit that variable and keep its configuration in the saved file.

Preflight verifies dataset content/order against Stage 1, the saved scientific
settings, complete handoff sources and combined legacy evidence, complete split
partitions and exact agreement with the TF-IDF evidence, cache availability,
selection authority, frozen snapshots when present, and saved model IDs.
Guarded reselection additionally runs its ordinary compatibility checks in
read-only mode. Preflight does not test endpoint availability or fit a model.
The ordinary workflow verifies served model identity when it starts.

For a completed legacy Stage 2 run, add `STAGE2_RESELECT=1` to preflight and to
its first real launch. If the previous tree was renamed, also provide
`OCI_STAGE2_SOURCE=/existing/cohort-run/stage2_pre_roles_refactor`. Remove
`OCI_PREFLIGHT_ONLY` to launch. The source archive remains intact; files in the
active copy are archived under `stage2/reselection_archives/` by the existing
migration. Compatible all-candidate training measurements are reused. Held-out
measurements are reused only for compatible selected definitions; new
measurements may still require extraction.

For normal resume after preparation, omit both `STAGE2_RESELECT` and
`OCI_STAGE2_SOURCE`. A completed independent run is not migrated again by an
ordinary resume. If `stage2/reselection_state.json` says `preparing`, repeat with
`STAGE2_RESELECT=1` and without `OCI_STAGE2_SOURCE`. If only archive copying was
interrupted, repeat the first launch; a partial `.stage2-restore-*` sibling is
never adopted. Keep all such directories until you have verified the successful
copy. Do not run two launchers concurrently against the same output root.

The core CLI also provides read-only validation:

```bash
python scripts/run_all_evidence.py --config /private/cohort-independent.json \
  --stage2-only --stage2-reselect --preflight-only
```

Add `--stage2-source /path/to/archive` to validate a renamed archive. This core
option is read-only; the wrapper's first-launch copy handles restoration.

## Nuisance calibration and propensity eligibility

Set `stage2.min_propensity` and/or `stage2.max_propensity` to enable exclusion
from effect estimation. Both default to `null` (no exclusion); either may be
specified alone. Values must be finite probabilities, and the minimum must be
less than the maximum. Bounds are inclusive: with 0.10 and 0.90, patients at
exactly either boundary remain eligible. These settings are separate from
`stage2.propensity_clip`, which limits AIPW denominators without excluding rows.

```json
{
  "stage2": {
    "min_propensity": 0.10,
    "max_propensity": 0.90
  }
}
```

Selection retains all training patients for nuisance screening/prediction.
The joint modifier model uses nested out-of-fold training propensities to
exclude rows before encoding interactions, centering, penalty tuning and
fitting. Its held-out R-loss uses only eligible inner-held-out patients,
whose propensity model was trained on the inner-training partition. The
candidate-specific R-loss screen uses this same base-propensity eligibility
population for every candidate; augmentation cannot change the eligible set.
Excluded rows do not enter either candidate effect regression or its score.

Final modeling cross-fits the external propensity model inside each outer
training partition, with fold-local encoders, to decide eligibility for
causal-forest training. External AIPW nuisance models still train on all outer
training patients. The causal forest, including its internal nuisance fits,
uses only eligible training patients. Outer-held-out eligibility uses the
external propensity model fitted on the outer training partition, before any
AIPW clipping. Held-out outcomes never determine eligibility. No iterative
trimming or automatic threshold search is performed.

All held-out patients remain in `estimation/predictions.csv` and the pooled
`cross_fitted_predictions.csv`, with `effect_eligible` flags. Excluded patients
have missing AIPW scores, CATEs and CATE intervals, and do not contribute to
ATEs, their standard errors, mean CATEs or oracle effect metrics. Reports label
the estimand `average_treatment_effect_in_propensity_eligible_population`,
retain the total `rows`, and add `effect_estimation_rows` and exclusion counts.
Eligibility is estimated and fold-specific; these estimates are no longer
full-cohort ATEs. The existing standard-error calculation treats the learned
eligibility rule as fixed; it does not add uncertainty for the hard boundary.
Insufficient eligible training rows/treatment arms or an empty held-out set
raises an explicit error instead of silently falling back to an untrimmed fit.

Calibration diagnostics are descriptive and never recalibrate predictions.
Binary models report Brier score, log loss, AUROC, observed/predicted means,
mean prediction error, ten fixed-width reliability bins, bin-weighted absolute
calibration error, and joint logistic calibration intercept/slope (ideal 0/1).
The intercept is from the joint intercept-and-slope fit, not an intercept-only
calibration-in-the-large fit. Single-class, constant-prediction, separated or
failed calibration fits have explicit statuses and null coefficients.
Continuous outcomes report mean error, MSE/RMSE and linear calibration
intercept/slope. Propensity reports include quantiles, counts below 0.10 and
above 0.90, and configured-bound exclusion counts.

Artifacts:

- `selection/statistical_evidence.json`: per-inner-fold nuisance diagnostics,
  nested training calibration, pooled inner-out-of-fold calibration, overlap
  summaries and per-patient nuisance predictions.
- `selection/nuisance_predictions.csv`: the pooled inner-out-of-fold
  probabilities and outcomes used for selection diagnostics. These rows are
  outer-training patients; do not pool the five outer files as unique patients.
- `estimation/nuisance_fit_predictions.csv`: out-of-fold external nuisance
  predictions and eligibility for causal-forest training.
- `estimation/diagnostics.json`: training/out-of-fold and outer-held-out
  calibration, factual outcome calibration by treatment arm, and eligible-only
  held-out calibration. Counterfactual outcome calibration is not observable.
- `estimation/propensity_overlap.json`: training and held-out exclusion audit.
- `causal_estimate.json`: pooled outer-held-out external nuisance calibration
  and propensity summary, counting each patient once.

Bounds enter both selection and estimation fingerprints. A saved-run launch
with changed bounds requires guarded `--stage2-reselect`, which archives
selection and downstream results before reusing measurements. Statistical and
estimation schemas also distinguish the new diagnostics from old checkpoints.

For the local five-confounder/five-modifier cohort, the prepared config at
`artifacts/local_stage2_independent_setup/five_conf_five_mod.json` uses 0.10–0.90.
The previous config is preserved as `five_conf_five_mod_untrimmed.json`.
Run the read-only check, then launch when ready:

```bash
./artifacts/local_stage2_independent_setup/five_conf_five_mod.sh preflight
./artifacts/local_stage2_independent_setup/five_conf_five_mod.sh restart
```

`restart` performs guarded reselection of the completed active Stage 2 tree.
It moves current selection/estimation outputs into a new
`stage2/reselection_archives/reselection_<timestamp>/` directory, reuses frozen
measurements, and leaves `stage2_pre_roles_refactor` intact. `resume` continues
an interrupted run after preparation; `resume-preparation` continues interrupted
archival preparation. The script retains the existing saved server/model
settings and runtime override behavior. It does not start a vLLM server.
