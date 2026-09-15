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
