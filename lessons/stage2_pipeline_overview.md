# Stage 2 pipeline overview

Stage 2 is an outer-fold-honest evidence-to-measurement-to-estimate pipeline.
The current implementation deliberately separates three kinds of information:

- Stage 1 evidence proposes what could be measured.
- Outer-training notes determine how those candidates behave as measurements.
- Outer-heldout notes are touched only after definitions, roles, mappings, and
  latent rules are frozen.

```text
Stage 1 handoff
    |
    v
semantic_cluster_cards_v2 (fold-local cards + exact lineage)
    |
    v
exhaustive candidate discovery (all cards; no retrieval/candidate cap)
    |
    v
merge-only name consolidation -> one-feature ontology requests
    |
    v
extract every candidate on outer-training records
    |
    v
aggregate ontology supervision and incremental re-extraction
    |
    v
optional equivalence-only alias consolidation
    |
    v
treatment/outcome group-elastic-net + p/q evidence; candidate-wise + joint R-loss evidence
    |
    v
llm_roles: binding LLM adjudication; independent_tasks: numerical selection + optional annotations
    |
    v
freeze definitions and extract selected held-out dependencies
    |
    v
elastic-net nuisances + honest causal forest + held-out AIPW
    |
    v
cross-fold estimate and stability summary
```

## What changed from the older Stage 2 design

Current Stage 2 does **not** use an evidence-community graph, ColBERT routing,
candidate-to-evidence retrieval, top-N evidence-axis union, hard candidate cap,
or discovery-time role filter. Every compiled semantic card enters exhaustive
feature listing, and every discovered feature reaches lossless merge-only
consolidation. Documents that describe `selected_candidates.json`, a bounded
candidate registry, or ColBERT candidate scoring are historical.

Discovery creates names and evidence lineage, not extraction types or causal
roles. One independent operationalization request defines each consolidated
candidate. Roles are assigned later from outer-training data.

## Fold boundary

For one outer fold, all of the following are training-only:

- evidence compilation and candidate discovery;
- alias merging and ontology definition;
- extraction harmonization and aggregate ontology revisions;
- equivalence-only latent construction;
- nuisance-role and modifier selection;
- categorical encoders, missingness rules, and fitted models.

The held-out fold receives only frozen measurement dependencies. Treatment and
outcome are absent from discovery, ontology supervision, and alias consolidation.
Oracle effect columns are reserved for post-run evaluation.

## Evidence and candidate construction

`semantic_cluster_cards_v2` is the only supported compiler. It applies the
scientific evidence projection, exact-deduplicates with provenance unioned,
builds conservative fold-local cards, and verifies all frozen selected Stage 1
architectures are represented. Its outputs live in
`stage2/evidence_compilation/`.

The primary model lists every pretreatment patient-level feature supported by
the cards. Exact names coalesce first. Repeated batches can merge aliases but
cannot drop unmentioned features. Early rounds shift alphabetical boundaries;
later rounds use deterministic shuffles. Explicit features are immutable.

Operationalization sees one canonical name and readable supporting excerpts.
It defines type, unit/categories, extraction rule, and missingness. Prompt
packing and all repair/fallback behavior are checkpointed per feature.

## Extraction and supervision

The small model extracts every candidate from outer-training notes, one patient
per request and bounded feature/document chunks per request. Mixed continuous
and categorical representations get a training-only harmonization plan that is
frozen for later rows.

The primary supervisor sees aggregate values and validation failures only. It
can revise the same extraction ontology, but cannot add/drop/split/merge/rename
features or assign roles. Only changed prompt-facing definitions are
re-extracted. Explicit ontologies remain locked.

## Equivalence-only consolidation

Before supervised selection, the optional sequential pass visits the original
candidate order once. Each active pivot retrieves nearby active definitions.
Pairwise Spearman, bias-corrected Cramer's V, or correlation-ratio evidence is
computed on outer-training rows.

`stage2.selection_consolidation.enabled` defaults to **false** when omitted,
including fresh runs through the standard synthetic shell launchers. The
checked-in example config explicitly sets it to `true`; saved-run launches
preserve their supplied policy. This second pass is separate from the earlier
merge-only name consolidation.

A replacement must satisfy all of these conditions:

- every source pair is evaluable and meets the configured association threshold;
- sources represent the same attribute, entity, temporal scope, and granularity;
- source and output value types match;
- continuous units match and the rule is coalesce;
- categorical recodes map every declared source category injectively;
- all-source missingness remains missing;
- overlapping nonmissing sources agree numerically or after canonical recoding.

The last check matters: coalesce is first-source-wins mechanically, so accepting
conflicting overlaps would discard information. The validator now rejects such
proposals. Malformed values outside a declared continuous ontology are still
treated as missing, matching materialization semantics.

Accepted canonical measurements immediately replace their sources in the
active retrieval pool. Original columns and recursive lineage remain in the
registry. Whether enabled, disabled, or trivial, the pass writes the report
referenced from final definitions.

## Statistical evidence and selection modes

Every inner fold fits two group elastic nets: logistic treatment nuisance and
the outcome-appropriate marginal nuisance. Nominal contrasts plus missingness
form one group. Candidate-wise omnibus screens also test treatment, outcome,
and treatment-adjusted outcome association, with within-fold Benjamini-Hochberg
q-values. The `univariable_confounder_p_value_threshold` (0.05 by default) and
`univariable_confounder_q_value_threshold` (0.10) generate support flags, not
hard inclusion gates. Candidate discovery has already adapted to the
outer-training data, so these are exploratory evidence rather than
confirmatory tests of the full pipeline.

Cross-fitted nuisance predictions feed candidate-specific grouped calibration
and a ridge-stabilized R-learner. All estimable treatment interactions for one
candidate are scored together on inner-heldout rows. The top ten held-out
R-loss gains per fold are selected by rank, without a positivity or p-value
gate. In parallel, one grouped-elastic-net R-loss model per fold selects among
all candidate interaction groups jointly and records held-out whole-model gain.

`stage2.statistical_selection.selection_mode` chooses the binding rule:

- `llm_roles` (default): treatment/outcome any-fold support forms a provisional
  nuisance union used by both residual nuisance models. Candidate-wise top-N
  R-loss support forms a provisional modifier union. The primary LLM adjudicates
  final roles from all views. Disabling role adjudication uses the provisional
  roles; failures of enabled adjudication do not silently use that fallback.
- `independent_tasks`: residual nuisance models independently regularize over
  all candidates within each inner fold. Separate treatment, outcome, and joint
  R-loss tasks select any-fold nonzero groups. Neither nuisance support nor a
  candidate-wise top-N rank is required for effect selection. P/q values and
  candidate-wise ranks remain diagnostics, and optional LLM annotations cannot
  change selection or routing.

An allowlisted aggregate artifact packages all four views for the primary LLM.
The artifact is submitted in bounded candidate batches while each candidate's
global votes and ranks remain intact. The adjudicator covers every candidate,
explicitly reconciles method disagreement and fold consistency, and may assign
both roles or neither in `llm_roles`; in `independent_tasks` its interpretation
is advisory. Prompt construction has no dataset interface and excludes
row-level values, identifiers, outer-heldout data, oracle fields, paths or names,
and generation metadata. Explicit investigator roles remain locked.

At 64 or more candidates, selection is submitted to a loky worker so Python
optimization loops do not contend with thread-level outer-fold orchestration.
Loky serialization also makes this path safe from notebooks, `python -c`, and
stdin without an `if __name__ == "__main__"` wrapper.

## Estimation and audit

Selected modifiers form causal-forest `X`; pure confounders form `W`; dual-role
features occur once in `X`. A constant-effect design is used when no modifier
survives. Outer-heldout rows receive propensity, potential-outcome predictions,
AIPW scores, causal-forest effects, and available uncertainty intervals.

In independent-task mode, external propensity inputs are treatment-selected;
external outcome inputs are outcome-selected **plus every selected effect
modifier**. Forest X contains effect-selected features, and W contains the
treatment/outcome union excluding X. Both internal forest nuisance models
independently regularize over X plus W. The retained confounder/modifier fields
are routing labels; `selection_authority` and `taskwise_routing` identify the
binding numerical selections. See the
[independent-task guide](../docs/stage2_independent_tasks.md).

The only supported nuisance family is elastic net. Logistic elastic net is used
for treatment and binary outcomes; squared-error elastic net is used for
continuous outcomes. The obsolete strict random-forest runtime-config module
was retired with its disconnected tests and guidance. `fit_audit()` now records
both configured settings and every fitted EconML nuisance clone, including its
cross-fit position, effective CV folds, selected `C` or `alpha`, constant-model
fallback, iteration count, and iteration-limit status.

## Checkpoints worth inspecting

| Path | Meaning |
|---|---|
| `evidence_compilation/` | Cards, exact members, lineage, and reduction audit |
| `outer_NNN/feature_definitions.json` | Operational definitions before extraction |
| `outer_NNN/ontology_supervision/` | Aggregate review, revisions, and convergence |
| `outer_NNN/selection/candidate_consolidation/` | Alias decisions, repair events, report, and latent registry |
| `outer_NNN/selection/statistical_evidence.json` | Univariable and multivariable confounder evidence plus candidate-wise and joint modifier evidence |
| `outer_NNN/selection/role_adjudication/` | Binding LLM adjudication in `llm_roles`; nonbinding annotations under `advisory/` in `independent_tasks` |
| `outer_NNN/selection/elastic_net_selection.json` | Final decisions, selection policy/authority, and numerical/LLM audit (historical filename retained) |
| `outer_NNN/final_definitions.json` | Frozen selected definitions and dependencies |
| `outer_NNN/estimation/diagnostics.json` | Fitted model and nuisance-clone audit |
| `cross_fitted_predictions.csv` | One held-out prediction per patient |
| `causal_estimate.json` | Cross-fitted ATE and uncertainty |

Reruns validate semantic fingerprints rather than trusting file presence. The
guarded `--stage2-reselect` path archives selection and downstream artifacts,
then reuses post-ontology definitions and all-candidate training extraction.
