# Stage 2 selection and estimation

This page documents the statistical part of Stage 2: how OCI decides, within each
outer fold, which extracted variables to adjust for, which to treat as effect
modifiers, which effect models to use, and how it builds the final estimates and
diagnostics. It starts where [How OCI works](how_it_works.md) step 6 ends: each
outer fold has a frozen catalog of candidate variables, each measured on every
outer-training patient. Terms are defined in the [glossary](glossary.md).

Everything on this page runs separately inside each outer fold and uses only
that fold's outer-training patients, except the final held-out predictions, the
outer-fold confirmation, and the pooled diagnostics. No step on this page sends a request to an LLM, except
the optional modifier-concept review described in
[Optional extensions](optional_extensions.md). Configuration keys are under
`stage2.statistical_selection` unless stated otherwise; defaults are listed in
[Configuration](configuration.md).

Two modifier rules are available (`multi_model.modifier_selection.rule`):

- **`orthogonal_tests`** (default) selects modifiers with three pre-specified
  component tests per candidate, combined by a Cauchy combination and
  Benjamini–Hochberg, and averages every final effect model (the estimator
  ensemble). Held-out patients then confirm the selected modifiers, for
  reporting only.
- **`balanced_evidence`** ranks candidates by a fixed formula and chooses the
  number of modifiers and one effect model jointly by nested validation.

Both rules share the evidence families, the predictive nuisance selection, the
final refit of each effect model, the nuisance construction, and the
diagnostics.

## Contents

1. [Evidence families and resampling](#evidence-families-and-resampling)
2. [Predictive nuisance selection](#predictive-nuisance-selection)
3. [Modifier selection by orthogonal tests](#modifier-selection-by-orthogonal-tests)
4. [Estimator ensemble](#estimator-ensemble)
5. [Outer-fold confirmation](#outer-fold-confirmation)
6. [The balanced_evidence rule](#the-balanced_evidence-rule)
7. [Final refit](#final-refit)
8. [Nuisance construction](#nuisance-construction)
9. [Propensity eligibility and calibration](#propensity-eligibility-and-calibration)
10. [DINA](#dina)
11. [Heterogeneity and overlap diagnostics](#heterogeneity-and-overlap-diagnostics)
12. [Output files](#output-files)
13. [Limitations](#limitations)

## Evidence families and resampling

### Evidence families

An [evidence family](glossary.md#evidence-family) is one model type used to score
candidates. Every candidate is exposed to every active family.

| Family (config name) | Default | What it measures |
|---|---|---|
| Univariable models (`univariable`) | on | Per candidate: association with treatment; treatment-adjusted association with outcome; and (`balanced_evidence` only) the adjusted interaction test described below |
| Penalized outcome interactions (`penalized_interactions`) | `balanced_evidence` | Grouped elastic net for `Y ~ T + all candidates + all T × candidate`; the selection of each candidate's interaction group, and the whole model's validation R-loss gain over a constant effect |
| Orthogonal joint model (`orthogonal_linear`) | `balanced_evidence` | All candidates jointly: grouped elastic-net DINA for binary outcomes; grouped residual model `(Y − m̂) ~ (T − ê)·(1 + candidates)` for continuous outcomes |
| One-candidate effect models (`univariable_rlearner`) | `balanced_evidence` | One candidate at a time: validation gain of a ridge DINA model (binary) or ridge R-learner (continuous) over a constant effect |
| Causal forests (`causal_forest`) | `balanced_evidence` | Honest forests (800 trees, minimum leaf 10) fitted to the residuals `T − ê` and `Y − m̂`; validation R-loss increase when a candidate is permuted |
| Joint DINA permutation (`dina`) | `balanced_evidence`, binary only | Validation DINA-loss increase when a candidate is permuted in the joint DINA model |
| Penalized main effects (`penalized_main`) | off | Grouped elastic nets for treatment and outcome; diagnostic only |
| Predictive forests (`predictive_forest`) | off | Random-forest permutation importance for treatment and outcome; diagnostic only |
| Matched-batch contrasts (`matched_batch_contrast`) | off | See [Optional extensions](optional_extensions.md) |

`penalized_main` and `predictive_forest` produce treatment and outcome evidence
that no later step reads; list them in `multi_model.diagnostic_families` to run
them. Matched-batch evidence runs when `multi_model.matched_batch.enabled` is
true.

Under `balanced_evidence`, the effect families feed the
[modifier ranking](#modifier-ranking). Under the default `orthogonal_tests`
rule, modifiers come from the [orthogonal tests](#modifier-selection-by-orthogonal-tests)
and no role, nuisance model, forest W, or kept output reads effect evidence, so
`penalized_interactions`, `orthogonal_linear`, `univariable_rlearner`,
`causal_forest`, `dina`, and the univariable interaction test are not run;
`selection/statistical_evidence.json` lists them under
`skipped_effect_evidence`. The univariable treatment and outcome associations,
the split nuisance models, and any requested opt-in family still run. The
evidence scope is part of the numerical-evidence identity, so evidence computed
under one rule is never reused under the other. The nuisance supports always
come from the split nuisance models described below.

**Adjusted interaction test.** For each candidate `Z`, two models are fitted on
training patients:

- reduced: `g(E[Y]) = α + β_T T + β_Z Z + f(logit ê) + h(m̂)`
- full: the same terms plus `γ T·Z`

where `g` is the logit link for binary outcomes and the identity for continuous
outcomes, `ê` and `m̂` are propensity and marginal-outcome predictions
cross-fitted inside the training partition (`m̂` enters on the logit scale for
binary outcomes), and `f` and `h` are each a linear term plus one
natural-cubic-spline term with knots at the 10th, 50th, and 90th percentiles. Binary outcomes use a likelihood-ratio test and continuous outcomes
a nested F test. Benjamini–Hochberg [q-values](glossary.md#bh-q-value) are
computed across candidates within each resample. These are ordinary adjusted
tests with learned adjustment scores, not orthogonal score tests, and their
p-values carry no post-selection guarantee.

**Grouping.** A candidate's categorical levels, its missingness indicator, and
its [extraction-failure](glossary.md#extraction-failure) indicator form one
group. Penalties select or drop the whole group, interaction tests are omnibus
over the group, and permutations move all its columns together.

**Equivalent measurements.** Candidates that are near-duplicates are grouped by
complete linkage. Two candidates qualify when they have the same value type
(and the same units for numeric types), identical terminal name tokens and a
name-token similarity of at least 0.8 after removing the words "value",
"measurement", and "level", at least 30 jointly observed patients covering at
least 80% of either candidate's observed patients, and a Spearman correlation
(or, for categorical values, an agreement rate) of at least 0.95. Groups are learned in each training
partition. They are permuted together as an extra evidence channel, and the
ranking keeps one representative per group.

### Resampling

Evidence is computed on several partitions of the outer-training patients:

- **Partitions.** The original five inner folds plus two additional,
  independently seeded five-fold partitions (`multi_model.stability.partition_repeats`,
  default 3). Each patient is a validation patient once per partition, giving 15
  training/validation splits per outer fold. No outer held-out patient enters.
- **Nuisances.** In each split, all-candidate elastic nets for treatment, the
  marginal outcome, and each arm's outcome are cross-fitted
  inside the training patients (3 subfolds, `internal_cv_folds`). Training
  patients get out-of-fold predictions; validation patients get predictions from
  models fitted on all training patients of the split. These nuisances stay fixed
  across the perturbations within the split, so the evidence measures stability
  given one honest nuisance construction, not a bootstrap of the whole pipeline.
- **Perturbations.** Single-candidate and linear families run three times per
  split (`multi_model.repeats`): once on all training rows and twice on 80% row
  subsamples (`multi_model.row_fraction`), with the penalized families cycling
  their elastic-net mixing ratio through 0.2, 0.8, and 1.0. The
  joint families (causal forest, joint DINA) run once on all candidates and then
  on four random partitions of the candidates into subsets of about 32
  (`multi_model.feature_subset_size`, `multi_model.stability.subset_partitions`),
  so every candidate is assessed in several covariate contexts. Permutation
  importance uses 10 paired permutations of validation patients.
- **Overlap.** Every effect family uses the same propensity eligibility rule
  (see [Propensity eligibility](#propensity-eligibility-and-calibration)); the
  treatment and outcome association tests and the nuisance models use all
  sampled patients.

Each fitted cell records whole-model validation gain over a constant effect
alongside per-candidate importance, because importance from a model that does
not generalize is weak evidence. A failed or nonconverged fit is recorded as
"not evaluable", never as a zero.

## Predictive nuisance selection

For each split, the full-training elastic nets above (treatment, marginal
outcome, control-arm outcome, treated-arm outcome) define supports. A candidate
is routed to the **treatment** task if its coefficient group is nonzero (norm
above `coefficient_tolerance`, default `1e-7`) in the treatment model of any
split, and to the **outcome** task if it is nonzero in any of the three outcome
models of any split. Penalties are tuned by training-only cross-validated
prediction loss. A candidate routed to either task receives the `confounder`
role.

These are predictive routing decisions, not identified causal roles, and no
p-value or LLM decision gates them. Their downstream use is limited: they decide
which variables enter the final effect models as adjustment covariates (the
forest's W and the interaction model's main effects). The nuisance predictions
used for scoring, eligibility, AIPW, and DINA always use the complete catalog
(see [Nuisance construction](#nuisance-construction)).

Investigator-specified variables (`stage2.explicit_features`) keep their
configured roles exactly; a confounder-only explicit variable never becomes a
modifier, and an explicit modifier is always included, outside the
Benjamini–Hochberg family of the orthogonal tests and outside the modifier
budget of `balanced_evidence`.

## Modifier selection by orthogonal tests

The default rule, `orthogonal_tests`, selects effect modifiers with
pre-specified tests on the outer-training patients. It runs no nested budget
search and fits no effect model to choose modifiers; held-out patients are used
afterwards to [confirm](#outer-fold-confirmation) the selection, never to make
it.

### Testing nuisances

Before any modifier is chosen, the training half of the final
[nuisance construction](#nuisance-construction) is computed: elastic nets on the
complete catalog for treatment and for the outcome in each treatment arm,
fitted in the inner cross-fitting folds with the final estimation's seeds,
encoders, and standardization, followed by cross-fitted Platt recalibration.
Each training patient receives `ê`, `μ̂0`, and `μ̂1` from the inner-fold model
that did not see it. Held-out patients take no part; their measurements do not
exist yet. The fitted inner-fold models are stored (see
[Nuisance construction](#nuisance-construction)), and the final estimation
reuses them to score the held-out patients instead of refitting them. Either
way the training values are reproduced exactly, because each inner-fold model
predicts its inner-validation patients and the held-out patients in separate
calls. The tests use the marginal outcome prediction

$$
\hat m = \hat e\,\hat\mu_1 + (1 - \hat e)\,\hat\mu_0 .
$$

When `stage2.min_propensity` or `stage2.max_propensity` is set, the tests use
only the training patients whose recalibrated `ê` lies within the bounds, as
the final estimation does.

### Grouped design

Each candidate `j` is tested on its grouped design `Z_j`, encoded on the
eligible training patients: the value centered at its median (continuous
candidates, with missing values imputed at the median) or one indicator per
non-reference category (categorical candidates; the reference is the first
declared category that is observed), plus the "not documented" indicator and,
for candidates with training extraction failures, the
[extraction-failure](glossary.md#extraction-failure) indicator. Indicator
columns with fewer than `categorical_min_count` patients (default 5) on either
side are dropped and listed in `dropped_sparse_columns`, because they cannot be
estimated reliably and give a patient a leverage of one. The remaining columns
are tested jointly; a column that adds no rank is skipped.

### Component tests

Each candidate receives up to three p-values, one per component:

| Component | Model and test | Scale |
|---|---|---|
| `r_projection` | Ordinary least squares of `Y − m̂` on `[(T − ê), (T − ê)·Z_j]` with no intercept and no penalty; HC3 covariance; joint Wald χ² test of the `Z_j` interaction block | Absolute (risk difference for binary outcomes), overlap-weighted: each patient weighs in proportion to `(T − ê)²`, as in the R-learner |
| `aipw_projection` | Ordinary least squares of the AIPW pseudo-outcome `ψ` on `[1, Z_j]`; HC3 covariance; joint Wald χ² test of the `Z_j` block | Absolute, population-weighted, as in the DR-learner projection |
| `outcome_interaction` | Reduced `g(E[Y]) = α + β_T T + f(logit ê) + h(m̂) + β_Zᵀ Z_j` versus the same model plus `γᵀ T·Z_j`; likelihood-ratio test (logit link) for binary outcomes, partial F test (identity link) for continuous outcomes | Multiplicative (odds ratios) for binary outcomes, additive for continuous outcomes |

Here

$$
\psi = \hat\mu_1 - \hat\mu_0 + \frac{T\,(Y - \hat\mu_1)}{\hat e_c} - \frac{(1 - T)(Y - \hat\mu_0)}{1 - \hat e_c},
$$

with `ê_c` the propensity clipped to `[c, 1 − c]`, `c = stage2.propensity_clip`,
exactly as in the AIPW ATE. In `outcome_interaction`, `f` and `h` are a linear
term plus one natural-cubic-spline term with knots at the 10th, 50th, and 90th
percentiles (`m̂` enters on the logit scale for binary outcomes), as in the
[adjusted interaction test](#evidence-families) of the evidence families, but
with the testing nuisances and the whole grouped design interacted with
treatment.

A component without a usable fit gives no p-value, and the reason is recorded:
a design without an independent candidate column, too few residual degrees of
freedom, a patient with leverage one, a singular covariance, too few eligible
patients or a single treatment arm, or a logistic fit that warned of
nonconvergence or separation.

### Cauchy combination

The available component p-values of candidate `j` are combined with a weighted
[Cauchy combination](glossary.md#cauchy-combination) (Liu and Xie 2020):

$$
T_j = \sum_k \tilde w_k \tan\{(0.5 - p_{jk})\,\pi\}, \qquad p_j = 0.5 - \frac{\arctan T_j}{\pi},
$$

where each `p_jk` is clamped to `[10⁻¹⁵, 1 − 10⁻¹⁵]` and the weights `w̃_k` are
the configured weights renormalized over the components that have p-values.
The default weights (`modifier_selection.weights` null) balance the two effect
scales:

| Outcome | `r_projection` | `aipw_projection` | `outcome_interaction` |
|---|---|---|---|
| Binary | 0.25 | 0.25 | 0.5 |
| Continuous | 1/3 | 1/3 | 1/3 |

For binary outcomes half the weight goes to absolute and half to
multiplicative evidence; for continuous outcomes all three components are on
the additive scale. The combined p-value is exactly uniform when the component
tests are independent. The three components test related hypotheses on the
same patients and are strongly positively correlated; the combination then
remains accurate in the lower tail, where selection happens, but is not uniform
over its whole range. In simulations without heterogeneity, rejection rates at
0.01, 0.05, and 0.10 matched their nominal levels, and the excess probability
sat near `p = 1`.

### Selection

1. **Family.** Every candidate that is not an investigator-locked variable and
   has a combined p-value. Benjamini–Hochberg
   [q-values](glossary.md#bh-q-value) are computed across the family.
2. **Discoveries.** Candidates with combined `q ≤ modifier_selection.q`
   (default 0.10), ordered by combined p-value.
3. **Cap.** At most `modifier_selection.max_modifiers` discoveries (default
   64; null for no cap) receive the `effect_modifier` role; the rest are marked
   `excluded_by_cap`.
4. **Locked modifiers.** Investigator-specified modifiers
   (`stage2.explicit_features`) are always included. They are tested and
   reported but stay outside the family and receive no q-value.
   Confounder-only explicit variables never become modifiers.

When no candidate modifies the effect and the p-values are valid and
independent or positively dependent, Benjamini–Hochberg at level `q` bounds the
probability of selecting any modifier by `q`. Zero discoveries is a valid
outcome; estimation then proceeds with no modifiers (see
[Zero modifiers](#final-refit)).

### Labels

Each component's p-values also receive Benjamini–Hochberg q-values across the
same family at the same `q`; a component passes when its q-value is at most
`q`. Each selected modifier is labeled:

- `absolute` when `r_projection` or `aipw_projection` passes;
- `multiplicative` (binary outcomes) or `additive` (continuous outcomes) when
  `outcome_interaction` passes;
- `combined_only` when it was selected although no single component passes.

A candidate is flagged `extreme_propensity_signal` when `aipw_projection`
passes and `r_projection` does not: the population-weighted projection gives
patients with extreme propensities far more weight than the overlap-weighted
one, so such a signal may rest on a few poorly overlapping patients. Labels and
the flag are reported only; they do not affect selection or estimation.

For binary outcomes the two scales differ in substance. Under a constant odds
ratio the risk difference still varies with baseline risk, so a strongly
prognostic variable is a genuine risk-difference modifier: the absolute
components detect it and `outcome_interaction` does not, and it is labeled
`absolute` only. A log-odds modifier also changes the risk difference, so
multiplicative modifiers usually carry both labels.

### Roles

Confounder roles and nuisance routing come from the one-pass
[predictive nuisance selection](#predictive-nuisance-selection), unchanged; so
does the causal forest's W. Only the effect-modifier role is assigned here. Each
modifier in `final_definitions.json` carries a compact `modifier_evidence`
record (combined and component p- and q-values, labels, the flag, rank, and
whether it is investigator-locked), and every candidate's full test record is in
`selection/modifier_tests.json` (see [Configuration](configuration.md)).

### Assumptions and limitations

- The p-values are nominal and conditional on the frozen catalog and
  extractions. The testing nuisances are cross-fitted, so the two projection
  tests are first-order insensitive to nuisance error; `outcome_interaction` is
  an ordinary adjusted test with learned adjustment scores.
- Stage 1 evidence, candidate discovery, and the extraction definitions used
  the same outer-training patients, and candidates reach the catalog partly
  because they appeared related to treatment or outcome. Training p-values are
  therefore somewhat optimistic. This is why the
  [outer-fold confirmation](#outer-fold-confirmation) repeats the tests on
  held-out patients.
- Each test examines one candidate at a time. A candidate correlated with a
  true modifier can be selected in its place, and Benjamini–Hochberg does not
  account for this redundancy.
- The Cauchy combination is accurate in the tail, not uniform, under the
  strong dependence between components.
- Rare indicator columns are dropped; a candidate whose only variation lies in
  a rare level cannot be tested.
- The HC3 Wald and likelihood-ratio tests are large-sample approximations.

## Estimator ensemble

Under `orthogonal_tests`, `modifier_selection.estimator_policy` decides the
final effect model. The default (null) is `ensemble`: every final estimator
available for the outcome type is fitted on the selected modifiers, exactly as
described in [Final refit](#final-refit) and with the same seeds as when it is
chosen alone, and `estimated_cate` is the unweighted mean of their predictions:

| Outcome | Components |
|---|---|
| Binary | causal forest, penalized interaction model, DINA |
| Continuous | causal forest, penalized interaction model |

The prognostic score enters each component as it does alone: in the forest's
X for binary outcomes, as an unpenalized covariate of the interaction model,
and through the DINA offsets. The policy can instead name one estimator
(`causal_forest`, `linear_interactions`, or `dina`; `dina` requires a binary
outcome).

- **Per-estimator columns.** `estimated_cate_causal_forest` and
  `estimated_cate_linear_interactions` are always written, and
  `estimated_cate_dina` when DINA was fitted, together with DINA's
  `dina_log_odds_ratio`, `dina_mu0`, and `dina_mu1`.
- **Intervals.** `estimated_cate_lower_95`, `estimated_cate_upper_95`, and
  `estimated_cate_standard_error` stay empty, because the forest's standard
  error does not describe the average. The forest's own interval is reported in
  `estimated_cate_causal_forest_lower_95`, `estimated_cate_causal_forest_upper_95`,
  and `estimated_cate_causal_forest_standard_error`; it describes the forest
  alone.
- **Unestimable components.** A component that is not estimable (for example a
  DINA fit that converges for no penalty) is recorded in
  `ensemble.unavailable_estimators` and left out of the mean; its column stays
  empty (DINA's is not written). If no component is estimable, the fold fails.
- **Audit.** `estimation/diagnostics.json` records `estimator_policy`
  (`ensemble`), the component and fitted estimators, each component's fit audit
  and architecture diagnostics, and `mean_estimated_cate_by_estimator`.

## Outer-fold confirmation

After every outer fold is finished and its predictions are frozen, the
selected modifiers of each fold are re-tested on that fold's held-out patients.
The confirmation is reporting only: it is computed in the pooled diagnostics
step from frozen artifacts, and no selection, estimation, ensemble, or other
fold reads it.

- **Inputs per fold.** The selected modifiers (features with the
  `effect_modifier` role in `final_definitions.json`), their held-out
  measurements (`estimation/nuisance_candidate_heldout.csv`, or
  `extraction/heldout/harmonized.csv` with the held-out failure mask), and the
  held-out nuisances and labels in `cross_fitted_predictions.csv`
  (`propensity`, `mu0`, `mu1`, `treatment`, `outcome`, `effect_eligible`).
- **Tests.** The same three component tests and Cauchy combination, with the
  configured `q` and weights, on the eligible held-out patients, with
  Benjamini–Hochberg within the fold's selected set. A modifier is confirmed
  when its held-out combined q-value is at most `q`. Because the held-out
  patients took no part in discovery or selection, these tests are valid
  conditional on the selection (sample splitting).
- **Pooled.** Counts of selected and confirmed modifiers per fold and in total.
- **Across folds.** Each fold defines and extracts its own variables, so
  variables are matched across folds only by exact normalized name (lower case,
  runs of other characters replaced by `_`), reported as "selected in `k`
  folds, confirmed in `j`". For a name selected in several folds, the held-out
  combined p-values are combined by Stouffer's method,
  `Z = Σ_k Φ⁻¹(1 − p_k) / √K`.

The results are written to `stage2/heterogeneity_diagnostics.json` under
`modifier_confirmation`, with an `interpretation` string, and summarized in
`stage2/causal_estimate.json`. Missing inputs make a fold or the section
`unavailable`; they never fail the run. The confirmation also runs for
`balanced_evidence` selections.

## The balanced_evidence rule

`multi_model.modifier_selection.rule = "balanced_evidence"` replaces the
orthogonal tests and the estimator ensemble with a deterministic ranking of the
candidates and a nested, joint choice of the number of modifiers and one
effect model. Its settings are under `multi_model.modifier_count`;
`modifier_selection.estimator_policy` must stay null, and
`modifier_selection.q`, `weights`, and `max_modifiers` are not used for
selection (`q` and `weights` still configure the outer-fold confirmation). The
optional [modifier-concept review](optional_extensions.md#modifier-concept-review)
works only with this rule.

### Modifier ranking

The ranking orders all unlocked candidates by a fixed formula over the saved
numerical evidence; it involves no LLM. It combines two equally weighted groups
of evidence: the adjusted interaction tests, and predictive evidence from the
effect models.

**Predictive sources.** For binary outcomes the sources are the one-candidate
DINA gain, the joint DINA permutation importance, and the causal-forest
permutation importance. For continuous outcomes they are the one-candidate
R-learner gain and the causal-forest permutation importance. Each joint source
has two contexts (the full candidate set and the feature subsets); the
one-candidate source has one.

For candidate `j`, source `f`, and context `c`:

1. **Patient-level gain.** Each validation patient contributes a paired loss
   difference: the constant-effect loss minus the candidate model's loss for
   one-candidate sources, or the permuted loss minus the unpermuted loss for
   permutation sources. A patient's repeated appearances are averaged first.
   `ḡ` is the mean over patients, and `σ` is the larger of the patient-level
   standard error and the standard deviation of the per-partition means.
2. **Strength** `s = max(ḡ, 0) / (|ḡ| + σ)`, which shrinks noisy gains toward 0.
3. **Percentile** `p`: the midrank of `ḡ` among the strictly positive gains of
   all candidates in the same source and context (0 if `ḡ ≤ 0`).
4. **Coverage** `κ` = evaluated fits / planned fits.
5. **Component score** `S_jfc = p · s · κ`.
6. **Model reliability** for joint sources: `r_jfc = max(G, 0) / (|G| + σ_G)`,
   where `G` is the whole model's patient-level gain over a constant effect in
   the fits that included `j`. One-candidate sources have `r = 1`.
7. **Source score** `S_jf = (Σ_c r_jfc S_jfc / Σ_c r_jfc) · max_c r_jfc`
   (0 when every reliability is 0).

The **predictive score** combines sources with a bounded agreement bonus:

$$
P_j = \frac{\max_f S_{jf} + 0.25\,\operatorname{mean}_f S_{jf}}{1.25}.
$$

The **interaction score** uses the adjusted interaction test's q-values, mapped
to a bounded strength

$$
s(q) = \frac{-\log_{10} q}{1 - \log_{10} q}
$$

(with `q` floored at `1e-300`; `s(0.1) = 0.5`). Strengths are averaged over
repeats within each inner fold, the median is taken across folds, and the result
is multiplied by coverage: `I_j = κ_j · median_fold(mean_repeat s(q))`. Missing
or unevaluable tests contribute no support.

The **rank score** is

$$
\text{score}_j = (1 - w)\,P_j + w\,I_j ,
$$

with `w = multi_model.interaction_ranking_weight` (default 0.5; 0 gives a
purely predictive ranking). A source that fails for a candidate keeps its place
in the mean and contributes zero.

**Equivalent measurements and ties.** For members of an equivalent-measurement
group, the score is recomputed with the group's permutation evidence added as
extra sources, and the larger of the two scores is used. Each group keeps one
representative (the member with the best individual score, then the highest
observed coverage, then the stable ID), which takes the group's best score; the
other members are dropped from the shortlist but kept in the audit. Exact ties
are broken by joint selection stability (how often the candidate's group was
selected in the two penalized joint models), then by median interaction q-value
(only when every tied candidate has one), then by feature ID.

The scores are heuristics. They are not probabilities that a variable is a
modifier, and adding predictive sources does not multiply their weight against
the interaction test. Every component, coverage count, and rationale is saved in
`ranking.json`.

### Modifier count and model selection

The number of modifiers and the final effect model are chosen together by
nested validation on the five original inner folds. The count rule
(`multi_model.modifier_count.selection_rule`) has one allowed value,
`balanced_evidence`.

**Candidate budgets.** Let `M = min(number of unlocked candidates, 64)`
(`max_ranked_modifiers`). The budgets are `{min(k, M) : k ∈ 0, 4, 8, 12, 16, 24, 32} ∪ {M}`
(`candidate_counts`), deduplicated. Locked modifiers are added on top of every
budget.

**Nested rankings.** For each inner fold `k`, the evidence of the previous
sections is recomputed using only that fold's training patients (with three
subfolds for validation), and a ranking is built from it. The labels of fold
`k`'s validation patients and of all outer held-out patients are physically
removed before this computation. A ranking built from all outer-training
patients would already have seen fold `k`'s validation outcomes, which would
favor it unfairly.

**Candidate models.** At each budget `K`, the top `K` candidates of fold `k`'s
ranking (plus locked modifiers) are the modifiers of three models
(`modifier_count.estimators`, default all three; DINA is skipped for
continuous outcomes):

| Model | Fit on fold `k`'s eligible training patients |
|---|---|
| `causal_forest` | An honest causal forest (`stage2.estimation_trees` trees, default 200; minimum leaf 10; square-root feature sampling; 45% subsampling) on the fixed residuals `T − ê` and `Y − m̂`, with the modifiers as features. For binary outcomes the prognostic score `logit μ̂0` is an extra feature. Three seeds are averaged (`forest_seeds`). |
| `linear_interactions` | A penalized outcome model with every catalog candidate as a main effect and treatment interactions only for the modifiers, plus unpenalized `logit ê` and prognostic-score covariates (see [Final refit](#final-refit)). CATE = predicted outcome under `T = 1` minus under `T = 0`. |
| `dina` | A grouped DINA model of the conditional log odds ratio over the modifiers, converted to a risk difference. |

The scoring residuals and the eligibility rule come from the all-candidate
elastic nets of the parent evidence pass (out-of-fold for training patients,
training-fitted for validation patients). They are identical across budgets and
models, so every combination is scored on the same validation patients against
the same residuals. DINA's offsets come from all-candidate elastic nets refitted
on the eligible training patients and shared across budgets.

**Scoring.** Each combination `(a, K)` is scored on fold `k`'s eligible
validation patients by

$$
L_k(a, K) = \frac{1}{n_k}\sum_i \big[(Y_i - \hat m_i) - (T_i - \hat e_i)\,\hat\tau_{a,K}(X_i)\big]^2 ,
$$

with `τ̂` on the probability scale for binary outcomes. A combination competes
only if it is estimable on every fold; at least two folds are required.

**Rule.** Let `L̄(a, K)` be the mean over folds and `(a*, K*)` the combination
with the smallest `L̄` (ties to fewer modifiers, then to the model order below).

1. **Predictive guardrail.** For each combination, compute the per-fold excess
   `d_k = L_k(a, K) − L_k(a*, K*)`, its mean `d̄`, and its paired standard error
   `sd(d_k)/√(number of folds)`. The combination is eligible if
   `d̄ ≤ SE + t`, where `t = max(10⁻⁴ · |L̄(a*, K*)|, 10⁻¹²)`
   (`relative_r_loss_tolerance`, default `1e-4`).
2. **Interaction utility.** In fold `k`'s ranking of length `R_k`, each candidate
   has utility `u_j = κ_j · (median_fold(mean_repeat s(q)) − s(q₀))`, where
   `q₀ = multi_model.q_threshold` (default 0.1, so `s(q₀) = 0.5`): positive for
   repeated adjusted-interaction support stronger than `q ≈ 0.1`, negative for
   weaker support, zero for missing tests. The budget's utility in fold `k` is
   `U_k(K) = Σ_{j ≤ K} u_j / R_k`, and `Ū(K)` is its mean over folds. It uses only
   training evidence; validation outcomes never enter it.
3. **Budget.** Among eligible combinations, choose the budget with the largest
   `Ū(K)`; ties go to the smaller budget.
4. **Model.** At that budget, among eligible combinations, choose the model with
   the smallest `L̄`; models within `t` of it are tied and resolved in the order
   `linear_interactions`, `dina`, `causal_forest`.

The rule lets strong interaction evidence influence how many modifiers are kept
without overriding a clear predictive disadvantage. It is a selection heuristic,
not a test, and it carries no error-rate guarantee.

A budget of zero is always a candidate (see
[Zero modifiers](#final-refit)).

**Final modifier set.** A ranking built from all outer-training evidence
supplies the top `K` candidates of the chosen budget, which receive the
`effect_modifier` role, and the chosen model is the only final effect model.
The nested comparison holds the catalog, extractions, and definitions fixed;
see [Limitations](#limitations).

## Final refit

Each final effect model (every ensemble component, or the single chosen
model) is refitted on the outer-training patients that pass the eligibility
rule, and predicts the outer held-out patients. Let the retained variables be
the confounders (from nuisance selection) and the modifiers (from the modifier
rule); a variable can be both.

- **Causal forest.** EconML `CausalForestDML` with honest trees, inference
  enabled, `stage2.estimation_trees` trees, minimum leaf 10, and square-root
  feature sampling. **X** contains the modifiers and, for binary outcomes, the
  prognostic score `logit μ̂0(X)`; **W** contains the confounders that are not
  modifiers. The forest fits its own internal cross-fitted elastic-net nuisances
  on X and W. The prognostic score is included because, on the risk-difference
  scale, the CATE varies with baseline risk even when the odds ratio is constant,
  and a forest restricted to the selected modifiers could not represent that.
  When there are no modifiers and the outcome is continuous, a constant X gives
  one effect for the fold. The forest is the only model that produces per-patient
  95% confidence intervals (written to the `estimated_cate_causal_forest_*`
  columns under the ensemble).
- **Penalized interaction model.**

  $$
  g\big(E[Y \mid T, Z]\big) = \alpha + \beta_T T + \lambda_1 \operatorname{logit}\hat e + \lambda_2\, \text{prog} + \beta^\top Z + T\,\gamma^\top Z_{\text{mod}}
  $$

  where `Z` holds all retained variables, `Z_mod` the modifiers, and `prog` the
  prognostic score (`logit μ̂0` for binary outcomes, `μ̂0` for continuous). The
  intercept, `β_T`, and the two adjustment covariates are unpenalized; the main
  effects and interactions carry a grouped elastic-net penalty (mixing ratio
  `l1_ratio`, default 0.8) tuned by training-only cross-validated outcome loss.
  The unpenalized adjustment covariates guard against regularization-induced
  confounding (Hahn et al., *Bayesian Analysis* 2018): when many confounder main
  effects are shrunk toward zero but `β_T` is not, the confounding they should
  absorb moves into `β_T` and the interactions. The covariates are never
  interacted with treatment. CATE is the predicted outcome under `T = 1` minus
  under `T = 0`.
- **DINA.** A grouped DINA model over the modifiers, with the
  [final nuisances](#nuisance-construction) as offsets for both fitting and
  prediction. See [DINA](#dina).

**Zero modifiers.** Zero modifiers does not force a zero effect: the forest
then fits a constant effect (or, for binary outcomes, an effect that varies
only with baseline risk); the interaction model keeps an unpenalized treatment
coefficient; and DINA fits a constant log odds ratio, which still implies risk
differences that vary with baseline risk.

The effect predictions are frozen before the held-out treatment and outcome
labels are read.

## Nuisance construction

One nuisance construction feeds every downstream quantity of the final
estimation: propensity eligibility, the AIPW scores, the DINA offsets, the
interaction model's adjustment covariates, the forest's prognostic score, and
the diagnostics. Its training half, computed earlier and identically, also
supplies the [testing nuisances](#testing-nuisances) of the orthogonal tests.

1. **Models.** Within the outer-training patients, the five inner folds are
   used for cross-fitting. In each inner fold, elastic nets are fitted on the
   other four folds: a logistic model for treatment and, separately in each
   treatment arm, a model for the outcome (logistic for binary outcomes). Every
   model uses the **complete catalog** of candidates, with categorical encoders,
   standardization, and the penalty (3-fold cross-validation, mixing ratio 0.8)
   fitted on that model's training rows. The selected nuisance supports are
   recorded for transparency but do not restrict these models: a support chosen
   using all outer-training patients would otherwise screen the inputs of their
   own out-of-fold predictions.
2. **Training patients** receive the prediction of the one inner-fold model that
   did not see them.
3. **Held-out patients**, measured on the same complete catalog, receive the
   unweighted average of the five inner-fold models' predictions (on the
   probability scale for binary targets). There is no separate model fitted on
   all outer-training patients.
4. **Model reuse.** Under `orthogonal_tests`, the modifier tests fit these
   inner-fold models before selection and store them with joblib in
   `selection/nuisance_models/<id>/`, together with a `manifest.json` holding
   each file's SHA-256 and a fingerprint of the training measurements and
   labels, candidate definitions, folds, seed, source code, and library
   versions (Python, NumPy, pandas, SciPy, scikit-learn, joblib). The final
   estimation recomputes that fingerprint from its own inputs and loads a model
   only when the fingerprint and file hash match and the model's input width
   agrees; otherwise, or on any load error, it refits that model. Reused and
   refitted models give identical predictions, and each model's fit audit is
   written either way, marked `model_source` `reused_stored_model` or
   `fitted`. Under `balanced_evidence` no models are stored and the final
   estimation fits them.
5. **Recalibration.** The propensity and, for binary outcomes, both arm-specific
   outcome predictions are Platt-recalibrated: a logistic regression of the label
   on `[1, logit(prediction)]`. Each inner fold's training predictions use a
   calibrator fitted on the out-of-fold predictions of the other inner folds;
   held-out predictions use a calibrator fitted on all training out-of-fold
   predictions. The `μ0` (`μ1`) calibrator is fitted on comparator (treated)
   training patients and applied to every patient. A calibrator falls back to
   the identity map when it has fewer than 20 rows, a single class, constant
   predictions, separation, or a nonpositive slope. Continuous-outcome
   predictions are not recalibrated.

The recalibrated predictions are written as `propensity`, `mu0`, and `mu1`; the
raw predictions are kept as `propensity_uncalibrated`, `mu0_uncalibrated`, and
`mu1_uncalibrated`. Each fold's `estimation/diagnostics.json` reports
calibration before and after recalibration (`nuisance_calibration`, including a
`calibration_slope_summary`), elastic-net fit audits, including any fit that
reached its iteration limit (with `model_sources` counts when a model store was
consulted), and, under `orthogonal_tests`, `nuisance_model_store` (`available`
or `absent`, with the store fingerprint). The held-out calibrator is learned from
single-model out-of-fold predictions but applied to an average of five models,
which is slightly less dispersed; held-out calibration slopes after
recalibration can therefore sit modestly above 1 (see
[Limitations](#limitations)).

## Propensity eligibility and calibration

### Eligibility bounds

`stage2.min_propensity` and `stage2.max_propensity` (both unset by default;
either may be set alone) exclude patients with estimated propensities outside
the bounds from effect fitting and effect estimation. Bounds are inclusive: with
0.1 and 0.9, a patient at exactly 0.1 is eligible. They are separate from
`stage2.propensity_clip` (default 0.02), which bounds AIPW denominators and
excludes no one.

The same rule applies at every stage:

| Stage | Training patients judged by | Validation or held-out patients judged by |
|---|---|---|
| Evidence families and modifier ranking | Out-of-fold training propensity | Propensity from models fitted on the split's training patients |
| Orthogonal modifier tests | Recalibrated out-of-fold propensity (testing nuisances) | Not used |
| Modifier count and model selection (`balanced_evidence`) | Out-of-fold training propensity | Propensity from models fitted on the fold's training patients |
| Final refit and estimation | Recalibrated out-of-fold propensity | Recalibrated inner-fold-ensemble propensity |
| Outer-fold confirmation | Not used | `effect_eligible` from the final estimation |

Held-out outcomes never determine eligibility, and there is no automatic
threshold search. Too few eligible patients or treatment arms makes an evidence
cell or a modifier test unevaluable and stops count selection or final
estimation with an error; no step falls back to all patients. Excluded held-out patients stay
in `predictions.csv` and `cross_fitted_predictions.csv` with
`effect_eligible = false` and missing `aipw_score` and `estimated_cate`; they do
not contribute to the ATE, its standard error, the mean CATE, or oracle metrics.
The estimand is then labeled `average_treatment_effect_in_propensity_eligible_population`.
The eligible population is estimated and differs by fold, and the reported
standard error treats the eligibility rule as fixed.

### Calibration diagnostics

Calibration is reported for treatment and for the factual outcome (the outcome
model of the arm each patient received), on training out-of-fold predictions
and on held-out predictions, overall, by treatment arm, and among eligible
held-out patients:

- binary targets: Brier score, log loss, AUROC, observed and predicted means,
  ten fixed-width reliability bins, expected calibration error, and the
  intercept and slope of a logistic regression of the label on the logit of the
  prediction (ideal 0 and 1);
- continuous targets: mean error, MSE, RMSE, and linear calibration intercept
  and slope.

Propensity summaries (`estimation/propensity_overlap.json`) give quantiles,
counts below 0.1 and above 0.9, and exclusion counts. Counterfactual outcome
calibration cannot be observed.

## DINA

DINA (**DI**fference in **NA**tural parameters; Gao and Hastie, "Estimating
heterogeneous treatment effects for general responses," *Biometrics*
81(4):ujaf162, 2025, [doi:10.1093/biomtc/ujaf162](https://doi.org/10.1093/biomtc/ujaf162))
models a treatment effect on the scale of the outcome distribution's natural
parameter. For a binary outcome that scale is the log odds, so the DINA effect
`δ(X)` is the conditional log odds ratio.

Given propensity `e`, arm-specific outcome probabilities `μ0` and `μ1`, and
`v_t = μ_t (1 − μ_t)`, OCI computes

$$
a(X) = \frac{e\,v_1}{e\,v_1 + (1 - e)\,v_0}, \qquad
\nu(X) = a\,\operatorname{logit}\mu_1 + (1 - a)\,\operatorname{logit}\mu_0 ,
$$

and fits `δ` under the working model

$$
\operatorname{logit} P(Y = 1 \mid T, X) = \nu(X) + \{T - a(X)\}\,\delta(X)
$$

by minimizing the Bernoulli loss `softplus(η) − Yη` with
`η = ν + (T − a) δ`. Centering the treatment at `a` makes the fit insensitive to
first-order errors in the nuisances. The probability-scale effect reported as
`estimated_cate` is

$$
\hat\tau(X) = \operatorname{expit}\{\nu + (1 - a)\,\delta\} - \operatorname{expit}\{\nu - a\,\delta\}.
$$

**Implementation** (`oci/models/dina.py`, `oci/inference/stage2_dina.py`).
`δ(X)` is linear in the encoded modifiers plus an unpenalized constant. A group
elastic-net penalty (mixing ratio 0.8) keeps or drops each variable's encoded
columns together; continuous modifiers are winsorized at the 0.5th and 99.5th
percentiles. The penalty is tuned by 3-fold cross-validated DINA loss over 8
values from `10⁻¹` down to `10⁻⁵`, warm-starting within each partition. A
penalty that fails to converge in any cross-validation fold is excluded; if
none converges, or the final fit fails, the model is marked not estimable. The
one-candidate evidence models use a ridge penalty instead. DINA exports
`dina_log_odds_ratio`, `dina_mu0`, and `dina_mu1` beside `estimated_cate`; it
does not produce per-patient confidence intervals.

**Where OCI uses DINA (binary outcomes only).**

| Place | Use |
|---|---|
| Stage 1 `bow_r_loss` | Sparse ridge DINA on bag-of-words features |
| Stage 1 `htr_neural` effect heads | DINA loss with nuisances frozen inside the effect-training fold |
| Stage 1 `tfidf_topics`, `tfidf_orphan_ngrams`, `neural_query_moments`, embedding effect directions | DINA score `(T − a)(Y − expit(ν + (T − a)δ₀))` around a training-fitted constant `δ₀`, with Fisher-information weights |
| Stage 2 evidence | One-candidate DINA gains, the joint grouped DINA model, and joint DINA permutation importance |
| Stage 2 estimator ensemble, `balanced_evidence` model search, and final refit | The `dina` effect model |

**Scale.** DINA, the logistic interaction model, and the adjusted interaction
test describe variation in log odds ratios; the causal forest describes
variation in risk differences. A constant odds ratio implies risk differences
that vary with baseline risk, so these methods can legitimately support
different modifiers. The `balanced_evidence` model search compares all models
on the risk-difference scale; the orthogonal tests combine evidence from both
scales.

## Heterogeneity and overlap diagnostics

After every outer fold finishes, `stage2/heterogeneity_diagnostics.json` is
computed from the pooled held-out predictions (`aipw_score`, `estimated_cate`,
`propensity`, `mu0`, `mu1`, treatment, and outcome) and from the per-fold
measurement files. It never reads the dataset, so oracle columns cannot enter,
and a failing diagnostic is recorded as `unavailable` or `error` without failing
the run. A compact summary is copied into `stage2/causal_estimate.json` under
`heterogeneity_diagnostics`.

Each section has an `interpretation` string, `per_fold` results, and a `pooled`
result. `ψ` below is the held-out AIPW score and `τ̂` is `estimated_cate` (the
ensemble mean under the default estimator policy). The `modifier_confirmation`
section is described in [Outer-fold confirmation](#outer-fold-confirmation).

| Section | What it computes | How to read it |
|---|---|---|
| `blp` | OLS of `ψ` on `[1, τ̂ − mean(τ̂)]` with HC3 standard errors | `beta1_ate` estimates the ATE among eligible patients. `beta2_heterogeneity` is the key quantity: a positive value with a small `beta2_p_value_one_sided_greater_than_0` is evidence that the CATE ranking captures real heterogeneity; near 0, none is detectable; near 1, the predictions are well scaled (`beta2_p_value_two_sided_equal_1` tests this); between 0 and 1, the predictions are too spread out; above 1, too compressed. |
| `gates` | Mean `ψ` within quintiles of `τ̂` (G1 lowest predicted effect, G5 highest), formed within each fold, plus a top-minus-bottom difference and a linear trend across groups | An increasing pattern and a positive top-minus-bottom difference mean that the ranking separates patients with larger and smaller effects. Similar group means mean no detectable heterogeneity. |
| `clan` | Mean of each retained modifier and confounder (held-out measurements; numeric values or category indicators) in the top versus bottom `τ̂` quintile | Describes which measured characteristics distinguish predicted high- and low-effect patients. Descriptive, not causal, and not adjusted for multiple comparisons. |
| `rate` | AUTOC and Qini of the `τ̂` ranking, with stratified bootstrap standard errors (500 replicates) | `TOC(u)` is the ATE among the top fraction `u` minus the overall ATE. AUTOC weights every `u` equally and is sensitive to the top of the ranking; Qini weights by `u` and has more power for diffuse heterogeneity. A positive value whose interval excludes 0 means the ranking is useful for prioritizing patients. |
| `overlap_weighted_ate` | The [ATO](glossary.md#ato), with weights `1 − e` for treated and `e` for comparator patients; a Hajek (weighted means) and an augmented (adds the outcome models) estimate, with effective sample size | A large gap between the ATO and the AIPW ATE means the ATE depends on patients with extreme propensities, or that effects differ in the region of overlap. |
| `ipw_weights` | Inverse-propensity weights `1/ê` (treated) and `1/(1 − ê)` (comparator) with `ê` clipped as in AIPW: Kish effective sample size `(Σw)²/Σw²`, its fraction of rows, and the maximum weight | An effective-sample-size fraction well below 1 or a very large maximum weight means the AIPW ATE leans on a few patients: weak overlap and high variance. |
| `trimming_curve` | The AIPW ATE restricted to `α ≤ ê ≤ 1 − α` for `α` in 0, 0.01, 0.025, 0.05, 0.10, 0.15 | Each `α` targets a different population. A flat curve suggests the ATE is not driven by poor-overlap patients; drift signals dependence on extreme propensities. `α = 0` uses all held-out patients, including any outside configured eligibility bounds. |
| `balance` | [SMDs](glossary.md#smd) of every catalog candidate before and after inverse-propensity weighting, for training patients (per fold, weighted by out-of-fold `ê`) and held-out patients (pooled, weighted by their `ê`); variables with weighted `|SMD| > 0.1` are listed | The **mean of a variable after weighting should be similar in both arms**. A weighted `|SMD|` above 0.1 flags a variable the propensity model does not balance: a likely gap in the propensity model, or a variable that strongly predicts treatment. |

**Pooling.** BLP coefficients are pooled across folds by fixed-effect
inverse-variance weighting (the fold median is reported as a robustness check);
the outer folds have disjoint held-out patients. GATES, CLAN, ATO, IPW weights,
the trimming curve, and held-out balance are recomputed on the pooled held-out
patients, with quintiles always formed within each fold because each fold has a
different model. RATE ranks pooled patients by their within-fold percentile of
`τ̂`, with the bootstrap stratified by fold. Training balance is reported per
fold only, because training sets overlap across folds. A fold whose `τ̂` is
constant cannot rank patients: its BLP slope is not identified, and it is left
out of the pooled GATES, CLAN, and RATE (listed in
`inputs.folds_with_constant_cate`).

All standard errors treat the cross-fitted nuisances as fixed. The section
`inputs.recomputed_psi_max_abs_difference_from_aipw_score` checks that the
saved AIPW scores match a recomputation from the saved nuisances.

## Output files

Per outer fold, under `stage2/outer_NNN/`:

| Path | Contents |
|---|---|
| `extraction/candidate_missingness_filter.json` | The informativeness filter's decision for every candidate |
| `selection/statistical_evidence.json` | All evidence cells, family summaries, nuisance supports, and policy (under `orthogonal_tests`, also `skipped_effect_evidence`) |
| `selection/nuisance_predictions.csv` | Out-of-fold nuisance predictions of the evidence families (outer-training patients only) |
| `selection/modifier_tests.json` | `orthogonal_tests`: every candidate's component and combined p-values, q-values, selection, and labels |
| `selection/modifier_count/selection.json` | The modifier selection report: for `orthogonal_tests`, the selected modifiers, estimator policy, and test summary; for `balanced_evidence`, the loss surface, guardrail, utilities, fold rankings, and the full-training ranking |
| `selection/modifier_count/<id>/testing_nuisances.csv` | `orthogonal_tests`: the testing nuisances and eligibility of each training patient |
| `selection/nuisance_models/<id>/` | `orthogonal_tests`: the stored inner-fold nuisance models (joblib) and their `manifest.json`, reused by the final estimation |
| `selection/modifier_count/<id>/fold_NNN/ranking/ranking.json` | `balanced_evidence`: one nested ranking with every candidate's component scores |
| `selection/elastic_net_selection.json` | Final role decisions with their sources |
| `final_definitions.json` | Retained variables, roles (with `modifier_evidence` for each modifier under `orthogonal_tests`), definitions, and the final estimator |
| `estimation/predictions.csv` | Held-out predictions for this fold, including per-estimator CATE columns under the ensemble |
| `estimation/diagnostics.json` | Fit audits (each ensemble component's), AIPW ATE, nuisance calibration, overlap, and methodology notes |
| `estimation/nuisance_fit_predictions.csv` | Training out-of-fold nuisances, raw and recalibrated, with eligibility |
| `estimation/propensity_overlap.json` | Training and held-out eligibility audit |
| `estimation/extraction_failure_diagnostics.json` | Extraction-failure rates on training and held-out patients |

Across folds, under `stage2/`: `cross_fitted_predictions.csv`,
`causal_estimate.json`, `heterogeneity_diagnostics.json`, and, for synthetic
data, `posthoc_oracle_ite_metrics.json` and
`posthoc_predictions_with_oracle_ite.csv`. The complete directory layout is in
[Configuration](configuration.md).

## Limitations

- **Selection optimism.** Stage 1 evidence, variable discovery, the
  extraction definitions, and the informativeness filter for an outer fold all
  used every outer-training patient's labels. The orthogonal tests' training
  p-values therefore do not account for the adaptive discovery that produced
  the candidates and are somewhat optimistic; the outer-fold confirmation
  re-tests the selected modifiers on held-out patients. Under
  `balanced_evidence`, the nested count and model selection re-derives the
  evidence and ranking inside each inner fold but holds the catalog and
  extractions fixed, so its validation R-loss is optimistic, as are the
  q-values used in ranking. Only the outer held-out predictions evaluate the
  complete pipeline, and the reported ATE standard error does not account for
  selection.
- **Measurement error from LLM extraction.** Extracted values can be wrong,
  missing, or inconsistent across patients, and the definitions themselves can
  differ between folds. Error in a confounder leaves residual confounding; error
  in a modifier attenuates estimated heterogeneity. Extraction failures are
  flagged rather than treated as "not documented", and failure rates by arm are
  reported, but extraction accuracy is not otherwise validated against a
  reference standard.
- **Pretreatment text.** The analysis assumes that every supplied note was
  written before treatment assignment. OCI does not verify this. Post-treatment
  text, such as documentation of early response or toxicity, can act as a
  mediator or collider and bias both the ATE and the CATEs.
- **Unmeasured confounding.** The estimates are causal only if, given the
  extracted variables, treatment assignment is unconfounded, every eligible
  patient could have received either treatment (positivity), and the usual
  consistency assumptions hold. Characteristics that the notes do not document,
  or that Stage 1 and Stage 2 fail to surface, are not adjusted for. Variable
  roles are predictive routing decisions, not demonstrated causal structure.
  OCI provides overlap and balance diagnostics but no sensitivity analysis for
  unmeasured confounding.
- **Nuisance recalibration.** Held-out nuisances are an average of five models,
  while their calibrator is learned from single-model predictions. Recalibration
  improves log loss, but held-out calibration slopes after recalibration can sit
  modestly above 1 (about 1.04 with five inner folds and about 1.2 with three).
  Slopes before and after recalibration are in each fold's diagnostics.
- **Inference.** Standard errors for the AIPW ATE and the diagnostics treat the
  cross-fitted nuisances and the eligibility rule as fixed. Only the causal
  forest produces per-patient intervals, and they do not describe the ensemble
  mean, which has none. The modifier tests' error control is nominal (see
  [Assumptions and limitations](#assumptions-and-limitations)). Under
  `balanced_evidence`, rankings, budgets, and guardrails are heuristics without
  error-rate guarantees, and validation R-loss is a noisy criterion that does
  not directly optimize recovery of true modifiers.
- **Synthetic benchmarks.** The bundled cohorts are LLM-written narratives
  generated from a known, simple data-generating process with one or five
  confounders and modifiers among 1,000 patients. Good recovery on them shows
  that the pipeline can find planted signals in synthetic text. It does not
  establish performance on real records, where documentation is noisier,
  confounding is richer, and the truth is unknown. See
  [Synthetic data](synthetic_data.md).

## References

- Benjamini Y, Hochberg Y. Controlling the false discovery rate: a practical
  and powerful approach to multiple testing. *Journal of the Royal Statistical
  Society, Series B* 1995;57(1):289–300.
- Chernozhukov V, Chetverikov D, Demirer M, et al. Double/debiased machine
  learning for treatment and structural parameters. *Econometrics Journal*
  2018;21(1):C1–C68.
- Chernozhukov V, Demirer M, Duflo E, Fernández-Val I. Generic machine learning
  inference on heterogeneous treatment effects in randomized experiments.
  NBER Working Paper 24678, 2018.
- Gao Z, Hastie T. Estimating heterogeneous treatment effects for general
  responses. *Biometrics* 2025;81(4):ujaf162.
- Hahn PR, Carvalho CM, Puelz D, He J. Regularization and confounding in linear
  regression for treatment effect estimation. *Bayesian Analysis* 2018;13(1):163–182.
- Li F, Morgan KL, Zaslavsky AM. Balancing covariates via propensity score
  weighting. *Journal of the American Statistical Association* 2018;113(521):390–400.
- Liu Y, Xie J. Cauchy combination test: a powerful test with analytic p-value
  calculation under arbitrary dependency structures. *Journal of the American
  Statistical Association* 2020;115(529):393–402.
- MacKinnon JG, White H. Some heteroskedasticity-consistent covariance matrix
  estimators with improved finite sample properties. *Journal of Econometrics*
  1985;29(3):305–325.
- Nie X, Wager S. Quasi-oracle estimation of heterogeneous treatment effects.
  *Biometrika* 2021;108(2):299–319.
- Semenova V, Chernozhukov V. Debiased machine learning of conditional average
  treatment effects and other causal functions. *Econometrics Journal*
  2021;24(2):264–289.
- Stouffer SA, Suchman EA, DeVinney LC, Star SA, Williams RM. *The American
  Soldier, Vol. 1: Adjustment During Army Life.* Princeton University Press,
  1949.
- Yadlowsky S, Fleming S, Shah N, Brunskill E, Wager S. Evaluating treatment
  prioritization rules via rank-weighted average treatment effects. *Journal of
  the American Statistical Association* 2025;120(549):38–51.
- EconML: [CausalForestDML](https://www.pywhy.org/EconML/_autosummary/econml.dml.CausalForestDML.html)
  and [RScorer](https://www.pywhy.org/EconML/_autosummary/econml.score.RScorer.html).
