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
  ensemble), whose interaction model leaves the selected modifiers unpenalized.
  Held-out patients then confirm the selected modifiers, for reporting only.
- **`balanced_evidence`** ranks candidates by a fixed formula and chooses the
  number of modifiers and one effect model jointly by nested validation.

Both rules share the evidence families, the predictive nuisance selection, the
final refit of each effect model (apart from the interaction model's
[selected-modifier penalty](#selected-modifier-penalty)), the nuisance
construction, and the diagnostics.

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
12. [Sensitivity to unmeasured confounding](#sensitivity-to-unmeasured-confounding)
13. [Output files](#output-files)
14. [Limitations](#limitations)

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

**Grouping.** A candidate's categorical levels, its missingness indicator, its
[documented-unknown](glossary.md#documented-unknown) indicator, and its
[extraction-failure](glossary.md#extraction-failure) indicator form one group
(see [Not documented, documented unknown, and failed cells](#not-documented-documented-unknown-and-failed-cells)).
Penalties select or drop the whole group, interaction tests are omnibus over the
group, and permutations move all its columns together.

#### Not documented, documented unknown, and failed cells

Every encoder (the selection designs, the modifier tests' grouped design, the
final nuisance and effect models, and the diagnostics) keeps three non-value
states apart:

| State | Stored as | Encoded as |
|---|---|---|
| Not documented: the record does not mention the variable | null | The "missing" indicator; a continuous value is imputed at the training median |
| [Documented unknown](glossary.md#documented-unknown): the record states that the value is unknown, indeterminate, pending, not assessed, or equivocal | The reserved value `documented_unknown`, for every value type | Its own `<variable>:documented_unknown` indicator in the variable's group; a continuous value is imputed at the training median; never a category level, never pooled into a rare-level "other" column, and never the reference level |
| [Extraction failure](glossary.md#extraction-failure) | null, flagged in `<variable>__oci_extraction_failed` | Its own failure indicator in the variable's group |

The documented-unknown indicator exists when the training rows contain the
state; a held-out documented unknown for a variable without training
occurrences is encoded as missing and listed in
`documented_unknown_report_heldout.json`. In the modifier tests the indicator is
part of the tested group, as an observed patient state. Unknown-like wording is
mapped conservatively: only a whole value equal to one of "unknown", "not
known", "status unknown", "indeterminate", "undetermined", "pending", "result
pending", "results pending", "not assessed", "not evaluated", "equivocal", or
"inconclusive" (case, spacing, and punctuation ignored) is treated as the
reserved state, so a category such as "Unknown primary" is unchanged. A
declared category with one of these labels is removed from the definition only
when the remaining categories still form a valid ontology (two for binary,
at least two otherwise); otherwise it is a genuine answer, such as "Not
assessed" in an assessed/not-assessed variable. Investigator-specified
variables (`stage2.explicit_features`) keep every declared category, because
their definitions are locked; an extracted value matching a declared category
is that category. An extraction answer of bare
"unknown" stays a missing-value placeholder, because extraction models often
write it for an unmentioned variable; the prompts ask for `documented_unknown`
for an explicit statement. When repeated observations disagree, a documented
value takes precedence over documented-unknown statements.

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
[nuisance construction](#nuisance-construction) is computed: nuisance models on
the complete catalog for treatment and for the outcome in each treatment arm,
fitted in the inner cross-fitting folds with the final estimation's learner,
seeds, encoders, and standardization. Each training patient receives the raw
`ê`, `μ̂0`, and `μ̂1` of the inner-fold model that did not see it; training
predictions are not recalibrated. Held-out patients take no part; their measurements do not
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
only the training patients whose out-of-fold `ê` lies within the bounds, as
the final estimation does.

### Grouped design

Each candidate `j` is tested on its grouped design `Z_j`, encoded on the
eligible training patients: the value centered at its median (continuous
candidates, with missing values imputed at the median) or one indicator per
non-reference category (categorical candidates; the reference is the first
declared category that is observed), plus the "not documented" indicator, the
[documented-unknown](glossary.md#documented-unknown) indicator when the
training patients contain that state, and, for candidates with training
extraction failures, the [extraction-failure](glossary.md#extraction-failure)
indicator. Indicator
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
  therefore optimistic **screening** values (`p_value_role` in
  `modifier_tests.json`), not inferential ones. The inferential result is the
  [outer-fold confirmation](#outer-fold-confirmation), which repeats the tests
  on held-out patients.
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
  alone. No per-patient interval is reported for the ensemble because no
  available interval accounts for the uncertainty of variable discovery,
  extraction, and selection. The honest uncertainty statements are the GATES
  group effects (see [Heterogeneity and overlap diagnostics](#heterogeneity-and-overlap-diagnostics))
  and the per-modifier held-out effect sizes of the
  [outer-fold confirmation](#outer-fold-confirmation).
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
  patients took no part in discovery, extraction definitions, or selection,
  these tests are valid conditional on that fold's own selection (sample
  splitting). They are the inferential result (`p_value_role`); the training
  p-values are screening values.
- **Effect sizes.** For each selected modifier, `effects` gives the held-out
  R-learner projection coefficients (the change in the absolute treatment
  effect, a risk difference for binary outcomes, per unit of each design
  column) and the AIPW projection coefficients, each with an HC3 standard
  error, 95% interval, and two-sided p-value. Categorical modifiers have one
  coefficient per level versus the reference level; continuous modifiers also
  get the coefficient per standard deviation of the fold's observed
  outer-training values (`training_sd`). Missingness, documented-unknown, and
  extraction-failure columns are reported too. Each modifier is analysed alone, as in the tests.
- **Pooled.** Counts of selected and confirmed modifiers per fold and in total.
- **Across folds.** Each fold defines and extracts its own variables, so the
  selected variables are grouped into **concepts**: identical normalized names
  always match, and other pairs match when a deterministic lexical similarity
  (the mean of TF-IDF cosines of word 1–2-grams and character 3–5-grams of the
  name, description, and measurement definition) is at least 0.7 for every
  cross-fold pair of the group (complete linkage), with at most one variable
  per fold per concept, assigned in order of decreasing similarity. The
  threshold is conservative: a false merge would combine different concepts,
  while a missed merge only forgoes aggregation, so differently worded synonyms
  can stay separate (lexical overlap scores, for example, brain versus liver
  metastases at about 0.65). For each concept, the held-out combined p-values
  of all outer folds are aggregated by the Meinshausen–Meier–Bühlmann quantile
  rule with `γ = 0.5`, `Q = min(1, 2 · median_j p_j)`, with `p_j = 1` for folds
  where the concept was not selected or could not be tested, and
  Benjamini–Hochberg then runs across concepts (`aggregated_q_value`). Each
  concept lists its matched variables, definitions, similarity scores, and
  "selected in `k` folds, confirmed in `j`". A concept selected in one fold is
  never tested on another fold's held-out patients, because those patients were
  in that fold's training data. The aggregation treats the folds' tests as
  tests of one concept, which holds only approximately when folds define the
  variable differently.
- **Pooled effect sizes.** For a concept matched in two or more folds whose
  members have the same columns (a continuous value, or the same category
  labels), the effects are pooled by fixed-effect inverse-variance weighting,
  with Cochran's `Q`, `I²`, and the fold range. Continuous modifiers are pooled
  per training SD, because folds can define units differently; otherwise the
  effects are reported per fold only, with the reason.

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
  intercept, `β_T`, and the two adjustment covariates are unpenalized; the
  confounder main effects carry a grouped elastic-net penalty (mixing ratio
  `l1_ratio`, default 0.8) tuned by training-only cross-validated outcome loss
  (log loss for binary outcomes, squared error for continuous outcomes). Under
  `orthogonal_tests` the modifiers' interactions and main effects are left out
  of that penalty, subject to a guard (see
  [Selected-modifier penalty](#selected-modifier-penalty)); otherwise every
  main effect and interaction shares it.
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

### Selected-modifier penalty

One penalty shared by the main effects and the interactions is tuned for
outcome prediction, to which the interactions contribute little. When many of
the main effects do not predict the outcome (for example confounders that only
predict treatment), cross-validation chooses a strong penalty, which also
shrinks the interactions and compresses the estimated heterogeneity. Under
`orthogonal_tests` the modifiers were already chosen by calibrated tests, and
the final interaction model does not shrink them a second time: the interaction columns `T·Z_mod` of every selected modifier (including
investigator-specified modifiers) and that modifier's main-effect columns
(strong hierarchy) are exempt from the tuned penalty in every cross-validation
fit and in the refit. The penalty, still tuned by cross-validated outcome loss,
applies to the remaining main effects. The exemption removes the second
shrinkage at the price of variance: a falsely selected modifier enters at full
strength. `modifier_selection.unpenalized_selected_interactions: false` keeps
the single shared penalty.

- **Guard.** The exemption adds `2c` unpenalized parameters, where `c` is the
  number of encoded design columns of the selected modifiers (values, category
  contrasts, and missingness, documented-unknown, and extraction-failure
  indicators). It is used
  only when `2c` is at most a limit computed on the effect-eligible training
  patients: `min(events, non-events) / r` for binary outcomes and
  `rows / r` for continuous outcomes, where `r` is
  `modifier_selection.unpenalized_events_per_parameter` (default 10 for binary
  and 15 for continuous outcomes). Otherwise the shared-penalty model is
  fitted.
- **Separation fallback.** An unpenalized logistic term diverges under complete
  or quasi-complete separation, for example when every treated patient in a
  rare category has the outcome. The exempt fit is accepted when the penalty
  cross-validation and the refit converge and no selected-modifier interaction
  column changes the linear predictor by more than 10 across its observed
  training range (in log odds for binary outcomes, in training-outcome standard
  deviations for continuous outcomes). A failed fit is refitted with a fixed
  ridge penalty `0.5 · 0.01 · ‖b‖²` (on the mean-loss scale and the
  standardized columns; for binary outcomes a `N(0, 1/(0.01 n))` prior on each
  coefficient, with `n` training patients) on the selected-modifier columns
  only; they stay outside the tuned penalty, and the same checks apply. If that
  fit also fails, the shared-penalty model is fitted. Separation that moves the
  CATE always moves an interaction coefficient. The main-effect columns' range
  effects are reported but not checked, because their size also reflects
  collinearity with the prognostic adjustment covariate, with or without a
  penalty.
- **Zero modifiers.** With no selected modifier (or no encoded modifier
  column) the model is exactly the shared-penalty model.
- **Audit.** `interaction_fit_audit.selected_modifier_penalty` in
  `estimation/diagnostics.json` records the settings, the guard (encoded
  columns per modifier, events and non-events or rows, the limit, and the
  decision), each attempted fit with its convergence and range-effect
  diagnostics, and the path used: `unpenalized`, `separation_ridge`, or
  `fully_penalized` with its reason (`no_selected_modifiers`,
  `selected_modifiers_have_no_encoded_columns`, `events_per_parameter_guard`,
  or `separation_or_nonconvergence`). The architecture diagnostics, including
  each ensemble component's audit, repeat a compact `selected_modifier_penalty`
  summary, which reads `disabled_by_configuration` when the setting is false.

These settings do not affect selection, so changing them refits only the
final estimation. Under `balanced_evidence`, its nested validation fits, the
`penalized_interactions` evidence family, and the final interaction model all
use the single shared penalty.

## Nuisance construction

One nuisance construction feeds every downstream quantity of the final
estimation: propensity eligibility, the AIPW scores, the DINA offsets, the
interaction model's adjustment covariates, the forest's prognostic score, and
the diagnostics. Its training half, computed earlier and identically, also
supplies the [testing nuisances](#testing-nuisances) of the orthogonal tests.

1. **Models.** Within the outer-training patients, the five inner folds are
   used for cross-fitting. In each inner fold, three nuisance models are fitted
   on the other four folds: one for treatment and, separately in each treatment
   arm, one for the outcome. Every model uses the **complete catalog** of
   candidates, with categorical encoders fitted on that model's training rows.
   The selected nuisance supports are recorded for transparency but do not
   restrict these models: a support chosen using all outer-training patients
   would otherwise screen the inputs of their own out-of-fold predictions. The
   learner is `stage2.nuisance_learner` (see [Nuisance learner](#nuisance-learner)):
   by default a super learner, otherwise a standardized elastic net (logistic
   for binary targets; penalty by 3-fold cross-validation, mixing ratio 0.8).
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
   labels, candidate definitions, folds, seed, nuisance learner, source code,
   and library versions (Python, NumPy, pandas, SciPy, scikit-learn, joblib). The final
   estimation recomputes that fingerprint from its own inputs and loads a model
   only when the fingerprint and file hash match and the model's input width
   agrees; otherwise, or on any load error, it refits that model. Reused and
   refitted models give identical predictions, and each model's fit audit is
   written either way, marked `model_source` `reused_stored_model` or
   `fitted`. Under `balanced_evidence` no models are stored and the final
   estimation fits them.
5. **Held-out calibration.** Training out-of-fold predictions are used raw. For
   the held-out patients, the propensity and, for binary outcomes, both
   arm-specific outcome predictions of each inner-fold model are
   Platt-calibrated before averaging: inner-fold model `k`'s calibrator, a
   logistic regression of the label on `[1, logit(prediction)]`, is fitted on
   model `k`'s own inner-validation patients (its out-of-fold predictions and
   their training labels; for `μ0` and `μ1`, only the validation patients in
   the comparator or treated arm), applied to model `k`'s held-out predictions,
   and the five calibrated predictions are averaged. Each calibrator therefore
   matches the model it corrects, and no held-out label is used. A calibrator
   falls back to the identity map when it has fewer than 20 rows, a single
   class, constant predictions, separation, or a nonpositive slope.
   Continuous-outcome predictions are not calibrated; their held-out value is
   the average of the five models.

The final predictions are written as `propensity`, `mu0`, and `mu1` (training
rows: raw out-of-fold; held-out rows: the average of the calibrated fold-model
predictions); the uncalibrated values are kept as `propensity_uncalibrated`,
`mu0_uncalibrated`, and `mu1_uncalibrated` (equal to the final values for
training rows; the uncalibrated average for held-out rows). Each fold's
`estimation/diagnostics.json` reports calibration before and after held-out
calibration (`nuisance_calibration`, including a `calibration_slope_summary`
for training and held-out rows), each fold model's calibrator parameters, status,
and fit rows (`nuisance_calibration.recalibration`), nuisance fit audits,
including any elastic-net fit that reached its iteration limit (with
`model_sources` counts when a model store was consulted), and, under
`orthogonal_tests`, `nuisance_model_store` (`available` or `absent`, with the
store fingerprint).

### Nuisance learner

The default `stage2.nuisance_learner`, `super_learner`, stacks three base
learners (`stage2.super_learner.learners`, any nonempty subset):

| Learner | Settings |
|---|---|
| `elastic_net` | The standardized elastic net above, with its own penalty cross-validation |
| `gradient_boosting` | scikit-learn histogram gradient boosting: depth 3, learning rate 0.05, at most 300 iterations, L2 regularization 1.0, at least 20 rows per leaf; early stopping on a 15% validation fraction (20 iterations without improvement) when the training rows number at least 100 and, for binary targets, the smaller class has at least 10 rows |
| `random_forest` | scikit-learn random forest: 500 trees, at least 5 rows per leaf, square-root feature sampling |

Each nuisance model is stacked inside its own training rows:

1. **Internal cross-validation.** `stage2.super_learner.cv_folds` folds (default
   5; stratified on the label for binary targets) give out-of-fold predictions
   of every base learner. A small arm or rare class reduces the number of folds
   (to the smaller class count, or half the rows for continuous targets), and
   the reduction is recorded; with fewer than two possible folds the elastic net
   alone is used.
2. **Weights.** Non-negative weights that sum to one minimize the internal-CV
   log loss of the combined probability (binary targets, probabilities clipped
   to `[10⁻⁶, 1 − 10⁻⁶]`, SLSQP on the simplex) or the squared error
   (continuous targets, non-negative least squares normalized to sum to one;
   the best single learner if every weight is zero).
3. **Refit.** Learners with positive weight are refitted on all training rows,
   and the model predicts the weighted combination. A learner with zero weight
   does not affect predictions and is not refitted.

Seeds derive from the inner-fold seed (elastic net, boosting, forest, and the
internal folds each have a fixed offset), so fits are deterministic. Each fit
runs in one modeling-pool job with one thread (the forest uses `n_jobs=1`;
boosting's OpenMP threads follow the job's thread limit), so the learners stay
within `stage2.modeling_workers`. The held-out calibration (step 5 above)
applies to the stacked predictions exactly as to the elastic net.

Every model's fit audit in `estimation/nuisance_fit_audits/` records the stack
weights, each learner's internal-CV loss, the stack's internal-CV loss (with
the weights refitted on the other internal folds, so it is not optimistic),
the requested and effective internal folds, the seeds and settings, each
refitted learner's audit (the elastic net's selected regularization and
iteration-limit flags; boosting's iteration count), and the elastic net's
per-fold regularization and iteration-limit flags in the internal folds.
`estimation/diagnostics.json` summarizes them under `nuisance_learner`: mean
stack weights, mean learner and stack CV losses, and fit modes for `propensity`,
`mu0`, and `mu1`.

With `nuisance_learner: "elastic_net"`, every nuisance model is the elastic net
alone. The models that assign nuisance roles during selection and the causal
forest's internal nuisances are elastic nets under either setting.

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
| Orthogonal modifier tests | Out-of-fold propensity (testing nuisances) | Not used |
| Modifier count and model selection (`balanced_evidence`) | Out-of-fold training propensity | Propensity from models fitted on the fold's training patients |
| Final refit and estimation | Out-of-fold propensity | Inner-fold-ensemble propensity, each fold model calibrated on its own validation patients |
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

### Estimands: clipping, trimming, and overlap weighting

Three different tools address extreme propensities, and they answer different
questions:

- **Clipping** (`stage2.propensity_clip`, default 0.02) bounds the AIPW
  denominators to `[c, 1 − c]`. It keeps every patient and the ATE population,
  trading a little bias for variance. `clip_sensitivity` in the diagnostics
  recomputes the ATE at clips 0.01, 0.02, and 0.05.
- **Trimming** (the eligibility bounds above, and the `trimming_curve`
  diagnostic) removes patients with estimated propensities near 0 or 1. It
  changes the target population, and that population is data-dependent: it is
  defined by each fold's fitted propensity model, so it differs between folds
  and between learners and is hard to describe clinically.
- **Overlap weighting** (the ATO) weights every patient by `e(1 − e)`. It
  defines a smooth target population, the patients in clinical equipoise,
  without a cutoff, and its weights vanish at the extremes so no clipping is
  needed.

`stage2/causal_estimate.json` reports the clipped AIPW ATE and the augmented ATO
as **co-primary estimands** (`co_primary_estimands`), per fold and pooled. The ATE
is the mean of the eligible patients' AIPW scores; the ATO is the augmented
overlap-weighted estimator of Li, Morgan, and Zaslavsky on every held-out patient
with the cross-fitted nuisances, with an influence-function standard error that
treats the nuisances as fixed. `stage2.lead_estimand` (`"ate"` or `"ato"`)
records which estimand the study pre-specifies to lead with; it changes only
the order and the role labels, never the computation, and should be set before
the results are seen. The gap `ate_minus_ato`, the trimming curve, and the clip
sensitivity are secondary sensitivity analyses.

`cross_fitted_predictions.csv` flags patients whose final held-out propensity
lies outside `stage2.overlap_flag_bounds` (default `[0.05, 0.95]`) in
`outside_overlap`, with counts per fold and pooled in `causal_estimate.json` and
the diagnostics (`overlap_flags`). The flag is reporting only and never removes
patients or changes estimates; trimming remains the separate, opt-in eligibility
bounds.

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

**Primary and secondary analyses.** The pre-specified primary heterogeneity
test is the pooled BLP `beta2` with its one-sided p-value for `beta2 > 0`,
reported as `primary_heterogeneity_test` in `heterogeneity_diagnostics.json` and
`causal_estimate.json`. GATES and RATE are secondary analyses; CLAN, the CATE
variance decomposition, and the modifier confirmation are descriptive. No
diagnostic is used for selection or estimation. Per-fold results rest on one
fold's held-out patients (about 200 at n = 1000) and are low-powered. Every
result also rests on a single random outer split: repeated splitting with
medians across splits (Chernozhukov, Demirer, Duflo and Fernández-Val) is not
performed, and would require rerunning the pipeline with other split seeds.

| Section | What it computes | How to read it |
|---|---|---|
| `blp` | OLS of `ψ` on `[1, τ̂ − mean(τ̂)]` with HC3 standard errors | `beta1_ate` estimates the ATE among eligible patients. `beta2_heterogeneity` is the key quantity: a positive value with a small `beta2_p_value_one_sided_greater_than_0` is evidence that the CATE ranking captures real heterogeneity; near 0, none is detectable; near 1, the predictions are well scaled (`beta2_p_value_two_sided_equal_1` tests this); between 0 and 1, the predictions are too spread out; above 1, too compressed. |
| `gates` | Mean `ψ` within quintiles of `τ̂` (G1 lowest predicted effect, G5 highest), formed within each fold, each with its `mean_estimated_cate`; a top-minus-bottom difference; a linear trend across groups; and `calibration`, the HC3 slope of `ψ` on each patient's group mean `τ̂` | An increasing pattern and a positive top-minus-bottom difference mean that the ranking separates patients with larger and smaller effects. Similar group means mean no detectable heterogeneity. A calibration slope near 1 means the group effects match the predicted group means; near 0, the predicted spread is not detectably real. |
| `clan` | Mean of each retained modifier and confounder (held-out measurements; numeric values or category indicators) in the top versus bottom `τ̂` quintile | Describes which measured characteristics distinguish predicted high- and low-effect patients. Descriptive, not causal, and not adjusted for multiple comparisons. |
| `cate_variance_decomposition` | Per fold, OLS `R²` of `τ̂` (and of each ensemble component's CATE column) on (a) the prognostic score alone (`logit μ̂0` for binary outcomes, `μ̂0` for continuous outcomes, as in the forest's X and the interaction model's adjustment), (b) the selected modifiers' grouped design alone (the modifier tests' encoding of the held-out measurements), and (c) both; `unique_prognosis` = `R²(c) − R²(b)`, `unique_modifiers` = `R²(c) − R²(a)`, `shared` = `R²(a) + R²(b) − R²(c)`, `unexplained` = `1 − R²(c)` | Shows how much of the predicted CATE spread follows baseline risk rather than the named modifiers: on the risk-difference scale effects vary with baseline risk even under a constant odds ratio, so a wide spread with few modifiers can be mostly prognosis. Descriptive, not causal; nonlinear or interaction structure that a linear projection misses (for example a forest's splits) lands in `unexplained`, and many modifier columns relative to rows inflate `R²(b)`. A fold with a constant `τ̂` is `unavailable`. |
| `rate` | AUTOC and Qini of the `τ̂` ranking, with stratified bootstrap standard errors (500 replicates) | `TOC(u)` is the ATE among the top fraction `u` minus the overall ATE. AUTOC weights every `u` equally and is sensitive to the top of the ranking; Qini weights by `u` and has more power for diffuse heterogeneity. A positive value whose interval excludes 0 means the ranking is useful for prioritizing patients. |
| `overlap_weighted_ate` | The [ATO](glossary.md#ato), with weights `1 − e` for treated and `e` for comparator patients; a Hajek (weighted means) and an augmented (adds the outcome models) estimate, with effective sample size | A large gap between the ATO and the AIPW ATE means the ATE depends on patients with extreme propensities, or that effects differ in the region of overlap. |
| `ipw_weights` | Inverse-propensity weights `1/ê` (treated) and `1/(1 − ê)` (comparator) with `ê` clipped as in AIPW: Kish effective sample size `(Σw)²/Σw²`, its fraction of rows, and the maximum weight | An effective-sample-size fraction well below 1 or a very large maximum weight means the AIPW ATE leans on a few patients: weak overlap and high variance. |
| `clip_sensitivity` | The AIPW ATE over the eligible patients recomputed with the propensity clipped at 0.01, 0.02, and 0.05 (and the configured clip, marked `configured`), with standard errors | Clipping changes the weights, not the population. Estimates that move with the clip show dependence on patients with extreme propensities. The configured entry equals the reported ATE. |
| `overlap_flags` | Counts of held-out patients outside `stage2.overlap_flag_bounds`, per fold and pooled | Reporting only; the same flag as the `outside_overlap` column. |
| `unmeasured_confounding_sensitivity` | E-values, omitted-variable-bias bounds with robustness values, and benchmark-implied bounds for the ATE and the ATO | See [Sensitivity to unmeasured confounding](#sensitivity-to-unmeasured-confounding). |
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
`inputs.folds_with_constant_cate`). The CATE variance decomposition is computed
per fold; its pooled shares are row-weighted means of the fold partitions,
with the fold range.

All standard errors treat the cross-fitted nuisances as fixed. The section
`inputs.recomputed_psi_max_abs_difference_from_aipw_score` checks that the
saved AIPW scores match a recomputation from the saved nuisances.

## Sensitivity to unmeasured confounding

The estimates assume that the extracted variables capture every confounder.
Three reporting-only analyses ask how strong an unmeasured confounder would
have to be to change the conclusion. They run after every fold's predictions
are frozen, use only frozen artifacts (the pooled held-out predictions and each
fold's own refits on its outer-training patients), and are never read by
selection, estimation, or another fold. The full results are in
`unmeasured_confounding_sensitivity` in `stage2/heterogeneity_diagnostics.json`;
`stage2/causal_estimate.json` repeats the pooled E-values, robustness values,
and strongest benchmark next to the co-primary estimands. E-values and bounds
are computed for both co-primary estimands, per fold on that fold's held-out
patients and pooled by stacking every fold's held-out patients; benchmarks cover
the ATE. A result that cannot be computed is `unavailable` with a reason and
does not affect the rest of the run.

### E-values

The E-value (VanderWeele and Ding 2017) is the minimum strength of association,
on the risk-ratio scale, that an unmeasured confounder would need to have with
both the treatment and the outcome, conditional on the measured covariates, to
fully explain away the observed association. For a risk ratio `RR ≥ 1`,

```text
E = RR + sqrt(RR × (RR − 1))
```

and for `RR < 1` the formula is applied to `1/RR`. The confidence-limit E-value
(`evalue_confidence_limit`) is the E-value of the confidence limit closest to
the null; it is 1 when the interval includes 1.

- **Binary outcomes.** The risk ratio is the causal risk ratio
  `E[Y(1)] / E[Y(0)]` built from the same two arm means whose difference is the
  reported risk difference, so this is the E-value for the estimated causal risk
  ratio corresponding to the reported risk difference. For the ATE the arm means
  are the clipped AIPW means `mean(μ̂1 + T(Y − μ̂1)/ê)` and
  `mean(μ̂0 + (1 − T)(Y − μ̂0)/(1 − ê))` over the effect-eligible patients; for
  the ATO they are the augmented overlap-weighted arm means. The 95% interval is
  the delta-method interval on `log RR` from the arm-mean influence functions,
  with the cross-fitted nuisances held fixed. If an estimated arm mean is not
  positive, the risk ratio and its E-values are `unavailable`.
- **Continuous outcomes.** The estimate is converted to a standardized mean
  difference `d = estimate / SD(Y)`, where `SD(Y)` is the marginal standard
  deviation of the observed outcome over the patients that enter the estimate
  (the effect-eligible held-out patients for the ATE, every held-out patient for
  the ATO). A residual standard deviation would give a larger `d` and a larger
  E-value, so the marginal one is the conservative choice. Then `RR ≈ exp(0.91·d)`
  with interval `exp(0.91·(d ± 1.96·SE(d)))`, `SE(d) = SE / SD(Y)`. This is an
  approximate conversion that combines Chinn (2000), `ln(OR) ≈ 1.81·d`, with
  VanderWeele (2017), `RR ≈ sqrt(OR)` for a common outcome, as the EValue R
  package does for continuous outcomes (`evalues.MD`, `evalues.OLS`):
  `RR ≈ sqrt(OR) = exp(1.81·d/2) ≈ exp(0.91·d)`. The statement
  `RR ≈ exp(0.91·d)` is this documentation's statement of that combination, not
  a verbatim formula from either paper.

### Omitted-variable-bias bounds and robustness values

The omitted-variable-bias framework of Chernozhukov, Cinelli, Newey, Sharma and
Syrgkanis bounds the bias of a debiased estimate of a linear functional of the
outcome regression, such as the ATE, by how much omitted confounders would
explain the outcome and the treatment (through the Riesz representer `α`). OCI
follows the formulas of the DoubleML sensitivity guide:

```text
W = (Y, T, X)
σ² = E[(Y − g(T, X))²],   g(T, X) = T·μ̂1 + (1 − T)·μ̂0
ν² = E[α²],               estimated by  ν̂² = Eₙ[2·m(W, α̂) − α̂²]
|θ̃ − θ| ≤ |ρ| · σ · ν · C_Y · C_D,   C_Y² = cf_y,   C_D² = cf_d / (1 − cf_d)
bounds:  θ ± |ρ|·σ·ν·C_Y·C_D,  with ρ = 1
```

- `θ̃` is the estimate that adjusts for the measured variables and `θ` the one
  that would also adjust for the omitted confounders. `m(W, α)` applies the
  effect functional to `α`: `α(1, X) − α(0, X)` for the ATE, weighted by
  `h(X)/E[h]` for the ATO.
- `cf_y` is the share of the outcome's residual variation, given treatment and
  the measured covariates, that the omitted confounders would explain (a
  partial R²). `cf_d` is the share of the variation of the Riesz representer
  that they would explain; for the ATE the representer is the
  inverse-propensity weight. `ρ` is the correlation between the two omitted
  parts; `ρ = 1`, the most adversarial value, makes the bounds conservative.
- **ATE**: `α = T/ê − (1 − T)/(1 − ê)` and `m(W, α) = 1/ê + 1/(1 − ê)`, with `ê`
  clipped at `stage2.propensity_clip` as in the AIPW scores, over the
  effect-eligible held-out patients.
- **ATO**: the overlap weights `h(X) = ê(X)(1 − ê(X))` are treated as a fixed
  function of the measured covariates, so the target is the `h`-weighted average
  effect. Its representer is `α = h/E[h] · (T/ê − (1 − T)/(1 − ê)) = (T − ê)/E[h]`,
  with `m(W, α) = h/E[h]²`, over every held-out patient. This bounds the bias
  for the measured-covariate equipoise population; it does not describe how an
  omitted confounder would redefine that population.
- **Confidence bounds.** The adjusted scores `ψ ∓ ψ_bias`, with
  `ψ_bias = |ρ|·C_Y·C_D·(σ²·ψ_ν² + ν²·ψ_σ²)/(2σν)` from the centered score
  elements of `σ²` and `ν²`, give standard errors for each bound (sample variance
  with `n − 1`, as for the reported standard errors). The lower confidence
  bound is `θ − bias − 1.96·SE₋` and the upper is `θ + bias + 1.96·SE₊`; at
  `cf_y = cf_d = 0` they equal the reported 95% confidence interval. Each
  adjusted score is its bound's delta-method influence function: `ψ` is the
  centered AIPW (or ATO) score, and `θ − bias` uses `ψ − ψ_bias`. DoubleML 0.11
  adds `ψ_bias` to its Jacobian-scaled score, which for this model is `−ψ`, so
  its lower and upper bound standard errors are exchanged relative to these;
  the tests check OCI's assignment by simulation, and an optional test
  (`tests/test_stage2_sensitivity_doubleml.py`, run when DoubleML is installed)
  checks the estimate, `σ²`, `ν²`, the bounds, RV, and the benchmark gain
  statistics against DoubleML given the same nuisance predictions.
- **Robustness values.** `robustness_value` (RV) is the common value
  `x = cf_y = cf_d` at which the bound reaches zero effect, found by bisection
  (Cinelli and Hazlett introduced the robustness value). Confounders that
  explain a share `x` of the outcome's residual variation and a share `x` of the
  representer's variation could explain the estimate away; weaker ones could
  not. `robustness_value_confidence` (RVa) is the same for the confidence bound;
  it is 0 when the confidence interval already includes zero.
- **Bound table.** `contour` lists the bounds and confidence bounds for every
  pair of `cf_y` and `cf_d` in 0.01, 0.02, 0.05, 0.10, and 0.20 at `ρ = 1`, with
  `confidence_bounds_exclude_zero`, ready for a contour plot.

### Benchmarking against discovered variables

An unmeasured confounder is easier to judge against the measured ones
(Cinelli and Hazlett 2020; gain statistics as in the DoubleML guide). In each
outer fold, OCI ranks that fold's discovered nuisance candidates, refits the
nuisances without each of the top `stage2.sensitivity.benchmark_groups`
(default 5; 0 disables the refits), and measures what the omitted variable
explained:

1. **Long reference.** The propensity model and the two arm outcome models are
   refitted with every candidate as **elastic nets**, one fit per model on all
   outer-training patients, scored on the fold's held-out patients without
   recalibration. The reported estimate may use the super learner; the
   benchmark uses the elastic net for both the long reference and the short
   models so that the comparison isolates the omitted variable rather than the
   learner, and to keep the cost bounded.
2. **Ranking.** Each variable's importance is the product of the L2 norm of its
   standardized elastic-net coefficients (all of its encoded columns) in the
   long reference's propensity model and the combined norm over both arm outcome
   models; ties are broken by the sum of the two norms, then by name. Variables
   whose product is zero, that is, that enter only one of the models, are not
   benchmarked. Each variable's role in the fold's final selection is
   reported with it.
3. **Short models.** For each benchmarked variable the same three models are
   refitted without its columns, and the held-out `σ²_short`, `ν²_short`, and
   `θ_short` (the ATE over the same effect-eligible held-out patients) are
   computed as above.
4. **Gains.** With `R² = 1 − σ²/Var(Y)` from the long reference,

   ```text
   cf_y = (R² − R²_short) / (1 − R²)
   cf_d = (1 − R²_α) / R²_α,          R²_α = ν²_short / ν²
   ρ̂   = (θ_short − θ) / sqrt((σ²_short − σ²)(ν² − ν²_short))
   ```

   with `cf_y` and `cf_d` limited to `[0, 1]` and `|ρ̂|` to 1. When omitting the
   variable does not raise `σ²` and lower `ν²`, `ρ̂` is not identified and is
   reported as null.
5. **Implied bounds.** Each variable's `(cf_y, cf_d)` is applied to the fold's
   ATE and to the pooled ATE, at `ρ = 1` and at `ρ̂`, giving statements such as
   "a confounder as strong as *variable* (fold 2) would move the pooled ATE
   bound to [a, b] at ρ = 1".

Benchmarking runs `3 + 3k` elastic-net fits per fold for `k` benchmarked
variables, inside the shared CPU modeling pool (`stage2.modeling_workers`), with
checkpoints in `outer_NNN/estimation/sensitivity_benchmarks/fits/<id>/` and the
results in `outer_NNN/estimation/sensitivity_benchmarks/benchmarks.json`; a
failure is recorded in that file, never raised. The DoubleML guide notes that
benchmarking is computationally heavy and that `ρ̂` can be unstable when the
benchmark variables are weak confounders. The ATO is not benchmarked, because
omitting a variable also changes its overlap weights and therefore its target
population.

### Reading the sensitivity analyses

- **E-value.** An E-value of 2 means that a confounder associated with both
  treatment and outcome by risk ratios of 2 each, beyond the measured
  variables, could explain the estimate away, and a weaker one could not. Values
  near 1 mean weak confounding could explain the estimate; a
  confidence-limit E-value of 1 means the interval already includes the null.
- **Robustness value.** An RV of 0.05 means that confounders explaining 5% of
  the outcome's residual variation and 5% of the representer's variation could
  reduce the estimate to zero. RVa answers the same question for statistical
  significance.
- **Bounds assume ρ = 1** and are therefore conservative; the benchmark `ρ̂`
  shows how adversarial a confounder like a measured one actually was.
- **Benchmarks** describe hypothetical confounders as strong as the discovered
  variables. An unmeasured confounder may be weaker or stronger than every
  measured one, so benchmarks can understate or overstate the risk. Each fold
  discovers its own variables, so benchmarks are listed per fold.
- **Measurement error.** Discovered confounders are measured by an extraction
  model. Measurement error in them leaves part of their confounding unadjusted,
  which acts like partial unmeasured confounding; these analyses cover that
  residual bias too.
- None of these analyses addresses positivity violations or post-treatment text
  (see [Limitations](#limitations)).

### References for the sensitivity analyses

1. **VanderWeele TJ, Ding P.** Sensitivity analysis in observational research:
   introducing the E-value. *Annals of Internal Medicine.* 2017;167(4):268–274.
   doi:10.7326/M16-2607. PMID 28693043.
2. **Ding P, VanderWeele TJ.** Sensitivity analysis without assumptions.
   *Epidemiology.* 2016;27(3):368–377.
3. **VanderWeele TJ.** On a square-root transformation of the odds ratio for a
   common outcome. *Epidemiology.* 2017;28(6):e58–e60.
   doi:10.1097/EDE.0000000000000733.
4. **Chinn S.** A simple method for converting an odds ratio to effect size for
   use in meta-analysis. *Statistics in Medicine.* 2000;19(22):3127–3131.
   PMID 11113947.
5. **Mathur MB, Ding P, Riddell CA, VanderWeele TJ.** Website and R package for
   computing E-values. *Epidemiology.* 2018;29(5):e45–e47 (PMC6066405).
6. **Chernozhukov V, Cinelli C, Newey W, Sharma A, Syrgkanis V.** Long story
   short: omitted variable bias in causal machine learning. NBER Working Paper
   30302 (2022); arXiv:2112.13398; *The Review of Economics and Statistics*
   (MIT Press), doi:10.1162/REST.a.1705.
7. **Cinelli C, Hazlett C.** Making sense of sensitivity: extending omitted
   variable bias. *Journal of the Royal Statistical Society Series B.*
   2020;82(1):39–67. doi:10.1111/rssb.12348.
8. **Bach P, Chernozhukov V, Kurz MS, Spindler M.** DoubleML — an
   object-oriented implementation of double machine learning in Python.
   *Journal of Machine Learning Research.* 2022;23(53):1–6. Sensitivity user
   guide: https://docs.doubleml.org/stable/guide/sensitivity.html
9. **Li F, Morgan KL, Zaslavsky AM.** Balancing covariates via propensity score
   weighting. *Journal of the American Statistical Association.*
   2018;113(521):390–400. doi:10.1080/01621459.2016.1260466.

## Output files

Per outer fold, under `stage2/outer_NNN/`:

| Path | Contents |
|---|---|
| `extraction/candidate_missingness_filter.json` | The informativeness filter's decision for every candidate |
| `extraction/documented_unknown_report_fit.json`, `documented_unknown_report_heldout.json` | Documented-unknown counts and fractions per candidate, overall and by treatment arm |
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
| `estimation/nuisance_fit_predictions.csv` | Training out-of-fold nuisances (raw), with eligibility |
| `estimation/propensity_overlap.json` | Training and held-out eligibility audit |
| `estimation/extraction_failure_diagnostics.json` | Extraction-failure rates on training and held-out patients |
| `estimation/sensitivity_benchmarks/benchmarks.json` | Benchmark variables, their importance, and their `cf_y`, `cf_d`, and `ρ̂` gains (reporting only), with fit checkpoints under `fits/` |

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
  flagged rather than treated as "not documented", documented-unknown answers
  are kept as their own state, and both rates are reported by arm, but
  extraction accuracy is not otherwise validated against a reference standard.
- **Nuisance models.** The default super learner can follow thresholds,
  curvature, and interactions that a linear elastic net misses, and the stack
  falls back toward the elastic net when the flexible learners do not improve
  the internal-CV loss. Its library is fixed, its tree learners are noisy in
  small arms (which reduce the internal folds or use the elastic net alone),
  and when both the propensity and outcome models are misspecified, AIPW is
  biased roughly in proportion to the product of their errors, whichever
  learner is used.
- **Pretreatment text.** The analysis assumes that every supplied note was
  written before treatment assignment. Post-treatment text, such as
  documentation of early response or toxicity, can act as a mediator or
  collider and bias both the ATE and the CATEs, and text that names the planned
  treatment undermines overlap. OCI cannot verify the assumption for a single
  text column. [Dated-notes input](configuration.md#notes) mitigates the risk:
  notes dated after the index day, and by default those dated on it, never
  enter the analysis text, and every dropped note is counted. What remains
  unchecked is text without a trustworthy date (a single text column, or a note
  date that does not reflect when its content was written), plan statements
  and copied-forward text that name the intended treatment in notes dated
  before the index date, and a wrong index date.
- **Unmeasured confounding.** The estimates are causal only if, given the
  extracted variables, treatment assignment is unconfounded, every eligible
  patient could have received either treatment (positivity), and the usual
  consistency assumptions hold. Characteristics that the notes do not document,
  or that Stage 1 and Stage 2 fail to surface, are not adjusted for. Variable
  roles are predictive routing decisions, not demonstrated causal structure.
  The [sensitivity analyses](#sensitivity-to-unmeasured-confounding) describe
  how strong unmeasured confounding would have to be to change the conclusion;
  they do not adjust for it.
- **Nuisance calibration.** Training out-of-fold nuisances are used without
  recalibration, so their calibration is that of the learner. Each held-out
  calibrator is fitted on one fold model's validation patients (about a fifth of
  the outer-training patients, fewer within an arm), so it is noisier than a
  pooled calibrator and falls back to the identity map in small arms. Slopes
  before and after held-out calibration are in each fold's diagnostics.
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

The references for the E-values, omitted-variable-bias bounds, and benchmarks
are listed with full identifiers in
[References for the sensitivity analyses](#references-for-the-sensitivity-analyses).

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
- Higgins JPT, Thompson SG. Quantifying heterogeneity in a meta-analysis.
  *Statistics in Medicine* 2002;21(11):1539–1558.
- Li F, Morgan KL, Zaslavsky AM. Balancing covariates via propensity score
  weighting. *Journal of the American Statistical Association* 2018;113(521):390–400.
- Liu Y, Xie J. Cauchy combination test: a powerful test with analytic p-value
  calculation under arbitrary dependency structures. *Journal of the American
  Statistical Association* 2020;115(529):393–402.
- MacKinnon JG, White H. Some heteroskedasticity-consistent covariance matrix
  estimators with improved finite sample properties. *Journal of Econometrics*
  1985;29(3):305–325.
- Meinshausen N, Meier L, Bühlmann P. P-values for high-dimensional regression.
  *Journal of the American Statistical Association* 2009;104(488):1671–1681.
- Nie X, Wager S. Quasi-oracle estimation of heterogeneous treatment effects.
  *Biometrika* 2021;108(2):299–319.
- Semenova V, Chernozhukov V. Debiased machine learning of conditional average
  treatment effects and other causal functions. *Econometrics Journal*
  2021;24(2):264–289.
- van der Laan MJ, Polley EC, Hubbard AE. Super learner. *Statistical
  Applications in Genetics and Molecular Biology* 2007;6(1):Article 25.
- Yadlowsky S, Fleming S, Shah N, Brunskill E, Wager S. Evaluating treatment
  prioritization rules via rank-weighted average treatment effects. *Journal of
  the American Statistical Association* 2025;120(549):38–51.
- EconML: [CausalForestDML](https://www.pywhy.org/EconML/_autosummary/econml.dml.CausalForestDML.html)
  and [RScorer](https://www.pywhy.org/EconML/_autosummary/econml.score.RScorer.html).
