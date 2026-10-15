# How OCI works

OCI estimates how the effect of a treatment varies across patients when the
relevant patient characteristics are documented only in clinical text. It
proceeds in two stages:

- **Stage 1** fits several independent text models, each of which points to
  parts of the record associated with treatment choice, outcome, or variation
  in treatment effect. Their evidence is frozen in a
  [handoff](glossary.md#handoff).
- **Stage 2** turns that evidence into named, measurable clinical variables,
  extracts the variables from every patient's record with an LLM, selects
  [confounders](glossary.md#confounder) and
  [effect modifiers](glossary.md#effect-modifier) statistically, and estimates
  per-patient conditional average treatment effects
  ([CATEs](glossary.md#cate-ite-and-ate)) and an average treatment effect
  (ATE).

This page follows one analysis from start to finish. Terms of art are defined in
the [glossary](glossary.md). The statistical details of Stage 2 selection are in
[Stage 2 selection](stage2_selection.md), and each Stage 1 model is described in
[Stage 1 architectures](stage1_architectures.md).

> **Settings this description assumes**
>
> - A binary treatment and a binary outcome. Differences for continuous
>   outcomes are noted where they occur.
> - Five outer folds and five inner folds (`science.outer_folds`,
>   `science.inner_folds`).
> - No Stage 1 architecture selector, so the eight default architectures run.
> - Stage 2 settings at their code defaults, which is what
>   `./run_synthetic.sh` uses: no propensity eligibility bounds, the optional
>   pre-selection measurement consolidation off, matched-batch evidence off,
>   estimand-informed ontology search off, and modifier-concept review off.
>   Effect modifiers are selected by the default `orthogonal_tests` rule, and
>   all three final effect models (causal forest, penalized interaction model,
>   DINA) are fitted and averaged.
> - The example configuration
>   `example_configs/research_all_evidence_multi_model.json` differs in three
>   scientific settings: it sets propensity eligibility bounds of 0.1 to 0.9,
>   enables pre-selection measurement consolidation, and enables matched-batch
>   evidence. See [Optional extensions](optional_extensions.md) and
>   [Configuration](configuration.md).
>
> All patient counts below come from an **illustrative** cohort of 1,000
> patients. They describe how the folds divide a cohort of that size, not the
> results of any particular run.

## Study design and fold structure

Each row of the input dataset is one patient (or one treatment decision) with
four columns: an identifier, the pretreatment clinical text, a binary treatment
indicator, and the outcome. **All supplied text is assumed to have been written
before treatment assignment.** OCI does not check dates to remove
post-treatment text; preparing a pretreatment record is the analyst's
responsibility.

The cohort is split into five outer folds, stratified jointly by treatment and
outcome. Stage 1 and Stage 2 use the same folds. For each outer fold, the
pipeline is run separately:

| Partition (illustrative 1,000-patient cohort) | Approximate size | Purpose |
|---|---:|---|
| Outer training set | 800 | Everything for this fold: Stage 1 evidence, variable discovery, extraction refinement, selection, and model fitting |
| Outer held-out set | 200 | Receives the fold's final effect predictions |
| One inner training partition | 640 | Fits the evidence models and the cross-fitted nuisance models (and, under the alternative `balanced_evidence` rule, a modifier ranking and candidate effect models) |
| Its inner validation partition | 160 | Scores the evidence models and receives out-of-fold nuisance predictions |
| Subfolds of the 640 | 3 × about 213 | Cross-fitting and penalty tuning inside an inner training partition |

Every patient is in exactly one outer held-out set, so the pooled predictions
cover the whole cohort once. Different outer folds can discover different
variables and select different modifiers; this variation is itself informative
about stability.

**What each outer fold may use.** Everything before the final prediction step
uses only the outer-training patients: evidence compilation and variable
discovery, duplicate merging and extraction definitions, value harmonization
and definition revisions, nuisance-input and modifier selection, categorical
encoders, and every fitted model. The held-out patients' text is read only to
measure the frozen variables, and their treatment and outcome labels are read
only after their effect predictions are fixed. Patient-level treatment and
outcome labels are never shown to an LLM. (The Stage 1 evidence that the LLM
reads was itself derived from outer-training labels; see the limitations in
[Stage 2 selection](stage2_selection.md#limitations).) [Oracle columns](glossary.md#oracle-columns) in synthetic
data are used only for evaluation after all predictions are frozen.

## Stage 1: evidence from ten architectures

Stage 1 runs up to ten [architectures](glossary.md#architecture), each a
different way of finding clinically meaningful text. By default eight run (seven
for continuous outcomes); the two optional ones are enabled by name. Each
architecture is fitted within the outer-training data of each fold and writes
evidence such as ranked phrases, topics, attention spans, or retrieved text
excerpts. None of this evidence is a variable yet. The architectures fall into
four families.

**Sparse lexical models** (`bow_nuisance`, `bow_r_loss`). Regularized
[bag-of-words](glossary.md#bow-and-tf-idf) models report which words and short
phrases predict treatment assignment or outcome, and which explain variation in
the treatment effect after the nuisance predictions are accounted for. They are
sensitive to explicit chart language such as diagnoses, performance status, and
laboratory abnormalities.

**Contextual neural models** (`htr_neural`, `matched_pair_uplift`). A
[hierarchical transformer](glossary.md#htr) reads long records as ordered chunks
and learns treatment, outcome, and effect heads; its attention points back to
influential passages. The matched-pair architecture pairs treated and untreated
patients with similar predicted propensity and outcome risk, and asks which text
features explain the difference in their outcomes. It requires a binary
outcome.

**Embedding models** (`embedding_whole_cohort`, `embedding_clustered`,
`neural_query_moments`, and the optional `tfidf_semantic_retrieval_contrasts`).
Record chunks are encoded by a frozen sentence-embedding model. Directions that
separate treated from untreated patients, outcome from no outcome, or high from
low effect scores are computed across the whole cohort and within clusters of
similar patients, and real text excerpts at both ends of each direction are
retrieved. Learned neural queries search the same embedding space for recurring
patterns. The optional semantic-retrieval architecture summarizes the retrieved
excerpts as distinguishing words.

**TF-IDF topic models** (`tfidf_topics` and the optional
`tfidf_orphan_ngrams`). A fold-local TF-IDF screen finds terms associated with
treatment, outcome, or effect variation; [NMF](glossary.md#nmf) groups them into
topics that recur across random seeds. The optional orphan architecture keeps
effect-associated phrases that no topic captured.

For binary outcomes, every effect-variation screen in Stage 1 uses the
[DINA](glossary.md#dina) objective with cross-fitted propensity and arm-specific
outcome predictions; for continuous outcomes the screens use
[R-loss](glossary.md#r-loss) forms. Agreement across architectures suggests that
a concept is not an artifact of one representation; disagreement is preserved
for Stage 2 rather than resolved by a global importance ranking.

The handoff (`handoff/evidence.jsonl`) holds the selected architectures' evidence
for every outer fold. An unselected architecture may still run as a private
prerequisite of a selected one (for example, the lexical nuisance models whose predictions the
embedding architectures need for their effect directions); its evidence then
stays out of the handoff. See [Stage 1 architectures](stage1_architectures.md) for each
architecture's estimand and output, selection rules, and oracle-safe evaluation.

## Stage 2: from evidence to effect estimates

The steps below run once per outer fold. Steps 1 to 6 build and measure the
fold's variables; steps 7 to 9 choose how to use them; steps 10 to 13 produce the
held-out estimates.

```text
handoff ─► 1 evidence cards ─► 2 candidate discovery ─► 3 merge + define
        ─► 4 extract (training) ─► 5 review, refine, harmonize ─► 6 informativeness filter
        ─► 7 statistical evidence ─► 8 nuisance inputs ─► 9 modifier tests + estimator policy
        ─► 10 extract (held-out) ─► 11 final nuisances ─► 12 refit + predict
        ─► 13 pooled ATE, diagnostics, confirmation, oracle comparison
```

### 1. Compile the evidence into cards

The handoff is redundant: the same phrase or excerpt recurs across architectures,
views, and training contexts. A deterministic compiler deduplicates it within
the fold, aggregates provenance and stability, clusters related evidence, and
writes up to 400 [evidence cards](glossary.md#evidence-card) per outer fold,
each with up to four representative excerpts. Every card keeps a lineage record
back to its raw Stage 1 occurrences. The cards are written to
`stage2/evidence_compilation/`.

### 2. Discover candidate variables

The [interpretation LLM](glossary.md#interpretation-llm-and-extraction-llm)
receives one card per request and lists every distinct, measurable,
pretreatment patient attribute its excerpts support. An illustrative excerpt
mentioning "ECOG 2, elevated neutrophil-to-lymphocyte ratio, and liver
metastases" could yield three candidates. Discovery asks what could be measured;
it assigns no causal roles. Because one card can mention several attributes and
one attribute can appear on many cards, the number of candidates is not tied to
the number of cards.

### 3. Merge duplicates and write extraction definitions

Candidates from different cards overlap. The interpretation LLM reviews them in
bounded batches (20 names per batch by default), grouped partly by semantic
similarity and partly alphabetically, over repeated rounds that stop when merges
become rare. The operation is merge-only: names for the same scalar measurement
are combined with their evidence, distinct measurements stay separate (a
neutrophil count and a neutrophil-to-lymphocyte ratio are not merged into one
"inflammation" variable), and no candidate is dropped.

The interpretation LLM then writes one
[extraction definition](glossary.md#extraction-definition-ontology) per merged
candidate: its meaning, value type, units or allowed categories, missing-value
rule, and rule for repeated or conflicting mentions (latest value, maximum,
most frequent value, any positive mention, and so on). These rules define
different measurements, so they are part of the variable. Investigator-specified
variables, if configured, keep their supplied definitions and roles.

### 4. Extract every candidate from the outer-training records

The [extraction LLM](glossary.md#interpretation-llm-and-extraction-llm) reads one
patient's record per request and returns up to 10 variables per request.
Records longer than the context budget are processed in ordered, lossless
chunks, with the current values carried forward from chunk to chunk; a request
that cannot fit fails rather than truncating text. Variables defined by the most
frequently documented value use a separate path that collects countable
observations across the whole record. Python validates every response; invalid
responses receive validator-guided repair requests, and reasoning is turned on
after a configured number of failed repairs. A cell that still fails is stored
as an [extraction failure](glossary.md#extraction-failure), distinct from "not
documented". The result is one row per outer-training patient and one column
per candidate.

### 5. Review definitions and harmonize values

Several checks precede any statistical modeling:

- **Definition refinement.** When the same validation problem recurs across
  patients (by default in at least 3 patients), the interpretation LLM can
  revise the affected definition, for example to clarify categories that keep
  producing incompatible values. Only variables whose definitions changed are
  re-extracted.
- **Aggregate review.** The interpretation LLM reviews value distributions and
  failure summaries for each variable (by default for up to 2 rounds). It can
  revise a definition but cannot add, drop, rename, or assign roles to
  variables, and it sees no treatment or outcome information.
- **Harmonization.** Different written forms of the same value are mapped to one
  canonical value by a plan learned on training patients only. The plan is
  frozen and later applied unchanged to held-out patients.

### 6. Remove uninformative candidates

Using outer-training patients only, a candidate is kept when all of the
following hold:

1. At most 10% of patients have an extraction failure for it
   (`stage2.max_extraction_failure_fraction`).
2. At least 20 patients have an observed value (`stage2.min_informative_patients`).
3. It is not nearly constant: a categorical, binary, or ordinal variable needs
   at least two levels with at least 20 patients each (counting "not documented"
   as a level), and a continuous variable needs at least two distinct values.

With the illustrative 800 outer-training patients, a candidate observed in 19
patients is removed and one observed in 20 is kept. The surviving candidates are
the fold's [catalog](glossary.md#candidate-variable-and-catalog). The full
extraction remains on disk for audit. If enabled, the optional estimand-informed
ontology search and the pre-selection measurement consolidation run at this
point; see [Optional extensions](optional_extensions.md).

### 7. Build statistical evidence about every candidate

This is the first large CPU phase. Within each inner split, cross-fitted
elastic-net nuisance models are fitted (step 8 reads their supports), and the
univariable family tests every candidate's association with treatment and with
the outcome. Under the default `orthogonal_tests` modifier rule nothing else
runs here, because the step 9 tests choose the modifiers and nothing reads
further effect evidence. Under `balanced_evidence`, several more
[evidence families](glossary.md#evidence-family) score every candidate's
relation to treatment-effect variation:

- a conventional treatment-by-candidate interaction test, adjusted for
  cross-fitted propensity and outcome predictions, with
  [BH q-values](glossary.md#bh-q-value);
- grouped elastic-net outcome models with treatment interactions, and a grouped
  DINA model of the conditional log odds ratio;
- one-candidate-at-a-time DINA effect models compared with a constant effect;
- permutation importance in causal forests and in the joint DINA model.

Each candidate's categorical levels and missingness indicators always move
together as one group. The evidence is repeated over three partitions of the
outer-training patients, over row subsamples, and over random subsets of about
32 candidates, so each candidate is assessed in several covariate contexts.
Whole-model gains over a constant effect are recorded alongside per-candidate
importance, because an importance score from a model that does not generalize
is weak evidence. See
[Stage 2 selection](stage2_selection.md#evidence-families-and-resampling).

### 8. Select nuisance-model inputs

Within each inner training partition, grouped elastic nets predict treatment,
the marginal outcome, and the outcome in each treatment arm. A candidate is
routed to a nuisance task if its coefficient group is nonzero in any training
fold for that task, and a candidate routed to either task receives the
`confounder` role. The decision is predictive, not causal, and uses no LLM. These
roles decide which variables enter the final effect models as adjustment
covariates (the forest's W and the interaction model's main effects); the
nuisance predictions themselves always use the complete catalog (step 11).

### 9. Test candidate modifiers and set the estimator policy

The modifier tests use the training half of the step 11 nuisance construction,
computed at this point because it needs no held-out measurements: each of the
800 outer-training patients receives a recalibrated propensity and arm-specific
outcome predictions from the inner-fold model that did not see that patient.
The final estimation later reproduces exactly these training values.

Each candidate, with its categorical levels, "not documented" indicator, and
extraction-failure indicator tested jointly, receives three p-values:

- **R-learner projection** (`r_projection`): does the candidate explain
  variation in the risk-difference effect, weighting patients by how uncertain
  their treatment assignment was?
- **AIPW projection** (`aipw_projection`): does the candidate predict each
  patient's doubly robust effect score, weighting all patients equally?
- **Adjusted interaction test** (`outcome_interaction`): does the candidate
  interact with treatment in a logistic outcome model (on the odds-ratio scale;
  an additive model for continuous outcomes)?

A weighted [Cauchy combination](glossary.md#cauchy-combination) merges the
three p-values (half the weight on the two risk-difference tests and half on
the odds-ratio test for binary outcomes), and Benjamini–Hochberg is applied
across candidates. Candidates with q-values of at most 0.10 become the fold's
effect modifiers, ordered by p-value and capped at 64; investigator-specified
modifiers are always included. Zero modifiers is a valid outcome and does not
force a zero treatment effect. Each selected modifier is labeled by the scales
on which it is supported (`absolute`, `multiplicative`, or `combined_only`).

The estimator policy is the ensemble: in step 12 the causal forest, the
penalized interaction model, and DINA are all refitted and their predictions
averaged. No nested budget search runs, and confounder roles from step 8 are
unchanged. The tests reuse the patients whose evidence built the catalog, so
their p-values are somewhat optimistic; step 13 repeats them on held-out
patients. The alternative `balanced_evidence` rule instead ranks the
candidates and chooses the number of modifiers and one effect model by nested
validation. See
[Stage 2 selection](stage2_selection.md#modifier-selection-by-orthogonal-tests).

### 10. Measure the held-out patients

The extraction LLM measures the outer held-out patients on the complete catalog,
using the frozen definitions and the training harmonization plans. Only their
text is used.

### 11. Build the final nuisance predictions

One nuisance construction is used for every downstream quantity. Within the
outer-training patients, nuisance models on the complete catalog are fitted in
the five inner folds. By default each is a super learner: an elastic net,
histogram gradient boosting, and a random forest, combined with non-negative
weights chosen by internal cross-validation (`stage2.nuisance_learner:
"elastic_net"` uses the elastic net alone). Each training patient gets the
prediction of the model that
did not see it, and each held-out patient gets the average of the five models'
predictions. The propensity and, for binary outcomes, both arm-specific outcome
predictions are then Platt-recalibrated using calibrators learned from training
predictions only. Under `orthogonal_tests` the step 9 tests already fitted
these five models and stored them, so they are reused (after a fingerprint and
file-hash check) to score the held-out patients instead of being refitted; the
training predictions equal those used by the tests. These predictions drive
eligibility, the AIPW scores, the DINA offsets,
the interaction model's adjustment covariates, the forest's prognostic score,
and the diagnostics. See
[Stage 2 selection](stage2_selection.md#nuisance-construction).

### 12. Refit the effect models and predict held-out effects

Each effect model of the ensemble is refitted on the outer-training patients
with the retained confounders and modifiers:

- **Causal forest:** modifiers (and, for binary outcomes, the prognostic score
  `logit μ̂0`) in X, the remaining confounders in W, with the forest fitting its
  own cross-fitted nuisances.
- **Penalized interaction model:** all retained variables as main effects,
  treatment interactions with the selected modifiers, and unpenalized
  `logit ê` and prognostic-score covariates.
- **DINA:** a grouped log-odds-ratio model over the selected modifiers, using
  the step 11 nuisances as offsets.

Each held-out patient receives one `estimated_cate`, the unweighted mean of the
three models' predictions, which are also kept as `estimated_cate_causal_forest`,
`estimated_cate_linear_interactions`, and `estimated_cate_dina` (patients
outside propensity eligibility bounds, when bounds are set, receive none). The
ensemble mean has no confidence interval; the causal forest's own interval is
kept in separate columns.

### 13. Estimate the ATE and run diagnostics

After the effect predictions are frozen, the held-out labels are read and each
held-out patient receives an [AIPW](glossary.md#aipw) score `aipw_score` from
the step 11 nuisances. When all outer folds finish, the predictions are pooled
in `stage2/cross_fitted_predictions.csv` and:

- the mean of the AIPW scores is the ATE (`stage2/causal_estimate.json`);
- heterogeneity, overlap, and balance diagnostics are written to
  `stage2/heterogeneity_diagnostics.json` (see
  [Stage 2 selection](stage2_selection.md#heterogeneity-and-overlap-diagnostics));
- each fold's selected modifiers are re-tested on its held-out patients with
  the step 9 tests, for reporting only (`modifier_confirmation` in the same
  file, summarized in `causal_estimate.json`; see
  [Outer-fold confirmation](stage2_selection.md#outer-fold-confirmation));
- for synthetic data, the frozen predictions are joined to the oracle effects
  and correlation and error metrics are written to
  `stage2/posthoc_oracle_ite_metrics.json`.

The AIPW ATE and the mean of `estimated_cate` are separate estimates, and the
`mu0` and `mu1` columns are the nuisance predictions, not the effect model's
counterfactual predictions. An effect model can predict a constant or zero
effect while the AIPW ATE is nonzero.

## Continuous outcomes

The same steps run with these differences: `matched_pair_uplift` does not run;
DINA is replaced by R-loss forms in Stage 1 screens and is removed from Stage 2
evidence, the estimator ensemble, and the `balanced_evidence` model search; the
interaction model uses an identity link; the outcome interaction test is a
partial F test and the three modifier-test components are weighted equally;
outcome nuisances are not recalibrated; and the forest's X carries no
prognostic score.

## Checkpoints and parallel work

Each component and each fold-level step writes its own checkpoints, so
rerunning the same command resumes an interrupted run. LLM requests from all
outer folds share the configured request concurrency, and CPU model fits share
one pool sized by `stage2.modeling_workers` across the whole run. Dependent
steps, such as ranking before scoring that ranking's prefixes, keep their order.
See [Configuration](configuration.md) for resume, reselection, and parallelism
settings, and [Serving](serving.md) for LLM endpoints.
