# Glossary

This page defines the terms of art used throughout the OCI documentation. Other
pages link here instead of redefining terms. Notation: `T` is the binary
treatment (1 = treated, 0 = comparator), `Y` is the observed outcome, and `X` is
the vector of patient characteristics measured from pretreatment text.

## Causal inference

### Confounder

A pretreatment characteristic that influences both which treatment a patient
receives and the patient's outcome. Ignoring a confounder biases a comparison of
treated and untreated patients. OCI adjusts for confounders through its nuisance
models. In OCI output, the `confounder` role means that a variable was selected
as an input to at least one nuisance model; it is a modeling decision, not proof
of a causal pathway.

### Effect modifier

A pretreatment characteristic that changes how much a patient benefits from (or
is harmed by) the treatment. For example, a biomarker-positive subgroup might
have a larger treatment effect than a biomarker-negative subgroup. In OCI output,
the `effect_modifier` role means that the variable is an input to the model of
treatment-effect variation. A variable can hold both roles.

### CATE, ITE, and ATE

- The **average treatment effect (ATE)** is the mean difference in outcome if
  everyone were treated versus if no one were treated:
  `E[Y(1) − Y(0)]`, where `Y(1)` and `Y(0)` are potential outcomes under each
  treatment.
- The **conditional average treatment effect (CATE)** is the same difference
  among patients who share the characteristics `x`: `τ(x) = E[Y(1) − Y(0) | X = x]`.
- The **individual treatment effect (ITE)** is `Y(1) − Y(0)` for one patient.
  It is never observed. OCI's per-patient prediction, `estimated_cate`, is an
  estimate of the CATE at that patient's measured characteristics. The
  documentation and some output fields call this prediction an "estimated ITE".

For binary outcomes, all three are reported on the probability (risk
difference) scale.

### Nuisance models

Models that are needed to estimate a treatment effect but are not themselves the
quantity of interest:

| Symbol | Meaning |
|---|---|
| `e(X)` | Propensity: probability of treatment, `P(T = 1 | X)` |
| `μ0(X)`, `μ1(X)` | Arm-specific outcome models: expected outcome under comparator and under treatment, `E[Y | T = t, X]` |
| `m(X)` | Marginal outcome model: expected outcome ignoring treatment, `E[Y | X]` |

OCI fits all Stage 2 nuisance models as cross-validated elastic-net regressions
(logistic for binary targets). Estimates carry a hat, for example `ê(X)`.

### Cross-fitting

Each patient's nuisance prediction comes from a model that was fitted without
that patient. The patients are split into folds; for each fold, models are fitted
on the other folds and used to predict the held-out fold. Cross-fitting prevents
a model from grading its own training data and is required for the validity of
R-loss, AIPW, and DML estimates.

### Outer and inner folds

OCI uses nested cross-validation. The cohort is divided into **outer folds**
(five by default). For each outer fold, the remaining **outer-training**
patients are used for all discovery, extraction refinement, selection, and model
fitting, and the fold's **outer held-out** patients receive the final effect
predictions. Each outer-training set is divided again into **inner folds** (five
by default) for cross-fitting and for comparing candidate models. Every patient
is in exactly one outer held-out set, so the pooled predictions cover the whole
cohort once.

### R-loss

The R-learner loss (Nie and Wager, *Biometrika* 2021) scores a candidate effect
function `τ̂` using cross-fitted nuisance predictions:

$$
\text{R-loss}(\hat\tau) = \frac{1}{n}\sum_{i=1}^{n}
\Big[\big(Y_i - \hat m(X_i)\big) - \big(T_i - \hat e(X_i)\big)\,\hat\tau(X_i)\Big]^2 .
$$

Lower is better. OCI uses validation R-loss in its evidence families and, under
the `balanced_evidence` modifier rule, to compare effect models that have
different internal objectives (causal forest, penalized interaction model, DINA)
on a common scale. For binary outcomes, `τ̂` is a risk difference.

### AIPW

Augmented inverse-probability weighting, a doubly robust ATE estimator. Each
patient receives a score

$$
\psi_i = \hat\mu_1(X_i) - \hat\mu_0(X_i)
+ \frac{T_i\{Y_i - \hat\mu_1(X_i)\}}{\hat e(X_i)}
- \frac{(1 - T_i)\{Y_i - \hat\mu_0(X_i)\}}{1 - \hat e(X_i)} .
$$

The mean of `ψ_i` is the AIPW ATE, and its standard error is the standard
deviation of `ψ_i` divided by `√n`. The estimate remains consistent if either
the propensity model or the outcome models are correct. OCI clips `ê` to
`[c, 1 − c]` in the denominators, where `c` is `stage2.propensity_clip`
(default 0.02). The per-patient score is the `aipw_score` output column.

### DINA

**DI**fference in **NA**tural parameters, the heterogeneous-effect estimand of
Gao and Hastie (Z. Gao and T. Hastie, "Estimating heterogeneous treatment
effects for general responses," *Biometrics* 81(4):ujaf162, 2025,
[doi:10.1093/biomtc/ujaf162](https://doi.org/10.1093/biomtc/ujaf162)). For a
binary outcome the natural parameter is the log odds, so DINA `δ(X)` is the
conditional log odds ratio between treatment arms. OCI uses DINA for binary
outcomes in Stage 1 effect evidence, in Stage 2 effect evidence, and as one of
the three final effect models, then converts `δ(X)` to a risk difference. The
equations are in [Stage 2 selection](stage2_selection.md#dina).

### DML

Double (debiased) machine learning (Chernozhukov et al., *Econometrics Journal*
2018). Outcome and treatment are residualized on covariates with cross-fitted
machine-learning models, and the effect is estimated from the residuals. OCI's
final causal forest is EconML's `CausalForestDML`, which follows this recipe.

### Causal forest X versus W

`CausalForestDML` takes two covariate matrices. **X** holds the effect
modifiers: the forest splits on X, so the CATE is a function of X. **W** holds
adjustment-only covariates that enter the internal nuisance models but are not
used to describe effect variation. In OCI's final forest, X contains the
selected effect modifiers (plus a prognostic score for binary outcomes; see
[Stage 2 selection](stage2_selection.md#final-refit)) and W contains the
selected confounders that are not also modifiers.

### Propensity eligibility and clipping

Two separate overlap controls:

- **Eligibility bounds** (`stage2.min_propensity`, `stage2.max_propensity`,
  both unset by default) exclude patients whose estimated propensity lies
  outside the bounds from effect fitting and effect estimation. Bounds are
  inclusive. Excluded patients are flagged `effect_eligible = false` and the
  estimand becomes the ATE in the propensity-eligible population.
- **Clipping** (`stage2.propensity_clip`) only bounds the AIPW denominators and
  excludes no one.

### Prognostic score

The predicted outcome under the comparator, `μ̂0(X)`. OCI uses it, on the logit
scale for binary outcomes, as an unpenalized adjustment covariate in the
penalized interaction model and as an extra column of the forest's X for binary
outcomes.

## Statistics and models

### BH q-value

A Benjamini–Hochberg adjusted p-value. Among all tests with `q ≤ 0.10`, the
expected fraction of false discoveries is at most 10% when the tests' p-values
are valid. The default modifier rule selects candidates whose combined test has
`q ≤ modifier_selection.q` (default 0.10). In the evidence families, q-values are
computed within each resample and endpoint and used as graded evidence, never
as a hard inclusion gate.

### R-learner projection

A test of whether a candidate `Z` explains variation in the treatment effect on
the absolute scale: ordinary least squares of `Y − m̂` on
`[(T − ê), (T − ê)·Z]` with cross-fitted nuisances, and a Wald test of the
`(T − ê)·Z` coefficients with HC3 standard errors. Patients weigh in proportion
to `(T − ê)²`, so the test emphasizes patients with substantial uncertainty in
their treatment assignment (overlap-weighted). It is the `r_projection`
component of the default modifier tests; see
[Stage 2 selection](stage2_selection.md#component-tests).

### AIPW projection

A test of whether a candidate `Z` predicts the [AIPW](#aipw) score `ψ`:
ordinary least squares of `ψ` on `[1, Z]` and a Wald test of the `Z`
coefficients with HC3 standard errors (a DR-learner projection; Semenova and
Chernozhukov 2021). Every patient weighs equally (population-weighted), so
patients with extreme propensities can dominate. It is the `aipw_projection`
component of the default modifier tests.

### Cauchy combination

A way to merge several p-values for the same hypothesis (Liu and Xie, *JASA*
2020): `T = Σ_k w_k tan{(0.5 − p_k)π}` with weights summing to 1, and combined
`p = 0.5 − arctan(T)/π`. It is exact when the p-values are independent and
remains accurate for small p-values under arbitrary dependence. OCI combines
its three modifier-test components this way; see
[Stage 2 selection](stage2_selection.md#cauchy-combination).

### BLP, GATES, CLAN, and RATE

Diagnostics that ask whether a set of CATE predictions captures real
heterogeneity. They use only held-out AIPW scores and the frozen predictions.

- **BLP** (best linear predictor; Chernozhukov, Demirer, Duflo, and
  Fernández-Val): regress `ψ_i` on `[1, τ̂_i − mean(τ̂)]`. The intercept `β1`
  estimates the ATE; the slope `β2` measures how well the predicted ranking
  tracks true effect variation (`β2 = 0`: no detectable signal; `β2 = 1`:
  well-scaled predictions).
- **GATES** (sorted group average treatment effects): the mean `ψ_i` within
  quintiles of `τ̂`. An increasing pattern supports heterogeneity.
- **CLAN** (classification analysis): the mean of each retained variable in the
  top versus bottom `τ̂` quintile. Descriptive only.
- **RATE** (rank-weighted average treatment effect; Yadlowsky et al.): how much
  better than average the patients ranked highest by `τ̂` do. **AUTOC** weights
  every ranking depth equally and is sensitive to the top of the ranking;
  **Qini** weights depth linearly and has more power for diffuse heterogeneity.

### ATO

The average treatment effect in the overlap population (Li, Morgan, and
Zaslavsky, *JASA* 2018). Patients are weighted by `e(1 − e)`, which emphasizes
patients who could plausibly have received either treatment.

### SMD

Standardized mean difference: the treated-minus-comparator difference in a
variable's mean, divided by the pooled standard deviation. An inverse-propensity
weighted `|SMD|` above 0.1 suggests that the propensity model does not balance
that variable.

### Platt recalibration

A two-parameter logistic regression of a binary label on
`[1, logit(prediction)]`. OCI applies it to its final nuisance predictions,
learned from out-of-fold training predictions only (see
[Stage 2 selection](stage2_selection.md#nuisance-construction)).

### HTR

Hierarchical transformer. A long record is split into overlapping word chunks,
each chunk is encoded by a small pretrained text encoder, and a document-level
transformer combines the chunk encodings. Used by the `htr_neural` and
`matched_pair_uplift` Stage 1 architectures.

### BoW and TF-IDF

Bag of words: a text representation that counts words and short phrases
(n-grams) and ignores their order. TF-IDF (term frequency–inverse document
frequency) weights those counts so that terms frequent in one record but rare in
the cohort count more. OCI's lexical Stage 1 architectures use TF-IDF-weighted
bag-of-words features.

### NMF

Non-negative matrix factorization. Applied to a TF-IDF matrix, it finds groups
of terms that tend to occur together (topics). Used by `tfidf_topics`.

## Pipeline

### Stage 1 and Stage 2

**Stage 1** fits text models that point to parts of the clinical record related
to treatment, outcome, or treatment-effect variation, and freezes that evidence.
**Stage 2** turns the evidence into named, measurable variables, extracts them
from every patient's record with an LLM, selects which variables to adjust for
and which to treat as effect modifiers, and estimates treatment effects. See
[How OCI works](how_it_works.md).

### Architecture

One of the ten Stage 1 evidence generators, such as `bow_nuisance` or
`tfidf_topics`. The documentation uses "architecture" for these; configuration
comments and some code use "lane" for the same thing. See
[Stage 1 architectures](stage1_architectures.md).

### Handoff

The frozen collection of Stage 1 evidence passed to Stage 2,
`handoff/evidence.jsonl`, with an index file. It contains only the selected
architectures' evidence, organized by outer fold.

### Evidence card

A compact, readable summary of related Stage 1 evidence for one outer fold:
up to four representative text excerpts plus provenance. Stage 2 compiles the
handoff into at most 400 cards per outer fold by default, and the interpretation
LLM reads one card per request. A card is neither a patient nor a variable; one
card can suggest several variables.

### Candidate variable and catalog

A variable proposed by the interpretation LLM from evidence cards, such as
"ECOG performance status". After duplicates are merged and sparse candidates are
removed, the remaining candidates form a fold's **catalog**. Every catalog
variable is extracted for every outer-training patient and later for every
outer held-out patient.

### Extraction definition (ontology)

The rule for measuring one candidate variable from text: its meaning, value type,
units or allowed categories, missing-value rule, and how to resolve repeated or
conflicting mentions (for example, latest value, maximum, or most frequent
value). The code and logs call the set of definitions the **ontology**.

### Extraction failure

A patient–variable cell whose LLM response could not be validated after all
repair attempts. The cell is stored as missing but, unlike "not documented", is
flagged as a failure. Variables with training failures get a companion indicator
column named `<variable>__oci_extraction_failed`, which models treat as part of
the parent variable.

### Interpretation LLM and extraction LLM

Stage 2 uses two LLM roles, which may be served by the same model:

- The **interpretation LLM** (`stage2.endpoint`, `stage2.model`) reads evidence
  cards, names and merges candidates, writes extraction definitions, reviews
  aggregate extraction quality, and plans value harmonization. Configuration
  comments sometimes call it the "primary" model.
- The **extraction LLM** (`stage2.extraction_llm`) reads one patient's record at
  a time and returns that patient's values.

### Oracle columns

Columns in a synthetic dataset that record the true data-generating quantities,
such as `true_ite`, `true_ite_prob`, `true_y0`, `true_y1`, and
`true_treatment_prob`. Columns whose names start with `true_`, `oracle_`, or
`ground_truth` are never read during discovery, extraction, selection, or
fitting. They are joined to predictions only after all predictions are frozen,
for evaluation. See [Synthetic data](synthetic_data.md).

### Evidence family

One statistical model type used in Stage 2 to score candidates, such as the
adjusted interaction test or the causal-forest permutation importance. See
[Stage 2 selection](stage2_selection.md#evidence-families).

### Modifier budget

Under the `balanced_evidence` modifier rule, the number of top-ranked
candidates, beyond any investigator-locked modifiers, given the effect-modifier
role. Stage 2 chooses it from a configured list (`0, 4, 8, 12, 16, 24, 32` plus
the full shortlist, capped at 64). The default `orthogonal_tests` rule has no
budget: the number of modifiers is the number of Benjamini–Hochberg
discoveries, capped at `modifier_selection.max_modifiers`.

### Outer-fold confirmation

Re-testing each outer fold's selected modifiers on that fold's held-out
patients, with the same component tests and Cauchy combination as the training
selection and Benjamini–Hochberg within the fold's selected set. Because the
held-out patients took no part in selection, the tests are valid conditional on
it. The confirmation is reported in `stage2/heterogeneity_diagnostics.json`
and never feeds back into selection or estimation; see
[Stage 2 selection](stage2_selection.md#outer-fold-confirmation).

### Fresh run and saved run

A **fresh run** starts from a configuration file in an output directory that
has no completed work. A **saved run** is an existing output directory, whose
resolved settings are stored in `run_config.json`. Rerunning a command resumes a
saved run from its checkpoints; `--stage2-only` relaunches Stage 2 of a saved
run; `--stage2-reselect` recomputes selection and estimation of a saved run
whose Stage 2 has completed. The synthetic launcher applies its presets to full
launches but not to Stage 2-only relaunches of a saved run (`STAGE2_ONLY=1`),
which use the saved settings. See [Configuration](configuration.md).

### Three uses of "multi_model"

The string appears in three unrelated configuration names:

| Name | Meaning |
|---|---|
| `stage2.statistical_selection.multi_model` | Settings for Stage 2 variable selection and final-model choice |
| `architecture.multi_model_forest` (inside the Stage 1 template) | Settings for the Stage 1 text models |
| `multi_model_agentic_forest` | A model type of the standalone `oci run` workflow; see [Optional extensions](optional_extensions.md) |
