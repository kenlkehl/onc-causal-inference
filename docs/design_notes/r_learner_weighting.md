# Design note: weighted R-loss versus unweighted pseudo-target regression

This note explains a trade-off that shapes how OCI screens text for
treatment-effect heterogeneity in Stage 1 and why final effect estimation is
left to Stage 2. Notation follows the [glossary](../glossary.md): `e(X)` is the
propensity, `m(X)` the marginal outcome model, and `τ(X)` the conditional
treatment effect.

## The two objectives

The R-learner starts from the residualized treatment-effect equation

```text
Y − m(X) = τ(X) · (T − e(X)) + noise
```

and fits `τ(X)` by minimizing the [R-loss](../glossary.md#r-loss)

```text
Σ_i [ (Y_i − m̂_i) − τ(X_i) · (T_i − ê_i) ]²
```

This objective can be rewritten as a weighted regression on a pseudo-target.
Define

```text
pseudo_τ_i = (Y_i − m̂_i) / (T_i − ê_i)
weight_i   = (T_i − ê_i)²
```

Then the same objective is

```text
Σ_i weight_i · [ pseudo_τ_i − τ(X_i) ]²
```

So the treatment-residual weight is not an arbitrary heuristic: it is the
algebraically equivalent form of the residualized causal loss.

## Why the weighted form is principled

The weighted R-loss respects the amount of causal information in each
observation. If `T_i` is close to `ê_i`, the residualized treatment is small, the
pseudo-target divides by that small number, and the label becomes very noisy.
The weight down-weights exactly those unstable labels. Rows with larger absolute
treatment residuals carry more information about the treatment-effect slope and
get more influence.

The weighting also preserves the R-learner's orthogonal structure. The loss is
built from the residualized outcome and the residualized treatment, which gives
the estimator some robustness to imperfect nuisance models: first-order nuisance
errors tend to cancel rather than bias `τ` directly. Unweighted pseudo-target
regression changes the objective. It does not fit the residualized equation;
it fits the noisy ratio labels as if every ratio were equally reliable.

## Why it can be conservative

The clean theory can be conservative in small, binary, stochastic settings.
Binary treatments and outcomes create large irreducible noise. The pseudo-target
can contain a visible subgroup contrast while the weighted, regularized learner
still prefers a nearly constant `τ`, because the subgroup contrast explains only
a small fraction of total residualized outcome variation. As an illustration,
in a synthetic cohort where patients with a high biomarker value benefit more,
the high-biomarker group can have a clearly higher mean pseudo-target while a
weighted sparse R-learner still compresses its predictions toward a flat
treatment effect.

That does not mean the R-loss is wrong. It optimizes a different, more
conservative criterion: fit the residualized causal equation while discounting
observations whose treatment residual makes their pseudo-label unstable. When
the subgroup signal is weak relative to stochastic treatment and outcome
variation, this weighting makes heterogeneity harder to recover.

Unweighted pseudo-target regression makes the opposite trade-off. It can be more
sensitive to visible subgroup differences, because every pseudo-label gets equal
weight. That is useful diagnostically: if unweighted regression finds a known
modifier but the weighted R-loss does not, the representation probably contains
the modifier signal and the bottleneck is the objective's variance trade-off.
But unweighted regression also gives too much authority to high-variance ratio
labels, especially when `|T − ê|` is small, and it loses some of the formal
orthogonality that motivates the R-learner.

A sensitivity sweep can locate the problem:

1. weighted R-loss: `weight = (T − ê)²`
2. floored weights: `weight = max((T − ê)², floor)`
3. square-root weights: `weight = |T − ê|`
4. unweighted pseudo-target regression
5. oracle or group-balanced weighting, for diagnostics only

If only the unweighted or floored variants recover a known modifier, the
modifier signal exists but is too weak for the strict weighted objective. That is
a scientific finding, not only a modeling failure.

## Discovery versus final estimation

The case for the weighted R-loss depends on how its output is used. If the
fitted `τ(X)` is the final CATE estimator, the weighted objective has the
stronger argument: the goal is to estimate treatment effects under the
residualized causal moment, and preserving the R-loss, down-weighting unstable
labels, and retaining orthogonality are central.

If the fitted model is used only to discover candidate variables, the question
changes to: which pretreatment text features might relate to treatment-effect
heterogeneity? The model then acts as a screening instrument, not the final
estimator, and high-recall discovery may matter more than low-variance effect
prediction. An unweighted pseudo-target that surfaces a known modifier the
weighted R-loss suppresses can be the better discovery tool even though it is
the less principled estimator.

Noisy ratio labels are a liability in both roles, but in discovery the
downstream process can filter candidates: Stage 2 names explicit variables,
extracts them for every patient, and tests them again with honest,
cross-fitted models. The screening model only has to surface plausible
variables. The risk is false positives: unweighted pseudo-targets can elevate
template artifacts, names, documentation style, or variables associated with
nuisance-model errors. Saving each screen's evidence, the variables proposed
from it, their extraction coverage, and the final model diagnostics lets a
researcher separate clinically meaningful candidates from textual artifacts.

In short:

- final CATE estimation: the weighted R-loss is the cleaner target;
- feature discovery: unweighted or softened pseudo-target regression can be
  defensible, because sensitivity may matter more than formal optimality.

## How OCI applies this

**Stage 1 screens use both forms where the outcome is continuous.**

- `bow_r_loss` fits each bag-of-words view twice: an unweighted pseudo-target
  regression and a weighted R-loss regression.
- `htr_neural` trains two effect heads: one on pseudo-outcome squared error and
  one on the squared R-loss.
- `embedding_whole_cohort` builds a direction from the upper and lower tails of
  the pseudo-target, weighted by `(T − ê)²` by default
  (`embedding_contrast.pseudo_target_weighted`), and a second direction from the
  unweighted orthogonal score `(Y − m̂)(T − ê)`.

**For binary outcomes, Stage 1 screens use DINA instead of either form.** The
[DINA](../glossary.md#dina) objective models the conditional log odds ratio with
Bernoulli variance weighting built into its likelihood. Where a screen uses a
per-patient score, such as the embedding effect directions and the TF-IDF and
neural-query effect screens, it uses the DINA score with Fisher-information
weights, and the embedding architectures also add an unweighted version of the
score.

**Final estimation does not use a Stage 1 model.** Stage 2 estimates CATEs from
explicitly extracted variables. It compares a causal forest, a penalized
interaction outcome model, and a DINA model by the weighted validation R-loss on
cross-fitted residuals, and then refits the chosen model; see
[Stage 2 selection](../stage2_selection.md). The screening objectives above
therefore affect which variables are proposed, while the principled weighted
criterion decides how they are used.
