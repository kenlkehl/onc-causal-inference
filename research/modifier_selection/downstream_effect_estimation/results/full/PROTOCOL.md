# Frozen protocol: downstream effect estimation

Status: frozen before execution on 2026-09-29.

## Question

Does replacing the true effect-modifier set with modifiers selected by the Gao
sparse Stage 2 procedure explain a loss in patient-level treatment-effect
prediction, and does a binary/log-odds counterfactual estimator reduce the
remaining loss relative to OCI's causal forest?

## Factorial design

- NSCLC mechanisms: saved 1-confounder/1-modifier and
  5-confounder/5-modifier datasets.
- Training sizes: 800, 3,200, and 12,800 observations.
- Modifier input:
  - `true`: every metadata-designated modifier group;
  - `gao_selected`: groups with at least one nonzero Gao sparse coefficient.
- Downstream estimator:
  - `causal_forest`: OCI's current residualized `econml.grf.CausalForest` setup;
  - `gao_counterfactual`: the unpenalized DINA counterfactual model after
    modifier selection.
- Ten fixed Monte Carlo replications.

This produces 2 mechanisms x 3 sample sizes x 2 modifier inputs x 2
estimators x 10 replications.

## Frozen data construction

For each replication, 200 saved patient profiles are held out for evaluation.
Training profiles are sampled with replacement from the other 800 profiles.
The draws are nested: the first 800 observations in the 3,200 and 12,800
conditions equal the smaller conditions. Treatment and binary outcomes are
new independent draws from each sampled profile's saved true propensity and
potential-outcome probabilities. One hundred independent standard-normal
noise candidates are added for modifier selection.

Gao selection receives the true propensity and true arm-specific outcome
probabilities. Both downstream estimators therefore use oracle nuisances. This
is deliberate: the experiment isolates modifier selection and downstream
effect representation from nuisance-estimation error.

Categorical clinical variables are reference coded. Selection is fit at the
coordinate level, exactly as in Note 030; a clinical variable is passed
downstream if any of its coordinates is selected. Both downstream estimators
receive the same selected variable groups in a given data cell.

## Estimators

Let `e=P(A=1|X)`, `m=(1-e) mu0 + e mu1`, `rA=A-e`, and `rY=Y-m`.
The causal forest fits `rY` on `rA` with the supplied modifier coordinates,
using 200 trees, minimum leaf size 10, sample fraction 0.45, all supplied
features at each split, honesty enabled, and inference disabled.

For the Gao-Hastie/DINA model, define

```
V_a = mu_a (1-mu_a)
a   = e V_1 / {e V_1 + (1-e) V_0}
nu  = a logit(mu1) + (1-a) logit(mu0).
```

After support selection, it fits the convex Bernoulli likelihood

```
logit P(Y=1|A,X) = nu(X) + {A-a(X)} [alpha + B(X)' beta]
```

without a selection penalty. The model directly estimates the conditional
log-odds ratio.

## Evaluation

The primary endpoint is the Pearson correlation between true and predicted
risk difference on the same 200 held-out profiles, matching the metric in the
email to Ken. The run also records risk-difference RMSE, prediction R-squared,
and Spearman correlation, plus the analogous four log-odds-ratio metrics.

Each estimator is evaluated directly on its native scale. For the other scale,
the oracle held-out baseline nuisance is fixed:

- forest RD predictions imply `mu0=m-e*tau_RD` and
  `mu1=m+(1-e)*tau_RD`;
- DINA log-OR predictions imply `mu0=expit(nu-a*tau_logOR)` and
  `mu1=expit(nu+(1-a)*tau_logOR)`.

Probabilities used in logit conversion are clipped to `[1e-6, 1-1e-6]`.
If a model predicts an exactly constant effect, its heterogeneity correlation
is recorded as zero rather than the mathematically undefined `NaN`.
Held-out outcomes and treatments are never used for fitting or evaluation.

## Interpretation boundary

This experiment can distinguish loss due to selected modifier support from
loss due to the downstream estimator. It does not measure Stage 1 discovery,
patient-level text extraction, estimated nuisances, or uncertainty coverage.
