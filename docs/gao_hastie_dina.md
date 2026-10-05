# Binary Gao–Hastie DINA

Binary modifier discovery uses the Bernoulli DINA objective from [Gao and Hastie, *Biometrics*](https://academic.oup.com/biometrics/article/81/4/ujaf162/8403946). Continuous outcomes retain their R-learner paths. The native fitted effect is a **conditional log odds ratio**; final CATE predictions are **probability differences**.

For cross-fitted propensity `e` and arm-specific outcome probabilities `mu0`, `mu1`, the shared implementation computes:

```
v0 = mu0 * (1 - mu0)
v1 = mu1 * (1 - mu1)
a  = e*v1 / ((1-e)*v0 + e*v1)
nu = (1-a)*logit(mu0) + a*logit(mu1)
eta = nu + (T-a)*delta(X)
loss = softplus(eta) - Y*eta
p0 = expit(nu - a*delta(X))
p1 = expit(nu + (1-a)*delta(X))
CATE = p1 - p0
```

`oci/models/dina.py` contains the likelihood, counterfactual conversion, score, neural loss, and convex group elastic-net solver. The constant log odds ratio is unpenalized. Group lasso either retains or drops the clinical variable; every encoded categorical level and missingness column uses that variable's group ID. Ridge-only fits retain all encoded columns.

## Integration

| Component | Binary behavior |
|---|---|
| Stage 1 sparse text effect fits and lexical importance | Sparse DINA ridge; probability-scale effect predictions. The two former active BoW effect lanes share one DINA fit and contribute one effect feature. |
| Stage 1 staged and joint HTR effect heads | Bernoulli DINA with nuisance predictions fitted and frozen inside the effect-training fold; neural outputs are converted from log odds to risk differences. |
| TF-IDF effect topics and orphan n-grams | DINA score about a training-fitted constant log odds ratio, with Fisher-information centering for score tests. |
| Learned neural query effect bank | The same DINA score and Fisher weights, including nested discovery and final refitting. |
| Whole-cohort embedding effect contrasts | DINA score contrasts, including an optional Fisher-weighted version. Native treatment/outcome cell and cluster contrasts keep their existing definitions. |
| Stage 2 candidate modifier tests | Joint candidate-group DINA fit versus a constant log odds ratio, evaluated with held-out Bernoulli likelihood. |
| Stage 2 joint orthogonal selection | Group elastic-net DINA; CV chooses the penalty, including the configured one-standard-error rule. |
| Stage 2 evidence families | Causal forest remains. An additional `dina` family measures held-out likelihood loss after jointly permuting every column of a clinical feature. |
| Optional estimand ontology effect probes | Ridge DINA with fixed cross-fitted reference nuisances. |
| Stage 2 architecture/count search | `causal_forest`, `linear_interactions`, and `dina`. All compete using held-out risk-difference R-loss on identical eligible patients. |
| Final CATE estimation | DINA exports `dina_log_odds_ratio`, `dina_mu0`, `dina_mu1`, and `estimated_cate = dina_mu1 - dina_mu0`. |

DINA augments forest evidence; it does not replace the forest or the predictive treatment-interaction evidence family. The final S-learner architecture remains available. DINA is omitted from continuous-outcome searches. Unsupported DINA evidence cells are marked not estimable, and a final architecture/count combination must be estimable on every scoring fold to compete.

The default architecture list and `example_configs/research_all_evidence_multi_model.json` include DINA. An existing configuration that explicitly lists only the earlier two architectures continues to use that explicit list; add `"dina"` to `stage2.statistical_selection.multi_model.modifier_count.estimators` to enable it there.

## Boundaries and interpretation

Nuisance models require both treatment arms. TF-IDF stacks now supply cross-fitted arm outcome probabilities alongside the existing propensity and marginal outcome predictions. The sparse-text and HTR adapters use nested text nuisance fits; the HTR auxiliary nuisance view is TF-IDF with unigrams through trigrams and ridge logistic regression, independent of the neural effect head. Stage 2 arm nuisances use the existing grouped modeling infrastructure. Outer evaluation labels never enter the final fit APIs.

Selection now concerns variation in **log odds ratios** for these binary DINA paths. A constant log odds ratio can still imply heterogeneous risk differences. Consequently DINA and a causal forest can support different modifiers without either result being a coding error. The final common R-loss comparison operates on converted risk differences, never on log odds ratios.

The independent AIPW ATE calculation remains in place. Its nuisance probability columns are distinct from DINA's fitted counterfactual columns. DINA CATE confidence intervals are not implemented; interval columns remain missing, while the separate AIPW ATE interval remains available.

Legacy selection reports retain `r_loss` compatibility keys used by downstream selectors. For binary DINA screens these have explicit objective/scale metadata and native `dina_loss` aliases; they denote Bernoulli likelihood loss. Architecture search and probability-scale diagnostics still report actual squared R-loss. Changed checkpoint identities prevent reuse of earlier objective results.

## Validation

`tests/test_dina.py` checks nuisance orthogonality, likelihood derivatives, sparse/dense equivalence, whole-group selection, categorical encoding, held-out score construction, neural gradients, and held-out-label exclusion. The Stage 2 integration tests check the expanded evidence/search grid, final DINA dispatch, probability-scale exports, overlap handling, and checkpoint reuse. TF-IDF tests cover serial/process consistency for both arm-specific nuisance stacks as well as the original nuisance targets.
