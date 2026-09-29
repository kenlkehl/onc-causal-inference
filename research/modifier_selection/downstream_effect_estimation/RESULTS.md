# Downstream effect-estimation results

The frozen run completed 10 replications per cell. The primary entries below
are mean Pearson correlation and mean RMSE for the patient-level risk
difference on 200 held-out NSCLC profiles. `M/N` is the mean number of true
modifier groups and other groups in the supplied support. Oracle nuisances are
used throughout.

| Mechanism | n | Modifier input | M/N | Causal forest correlation / RMSE | Gao counterfactual correlation / RMSE |
|---|---:|---|---:|---:|---:|
| 1+1 | 800 | True | 1.0 / 0.0 | 0.881 / 0.093 | **0.987 / 0.046** |
| 1+1 | 800 | Gao selected | 0.0 / 0.0 | 0.000 / 0.170 | **0.059 / 0.169** |
| 1+1 | 3,200 | True | 1.0 / 0.0 | 0.885 / 0.084 | **0.994 / 0.026** |
| 1+1 | 3,200 | Gao selected | 0.1 / 0.1 | 0.065 / 0.164 | **0.134 / 0.158** |
| 1+1 | 12,800 | True | 1.0 / 0.0 | 0.889 / 0.080 | **0.999 / 0.012** |
| 1+1 | 12,800 | Gao selected | 0.1 / 0.1 | 0.077 / 0.160 | **0.139 / 0.152** |
| 5+5 | 800 | True | 5.0 / 0.0 | 0.556 / 0.155 | **0.823 / 0.124** |
| 5+5 | 800 | Gao selected | 0.7 / 0.4 | 0.186 / 0.181 | **0.451 / 0.174** |
| 5+5 | 3,200 | True | 5.0 / 0.0 | 0.675 / 0.132 | **0.955 / 0.057** |
| 5+5 | 3,200 | Gao selected | 3.0 / 0.4 | 0.602 / 0.150 | **0.876 / 0.091** |
| 5+5 | 12,800 | True | 5.0 / 0.0 | 0.749 / 0.119 | **0.987 / 0.030** |
| 5+5 | 12,800 | Gao selected | 4.5 / 0.2 | 0.726 / 0.126 | **0.972 / 0.041** |

## Interpretation

The result supports the proposed downstream claim under the oracle-nuisance
experiment. With the true modifier set, Gao's counterfactual estimator moves
close to correlation 1 as sample size increases: 0.999 in 1+1 and 0.987 in
5+5 at `n=12,800`. The causal forest reaches 0.889 and 0.749, respectively.
The corresponding RMSE reductions are 84% and 75%.

The two mechanisms reveal different selection behavior. In 5+5, Gao selection
improves from 0.7/5 true modifier groups at `n=800` to 4.5/5 at `n=12,800`.
Its downstream correlation consequently rises from 0.451 to 0.972. At the
largest size, most of the remaining gap to the true-support Gao result is gone.

In 1+1, the designated PD-L1 group is selected in only one of ten replications
at both `n=3,200` and `n=12,800`, and never at `n=800`. This is a selection
failure, not a downstream fitting failure: when PD-L1 is supplied, Gao reaches
0.987 even at `n=800`. A constant log-odds ratio can still imply varying risk
differences across baseline risks, so the Gao model can have a small positive
RD correlation even when its selected modifier support is empty.

The causal-forest benchmark is the oracle-residualized `econml.grf.CausalForest`
used by the current Stage 2 code, with its fixed 200-tree settings. It is not a
rerun of Ken's full `CausalForestDML` head with estimated nuisances and tuning.
The modifier input excludes confounders as splitting variables. Gao can still
translate a log-odds effect to the probability scale through the oracle
baseline nuisance; this is precisely the proposed advantage for a log-odds DGP,
but it makes these results an estimand-aware comparison rather than a claim
that every logistic learner dominates every forest configuration.

The frozen machine-readable results are in `results/full/`. `audit.json`
independently reconstructed all 240 replication rows from the held-out
predictions with maximum numerical error `7.2e-15`.
