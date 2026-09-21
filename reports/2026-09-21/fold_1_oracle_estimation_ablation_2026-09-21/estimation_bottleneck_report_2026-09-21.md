# Fold 1: why the oracle-covariate causal forest remains near 0.67 correlation

Date: September 21, 2026. Exploratory estimation diagnostics using the existing 800/200 outer split.

## 1. Main finding

**The observed dataset supports much better effect estimation when the estimator is given the correct functional structure.** A logistic treatment-interaction model, with coefficients estimated from the same 800 observed binary outcomes, achieved held-out CATE correlation **0.977** with elastic-net regularization and **0.981** without regularization. Their RMSEs were **0.0414** and **0.0445**, respectively.

For comparison, the best earlier forest configuration averaged correlation **0.676** and RMSE **0.1500**. Supplying that forest with exact nuisance functions increased correlation to **0.717**; additionally removing binary outcome noise increased it to **0.784**.

These findings make the forest's estimation strategy and current settings the central issue to investigate. They do not establish a universal limitation of causal forests. The logistic benchmark receives favorable knowledge of the DGP's functional structure, and this remains one previously inspected fold.

## 2. Controlled forest comparisons

### 2.1. What changed

Let `e` be the true treatment probability, `mu0` and `mu1` the true potential-outcome probabilities, and `m = (1−e)mu0 + e mu1` the true marginal outcome probability.

1. **Original baseline:** observed binary Y and T, with fitted elastic-net nuisance models. The existing frozen forests were reused.
2. **Exact nuisance functions:** kept the same observed Y and T; fitted the final forest to residuals `Y−m` and `T−e`. Only training-row oracle probabilities were used in fitting.
3. **Exact nuisances and noiseless factual outcomes:** kept each patient's actual treatment assignment, replaced the binary outcome by its true factual probability `muT`, and fitted to `muT−m` and `T−e`.
   - This intentionally supplies unattainable oracle information as a diagnostic control.
   - The identity `muT−m = (mu1−mu0)(T−e)` was verified numerically.
   - Treatment assignment, the finite covariate sample, and forest settings remained unchanged. Thus this removes Bernoulli outcome noise, not all sources of estimation error.

All forests used the earlier exact covariate encodings, 200 trees, honesty, sample fraction 0.45, minimum leaf size 10, and seeds 120042, 1120042, and 2120042. Both square-root and all-feature split search were retained. Configuration (a) has all ten variables in X; configuration (c) has modifiers in X and confounders in nuisance adjustment. Configuration (b), which duplicates confounders in the nuisance design, was not repeated.

### 2.2. Held-out performance

Values are arithmetic means of the three per-seed metrics, evaluated against the same 200 full patient-level true probability differences.

| Residual/outcome condition | Forest X | Split search | Correlation | CATE RMSE |
|---|---|---|---:|---:|
| Fitted nuisances, observed outcome | All variables | Square root | 0.6228 | 0.1607 |
| Exact nuisances, observed outcome | All variables | Square root | 0.6174 | 0.1599 |
| Exact nuisances, noiseless outcome | All variables | Square root | 0.7650 | 0.1445 |
| Fitted nuisances, observed outcome | Modifiers | Square root | 0.6758 | 0.1500 |
| Exact nuisances, observed outcome | Modifiers | Square root | 0.7171 | 0.1477 |
| Exact nuisances, noiseless outcome | Modifiers | Square root | 0.7845 | 0.1353 |
| Fitted nuisances, observed outcome | All variables | All | 0.6089 | 0.1562 |
| Exact nuisances, observed outcome | All variables | All | 0.5932 | 0.1559 |
| Exact nuisances, noiseless outcome | All variables | All | 0.7635 | 0.1255 |
| Fitted nuisances, observed outcome | Modifiers | All | 0.6247 | 0.1505 |
| Exact nuisances, observed outcome | Modifiers | All | 0.6427 | 0.1470 |
| Exact nuisances, noiseless outcome | Modifiers | All | 0.7690 | 0.1232 |

1. Exact nuisances helped the modifier-only forest, but did not rescue the forest with all variables in X. Nuisance error is therefore not a sufficient explanation for the overall performance gap.
2. Removing binary outcome noise improved correlation and RMSE at every matched forest seed and search setting relative to the exact-nuisance, observed-outcome condition.
3. Even the noiseless diagnostic with **all ten true variables in X** remained near correlation 0.76. This leaves a substantial limitation in the current finite-sample forest fitting procedure after both missing inputs and nuisance/outcome noise have been addressed.
4. These differences are conditional ablations, not additive percentages of error attributable to independent causes.

## 3. A model with the correct functional structure

### 3.1. Model definition

The logistic model's linear predictor contained:

1. Confounder main effects: age, sex, ECOG, creatinine clearance, and prior platinum therapy.
2. The baseline age × creatinine-clearance interaction present in the DGP.
3. A treatment main effect.
4. Treatment interactions with histology, EGFR status, NLR, brain metastasis status, and hemoglobin.

The design contained 27 encoded columns, plus the fitted intercept. Its continuous variables used the same outer-training standardization as the forests. Full categorical levels were retained. The known DGP link and interaction structure were supplied, but **no true coefficients, potential-outcome probabilities, or ITE labels entered these logistic fits**. Coefficients were estimated using the observed training outcomes. CATE predictions were `predicted P(Y=1|T=1,C,M) − predicted P(Y=1|T=0,C,M)`.

Two fitting procedures were fixed before evaluating their predictions:

- The repository's elastic-net logistic wrapper with the original nuisance settings: L1 ratio 0.8, three-fold internal CV, and the same 16-value C grid. It selected C = 1.0.
- Unpenalized logistic regression using L-BFGS, tolerance 10⁻⁸, and maximum 5,000 iterations. It converged in 61 iterations.

### 3.2. Results

| Model | Correlation | CATE RMSE | Mean prediction minus mean truth | Prediction SD |
|---|---:|---:|---:|---:|
| Logistic interactions, elastic net | 0.9770 | 0.0414 | +0.0047 | 0.1926 |
| Logistic interactions, unpenalized | 0.9814 | 0.0445 | +0.0057 | 0.2099 |

The true held-out effect SD was 0.1881. The regularized model had slightly lower correlation but better RMSE and effect-scale calibration than the unpenalized model.

The models share a small set of coefficients across all 800 training patients. That structural assumption matches the synthetic generator. Their success shows that the observed data contain recoverable treatment-effect information that the current forest is using less efficiently. It does not establish that this functional form will be correct in other datasets or after extraction.

## 4. What the existing forests reveal

These checks inspected the 18 previously frozen forests without refitting them. Detailed values are saved in the [structure and overlap audit](../fold_1_true_oracle_xw_comparison_2026-09-21/existing_forest_structure_diagnostics_2026-09-21.json).

### 4.1. Local sample sizes and resolution

1. Every tree used 360 training patients: 180 for split selection and 180 for leaf estimation. This was verified from regenerated subsample and honesty indices, consistent with the [EconML honesty and sampling definitions](https://www.pywhy.org/EconML/_autosummary/econml.dml.CausalForestDML.html).
2. In the best earlier modifier-only, square-root configuration, the median tree had 9 leaves; the median leaf had 17 patients contributing to effect estimation. Approximately 19% of leaves had fewer than five patients in one treatment arm.
3. With all variables in X and square-root search, the median tree had 11 leaves and the median effect-estimation leaf had 15 patients. Approximately 39% of leaves had fewer than five patients in one arm.
4. The full forest combines information across trees and all 800 patients. These per-tree counts are not the ensemble's effective sample size, but illustrate the local precision/resolution tradeoff.

### 4.2. Training effects and forest randomness

1. The modifier-only, square-root forest's mean apparent training correlation with true effects was 0.7165, compared with held-out correlation 0.6758. These are correlations with true effects, not fit to the noisy observed outcome.
2. Averaging its three already fitted seed predictions increased held-out correlation to 0.6906 and reduced RMSE to 0.1489.
3. Forest randomness accounts for some error, but simple seed averaging did not approach the logistic benchmark. No claim is made about the performance of a separately fitted larger forest.

### 4.3. Overlap and outcome noise

1. The actual dataset's `generation_config.json` and metadata both specify **`enforce_positivity: false`**. The configured bounds 0.1 and 0.9 were inactive.
2. Among 800 training patients, 101 had true propensity below 0.1, including only 5 treated patients; 122 had propensity above 0.9, including only 4 controls. Thus 223 patients were in these extreme-propensity groups.
3. Perfect covariates do not create the unobserved treatment comparisons in those regions. This is a real challenge for flexible local estimation, although the parametric model can use its global structural assumptions to estimate effects there.
4. Weak overlap is not established as the sole explanation: restricting the *evaluation* of the unchanged best forest to the 151 held-out patients with true propensity in [0.1, 0.9] yielded mean correlation 0.6784. This is a descriptive subgroup check, not a new fitted model or a causal attribution of the error.
5. On the training covariates, the mean conditional Bernoulli outcome variance was 0.1592. The mean squared oracle residual treatment component, `E[e(1−e)tau²]`, was 0.00669. This describes a noisy effect-learning signal; it is not a correlation ceiling or a formal variance decomposition of estimator error.

## 5. Implications for the estimation work

1. **Prioritize the final estimator and its structural assumptions.** The current forest has difficulty recovering this smooth logistic interaction signal, even in diagnostic conditions with exact nuisances and noiseless factual probabilities.
2. **Retain a structured treatment-interaction model as a benchmark.** It demonstrates much stronger recovery from the same observed outcomes when the functional structure is known. A practical version needs assessment without assuming the exact synthetic interaction structure and across other folds/DGPs.
3. **Study forest resolution and sample use directly.** A fixed comparison of leaf size, honesty, and sample fraction would test whether forest tuning closes the gap. The current results do not identify which setting is responsible, and smaller leaves may also increase noise.
4. **Use a sample-size curve to distinguish data demand from persistent approximation.** More simulated training patients under the same DGP would test whether the forest catches up. No such curve was run in this diagnostic.
5. The earlier confounder-placement question remains relevant to the estimand, but does not explain the much larger contrast between these estimators.

## 6. Validation and limitations

1. All 12 original baseline forests used in this comparison were refitted directly from their cached residuals and saved parameters. Every held-out prediction was reproduced exactly: maximum absolute error 0.0. This verifies that the ablations use the same final forest fitting path.
2. Twenty-four new forests and two logistic models completed without warnings. Both logistic procedures converged without reaching their iteration limits.
3. All 26 diagnostic prediction sets were frozen at **2026-09-21 15:43:29 UTC**, before running their evaluation phase. Fifty-three new fit artifacts were hashed; the original frozen artifacts were also verified unchanged.
4. Forest parameters were copied from the baseline. Neither the new forest settings nor logistic specifications were selected using their held-out oracle scores.
5. This fold's truth had already been examined in prior experiments. These are explicitly exploratory diagnostics, not an untouched confirmatory evaluation.
6. The true-nuisance and noiseless-outcome controls intentionally use training-row oracle probabilities. The logistic benchmarks intentionally use oracle knowledge of model structure. These advantages must remain visible in any comparison to the production workflow.
7. No production source, extraction result, or original experiment output was modified.

Artifacts:

- [Reproducible fitting/evaluation script](run_estimation_ablation_2026-09-21.py)
- [Input manifest](input_manifest_2026-09-21.json)
- [Prediction freeze and artifact hashes](predictions_frozen_2026-09-21.json)
- [Per-seed metrics for all 38 new and baseline fits](metrics_by_seed_2026-09-21.csv)
- [Aggregate metrics](summary_metrics_2026-09-21.csv)
- [Exact baseline replay and convergence checks](fit_validation_2026-09-21.json)
- [Evaluation validation](evaluation_validation_2026-09-21.json)
- [Independent metric recalculation and model reload checks](independent_validation_2026-09-21.json)
