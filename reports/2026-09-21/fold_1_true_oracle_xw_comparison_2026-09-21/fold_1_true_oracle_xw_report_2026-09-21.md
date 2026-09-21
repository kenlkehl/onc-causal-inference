# Fold 1: true oracle covariates in three X/W configurations

Date: September 21, 2026. This is an isolated, user-authorized oracle diagnostic.

## 1. Main finding

### 1.1. Performance on all 200 held-out patients

Configuration **(c), confounders in W and only modifiers in X, performed best on this fold**. Results below are the arithmetic mean of the metrics from three forest seeds, not the performance of an ensemble. Lower CATE RMSE and higher Pearson correlation are better. RMSE is on the outcome-probability difference scale.

| Configuration | X | W | Square-root search: RMSE | Square-root search: correlation | All-feature search: RMSE | All-feature search: correlation |
|---|---|---|---:|---:|---:|---:|
| (a) | Confounders + modifiers | Empty | 0.1607 | 0.623 | 0.1562 | 0.609 |
| (b) | Confounders + modifiers | Confounders | 0.1603 | 0.626 | 0.1560 | 0.605 |
| (c) | Modifiers | Confounders | **0.1500** | **0.676** | **0.1505** | **0.625** |

1. Relative to (a), (c) reduced mean RMSE by **6.67%** with square-root search and **3.66%** with all-feature search.
2. For each search setting, (c) had lower RMSE and higher correlation than both (a) and (b) at every matched seed.
3. Adding confounders to W when they were already in X produced only small changes: (a) and (b) were very close.
4. Searching all features improved RMSE for (a) and (b). For (c), it left mean RMSE nearly unchanged and reduced correlation from 0.676 to 0.625. More split candidates did not uniformly help.

### 1.2. Variation across the three forest seeds

| Configuration | Search | RMSE range | Correlation range |
|---|---|---:|---:|
| (a) | Square root | 0.1586–0.1630 | 0.605–0.644 |
| (b) | Square root | 0.1575–0.1635 | 0.612–0.636 |
| (c) | Square root | 0.1461–0.1566 | 0.658–0.687 |
| (a) | All | 0.1541–0.1590 | 0.600–0.621 |
| (b) | All | 0.1535–0.1590 | 0.597–0.620 |
| (c) | All | 0.1475–0.1548 | 0.610–0.640 |

These ranges describe forest randomness on one fixed fold. They are not confidence intervals or independent cohort replications.

## 2. What was fitted

### 2.1. Inputs and patients

1. The original fold 1 split supplied **800 training patients and 200 held-out patients**, with the same five inner cross-fitting splits. No patients were removed using estimated propensity scores.
2. Confounders were the true values of age, sex, ECOG performance status, creatinine clearance, and prior platinum therapy.
3. Modifiers were the true values of histology, EGFR mutation status, baseline neutrophil-to-lymphocyte ratio, brain metastasis status, and baseline hemoglobin.
4. All declared categorical levels were retained, including “Unknown” and “Other.” Continuous variables were standardized using outer-training means and standard deviations. There were no missing oracle covariate values, no imputation, and no extraction or candidate-feature inputs. Continuous oracle values were used as supplied, including any unusual simulated values.
5. Models learned from observed binary treatment and observed binary outcome. True treatment propensities, potential-outcome probabilities, and treatment effects were excluded from fitting and loaded for evaluation only after all 18 prediction sets were frozen.

### 2.2. Model settings held fixed

1. Estimator: native EconML `CausalForestDML`, with binary treatment and binary outcome enabled.
2. Forest: 200 trees, honest estimation, minimum leaf size 10, sample fraction 0.45, no maximum depth, MSE split criterion, subforest size 4, and one worker.
3. Split search: `max_features="sqrt"` or `max_features=1.0`. “All” refers to all encoded columns in that configuration's X, not to the earlier extracted candidate pool.
4. Forest seeds: 120042, 1120042, and 2120042. This produced **3 configurations × 2 search settings × 3 seeds = 18 fits**.
5. Both nuisance models were the repository's elastic-net logistic wrapper. Settings were L1 ratio 0.8, three-fold internal regularization selection, 16 C values from 0.01 to 10,000, maximum 5,000 iterations, tolerance 0.0001, and fixed seed 120042.
6. Each configuration fitted nuisance models once using the same five cross-fitting splits. Cached residuals were reused across its six final forests, so changing forest seed or split search did not change its nuisance estimates.
7. Versions: EconML 0.16.0, scikit-learn 1.6.1, NumPy 2.3.5, pandas 3.0.5.

### 2.3. How X and W actually enter this estimator

The final effect forest splits on X. The nuisance models receive the concatenation of X and W. This behavior was checked in the installed source and agrees with the [CausalForestDML documentation](https://www.pywhy.org/EconML/_autosummary/econml.dml.CausalForestDML.html).

| Configuration | Encoded X columns | Encoded W columns | Nuisance columns | Consequence |
|---|---:|---:|---:|---|
| (a) | 25 | 0 | 25 | Confounders still enter nuisance adjustment through X. |
| (b) | 25 | 13 | 38 | All 13 encoded confounder columns are literally duplicated in the nuisance design. |
| (c) | 12 | 13 | 25 | Nuisances have the same information as (a); the effect forest can split only on modifiers. |

1. The implementation does not deduplicate variables present in both X and W. The requested literal configuration (b) was preserved.
2. Duplication adds no information, but can change elastic-net regularization and the selected penalty. Thus (a) and (b) need not give identical predictions.
3. In this run, (a) versus (b) changed held-out propensity predictions by at most 0.011262 and outcome nuisance predictions by at most 0.013218.
4. In contrast, nuisance predictions for (a) and (c) were identical to numerical precision: maximum held-out differences below 9 × 10⁻¹⁶ and maximum out-of-fold training differences below 3.5 × 10⁻¹⁵. Their performance difference therefore comes from the final forest design, apart from negligible numerical differences.

## 3. Other estimation diagnostics

### 3.1. Bias, variation, interval coverage, and residual loss

The true held-out mean effect was **−0.060739**, with standard deviation **0.188074**.

| Configuration | Search | Mean prediction minus mean truth | Prediction SD | Coverage of full patient-level truth by nominal 95% intervals | Fitted-nuisance R-loss | Common oracle R-loss |
|---|---|---:|---:|---:|---:|---:|
| (a) | Square root | −0.030142 | 0.060706 | 60.2% | 0.169125 | 0.169037 |
| (b) | Square root | −0.028905 | 0.060594 | 60.2% | 0.169237 | 0.168942 |
| (c) | Square root | −0.030531 | 0.079133 | 66.3% | 0.168449 | 0.168489 |
| (a) | All | −0.028897 | 0.078811 | 62.0% | 0.169778 | 0.169574 |
| (b) | All | −0.027452 | 0.080979 | 61.8% | 0.169806 | 0.169413 |
| (c) | All | −0.030854 | 0.109251 | 70.7% | 0.168969 | 0.168881 |

1. Fitted-nuisance R-loss uses each configuration's estimated propensity and marginal outcome. Because (b)'s nuisance predictions differ, this score is not a completely common scoring rule across configurations.
2. Common oracle R-loss uses the same true propensity and true marginal outcome for every model, strictly during evaluation:

   `mean((Y − true_marginal_outcome − estimated_CATE × (T − true_propensity))²)`.

3. These losses evaluate predictions against noisy observed outcomes. RMSE and correlation directly compare predictions with the simulated probability-scale effect.
4. Interval coverage is descriptive coverage of `true_ite_prob` on these 200 patients, not a repeated-sampling coverage study. In particular, configuration (c) conditions its final predictions only on modifiers, so the full patient-level truth is finer than its conditioning set.

### 3.2. Nuisance accuracy

| Configuration | Held-out propensity RMSE | Held-out marginal outcome RMSE |
|---|---:|---:|
| (a) | 0.064835 | 0.073509 |
| (b) | 0.065630 | 0.073488 |
| (c) | 0.064835 | 0.073509 |

True covariates do not make fitted nuisance functions exact. This experiment tests forests with true inputs and estimated nuisances, rather than forests supplied with exact propensity and outcome functions.

## 4. Interpretation

### 4.1. What this fold supports

1. Restricting forest splits to the five named modifiers improved prediction here, even when all measurements were exact and both nuisance models retained all ten variables.
2. The controlled (a)-versus-(c) comparison points to the final forest's handling of its inputs, rather than different nuisance fits, as the source of that improvement. A finite-sample tradeoff between additional conditioning information and the difficulty of finding useful splits is a plausible explanation; this experiment does not isolate that mechanism further.
3. Configuration (b) supplies no additional adjustment information beyond (a). Its small changes are consistent with the effect of duplicated columns on regularized nuisance fitting.

### 4.2. Why this does not settle the best configuration generally

1. The simulated probability-scale treatment effect is

   `tau(C, M) = sigmoid(g(C) + h(M)) − sigmoid(g(C))`.

   Consequently, confounders affect the individual probability difference through baseline risk, even though the five named modifiers define the treatment interaction on the log-odds scale. This equation was independently verified against the dataset in the [earlier correlation diagnosis](../fold_1_oracle_modifier_candidates_2026-09-21/oracle_modifier_correlation_diagnosis_2026-09-21.md).
2. Configurations (a) and (b) therefore have inputs needed to represent the full true probability-scale effect. Configuration (c) cannot distinguish patients with identical modifiers but different confounders. Its empirical advantage here does not establish that excluding confounders from X is universally preferable.
3. Even (c) retained substantial error: RMSE approximately 0.150, correlation approximately 0.676, and predicted heterogeneity smaller than the true heterogeneity. Observed outcomes remain noisy, nuisance functions remain estimated, and these forest settings are unchanged. This run does not assign a numerical share of the remaining error to each source.
4. Low prediction spread alone does not explain imperfect correlation: positive affine rescaling leaves Pearson correlation unchanged. The all-feature results illustrate this distinction—greater prediction spread in (c) did not improve its correlation.
5. These are exploratory comparisons on a single outer fold. No model selection or production setting was changed in response to oracle performance.

## 5. Reference results on the earlier 180-patient evaluation subset

These use the same newly fitted models, still trained on all 800 patients, evaluated on the 180 held-out IDs retained by the earlier extraction-based comparison.

| Configuration | Square-root RMSE | Square-root correlation | All-feature RMSE | All-feature correlation |
|---|---:|---:|---:|---:|
| (a) | 0.1673 | 0.622 | 0.1624 | 0.606 |
| (b) | 0.1670 | 0.623 | 0.1622 | 0.602 |
| (c) | 0.1568 | 0.676 | 0.1569 | 0.621 |

For context, the earlier oracle-matched **extracted candidate** forest with square-root search had RMSE 0.1770 and correlation 0.461 on those 180 patients. That comparison is not a pure measurement-error ablation: it used 720 training patients, extracted candidate encodings, and the original grouped elastic-net nuisance fits. Here both the covariates and nuisance fits were rebuilt, with 800 training patients. The three configurations within this report are the controlled comparison.

## 6. Validation and reproducibility

1. All 18 fits completed. All 30 fitted nuisance clones completed without warnings or reaching the 5,000-iteration limit.
2. Cross-fitting covered each of the 800 training patients exactly once as an inner validation patient. Training and outer-held-out IDs were disjoint.
3. Every prediction file contains the same ordered 200 held-out IDs. All predictions and interval bounds are finite and ordered.
4. Nuisance estimates remained unchanged across each configuration's six forest fits. Saved forest parameters were checked against the intended seed and feature-search setting.
5. Manually calculated R-loss matched native `CausalForestDML.score` within 10⁻¹² for every fit.
6. All 18 prediction sets were frozen at **2026-09-21 15:13:51 UTC**, before the evaluation phase read true treatment effects and probabilities. All 63 frozen fit artifacts and source hashes passed verification after evaluation.
7. No production code, extraction artifact, or parent experiment output was changed. The fitting script refuses to overwrite an already frozen run.

Artifacts:

- [Reproducible fitting and evaluation script](run_true_oracle_xw_comparison.py)
- [Input manifest, exact splits, settings, and source hashes](input_manifest_2026-09-21.json)
- [Frozen fit artifact hashes](predictions_frozen_2026-09-21.json)
- [Metrics for every seed and evaluation population](metrics_by_seed_2026-09-21.csv)
- [Predictions joined to oracle values after freezing](predictions_with_oracle_2026-09-21.csv)
- [Aggregate evaluation and paired prediction differences](evaluation_2026-09-21.json)
- [Validation results](validation_2026-09-21.json)

The `fits/` directory also contains fitted model artifacts, nuisance audits, training nuisance predictions, and each forest's held-out predictions and settings.
