# Fold 1 oracle-variable forests: 95% sampling with inference off

Date: September 21, 2026.

## 1. Result

**Increasing the sampling fraction from 45% to 95% did not improve mean correlation.** With honesty on, correlation was essentially unchanged and RMSE improved slightly. With honesty off, both correlation and RMSE worsened at every matched seed.

Inference was **False in every comparison**. Results are means of the per-seed metrics for three fixed seeds on the same 200 held-out patients.

| Honesty | Sample fraction | Correlation | CATE RMSE | Prediction SD | Bias | R-loss |
|---|---:|---:|---:|---:|---:|---:|
| True | 0.45 — reused | 0.66752 | 0.15020 | 0.08221 | −0.03234 | 0.16816 |
| True | 0.95 | 0.66627 | 0.14507 | 0.10632 | −0.03163 | 0.16816 |
| False | 0.45 — reused | 0.67761 | 0.14276 | 0.14545 | −0.02996 | 0.16802 |
| False | 0.95 | 0.59344 | 0.17209 | 0.19068 | −0.02109 | 0.16992 |

1. With honesty on, mean RMSE improved **3.41%**; correlation changed by −0.00125. RMSE improved at all three seeds, while correlation improved at two and declined at one.
2. With honesty off, mean RMSE worsened **20.55%** and correlation fell by 0.08417. Both metrics worsened at all three seeds.
3. The true held-out effect SD is 0.18807. The non-honest 95% forests produced nearly the correct overall spread but a worse patient-level effect pattern. Matching the spread of true effects does not ensure accurate effects for individual patients.

### 1.1. New-run seed ranges

| Honesty at 95% | Correlation range | RMSE range |
|---|---:|---:|
| True | 0.65360–0.67371 | 0.14343–0.14619 |
| False | 0.59154–0.59586 | 0.17053–0.17404 |

These describe forest randomness on one fold, not uncertainty across independent cohorts.

## 2. Controlled setup

1. Same fold 1: 800 training patients and 200 held-out patients, without propensity filtering.
2. X contains the five true modifier variables, encoded into 12 columns. W contains the five true confounders, encoded into 13 columns for nuisance adjustment.
3. Original fitted elastic-net nuisance residuals and observed binary outcomes were reused. No nuisance models were refitted. True effects and outcome/propensity probabilities were not used in fitting.
4. Each new forest copied the corresponding inference-off baseline's parameters and changed **only `max_samples`, from 0.45 to 0.95**.
5. Fixed settings include 200 trees, square-root feature search, minimum leaf size 10, no maximum depth, and one worker.
6. Both honesty settings were evaluated with seeds 120042, 1120042, and 2120042: six new forests. Their six 45% baselines were reused from the preceding inference/honesty grid.

## 3. How the trees used the additional patients

| Honesty | Sample fraction | Patients per tree | Split-selection patients | Effect-estimation patients | Median leaves per tree | Median effect-estimation patients per leaf |
|---|---:|---:|---:|---:|---:|---:|
| True | 0.45 | 360 | 180 | Separate 180 | 9 | 17 |
| True | 0.95 | 760 | 380 | Separate 380 | 17–18 | 17 |
| False | 0.45 | 360 | 360 | Same 360 | 20 | 14 |
| False | 0.95 | 760 | 760 | Same 760 | 42 | 13–14 |

The ranges above reflect variation across the three seeds. All forests used all 800 training patients across their trees, in both roles.

More patients per tree allowed more leaves, while typical leaf sample sizes stayed about the same. This experiment therefore changes both per-tree data coverage and the resulting partition resolution; it is not a test of more patients estimating effects within fixed partitions.

## 4. Interpretation

1. At these fixed settings, larger subsamples did not resolve the correlation plateau and were harmful with honesty off.
2. The result strengthens the case that simply exposing each tree to more of the existing patients is insufficient. It does not establish which combination of splitting behavior, leaf size, and averaging is responsible.
3. The larger trees and deterioration without honesty are consistent with a less favorable complexity/variance tradeoff. That is an interpretation, not an isolated measurement of overfitting.
4. Increasing the total number of independent training patients was not tested: the cohort remained fixed at 800.
5. This is an exploratory comparison on a previously inspected fold. No production configuration was changed using these oracle scores.

## 5. Validation and artifacts

1. All six new fits completed without warnings. Their per-tree sample sizes were verified as exactly 760, with either disjoint 380/380 honest halves or the same 760 patients serving both roles.
2. Original nuisance residual hashes remained unchanged, and only the sample-fraction parameter differed from each matched baseline.
3. Each baseline's saved predictions was reproduced exactly from its saved model without refitting.
4. Predictions were frozen at **2026-09-21 16:48:18 UTC** before this experiment's evaluation phase. The fold's truth had been inspected in earlier experiments, so this is not an untouched validation study.
5. All original source and fit artifacts, previous-grid artifacts, and 19 new frozen artifacts passed hash verification after evaluation.

- [Reproducible fitting/evaluation script](run_sampling_095_2026-09-21.py)
- [Input manifest](input_manifest_2026-09-21.json)
- [Prediction freeze and artifact hashes](predictions_frozen_2026-09-21.json)
- [Metrics for every seed and sampling fraction](metrics_by_seed_2026-09-21.csv)
- [Aggregate evaluation](evaluation_2026-09-21.json)
- [Fit validation](fit_validation_2026-09-21.json)
- [Evaluation validation](evaluation_validation_2026-09-21.json)
- [Independent metric and saved-model checks](independent_validation_2026-09-21.json)
