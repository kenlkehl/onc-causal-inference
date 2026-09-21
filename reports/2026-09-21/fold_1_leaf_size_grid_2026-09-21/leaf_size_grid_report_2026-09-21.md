# Fold 1 oracle-variable forests: minimum leaf size at 95% sampling

Date: September 21, 2026.

## 1. Result

**Larger leaves did not resolve the correlation plateau and worsened CATE RMSE.** Correlation remained around 0.67. RMSE increased at every step from 10 to 20 to 40 to 80, at all three matched seeds.

Inference was **off**, honesty was **on**, and sampling was **95%** throughout. Results below are means of per-seed metrics on the same 200 held-out patients.

| Minimum leaf size | Correlation | CATE RMSE | RMSE change vs. 10 | Prediction SD | Bias | R-loss |
|---|---:|---:|---:|---:|---:|---:|
| 10 — reused | 0.66627 | 0.14507 | Reference | 0.10632 | −0.03163 | 0.16816 |
| 20 | 0.67082 | 0.14922 | +2.86% | 0.08479 | −0.03305 | 0.16835 |
| 40 | 0.66868 | 0.15504 | +6.87% | 0.06839 | −0.03434 | 0.16891 |
| 80 | 0.67540 | 0.16420 | +13.18% | 0.04647 | −0.03515 | 0.16994 |

1. Size 80 had the highest mean correlation, but the increase over size 10 was only 0.00913, while RMSE was 13.18% worse.
2. Increasing leaf size progressively reduced the variation in predicted effects. The true held-out effect SD was 0.18807; prediction SD fell from 0.10632 to 0.04647.
3. Observed-data R-loss also worsened at every step for each seed. The deterioration was not limited to the oracle RMSE metric.
4. Size 10 had the lowest RMSE among these four settings. This comparison did not test sizes below 10.

### 1.1. Variation across forest seeds

| Minimum leaf size | Correlation range | RMSE range |
|---|---:|---:|
| 10 | 0.65360–0.67371 | 0.14343–0.14619 |
| 20 | 0.66399–0.68033 | 0.14815–0.15096 |
| 40 | 0.65732–0.68498 | 0.15368–0.15718 |
| 80 | 0.66839–0.67988 | 0.16318–0.16518 |

These ranges describe forest randomness on one previously inspected fold, not uncertainty across independent datasets.

## 2. How the trees changed

| Minimum leaf size | Median leaves per tree | Median estimation patients per leaf | Median tree depth |
|---|---:|---:|---:|
| 10 | 17–18 | 17 | 7 |
| 20 | 9 | 32–33 | 5 |
| 40 | 5 | 62–64 | 3 |
| 80 | 3 | 113.5–117 | 2 |

The entries are per-forest medians; ranges span the three seeds. Half-integer medians are possible when pooling an even number of leaves.

Every tree used 760 training patients, divided into 380 for choosing splits and a separate 380 for estimating effects. Increasing minimum leaf size therefore pooled more patients within fewer leaves, as intended. All 800 training patients contributed to both roles across every forest.

The gain in patients per leaf came with a loss of partition detail. In this setting, that tradeoff narrowed the predicted effect distribution and worsened estimation error without substantially changing correlation. Simply increasing patients per leaf does not explain or fix the approximately 0.67 correlation plateau. This finding applies to the tested honesty-on configuration.

## 3. Controlled setup

1. Original fold 1: 800 training patients and 200 held-out patients, without propensity filtering.
2. X contained the five true modifier variables, encoded into 12 columns. W contained the five true confounders, encoded into 13 columns for nuisance adjustment.
3. The original fitted elastic-net nuisances and cached training residuals were reused. Fitting used observed binary outcomes; true effect and outcome/propensity probability columns were not supplied to the fits.
4. Fixed settings: 200 trees, square-root feature search, no maximum depth, inference off, honesty on, sample fraction 0.95, and one worker.
5. Fixed seeds: 120042, 1120042, and 2120042. Three existing size-10 forests were reused; nine new forests covered sizes 20, 40, and 80.
6. Each new forest changed **only `min_samples_leaf`** from its matched baseline. The actual tree subsamples and the patient indices in each honesty half were verified to be identical across leaf sizes within each seed.
7. Metrics are averages of the three separate forests' scores, not scores for an ensemble of their predictions.

## 4. Validation and scope

1. All nine new fits completed without warnings. Cached residual hashes remained unchanged.
2. The three reused baselines reproduced their saved predictions exactly, without refitting.
3. New predictions and 29 fit artifacts were frozen at **2026-09-21 17:52:09 UTC** before this experiment's evaluation phase read true effects.
4. Original sources, prior fit artifacts, and new fit artifacts passed hash verification after evaluation.
5. A separate validation script reloaded all 12 saved forests, reproduced every prediction exactly, independently recomputed all reported metrics and tree summaries, and checked nuisance predictions against the original fitted nuisance models. It also verified that every estimation leaf satisfied its configured minimum size.
6. This is an exploratory diagnostic on a previously examined fold. Production settings were not changed, and these scores do not constitute untouched validation.

## 5. Artifacts

- [Reproducible fitting and evaluation script](run_leaf_size_grid_2026-09-21.py)
- [Input manifest](input_manifest_2026-09-21.json)
- [Frozen prediction and model hashes](predictions_frozen_2026-09-21.json)
- [All per-seed metrics](metrics_by_seed_2026-09-21.csv)
- [Aggregate evaluation and paired contrasts](evaluation_2026-09-21.json)
- [Fit validation](fit_validation_2026-09-21.json)
- [Evaluation validation](evaluation_validation_2026-09-21.json)
- [Independent validation script](validate_leaf_size_grid_2026-09-21.py)
- [Independent validation results](independent_validation_2026-09-21.json)
