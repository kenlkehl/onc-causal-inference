# Fold 1 oracle-variable forests: 10% versus 45% and 95% sampling

Date: September 21, 2026.

## 1. Result

**Ten percent sampling did not improve on 45% sampling.** It produced slightly lower mean correlation and higher CATE RMSE with either honesty setting. With honesty off, 10% did outperform 95%.

Inference was **off** and minimum leaf size was **10** throughout. Results are means of per-seed metrics across three fixed seeds on the same 200 held-out patients.

| Honesty | Sample fraction | Correlation | CATE RMSE | Prediction SD | Bias | R-loss |
|---|---:|---:|---:|---:|---:|---:|
| On | 10% — new | 0.65325 | 0.17064 | 0.03676 | −0.03792 | 0.17080 |
| On | 45% — reused | 0.66752 | 0.15020 | 0.08221 | −0.03234 | 0.16816 |
| On | 95% — reused | 0.66627 | 0.14507 | 0.10632 | −0.03163 | 0.16816 |
| Off | 10% — new | 0.66655 | 0.14927 | 0.08687 | −0.03121 | 0.16871 |
| Off | 45% — reused | 0.67761 | 0.14276 | 0.14545 | −0.02996 | 0.16802 |
| Off | 95% — reused | 0.59344 | 0.17209 | 0.19068 | −0.02109 | 0.16992 |

1. With honesty on, 10% sampling increased mean RMSE by **13.61% versus 45%** and **17.62% versus 95%**. RMSE was worse at all three matched seeds against both baselines.
2. With honesty off, 10% sampling increased mean RMSE by **4.56% versus 45%**, but reduced it by **13.26% versus 95%**. These RMSE directions held at every matched seed. Correlation also improved over 95% at every seed.
3. Among these six inference-off settings, 45% sampling with honesty off had the highest mean correlation and lowest mean RMSE.
4. The true held-out effect SD was 0.18807. At 10%, the predicted effect distribution was especially narrow with honesty on: SD 0.03676.

### 1.1. Ten-percent runs by seed

| Honesty | Forest seed | Correlation | CATE RMSE |
|---|---:|---:|---:|
| On | 120042 | 0.55689 | 0.17488 |
| On | 1120042 | 0.70526 | 0.16533 |
| On | 2120042 | 0.69760 | 0.17170 |
| Off | 120042 | 0.65550 | 0.14673 |
| Off | 1120042 | 0.64845 | 0.15117 |
| Off | 2120042 | 0.69571 | 0.14991 |

With honesty on, two of the three 10% seeds had higher correlation than their 45% and 95% counterparts, but the first seed performed substantially worse. The 0.55689–0.70526 correlation range was much wider than the earlier honesty-on ranges: 0.66517–0.66970 at 45% and 0.65360–0.67371 at 95%. These ranges describe forest randomness on this fold, not uncertainty across independent datasets.

## 2. How the trees changed

| Honesty | Sample fraction | Patients per tree | Split / estimation patients | Median leaves per tree | Median estimation patients per leaf |
|---|---:|---:|---:|---:|---:|
| On | 10% | 80 | Separate 40 / 40 | 2 | 15–16 |
| On | 45% | 360 | Separate 180 / 180 | 9 | 17 |
| On | 95% | 760 | Separate 380 / 380 | 17–18 | 17 |
| Off | 10% | 80 | Same 80 / 80 | 5 | 13–14 |
| Off | 45% | 360 | Same 360 / 360 | 20 | 14 |
| Off | 95% | 760 | Same 760 / 760 | 42 | 13–14 |

The leaf entries are medians within each forest; ranges span the three seeds. All forests still used all 800 training patients in both roles across their 200 trees.

1. With honesty on at 10%, median depth was one. Between 36 and 43 of 200 trees, or **18–21.5%**, made no split at all. The remaining trees had at most depth two.
2. With honesty off at 10%, median depth was three. Between zero and two trees per forest made no split.
3. Lower sampling chiefly reduced the number of leaves; typical leaf sample sizes stayed near the unchanged minimum. The resulting shallow honest forests produced strongly compressed effect predictions. The change did not resolve the correlation plateau.
4. All settings retained 200 trees. This comparison does not test whether more trees would reduce the greater seed variation seen at 10%.

## 3. Controlled setup

1. Same fold 1: 800 training patients and 200 held-out patients, without propensity filtering.
2. X contained the five true modifier variables, encoded into 12 columns. W contained the five true confounders, encoded into 13 columns for nuisance adjustment.
3. Original fitted elastic-net nuisances and cached training residuals were reused. No nuisance models were refitted. Fitting used observed binary outcomes, with no true effect or outcome/propensity probability columns supplied.
4. Each new forest copied its matched 45% baseline and changed **only `max_samples` to 0.10**. The corresponding 95% baseline was also checked to differ only in that parameter.
5. Fixed settings included 200 trees, square-root feature search, minimum leaf size 10, no maximum depth, inference off, and one worker. Both honesty settings used seeds 120042, 1120042, and 2120042.
6. Six new forests were fitted. Twelve existing 45%/95% forests were reused. Reported means average the separate per-seed scores; they do not score an ensemble of the three forests.

## 4. Validation and scope

1. All six new fits completed without warnings. Their actual tree subsamples were verified to contain exactly 80 distinct training patients, with disjoint 40/40 honest halves or the same 80 patients serving both roles.
2. Cached residual hashes remained unchanged. All twelve reused baselines reproduced their saved predictions exactly without refitting.
3. New predictions and 20 fit artifacts were frozen at **2026-09-21 18:24:36 UTC** before this experiment's evaluation phase read true effects.
4. Original source files, prior fit artifacts, and new fit artifacts passed hash verification after evaluation.
5. A separate validation script reloaded all 18 saved forests, reproduced every prediction exactly, independently recomputed all metrics and tree summaries, and checked nuisance predictions against the original fitted nuisance models. Every estimation leaf met the minimum size of 10.
6. This remains an exploratory diagnostic on a previously inspected fold. Production settings were not changed, and these results do not constitute untouched validation.

## 5. Artifacts

- [Fitting and evaluation script](run_sampling_010_2026-09-21.py)
- [Input manifest](input_manifest_2026-09-21.json)
- [Frozen predictions and model hashes](predictions_frozen_2026-09-21.json)
- [All per-seed metrics](metrics_by_seed_2026-09-21.csv)
- [Aggregate evaluation and paired contrasts](evaluation_2026-09-21.json)
- [Fit validation](fit_validation_2026-09-21.json)
- [Evaluation validation](evaluation_validation_2026-09-21.json)
- [Independent validation script](validate_sampling_010_2026-09-21.py)
- [Independent validation results](independent_validation_2026-09-21.json)
