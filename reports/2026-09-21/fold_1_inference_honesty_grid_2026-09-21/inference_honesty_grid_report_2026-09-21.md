# Fold 1 oracle-variable forests: inference × honesty

Date: September 21, 2026.

## 1. Results

**Keeping inference on and turning honesty off performed best on this fold.** It reduced mean CATE RMSE by **7.97%**, from 0.14997 to 0.13803, while increasing mean correlation from 0.67577 to 0.69442.

All results use true modifiers in X and true confounders in W, the same 800 training and 200 held-out patients, observed binary outcomes, and the original fitted nuisance functions. The sampling fraction remained **45% in every cell**. Values below are arithmetic means of metrics across three seeds, not ensemble-prediction metrics.

| Inference | Honesty | Correlation | CATE RMSE | Prediction SD | Mean prediction minus mean truth | R-loss |
|---|---|---:|---:|---:|---:|---:|
| True | True — reused baseline | 0.67577 | 0.14997 | 0.07913 | −0.03053 | 0.16845 |
| True | False | **0.69442** | **0.13803** | 0.14340 | −0.02364 | **0.16735** |
| False | True | 0.66752 | 0.15020 | 0.08221 | −0.03234 | 0.16816 |
| False | False | 0.67761 | 0.14276 | 0.14545 | −0.02996 | 0.16802 |

The true held-out effect SD is 0.18807. RMSE measures error in the outcome-probability difference; R-loss uses the same held-out nuisance predictions in every cell, so it is directly comparable here.

### 1.1. Seed variation

| Inference | Honesty | Correlation range | RMSE range |
|---|---|---:|---:|
| True | True | 0.65846–0.68678 | 0.14612–0.15659 |
| True | False | 0.68156–0.70097 | 0.13603–0.14083 |
| False | True | 0.66517–0.66970 | 0.14869–0.15134 |
| False | False | 0.66966–0.68757 | 0.14138–0.14410 |

1. Turning honesty off reduced RMSE at every matched seed, with inference either on or off.
2. With inference on, turning honesty off improved correlation at two of three seeds; with inference off, it improved correlation at all three seeds.
3. Turning inference off reduced average correlation under both honesty settings. With honesty off, it worsened RMSE and correlation at all three matched seeds. With honesty on, its paired effects were mixed.
4. These ranges describe algorithmic randomness on one fixed fold. They are not confidence intervals or independent cohort replications.

## 2. What was held fixed

1. X: the five true modifiers—histology, EGFR status, NLR, brain metastasis status, and hemoglobin—encoded into 12 columns.
2. W: the five true confounders—age, sex, ECOG, creatinine clearance, and prior platinum therapy—encoded into 13 columns. These remain in nuisance adjustment.
3. The original cross-fitted elastic-net nuisance estimates and exact cached treatment/outcome residuals were reused. No nuisance model was refitted.
4. The outcomes remain the observed binary outcomes. This experiment uses oracle covariate values, not the later exact-nuisance or noiseless-outcome controls.
5. Forest settings: 200 trees, minimum leaf size 10, square-root feature search, sample fraction 0.45, no maximum depth, and one worker. All other parameters were copied from the original frozen forests.
6. Seeds: 120042, 1120042, and 2120042.
7. The three `inference=True, honest=True` forests were reused. Only the other three grid cells were fitted, producing nine new forests.

## 3. What the two flags changed

### 3.1. Inference changes the subsampling arrangement

The installed EconML implementation was inspected directly:

1. With inference enabled, groups of four trees share a random half-sample of 400 patients; each tree draws 360 patients from that half-sample.
2. With inference disabled, each tree draws 360 patients directly from the full pool of 800.
3. Thus inference changes the arrangement of tree samples as well as enabling uncertainty estimation. Turning it off does not automatically increase the sampling fraction.

### 3.2. Honesty changes how each tree uses its patients

| Honesty | Unique patients per tree | Patients choosing splits | Patients estimating effects | Median leaves per tree | Median effect-estimation patients per leaf |
|---|---:|---:|---:|---:|---:|
| True | 360 | 180 | A separate 180 | 9 | 17 |
| False | 360 | All 360 | The same 360 | 20 | 14 |

These medians were identical across the three seeds and both inference settings within each honesty condition. Every forest used all 800 training patients across its trees, in both split-selection and effect-estimation roles.

Turning honesty off supplied more patients for each stage, but also allowed finer partitions. Consequently, the typical leaf contained fewer effect-estimation patients, rather than twice as many. The change combines reuse of outcomes for splitting and estimation with a change in tree resolution.

## 4. Interpretation

1. Honesty contributed to the earlier attenuation of predicted heterogeneity under these settings. With inference on, turning honesty off increased prediction SD from 0.079 to 0.143 and improved RMSE substantially.
2. Correlation improved only modestly. This supports honesty/sample allocation as one contributor, but it does not explain the large remaining gap in effect-pattern recovery. Increasing spread alone cannot improve Pearson correlation through a simple positive affine rescaling.
3. Disabling inference alone did not improve average correlation at the fixed 45% sampling fraction. The result does not test whether its ability to permit larger sampling fractions could be useful.
4. The best observed cell is descriptive performance on this previously inspected fold. No production setting was changed or selected using these oracle scores.
5. This report evaluates held-out point predictions. Interval calibration was not evaluated.

## 5. Validation and provenance

1. All nine new forests completed without warnings. Every prediction was finite and aligned to the same ordered 200 held-out IDs.
2. Only the `inference` and `honest` parameters changed. Cached nuisance residual hashes remained identical throughout all fits.
3. The baseline forests were loaded and their predictions reproduced exactly without refitting. Their direct residual-fitting equivalence had also been verified by exact replay in the previous estimation-ablation experiment.
4. Per-tree subsamples, honest splitting indices, leaf sizes, and whole-forest patient coverage were inspected from the saved models.
5. New predictions were frozen at **2026-09-21 16:04:18 UTC** before this experiment's evaluation phase read true effects. The fold's oracle effects had already been inspected in prior experiments; this is exploratory, not a blinded validation study.
6. All original source and fit hashes and all 29 new frozen artifacts were verified after evaluation. Production source and the ongoing parent experiment were unchanged.

Artifacts:

- [Reproducible fitting and evaluation script](run_inference_honesty_grid_2026-09-21.py)
- [Input manifest](input_manifest_2026-09-21.json)
- [Prediction freeze and artifact hashes](predictions_frozen_2026-09-21.json)
- [Metrics for all 12 seed/configuration combinations](metrics_by_seed_2026-09-21.csv)
- [Aggregate metrics and paired contrasts](evaluation_2026-09-21.json)
- [Baseline reuse and tree audits](baseline_reuse_validation_2026-09-21.json)
- [Fit validation](fit_validation_2026-09-21.json)
- [Evaluation validation](evaluation_validation_2026-09-21.json)
- [Independent metric, model reload, and patient-usage checks](independent_validation_2026-09-21.json)
