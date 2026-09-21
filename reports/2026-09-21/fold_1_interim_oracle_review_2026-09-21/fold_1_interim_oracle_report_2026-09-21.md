# Fold 1 interim oracle review — September 21, 2026

1. **Scope and timing**
   1. This is the user-requested interim review of the completed first outer fold of the four-way Stage 2 selection comparison. It evaluates the final causal-forest stage of that comparison, with shared nuisance predictions across all four arms.
   2. All 12 fold 1 prediction files and 46 supporting files were verified and hashed at 2026-09-21T13:07:00.682309+00:00 before this review loaded oracle values. Predictions were already complete by 03:53:45 UTC.
   3. The original execution plan deferred oracle evaluation until all 60 fits were frozen. The user explicitly requested this early fold 1 review. This is a disclosed change in evaluation timing; the running experiment, its sources, predictions, selection rules, and settings were not modified.
   4. No forest or nuisance model was fitted, tuned, or selected using this review. The other folds remain part of the prespecified comparison. This is an exploratory interim result on an already-inspected cohort.

2. **Oracle feature recovery**
   1. All five confounder concepts and all five modifier concepts remain represented among the 352 candidate measurements after refreshed training extraction and ontology supervision. The current definitions were checked against the earlier direct concept mapping. Separate gender identity was excluded as a direct substitute for biological sex.
   2. All five confounders were used by at least one of the five shared cross-fitted treatment or marginal-outcome nuisance models. The table counts inner models with a nonzero group for a direct feature.

      | Confounder | Treatment model: inner folds / 5 | Outcome model: inner folds / 5 |
      | --- | ---: | ---: |
      | Age | 5/5 | 5/5 |
      | Sex | 0/5 | 3/5 |
      | ECOG performance status | 5/5 | 5/5 |
      | Creatinine clearance | 5/5 | 0/5 |
      | Prior platinum therapy | 1/5 | 4/5 |

      The final full-training nuisance fits used all 352 candidates as their input design, shared across arms. Their saved audit records contain regularization and convergence information, but not their individual fitted coefficient lists; the counts above refer specifically to the logged cross-fitted models.

   3. Direct modifier concepts admitted to the final effect forest differed sharply by policy. Presence in a nuisance model does not make a feature available for effect-forest splits.

      | True modifier | Current joint | LLM | Permissive union | All candidates |
      | --- | ---: | ---: | ---: | ---: |
      | Histology | No | Yes | Yes | Yes |
      | EGFR mutation status | No | No | No | Yes |
      | Baseline NLR | Yes | Yes | Yes | Yes |
      | Brain metastases | No | No | No | Yes |
      | Baseline hemoglobin | No | No | No | Yes |
      | **Total** | **1/5** | **2/5** | **2/5** | **5/5** |

   4. Recovery of a concept is not full recovery of its oracle information. Prior platinum history was reduced from four oracle categories to a binary any-prior-treatment feature. The sex definition omits Other; EGFR and brain-metastasis definitions omit an explicit Unknown category; generic NSCLC does not identify an oracle histologic subtype.
   5. Measurement agreement was checked on all 200 held-out patients, before overlap exclusions. Continuous statistics use available paired measurements; categorical agreement uses available extracted values with explicit semantic label mappings. Missing values are reported separately.

      | Concept | Available / 200 | Agreement with latent oracle |
      | --- | ---: | --- |
      | Age | 198/200 | r = 0.989; MAE = 1.043 |
      | Sex | 196/200 | 196/196 (100.0%) |
      | ECOG performance status | 198/200 | 186/198 (93.9%) |
      | Creatinine clearance | 195/200 | r = 0.991; MAE = 1.204 |
      | Prior platinum therapy | 189/200 | 183/189 (96.8%) for any prior platinum vs none |
      | Histology | 198/200 | 175/198 (88.4%) |
      | EGFR mutation status | 192/200 | 157/192 (81.8%) |
      | Baseline NLR | 155/200 | r = 0.915; MAE = 0.322 |
      | Brain metastases | 197/200 | 185/197 (93.9%) |
      | Baseline hemoglobin | 198/200 | r = 0.906; MAE = 0.489 |

      NLR was missing in 45/200 held-out records. Eleven held-out latent oracle NLR values were negative, demonstrating an additional synthetic-data realism limitation. These comparisons measure agreement with the simulation values, not manually adjudicated extraction accuracy: generated text, clinical timing, ontology definitions, and extraction can all contribute to disagreement.

3. **Final causal-estimation performance**
   1. Every arm used the same 720 eligible training patients and 180 eligible held-out patients, from 800 and 200 respectively. Eligibility was based on shared estimated propensities in [0.1, 0.9]. Row order, treatment, outcome, nuisance predictions, and eligibility were identical in all 12 prediction files.
   2. The table reports means across the three prespecified forest seeds. CATE RMSE and bias are on the probability-difference scale; 0.188 corresponds to about 18.8 percentage points. Lower RMSE and R-loss are better; higher correlation is better. Seed ranges represent algorithmic variation, not confidence intervals.

      | Method | Features in X | Modifiers | CATE RMSE: mean (seed range) | CATE correlation | Held-out R-loss |
      | --- | ---: | ---: | ---: | ---: | ---: |
      | Current joint gate | 18 | 1/5 | 0.1877 (0.1850–0.1909) | 0.296 | 0.19076 |
      | LLM adjudication | 11 | 2/5 | 0.1917 (0.1899–0.1926) | 0.286 | 0.19130 |
      | Permissive union | 49 | 2/5 | 0.1875 (0.1871–0.1878) | 0.338 | 0.18925 |
      | All candidates | 352 | 5/5 | 0.1941 (0.1909–0.1967) | 0.184 | 0.19030 |

   3. A constant treatment effect fitted only on training residuals had CATE RMSE **0.1967** and held-out R-loss **0.19073**. The current gate reduced RMSE by 4.6% relative to this constant; the permissive union reduced it by 4.7%.
   4. The permissive union and current gate were effectively tied on RMSE: 0.1875 versus 0.1877, a 0.10% relative difference. The union had the best correlation and a 0.78% R-loss improvement over the constant comparator. The current gate's mean R-loss was essentially the same as the constant's.
   5. Retaining all five modifier concepts did not improve CATE accuracy in this fold. The all-candidate arm had lower correlation and more compressed predictions than the current gate. This observation does not identify whether dimensionality, measurement quality, forest settings, or another component caused the difference.

4. **Bias, heterogeneity, and intervals**
   1. The true average treatment effect in the 180 eligible held-out patients was -0.0628; true individual-effect standard deviation was 0.1951.

      | Method | Mean estimated effect | Bias | Predicted effect SD | Empirical coverage of nominal 95% intervals |
      | --- | ---: | ---: | ---: | ---: |
      | Current joint gate | -0.0417 | +0.0211 | 0.0568 | 52.6% |
      | LLM adjudication | -0.0350 | +0.0278 | 0.0870 | 56.3% |
      | Permissive union | -0.0350 | +0.0278 | 0.0410 | 50.9% |
      | All candidates | -0.0366 | +0.0262 | 0.0250 | 52.4% |

   2. Every method substantially compressed true treatment-effect heterogeneity. Predicted standard deviations ranged from about 0.025 to 0.087, compared with 0.195 for the oracle. Nominal 95% intervals covered the true conditional effects only about 51–56% of the time on average across seeds.
   3. Confounder-concept recovery is stronger than modifier admission, but the final models still recover only a modest portion of patient-level treatment-effect variation. Broader admission alone did not solve that problem in this fold.

5. **Limits and operational context**
   1. This is one outer fold, with 180 evaluated patients, and it does not establish a winning policy for the complete experiment. The three seeds do not provide three independent samples.
   2. All methods share refreshed Gemma 4 26B A4B measurements and nuisances. This is not a controlled old-extractor-versus-new-extractor comparison.
   3. Extraction reliability and structural-repair reasoning policies changed during the refresh, with compatible earlier completed fallback records retained. The completed candidate measurements therefore include that documented history; the four arms use the same resulting measurements.
   4. Binding LLM role adjudication used 15 candidates per batch in fold 1 after the original 20-candidate rendered prompt exceeded the request budget. Its 24 batches covered all 352 candidates without additional evidence truncation. Smaller batch context is an operational change relevant to interpretation.
   5. Subsequent fitting and operational recovery must continue to use the frozen experiment rules; these interim oracle findings do not authorize adaptive tuning or oracle-based changes.

6. **Saved audit and results**
   1. [Prediction and supporting-input freeze](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-21/fold_1_interim_oracle_review_2026-09-21/frozen_fold_1_inputs_2026-09-21.json).
   2. [Causal metrics and seed summaries](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-21/fold_1_interim_oracle_review_2026-09-21/fold_1_causal_metrics_2026-09-21.json).
   3. [Metrics by seed, CSV](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-21/fold_1_interim_oracle_review_2026-09-21/fold_1_causal_metrics_by_seed_2026-09-21.csv).
   4. [Reviewed concept lineage, nuisance selection, and measurement agreement](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-21/fold_1_interim_oracle_review_2026-09-21/fold_1_feature_recovery_2026-09-21.json).
   5. [Fold 1 predictions joined to oracle values](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-21/fold_1_interim_oracle_review_2026-09-21/fold_1_predictions_with_oracle_2026-09-21.csv).
