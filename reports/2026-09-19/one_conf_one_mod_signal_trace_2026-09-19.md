# One-confounder / one-modifier run: where effect-modification signal weakened

Analysis date: **2026-09-19**

1. **Answer and run identification**

   1. **PD-L1 was retained as an effect modifier in all five outer folds.** The saved evidence does not show the concept being dropped during the Stage 1 handoff, discovery, consolidation, or final role assignment.
   2. The first measurable information loss is in **measurement: extraction, re-extraction, and—in folds 2–4—harmonization**. The final estimates then show considerable attenuation of the remaining PD-L1 pattern, particularly in folds 3 and 4.
   3. The intermediate joint interaction screen selected no modifiers in outer folds 1, 4, and 5. However, the candidate-wise screen still supported PD-L1, and LLM role adjudication retained it. Thus, the joint screen's failure was **rescued**, not a terminal loss in this run.
   4. Run examined: `one_conf_one_mod_nsclc_full/stage2_pre_roles_refactor`, completed **September 17, 2026, at 11:17:06 p.m. Eastern** (`2026-09-18T03:17:06Z`). This is the latest completed saved run found for this cohort; there is no newer active `stage2/` run.
   5. This was the archived **`llm_roles`** workflow, with the optional post-extraction selection-consolidation pass **disabled**. These findings describe its saved outputs, not a rerun of the current branch's newer selection policy.

2. **How signal was assessed**

   1. The synthetic truth identifies age as the confounder and PD-L1 category as the explicit treatment-interaction variable: `<1%`, `1–49%`, and `≥50%`.
   2. “Effect” below means the difference in outcome probability under treatment versus control. Negative means lower outcome probability under treatment; no clinical benefit/harm interpretation is assumed.
   3. The true pattern is nonmonotonic: approximately **+3 percentage points** below 1%, **−6 points** at 1–49%, and **+28 to +34 points** at ≥50%. The probability-scale effect also varies with age through the logistic outcome model.
   4. Each outer fold has 800 training patients and 200 heldout patients. Extraction accuracy means agreement with the three synthetic PD-L1 categories; missing values count as incorrect. “Wrong” excludes missing values.
   5. The high-minus-middle contrast is the mean effect in the true ≥50% group minus the mean effect in the true 1–49% group. The percentage retained is the estimated contrast divided by the oracle contrast. It is a descriptive measure of attenuation, **not a causal decomposition of error or a significance test**.

   | Outer fold | Final training extraction correct | Heldout extraction correct | Heldout missing / wrong | Candidate-wise / joint screen support | True → estimated high-minus-middle contrast | Contrast retained |
   |---|---:|---:|---:|---|---:|---:|
   | 1 | 698/800 (87.3%) | 167/200 (83.5%) | 29 / 4 | 3/5 / 0/5 inner folds | 38.4 → 13.3 percentage points | 34.8% |
   | 2 | 745/800 (93.1%) | 145/200 (72.5%) | 23 / 32 | 3/5 / 4/5 inner folds | 33.9 → 11.8 points | 34.8% |
   | 3 | 664/800 (83.0%) | 160/200 (80.0%) | 30 / 10 | 3/5 / 2/5 inner folds | 39.8 → 6.4 points | 16.2% |
   | 4 | 661/800 (82.6%) | 161/200 (80.5%) | 32 / 7 | 3/5 / 0/5 inner folds | 39.8 → 4.5 points | 11.3% |
   | 5 | 725/800 (90.6%) | 183/200 (91.5%) | 15 / 2 | 3/5 / 0/5 inner folds | 40.1 → 17.2 points | 43.0% |

   Candidate-wise support counts selection into the saved top-N evidence list. Joint support counts a nonzero selected PD-L1 feature group. Neither is the final role decision: **all five final decisions retained PD-L1**.

3. **Trace through the shared pipeline**

   1. **Stage 1 evidence → Stage 2 discovery: the concept survives.**
      1. PD-L1 appears explicitly in 55, 66, 57, 61, and 56 of the 400 compiled evidence cards for outer folds 1–5, respectively.
      2. Discovery produced 18, 13, 15, 13, and 14 PD-L1-related candidate descriptions, spanning 8–9 source architectures in each fold.
      3. A PD-L1 feature is present in every consolidated definition set and every final selected definition set.
      4. These are evidence-lineage checks. They establish that the concept crossed the boundary; they do not prove that every quantitative Stage 1 effect estimate was accurate or that no information was compressed upstream.
   2. **Ontology and extraction: the first observable degradation.**
      1. PD-L1 initially had a continuous percentage representation in all folds. Ontology supervision changed folds 1 and 5 to three ordered categories; folds 2–4 remained continuous.
      2. Across supervision and re-extraction, fold 1's training correct count fell from 720 to 698, and fold 5's from 734 to 725. Wrong-category assignments decreased, but missingness increased. Because a new extraction was involved, this is not evidence that changing the type alone caused the decline.
      3. The continuous folds confuse the clinical boundary `<1%` with the numeric value `1`, moving true negative-expression patients into the middle category. This is especially pronounced in fold 2's heldout extraction.
      4. Harmonization adds 0, 1, 5, 2, and 0 heldout missing values in folds 1–5. Examples include `1-49%` in fold 2; `70%` and `< 1%` in fold 3; and `< 1 %` and `>= 50` in fold 4. Fold 3 also loses the uninterpretable value `Present`, so not every newly missing value represents usable quantitative information.
      5. All but one of the 129 heldout records missing PD-L1 after harmonization still mention PD-L1 or tumor proportion score in their text. A keyword mention alone does not prove that an unambiguous eligible measurement was available.
   3. **Statistical screening → role adjudication: weak joint support, but no final exclusion.**
      1. The candidate-wise residualized treatment-interaction screen places PD-L1 in its top-N list in 3 of 5 inner folds for every outer fold. Mean heldout R-loss improvement is positive in all five outer folds.
      2. The joint elastic-net interaction screen selects PD-L1 in 0, 4, 2, 0, and 0 inner folds, respectively. In outer folds 1, 4, and 5, those joint fits select no modifier features at all.
      3. LLM adjudication reconciles the two evidence sources and retains PD-L1 as an effect modifier in every fold. The saved final definitions confirm that it reaches the effect model.
      4. The candidate-wise PD-L1 interaction uses a single ordered score in folds 1 and 5 and a single numeric value in folds 2–4. That interaction basis cannot fully represent the true low → negative-middle → high shape. Nevertheless, it retained enough evidence for selection here.
   4. **Final CATE estimation: remaining heterogeneity is strongly flattened.**
      1. The final models include 8, 7, 8, 11, and 6 effect-modifier concepts, respectively, including PD-L1. These are concept counts, not encoded matrix widths.
      2. Age is retained as a confounder in every fold but is not explicitly included among the forest's effect-modifier inputs. Adjustment for age and allowing the predicted effect to vary with age are different operations.
      3. The run uses an honest causal forest with 200 trees and no propensity trimming. PD-L1's final attenuation is not explained by a fallback to an empty/constant modifier design.
      4. The true middle group's mean effect is negative in every fold; its mean predicted effect is positive in every fold. This remains true when restricting to heldout patients whose PD-L1 category was extracted correctly.

4. **Where the loss appears in each outer fold**

   1. **Outer fold 1: supervision/re-extraction loss, rescued screening, then attenuation.**
      1. PD-L1 survives discovery and consolidation. Across the first saved supervision extraction and the final training extraction, missing values increase from 51 to 92; wrong categories decrease from 29 to 10.
      2. The heldout model input has 29 missing and 4 wrong PD-L1 categories. Harmonization adds no further missingness.
      3. The joint screen gives PD-L1 0/5 votes, but candidate-wise support of 3/5 leads to retention by the LLM.
      4. Final estimates preserve some separation of the high group but lose the middle group's negative effect: its mean is **+9.4 points instead of −6.3**. The high-minus-middle contrast retains **34.8%** of its true size. Predicted versus oracle individual-effect correlation is **0.511**.
   2. **Outer fold 2: the clearest heldout extraction failure, followed by upward-biased effects.**
      1. Training PD-L1 extraction is the best of the five folds at 93.1%, but heldout accuracy falls to 72.5%.
      2. Of 80 truly `<1%` heldout patients, **32 are assigned to the middle group**, including **31 encoded as exactly `1.0`**. Inspection of example notes confirms explicit `<1%` text despite the recorded value of 1.0.
      3. Harmonization adds one missing value by dropping `1-49%`; total heldout missingness is 23/200.
      4. This is not a screening failure: PD-L1 has 3/5 candidate-wise and 4/5 joint support and is retained.
      5. The high group's estimated effect is close to its oracle mean, but the middle group's estimate is **+16.0 points instead of −6.0**. The contrast retains **34.8%**; individual-effect correlation is **0.334**. Even correctly measured middle-group patients average **+16.9 instead of −6.0 points**, so heldout miscoding alone cannot explain the result.
   3. **Outer fold 3: noisy measurement and harmonization, followed by substantial final flattening.**
      1. Final training extraction contains 88 missing and 48 wrong PD-L1 categories. Heldout extraction contains 10 wrong categories; harmonization increases missingness from 25 to 30.
      2. PD-L1 is retained with 3/5 candidate-wise and 2/5 joint support.
      3. The middle group's estimate is **+14.0 points instead of −6.7**, while the high group's estimate is **+20.5 instead of +33.2**. Only **16.2%** of the high-minus-middle contrast remains; individual-effect correlation is **0.466**.
      4. The flattening persists among correctly measured patients: middle **+14.1 versus −6.7**, high **+21.1 versus +33.8 points**.
   4. **Outer fold 4: measurement loss plus the strongest final flattening.**
      1. Training has 103 missing and 36 wrong PD-L1 categories. Heldout data have 32 missing and 7 wrong categories after harmonization; two of the missing values are introduced by harmonization.
      2. The joint screen gives 0/5 support, but candidate-wise support of 3/5 again leads the LLM to retain PD-L1.
      3. This fold has the most selected effect-modifier concepts, 11. That is a possible source of competition for splits, not a demonstrated cause of failure.
      4. The middle and high groups are estimated at **+13.3 and +17.8 points**, versus true effects of **−6.3 and +33.6**. Only **11.3%** of the contrast remains; individual-effect correlation is **0.243**, the lowest of the five folds.
      5. Correctly measured middle-group patients still average **+13.1 instead of −6.2 points**. The failure therefore persists after excluding heldout measurement mistakes.
   5. **Outer fold 5: best measurement and strongest retained final signal, but still the wrong middle-group sign.**
      1. The categorical revision reduces training wrong-category assignments from 10 to 7 but increases missingness from 55 to 68; one initially unparsed value also disappears.
      2. Heldout accuracy is 91.5%, the best of the five folds: 15 missing and 2 wrong categories, with no additional loss in harmonization.
      3. PD-L1 has 0/5 joint votes but 3/5 candidate-wise support and is retained by the LLM.
      4. Final estimates retain **43.0%** of the high-minus-middle contrast, the best result here; individual-effect correlation is **0.678**. The middle group's estimate remains **+7.9 points instead of −6.6**, including **+6.7 instead of −6.7** among correctly measured patients.

5. **How much information remains after measurement?**

   1. A diagnostic lookup was constructed using each outer training fold only: take the mean oracle individual effect within a PD-L1 category, then assign that mean to heldout patients in the same category. One version uses the true category; another uses the measured category, including a missing category.
   2. This uses unavailable oracle effects and is **not a deployable causal estimator, an achievable-performance guarantee, or a formal error decomposition**. It only checks whether the measured feature still distinguishes groups with different true effects.

   | Outer fold | Correlation using true-category oracle lookup | Correlation using measured-category oracle lookup | Actual final CATE correlation |
   |---|---:|---:|---:|
   | 1 | 0.879 | 0.781 | 0.511 |
   | 2 | 0.863 | 0.754 | 0.334 |
   | 3 | 0.888 | 0.845 | 0.466 |
   | 4 | 0.892 | 0.772 | 0.243 |
   | 5 | 0.875 | 0.818 | 0.678 |

   3. The measured PD-L1 column retains substantial information in every fold. In particular, fold 3 has a measured-category diagnostic correlation of 0.845 despite much weaker final predictions. Final estimation is not recovering all of the pattern still identifiable from the measurements.
   4. The saved artifacts cannot uniquely separate noisy training measurements, nuisance-model error, finite-sample uncertainty, forest regularization, competing features, and omission of age from the explicit effect inputs. Distinguishing those contributions requires controlled refits in a separate diagnostic run.

6. **Improvements most directly supported by this trace**

   1. **Make PD-L1 measurement consistent across folds.** Preserve the three clinically stated categories or an interval-aware representation. Never silently interpret `<1%` as the point value `1%`. Verify both training and heldout extraction against examples containing comparison symbols, Unicode spaces/dashes, and percentage ranges. Generalize this validation to other thresholded measurements.
   2. **Make harmonization preserve valid measurements.** Handle values such as `70%`, `< 1 %`, and `>= 50` according to the frozen feature contract. Record when harmonization changes a usable extraction into missingness. Do not infer a numeric score from an unquantified value such as `Present`.
   3. **Test nonlinear/category-specific effect contrasts.** A single numeric or ordered-score interaction cannot express the observed nonmonotonic truth. For a future prespecified method comparison, use category-specific treatment interactions or flexible effect bases, while keeping selection and tuning inside the training folds.
   4. **Separate adjustment from effect prediction.** Consider whether relevant baseline predictors such as age should also enter the CATE inputs on the chosen effect scale. In this synthetic model, age changes risk differences through the logistic link even without an explicit age-by-treatment logit interaction.
   5. **Use controlled estimation diagnostics.** Compare the frozen extracted inputs with a compact effect model, alternative forest settings, and nuisance-model diagnostics, one change at a time. Evaluate any oracle-input ablation only as a separately labeled simulation diagnostic. Do not tune the production pipeline to these outer-heldout results.
   6. **Track signed group effects, not just overall CATE correlation.** Every fold missed the middle group's negative mean effect. A single aggregate correlation obscures that recurring failure.

7. **Provenance and limits**

   1. No discovery, extraction, screening, or causal model was rerun. Production artifacts were read only. The post-hoc analysis checks patient IDs, treatment/outcome alignment, row coverage, and separation of training and heldout rows.
   2. The saved prediction SHA-256 is `b2131f611e960c67191d657e14fc8808d155be5095bd9c521b45281c64133f4e`, matching the run's saved oracle-evaluation hash. The analysis records hashes for 66 source files before joining oracle values.
   3. “First saved extraction” refers to the extraction saved for the first ontology-supervision round; it need not be the first individual LLM response. Aggregate comparisons cannot identify the precise prompt or extraction attempt responsible for every change.
   4. The oracle audit compares extracted categories with generator truth. Synthetic notes can themselves contain ambiguity or inconsistencies; individual errors require note review before assigning responsibility to a particular extraction rule.
   5. Saved artifacts establish where the feature and the effect pattern remain visible. They do not identify a unique causal explanation for the final attenuation.
   6. Reproducibility and evidence:
      1. [Aggregate audit, confusion matrices, screening evidence, and source hashes](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-19/one_conf_one_mod_signal_trace_2026-09-19.json).
      2. [Read-only analysis script](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-19/one_conf_one_mod_signal_trace_2026-09-19.py).
      3. [Frozen cross-fitted predictions](/data1/ken/pcori_dev/causal-dragonnet-text/artifacts/research_all_evidence/one_conf_one_mod_nsclc_full/stage2_pre_roles_refactor/cross_fitted_predictions.csv).
      4. [Run configuration](/data1/ken/pcori_dev/causal-dragonnet-text/artifacts/research_all_evidence/one_conf_one_mod_nsclc_full/stage2_pre_roles_refactor/config.json).
      5. Final selected definitions: [fold 1](/data1/ken/pcori_dev/causal-dragonnet-text/artifacts/research_all_evidence/one_conf_one_mod_nsclc_full/stage2_pre_roles_refactor/outer_001/final_definitions.json), [fold 2](/data1/ken/pcori_dev/causal-dragonnet-text/artifacts/research_all_evidence/one_conf_one_mod_nsclc_full/stage2_pre_roles_refactor/outer_002/final_definitions.json), [fold 3](/data1/ken/pcori_dev/causal-dragonnet-text/artifacts/research_all_evidence/one_conf_one_mod_nsclc_full/stage2_pre_roles_refactor/outer_003/final_definitions.json), [fold 4](/data1/ken/pcori_dev/causal-dragonnet-text/artifacts/research_all_evidence/one_conf_one_mod_nsclc_full/stage2_pre_roles_refactor/outer_004/final_definitions.json), [fold 5](/data1/ken/pcori_dev/causal-dragonnet-text/artifacts/research_all_evidence/one_conf_one_mod_nsclc_full/stage2_pre_roles_refactor/outer_005/final_definitions.json).
      6. [Synthetic data-generating metadata](/data1/ken/pcori_dev/causal-dragonnet-text/synthetic_data/example_synthetic_datasets/one_confounder_one_effect_modifier_nsclc_with_structured/metadata.json), used only after freezing the analysis inputs.
