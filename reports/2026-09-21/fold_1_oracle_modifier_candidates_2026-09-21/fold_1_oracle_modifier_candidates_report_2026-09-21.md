# Fold 1: causal forests restricted to oracle-matched modifier candidates

Date: September 21, 2026. User-requested diagnostic using oracle identities for candidate selection.

1. **Result**
   1. Restricting the final forest to candidates representing the five known effect-modifier concepts improved held-out CATE performance. Across the same three seeds, mean RMSE was **0.1770** with square-root split sampling and **0.1780** with every encoded feature available at a split.
   2. Relative to all 352 candidates under the corresponding split setting, RMSE decreased **8.83%** and **5.45%**, respectively. Both diagnostic settings improved on their corresponding all-candidates run in every paired seed.
   3. Mean correlation with oracle effects increased to **0.461** and **0.454**. The square-root diagnostic also reduced RMSE by **5.72%** relative to current joint selection.
   4. This is an oracle-informed candidate-selection diagnostic. It measures what the existing extracted data can achieve when the relevant candidate identities are supplied.

2. **Candidates and scope**
   1. Five modifier concepts correspond to **six saved candidates**, because EGFR has two direct representations. Both EGFR candidates were retained without choosing between them based on performance.

   | Modifier concept | Saved candidate names | Candidate IDs |
   |---|---|---|
   | Hemoglobin | hemoglobin_level | outer_001_feature_155 |
   | NLR | neutrophil_to_lymphocyte_ratio | outer_001_feature_228 |
   | Brain metastases | brain_metastases_presence | outer_001_feature_052 |
   | EGFR mutation status | biomarker_egfr_status; egfr_mutation_status | outer_001_feature_032; outer_001_feature_111 |
   | Histology | lung_cancer_histology | outer_001_feature_192 |

   2. These six candidates produce **18 encoded columns**. Square-root sampling requests about four columns per split; the all-feature setting makes all 18 available. The existing numeric imputation, categorical levels, and missingness indicators were preserved.
   3. Predictors are the existing LLM-extracted measurements, including extraction errors, missing values, and ontology limitations. No latent oracle covariate values replaced or repaired them. NLR was not recalculated from other measurements.
   4. Oracle knowledge was used explicitly to restrict the candidate identities. Oracle treatment-effect values were loaded for evaluation only after all six new predictions were frozen. Earlier fold 1 oracle results had already been inspected.
   5. The restriction applies only to the final causal forest. Treatment and outcome nuisance predictions, residuals, and overlap eligibility remain the original shared all-candidate estimates. Confounding adjustment was not restricted to these six candidates.
   6. This analysis uses the same **720 eligible training patients**, **180 eligible held-out patients**, and seeds 120042, 1120042, and 2120042. Forest settings remain 200 trees, honest estimation, minimum leaf size 10, and the original remaining parameters.

3. **Performance**
   1. Values below are means of the three seed-specific metrics. All seeds evaluate the same 180 patients; the table does not pool them as 540 independent patients. Lower RMSE and R-loss are better.

   | Model | CATE RMSE | Correlation | R-loss | Predicted effect SD | Nominal 95% interval coverage |
   |---|---:|---:|---:|---:|---:|
   | Oracle-matched candidates; square-root splits | 0.176965 | 0.461 | 0.188194 | 0.06333 | 60.0% |
   | Oracle-matched candidates; all-feature splits | 0.177991 | 0.454 | 0.187859 | 0.11671 | 64.1% |
   | All candidates; square-root splits | 0.194097 | 0.184 | 0.190295 | 0.02498 | 52.4% |
   | All candidates; all-feature splits | 0.188257 | 0.336 | 0.189796 | 0.03307 | 54.6% |
   | Current joint selection; square-root splits | 0.187692 | 0.296 | 0.190762 | 0.05678 | 52.6% |
   | Permissive union; square-root splits | 0.187511 | 0.338 | 0.189252 | 0.04102 | 50.9% |
   | LLM admission; square-root splits | 0.191657 | 0.286 | 0.191296 | 0.08697 | 56.3% |

   2. The square-root diagnostic's seed-specific RMSE range was **0.176084–0.177429**; the all-feature diagnostic's range was **0.177277–0.178876**.
   3. Within the six-candidate set, all-feature search produced slightly lower R-loss and substantially more predicted variation, but slightly higher mean CATE RMSE. Square-root search had lower RMSE in two of the three seeds. No single split setting dominated every metric.
   4. Oracle effect SD was **0.19510**. Predicted SD reached 32.5% of that with square-root splits and 59.8% with all-feature splits. Considerable compression of heterogeneity remains.
   5. Oracle mean effect was -0.06278; mean predictions were -0.03857 and -0.03897. Bias was +0.02421 and +0.02381. Empirical coverage of the nominal 95% intervals remained low at 60.0% and 64.1%.
   6. The shared constant-effect reference had RMSE 0.196669 and R-loss 0.190734. Both restricted forests improved on that reference.

4. **Interpretation**
   1. Candidate admission materially affects this fold's performance. Supplying the relevant identities yields better CATE estimates than either the broad all-candidates sets or the currently selected smaller sets, using the same extracted measurements and nuisance estimates.
   2. This comparison changes candidate number and relevance together; it does not identify a pure effect of feature count. It also does not establish that an implementable selector can reliably recover this set without oracle knowledge.
   3. Remaining error may reflect measurement quality, representation, nuisance estimation, finite sample size, or forest settings. This experiment does not separate those sources. It is not a perfect-measurement upper bound.
   4. Results are exploratory and restricted to one previously inspected fold. Three seeds characterize some forest randomness, not uncertainty across independent cohorts. The six-candidate oracle-informed filter is not part of the ongoing production comparison.
   5. The prior refresh's extraction-policy revisions, retained earlier fallback records, and the already-inspected synthetic cohort still apply. The unchanged binding-LLM reference additionally retains its documented prompt-batching correction.

5. **Validation and provenance**
   1. Reconstructed full training and held-out design hashes matched the earlier verified original design. The restricted matrices equal the corresponding original encoded columns exactly. Reconstruction uses validated measurement checkpoints to preserve categorical values that ordinary CSV reloading can alter.
   2. All evaluated models share identical held-out row IDs, observed treatment/outcome values, nuisance predictions, eligibility, and replicate identifiers. The nuisance identity is `fc0814beb20fdbc3001ad6bc259896e87dd758f4a72f5d71d86b228d7541ee29`.
   3. Six new models were fitted, with predictions frozen at **2026-09-21T14:16:17+00:00**, before this diagnostic's oracle-effect evaluation. All 471 input files and 12 newly frozen artifact files passed hash checks after evaluation.
   4. R-loss, constant R-loss, mean CATE, and predicted SD were recomputed from saved predictions for all 21 evaluated fits and matched saved fitting metrics within 1e-12. Every new interval is finite and ordered.
   5. The ongoing five-fold comparison's source, parameters, measurements, selections, nuisances, and predictions were left unchanged.

6. **Saved artifacts**
   1. [Input manifest and exact candidate definitions](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-21/fold_1_oracle_modifier_candidates_2026-09-21/input_manifest_2026-09-21.json).
   2. [Metrics by seed](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-21/fold_1_oracle_modifier_candidates_2026-09-21/metrics_by_seed_2026-09-21.csv) and [evaluation summary](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-21/fold_1_oracle_modifier_candidates_2026-09-21/evaluation_2026-09-21.json).
   3. [Frozen predictions](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-21/fold_1_oracle_modifier_candidates_2026-09-21/predictions_frozen_2026-09-21.json), [predictions with oracle effects](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-21/fold_1_oracle_modifier_candidates_2026-09-21/predictions_with_oracle_2026-09-21.csv), and [validation](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-21/fold_1_oracle_modifier_candidates_2026-09-21/validation_2026-09-21.json).
   4. [Prior all-candidates split-search comparison](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-21/fold_1_all_candidates_all_split_features_2026-09-21/fold_1_all_features_report_2026-09-21.md).
