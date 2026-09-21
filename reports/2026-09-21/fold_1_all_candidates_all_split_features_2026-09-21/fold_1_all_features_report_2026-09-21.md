# Fold 1: all candidates with all features available at each split

Date: September 21, 2026. Separate user-requested exploratory follow-up.

1. **Answer**
   1. Allowing every encoded feature at each split improved the all-candidates forest in this fold. Mean oracle CATE RMSE decreased from 0.1941 to 0.1883, a **3.01% reduction**, and improved in all three paired seeds.
   2. Mean correlation with oracle effects increased from 0.184 to 0.336. Predicted effect spread increased by 32.4%, from 0.02498 to 0.03307, but remained far below oracle SD 0.19510.
   3. This supports limited feature exposure at splits as one contributor to weak all-candidates performance. It does not establish candidate count as the sole cause or resolve the compressed heterogeneity.

2. **What changed and what was held fixed**
   1. Candidate set: the original all-candidates arm, with **352 clinical features and 1,096 encoded columns**, including all five directly mapped oracle modifier concepts according to the earlier post-hoc lineage review. No candidate selection was rerun.
   2. Only estimator parameter changed: `max_features="sqrt"` to `max_features=1.0`. The square-root setting requests approximately 33 columns per split; the new setting makes all 1,096 available. Constant columns cannot yield useful splits, and the splitter can inspect beyond a nominal subset to find a valid partition.
   3. Training and evaluation: the same 720 eligible training patients and 180 eligible held-out patients from fold 1; unchanged definitions, extraction values, encoding, overlap eligibility, nuisance predictions, treatment and outcome residuals.
   4. Forest settings: 200 trees, honest fitting, minimum leaf size 10, unrestricted maximum depth, inference enabled, subforest size 4, one worker, and the installed defaults including `max_samples=0.45` and `criterion="mse"`.
   5. Paired seeds: 120042, 1120042, and 2120042. The three seeds reuse the same 180 held-out patients; they are not three independent cohorts. Summary metrics are averages across seeds, not predictions from an ensemble of the three forests.
   6. The ongoing five-fold, four-way comparison retains its original settings. This separate follow-up did not change its source, measurements, admission decisions, nuisance fits, predictions, or execution.

3. **Held-out results**
   1. Lower RMSE and R-loss are better. Correlation measures alignment with oracle heterogeneity. Spread is the population standard deviation of predicted CATE. Coverage is the observed fraction of oracle effects inside the nominal 95% forest intervals.

   | Model | CATE RMSE | Correlation | R-loss | Predicted SD | 95% interval coverage |
   |---|---:|---:|---:|---:|---:|
   | All candidates; square-root split search | 0.194097 | 0.184 | 0.190295 | 0.02498 | 52.4% |
   | All candidates; all-feature split search | 0.188257 | 0.336 | 0.189796 | 0.03307 | 54.6% |
   | Current joint selection; square-root search | 0.187692 | 0.296 | 0.190762 | 0.05678 | 52.6% |
   | Permissive union; square-root search | 0.187511 | 0.338 | 0.189252 | 0.04102 | 50.9% |
   | LLM admission; square-root search | 0.191657 | 0.286 | 0.191296 | 0.08697 | 56.3% |

   2. The all-feature setting lowered mean R-loss by **0.26%**; two seeds improved and one worsened. Its mean R-loss was about 0.49% below the shared constant-effect reference.
   3. The constant-effect reference has CATE RMSE 0.196669 and R-loss 0.190734. The new all-feature forest reduced RMSE by 4.28% relative to that reference.
   4. The new result approached the current joint and permissive-union arms, but its mean RMSE remained slightly higher than theirs. This fold does not show an advantage for using all candidates over those smaller sets.
   5. Predicted effect SD remained only 17.0% of oracle SD. Mean predicted CATE was -0.03865, compared with oracle mean -0.06278; average bias was +0.02414. Empirical interval coverage remained low at 54.6%.
   6. Paired results:

   | Seed | Square-root RMSE | All-feature RMSE | RMSE change | Square-root R-loss | All-feature R-loss |
   |---|---:|---:|---:|---:|---:|
   | 120042 | 0.190927 | 0.186044 | -0.004883 | 0.190829 | 0.189824 |
   | 1120042 | 0.194622 | 0.189069 | -0.005553 | 0.189996 | 0.189198 |
   | 2120042 | 0.196743 | 0.189659 | -0.007084 | 0.190060 | 0.190366 |

4. **Verification and provenance**
   1. The original design fingerprints matched exactly for all three seeds. Replaying the original square-root forests reproduced all 180 predictions and both interval endpoints exactly in every seed: maximum absolute difference **0.0**.
   2. The saved combined held-out CSV does not preserve all categorical Python types on ordinary reload. To reproduce the original matrices, this analysis reused the same 125-feature standard panel, reconstructed the other 227 features from their 200 validated patient checkpoints, and reapplied the original frozen harmonization rules. No extraction request or measurement correction was made.
   3. All three new fits were completed and their prediction hashes frozen at **2026-09-21T14:05:19+00:00** before this follow-up's evaluation loaded oracle values. Input provenance covers 461 files. Shared eligible row IDs, observed values and nuisance predictions match the original all-candidates arm exactly.
   4. Every original and new R-loss, constant R-loss, mean predicted CATE, and predicted SD was independently recomputed from saved predictions and matched the saved fitting metrics within 1e-12. Input and frozen prediction hashes were rechecked after evaluation.
   5. Runtime versions: EconML 0.16.0, scikit-learn 1.6.1, NumPy 2.3.5, pandas 3.0.5.

5. **Interpretation and limits**
   1. This was requested after the fold 1 oracle review. It is a post-hoc exploratory sensitivity analysis, not an untouched confirmation or a prespecified fifth arm. Oracle values were excluded from the new fitting computations, but prior oracle results informed the motivation for this follow-up.
   2. Three paired seeds on one fold provide evidence about this setting on these data; they do not establish general superiority or statistical significance. Changing the feature-search setting changes subsequent tree structures as well as which variables are considered.
   3. All candidates still share the existing extraction limitations, representations and regularized nuisance estimates. This analysis does not distinguish their remaining contributions from limited sample size, honest-tree splitting, or other forest settings.
   4. The measurement refresh previously changed transport/repair policies in two documented revisions and retained earlier fallback records. The binding LLM arm also uses the documented prompt-budget batching correction. These operational facts and the already-inspected synthetic cohort limit broader interpretation; this is not a controlled extractor comparison.

6. **Saved artifacts**
   1. [Input manifest](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-21/fold_1_all_candidates_all_split_features_2026-09-21/input_manifest_2026-09-21.json): parameters, versions, input hashes and design hashes.
   2. [Baseline reproduction checks](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-21/fold_1_all_candidates_all_split_features_2026-09-21/reproduction_validation_2026-09-21.json) and [final validation](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-21/fold_1_all_candidates_all_split_features_2026-09-21/validation_2026-09-21.json).
   3. [Frozen new predictions](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-21/fold_1_all_candidates_all_split_features_2026-09-21/predictions_frozen_2026-09-21.json), with three seed-specific prediction files under `fits/`.
   4. [Metrics by seed](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-21/fold_1_all_candidates_all_split_features_2026-09-21/metrics_by_seed_2026-09-21.csv), [evaluation summary](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-21/fold_1_all_candidates_all_split_features_2026-09-21/evaluation_2026-09-21.json), and [predictions with oracle values](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-21/fold_1_all_candidates_all_split_features_2026-09-21/predictions_with_oracle_2026-09-21.csv).
   5. [Earlier fold 1 review](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-21/fold_1_interim_oracle_review_2026-09-21/fold_1_interim_oracle_report_2026-09-21.md) and [NLR measurement audit](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-21/fold_1_interim_oracle_review_2026-09-21/nlr_extraction_audit_2026-09-21/nlr_extraction_audit_2026-09-21.md).
