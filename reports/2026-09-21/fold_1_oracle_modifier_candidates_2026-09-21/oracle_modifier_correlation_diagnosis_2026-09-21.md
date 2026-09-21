# Why oracle-matched candidates still give only moderate CATE correlation

Date: September 21, 2026. Interpretation supplement to the fold 1 diagnostic.

1. **A correction to the oracle-modifier interpretation**
   1. The five named modifiers in this simulation modify the treatment coefficient on the **log-odds scale**. The evaluated oracle, `true_ite_prob`, is a **difference in outcome probabilities**.
   2. The actual generating equation is:
      - `P(Y=1 | T=0, C, M) = sigmoid(g(C))`
      - `P(Y=1 | T=1, C, M) = sigmoid(g(C) + h(M))`
      - `tau(C, M) = sigmoid(g(C) + h(M)) - sigmoid(g(C))`
   3. Here `C` contains the five baseline-risk/confounding variables and `M` contains the five explicitly interacting modifiers. Probability-scale treatment effects depend on both sets. The six-candidate forest has only measurements of `M` in its final effect model.
   4. Keeping `C` in the nuisance adjustment does not supply its values to the final forest's split decisions or predictions. The restricted effect model must average over baseline-risk differences it cannot directly distinguish. Thus the earlier modifier-only diagnostic is incomplete as an oracle benchmark for the probability-scale target.
   5. This is verified against the actual saved data: the recorded equations reproduce both potential-outcome probabilities and their difference for all **1,000 patients**, with maximum absolute error 2.22e-16. The generator's conversion is in [generator.py](/data1/ken/pcori_dev/causal-dragonnet-text/synthetic_data/generator.py:1275).
   6. For illustration, holding the treatment's log-odds increment at 1 gives probability differences of 0.132, 0.231, and 0.061 at untreated risks 0.1, 0.5, and 0.9. Those are mathematical examples, not fitted patient estimates. In the actual 180 eligible patients, the 10th and 90th percentiles of untreated risk are 0.141 and 0.848.

2. **Candidate identities are known; measurements remain imperfect**
   1. Among the **180 eligible held-out patients**, NLR is missing for **39 (21.7%)**. Among the 141 with an extracted value, its correlation with the latent NLR is 0.910. This differs from the earlier full-200-patient audit, where 45 values were missing.
   2. The simulation assigns distinct effects to EGFR `Unknown`, brain-metastasis `Unknown`, and histology `Other`. The retained extracted definitions cannot directly preserve these categories.
      - Of 30 oracle EGFR-Unknown patients, the primary candidate records 19 as Wild-type, five as Mutant, and six as missing. The second EGFR candidate also has only two substantive categories.
      - Of 11 oracle brain-metastasis-Unknown patients, ten are extracted as absent and one as missing.
      - Of 23 oracle histology-Other patients, 19 are extracted as Large-cell carcinoma, three as unspecified NSCLC, and one as missing.
   3. Hemoglobin is available for 178 patients, with correlation 0.903 and MAE 0.506 g/dL against the latent value. Availability does not establish measurement accuracy.
   4. These discrepancies combine source-note generation, category definitions, timing, and extraction. They cannot all be attributed to extraction errors without source-level adjudication. The previous [NLR audit](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-21/fold_1_interim_oracle_review_2026-09-21/nlr_extraction_audit_2026-09-21/nlr_extraction_audit_2026-09-21.md) distinguishes some of those mechanisms.

3. **Learning treatment effects remains statistically difficult**
   1. The learner receives one observed binary outcome under one observed treatment per patient. It does not receive each patient's true treatment-effect label.
   2. Only 720 training patients are eligible. With the existing 45% subsample and honest half-split, approximately 162 patients determine each tree's structure and another 162 estimate its node values. The forest collectively uses the 720-patient training pool. Minimum leaf size remains 10. These design choices favor stable estimates while limiting fine distinctions in a modest sample. [EconML causal-forest documentation](https://www.pywhy.org/EconML/_autosummary/econml.grf.CausalForest.html).
   3. Treatment and outcome nuisance estimates were held fixed, rather than replaced by oracle probabilities. On the 180 evaluated patients, their RMSEs against full-latent oracle probabilities are 0.124 for propensity and 0.152 for marginal outcome risk.
   4. The oracle marginal outcome probability for that audit is `(1-e)*P(Y(0)=1) + e*P(Y(1)=1)`, using the saved oracle propensity `e`. These comparisons involve different information sets: extracted measurements versus latent covariates. They establish a gap, not how much CATE error that gap causes.

4. **Prediction compression alone does not explain correlation**
   1. Pearson correlation is unchanged by multiplying predictions by a positive constant or adding a constant. Expanding their standard deviation alone cannot improve it.
   2. This is visible in the diagnostic: changing from square-root to all-feature split search increases predicted SD from 0.063 to 0.117 while correlation changes from 0.461 to 0.454. Individual differences in estimated effects remain imperfectly aligned with the oracle.
   3. A direct check on the saved seed-120042 predictions confirms that multiplying by three and adding 0.1 leaves correlation unchanged to numerical precision.

5. **What remains unresolved**
   1. We have verified the scale/feature-set issue, measurement limitations, and the use of estimated nuisances. We have not quantified each one's contribution to the remaining correlation loss, so none should be presented as the uniquely established dominant cause.
   2. The next separating diagnostics would add the five baseline-risk candidates to the final forest, then compare extracted versus true covariate values, and finally estimated versus oracle nuisance probabilities under fixed forest settings. Such runs would remain explicitly oracle-informed diagnostics outside the main comparison.
   3. No new forests were fitted for this explanation. Existing frozen predictions and the ongoing experiment remain unchanged.
   4. [Machine-readable audit](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-21/fold_1_oracle_modifier_candidates_2026-09-21/oracle_modifier_correlation_diagnosis_2026-09-21.json) records the reconstruction errors, measurement counts, nuisance diagnostics, and source hashes.
