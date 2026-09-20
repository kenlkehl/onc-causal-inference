# Five-confounder/five-modifier run: where features were lost before the causal forest

Date: September 19, 2026.

1. **Answer and scope**
   1. The hard feature exclusions occurred at the **Stage 2 statistical task-selection step**, after discovery, consolidation, and extraction and before fitting the causal forest. For direct effect-modifier inputs, the binding gate was the **joint grouped elastic-net R-loss model**.
   2. This audit covers the latest completed [five-confounder/five-modifier Stage 2 run](/data1/ken/pcori_dev/causal-dragonnet-text/artifacts/research_all_evidence/five_conf_five_mod_nsclc_full/stage2/complete.json), completed September 18, 2026, at 10:45:33 p.m. Eastern (September 19 at 02:45:33 UTC).
   3. The preceding [card-recall audit](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-19/five_conf_five_mod_card_recall_2026-09-19.md) found all ten oracle concepts represented among the 400 evidence cards in every fold. The present audit follows the resulting measurements to the forest inputs.
   4. Some measurement detail was already lost when definitions were formed or revised. Therefore, complete feature exclusion and loss of information within a retained feature are separate findings.

2. **How the binding selection works in this run**
   1. The saved selection policy is `independent_tasks`. Treatment prediction, outcome prediction, and effect prediction have separate selection results. A nonzero feature group in at least one of the five inner folds selects that feature for that task.
   2. The final routing is:
      - **X — effect input:** selected by the joint effect model in at least one inner fold; available to the forest for predicting variation in treatment effects.
      - **W — adjustment only:** no effect-selection votes, but at least one treatment- or outcome-selection vote; available for adjustment, without being a direct effect-prediction input.
      - **Excluded:** zero votes for all three tasks; absent from the final forest inputs.
   3. Every candidate is eligible for effect screening. The effect candidate list is not restricted to features selected by the treatment or outcome screens. Residual construction uses all candidates with independent regularization within the relevant folds.
   4. Candidate-by-candidate effect screening and its top-N rankings are diagnostic. Positive results there cannot override zero votes from the joint effect model. The LLM's role annotations are also advisory and cannot retain or reroute a feature.
   5. The forest subsequently fits its own nuisance models on its supplied X and W inputs. The exclusions documented here have already happened by that point.
   6. These rules are implemented in [task-wise routing](/data1/ken/pcori_dev/causal-dragonnet-text/oci/inference/stage2_taskwise_policy.py:28), [selection-report finalization](/data1/ken/pcori_dev/causal-dragonnet-text/oci/inference/stage2_taskwise_policy.py:144), and [advisory annotation](/data1/ken/pcori_dev/causal-dragonnet-text/oci/inference/stage2_taskwise_annotation.py:27).

3. **Where each direct oracle modifier ended up**
   1. All five direct modifier measurements survived into statistical selection in every outer fold. Their candidate-wise effect tests completed successfully in all 125 combinations of modifier, outer fold, and inner fold.
   2. The final routing was as follows. These rows track explicit measurements of each concept, including reviewed aliases; related biomarkers and lesion-count proxies are not counted as the same measurement.

      | Direct oracle modifier | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
      |---|---|---|---|---|---|
      | Histology | Excluded | W | Excluded | Excluded | Excluded |
      | EGFR mutation status | W | Excluded | Excluded | Excluded | W |
      | Neutrophil-to-lymphocyte ratio (NLR) | X | W | X | X | X |
      | Brain-metastasis status | Excluded | Excluded | Excluded | Excluded | X |
      | Hemoglobin | X | W | Excluded | Excluded | Excluded |

   3. Across the 25 modifier–outer-fold combinations, **6 (24%) reached X**, **5 (20%) reached W only**, and **14 (56%) were excluded entirely**. Thus 19 of 25 direct modifier measurements could not directly drive the forest's effect predictions.
   4. Histology and EGFR never entered X. NLR entered X in four folds, brain-metastasis status in one, and hemoglobin in one. In fold 2, none of these five direct modifier measurements entered X; the two effect inputs were `anxiety_presence` and `ttf1_expression`.

4. **Examples locating the loss at the joint effect gate**
   1. **Fold 3, brain-metastasis status:** candidate-wise held-out R-loss improvement was positive in all five inner folds: 0.004828, 0.001183, 0.006312, 0.000089, and 0.001475. It ranked in the top-N set in two folds. Nevertheless, the joint effect model selected it in zero folds; treatment and outcome models also selected it in zero folds. It was therefore excluded entirely. The advisory LLM called it an effect modifier, but that annotation could not change the result.
   2. **Fold 1, histology:** two candidate-wise top-N votes and an advisory effect-modifier annotation, but zero joint-effect, treatment, and outcome votes. It was excluded entirely.
   3. **Fold 2, NLR:** two candidate-wise top-N votes, but zero joint-effect votes. One outcome-selection vote retained it as W only. Its values could support adjustment but could not directly define the forest's conditional effect predictions.
   4. The discrepancy does not establish that every exclusion was statistically unreasonable. EGFR had zero candidate-wise top-N votes in every fold, and its individual held-out gains were often negligible or negative. Positive individual gains also do not establish additional benefit after other variables enter a joint model.
   5. Across the 25 joint inner-fold effect models, 11 selected no feature groups at all. All 25 reported successful status and solver convergence. Locating the exclusion is conclusive; separating the contributions of regularization, correlated alternatives, measurement error, and limited signal would require controlled refits.

5. **What happened to the confounders**
   1. The corresponding confounder representations had the following routing. Renal-function and prior-treatment rows require the measurement qualifications in item 6.

      | Oracle confounder concept | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
      |---|---|---|---|---|---|
      | Age | W | W | W | W | W |
      | Sex | Excluded | W | Excluded | W | W |
      | ECOG performance status | W | W | W | W | W |
      | Creatinine clearance / renal-function representation | W | W | W | W | W |
      | Prior platinum / prior-chemotherapy representation | X | W | Excluded | W | X |

   2. Complete exclusions were **sex in folds 1 and 3** and **prior-platinum history in fold 3**. Reviewed sex/gender aliases did not rescue the two sex exclusions.
   3. Prior platinum entering X in folds 1 and 5 is a model-routing outcome, not evidence that its oracle causal role changed. Likewise, W membership does not prove a variable is a true confounder.

6. **Earlier losses of measurement detail**
   1. **EGFR and brain metastases:** the initial consolidated definitions were binary in every fold. Neither included the oracle's explicit `Unknown` category. This loss of an explicit category was already present before statistical selection. Null handling may retain some related information, but it is not an explicit reconstruction of the oracle categories.
   2. **Prior platinum:** folds 1, 2, 3, and 5 represented prior platinum exposure as yes/no, collapsing the oracle's treatment-history categories of None, Adjuvant, First-line, and Multiple. Fold 4 used a broader prior-chemotherapy-regimen feature. Retaining these features would not by itself recover the original distinctions.
   3. **Creatinine clearance:** in folds 3 and 5, consolidation combined CrCl and eGFR into a single measurement whose definition allowed either value. In fold 2, definition supervision changed continuous clearance to ordinal kidney-function stages. These are broadening or coarsening of the measurement rather than complete concept deletion.
   4. No feature IDs disappeared between the initial consolidated definitions and the definitions supplied to statistical selection. The optional second consolidation pass was disabled. Identity preservation does not establish that extraction was accurate, complete, or faithful to baseline timing; patient-level measurement accuracy was not assessed here.

7. **Overall feature counts and remaining interpretation limits**
   1. The counts below are feature definitions, not encoded columns. Discovery counts include candidates subsequently merged into shared definitions.

      | Outer fold | Discovered candidates | Consolidated / entering selection | Final forest inputs | X | W |
      |---|---:|---:|---:|---:|---:|
      | 1 | 934 | 352 | 130 | 28 | 102 |
      | 2 | 866 | 304 | 121 | 2 | 119 |
      | 3 | 988 | 351 | 30 | 2 | 28 |
      | 4 | 1,105 | 419 | 130 | 5 | 125 |
      | 5 | 886 | 318 | 139 | 35 | 104 |

   2. Excluding a direct measurement does not establish that all related information disappeared. For example, fold 1 retained brain-lesion count in X, and fold 2 retained TTF-1 expression. Such features can carry related information without reproducing the original oracle variable.
   3. This trace establishes which measurements could reach the forest and where their routing was determined. It does not measure how much each exclusion changed CATE/ITE accuracy, or how faithfully retained proxies substituted for missing direct measurements.

8. **Audit provenance and reproducibility**
   1. This is a post-hoc inspection of saved run artifacts. No model was refitted, no LLM was called, and no patient-level oracle values were used. Oracle metadata was used only to label the audit after freezing the analyzed run artifacts.
   2. The [detailed selection trace](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-19/five_conf_five_mod_selection_trace_2026-09-19.json) records feature IDs, original and screened definitions, discovery lineage, all task votes, candidate-wise results, advisory annotations, final routing, and SHA-256 hashes for 43 source files.
   3. The [audit script](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-19/five_conf_five_mod_selection_trace_2026-09-19.py) checks that every initial feature enters selection, that the final feature set exactly matches retained numerical decisions, and that final forest roles agree with those decisions.
