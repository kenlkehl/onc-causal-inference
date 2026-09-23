# Fold 1: assessment of cross-fold modifier concepts — 2026-09-23

1. **Outcome**
   1. The completed LLM pass selected **21 representative modifiers from 21 retained concepts**, preserving all **189 confounders**. It assigned the 217 candidates to 30 groups: 21 retained, 3 uncertain, and 6 excluded.
   2. All five original top-64 prefixes were preserved when extending the rankings to 100. The five top-100 lists contained 217 distinct candidates; 15 occurred in all five lists.
   3. **Direct oracle modifier recovery remained 1/5: NLR.** All five oracle modifier variables had a corresponding candidate in the review inputs.
   4. The pass recognized an anemia concept containing hemoglobin and hematocrit, then selected hematocrit. That is related concept/proxy retention, not direct hemoglobin recovery.
   5. Histology, EGFR status, and brain-metastasis presence were placed in excluded groups. TTF-1 was selected separately as a lung-adenocarcinoma marker; this does not count as direct histology recovery.
   6. Fifteen selected variables overlap the previous final 64; six are new. No causal-effect estimator was fitted for this list, so its ITE accuracy is unknown.

2. **Assessment of the LLM reasoning**
   1. **Some stated reasons conflict with the supplied recurrence data.** The model described histologic markers other than TTF-1 as having low recurrence, but CK5/6 ranked 29, 9, 7, 28, and 4 in the five folds. Histology itself appeared in four of five lists. Exclusion can still be defensible for redundancy or weak effect evidence; lack of recurrence does not describe these examples.
   2. The same issue appears in other excluded groups: CDKN2A status and blood urea nitrogen each occurred in all five lists. Across excluded groups, 20 candidates occurred in at least four folds. Frequency alone is not an inclusion rule, but factual recurrence claims should match the inputs.
   3. **One excluded group collects 99 heterogeneous candidates** under “Patient Baseline and Comorbidities.” This is a broad residual category rather than a specific underlying modifier concept.
   4. The first response accounted for only 67 candidates and already chose the final 21 representatives. The selected set remained identical while repairs increased coverage to 211 and then 217. Structural completeness therefore does not demonstrate that the extra candidates received a substantive reassessment.
   5. The selection still includes low-recurrence concepts: pneumonia appeared in one fold and glucose in two. No hard frequency threshold was imposed, so these are allowed choices, but they do not provide strong cross-fold recurrence.
   6. A misleading upstream name deserves care: `biomarker_fgfr_status` is defined as numeric GFR in ml/min. Its renal-group assignment follows that frozen definition; the name alone should not be used to interpret it as a tumor FGFR biomarker.
   7. This produced a more compact shortlist, with some useful proxy consolidation. The inconsistent rationales and broad exclusion categories limit confidence in it as a stable concept-selection method.

3. **Selected measurements**

   | Concept | Representative | Concept folds | Representative folds | Ranks in folds 1–5 |
   | --- | --- | ---: | ---: | --- |
   | Systemic Inflammation (NLR) | neutrophil_to_lymphocyte_ratio | 5/5 | 5/5 | 2, 1, 2, 1, 1 |
   | Renal Function Status | creatinine_level | 5/5 | 5/5 | 7, 12, 16, 93, 53 |
   | KRAS G12C Variant Burden | kras_g12c_variant_allele_frequency | 4/5 | 4/5 | 22, 40, 9, outside 100, 55 |
   | Lung Adenocarcinoma Marker | ttf1_expression | 5/5 | 5/5 | 48, 2, 8, 9, 38 |
   | Prior Platinum Exposure | prior_platinum_chemotherapy | 5/5 | 4/5 | 24, 3, outside 100, 15, 46 |
   | Distant Visceral Metastasis | liver_metastasis_presence | 5/5 | 5/5 | 46, 92, 32, 5, 69 |
   | Intracranial Hemorrhage Burden | epidural_hemorrhage | 5/5 | 4/5 | outside 100, 29, 1, 21, 15 |
   | Dyspnea Severity | dyspnea_severity | 5/5 | 5/5 | 1, 17, 5, 23, 50 |
   | Cellular Proliferation Index | ki67_proliferation_index | 5/5 | 5/5 | 42, 4, 30, 89, 7 |
   | White Blood Cell Count | white_blood_cell_count | 5/5 | 5/5 | 4, 11, 6, 66, 5 |
   | MET Gene Amplification | met_amplification_status | 5/5 | 5/5 | 26, 56, 10, 55, 42 |
   | Cognitive Confusion Status | confusion_present | 4/5 | 4/5 | 25, outside 100, 26, 29, 17 |
   | Chronic Lung Disease | lung_disease_diagnosis | 5/5 | 3/5 | 27, 26, 37, outside 100, outside 100 |
   | Pulmonary Nodularity | pulmonary_nodules_presence | 4/5 | 3/5 | 6, 73, outside 100, outside 100, 20 |
   | Radiation Therapy Burden | radiation_therapy_session_count | 5/5 | 3/5 | outside 100, 23, 72, 24, outside 100 |
   | Lymph Node Dissection Status | lymph_node_dissection_status | 4/5 | 4/5 | 63, 45, 94, outside 100, 33 |
   | Anemia Burden | hematocrit | 4/5 | 3/5 | outside 100, 36, 48, outside 100, 2 |
   | Targeted Therapy Exposure | targeted_therapy_use | 4/5 | 3/5 | 83, outside 100, outside 100, 58, 34 |
   | Baseline Glucose Level | glucose_level | 2/5 | 2/5 | 21, outside 100, outside 100, 3, outside 100 |
   | Pneumonia Burden | pneumonia_diagnosis | 1/5 | 1/5 | 18, outside 100, outside 100, outside 100, outside 100 |
   | Skeletal Compression Burden | skeletal_compression_fracture | 5/5 | 3/5 | 17, outside 100, 51, 33, outside 100 |


4. **Execution and verification**
   1. Baseline code, tests, and the prior completed experiment were committed on main as `568ef7a` before this experiment was created.
   2. Gemma 4 31B reviewed the union with definitions, fold ranks, and fold-specific numerical evidence. The top-100 extension reused saved extraction and numerical results; no extraction or statistical model was rerun.
   3. Five completed concept responses were saved. After the fourth, an in-progress repair was canceled and resumed from its checkpoint with all outstanding validation issues in one message. The original prompt, selection criteria, and total time/response budget were preserved.
   4. Final checks verified complete unique candidate coverage, representative membership, evidence citations, retained confounders, and all selection-freeze hashes. Oracle identity matching occurred after the selection freeze.
   5. The raw LLM decisions and their generated report remain unchanged. This assessment is separate from those decisions and includes no additional model selection or fitting.
   6. This is exploratory work on a previously examined outer fold. Prior R-loss results do not validate this new concept-selection procedure.

5. **Artifacts**
   1. [Full LLM report and every concept decision](REPORT_2026-09-23.md).
   2. [Selected modifier table](selected_modifiers.csv) and [exported feature definitions](selected_definitions.json).
   3. [Independent checks and response history](assessment_checks_2026-09-23.json).
   4. [Oracle identity audit](oracle_recovery.json), [selection freeze](selection_frozen.json), and [protocol](PROTOCOL_2026-09-23.md).
