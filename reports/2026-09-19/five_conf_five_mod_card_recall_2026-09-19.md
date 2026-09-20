# Five-confounder / five-modifier run: raw-evidence-to-card recall

Analysis date: **2026-09-19**

1. **Result: no complete oracle-concept losses**
   1. All **five oracle confounders and five oracle effect modifiers appear in the readable card text in every outer fold**.
   2. The count of completely missing concepts is **0/25 confounder–fold combinations and 0/25 modifier–fold combinations: 0/50 overall (0%)**.
   3. This concerns the combined **400-card set within each outer fold**. There are five such sets, containing 2,000 cards in total. It is not a claim that each individual architecture retained every concept.
   4. Run examined: `artifacts/research_all_evidence/five_conf_five_mod_nsclc_full/stage2`, completed **September 18, 2026, at 10:45:33 p.m. Eastern** (`2026-09-19T02:45:33Z`). The earlier `stage2_pre_roles_refactor` run and the separate August 31 cohort were not used.

2. **Number of cards containing each concept**
   1. Counts below use only readable `representative_evidence[].text`, with clinical aliases and context checks. A card counts once per concept, even if several of its excerpts contain that concept. One card can contain multiple concepts.
   2. These are reproducible lexical/context counts, not an exhaustive semantic annotation of every card. The zero-dropout conclusion is supported by explicit examples of each concept in every fold.

   | Oracle role | Concept | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
   |---|---|---:|---:|---:|---:|---:|
   | Confounder | Age | 115 | 116 | 125 | 113 | 119 |
   | Confounder | Sex | 72 | 80 | 83 | 84 | 79 |
   | Confounder | ECOG performance status | 76 | 82 | 72 | 75 | 87 |
   | Confounder | Creatinine clearance | 28 | 33 | 30 | 23 | 26 |
   | Confounder | Prior platinum therapy | 48 | 41 | 52 | 54 | 57 |
   | Effect modifier | Histology | 117 | 108 | 108 | 110 | 111 |
   | Effect modifier | EGFR mutation status | 54 | 57 | 51 | 54 | 50 |
   | Effect modifier | Neutrophil-to-lymphocyte ratio (NLR) | 13 | 14 | 17 | 13 | 11 |
   | Effect modifier | Brain metastases | 51 | 48 | 51 | 53 | 53 |
   | Effect modifier | Hemoglobin | 61 | 63 | 63 | 44 | 58 |

3. **How the comparison was defined**
   1. Oracle names and roles come from the saved synthetic-generation metadata. They were used only for this retrospective audit, not supplied to discovery, card selection, extraction, or estimation.
   2. A complete loss would require evidence of a concept in the raw handoff and **zero readable card excerpts containing it** in that outer fold. Every concept instead has positive card coverage in every fold.
   3. Raw evidence and prompt visibility are different objects. The compiler assigns members to clusters and records their lineage, but only selected, sometimes shortened, excerpts are visible in the cards. Merely appearing in a member list or a card's metadata was not counted as survival.
   4. The saved compiler creates card text by selecting and truncating source excerpts; it does not use an LLM to invent card content. The audit also checks that each fold's saved interpretation-input packets contain exactly the same 400 card contents as the compiled card file.
   5. Ambiguities were handled explicitly:
      1. **EGFR:** required mutation/molecular context, such as mutation, wild type, exon, or variant language. A bare `eGFR` renal-function value is insufficient.
      2. **Prior platinum:** required platinum/cisplatin/carboplatin language accompanied by treatment-history cues. A drug name by itself is insufficient for the qualified counts.
      3. **Creatinine clearance:** searched clearance, CrCl/ClCr, and Cockcroft–Gault language; generic serum creatinine alone is insufficient.
      4. **NLR:** required NLR or an explicit neutrophil-to-lymphocyte-ratio phrase. Separate neutrophil and lymphocyte counts alone were not counted.
      5. **Brain metastases:** required brain/CNS/intracranial language close to metastasis language. A generic brain scan alone was not counted.
   6. The audit script retains card IDs, short source excerpts, matching rules, and source identities so individual classifications can be reviewed.
   7. Raw-source checks found all ten concepts in each fold's full-outer-training handoff row: rows **1, 7, 13, 19, and 25**, respectively. These were read through the compiler's scientific-text adapters. This confirms all **50 raw-presence checks**; scanning stopped once every presence query had a witness, so no raw per-concept frequency is claimed.
   8. The five card files were freshly hashed before the oracle-label audit, and all 2,000 interpretation-input packets were checked against their saved card contents. The raw-handoff fingerprint is the compiler's previously recorded fingerprint; a fresh hash of the entire 3.7 GB handoff was not completed. No production artifacts or fitted models were changed.

4. **What compression did change**
   1. The compiler's saved totals are approximately **4.3–4.5 million raw evidence occurrences per outer fold**, including repeated evidence across contexts and views. These become approximately **693,000–744,000 exact members**, then 400 cards per fold.
   2. Visibility is uneven. NLR is the sparsest oracle concept: **11–17 cards per fold**, or **2.75–4.25%** of the cards. Creatinine clearance appears in **23–33 cards**, or **5.75–8.25%**. Age and histology each appear in more than 100 cards in every fold.
   3. These percentages measure visibility, not model attention, clinical importance, statistical strength, or the share of raw evidence retained. Cards overlap in their concepts.

5. **What the zero-loss result does and does not establish**
   1. The 400-card compression boundary did **not completely remove any of these ten oracle clinical concepts** from any fold's interpretation inputs.
   2. This does not establish preservation of every informative raw passage, rare category, numerical association, effect contrast, or architectural source. It also does not establish that the concept was equally salient before and after compression.
   3. It does not establish that Stage 2 subsequently discovered the correct measurement, preserved the baseline timing qualifier, extracted it correctly, assigned its true causal role, or estimated its effect accurately. Those are later boundaries requiring separate checks.
   4. In particular, the labels “baseline NLR,” “baseline hemoglobin,” and “prior platinum therapy” impose temporal requirements beyond simply recognizing the clinical concept. The presence audit should not be interpreted as patient-level validation of those requirements.

6. **Supporting files**
   1. [Machine-readable counts, card IDs, contextual examples, and raw witnesses](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-19/five_conf_five_mod_card_recall_2026-09-19.json).
   2. [Read-only audit script](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-19/five_conf_five_mod_card_recall_2026-09-19.py).
   3. [Frozen card-file identities](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-19/five_conf_five_mod_card_sets_frozen_2026-09-19.json).
   4. [Saved compiler summary and raw-handoff fingerprint](/data1/ken/pcori_dev/causal-dragonnet-text/artifacts/research_all_evidence/five_conf_five_mod_nsclc_full/stage2/evidence_compilation/summary.json).
   5. Card text: [fold 1](/data1/ken/pcori_dev/causal-dragonnet-text/artifacts/research_all_evidence/five_conf_five_mod_nsclc_full/stage2/evidence_compilation/outer_001/cards.jsonl), [fold 2](/data1/ken/pcori_dev/causal-dragonnet-text/artifacts/research_all_evidence/five_conf_five_mod_nsclc_full/stage2/evidence_compilation/outer_002/cards.jsonl), [fold 3](/data1/ken/pcori_dev/causal-dragonnet-text/artifacts/research_all_evidence/five_conf_five_mod_nsclc_full/stage2/evidence_compilation/outer_003/cards.jsonl), [fold 4](/data1/ken/pcori_dev/causal-dragonnet-text/artifacts/research_all_evidence/five_conf_five_mod_nsclc_full/stage2/evidence_compilation/outer_004/cards.jsonl), [fold 5](/data1/ken/pcori_dev/causal-dragonnet-text/artifacts/research_all_evidence/five_conf_five_mod_nsclc_full/stage2/evidence_compilation/outer_005/cards.jsonl).
   6. [Oracle feature names and roles](/data1/ken/pcori_dev/causal-dragonnet-text/synthetic_data/example_synthetic_datasets/five_confounders_five_effect_modifiers_nsclc_with_structured/metadata.json).
