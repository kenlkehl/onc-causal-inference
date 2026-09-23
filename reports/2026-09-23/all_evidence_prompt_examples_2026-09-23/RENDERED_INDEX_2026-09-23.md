# Full rendered prompt examples — September 23, 2026

Current instructions and response schemas with invented miniature inputs. No model calls were made.

[Readable guide](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-23/all_evidence_prompt_examples_2026-09-23/PROMPT_EXAMPLES_2026-09-23.md)

- [01. Discover atomic clinical features](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-23/all_evidence_prompt_examples_2026-09-23/rendered/01_discover.md) — Core pathway
- [02. Audit evidence left uncited](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-23/all_evidence_prompt_examples_2026-09-23/rendered/02_audit_unmapped.md) — Conditional: initial discovery leaves an evidence item uncited
- [03. Consolidate candidate names before extraction](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-23/all_evidence_prompt_examples_2026-09-23/rendered/03_merge_aliases.md) — Core pathway; repeated bounded partitions
- [04. Define a measurement ontology](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-23/all_evidence_prompt_examples_2026-09-23/rendered/04_define_ontology.md) — Core pathway; per discovered feature
- [05. Extract one patient's values](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-23/all_evidence_prompt_examples_2026-09-23/rendered/05_extract_patient.md) — Training and heldout extraction; same template
- [06. Update extraction with the next record chunk](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-23/all_evidence_prompt_examples_2026-09-23/rendered/06_serial_chunk.md) — Conditional: serial long-record extraction
- [07. Extract source-grounded page observations](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-23/all_evidence_prompt_examples_2026-09-23/rendered/07_page_observations.md) — Alternate oversized-record/page path in shared extraction code
- [08. Normalize an out-of-ontology categorical value](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-23/all_evidence_prompt_examples_2026-09-23/rendered/08_map_categories.md) — Conditional: extracted categorical values violate declared categories
- [09. Refine a schema after repeated extraction failures](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-23/all_evidence_prompt_examples_2026-09-23/rendered/09_refine_ontology.md) — Conditional: repeated feature-attributable training extraction failures
- [10. Harmonize mixed numeric and text values](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-23/all_evidence_prompt_examples_2026-09-23/rendered/10_harmonize_values.md) — Conditional: mixed training representations
- [11. Extend a frozen map for new training text values](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-23/all_evidence_prompt_examples_2026-09-23/rendered/11_extend_value_map.md) — Conditional: new training tokens during incremental extraction
- [12. Supervise extraction ontology using aggregate diagnostics](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-23/all_evidence_prompt_examples_2026-09-23/rendered/12_supervise_ontology.md) — Core training extraction/supervision loop
- [13. Check extracted measurements for lossless alias consolidation](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-23/all_evidence_prompt_examples_2026-09-23/rendered/13_post_extraction_aliases.md) — Optional: sequential_consolidation.enabled
- [14. Adjudicate roles in the llm_roles mode](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-23/all_evidence_prompt_examples_2026-09-23/rendered/14_default_roles.md) — Alternative selection branch: llm_roles
- [15. Review multi-model evidence for themes](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-23/all_evidence_prompt_examples_2026-09-23/rendered/15_model_themes.md) — Multi-model selection branch
- [16. Merge theme summaries across batches](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-23/all_evidence_prompt_examples_2026-09-23/rendered/16_merge_themes.md) — Conditional: multi-model themes exceed the request batch limit
- [17. Assign roles using multi-model evidence and themes](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-23/all_evidence_prompt_examples_2026-09-23/rendered/17_model_roles.md) — Multi-model selection branch
- [18. Rank modifier candidates](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-23/all_evidence_prompt_examples_2026-09-23/rendered/18_rank_modifiers.md) — Multi-model branch with cross-validated modifier-count selection
- [19. Interleave ordered modifier lists](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-23/all_evidence_prompt_examples_2026-09-23/rendered/19_merge_rankings.md) — Conditional: ranking spans multiple batches
- [20. Annotate numerically selected roles without changing selection](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-23/all_evidence_prompt_examples_2026-09-23/rendered/20_advisory_roles.md) — Alternative selection branch: independent_tasks, with LLM annotation enabled
- [21. Experimental global cross-fold concept review](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-23/all_evidence_prompt_examples_2026-09-23/rendered/21_cross_fold_concepts.md) — Experimental report-level pass; not integrated into the production pathway
- [22. Repair a response using the exact validation error](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-23/all_evidence_prompt_examples_2026-09-23/rendered/22_validation_repair.md) — Shared transport: schema/validation failure
- [23. Repair an overlong response](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-23/all_evidence_prompt_examples_2026-09-23/rendered/23_length_repair.md) — Shared transport: completion length exceeded

Each rendered Markdown file has a matching `.json` file containing the exact message strings and source metadata.
