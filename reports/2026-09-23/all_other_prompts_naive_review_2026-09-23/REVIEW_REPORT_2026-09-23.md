# All-evidence prompt review and discovery adoption

**Superseded prompt proposals:** See the [September 23 plain-language revision and Qwen review](../all_evidence_plain_language_qwen_review_2026-09-23/REVIEW_REPORT_2026-09-23.md). This document preserves the earlier review and discovery implementation record. Its eligibility/process explanations are superseded in the new proposals.

Date: September 23, 2026.

The approved **01_discover** prompt and its Python caller/validator changes are implemented. **02–23 are reviewed replacement proposals only**, following the user's request to review replacements before integrating them. No extraction, selection, or causal-estimation experiment was rerun for this exercise.

## 1. What changed in production

1. Discovery sends one evidence card's readable excerpts per call, with the approved self-contained prose instructions.
2. The model returns clinical `name`, `description`, `basis`, and `uncertainty`. It does not manage source IDs, excerpt indices, machine names, or citation lists.
3. Python attaches the card's trusted provenance, normalizes names, preserves distinct candidates when names collide, validates the response, and records empty-card dispositions.
4. The prompt/response version changed. Old discovery results cannot silently satisfy the new fingerprint. Existing completed definitions are preserved; use a fresh Stage 2 output directory when regenerating definitions.
5. The conditional recall audit still uses its existing production prompt/validator. Its replacement is proposal 02 below. Other prompt families and the numerical estimation pipeline are unchanged.

Production files: [approved prompt](../../../oci/inference/stage2_discovery_prompt.py), [caller and validator](../../../oci/inference/plain_handoff_stage2.py), [behavioral/integration tests](../../../tests/test_plain_handoff_stage2.py), and [workflow documentation](../../../docs/all_evidence_workflow.md).

Validation: **245 tests passed, 11 warnings**, covering discovery, trusted provenance, naming collisions, malformed responses, recall-audit integration, extraction, checkpoints/resume, and runtime regressions. The adopted system message is the approved v3 text. One-card requests can increase request count relative to batching; this is an explicit execution tradeoff, not a measured improvement in recovery.

## 2. How the remaining prompts were reviewed

1. Each of the 22 original variants was given to a separate **GPT 5.6 Sol** subagent with no inherited conversation history. The agent could read only its assigned messages file, without repository code, other reviews, study results, or oracle information.
2. Reviewers explained the task, asked consequential clarification questions, and identified bookkeeping. They were not required to find defects or agree with the prompt author's criticism.
3. Replacements supplied purpose, scope, input meaning, decision authority, and an explicit output contract. Python responsibilities are documented beside each replacement.
4. Each replacement received another fresh GPT 5.6 Sol review and a miniature task execution. Where those checks raised a remaining question, the prompt was clarified and the recipient received a targeted follow-up. Those follow-ups retained that recipient's history and are not counted as fresh reviews.
5. Example responses were checked for JSON structure and task-specific invariants. This is a qualitative comprehension exercise on invented examples, **not an evaluation of clinical extraction accuracy, variable recovery, ranking quality, or causal estimation**.

The inventory covers all 23 generative prompt variants in the previously documented all-evidence path: core, conditional, alternative, and experimental. Stage 1's current all-evidence route uses numerical/text-encoder modeling rather than an additional generative chat prompt. Final count/architecture selection and ITE estimation are numerical procedures. The numbered prompts are therefore not 23 unconditional steps in one run.

## 3. Main findings

1. **Scope was often implicit.** A recipient needs to know whether it is naming variables, defining measurements, extracting values, grouping themes, assigning roles, ranking candidates, or merely explaining an already fixed decision.
2. **Eligibility and time handling need explicit ownership.** Extraction proposals say when the caller has already selected eligible records. A dated lab result does not lend its date to an unrelated CT statement. The latest/earliest proposals explicitly retain the existing dated-observation preference.
3. **Unavailable evidence must differ from negative evidence.** The statistical proposals explain coverage, support criteria, overlapping splits, effect scale, and the limits of each modeling family. They do not silently interpret an empty array as a failed screen.
4. **Some instructions conflicted with their output contracts.** The theme-only prompt inherited role-assignment instructions. Theme compression could request fewer themes than the number of unrelated clinical concepts. Annotation-only work still looked like role adjudication. The proposals give each step only its actual decision authority.
5. **Provenance and mechanics belong in Python.** The proposals remove opaque IDs, row/chunk indices, character-offset arithmetic, numerical ranks, and repeated source-ID lists. Semantic decisions still need references: grouping and ranking use unique clinical names. Python cannot decide clinical equivalence or quote-to-finding meaning by simply manipulating identifiers.
6. **Examples need review too.** Some original invented statistical fixtures contained inconsistent aggregate votes or duplicated evidence arrays. These were corrected in the new examples. They are not evidence that the corresponding real experiment outputs were inconsistent.
7. **A confident comprehension review is insufficient by itself.** The first replacement responses for 05, 22, and 23 interpreted an intended `values` wrapper as a flat feature-keyed object. The new contract explicitly adopts the simpler flat object and moves software wrappers to Python. The original responses remain in the record; a further fresh review of 05 and targeted repair follow-ups verified the clarified contract.

## 4. Replacement catalog

Each linked replacement contains the literal messages, its proposed Python responsibilities, and the changes from the original. Original and revised reviews are preserved separately. JSON files beside each Markdown prompt contain the exact proposed messages.

### 4.1 Discovery and measurement definition

| Prompt | Main clarification | Replacement | Original review | Fresh replacement check |
| --- | --- | --- | --- | --- |
| 02 Recall audit | Independently re-examine one empty card; reuse discovery scope without item tracking. | [02](revised/02_audit_unmapped.md) | [Questions](reviews/02_original.md) | [Check](checks/02_revised.md) |
| 03 Alias consolidation | Same scalar target, not related concepts; omit unchanged variables; Python retains them. | [03](revised/03_merge_aliases.md) | [Questions](reviews/03_original.md) | [Check](checks/03_revised.md) |
| 04 Measurement definition | Caller-supplied record scope, repeat/conflict policy, units/nulls, and no unsupported stability claim. | [04](revised/04_define_ontology.md) | [Questions](reviews/04_original.md) | [Check](checks/04_revised.md) |

### 4.2 Extraction and quality control

| Prompt | Main clarification | Replacement | Original review | Fresh replacement check |
| --- | --- | --- | --- | --- |
| 05 Patient extraction | One patient, fixed definitions, explicit date handling; Python attaches the row. | [05](revised/05_extract_patient.md) | [Questions](reviews/05_original.md) | [Check](checks/05_revised.md) |
| 06 Serial extraction | Prior decision notes are allowed evidence state; Python owns chunk order and identity. | [06](revised/06_serial_chunk.md) | [Questions](reviews/06_original.md) | [Check](checks/06_revised.md) |
| 07 Page observations | Extract occurrences and exact quotations; Python locates offsets and reconciles pages. | [07](revised/07_page_observations.md) | [Questions](reviews/07_original.md) | [Check](checks/07_revised.md) |
| 08 Category mapping | Normalize one token into the existing vocabulary; do not infer new patient facts. | [08](revised/08_map_categories.md) | [Questions](reviews/08_original.md) | [Check](checks/08_revised.md) |
| 09 Failure-driven ontology review | Failed outputs are not clinical truth; synonyms need not change the schema. | [09](revised/09_refine_ontology.md) | [Questions](reviews/09_original.md) | [Check](checks/09_revised.md) |
| 10 Value harmonization | Do not invent cutoffs or midpoints; Python builds intervals from supported boundaries. | [10](revised/10_harmonize_values.md) | [Questions](reviews/10_original.md) | [Check](checks/10_revised.md) |
| 11 Frozen-map extension | Interpret one token under the fixed representation; never redesign the bins. | [11](revised/11_extend_value_map.md) | [Questions](reviews/11_original.md) | [Check](checks/11_revised.md) |
| 12 Aggregate ontology supervision | Current values and historical failures may overlap; this is definition review, not selection. | [12](revised/12_supervise_ontology.md) | [Questions](reviews/12_original.md) | [Check](checks/12_revised.md) |
| 13 Extracted alias consolidation | Require compatible paired-value agreement, not correlation alone; Python compiles merge mechanics. | [13](revised/13_post_extraction_aliases.md) | [Questions](reviews/13_original.md) | [Check](checks/13_revised.md) |

### 4.3 Selection, interpretation, and the experimental concept review

| Prompt | Main clarification | Replacement | Original review | Fresh replacement check |
| --- | --- | --- | --- | --- |
| 14 Default role adjudication | Supply study context, distinguish predictive/causal roles and missing/negative evidence. | [14](revised/14_default_roles.md) | [Questions](reviews/14_original.md) | [Check](checks/14_revised.md) |
| 15 Model themes | Organize every feature, including singletons; do not assign roles or merge measurements. | [15](revised/15_model_themes.md) | [Questions](reviews/15_original.md) | [Check](checks/15_revised.md) |
| 16 Theme consolidation | Merge synonymous themes only; Python handles context limits without forced clinical umbrellas. | [16](revised/16_merge_themes.md) | [Questions](reviews/16_original.md) | [Check](checks/16_revised.md) |
| 17 Multi-model roles | Explain different evidence families and role-specific uncertainty/stability. | [17](revised/17_model_roles.md) | [Questions](reviews/17_original.md) | [Check](checks/17_revised.md) |
| 18 Modifier ranking | Order by heterogeneity evidence; Python owns rank numbers and subsequent count selection. | [18](revised/18_rank_modifiers.md) | [Questions](reviews/18_original.md) | [Check](checks/18_revised.md) |
| 19 Ranking merge | Compare two eligible front candidates; Python preserves each list's order. | [19](revised/19_merge_rankings.md) | [Questions](reviews/19_original.md) | [Check](checks/19_revised.md) |
| 20 Advisory annotation | Explain fixed numerical decisions without returning replacement assignments. | [20](revised/20_advisory_roles.md) | [Questions](reviews/20_original.md) | [Check](checks/20_revised.md) |
| 21 Cross-fold concepts | Recurrence is descriptive, not independent confirmation; uncertainty remains an allowed outcome. | [21](revised/21_cross_fold_concepts.md) | [Questions](reviews/21_original.md) | [Check](checks/21_revised.md) |

### 4.4 Shared repair conversations

| Prompt | Main clarification | Replacement | Original review | Fresh replacement check |
| --- | --- | --- | --- | --- |
| 22 Validation repair | Show the original task, failed response, and precise semantic error; return a complete correction. | [22](revised/22_validation_repair.md) | [Questions](reviews/22_original.md) | [Check](checks/22_revised.md) |
| 23 Length repair | Preserve every required feature; Python increases budgets or splits requests when necessary. | [23](revised/23_length_repair.md) | [Questions](reviews/23_original.md) | [Check](checks/23_revised.md) |

The old 22/23 examples were **appended repair fragments**. Their isolated reviewers correctly reported missing original context, but the production transport already retains the original conversation. The new examples show complete repair conversations; this fixes the review artifact's missing context and improves the repair wording, rather than demonstrating a production context-loss bug.

### 4.5 Clarifications after replacement checks

| Prompt | Final clarification | Verification |
| --- | --- | --- |
| 05, 22, 23 | Explicitly flat feature-keyed JSON; Python supplies wrappers. Latest/earliest explicitly prefers dated observations. | [Additional fresh check](checks/05_final_fresh.md), [repair follow-up](checks/22_followup.md), [length follow-up](checks/23_followup.md) |
| 06 | Dated versus undated precedence; threshold strings; route mode features through occurrence extraction so Python counts them. | [Follow-up](checks/06_followup.md) |
| 10 | Python constructs complementary intervals; the LLM does not enumerate bins. | [Follow-up](checks/10_followup.md) |
| 14, 17 | Define the qualitative assessment labels, especially uncertain versus not_supported. | [Default-role follow-up](checks/14_followup.md), [multi-model follow-up](checks/17_followup.md) |
| 15, 16 | Explicit top-level object and named array key. | [Theme follow-up](checks/15_followup.md), [merge follow-up](checks/16_followup.md) |
| 18, 19 | A univariable effect model can use nuisance-adjusted residuals; validation here is inside training, never the outer test set. | [Ranking follow-up](checks/18_followup.md), [comparison follow-up](checks/19_followup.md) |

## 5. Changes that go beyond wording

These are reviewable proposals, not silent production policy changes:

1. **Single-target calls and human-readable labels.** Several proposals simplify response identity by handling one patient, feature, or token per call. Integrating them requires adapters and can increase request count. Batched implementations could instead use unique semantic labels while retaining the same instructions.
   - Proposal 06 also routes mode aggregation through observation extraction, so Python owns occurrence counts. This requires caller routing and state changes; it is not accomplished merely by changing prose.
2. **Ontology defaults.** Proposal 04 explicitly prefers latest for ordinary repeated measurements. The runtime already has a compatibility fallback to latest, but putting that preference in the definition prompt changes how the model chooses a rule. Study-specific measurement policies still need to take precedence.
3. **Harmonization policy.** Proposal 10 uses only supported boundaries and adds an `insufficient_definition` path. Its caller must validate semantic interpretations against parsed numerical constraints, construct exhaustive/nonoverlapping bins, and handle unsupported text without silently discarding a variable. Free-form interpretation is explanatory evidence, not executable binning code. Complex unparsed language needs an explicit typed adapter or clarification path before adoption.
4. **Lossless consolidation.** Proposal 13 expects explicit paired-agreement diagnostics and lets Python compile recodes. It is not a drop-in replacement for a schema that expects model-authored transformation expressions.
5. **Role uncertainty and scientific context.** Proposals 14/17 require real study context and explicit evidence-availability states. Their supported/plausible/uncertain/not-supported judgments remain qualitative; no new numerical selection cutoff is hidden in the prompt. The illustrative study facts must never be substituted for a real study configuration.
6. **Theme context management.** Proposal 16 removes a forced semantic output-count cap. Its adoption requires Python batching/retrieval that can cope with legitimately distinct themes.
7. **Ranking mechanics.** Proposal 18 represents ties explicitly. Proposal 19 uses a pairwise comparison within a Python merge, which may require more requests and can expose nontransitive preferences. It needs cost and ranking-sensitivity evaluation before replacing the current batching strategy.
8. **Provenance granularity.** Python can attach the complete supplied evidence panel to a decision. That is an audit trail of what the model saw, not proof that the model relied on every source. Rationales name relevant methods in words; a future need for exact evidence-level citations requires a separate validated semantic association contract.

## 6. Artifacts and limits

- [Proposal manifest](proposal_manifest.json): per-prompt purpose, changes, and Python responsibilities.
- [All replacement messages in one document](ALL_REVISED_PROMPTS_2026-09-23.md): full prose and example inputs.
- [Verification record](verification.json): coverage, response checks, hashes, and follow-up status.
- `original/`: exact messages used for original isolated reviews.
- `first_drafts/`: proposals before the remaining clarification pass.
- `revised/`: final proposed messages; only 01 is implemented elsewhere.
- `reviews/`: original recipient questions.
- `checks/`: fresh replacement reviews, example responses, and targeted follow-ups.
- [Builder](build_replacements.py): reproduces the proposals without model calls or production mutation.

These examples are deliberately small and invented. Understanding them does not establish robustness to long notes, broad candidate pools, poor extraction, weak statistical power, or correlated predictors. The new proposals have not been benchmarked on the Gemma services or on fold 1. No performance claim is made for these changes.
