# All-evidence pathway: LLM prompt examples

Date: September 23, 2026. Repository snapshot: `568ef7a`.

These are **readable, abbreviated examples of the existing prompts**, not a proposed rewrite. Clinical snippets, patient identifiers, diagnostic counts, and illustrative modeling results below are invented. They are not results from fold 1. Every numbered example has an accompanying unabridged rendered template in `rendered/`, with its source location and a JSON copy. The templates were rendered locally without calling a model or modifying the pipeline.

## 1. Map of the pathway

1. **Stage 1:** numerical text modeling, retrieval, clustering, and evidence-card construction. The current all-evidence route does not ask a generative LLM to interpret each Stage 1 model. Pretrained encoders are used, but that is different from a chat prompt.
2. **Stage 2 discovery:** read the evidence-card text → discover atomic features → audit uncited items → consolidate aliases → define each measurement.
3. **Stage 2 measurement:** extract patient values → repair invalid responses when necessary → address category/schema/representation problems → supervise extraction quality. Some steps loop; this is not a single unconditional chain.
4. **Optional measurement consolidation:** check whether extracted variables are losslessly interchangeable aliases.
5. **Stage 2 selection:** numerical models generate evidence. Depending on the selection mode, the LLM adjudicates roles directly, reviews multi-model themes and roles, or merely annotates numerical decisions. In the multi-model modifier-count search it also ranks candidates and combines rankings.
6. **Final estimation:** cross-validation selects modifier count and, when enabled, model architecture numerically. Frozen selected measurements are extracted from heldout text; the estimator is fitted and evaluated in code. There is no final LLM judgment of the resulting ITEs.
7. **Latest experiment:** a separate cross-fold top-100 concept review uses saved rankings/evidence. It is not yet an integrated production step.

The configured **primary/adjudication model** handles interpretation and review. The configured **extraction model** handles the patient-extraction variants. Optional post-extraction consolidation can have its own model override. The Gemma 31B / 26B A4B services are choices of those roles, not hardcoded identities in the scientific prompts.

## 2. Discovery and measurement definition

### 01. Discover atomic clinical features

**Input:** readable evidence-card text and local item numbers. Architecture identity, numerical importance/ranks, provenance IDs, and the clinical question are not supplied in this prompt.

> Examine every supplied text item and identify all explicitly stated or unambiguously encoded patient-level clinical variables. Each variable must support one value for one patient. Do not stop at the dominant topic or combine independently varying findings into a broad profile. Cite the supporting item numbers. Do not choose causal roles, units, or extraction categories yet.
>
> Item 1: “Creatinine 1.2 mg/dL. CT documents emphysema.”
>
> Return candidate names, descriptions, supporting items, rationales, and caveats in JSON.

This can yield multiple candidates from one card. It does not require one feature per card or a minimum number of supporting cards.

### 02. Re-read evidence left uncited

**Trigger:** an initial discovery pass leaves an evidence item uncited. This is an omission-recovery pass, not a vote over all candidates.

> The supplied item was not cited in the initial review. Re-read every string for missed clinical variables, including secondary findings outside the apparent topic. One clear clue is sufficient; it need not recur. Apply the same atomic-variable rules. Return an empty list only if the text supports no valid variable.
>
> Previously uncited item: “CT documents emphysema; creatinine 1.2 mg/dL.”

### 03. Consolidate names before extraction

**Input:** candidate names and descriptions, reviewed in repeated bounded partitions. Every input survives unchanged or within one alias merge.

> Review these candidates: `serum_creatinine`, `creatinine_level`, and `emphysema`. Merge only names that represent the same underlying patient measurement. Prefer an informative canonical name. Keep independently varying measurements separate. Never drop a feature or create a broader umbrella concept. Return only merge directives using exact supplied names.

The real prompt also allows quantitative and coarsened names to be unified when one underlying measurement and extraction ontology can represent them. This happens **before measurement extraction**; it is less restrictive than the later lossless consolidation of already extracted values. Investigator-specified names have explicit protection rules.

### 04. Define the extraction ontology

**Input:** one canonical feature name and its supporting clinical text. This step determines how an extractor should measure it.

> Define exactly `serum_creatinine` from the supplied evidence. Choose its value type, unit or allowed categories, reproducible extraction rule, missing-value rule, and policy for conflicting observations. Prefer a continuous measurement when feasible. Preserve supported threshold reports when an exact number is unavailable. Do not rename the feature, combine it with another variable, or invent a score.
>
> Evidence: “January 1: creatinine 1.0 mg/dL. January 10: creatinine 1.2 mg/dL.”
>
> Return the definition and a conflict policy such as latest, earliest, maximum, minimum, mode, any-positive, or single-or-null.

Investigator-supplied definitions can bypass inferred ontology construction.

## 3. Extraction and its conditional quality-control prompts

### 05. Extract one patient's values

**Model:** extraction model. **Input:** one patient's text and prespecified feature definitions; no treatment/outcome columns or modeling statistics.

> Extract only the declared variables from this patient's supplied clinical text. Follow each measurement and missing-value rule literally. Consider all supported observations before applying the declared conflict policy. Use exact declared categories and one scalar value or null per feature. Return every requested feature exactly once.
>
> Variables: latest pretreatment creatinine in mg/dL; emphysema, using `Present` or `Absent`.
>
> Record: “January 1: creatinine 1.0. January 10: creatinine 1.2. CT documents emphysema.”
>
> Return a JSON row containing `serum_creatinine` and `emphysema`.

The same prompt family is reused for heldout extraction after selection is frozen. Unmentioned emphysema is missing, not automatically absent.

### 06. Update extraction across a long record

**Trigger:** serial chunked extraction. **Input:** current text chunk, prior validated values, and bounded per-feature decision state.

> Continue this patient's extraction using the prior values and the next record chunk. Preserve a prior nonmissing value unless the new evidence changes the result under the feature's conflict rule. For latest/earliest, compare governing dates when available; otherwise follow source-order rules. Do not treat a prior null as a negative finding. Return updated values and concise state needed for the next chunk.
>
> Prior: creatinine 1.0, dated January 1. New chunk: “January 10: creatinine 1.2 mg/dL.”

The model is not asked to re-read the entire preceding record at each update.

### 07. Extract individual page observations with provenance

**Status:** alternate page/oversized-record path in shared extraction code; distinct from serial cumulative extraction.

> Extract every supported observation on this page for the declared features. Do not collapse repeated or conflicting values. For each observation, provide its value, an exact source quote, character offsets, and a governing date only if that date is explicitly supported. Return no observation for an unsupported feature.
>
> Page: “January 1: creatinine 1.0 mg/dL. January 10: creatinine 1.2 mg/dL.”

Cross-page conflict resolution subsequently runs in code. There is **no extra LLM reconciliation prompt** for that operation.

### 08. Map an invalid category to the existing ontology

**Trigger:** category-validation failures that reach the normalization path. **Model:** primary model. **Input:** prior extracted tokens and allowed categories; no patient text.

> Normalize the previously extracted value using only the supplied definition and declared categories. Map by unambiguous semantic equivalence; otherwise return null. Do not perform new clinical extraction or infer additional patient information.
>
> Feature: emphysema. Allowed values: `Present`, `Absent`. Extracted token: “present on CT.”
>
> Return the correction for the supplied mapping ID.

This corrects a value to an existing schema; it does not revise the schema.

### 09. Refine an ontology after repeated extraction failures

**Trigger:** repeated failures attributable to the same feature in training extraction. **Input:** feature definition and aggregate failure examples, which are prior model outputs rather than verified clinical facts.

> Review this feature's definition and repeated validation failures. Decide whether the existing ontology should be kept or revised to make the same measurement extractable. Do not blindly add every failed output as a category. Keep the feature's identity and causal roles fixed. Return a reason and either the unchanged definition or revised description, type, categories/unit, measurement rule, and missing-value rule.
>
> Emphysema allows `Present`/`Absent`; three extractions returned “positive” or “present on CT.”

An acceptable conclusion can be “keep”: semantically equivalent category tokens do not automatically justify redesigning the feature. Generic malformed JSON is not, by itself, evidence that the clinical ontology needs revision.

### 10. Harmonize mixed numeric and textual representations

**Trigger:** a nominally continuous extracted variable contains both numbers and text. **Input:** its definition and aggregate training values; no treatment, outcome, or heldout data.

> Choose one common modeling representation for these observed values. Use continuous only if every nonnumeric token has an exact numeric meaning. Do not invent midpoints for ranges, inequalities, or qualitative labels. Otherwise define coherent categories, map each text token, and specify exhaustive, nonoverlapping numeric bins. Map unusable tokens to null.
>
> Creatinine values include `0.8`, `1.2`, “<1.0”, and “high”.

This is a representation decision, not evidence of effect modification.

### 11. Extend a frozen value map for new training tokens

**Trigger:** incremental training extraction introduces an unseen text representation after a harmonization plan exists.

> Map these newly observed text values into the existing harmonization plan. Keep its target representation, categories, and numeric bins unchanged. Copy each raw token exactly. Use only an existing category or an unambiguous exact number, as applicable; otherwise return null.
>
> Frozen categories: “Below 1.0”, “At least 1.0”. New token: “less than 1.0”.

The example threshold is invented for illustration. Heldout values do not trigger learning a new harmonization plan from test data.

### 12. Supervise extraction quality from aggregate diagnostics

**Input:** one feature's ontology, aggregate missingness/value summaries, and validation-failure patterns. Unlike failure-driven refinement, this is the broader training extraction supervision pass.

> Review whether the small model can apply this feature's ontology coherently. Use the observed value distribution and aggregate failures to decide keep versus revise. Preserve the same feature. You may clarify its measurement and missing-value rules or repair an unsuitable value schema. Do not select features or assign causal roles.
>
> Emphysema: 10 illustrative records; 5 `Present`, 3 `Absent`, 2 missing; repeated noncanonical category tokens.

The prompt explicitly excludes patient text, treatment/outcome values, causal-role evidence, performance measures, and p-values. A revised schema can cause affected training measurements to be re-extracted and reviewed again.

## 4. Optional consolidation of extracted measurements

### 13. Require semantic equivalence and lossless value agreement

**Input:** a pivot candidate, semantic neighbors, extraction definitions, observed training summaries, and pairwise associations. Treatment and outcome are unavailable.

> Decide whether these extracted fields are interchangeable encodings of the same measurement: same attribute, entity, time scope, granularity, and compatible scale. High association alone is insufficient. Any replacement must preserve nonmissing information, reconcile only equivalent category labels, and preserve all-source missingness as null. Do not hide conflicts with a first-source-wins rule. If uncertain, leave the fields unchanged.
>
> Candidates: creatinine and creatinine level, both mg/dL, with concordant observed values.
>
> Return `leave_unchanged` or explicit lossless replacement rules. Do not assign causal roles.

This prompt's schema uses the word `latents`, but the current instructions prohibit broader latent concepts or new burden/composite scores. The config class defaults `sequential_consolidation.enabled` to **false**; individual configurations can explicitly enable it. This optional second pass is separate from the core pre-extraction alias consolidation.

## 5. Selection: alternative branches, not one sequence of all prompts

### 14. Direct role adjudication in `llm_roles`

**Input:** definitions and aggregate evidence from nuisance elastic net, candidate-wise association screens, candidate R-learner tests, and joint modifier modeling.

> For each candidate, decide confounder, effect modifier, both, or neither. Treat statistical methods as evidence, not automatic gates. For confounding, distinguish a plausible common cause of treatment and outcome from an outcome-only predictor or treatment-only instrument. For modification, require evidence about treatment-effect heterogeneity rather than prognosis alone. Reconcile disagreement and compare inner folds. Preserve investigator-locked roles.
>
> Example evidence: creatinine repeatedly predicts treatment and outcome but has weak heterogeneity evidence.
>
> Return each feature's roles, evidence for and against, fold consistency, and rationale.

This is an alternative to the multi-model branch below, not an additional adjudication layered on top of it.

### 15. Find themes in multi-model evidence

**Input:** bounded batches of candidate definitions and summaries from seven modeling families: univariable tests, penalized main effects, penalized interactions, orthogonal linear models, univariable R-learners, predictive forests, and causal forests.

> Identify themes in the supplied candidate measurements and their modeling evidence. Cover every candidate, including weak and unevaluable candidates. Describe common or complementary evidence, disagreements, and distinctions between aliases and related proxies. No single modeling family, p-value, or support fraction is a mandatory gate. Cite supplied evidence IDs.
>
> Illustrative pattern: one measurement appears in penalized treatment/outcome models; another has logistic-interaction support but inconsistent adjusted R-loss and forest evidence.

The real system prompt explains the different effect scales, propensity-restricted modifier population, correlated predictors, and dependence among overlapping folds/resamples. A theme organizes evidence; it does not itself merge extracted measurements.

### 16. Combine themes across batches

**Trigger:** there are too many initial themes for the bounded review context.

> Combine these theme summaries into at most the supplied number of broader summaries. Preserve all candidate IDs and meaningful distinctions. A parent theme is an organizational device, not a declaration that its measurements are interchangeable. Preserve representative evidence citations and disagreements.

This is hierarchical compression. It has a per-request output-theme limit; that differs from the experimental global concept review, which does not target a fixed concept count.

### 17. Assign roles using multi-model evidence and themes

**Input:** candidate modeling cards plus their relevant consolidated themes.

> Assign each candidate confounder, effect modifier, both, or neither. Explain supporting and contradictory evidence, stability across folds/subsets, and disagreements across modeling methods. Use themes as context without transferring one member's evidence automatically to every other member. Cite the candidate's own evidence: effect evidence for modifier assignments, and treatment plus outcome evidence for confounder assignments. Preserve locked roles.

Those citations establish an auditable connection to the evidence; they are not proof that a variable is causally valid. The response also labels stability as consistent, mixed, or insufficient.

### 18. Rank candidates for modifier-count cross-validation

**Input:** all eligible unlocked candidates, not only those provisionally labeled modifiers, with numerical evidence computed in the applicable training split.

> Rank these candidates for modeling treatment-effect heterogeneity. Compare adjusted R-loss evidence, consistency, evaluability, redundancy, and complementary information. Outcome prognosis alone is insufficient. Put weak or unevaluable candidates toward the tail, but include every supplied candidate exactly once. Cite each candidate's own effect evidence. Do not merge measurements or choose how many modifiers to retain.

Rankings are computed within inner training splits, followed by a full outer-training ranking for the final fit. Numerical cross-validation, not the ranking LLM, chooses the prefix length and model architecture where architecture search is enabled. Locked roles are handled separately.

### 19. Interleave sorted modifier lists

**Trigger:** bounded ranking requests produce multiple ordered lists.

> Interleave these candidate lists into one ranking using the supplied modeling evidence. Preserve the order already established within each input list. Include every supplied candidate exactly once and justify comparisons using its own effect evidence.

This combines batch rankings. It is not the later experimental search for recurring concepts across inner folds.

### 20. Advisory annotation in `independent_tasks`

**Status:** separate selection mode. Numerical selection and feature routing are frozen first; the optional LLM pass only annotates them.

> Interpret the candidate measurements and their aggregate evidence, distinguishing possible confounding, prognosis, and effect modification. Explain disagreements and uncertainty. Your response is advisory: you cannot add or remove selected features or change their numerical routing. Preserve investigator-locked annotations.

An annotation failure does not overturn the numerical selection.

## 6. Latest experimental cross-fold concept review

### 21. Infer concepts from each fold's top 100 candidates

**Status:** experiment saved in reports, not an integrated production step. The actual review included the union of 217 candidates across five inner-training top-100 lists. Gemma and the blinded Sol comparison used the same substantive synthesis instructions; execution wrappers differed.

> Review the union of the top-100 candidate lists from five overlapping inner-training splits. Infer recurring underlying concepts, including those represented by different proxies in different folds. Use definitions, ranks, and supplied effect-model evidence; do not infer oracle truth or a known data-generating process.
>
> Assign every candidate to exactly one coherent concept. Distinguish aliases, distinct facets, and indirect proxies. Do not combine unrelated findings merely to reduce the count. For each concept choose retain, uncertain, or exclude. For a retained concept, choose a parsimonious set of existing representative feature IDs, usually one unless others add distinct information. There is no fixed number of concepts or minimum fold-frequency threshold.
>
> Reconcile unadjusted log-odds interactions with nuisance-adjusted probability-scale evidence. Missing fits are not negative votes; overlapping folds are not independent replications. Cite each selected representative's own effect evidence and relevant inner folds. Preserve confounders separately. Do not invent composite variables or claim that selection has improved causal estimation.

This is the dense JSON prompt that prompted this documentation request. It includes per-fold ranks and numerical arrays for each candidate/method, with legends for exposures, support, evaluability, scores, and p/q values.

The [actual saved blinded request](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-23/fold_1_blinded_sol_modifier_concepts_2026-09-23/blinded_input.json) contains the full system prompt and all 217 candidate records. The [Sol execution prompt](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-23/fold_1_blinded_sol_modifier_concepts_2026-09-23/review_prompt.txt) includes its execution wrapper. The rendered example accompanying this guide preserves the prompt and schema but substitutes two invented candidate records.

## 7. Shared response-repair prompts

### 22. Repair malformed or schema-invalid output

The task context is retained, with validator feedback and available failed-response content subject to prompt/context budgets. The repair goes to the model responsible for that request.

> The previous JSON failed validation. Correct this exact error: ValueError: missing required feature serum_creatinine in row 1. Return one corrected JSON object only.

This is the actual repair wording with an invented example error. For some tasks, feedback adds allowed feature IDs and valid expression examples. Repeated consolidation failures can explicitly require the conservative `leave_unchanged` response. Category mapping and ontology revision above are distinct from this generic structural repair.

### 23. Repair output that exceeded the response length

> The previous JSON exceeded the available response length. _Stage2OutputLengthError: response reached the configured completion-token limit. Return one materially shorter corrected JSON object using the same required schema. Remove redundancy, merge duplicate entries, and keep descriptions and rationales concise. Do not omit required records or fields. Return JSON only.

This is the current template with an invented error message. It asks for a shorter valid answer, not fewer required patients/features.

### Cross-cutting reasoning-control text

For supported model families, transport can append a reasoning instruction in addition to API/chat-template controls. For Gemma with reasoning enabled, the portable wording is:

> Enable the model's thinking mode for this request, but return only the final JSON object in response content.

With reasoning disabled:

> Disable thinking for this request and return only the final JSON object.

This is a transport setting, not another scientific review step. Repair reasoning escalation also applies to structural errors. The default allows up to 15 response repairs after an initial attempt, subject to request limits/deadlines; the configured threshold enables high reasoning after five completed repairs. Per-run settings can override these defaults.

## 8. What is not an additional LLM prompt

- Stage 1 numerical discovery and evidence-card compilation.
- Embedding retrieval, candidate/card routing, and deterministic deduplication.
- Computation of propensity scores, univariable tests, penalized models, R-loss, or forest importance.
- Cross-page application of declared observation-conflict rules.
- Cross-validation of modifier count and final estimator architecture.
- Final causal-forest or interaction-model fitting, ITE prediction, and effect-estimate evaluation.
- Application of frozen value maps to heldout extractions.
- Oracle-recovery evaluation after a blinded selection is frozen; oracle columns are excluded from the selector prompts.

Older modules and the legacy fold-analysis function contain additional prompt families, including earlier fusion/global selection and broader post-extraction review. They are not called by the current all-evidence Stage 2 entrypoint and are not presented here as active steps.

## 9. Reading burden and source fidelity

The prompts differ substantially in what the model has to digest. Discovery and operationalization largely involve clinical text and definitions. Extraction involves a patient record plus fixed schemas. Quality control sees aggregate values or failure summaries. The heaviest numerical interpretation is concentrated in role adjudication, multi-model theme review, modifier ranking, and the experimental cross-fold concept pass. The post-extraction alias check also contains numerical agreement evidence, for a different purpose.

These readable examples remove repetitive schema boilerplate and shrink the data, but do not represent a deployed plain-language replacement. The full rendered examples retain the current instructions and response contracts. Their metadata records source locations and source-file hashes. No selection, modeling, or extraction behavior was changed.

The [manifest](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-23/all_evidence_prompt_examples_2026-09-23/manifest.json) inventories all 23 examples; [build_examples.py](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-23/all_evidence_prompt_examples_2026-09-23/build_examples.py) reproduces them using the repository Python environment. The [rendered index](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-23/all_evidence_prompt_examples_2026-09-23/RENDERED_INDEX_2026-09-23.md) links to every exact template.
