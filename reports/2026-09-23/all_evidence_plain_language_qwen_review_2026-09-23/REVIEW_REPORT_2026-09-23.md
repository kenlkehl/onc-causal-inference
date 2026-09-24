# Plain-language review of all-evidence prompts

September 23, 2026

All 23 prompt variants have been rewritten around the clinical task, the meaning of the supplied information, and the answer needed. Each revised prompt was reviewed by Qwen in an isolated conversation and exercised in a separate direct call. The complete wording is in [the revised prompt catalog](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-23/all_evidence_plain_language_qwen_review_2026-09-23/ALL_REVISED_PROMPTS_2026-09-23.md); Qwen's final reviews and direct answers are in [the response record](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-23/all_evidence_plain_language_qwen_review_2026-09-23/QWEN_REVIEWS_AND_RESPONSES_2026-09-23.md).

**Final result:** Qwen reported it could complete all 23 tasks, with 0 remaining clarification questions. All 23 direct miniature examples passed their contract and applicable known-value checks. The 46 final-version calls returned reasoning and completed without truncation. The record includes 103 calls across all revisions.

These remain **reviewed proposals**, following the request to prepare replacements before production integration. This revision changes report artifacts only. The previously adopted discovery implementation remains at its earlier approved version.

## 1. The three requested corrections

1. **Extract one patient's declared variables** now starts:

   > Read the patient's clinical text and report a value for each listed clinical variable.

   The input description names the variable definitions and the patient's text. References to `record_scope`, upstream eligibility handling, and execution details have been removed from the model messages. A clinical restriction that affects a measurement belongs in the actual definition supplied to the model.

2. **Choose a common representation** now starts:

   > The measurements of one clinical variable contain numbers and text phrases. Choose a common representation that preserves their clinical meaning.

   This explicitly establishes the scope as **one clinical variable** in the system message.

3. **Assess roles using several models and clinical themes** now starts:

   > Decide whether one clinical variable should be used to adjust for confounding, to predict differences in treatment benefit between patients, or both.

   It then defines confounding and effect modification, explains the supplied model results, and describes the requested judgments in plain language.

## 2. Changes across the complete set

1. Each message begins with the action the recipient should take. Input descriptions explain clinical excerpts, measurement definitions, previous values, or statistical findings as appropriate.
2. Example user messages use clinical prose and labeled results. JSON remains the response format because the program consumes the answers.
3. IDs, indices, provenance attachment, column naming, offsets, numerical ranks, retry budgets, and orchestration are documented separately under **Developer integration notes — excluded from the prompt**. Clinical labels remain where the model must identify variables it groups or discusses.
4. Extraction prompts contain only the information needed to interpret the record and produce measurements. Refinement prompts focus on whether the measurement instructions need changing.
5. Statistical prompts state the observational study, treatment comparison, outcome, and effect scale. Method descriptions explain what the supplied evidence can support. The overlap population and dependence between patient subsets remain because they affect interpretation.
6. The wording avoids added contrastive boilerplate. A check of the actual messages also excludes `record_scope`, pipeline terminology, Python/caller instructions, and identifier/index bookkeeping.

The inventory includes core, optional, alternative, experimental, and repair variants. It represents 23 prompt types across the previously inventoried all-evidence pathway; an individual run uses the applicable variants.

## 3. Qwen review configuration

The service at `sn4622130540:8001` reported **`Inferact/Qwen3.8-Flash-Next-NVFP4`**, with a 262,144-token context limit, on vLLM `0.29.1rc1.dev452+g3df4ae153`.

Requests use temperature **1.0**, top-p **0.95**, top-k **20**, min-p **0.0**, presence penalty **0.0**, and repetition penalty **1.0**, with thinking enabled and preserved. These follow the [Qwen model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next#api-usage) and [served model's card](https://huggingface.co/Inferact/Qwen3.8-Flash-Next-NVFP4#api-usage). Every request specifies **`reasoning_effort: "xhigh"`**. The server's request schema accepts that value; every selected final call returned reasoning. Frequency penalty is neutral at 0.0.

The review uses a 65,536-token completion ceiling. This is the limit chosen for this exercise. Final calls must finish with `stop` to pass verification. Exact requests, usage, finish reasons, and reasoning lengths are saved; the readable artifacts contain final answers. [Server and sampling record](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-23/all_evidence_plain_language_qwen_review_2026-09-23/sources/server_and_sampling.json).

## 4. Exercise and findings

1. **Fresh comprehension review.** Qwen receives the target messages and example input, with no repository, earlier conversation, oracle truth, or earlier review. It explains the task and input, asks consequential clarification questions, identifies unnecessary context, and attempts the example.
2. **Independent direct execution.** A separate request sends the target messages themselves. This checks whether the task can be carried out without the review wrapper.
3. **Revision and recheck.** Changed prompts receive fresh reviews and direct executions. Verification matches the saved target messages exactly to the current proposal. Historical responses remain available for inspection.

The exercise uncovered concrete problems:

- One definition response treated a documented threshold such as `<1.0` as missing. The instructions now explicitly preserve threshold strings in both the measurement rule and the missing-value rule. The definition response uses a straightforward `unit` string and `categories` list; a future Python adapter will translate them into the existing representation.
- Page extraction initially returned exact numbers as strings. Its response instructions now distinguish JSON numbers from category and threshold strings.
- Definition refinement initially proposed revisions for ordinary synonyms of existing categories. The prompt now explicitly chooses `keep` when the clinical rule is clear and synonym handling is sufficient.
- The role reviewer asked how to judge uncalibrated R-loss gains and unavailable adjusted q-values. The prompt explains how to describe direction and consistency while leaving practical magnitude unresolved, and how to interpret nominal p-values with limited multiplicity information.
- A direct comparison answer conflated the R-learner's constant-effect baseline with causal-forest permutation importance. The forest description now names its comparison explicitly: the same fitted forest before and after shuffling the variable. The five prompts using that description received fresh reviews and direct checks.
- A review called the example randomized although the input had not specified the design. The study description now explicitly says observational.

Two early comprehension calls returned only the target task answer. The review wrapper was clarified, and fresh reviews were run. Those early calls are preserved as unsuccessful reviews and excluded from final review coverage.

Qwen also suggested removing information that was unnecessary for the tiny example but useful for the general task. The proposals retain the available measurement-selection rules, the study population, evidence-dependence information, and the explicit scope of one clinical variable. Other suggestions led to removal of an unrelated CT example, an unnecessary age-output example, generic software explanations, and irrelevant clinical background.

## 5. What the checks establish

The [verification record](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-23/all_evidence_plain_language_qwen_review_2026-09-23/verification.json) and [per-prompt index](/data1/ken/pcori_dev/causal-dragonnet-text/reports/2026-09-23/all_evidence_plain_language_qwen_review_2026-09-23/PROMPT_CHECK_INDEX_2026-09-23.md) identify the exact final review and direct response for every proposal. Checks cover the response contract, known extraction values, exact supporting quotations and dates, allowed labels, and candidate coverage. Clinical role, theme, ranking, and concept judgments are recorded without requiring agreement with a favored scientific conclusion.

These invented miniature examples establish whether the task and answer format are workable. They leave full-record extraction accuracy, variable recovery, selection stability, and causal-estimation performance unevaluated. One intermediate response even reversed a numerical count in its explanation while saying it understood the task. Comprehension and valid JSON remain separate from reliable interpretation of evidence.

No production extraction, selection, or estimation experiment was run for this revision. No new production-test result is claimed.

## 6. Reproducible artifacts

- `build_prompts.py`: writes all 23 proposals and the catalog.
- `run_qwen_review.py`: makes isolated comprehension and direct-execution calls with the saved sampling settings.
- `verify_reviews.py`: checks the saved current messages, actual request settings, response examples, and review coverage; writes the verification record and response index.
- `runs/`: preserves every round's exact messages, requests, final answers, and response metadata.

Developer integration notes describe the remaining Python changes for adopting each proposed contract. Those notes are absent from the model requests.
