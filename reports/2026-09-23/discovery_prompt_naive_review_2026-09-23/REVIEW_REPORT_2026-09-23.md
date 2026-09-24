# Discovery-prompt review with fresh GPT 5.6 Sol recipients

Date: September 23, 2026.

**Adoption update, September 23, 2026:** The approved v3 discovery prompt and
its Python caller/validator changes have now been implemented. The report below
records the earlier proposal exercise; its statements about being unimplemented
describe that earlier checkpoint. Current adoption and the remaining prompt
proposals are documented in [the follow-up review](../all_other_prompts_naive_review_2026-09-23/REVIEW_REPORT_2026-09-23.md).

## Baseline and scope

The existing repository state was committed first on `main` as `e5466c414140333b23b7c52b1c48e23f25ae925b`, **Record fold-one modifier concept comparisons and all-evidence prompt inventory**. That checkpoint contains 144 previously untracked report, experiment, and prompt-documentation files. No push was requested.

This exercise revises only the proposed `01_discover` prompt. Production source code and its execution path are unchanged. The new prompt changes the discovery request/response contract; the integration notes below describe what must change when adopting it. The original rendered inventory remains a faithful snapshot of the original implementation.

## Review method

Every reviewer was explicitly requested as `gpt-5.6-sol` and started with `fork_turns="none"`. Each was told to read only the designated messages file, without repository code, other prompt documentation, experiment results, or oracle information. Reviewers were asked to restate the task, identify consequential missing context, ask clarification questions, and identify bookkeeping. They were not instructed to agree that the prompt was defective or to produce a fixed number of objections.

1. **Original request:** `/root/naive_discovery_review` read only `original_messages.json`. It understood the broad goal but could not execute without six consequential assumptions. Its complete response is in `original_review_gpt_5_6_sol.md`.
2. **First revision:** `/root/fresh_revised_discovery_review` read only `revised_messages_v1.json`, without the first review or our answers. It said the request was sufficiently specified and produced the expected two clinical concepts. Its sample nevertheless reported an unnecessary date-selection caveat. This is preserved in `revised_v1_review_gpt_5_6_sol.md`.
3. **Second revision:** `/root/fresh_final_discovery_review` received only `revised_messages_v2.json` and an additional invented clinical example. It needed no clarification for either example, returned two and nine candidates respectively, and correctly left the routine creatinine repetition without an uncertainty caveat. It identified a remaining choice about detailed measurements versus inferred presence-only copies.
4. **Final revision and targeted follow-up:** revision 3 explicitly chooses the detailed measurement (cough duration) over a redundant inferred presence-only copy, while preserving separately stated severity and duration. The third reviewer read the full updated request and confirmed the boundary was resolved, with both earlier example outputs remaining appropriate. This follow-up inherited that reviewer's prior exchange; it was not a fourth fresh-context review. The full assessment and follow-up are in `revised_v2_review_gpt_5_6_sol.md`.

Reviewing the actual task response was useful: a recipient's statement that it understands the prompt did not, by itself, catch the unnecessary caveat in the first revision. This is a small qualitative prompt exercise, not evidence of better feature recovery or causal estimation.

## Questions and their explicit resolutions

| Fresh recipient's question | Resolution in the revised prompt |
| --- | --- |
| Do two dated creatinine values violate “one value per patient”? | Discovery identifies one measurement concept. A later step decides which observation to extract. Repetition and different dates are expected, not missing instructions. |
| Are dates and “pretreatment” labels additional variables? | Exact dates, timestamps, and those labels provide context. Explicit clinical time measures, such as symptom duration or age at diagnosis, can be variables. |
| Does a CT finding imply an imaging feature, a diagnosis, or CT performance? | Preserve the finding's supported meaning, such as emphysema on imaging. A test mentioned only as the source of a finding does not create an additional procedure variable. Explicit treatment/procedure history can. |
| Are plausible latent features allowed, or only explicit/unambiguous ones? | Use the original system message's direct-or-unambiguously-encoded threshold consistently. Standard abbreviation/notation expansion is allowed; a plausible additional diagnosis or risk factor is insufficient. |
| Can a description call something a concentration or presence/status variable? | Yes. Descriptions state the measured dimension in ordinary language. Formal units, categories, thresholds, missingness, and observation-selection rules belong to the later definition step. |
| Which strings are consensus versus representative excerpts, and do they carry different weights? | The recipient does not need to classify source types. Read all excerpts independently, with no hidden priority for a heading or summary. |

Additional clarifications make the recipient's job self-contained:

- The input can contain excerpts from different patients or dates, rather than one complete patient record.
- The output proposes clinical measurements; it does not extract patient values, select useful variables, or infer causal roles.
- A patient's own laboratory or tumor findings can support patient variables. A relative's illness can support family history without becoming the patient's diagnosis.
- Negative and uncertain mentions can identify a variable without implying a positive finding.
- Independently varying dimensions remain separate. Named overall scores/categories remain legitimate measurements, while concatenated multi-field codes are decomposed.
- Uncertainty concerns what the evidence means or supports. Deliberately deferred extraction choices are not missing context to report as uncertainty.

These choices are explicit scientific scope decisions. In particular, this is a text-grounded discovery prompt; it is not a separate speculative latent-concept generation pass. They should be assessed as such when the prompt is eventually evaluated on real evidence cards.

## Bookkeeping moved to Python

The proposed request contains **one evidence card's readable text at a time**. It has no card IDs, item numbers, excerpt indices, or character offsets. The response contains only `name`, `description`, `basis`, and `uncertainty` for each clinical candidate.

Python owns the following:

1. Keep the evidence card's ID, architecture, axes, and other provenance in the call context.
2. Attach that provenance to every returned candidate from the call.
3. Create internal identifiers and normalize human-readable clinical labels into machine-friendly names, resolving naming collisions without conflating distinct variables.
4. Validate the response schema and map `basis` to the existing internal `evidence_rationale` field and `uncertainty` to `caveats`.
5. Record an empty candidate list as a reviewed card with no supported candidate. Route any existing recall audit in code.
6. Preserve source associations when later semantic alias consolidation combines candidates from different cards.

The LLM retains tasks that require clinical interpretation: naming the measured dimension, deciding whether mentions describe the same attribute, explaining textual support, and identifying substantive ambiguity. `basis` and `uncertainty` do not ask it to manage source identifiers or bookkeeping. JSON remains an output serialization format; the instructions themselves are ordinary prose.

## Adoption implications

The current production caller batches several cards and its validator expects model-authored `supporting_items`. Therefore **replacing the prompt string alone would be incorrect**: the old validator could discard otherwise valid new-schema candidates for lacking citations.

Adoption requires one-card calls or an equivalent execution wrapper that keeps source association outside the model. The adapter can populate the old internal citation representation itself, or update the validator to accept the new response contract and attach provenance directly. Prompt/checkpoint versioning must also change so old discovery results are not reused as if generated by the new request.

One-card calls can increase request count relative to the current batching. They remove the need for the LLM to assign outputs to numbered inputs, while cross-card semantic consolidation remains a later task. This execution tradeoff is part of the proposal, not an already-deployed change.
