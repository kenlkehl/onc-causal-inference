# Stage 2 prompt adoption — 2026-09-23

1. **Production prompts**
   1. All 23 reviewed prompt types are adopted: discovery and recall audit; alias merging and definitions; patient, sequential-chunk, and occurrence extraction; category mapping and ontology refinement; value harmonization and extensions; aggregate supervision; extracted-measurement aliases; default roles, themes, theme consolidation, and multi-model roles; modifier ordering and comparisons; advisory annotations; modifier concepts; validation and length repairs.
   2. The 21 task-specific system messages in `oci/inference/stage2_prompt_catalog.py` exactly match the approved Qwen-reviewed proposals. The two shared repair instructions insert the current error.
   3. Python attaches identifiers and provenance, resolves exact quotations to source offsets, compiles supported numerical intervals, enforces role locks, and manages ranks, ties, and list merging. Responses use clinical variable names where matching is necessary.
   4. New prompt fingerprints prevent reuse of incompatible production checkpoints. The run below starts in a fresh output directory.
2. **Endpoint-aware request policy**
   1. Both endpoints are checked through `/models`. The advertised backing model supplies the sampling profile even when the served name is an alias.
   2. Both reasoning settings default to `auto`. For `Inferact/Qwen3.8-Flash-Next-NVFP4`, they resolve to `xhigh`, with thinking and `preserve_thinking` enabled.
   3. Thinking requests use temperature 1.0, top-p 0.95, top-k 20, min-p 0, presence/frequency penalties 0, and repetition penalty 1, following the [Qwen publisher recommendations](https://huggingface.co/Qwen/Qwen3.8-Flash-Next#best-practices).
   4. Explicit configuration overrides remain supported. Unknown model families use server sampling defaults. Flash Next requests fail visibly if the endpoint rejects its reasoning or sampling controls; compatibility retries can remove the JSON-response-format hint.
   5. Model manifests record resolved policies and sources; request events record actual controls sent. Token budgeting uses the detected model's tokenizer and thinking template.
3. **Modifier concepts and final estimation**
   1. The opt-in concept pass reviews the union of the top 100 modifier candidates from each inner fold, with recurrence and model evidence.
   2. It nominates existing representative measurements. Retained and explicitly nominated uncertain representatives enter modifier-count validation; confounder roles are preserved.
   3. Numerical evidence, rankings, and concept review repeat inside each count-validation training partition. The search compares causal forests and penalized outcome models with treatment interactions using held-out R-loss in the 0.1–0.9 propensity interval.
   4. The final concept review uses all outer-training evidence. The selected count is a budget; actual counts are recorded when fewer representatives are available.
   5. The inner comparison remains conditional on outer-training discovery and measurement definitions. Outer-held-out evaluation tests the complete adaptive procedure. Oracle columns are excluded from fitting and LLM inputs.
4. **Verification**
   1. Full repository test run: 818 passed; one old repair-prompt assertion required updating for the new contract. The follow-up run of runtime and sampling tests passed all 45 tests, including the corrected assertion and a new Flash Next rejection test.
   2. Direct clinical-contract tests cover trusted patient mapping, exact quote/date provenance, threshold boundaries and ambiguity, concept coverage, recurrence, and representative selection. The nested modeling test exercises concept-review routing with physically restricted training labels.
   3. A real extraction request to `sn4622130540:8001` returned the expected ECOG and emphysema values using the detected Qwen profile and `xhigh`. Saved requests, policies, and results are in `live_smoke/`.
   4. The exact Qwen tokenizer loaded successfully. The dated run preflight records input hashes, the four loaded dataset columns, model identities, and token-budget verification.
5. **Fresh experiment**
   1. Configuration and protocol: `../five_conf_five_mod_stage2_reviewed_prompts_2026-09-23/`.
   2. Uses preserved five-confounder/five-modifier Stage 1 evidence and split provenance. All five outer folds regenerate Stage 2 definitions and measurements.
   3. Interpretation and extraction both use `http://sn4622130540:8001/v1`. Port 8000 was unavailable during checks.
   4. Current execution state is recorded in that directory's `status.json`; completion is recorded in `result.json` only after the full workflow succeeds.
