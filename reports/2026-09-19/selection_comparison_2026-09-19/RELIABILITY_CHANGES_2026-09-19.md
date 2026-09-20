# Stage 2 extraction reliability changes — September 19, 2026

1. **Output budgets and progress**
   1. Ordinary extraction uses a 4,096-token output ceiling in the revised comparison and recommended example configuration.
   2. Reasoning-enabled extraction, including a repair that enables thinking, uses a separate 32,768-token total ceiling for reasoning plus final JSON. This follows the user's correction that 4,096 tokens is insufficient when reasoning is allowed.
   3. The request wrapper selects the ceiling after determining the actual reasoning mode. It reduces either ceiling when necessary to fit the remaining context. Long-record planning reserves the larger allowance.
   4. Existing configurations remain compatible: omitting `extraction_reasoning_max_tokens` retains the shared extraction ceiling; omitting `extraction_max_tokens` retains the legacy 75,000-token value.
   5. Streaming is enabled for this revision and the recommended example. Progress records distinguish generated reasoning from final-answer text using character counts. Provider token usage is saved when supplied, otherwise marked unavailable.
   6. The two-hour logical deadline remains unchanged. The 900-second transport timeout becomes a network read-inactivity limit during streaming. A watchdog closes a stream at the overall deadline, including a stream kept open by heartbeat bytes. Disconnected, unfinished, or token-truncated streams cannot become successful measurement checkpoints.

2. **Response-shape handling**
   1. A complete, exact feature-value map for the one requested patient can be wrapped into the required `rows` structure. Equivalent single-row wrappers are also accepted.
   2. Every requested value must already be present. Explicit row IDs, value types, and category ontologies remain subject to normal validation. The repair cannot supply missing measurements or assign another patient's values to the requested patient.
   3. Wrapper-only errors no longer automatically trigger high reasoning after five repair turns. Other validation failures retain the existing escalation policy and can use the larger reasoning ceiling.

3. **Failure recovery and audit trail**
   1. Logical requests receive unique IDs linked to patient IDs, feature IDs, and checkpoint paths. Journals record response/transport attempts, request settings, progress, elapsed time, final token usage, and failures.
   2. Request event journals contain counts and identifiers rather than patient notes or generated reasoning text.
   3. A patient or page whose request exhausts its budget is deferred while independent tasks continue. One retry pass then reuses that task's completed feature/chunk checkpoints. `extraction_deferred_retry_passes` controls this count; zero restores immediate abort.
   4. Remaining unresolved tasks prevent aggregate extraction completion and fitting. Consecutive exhausted tasks trip a circuit breaker during a persistent outage. Local configuration and programming errors still abort immediately.

4. **Comparison provenance and checkpoint reuse**
   1. The former worker was deliberately stopped before implementation. The model services were left running.
   2. Original source/configuration manifests and pre-change source copies remain preserved. The restart requires the explicit `results/revisions/reliability_v2/manifest.json` revision; it does not overwrite the original manifest or disable its checks.
   3. The revision hashes the new runtime, records configuration differences, and preserves an inventory of prior checkpoints. Model identities, candidate definitions, outer/inner splits, selection policies, nuisance fitting, and forest comparisons remain fixed.
   4. Completed patient and feature measurements remain eligible only through their existing model/text/definition fingerprint checks. An incomplete serial chunk additionally depends on its output reservation; if that changes its fingerprint, normal re-extraction is required.
   5. All retained refreshed measurements came from the requested 26B A4B extractor. The operational extraction policy changes partway through this refresh and is explicitly recorded. Every comparison arm still uses the same resulting measurements; this is not a separate randomized evaluation of extraction settings.
   6. The launcher completes all 60 forest fits, freezes prediction hashes, then performs oracle evaluation and produces the selection-comparison report. Oracle values remain excluded from fitting and selection.

5. **Validation**
   1. Full repository suite: **814 passed** after the separate reasoning-budget change.
   2. Reliability and experiment checks: **54 passed**, including 16 experiment checks in addition to tests already included in the full suite. These cover revision guards, shared comparison inputs, budget selection, context limits, incomplete streams, patient identity, deferred retry, unresolved failures, and checkpoint reuse.
   3. Live checks against `sn4622130540:8001` passed in both non-thinking and high-reasoning modes. Both extracted the two known values in a small artificial note correctly. Server records confirmed ceilings of 4,096 and 32,768 respectively, streamed progress, final `stop` finishes, and token usage (30 and 616 generated tokens).
   4. A package wheel built successfully from a temporary copy of the working source. Whitespace checks passed. The live checks verify transport and validation, not full-cohort extraction accuracy or the elimination of every long-tail timeout.

6. **Execution record**
   1. The revised comparison resumed at 01:11:35 UTC on September 20 (21:11 Eastern on September 19), with supervisor PID 2397100 and comparison child PID 2397144. The revision hashes passed validation, and fold 1 refresh was active at 01:12:01 UTC. Current state is recorded in `results/revisions/reliability_v2/runner_started.json` and `runner_state.json`.
   2. The pre-restart inventory contains 16,463 checkpoint files: completion records include 130 complete patient batches, 5,199 complete feature batches, and 174 complete serial chunks. Copies of 574 serial checkpoint/manifest files are retained under `before_serial_checkpoints/`, with hashes in `serial_checkpoint_backup.json`. These counts overlap at different checkpoint levels and must not be added as independent measurements.
   3. Detailed smoke-check evidence is in `results/revisions/reliability_v2/smoke/`.
   4. Completion monitoring remains attached to this task and reports completion or a new actionable failure, rather than unchanged progress.

7. **First-hour follow-up: remaining validation fallback**
   1. At 22:13 Eastern, the revised run was progressing without observed transport failures, but 32 requests had exhausted all format repairs and entered the existing null/prior-state fallback. The unresolved-request fitting gate applies to transport/deadline exhaustion; exhausted schema repairs still follow that separate pre-existing path. See `FORMAT_FAILURE_FOLLOWUP_2026-09-19.md` for counts, limits, and evidence.
