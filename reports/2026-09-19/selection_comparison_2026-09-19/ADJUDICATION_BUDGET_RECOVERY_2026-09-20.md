# Binding adjudication prompt-budget recovery — September 20, 2026

1. **Failure and completed work**
   1. At 21:55:16 UTC, the comparison worker exited during fold 1's binding LLM adjudication. The first batch's rendered prompt contained 126,503 characters, exceeding the existing 100,000-character request limit.
   2. The legacy adjudicator partitions by a maximum of 20 candidates per request. The comparison driver had not checked whether their complete aggregate evidence fit the request limit. The failure occurred before an HTTP attempt; no binding role decision had been produced.
   3. Fold 1's training measurement refresh, statistical evidence, and standard selected-feature held-out extraction were complete. The latter covers 200 patients and 125 measurement dependencies. Additional held-out candidate measurements for the four-way comparison remain pending.
   4. The failed supervisor and worker exited. Their final state, traceback, source hashes, and failed role prompts were archived under `results/revisions/adjudication_budget_v4/` before editing.

2. **Comparison-driver correction**
   1. Preflight every rendered role prompt using the existing system prompt, payload constructor, canonical JSON rendering, and transport character-count convention.
   2. Choose the largest uniform candidate batch size at or below the configured maximum for which every prompt fits the unchanged character limit.
   3. Preserve each complete allowlisted candidate evidence object, the global evidence boundary and methodology, candidate order, the binding role decision rules, and investigator locks. A runtime equality check verifies that changing the effective batch size has not changed the evidence package.
   4. Fail before sending any adjudication request if even one complete candidate cannot fit. This correction does not truncate additional evidence or increase the request limit.
   5. Save the configured and effective batch sizes, rendered lengths, evidence fingerprint, and complete candidate-to-batch mapping in each fold's `comparison_llm/request_budget.json`.
   6. The existing adjudicator fingerprints its effective policy and individual batch payloads. Resume requires the same recorded request-budget plan; stale decisions cannot silently substitute for a changed plan.

3. **Real-evidence preflight**
   1. All 352 fold 1 candidates remain in their original order with identical allowlisted evidence.
   2. The original 20-candidate batches reached 127,032 characters. Batches of at most 15 candidates produce 24 requests with a maximum of 96,474 characters.
   3. This preflight sent no HTTP requests, produced no role decisions, and opened no oracle values. Its evidence is saved under `results/revisions/adjudication_budget_v4/validation/`.

4. **Scientific and checkpoint compatibility**
   1. Only the isolated comparison driver changes. Production source, model identities, measurement policies, nuisance models, forest settings, admission rules, and all frozen configuration values remain unchanged.
   2. The existing extraction and statistical checkpoints remain subject to their normal compatibility checks. A hash inventory records completed patient markers and aggregate measurement/selection artifacts before the recovery.
   3. The new manifest extends the original-to-current source guard and authenticates the previous revision manifest, configuration, archived driver, failed prompts, validation records, and checkpoint inventory. Prior manifests are preserved.
   4. LLM batch context is smaller than originally configured. Report this operational grouping correction alongside the two earlier extraction-policy revisions; it is not a fifth experimental arm.
   5. Earlier audited extraction fallback results remain retained. No oracle data enter the recovery or subsequent fitting.

5. **Validation**
   1. All 29 focused comparison and role-adjudication checks passed, including 23 experiment checks outside the repository suite. Checks cover complete evidence/order preservation, safe prompt sizes, unchanged small batches, checkpoint reuse, changed-plan rejection, invalid limits, and oversized single-candidate failure before transport.
   2. All 817 repository tests passed. Existing dependency and numerical warnings remained warnings.
   3. The package wheel built successfully, and all 92 packaged Python sources matched the working source. Source whitespace checks passed.

6. **Recovery execution**
   1. `resume_adjudication_budget.py` is the one-shot launcher for this revision. Earlier launchers must remain stopped.
   2. The launcher runs the comparison with the explicit `adjudication_budget_v4` manifest, then evaluation and report generation. Current process identity and completion are recorded in that revision's `runner_state.json` and `runner_complete.json`.
   3. Confirm current processes before recovery, preserve completed measurements, and verify checkpoint hashes after restart. All 60 comparison prediction files must still be frozen before oracle evaluation.
