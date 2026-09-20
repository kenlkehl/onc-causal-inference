# Extraction timeout diagnosis — September 19, 2026

1. **Current status, checked around 20:24–20:26 America/New_York**
   1. The unchanged-input resume is active in fold 1's first training extraction round. The snapshot showed 127 of 800 training patients complete, 32 more started, and 5,046 completed ten-feature batches. No comparison forests have been fitted.
   2. The specific feature batch that exhausted its deadline in the original attempt completed after resumption at 23:11:38 UTC. Earlier successful checkpoints retained their original completion timestamps.
   3. Only one fatal extraction interruption had been logged at this check, but individual request timeouts continued. Resumption recovered progress; it did not establish that the underlying latency problem was fixed.

2. **What the timeout means**
   1. `request_timeout=7200` gives one logical JSON extraction request a two-hour budget. Here that usually means one patient's ten-feature batch, including response repairs and transport retries.
   2. `request_attempt_timeout=900` limits an individual non-streaming HTTP attempt to 15 minutes of waiting for a response. The transport wrapper passes this timeout to the OpenAI-compatible client. It does not receive incremental generation progress.
   3. `transport_max_attempts=6` permits up to six HTTP attempts within a response attempt. Invalid returned responses can cause up to 15 repair requests; each repair gets its own transport-attempt counter, while the two-hour logical deadline remains shared.
   4. Consequently, `attempts_used=1` on the final exception describes the last transport-retry cycle. It does not mean the entire logical extraction had only one HTTP attempt.
   5. Local semaphore queue time is excluded from the logical deadline in the current code. The observed error cannot be attributed to that earlier class of local queue accounting problem merely from its two-hour duration.

3. **Evidence of recurrence in this run**
   1. A log snapshot around 20:26 Eastern contained 106 transport-retry messages, all reporting `APITimeoutError` with a 900-second attempt timeout. These are retry events, not necessarily 106 distinct failed batches.
   2. The same snapshot contained 1,377 validation-failure messages, including 866 saying the extraction response required a `rows` array. Repeated attempts on one request contribute multiple messages.
   3. There were 6,373 logged extraction attempts in the initial non-thinking mode and 331 in high-reasoning mode. The counts include repairs and network retries and are not counts of unique patient measurements.
   4. The server metrics snapshot at 00:26:13 UTC on September 20 showed 32 running requests, zero waiting requests, and approximately 20% KV-cache use. It was serving ordinary requests successfully; this snapshot does not establish the resource state during every timeout.
   5. Successful requests averaged approximately 46 seconds end to end and 251 generated tokens. The metrics describe completed requests and therefore underrepresent requests that were still running or interrupted. They cannot rule out a slow tail.

4. **Most plausible mechanism, not yet proven per request**
   1. Repeated malformed-response repairs can promote an extraction to high-reasoning mode after five failed repairs. The output allowance remains 75,000 tokens, although a ten-feature JSON result is normally much shorter.
   2. A long reasoning response or generation loop can therefore outlast a 15-minute non-streaming attempt. A retry starts another model call; after enough retries and repairs, the shared two-hour budget expires.
   3. The current logs interleave concurrent requests without a stable per-request identifier. They also omit per-call generation counts and latency. It is therefore not possible to prove from these logs that every timeout occurred in thinking mode, or to distinguish slow generation from a server stall for the specific fatal attempt.
   4. The original two-hour interruption propagated out of the extraction task, cancelled queued work, and waited for active requests to drain. This converts one unresolved request into an interruption of the whole fold's extraction pass, despite its valid checkpoints remaining reusable.

5. **Recommended reliability changes for a separately recorded revision**
   1. Add request identifiers, patient/feature-batch identifiers, retry and repair counters, reasoning mode, elapsed time, token counts, and finish reasons to durable request records. Preserve failure records as well as successful responses.
   2. Address the repeated response-shape failures with a validated extraction schema or narrowly defined, tested normalization of unambiguous wrappers. Do not infer missing measurements or silently accept malformed data.
   3. Distinguish stalled calls from active generation, and explicitly align extraction output and reasoning budgets with the request timeouts. Streaming progress is one possible transport implementation, provided the overall deadline and final validation remain enforced.
   4. Consider checkpointing and deferring an exhausted individual request while other independent extraction tasks finish. Require every necessary measurement to resolve before fitting; do not silently drop difficult patients or features.
   5. Increasing the two-hour ceiling alone would leave both the recurring malformed responses and the failure granularity unchanged.

6. **Actions taken**
   1. The resumed experiment remains on its original frozen settings and source hashes. No timeout, prompt, model, admission rule, or production source was changed during this diagnosis.
   2. The previously authorized one-time resume is running. The existing completion follow-up will inspect any further failure before another recovery attempt.
   3. Supporting evidence is in `results/execution.log`, `results/monitoring.jsonl`, `results/recovery/transport_timeout_2026-09-19/`, and `results/extractor_metrics_20260920T002613Z.txt`.
