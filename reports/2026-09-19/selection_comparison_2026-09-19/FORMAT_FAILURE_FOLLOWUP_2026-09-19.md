# Remaining extraction format failures — September 19, 2026

1. **Observation after the reliability revision**
   1. At 22:13 Eastern, the revised worker was still processing fold 1 and had logged 1,245 validated logical requests since restart. Recent requests continued to complete successfully.
   2. No HTTP/stream failure or whole-extraction abort had been logged in this observation window. This is an early observation, not proof that all future timeouts are eliminated.
   3. Thirty-two logical requests exhausted all 15 format-repair turns. Every recorded final error was `extraction response requires a rows array`.
   4. The existing fallback warnings counted 272 fields replaced by null at the individual request boundary. This is not a count of final unique missing measurements: a serial chunk can instead retain its previously validated state, and later chunks or supervision can change final measurements.

2. **Why the run continues**
   1. The new deferred-retry gate applies to transport/deadline exhaustion (`Stage2RequestExhaustedError`). Those unresolved tasks prevent completion and fitting.
   2. Exhausted schema repairs use a different exception, `Stage2ResponseValidationError`. The pre-existing extraction path records an issue and returns nulls; serial extraction preserves prior validated chunk state where available. The revised worker retains that behavior.
   3. Therefore, the earlier statement that unresolved requests block fitting was too broad. It applies to transport/deadline exhaustion, not this existing validation fallback.

3. **Implication and action**
   1. A remaining response-format failure can still remove patient-feature information before statistical selection. Successful transport does not establish measurement completeness.
   2. The narrow wrapper normalization accepted 13 responses in the same snapshot, but did not resolve every malformed response. The exact unresolved wrapper cannot be inferred from the error alone; the request journal intentionally omits raw model output.
   3. The ongoing experiment, original checkpoints, frozen revision, and model services were left unchanged. No duplicate worker was started.
   4. Further corrective work should first reproduce one failing request in isolation and inspect its response shape, then test an exact schema/shape fix and explicitly version any resulting run change. No current selector results or oracle data are needed for that investigation.
   5. This issue is reported once as a remaining measurement-quality problem. Ordinary increases in the same known failure count should not generate repeated notifications; a new failure mode, substantial deterioration, or run termination should.

4. **Evidence**
   1. `results/revisions/reliability_v2/format_failure_followup.json` freezes the observation window, exact errors, counts, and example fallback warnings.
   2. `results/monitoring.jsonl` records the corresponding progress snapshot. The experiment remains in the extraction stage; no four-way CATE results are yet available.
