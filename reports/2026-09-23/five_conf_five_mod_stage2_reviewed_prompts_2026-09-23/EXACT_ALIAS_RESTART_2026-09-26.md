# Exact-duplicate consolidation restart — September 26, 2026

The user requested a pause, alphabetical/semantic-only grouping, an exact-duplicate merge prompt, a code commit, and a fresh consolidation start.

- The previous client (PID 201033) was stopped before changes.
- Every round now uses approximately **60% semantic and 40% alphabetical batches**, with no random grouping in the mixed strategy. Nonzero `random_fraction` settings produce a configuration error.
- The prompt permits only duplicate clinical attributes differing in spelling, synonymous wording, or abbreviation. Differences in quantity, scale, categories, method, timing, context, and specificity stay separate; uncertain or internally conflicting candidates remain unchanged.
- `cr_clearance` and `creatinine_clearance` can merge when their descriptions agree. Creatinine clearance and estimated GFR remain separate.
- The human-readable response must choose an existing member name as its canonical label; Python checks this constraint.
- A consolidation-specific prompt revision and grouping schema invalidate previous consolidation results without changing the discovery prompt or discovery checkpoint identity.
- Stopping remains two successful rounds each removing less than 0.5%, after at least three rounds, with a hard cap of 55 rounds.

All five prior mixed-consolidation directories were renamed with the suffix `before_exact_aliases_20260926T122441Z`. No fold had entered operationalization or downstream extraction. Discovery, evidence, and the earlier legacy-consolidation directories remain available.

| Fold | Fresh initial candidates | Previous completed round | Previous remaining candidates |
| --- | ---: | ---: | ---: |
| 1 | 4,467 | 19 | 1,552 |
| 2 | 4,234 | 30 | 1,279 |
| 3 | 4,497 | 24 | 1,465 |
| 4 | 4,328 | 25 | 1,393 |
| 5 | 4,406 | 25 | 1,470 |

Validation: **258 tests passed** across the Stage 2 workflow, candidate-consolidation, live-concurrency, and runtime-regression suites. Existing dependency and small-sample warnings occurred. The tests cover alphabetical/semantic allocation, deterministic coverage, merge-response name validation, retaining the distinct estimated-GFR candidate in a duplicate-merge fixture, checkpoint reuse, and stopping behavior. They do not establish that the LLM will always judge clinical equivalence correctly.

The experiment configuration retains `sn4622130540:8001`, automatic model detection and developer sampling, `xhigh` reasoning for the detected Qwen model, and live limits of 96 interpretation / 8 extraction / 96 total requests. Restart occurs after the code commit. The local `exact_alias_restart_2026-09-26.json` record captures the archive paths, launch PID, and committed revision.
