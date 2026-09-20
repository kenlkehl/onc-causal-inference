# Repository test review — 2026-09-20

The cleanup removes obsolete and redundant tests while preserving the scientific
and recovery checks used by the supported workflows.

## Inventory

| Measure | Before | After |
| --- | ---: | ---: |
| Test modules | 40 | 40 |
| Active test functions | 585 | 557 |
| Collected cases, including parameterization | 817 | 788 |
| Lines in test modules | 29,092 | 27,079 |

The difference between functions and cases comes from parameterized inputs.
Distinct malformed schemas, boundary values, outcome types, and failure modes
remain valuable even when they share one test function.

## Removed cases

| Group | Cases removed | Reason |
| --- | ---: | --- |
| ColBERT candidate prefilter and registry | 8 | Private historical helpers have no callers in the production Stage 2 path; discovery now carries all candidates into consolidation. |
| Monolithic consolidation validator | 7 | The old prompt is explicitly retired, and its historical validator has no production callers. Current iterative consolidation tests remain. |
| Legacy fold review and signal pruning | 6 | These exercise the unreachable `_run_fold_analysis_legacy` helper chain. Current ontology supervision, statistical selection, and binding LLM role adjudication remain covered. |
| Duplicate retry and reasoning checks | 4 | Stronger request-level regressions exercise real timeout classification, full attempt budgets, feedback, and reasoning escalation. |
| Literal defaults, metadata, and retired constructor arguments | 4 | These pin example epochs, batch-size defaults, packaging strings, or a removed signature. Retained tests exercise batching and fitted nuisance models; an actual package build validates packaging. |
| **Total** | **29** | |

Also deleted 14 already-disabled `_retired_test_*` functions, an unused candidate
scorer mock, obsolete pruning mocks, and unused imports and fixture arguments.
The disabled functions never contributed to the reported 817 cases.

## Useful coverage preserved during consolidation

- Replaced the mixed legacy nuisance/effect-model test with a compact current
  continuous-outcome nuisance test. It checks held-out predictions against a
  known linear response and checks treatment-arm means with an empty design.
- Extended the existing extraction reliability cases through repair 15. All
  four validation-error variants check reasoning, token budgets, error feedback,
  request identity, and successful recovery on the last allowed repair.
- Kept category-list normalization in current operationalization coverage and
  architecture-provenance fallback in the supporting-evidence test.
- Retained oracle exclusion, held-out isolation, investigator locks, complete
  text and citation handling, resumability, streaming/deadline recovery,
  concurrency limits, and fitted elastic-net nuisance audits.
- Retained public compatibility and explicit-feature tests, including supported
  legacy LLM role selection. A historical name alone was not a removal criterion.

## Validation

- Baseline: 817 passed with line and branch coverage instrumentation.
- Focused cleanup checks: 329 passed; the three subsequently strengthened shared
  behavior checks also passed.
- Final full suite: **788 passed** in 216.92 seconds under coverage (baseline:
  217.59 seconds). This reduces maintenance burden; measured runtime is essentially
  unchanged.
- Compared coverage line by line and branch by branch. The 684 fewer executed
  source lines and 244 fewer executed branches are confined to the retired
  prefilter, monolithic consolidation, and legacy review/pruning helper chains.
  All other measured line and branch coverage is unchanged. Coverage collection
  covers the parent process; subprocess-only execution is not included.
- `git diff --check` passed.
- Package wheel built through the configured `setuptools.build_meta` backend;
  all 92 packaged Python source files matched the repository, and the wheel
  metadata recorded the MIT license.
- Verified all 84 frozen source/input files in the active experiment revision.
  The experiment continues with its original runtime code and configuration.

The review used test inventory, caller searches, comparison of retained test
bodies, and before/after line and branch coverage. Test count alone was not a
removal criterion. The initial snapshot was pushed as `45c7b0c`; this cleanup is
a separate change.
