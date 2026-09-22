# Repository checkpoint — September 22, 2026

1. Automatic modifier-count selection is implemented for multi-model Stage 2.
   - Each count-validation fold rebuilds numerical evidence and an LLM ranking
     within its training portion, then compares ranked prefixes by held-out R-loss.
   - Minimum mean R-loss is the default; a paired one-standard-error rule is
     optional. Retained confounders and investigator-locked roles are preserved.
   - Numerical evidence, rankings, and count/seed fits have resumable checkpoints.
     Schema and preflight checks protect selection-policy and tree-count changes.
   - Documentation, the example configuration, and focused tests are updated.
2. Validation completed before the live experiment was launched.
   - The relevant Stage 2 and workflow suite passed: 319 tests.
   - After the final resume/tree-count change, 86 targeted tests passed.
   - New modules and tests passed targeted lint and formatting checks.
3. Earlier fold-1 experiments are saved alongside this checkpoint.
   - The broad multi-model review retained 189 confounders and 224 modifiers;
     its production refit was superseded by the consolidation experiments.
   - The compact consolidation and conservative-confounder/moderate-modifier
     experiments have completed reports under `reports/2026-09-21/`.
   - Their frozen evidence, decisions, metrics, manifests, and reproduction
     scripts are included. Model bundles, numerical caches, and logs stay local.
4. The nested fold-1 experiment is still running at this checkpoint.
   - Snapshot observed at 2026-09-22 13:03 UTC (09:03 America/New_York).
   - All six numerical evidence jobs are complete. The full-training aggregate
     evidence exactly reproduces the prior frozen numerical report.
   - The isolation audit checked 915 nested numerical row sets: parent-validation
     and outer-test patients are excluded from their training evidence.
   - Each of five count-validation folds has initially reviewed all 352
     candidates. LLM merge requests are still assembling the fold rankings.
   - No modifier count or new held-out causal-estimation result is available yet.
   - The run retains 189 confounders and compares budgets of 0, 4, 8, 12, 16,
     24, 32, and 64 modifiers with three forest seeds per fold.
   - Once selection finishes, the existing runner will freeze roles, fit three
     native and three fixed-residual forests, and evaluate oracle recovery and
     held-out ITE performance. The commit does not interrupt that run.
5. The live [status](STATUS_2026-09-22.md) continues to change after this snapshot.
   The [protocol](PROTOCOL_2026-09-22.md) records the frozen scientific procedure.
   Later results require another commit and are not claimed by this checkpoint.
