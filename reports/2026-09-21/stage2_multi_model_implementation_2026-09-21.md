# Stage 2 multi-model selection: implementation and validation

Date: September 21, 2026.

1. **Repository checkpoint before implementation**
   1. Commit `de37fff` preserves the completed estimation experiments, reports,
      scripts, coefficient tables, metrics, and provenance.
   2. Generated model bundles and large arrays remain local and are ignored by
      Git in the dated report directory.

2. **New selection mode**
   1. Enable `stage2.statistical_selection.selection_mode = "multi_model"`.
   2. Seven families contribute evidence: univariable models, grouped penalized
      main effects, grouped penalized outcome interactions, orthogonal linear
      models, candidate R-learners, predictive forests, and causal forests.
   3. Original inner folds, repeated patient subsets, and balanced forest feature
      subsets expose candidates to complementary models. Support counts are
      conditional on actual evaluation; missing fits are not negative votes.
   4. The LLM reviews themes across candidate batches, then assigns each candidate
      confounder/modifier/both/neither roles with appropriate empirical citations.
      Every candidate receives a decision; themes do not alter measurements.
   5. No single numerical method, p-value, or frequency threshold is binding.
      Investigator locks remain exact; LLM adjudication is required in this mode.

3. **Boundaries and recoverability**
   1. The selector projects only observed treatment and outcome from the source
      dataset. Training/validation partitions are checked before fitting.
   2. Elastic-net nuisances are cross-fitted within each inner training partition;
      encoders and penalty choices use training data only.
   3. Numerical fits, theme requests, and role decisions have separate atomic,
      fingerprinted checkpoints. Corrupted responses fail visibly. Changed
      policies cannot silently reuse a completed selection.
   4. Prompts contain allowlisted definitions and aggregate evidence. Row values,
      patient IDs, predictions, dataset paths, exception logs, and oracle metadata
      are excluded.

4. **Validation**
   1. Final full repository suite: **797 passed**, 149.84 seconds.
   2. Real-model tests cover binary/continuous outcomes, categorical encoding,
      all seven families, nested nuisances, group permutations, train/validation
      separation, and checkpoint reuse after changing outside labels/oracle data.
   3. Controlled LLM responses test cross-batch themes, complete coverage, locked
      roles, citations, invalid response rejection, and response integrity.
   4. The new complete example compiles into an active `full` workflow with
      `multi_model` selection. Disabled adjudication is rejected. Policy changes
      require guarded reselection.
   5. New modules pass formatting checks; `git diff --check` passes. Wheel and
      source distribution builds succeeded from a clean snapshot of package
      sources. The wheel's new modules match the working source hashes.

5. **Use and remaining empirical work**
   1. [Detailed design and configuration](../../docs/stage2_multi_model.md)
   2. [Complete example configuration](../../example_configs/research_all_evidence_multi_model.json)
   3. Existing extracted measurements can be reused through the established
      guarded `--stage2-reselect` workflow with compatible model identities.
   4. This turn implemented and tested the mode. It did not run a production
      selection benchmark or a live LLM adjudication experiment. Improved
      confounder/modifier recovery and causal estimation remain empirical questions.
