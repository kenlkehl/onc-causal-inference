# Abandoned fold-1 modifier-count experiment — September 22, 2026

1. Stopped at the user's request before final feature selection and estimation.
   - Six numerical jobs, five training-only rankings, and 120 count/seed validation fits completed.
   - Minimum mean validation R-loss favored 12 modifiers while retaining 189 confounders.
   - The full-training LLM ranking was unfinished. No final modifier list, production forest fits, or new held-out performance results were produced.
2. Interim direct oracle recovery was poor.
   - Every inner-fold top-12 list recovered NLR alone: 1/5 oracle modifiers.
   - The unfinished full-training ordering limited final direct recovery to at most NLR; all five oracle confounders remained retained.
   - The requested interim role comparison used previously published oracle-to-candidate mappings, not patient-level oracle outcomes or effects, and did not alter selection.
3. Checkpoints and audit records are preserved.
   - See `count_validation_snapshot_2026-09-22.json`, `interim_oracle_role_comparison_2026-09-22.json`, and `cancelled_2026-09-22.json`.
   - The model-selection processes and live monitor were terminated. The shared vLLM service was left available.
4. The next implementation will restrict modifier screening to estimated propensities 0.1–0.9 and compare causal forests with penalized outcome models containing treatment interactions.
