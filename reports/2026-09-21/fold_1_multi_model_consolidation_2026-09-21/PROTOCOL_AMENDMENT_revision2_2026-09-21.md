# Global review output revision 2

1. Two completed responses failed the requirement that the LLM explicitly enumerate all 352 IDs in a partition. No global response or final selection was accepted, and no oracle values were opened.
2. The LLM now proposes only actual consolidation groups. Code leaves every unlisted candidate as a separate singleton and labels that addition as automatic; this does not assert empirical distinctness or assign a role.
3. The entire 352-candidate training-evidence matrix remains in the prompt. Candidate IDs must be valid, no candidate may appear in two proposed groups, at most one representative per group may receive each role, and evidence citations and the 32-modifier shortlist cap are unchanged.
4. Validation errors identify offending IDs, and invalid responses are retained for audit. The per-attempt HTTP wait increases from 900 to 1800 seconds for this large global review on the heavily shared server. The total logical deadline remains 7200 seconds, with the existing token-aware context guard and reasoning enabled.
5. The original script, prompt, runtime manifest, and request log remain preserved. This is a change to output bookkeeping and transport tolerance, with no tuning from oracle or outer-test results. The compact-size selection and final estimation protocol are unchanged.
