# Selection comparison execution — September 19, 2026

1. **Status at launch**
   1. The complete measurement refresh started at 20:37 UTC on September 19, 2026. This document is a launch record, not a completed results report.
   2. The extractor was initially starting up, then advertised `nvidia/Gemma-4-26B-A4B-NVFP4` at `http://sn4622130540:8001/v1`. The adjudicator advertised `gemma4-31b`, rooted in `nvidia/Gemma-4-31B-IT-NVFP4`, at `http://sn4622130540:8000/v1`.
   3. Early extraction requests returned successfully and generated validated checkpoints. Some responses required the existing format-repair procedure.
   4. The saved extraction setup processes one patient and ten features per request, with 32 concurrent workers. The five initial candidate catalogues imply approximately 177,000 such batches for all training and test measurements, before adjustments from refinement or retries. At early observed throughput, the refresh may take several days; this is not a completion-time guarantee.

2. **Running process**
   1. Interpreter: `/home/klkehl/thisenv/bin/python`. The repository's `.venv` lacks required analysis packages.
   2. Launcher process at startup: host PID `2151402`, started September 19 at 16:28:22 America/New_York. The measurement/fitting child at startup was host PID `2161190`, started at 16:37:12.
   3. The launcher command is `python -u reports/2026-09-19/selection_comparison_2026-09-19/launch.py`. Its child runs the absolute path to `compare.py run` in this directory.
   4. Process identifiers are historical evidence. Confirm the exact command and start time before treating a process as this job. The default filesystem sandbox can hide host processes; absence there does not establish that the worker stopped.
   5. The launcher automatically executes preparation, refreshed extraction and selection, all 60 comparison forest fits, prediction freezing, and post-hoc evaluation. Report rendering follows after evaluation through the completion follow-up.

3. **Progress and checkpoints**
   1. `results/status.json` records the current major phase and outer fold. A fold may remain in the extraction phase for many hours.
   2. `results/execution.log` records individual requests and validation repairs. Recent successful requests are progress even when the major phase is unchanged.
   3. `results/refresh/outer_001/ontology_supervision/round_001/extraction/batches/` illustrates the initial extraction layout. Each patient directory contains checkpointed `feature_batches`; its own `complete.json` appears after all its feature batches finish.
   4. `results/inputs/source_manifest.json`, `splits.json`, `refresh_config.json`, and `experiment.json` freeze the experiment inputs. Successful checkpoints must be retained.
   5. `results/predictions_frozen.json` must contain all 60 prediction files before oracle evaluation. The four policies share nuisance predictions and eligible rows.

4. **Completion**
   1. Confirm evaluation has completed and every frozen prediction hash still matches. Then run `report.py` in this directory with `/home/klkehl/thisenv/bin/python`.
   2. Inspect the generated `selection_comparison_report_2026-09-19.md` and `results/comparison_summary.json`. Verify all five folds, four policies, and three forest seeds, and confirm shared nuisance hashes and held-out rows.
   3. Check the post-hoc mapping of direct modifier concepts against any revised measurement definitions before interpreting retention counts. Report any ambiguous lineage rather than assuming a retained identifier means faithful measurement.
   4. Report CATE RMSE, correlation, bias, effect spread, common-nuisance R-loss, and feature retention. Explain that this is an exploratory selector comparison on refreshed measurements, not a controlled comparison of the new extractor with E4B.

5. **Recovery boundaries**
   1. Do not launch a duplicate while the original worker is active. Do not change production artifacts, extraction settings, frozen input hashes, or candidate admission rules to improve observed outcomes.
   2. If the worker has actually exited, inspect the concrete failure before resuming. Reuse compatible checkpoints; do not silently bypass an input fingerprint mismatch.
   3. Any necessary experiment implementation correction must retain the old code and hashes, record the reason, and establish checkpoint compatibility. No oracle values may enter fitting, prompts, or selection.
   4. Do not change or restart the user's model servers. A service outage can be handled by the existing retries or a checkpointed resume once the same model identity is available.

6. **Checks completed before results**
   1. Nine focused tests passed for admission rules, exact investigator locks, missing-decision failures, independence from oracle labels, row alignment, shared nuisance losses, common nuisance fitting, resumability, and numerical equivalence with the final forest inside `CausalForestDML`.
   2. Report generation is checked separately with fabricated fixture values; no experiment oracle values are required for this check.

7. **Transport interruption and planned recovery, September 19 at 22:51 UTC**
   1. Fold 1's initial training extraction encountered a `Stage2RequestExhaustedError`: a logical extraction request exhausted its 7,200-second deadline after a network read timeout. The affected patient batch was `batch_00002`, row ID 1; its first 21 feature batches were complete and feature batch 22 had only an input checkpoint.
   2. The production extraction handler cancelled queued work and set its cancellation flag. Already active logical requests can finish or exhaust their own deadlines before the original process exits. The original worker was still draining when inspected at 22:56 UTC; it was not killed.
   3. `resume_after_timeout.py` was started as host supervisor PID `2275515` (initial tool session `25289`). It waits for the exact original worker commands to disappear, checks for another comparison worker, and permits only one resume for this incident. Its drain wait is bounded at three hours. The supervisor itself does not send extraction requests while the original worker is active.
   4. This recovery changes no scientific source, prompts, models, timeouts, concurrency, selection rules, or frozen configurations. `compare.py run` revalidates the original source hashes and the existing extraction code reuses only compatible checkpoints. The failed response is not converted into a measurement or a null value.
   5. Recovery state and output are saved under `results/recovery/transport_timeout_2026-09-19/`: `state.json`, `events.jsonl`, `resume_started.json` when a resume actually begins, and `worker_output.log`. A successful resumed run also performs evaluation and report rendering; the scheduled follow-up must still inspect and deliver the report.
   6. If this single resume fails, inspect the new concrete failure before taking another action. Do not duplicate the supervisor, bypass fingerprints, silently increase timeouts, or repeat the already reported incident as a new alert while it is merely draining.

8. **User-authorized reliability revision, September 19 evening (Eastern time)**
   1. The user requested the recommended reliability changes, then clarified that reasoning needs a larger output allowance than 4,096 tokens. The previous comparison child (host PID 2284061) was deliberately stopped at 00:32:46 UTC on September 20. This authorized revision supersedes the earlier unchanged-input-only recovery boundary for the documented implementation changes.
   2. `RELIABILITY_CHANGES_2026-09-19.md` describes the implementation and validation. The revised profile uses 4,096 output tokens for ordinary extraction, 32,768 total output tokens when reasoning is enabled, streaming progress, narrowly validated wrapper repair, and one deferred retry pass. Model services, scientific comparison policies, and the two-hour logical deadline remain unchanged.
   3. Original manifests/configuration and original source copies remain preserved. `results/revisions/reliability_v2/manifest.json` records the explicit revision and checkpoint inventory. Run the comparison only with that revision argument; do not re-run original preparation or modify the frozen manifests.
   4. `resume_reliability.py` is a one-shot launcher for refreshed extraction/fitting, post-hoc evaluation, and report generation. Its exact command, current child PID, timestamps, and exit status are recorded in `results/revisions/reliability_v2/runner_state.json`; `runner_started.json` guards against accidental duplicate launch. Verify current processes before any subsequent recovery.
   5. Validation after the separate reasoning-budget change: 814 repository tests passed, 54 reliability/comparison checks passed (including 16 comparison checks beyond the repository suite), and a package wheel built successfully. Live ordinary and reasoning extraction checks both passed against the user's 26B A4B server.
   6. The transport policy changed partway through the measurement refresh. Record this operational revision in the final comparison report; every selector still shares the same resulting measurements, nuisance predictions, and evaluation patients.
   7. Revised supervisor dispatched at 01:11:34 UTC on September 20 (September 19 Eastern), PID 2397100; child PID 2397144 started one second later. Revision validation passed at 01:11:53 UTC, and fold 1 refresh became active at 01:12:01 UTC. The completion heartbeat was reactivated. The pre-restart inventory records 5,199 completed feature batches and 130 completed patient batches; 574 serial checkpoint files were also backed up before launch.

9. **User-authorized structural-repair reasoning revision, September 20**
   1. The user requested reasoning for structural errors too. Repairs 6–15 now enable at least high reasoning for all validation errors, using the existing 32,768-token reasoning allowance. `STRUCTURAL_REPAIR_REASONING_2026-09-20.md` records the change and its limits.
   2. The previous worker was deliberately stopped at 09:41:22 UTC. The restart inventory recorded 532 completed patient checkpoints in fold 1, with 565 patient batches started. Completed measurements, including audited fallback results, are retained; compatible incomplete feature/chunk work resumes normally.
   3. `results/revisions/structural_reasoning_v3/manifest.json` supersedes `reliability_v2` for future execution and authenticates the earlier manifest and pre-change source copies. The runtime configuration is byte-identical to the previous revision. Do not rewrite either frozen manifest.
   4. The active launcher for this revision is `resume_structural_reasoning.py`; consult its `runner_state.json` under the new revision directory for current process identity. Earlier launchers remain stopped and must not be used. The completion follow-up must include this policy change and the earlier fallback burden in the final report.
   5. The new supervisor was dispatched at 09:51:24 UTC as host PID 2846055; its child PID 2846074 started one second later. The revision guard passed at 09:51:40 UTC, and fold 1 refresh resumed at 09:51:45 UTC. The hourly completion follow-up was updated and reactivated for this revision.
   6. Validation passed: 817 repository tests, 316 focused checks, the package build, and a live structural-repair escalation check on the 26B extractor. The live request journal confirms that repair 6 used high reasoning and the 32,768-token allowance.

10. **Adjudication prompt-budget recovery, September 20 at 22:22 UTC**
   1. The structural-reasoning worker stopped at 21:55:16 UTC on its first binding role prompt: 126,503 characters exceeded the existing 100,000-character limit before HTTP transport. Fold 1 training and the standard selected-feature held-out extraction were already saved.
   2. `ADJUDICATION_BUDGET_RECOVERY_2026-09-20.md` describes the comparison-only preflight correction. It preserves complete candidate evidence and chooses a smaller candidate batch size when needed. Fold 1 uses 24 batches of at most 15 candidates; all 352 candidates are covered, and the longest prompt is 96,474 characters. No model, scientific admission rule, or frozen configuration changed.
   3. `results/revisions/adjudication_budget_v4/manifest.json` supersedes v3 for further execution and preserves all prior provenance. Its SHA256 is `34edc68314bea5ab97954427fc862f478709cc1195c481aca9b1166035a00a4d`.
   4. Active launcher: `resume_adjudication_budget.py`. Supervisor PID 3435067 and child PID 3435118 started at 22:22:20 UTC. Revision validation passed at 22:22:30 UTC and fold 1 resumed at 22:22:34 UTC. Consult the v4 runner state for current identity; never reuse older launchers.
   5. The post-restart hash check matched all 2,600 saved patient completion markers across four extraction scopes and 14 aggregate files. Counts across repeated training passes are not unique patient counts. Compatible measurements remain reusable through their original fingerprint checks.
   6. Validation passed: 29 focused tests, all 817 repository tests, package build, and offline preflight on the exact saved aggregate evidence. The hourly follow-up now tracks v4. The final report must disclose the changed LLM batch context alongside earlier extraction-policy revisions.
