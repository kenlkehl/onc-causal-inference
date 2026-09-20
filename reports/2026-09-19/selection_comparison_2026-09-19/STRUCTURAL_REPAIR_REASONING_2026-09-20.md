# Structural extraction repairs with reasoning — September 20, 2026

1. **Authorized change**
   1. The user requested reasoning escalation for structural extraction errors as well as other validation failures.
   2. The initial response and repairs 1–5 retain the configured extraction reasoning policy, currently `none`. Repairs 6–15 now use at least `high` reasoning for every validation failure, including missing `rows`, malformed row objects, and missing `values` objects.
   3. Reasoning-enabled extraction uses the existing 32,768-token total allowance for reasoning plus final JSON, bounded by available context. Ordinary extraction retains its 4,096-token ceiling.
   4. Repairs continue to receive the concrete validation error, the original prompt, and the latest invalid response when space permits. The shared 7,200-second deadline and 15-repair limit remain in effect.

2. **Implementation and documentation**
   1. Removed the structural-error exemption in `oci/inference/plain_handoff_stage2.py`.
   2. Updated the workflow and quickstart descriptions to explain that structural errors also escalate.
   3. Regression checks cover all three previously exempted errors and a nonstructural error. They verify the escalation threshold, separate output ceilings, original prompt and validation feedback, and request audit continuity.

3. **Experiment revision and checkpoints**
   1. The previous comparison worker was deliberately stopped at 09:41:22 UTC on September 20, 2026. Its supervisor recorded the expected termination. Model servers were left running.
   2. Revision `structural_reasoning_v3` preserves the original experiment manifest, the earlier `reliability_v2` manifest, and copies of the source and documentation before this change. The existing validation guard authenticates original-to-current source hashes and the intervening revision through frozen input hashes.
   3. Runtime configuration is byte-identical to `reliability_v2`; only the code's structural-repair escalation policy changed. Candidate selection, nuisance modeling, forest settings, folds, and model identities retain their existing configuration.
   4. The stop-time inventory records 532 completed patient checkpoints out of 565 started in fold 1's first training extraction round. Compatible completed patient, feature, and serial-chunk checkpoints remain reusable through the normal fingerprint checks.
   5. Completed records that previously used the structural-validation fallback retain their values and failure audits. This revision does not retrospectively re-extract those completed records. Unfinished and new requests use the revised repair policy.
   6. The new one-shot supervisor is `resume_structural_reasoning.py`. Its manifest and process records are under `results/revisions/structural_reasoning_v3/`. It runs extraction and fitting, followed by evaluation and report generation. Earlier launchers must not be reused.

4. **Interpretation**
   1. This is an explicitly recorded change during the measurement refresh. All four selection methods will still share the resulting measurements and nuisance predictions.
   2. A live artificial-note check demonstrates that a missing-`rows` failure reaches a high-reasoning request and can return validated JSON. It does not establish that reasoning will eliminate all production formatting failures.
   3. The final comparison should report the repair-policy revisions and extraction fallback burden, including the earlier completed fallback records.

5. **Validation**
   1. The focused regression and comparison checks passed: 316 tests, including 16 comparison checks outside the repository suite.
   2. The full repository suite passed: 817 tests. Existing numerical and dependency warnings remained warnings.
   3. An artificial-note check injected six missing-`rows` failures, then sent repair 6 to the actual `nvidia/Gemma-4-26B-A4B-NVFP4` extractor at `sn4622130540:8001`. The journal confirms `reasoning_effort: high`, a 32,768-token output allowance, and a validated result.
   4. The package wheel built successfully using the installed setuptools backend; all 92 packaged Python source files matched the working source. The environment lacked the pip module, so the initial pip-based build attempt was superseded by the successful backend build.
   5. Source diff checks and the existing experiment revision guard passed before restart.

6. **Evidence**
   1. `results/revisions/structural_reasoning_v3/manifest.json` records the revised source hashes and preserved inputs.
   2. `results/revisions/structural_reasoning_v3/patient_checkpoints_before.json` records the completed patient markers at the restart boundary.
   3. `results/revisions/structural_reasoning_v3/validation/` contains regression results, the artificial-note live check and its request journal, and package build verification.
