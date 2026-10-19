# Branch `feat/orthogonal-modifier-tests`: what changed (2026-10-14 to 2026-10-17)

Sources: `git log main..feat/orthogonal-modifier-tests` and the review notes and
specs in `~/oci_review/` (outside the repository; the Spec columns name files
there). Commits with a `Co-Authored-By: Claude` trailer were written with Claude
Code. The seven later commits listed at the end are Ken's.

## 1. `main` (pushed, 983b9e1): cleanup, 2026-10-14

Plan and decisions: `~/oci_review/REVIEW.md` §8; outcome: §9. 1139 tests passed afterwards.

| Commit | Contents |
|---|---|
| 11d8fa3 | Hygiene: untracked `reports/` (~97 MB), removed unused datasets and scratch notes |
| 41d8c80 | Removed dead code, retired modes, and legacy compatibility paths |
| 22962f2 | Fixed bugs B1–B13 (REVIEW.md §1): `--architectures` crash, all-NaN CATE SEs, HTR calibrator fit on in-sample predictions, hard-coded sparsity filter, silent unknown `stage2` keys, silent KFold fallback, and others |
| d42f8b2 | Unified Stage 2 nuisance estimation; added heterogeneity diagnostics |
| 0ca3071 | Fixed credential handling and synthetic-data resume; corrected stale descriptions |
| 983b9e1 | Restructured the docs into README + `docs/` |

## 2. Branch `feat/orthogonal-modifier-tests` (pushed, not merged)

The branch differs from `main` in 129 files (+37.4k/−3.9k lines). At ab1dd79,
the last commit from this session, 1535 tests passed.

### Stage 2 selection and estimation
| Commit | Change | Spec |
|---|---|---|
| d242df2 | New default modifier selection `orthogonal_tests`: per-candidate R-projection, AIPW-projection, and outcome-interaction tests, combined by Cauchy combination, then BH at q = 0.10. The final estimate is the ensemble mean of forest, linear interactions, and DINA. `balanced_evidence` is still available. | ORTHOGONAL_TESTS_SPEC.md |
| ff29d56 | Skips effect-only evidence families under `orthogonal_tests`; `estimation_trees` no longer affects selection; reuses the stored nuisance models | |
| 9e61ed4 | Super-learner nuisances by default for final estimation and modifier tests (elastic net + HistGB + RF) | SUPER_LEARNER_SPEC.md |
| 872dd3c | Test-selected modifiers' T·Z interactions and main effects are unpenalized in the interaction model (events-per-parameter guard, separation fallback) | |
| 0ac054a | Training rows use raw out-of-fold nuisances; held-out rows average fold models, each Platt-calibrated on its own validation rows (chosen from a 740-job study) | |

### Stage 2 reporting (never feeds selection)
| Commit | Change | Spec |
|---|---|---|
| dc5fa71 | CATE variance decomposition, GATES calibration, pre-specified primary test (pooled BLP β2) | |
| b0d44a3 | Held-out effect sizes for selected modifiers; cross-fold concept matching with MMB aggregation; training p-values labeled as screening | |
| d941d62 | ATO as co-primary estimand with the clipped AIPW ATE; `outside_overlap` flag; clip sensitivity | |
| fb755c1 | Unmeasured-confounding sensitivity: E-values, OVB bounds and robustness values, elastic-net benchmark refits | SENSITIVITY_CITATIONS.md |
| e8a5878 | DoubleML cross-check test. DoubleML 0.11.4 swaps the lower and upper bound SEs; OCI is correct. Reporting this upstream needs your OK. | SENSITIVITY_CITATIONS.md |

### Measurement and input
| Commit | Change | Spec |
|---|---|---|
| a54f460 | Reserved `documented_unknown` extraction state, separate from not documented (null) and failure | |
| 81b90ec | Optional dated-notes input (`notes`): only notes before the index date, relative-day headers, derived `input/analysis_cohort.parquet` with manifest; fails closed on change | PRETREATMENT_NOTES_SPEC.md |

### Stage 1 and evaluation tooling
| Commit | Change | Spec |
|---|---|---|
| b2307f2 | `matched_pair_uplift` rewritten as a BoW conditional logit on discordant matched pairs (paired differences for continuous outcomes) | STAGE1_POINT12_SPEC.md |
| 830dde7 | Neural queries ranked on held-out statistics; Stage 1 evaluation scores the exact handed-off directions | STAGE1_POINT12_SPEC.md |
| f5746ea | `oci-evaluate-attribution` and `scripts/ablate_stage1_architectures.py` (leave-one-architecture-out) | STAGE1_POINT12_SPEC.md |

### Synthetic data and benchmark
| Commit | Change | Spec |
|---|---|---|
| 087e5c4 | Generator DGP factors (instruments, prognostic-only variables, covariate correlation, MCAR/MNAR documentation, post-treatment notes, proxies) and `synthetic_data.cli reequation` | BENCHMARK_POINT14_SPEC.md |
| c735208 | `scripts/benchmark.py` materialize/configs/aggregate; scenario registry S0–S6; frozen confirmatory seeds and ledger; oracle-covariate ceiling | BENCHMARK_POINT14_SPEC.md |
| 06eb4b0 | Fixed `--rerun text_models` discarding verifiable checkpoints; fixed preflight for Stage 1-only configs | |
| 0e99891 | Default `modifier_strength` raised to 2.0 (true-ITE SD of re-equated cells 0.08–0.23) | |
| efe9caa | `./run_benchmark.sh MACHINE BENCH`, end to end: note generation, materialize, configs, runs, aggregate; resumable; `--plan`, `--through`. New `dgx_station_gb300` preset; `run_synthetic.sh CONFIG.json` mode; `benchmarks/base_config.json` | |
| f2722ca | `--llm-url URLS` uses running vLLM servers for every LLM role; GPU detection by total memory (`MIN_TOTAL_GPU_GB`) | |
| 1d82fbf | Thousands of concurrent requests: raised the open-file limit, generator workers up to 1000 per server, 1800 s generator timeout | |
| ab1dd79 | `run_benchmark.sh` installs vLLM only when OCI starts its own servers | |

### Model support
| Commit | Change |
|---|---|
| 6d1fa84, b446222 | Nemotron 3.5: published sampling and reasoning switch; managed vLLM uses `--reasoning-parser nemotron_v3 --trust-remote-code` |
| 5bcf72d | Reflection Beam 1: sampling 0.7/0.99; every request names a reasoning level (`none` is sent as `minimal`); `--reasoning-parser beam`; `VLLM_ENGINE_READY_TIMEOUT_S=3600` |

### Ken's later commits on the branch (no Claude trailer; not reviewed in this session)
8d5ecd5 synthetic feature retries + S6 config fix · e2c4ea7 joblib 1.6 cloudpickle fix ·
17db49e, 6fd3d5d embedding batch size 16, then 128 · 9a0fc49 concurrent Stage 1 contexts on one GPU ·
d7e4c3c bug fix tweaks · 2767b39 more ontology review rounds

## 3. Open items
- No real end-to-end LLM/GPU benchmark run of the branch yet (all tests use fakes). Plan: merge to `main` after one succeeds; ask before merging.
- DoubleML upstream report (needs your OK).
- Deferred: fold-aware candidate testing and a permuted-outcome negative control (M-03); S1-21 shared Stage 1 nuisances, HTR cost, S1-25, S1-26 (after ablation results); extraction reliability substudy and MI sensitivity (M-11).
- Remaining `~/oci_review/REVIEW.md` §9 recommendations not covered above (SA-10, SA-26, operations items).
