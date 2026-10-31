# CLAUDE.md — OCI repository guide

## Purpose

OCI (Oncology Causal Inference) estimates treatment effects from clinical text
with a two-stage, science-first workflow:

1. **Stage 1** fits up to ten independent text-model architectures on each
   outer training fold and freezes their evidence (which parts of the record
   relate to treatment, outcome, or treatment-effect variation).
2. A **handoff** (`handoff/evidence.jsonl`) passes only the selected
   architectures' evidence to Stage 2.
3. **Stage 2** turns the evidence into named variables with an interpretation
   LLM, extracts them from every patient's text with an extraction LLM, selects
   confounders and effect modifiers with the deterministic `multi_model`
   procedure, and estimates CATEs and the AIPW ATE on outer held-out patients.
4. **Post-hoc evaluation** compares frozen outputs with oracle truth on
   synthetic data without letting oracle columns reach discovery or fitting.

User documentation: `README.md` (entry point) and `docs/` — `glossary.md`,
`how_it_works.md`, `stage1_architectures.md`, `stage2_selection.md`,
`configuration.md`, `serving.md`, `optional_extensions.md`,
`synthetic_data.md`, `benchmarks.md`, and `design_notes/`.

## Entry points

| Command | Code | Purpose |
|---|---|---|
| `python scripts/run_all_evidence.py` | `oci/inference/research_all_evidence_workflow.py` (`build_parser`, `main`) | Run, resume, or relaunch the all-evidence workflow (29 flags; everything else via `--set KEY=VALUE`) |
| `./run_synthetic.sh COHORT [OUT]` | `run_synthetic.sh`, `scripts/presets/{external,h100x8,rtxpro6000x8,dgx_station_gb300}.env` | Run a bundled synthetic cohort with a serving preset; `./run_synthetic.sh CONFIG.json` runs a configuration file's cohort with the preset's serving settings and the detected devices |
| `./run_benchmark.sh MACHINE BENCH` | `run_benchmark.sh`, `oci/evaluation/benchmark_driver.py`, `benchmarks/base_config.json` | The benchmark end to end on one machine (a preset name): LLM corpora the selected cells need (source expansion with the saved DGP; S5/S6 corpora) on a vLLM pool it starts and stops, `materialize`, `configs`, each run through `run_synthetic.sh CONFIG`, `aggregate`; resumes on rerun; `--plan` prints the plan and status; `--llm-url URLS` uses running vLLM servers for every LLM role in place of the preset's managed ones |
| `oci-evaluate-stage1` | `oci/evaluation/stage1.py` | Oracle-safe evaluation of frozen Stage 1 architectures; writes `OUT/evaluation/stage1/` |
| `oci-evaluate-attribution` | `oci/evaluation/attribution.py` | Credit Stage 2 variables to the Stage 1 architectures that supported them (any-source and sole-source); `--oracle` adds recovery by held-out R² |
| `python scripts/ablate_stage1_architectures.py` | `oci/evaluation/ablation.py` | `prepare`: leave-one-architecture-out handoffs and `run_stage2_from_artifacts.py` configs (never launches LLM jobs); `compare`: finished ablations versus the full run |
| `python scripts/benchmark.py` | `oci/evaluation/benchmark.py`, `oci/evaluation/benchmark_ceiling.py`, `benchmarks/{scenarios,seeds}.json` | `materialize`: one cohort per scenario, size, and seed (re-equation through `synthetic_data.reequation`, or the generator commands for S5/S6); `configs`: one workflow configuration per cell (never launches runs; confirmatory seeds need `--confirmatory` and are logged in `benchmarks/confirmatory_ledger.jsonl`); `aggregate`: finished runs versus oracle truth, with an oracle-covariate ceiling through Stage 2's `estimate_outer_fold` |
| `python scripts/run_stage2_from_artifacts.py` | same name | Stage 2 in a separate directory from a preserved handoff and split file (external endpoints only; for a dated-notes run, `dataset` is its `input/analysis_cohort.parquet`) |
| `python scripts/run_standalone_htr.py` | `oci/inference/standalone_htr.py` | `run`: HTR-only treatment effects per outer fold (no LLM, handoff, or Stage 2); `two_stage` (default) runs Stage 1's `MultiModelForestStage1HTRProvider` (cross-fitted HTR nuisance networks, then HTR effect networks; DINA for binary outcomes, against HTR propensity and per-arm outcome nuisances by default, from one three-head network or, with `run.nuisance_networks=separate`, one single-head network each, or Stage 1's bag-of-words ones with `--dina-nuisances bow`; the chunk encoder is first pretrained on the masked-LM task once per outer fold on its training text, `run.mlm_epochs=0` to skip, `run.mlm_scope=network` per network on its own training rows) with one Stage 1 checkpoint unit per inner network, `joint` one network with e, m, and tau heads (tau by the pseudo-outcome on its own detached nuisances); cross-fitted held-out predictions and an oracle-free summary; resumable; `--devices` and `--jobs-per-device` run outer folds in parallel; `evaluate`: oracle comparison after the prediction hash verifies |
| `oci run`, `oci init` | `oci/cli.py`, `oci/experiments/runner.py`, `oci/inference/applied.py` | Standalone explicit-feature workflows |
| `python -m synthetic_data.cli` | `synthetic_data/` (`generator.py`, `generation_factors.py`, `dgp_terms.py`) | Synthetic cohort generator: confounder, effect-modifier, instrument, and prognostic roles; generation-time factors (covariate correlation, MCAR/MNAR documentation, post-treatment dated notes, proxies) |
| `python -m synthetic_data.cli reequation` | `synthetic_data/reequation.py` (`reequation_cohort`) | Recompute treatment and outcome over an existing synthetic cohort's true characteristics for a scenario (no LLM; text copied unchanged; refuses cohorts with post-treatment notes) |

Helper scripts: `scripts/detect_all_evidence_hardware.py` (GPU and worker
sizing for `run_synthetic.sh`) and `scripts/configure_local_cuda.py` (exposes
the `local-llm` CUDA toolkit inside `.venv` for vLLM).

## Package map

```text
oci/
├── config.py                         # Stage 1 template and `oci run` dataclasses (ExperimentConfig)
├── cli.py                            # `oci init`, `oci run`
├── evaluation/stage1.py              # oci-evaluate-stage1 (scores the handed-off embedding directions)
├── evaluation/stage1_lexical_null.py # Random-chunk null for lexical recovery (--null-draws)
├── evaluation/stage1_timing_report.py  # Per-architecture Stage 1 wall time in the evaluation output
├── evaluation/attribution.py         # oci-evaluate-attribution (Stage 2 lineage, oracle recovery by R²)
├── evaluation/ablation.py            # scripts/ablate_stage1_architectures.py (prepare, compare)
├── evaluation/benchmark.py           # scripts/benchmark.py (registry, seed guard and ledger,
│                                     #   materialize, configs, aggregate metrics)
├── evaluation/benchmark_ceiling.py   # Oracle-covariate ceiling via estimate_outer_fold
├── evaluation/benchmark_driver.py    # run_benchmark.sh: generation, materialize, configs, runs, aggregate
├── extraction/                       # `oci run` extraction: explicit_features, complete_paged,
│                                     #   contract_lexical_context, llm_routing, cache
├── models/
│   ├── causal_forest_head.py         # CausalForestDML wrapper with elastic-net nuisances
│   ├── elastic_net_nuisance.py       # Clone-safe elastic-net nuisance estimators and fit audits
│   ├── super_learner_nuisance.py     # Stacked nuisance learner (elastic net, boosting, forest)
│   ├── dina.py                       # Gao–Hastie Bernoulli DINA objectives
│   ├── hierarchical_transformer_extractor.py   # HTR model
│   ├── htr_number_summary.py         # Per-chunk number summaries for HTR (template default)
│   ├── concept_embedding_cache.py    # Chunk embedding cache (Stage 1)
│   └── explicit_feature_featurizer.py          # `oci run` featurization
├── utils/calibration.py              # Platt calibration of nuisance predictions
└── inference/
    │ # Workflow and Stage 1
    ├── research_all_evidence_workflow.py   # Components, config compilation, resume, reselection
    ├── pretreatment_notes.py         # Optional dated-notes input (`notes`): builds and verifies
    │                                 #   input/analysis_cohort.parquet, its manifest, and the audit
    ├── stage1_architectures.py       # Canonical registry and selector semantics
    ├── stage1_architecture_artifacts.py    # Per-architecture views for explicit selections
    ├── stage1_checkpointing.py       # Per-context Stage 1 checkpoints
    ├── stage1_timing.py              # Reporting-only section timer; components/*/timing.jsonl
    ├── multi_model_forest_stage1.py  # BoW, HTR, embedding, matched-pair text models per context
    ├── htr_trainer.py, htr_evidence_provider.py, stage1_dina.py
    ├── multi_model_pair_uplift.py    # matched_pair_uplift: caliper matching on e and mu0, BoW pair
    │                                 #   conditional logit (binary) / paired-difference elastic net
    ├── sparse_text_modeling.py, stage1_modeling_support.py   # Shared Stage 1 modeling primitives
    ├── embedding_contrast_discovery.py     # Embedding contrasts (whole-cohort, clustered, retrieval)
    ├── tfidf_topic_stage1.py, tfidf_topic_discovery.py, tfidf_topic_score_selection.py
    │                                 # TF-IDF topics and orphan n-grams; writes split_provenance.jsonl
    ├── neural_query_agentic_forest.py, neural_query_discovery_runtime.py, neural_cohort_witness.py
    │ # Stage 2
    ├── plain_handoff_stage2.py       # Stage 2 config, LLM transport, managed-pool orchestration,
    │                                 #   evidence interpretation, consolidation, definitions, run loop
    ├── plain_handoff_stage2_evidence.py    # Handoff → evidence cards (uses all_evidence_fusion.py)
    ├── plain_handoff_stage2_analysis.py    # Extraction, aggregate review and refinement, filters,
    │                                 #   selection driver, held-out extraction, final estimation
    ├── stage2_multi_model_config.py, stage2_elastic_net_selection.py   # Selection policy (config)
    ├── stage2_multi_model_selection.py     # Resampled evidence families, predictive nuisance selection
    ├── stage2_modifier_tests.py      # Orthogonal modifier tests, Cauchy combination, BH, labels
    ├── stage2_modifier_ranking.py, stage2_modifier_count.py   # Modifier selection (orthogonal_tests,
    │                                 #   or balanced_evidence ranking and budget/estimator choice)
    ├── stage2_nuisance_selection.py, stage2_statistical_selection.py, stage2_interaction_screen.py
    ├── stage2_effect_estimators.py, stage2_dina.py, stage2_final_modeling.py   # Final effect models
    ├── stage2_diagnostics.py         # Oracle-free BLP (primary test), GATES, CLAN, RATE, ATO,
    │                                 #   trimming, SMDs, CATE variance decomposition, outer-fold
    │                                 #   modifier confirmation (reporting only)
    ├── stage2_sensitivity.py         # Unmeasured-confounding sensitivity: E-values, OVB bounds,
    │                                 #   robustness values, per-fold benchmark refits (reporting only)
    ├── stage2_extraction_failures.py # Extraction-failure masks and reports
    ├── stage2_documented_unknown.py  # Reserved documented_unknown state: mapping rules, reports
    ├── nuisance_diagnostics.py       # Held-out nuisance diagnostics, overlap policy
    ├── stage2_estimand_ontology*.py, stage2_modifier_concepts.py, stage2_multi_model_adjudication.py,
    │   stage2_matched_batch.py, stage2_sequential_consolidation.py   # Optional extensions
    ├── stage2_sampling.py, stage2_endpoint_pool.py, stage2_concurrency.py, vllm_server_pool.py
    │                                 # Sampling profiles, extraction routing, live limits, managed vLLM
    ├── stage2_preflight.py           # Read-only Stage 2-only validation
    ├── stage2_prompt_catalog.py, stage2_clinical_prompts.py, stage2_request_audit.py, ...
    ├── standalone_htr.py             # scripts/run_standalone_htr.py: HTR-only effects (Stage 1 two-stage or joint)
    ├── htr_mlm_pretraining.py        # Masked-LM pretraining of the HTR chunk encoder (standalone default)
    │ # `oci run`
    └── applied.py, applied_explicit_feature_forest.py, agentic_explicit_feature_forest.py,
        agentic_attention_variable_forest.py, multi_model_agentic_forest.py
```

Stage 1 uses `htr_trainer.py` and `sparse_text_modeling.py` and never imports
the `oci run` agentic runners (`tests/test_import_boundaries.py` enforces
this). The dependency runs the other way: `AgenticAttentionVariableForestRunner`
subclasses `HTRTrainer`, so changes to `htr_trainer.py` affect both Stage 1 and
`oci run`. `HTRTrainer._dina_effect_nuisance` and `_dina_effect_predictions`
supply a DINA effect network's nuisances (Stage 1: bag-of-words, cross-fitted
inside each effect fold); `scripts/run_standalone_htr.py` overrides them. `stage2_role_adjudication.py` holds only prompt-bounding limits and
JSON helpers; roles come from statistical selection.

## Stage 1 architectures

The canonical order in `oci/inference/stage1_architectures.py` is:

1. `bow_nuisance`
2. `bow_r_loss`
3. `matched_pair_uplift`
4. `htr_neural`
5. `embedding_whole_cohort`
6. `embedding_clustered`
7. `tfidf_semantic_retrieval_contrasts`
8. `tfidf_topics`
9. `tfidf_orphan_ngrams`
10. `neural_query_moments`

When `science.stage1_architectures` (`--architectures`) is omitted, the
architectures enabled by the Stage 1 template's switches run: eight with the
bundled template (all except `tfidf_semantic_retrieval_contrasts`
and `tfidf_orphan_ngrams`), for binary and continuous outcomes. `all` selects
all ten. An explicit selection is recorded in
`stage1_architectures/manifest.json` and cannot change on resume. Private
prerequisites (for example BoW nuisance fits for embedding architectures) may
run, but only selected architectures enter the handoff.
Keep the omitted-selector path's behavior unchanged.

`bow_r_loss` cross-validates each penalized linear effect model's ridge penalty
over `multi_model_forest.bow_effect_ridge_alpha_grid` (the lowest pooled
held-out effect loss over the effect folds; `[]` restores each view's fixed
`ridge_alpha`), and its term-ranking fit uses the chosen value.
Neural-query consensus groups are ordered by the median absolute held-out z of
their members (scored on inner partitions of the context's training rows,
never its held-out rows), then by the training score; the order sets query IDs
and never gates queries. Queries are fitted, scored, and retrieved only on
chunks with at least 12 letters or digits (`informative_chunk_text` in
`oci/models/concept_embedding_utils.py`, shared with embedding retrieval), and
each query keeps its 30 highest-contrast phrases (`evidence_ngram_top_k`).
For evaluation only, each text-model context writes
its held-out rows' scores on the exact handed-off embedding directions
(`EMBEDDING_DIRECTION_SCORE_SOURCES` in `stage1_architecture_artifacts.py`).
Every Stage 1 component writes reporting-only wall time per architecture
(`components/*/timing.jsonl`, `stage1_timing.py`); timing never enters a
checkpoint identity or a fitted value.

## Stage 2

`multi_model` is the only selection mode and the default. Per outer fold:
evidence cards → interpretation → candidate merging → extraction definitions →
outer-training extraction with aggregate review and failure-driven definition
refinement (each cell is a value, not documented (null), the reserved
`documented_unknown`, or an extraction failure; every encoder keeps the three
non-value states as separate grouped indicators) → informativeness and
extraction-failure filter
(`min_informative_patients`, `max_extraction_failure_fraction`) → optional
estimand search and selection consolidation → resampled evidence families and
predictive nuisance selection (under `orthogonal_tests` only the nuisance
models and the univariable treatment/outcome associations run; the effect-only
families are skipped) → modifier selection by `orthogonal_tests` (the
default: training-row final-estimation nuisances, whose inner-fold models are
stored in `selection/nuisance_models/` for reuse; `r_projection`,
`aipw_projection`, and `outcome_interaction` tests per candidate; Cauchy
combination; BH at `modifier_selection.q`) → held-out extraction of the full
nuisance-candidate catalog → final refit with nuisances from the configured
nuisance learner (super learner by default; raw out-of-fold predictions for
training rows; for held-out rows, the average of the inner-fold models, each
Platt-calibrated on its own inner-validation rows; the stored inner-fold models
score the held-out rows when their fingerprint verifies) of the
estimator ensemble (unweighted mean of `causal_forest`, `linear_interactions`,
and `dina` for binary outcomes; the interaction model leaves the selected
modifiers' interactions and main effects unpenalized, guarded by events per
parameter with a separation fallback) → cross-fitted predictions (with the
reporting-only `outside_overlap` flag), the co-primary clipped AIPW ATE and
overlap-weighted ATO (`stage2.lead_estimand` labels which leads) with their
reporting-only sensitivity to unmeasured confounding (E-values, omitted-variable-bias
bounds and robustness values, and per-fold elastic-net benchmark refits without each
of the top `stage2.sensitivity.benchmark_groups` discovered variables), and
`heterogeneity_diagnostics.json` (the pooled BLP `beta2` > 0 is the
pre-specified primary heterogeneity test; GATES and RATE are secondary),
including the reporting-only outer-fold confirmation of the selected modifiers
(the inferential result: held-out tests and effect sizes, cross-fold concept
matching with Meinshausen–Meier–Bühlmann aggregation; training-row modifier-test
p-values are screening values) and the CATE variance decomposition (prognosis
versus selected modifiers). No per-patient CATE interval is reported for the
ensemble.
`modifier_selection.rule:
"balanced_evidence"` instead runs the nested modifier ranking and
budget/estimator choice; concept review requires it. Details:
`docs/how_it_works.md` and `docs/stage2_selection.md`.

Configuration rules:

- Unknown keys under `stage2` are errors (`_reject_unknown_stage2_keys`), and
  so are unknown keys under `notes`.
- With `notes` (dated-notes input), every loader reads
  `input/analysis_cohort.parquet` (`ResearchStage1Config.analysis_dataset`, via
  `_load_analysis_dataset`), which the first Stage 1 start builds. Later starts
  verify the source hashes, the `notes` settings, and the derived file against
  `input/pretreatment_notes_manifest.json` and reuse it without rewriting it
  (Stage 1 checkpoints use its path, size, and mtime); any mismatch fails
  closed. `run_config.json` omits `notes` when it is not configured.
- Stage 2 requires the frozen split file
  `components/tfidf/split_provenance.jsonl` and never resamples folds.
- `run_config.json` redacts API keys; Stage 2 checks model identity
  (`stage2/model_identity.json`) and selection policy on resume. Selection or
  estimation policy changes on a completed run go through
  `--stage2-only --stage2-reselect`; under `orthogonal_tests`,
  `estimation_trees` is not part of the selection identity, so changing it
  refits only the final estimation. The same holds for
  `modifier_selection.unpenalized_selected_interactions` and
  `unpenalized_events_per_parameter` (`selection_dict()` omits them).
- Diagnostic evidence families (`multi_model.diagnostic_families`) and
  matched-batch evidence are opt-in and do not drive selection.
- External extraction pools can explicitly enable `extraction_llm.allow_model_variants`
  and select per-endpoint `model` IDs. All servers must share a recognized model
  family and compatible tokenization; actual identities are audited and the
  served-ID set is checked on resume. Identical backing models remain the default.
- Outer-fold confirmation (`modifier_confirmation`), the CATE variance
  decomposition (`cate_variance_decomposition`), and the unmeasured-confounding
  sensitivity (`unmeasured_confounding_sensitivity`) in
  `heterogeneity_diagnostics.json` are computed from frozen artifacts after
  every fold finishes; no selection, estimation, or other fold reads them. The
  sensitivity benchmarks are the only part that refits models: per fold, after
  the final estimation, on outer-training rows only, with their own checkpoints
  in `estimation/sensitivity_benchmarks/`.

## Scientific invariants

- Never use oracle columns (`true_*`, `oracle_*`, `ground_truth*`) during
  discovery, extraction, fitting, selection, or handoff construction. Join them
  only after predictions are frozen and hashed.
- In dated-notes mode, notes dated after the index date (and, unless
  `include_index_date` is set, on it) never enter the analysis text; note
  headers state relative days, so calendar dates never enter prompts through
  them. Only the configured note columns are read from the notes table, and the
  per-patient audit is never read by any modeling or LLM step.
- Hash frozen Stage 1 artifacts before loading oracle data for evaluation; the
  attribution and ablation reports and the benchmark aggregate likewise hash
  every Stage 2 artifact they read first, and write only under
  `OUT/evaluation/` or outside the run.
- Tune benchmark settings only on the development seeds of
  `benchmarks/seeds.json`; confirmatory seeds are used once per commit and
  scenario (`docs/benchmarks.md`).
- Keep outer held-out rows unavailable to inner-fold discovery, review,
  selection, and definition search.
- LLM prompts never contain dataset paths or names, synthetic-generation
  metadata, known data-generating roles, identifiers, or outer held-out
  information. Prompts that see selection evidence (concept review) accept only
  the allowlisted aggregate evidence bundle.
- Preserve exact row, split, component, architecture, and configuration
  provenance in sidecars.
- Fail closed when text capacity would truncate semantic content.
- The causal forest's internal nuisances and the selection-stage role models
  are elastic nets. Final Stage 2 nuisances (shared with the modifier tests) use
  the configured `stage2.nuisance_learner`: the super learner by default
  (elastic net, histogram gradient boosting, random forest), or `elastic_net`.
  Keep the fitted audit records: stack weights, per-learner and stack CV loss,
  and the elastic-net selected regularization and iteration limits.
- Estimand-informed definition search uses outcome information to choose
  definitions; keep it opt-in and training-only.

## Explicit-feature functionality

Explicit-feature extraction and modeling are supported and must not be removed
as incidental cleanup. The standalone `oci run` model types are:

- `explicit_feature_forest`
- `agentic_explicit_feature_forest`
- `agentic_attention_variable_forest`
- `multi_model_agentic_forest`

Feature contracts are role-aware: `roles: ["confounder"]`,
`roles: ["effect_modifier"]`, or both. Preserve the extraction strategies
(`full_record`, `complete_paged_v1`, `contract_lexical_rag`, `tail`),
complete-note paging and citation validation, extraction caching, the
`multi_model_agentic_forest` extracted-feature review, and the explicit-feature
featurizers. Stage 2 also accepts investigator-specified variables through
`stage2.explicit_features`; their definitions and roles are locked.

## Common commands

```bash
# Install (add --extra local-llm for managed vLLM pools)
uv sync --frozen

# Full or resumed workflow on your cohort
python scripts/run_all_evidence.py --config example_configs/research_all_evidence_multi_model.json

# Stage 1 only, or a subset of architectures
python scripts/run_all_evidence.py --config CONFIG.json --stage1-only
python scripts/run_all_evidence.py --config CONFIG.json --architectures bow_nuisance,tfidf_topics

# Progress, Stage 2-only relaunch, preflight, guarded reselection
python scripts/run_all_evidence.py --config OUT/run_config.json --status
python scripts/run_all_evidence.py --config OUT/run_config.json --stage2-only --preflight-only
python scripts/run_all_evidence.py --config OUT/run_config.json --stage2-only --stage2-reselect

# Bundled synthetic cohorts
./run_synthetic.sh one_conf_one_mod
PRESET=h100x8 ./run_synthetic.sh five_conf_five_mod /path/to/output

# Evaluate frozen Stage 1 outputs against oracle columns
oci-evaluate-stage1 --run-dir OUT --architectures all

# Credit Stage 2 variables to Stage 1 architectures; leave-one-architecture-out reruns
oci-evaluate-attribution --run-dir OUT --oracle
python scripts/ablate_stage1_architectures.py prepare --run-dir OUT --out ABL --stage2-config ENDPOINTS.json
python scripts/ablate_stage1_architectures.py compare --run-dir OUT --ablation-dir ABL --oracle

# Benchmark scenarios: cohorts per cell, run configurations (printed, not launched), aggregate
python scripts/benchmark.py materialize --scenarios S0,S1 --source SOURCE_DIR --out BENCH
python scripts/benchmark.py configs --base CONFIG.json --out BENCH
python scripts/benchmark.py aggregate --out BENCH
./run_benchmark.sh rtxpro6000x8 BENCH --scenarios S0 --sizes 1000 --seeds 101   # all steps, resumable

# HTR-only effects (Stage 1's cross-fitted HTR ensembles, five outer folds at once) and their oracle comparison
python scripts/run_standalone_htr.py run --dataset DATA.parquet --out OUT --seed 101 --devices cuda:0,cuda:1 --jobs-per-device 3 --evaluate

# Standalone explicit-feature workflow
oci run --config example_configs/agentic_explicit_feature_forest_config.json

# Tests
pytest -q
```

## Change checklist

When changing Stage 1 or Stage 2:

1. Update the canonical registry if architecture identity changes.
2. Preserve or deliberately version handoff, checkpoint, and artifact schemas.
3. Test omitted, subset, resume, and missing-artifact behavior.
4. Add architecture-native evaluation metrics without refitting.
5. Verify lightweight imports do not initialize optional runners
   (`tests/test_import_boundaries.py`).
6. Keep CLI flags, config keys, defaults, and output names in
   `docs/configuration.md` in step with the code.
7. Run `git diff --check`, focused tests, the full suite, and a package build.

**Documentation rule:** README.md and CLAUDE.md must stay in sync. When a
change affects user-visible behavior, update README.md and the relevant page in
`docs/` together with this file. Documentation describes current behavior only,
without change-log language.
