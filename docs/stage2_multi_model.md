# Stage 2 selection from multiple models

Implemented September 21, 2026; automatic modifier-count selection added September
22, 2026. Enable with
`stage2.statistical_selection.selection_mode: "multi_model"`.

## 1. Purpose and scope

1. Build a reproducible confounder/modifier list from complementary empirical
   evidence and LLM interpretation. A candidate discarded by one model remains
   available to other models and final review.
2. Use existing consolidated, ontology-reviewed, extracted measurements. This
   mode does not repeat discovery/extraction or change the final estimator.
3. Work separately inside each outer-training population. The numerical selector
   reads only observed treatment/outcome from the source dataset. Patient text,
   outer-test data, oracle columns, and generation metadata cannot enter its
   evidence or the LLM prompts.
4. The preceding checkpoint is `de37fff`: the oracle estimation diagnostics and
   extracted-candidate logistic experiments. Large reproducible arrays and model
   bundles remain local; reports, scripts, metrics, and provenance are versioned.

## 2. Evidence families

| Family | Treatment / outcome evidence | Modifier evidence |
| --- | --- | --- |
| Univariable models | Candidate association with treatment; outcome association adjusted for treatment | `Y ~ T + Z + T:Z`; logistic for binary outcomes, linear otherwise |
| Penalized main effects | Separate grouped elastic nets for treatment and marginal outcome | None; predictive support alone is not a modifier label |
| Penalized outcome interactions | Main-effect support in the joint outcome model | Grouped elastic net for `Y ~ T + all Z + all T:Z` |
| Orthogonal linear model | Elastic-net nuisances supply residuals | Grouped elastic net for `Y_res ~ T_res * (constant + Z)` without an additional intercept |
| Candidate R-learners | Same cross-fitted nuisances | One candidate at a time, ridge-stabilized contrasts, held-out R-loss gain over a constant effect |
| Predictive forests | Held-out group-permutation importance for treatment and outcome | None; these are evidence models, not causal-forest nuisances |
| Causal forests | All-candidate elastic-net nuisance adjustment | Honest forests on residuals; held-out R-loss permutation importance and grouped split importance |

1. Categorical columns remain one candidate group. Interaction tests are omnibus;
   permutations move all encoded columns of a candidate together, preserving its
   category/missingness coding.
2. Univariable modifier models include the candidate main effect and are
   unadjusted for other covariates. Their confounding sensitivity is explicit to
   the LLM; orthogonal models provide separate adjusted evidence.
3. Logistic interactions concern log odds. Orthogonal models and causal forests
   concern outcome/probability differences. These are complementary views, not
   interchangeable coefficient tests.
4. Nominal p < 0.05 and BH q < 0.10 create separate support flags within each
   resample and endpoint. Neither is an inclusion gate. These q-values do not
   correct upstream adaptive discovery.
5. Penalized models choose regularization inside training, with encoders refitted
   inside every tuning partition. Only penalties converging in every tuning fold
   qualify. Nonconverged final fits are unevaluable. Clone audits record selected
   penalties, boundary selections, iteration counts, and convergence.
6. Whole-model held-out gains versus constant predictions/effects accompany
   coefficient and forest importance evidence. A nonzero coefficient or positive
   split importance does not establish that the model generalized.

## 3. Resampling and honest evaluation

1. Preserve original inner-fold memberships. Fit on inner-training patients and
   score predictions on untouched inner-validation patients. Splits must partition
   exactly the supplied outer-training rows; each patient validates once.
2. Fit all-candidate treatment and marginal-outcome elastic nets. Cross-fit them
   again inside inner training to obtain training residuals; separate models
   trained on that population predict validation residuals. Encoder and penalty
   fitting are confined to training.
3. Keep nuisances fixed across that fold's perturbations. This assesses evidence
   stability conditional on one honest nuisance construction, not a bootstrap of
   the entire discovery process.
4. Defaults: three repetitions, using all inner-training patients once and two
   treatment-stratified 80% subsamples. Linear evidence models cycle through L1
   ratios 0.2, 0.8, and 1.0. Nuisances use their separately configured ratio.
5. Forests first see all candidates; later repetitions randomly partition the
   complete list into subsets of at most 32. Every candidate appears once per
   repetition. Causal forests retain all-candidate nuisance adjustment even when
   their effect-input subsets are small.
6. Default forests: 200 trees, minimum leaf size 10. Causal forests use honesty,
   45% tree subsampling, all supplied effect columns at each split, and inference
   disabled. Selection does not request confidence intervals.
7. Three group permutations score each candidate on validation patients. Causal
   scores hold nuisance residuals fixed while permuting effect inputs. These are
   marginal permutation diagnostics, not conditional importance estimates;
   correlated alternatives can dilute or redistribute their scores.
8. Configured propensity eligibility applies to orthogonal linear models,
   candidate R-learners, and causal forests. Association and outcome models use
   all sampled patients, so their populations and score scales differ.

## 4. Stability summaries and LLM interpretation

1. For each candidate, family, and task, report exposure/evaluable/support counts,
   support fraction, fold counts, score mean/range, p/q summaries when applicable,
   and model-quality diagnostics. Missing or nonconverged fits are not negative
   votes; support fractions divide by evaluated fits.
2. Keep families separate. Folds/repetitions share patients, so support fractions
   are not independent replications, causal probabilities, or formal
   error-controlled stability selection. There is no pooled popularity score.
3. The LLM reviews bounded batches for shared themes, complementary signals,
   competing proxies, and disagreements. Hierarchical reconciliation exposes
   themes across batches and preserves every candidate ID, including weak and
   unevaluable candidates.
4. Themes organize evidence; they do not merge, broaden, or relabel measurements,
   nor transfer roles to every member. Themes cite up to 12 representative
   evidence IDs. Final decisions still receive each candidate's complete
   aggregate evidence.
5. A second pass assigns provisional confounder, modifier, both, or neither to
   every candidate.
   Nonlocked retained roles require citations: modifiers cite heterogeneity
   evidence; confounders address both treatment and outcome evidence. No positive
   p-value or minimum-frequency gate is imposed. Investigator locks are exact.
   Automatic modifier-count selection subsequently preserves all retained
   confounders and chooses the modifier inputs as described below.
6. Decisions describe supporting/contrary evidence, fold consistency, disagreement
   across methods, rationale, and stability (`consistent`, `mixed`, or
   `insufficient`). Those labels are interpretations, not calibrated probabilities.
7. Explicit prompt allowlists exclude cell logs, row values/IDs, predictions,
   paths, exception messages, and oracle metadata. Oversized prompts split or
   fail with a budget error; no candidate is silently omitted.

## 5. Automatic modifier-count selection

1. Enabled by default **within `multi_model` mode**. The default rule is
   `minimum_r_loss`: select the modifier budget with the lowest mean validation
   R-loss, breaking exact ties toward fewer features. `one_standard_error` is
   available for stronger pruning: choose the smallest budget whose mean paired
   loss difference from the best is within one paired standard error across
   folds. This is a simplification heuristic, not a significance test.
2. The count step preserves **every confounder retained by the full-training
   role adjudication**. It does not force their elastic-net coefficients to be
   nonzero. Investigator-locked roles are exact; locked modifiers are always
   included and do not consume the additional-candidate budget. Locked
   confounder-only variables cannot acquire a modifier role.
3. Candidate budgets default to `0, 4, 8, 12, 16, 24, 32`, plus the complete
   ranked shortlist, capped at 64 unlocked candidates. Counts exceeding the
   available shortlist are clipped and deduplicated. Zero means a constant
   effect when no modifiers are locked, or the locked-modifier model otherwise.
   A constant effect is estimated from training residuals, not set to zero.
4. Preserve the original count-validation folds. **Inside each fold's training
   portion**, create treatment-stratified subfolds using `internal_cv_folds`,
   rerun all seven modeling families, and obtain a new LLM ranking. The evidence
   worker receives only that training portion's labels; every other label is
   masked. The count-validation outcomes cannot enter those evidence summaries
   or ranking prompts. A global ranking formed from all outer-training outcomes
   is not reused in these validation fits.
5. The ranking considers every unlocked candidate, including candidates not
   given a modifier role in the provisional broad review. It uses only the
   existing prompt-safe definitions and aggregate numerical evidence. Bounded
   requests sort batches; further requests merge leading windows while
   preserving each list's relative order. Every reviewed candidate is accounted
   for, candidate-specific effect citations are required, and oversized prompts
   fail rather than discard evidence. This ordering is an LLM heuristic, not an
   exhaustive search over feature subsets or a measurement consolidation.
6. For each count-validation fold, reuse the original numerical pass's
   all-candidate elastic-net nuisances. Training residuals are themselves
   cross-fitted; validation residuals use nuisance models fitted only on that
   fold's training patients. The residuals and propensity-eligible patients stay
   identical across modifier budgets. Selected confounder roles do not gate these
   broad scoring nuisances.
7. Fit the ranked prefixes using the production feature encoder, fitted only on
   eligible training patients. Validation forests use the configured
   `estimation_trees`, minimum leaf size 10, square-root feature sampling, 45%
   subsampling, honesty and inference enabled, and no tuning. Three seeds per
   fold are the default. This count-validation forest uses fixed residuals; the
   final production DML estimator subsequently refits its own nuisance models.
8. Score each held-out prediction using
   `mean(((Y - m_hat) - (T - e_hat) * tau_hat)**2)`. Average seeds within each
   fold, then average fold losses with equal fold weights. All budgets must have
   at least two common usable folds; insufficient overlap/variation fails visibly
   rather than selecting a budget using incomparable populations. Forest seeds
   are not counted as independent patient replications.
9. After choosing the budget, form a final ranking from all outer-training
   numerical evidence and take that prefix for the production refit. Preserve
   the broad confounder set, roles and citations before count selection, selected
   count, complete loss curve, fold rankings, predictions, encoded dimensions,
   seeds, and source/input fingerprints. The usual measurement-dependency and
   latent-materialization logic runs on the resulting roles.
10. This nesting is **conditional on the frozen upstream candidate catalog,
    extractions, ontology, and preselection consolidation**. It does not rerun
    Stage 1 or the unsupervised consolidation separately inside each split.
    R-loss is a noisy model-selection objective, not oracle-feature recall or
    Pearson ITE correlation, and does not establish causal identification.

## 6. Routing, checkpoints, and use

1. Final roles after count selection feed the existing estimator adapter:
   confounders enter nuisance adjustment, modifiers enter forest X, and dual-role
   variables receive both uses. No extraction definitions are changed by theme
   review.
2. This is opt-in. Omitted selectors preserve `llm_roles`; `independent_tasks`
   keeps binding joint-model selection and advisory LLM annotations. New defaults
   are omitted from both legacy policy fingerprints.
3. `role_adjudication.enabled` must be true. Disabled/incomplete LLM review cannot
   silently fall back to a numerical union.
4. Each nuisance fit and family/repetition/subset has an atomic, integrity-checked
   checkpoint under `selection/multi_model/<input-fingerprint>/fold_*/`. Identity
   includes measurements, labels, definitions, splits, policy, seed, and numerical
   source hashes. Completed compatible leaves are reused on restart.
5. Theme and role requests have separate checkpoints beneath
   `selection/role_adjudication/`. Evidence, prompt source, model identity, and
   policy changes invalidate them. Corrupted checkpoints fail visibly.
6. Standard selection reports, definitions, measurement dependencies, and final
   estimation artifacts remain. Changing the mode or a completed multi-model
   policy requires guarded `--stage2-reselect`.
7. Count selection stores checkpoints under `selection/modifier_count/` and
   nested numerical evidence under `selection/multi_model/modifier_count_training/`.
   `selection/modifier_count/selection.json` points to the active count result.
   Each LLM request and each fold/count/seed fit can resume independently;
   completed result hashes are validated. Count-only policy changes leave the
   underlying numerical-evidence fingerprint unchanged. When count selection is
   enabled, `estimation_trees` is also a selection parameter: changing it requires
   guarded reselection and invalidates the prior count-selection result.
8. This update uses `stage2_multi_model_selection_v2`. Version 1 configuration
   blocks are accepted as input, but an existing version 1 run requires guarded
   reselection before adopting the new defaults. The preflight check blocks the
   old completion shortcut. Set `modifier_count.enabled` to false to use only the
   broad role adjudication in a new/reselected version 2 run. Other selector modes
   retain their previous policies and fingerprints.

Configuration block:

```json
{
  "stage2": {
    "statistical_selection": {
      "selection_mode": "multi_model",
      "multi_model": {
        "repeats": 3,
        "row_fraction": 0.8,
        "l1_ratios": [0.2, 0.8, 1.0],
        "regularization_grid_size": 8,
        "forest_trees": 200,
        "forest_min_samples_leaf": 10,
        "feature_subset_size": 32,
        "permutation_repeats": 3,
        "nominal_p_threshold": 0.05,
        "q_threshold": 0.1,
        "max_prompt_chars": 100000,
        "modifier_count": {
          "enabled": true,
          "candidate_counts": [0, 4, 8, 12, 16, 24, 32],
          "max_ranked_modifiers": 64,
          "selection_rule": "minimum_r_loss",
          "forest_seeds": 3
        }
      }
    },
    "role_adjudication": {"enabled": true}
  }
}
```

Use [the full example](../example_configs/research_all_evidence_multi_model.json)
with configured data and model endpoints for a fresh run. For existing extracted
measurements, update the saved configuration and invoke guarded reselection:

```bash
python scripts/run_all_evidence.py --config /path/to/updated_run_config.json \
  --stage2-only --stage2-reselect
```

Before count selection, with 352 candidates, five inner folds, and default
repetitions, each forest family fits `5 × (1 + 2 × ceil(352/32)) = 115` subset models. Treatment/outcome
forests are separate: 230 predictive forests plus 115 causal forests, alongside
linear models and univariable tests. With the default three nested evidence
folds, count selection adds five training-only evidence runs of three folds each,
their LLM rankings, one final full-training ranking, and up to
`5 × 8 budgets × 3 seeds = 120` count-validation fits (zero-budget constant
models do not need a forest). The nested numerical work is substantially more
expensive than reusing a global ranking.

## 7. Validation and limits

1. Tests exercise real binary/continuous models, categorical encoding, nested
   nuisances, all evidence families, subset coverage, train/validation separation,
   and exact checkpoint reuse after changing outer-test labels and oracle columns.
2. Controlled-response LLM tests check theme reconciliation, candidate coverage,
   citations, locks, invalid-response rejection, and cache integrity. They do not
   establish real-LLM selection quality.
3. A production comparison must still measure extraction quality, frozen-selection
   confounder/modifier recovery, held-out effect estimates, and outer-fold
   variability. More evidence families do not establish better sensitivity,
   specificity, or causal identification by themselves.
4. Count-selection tests run real binary/continuous numerical models and forests
   with controlled LLM replies. They check nested row/label isolation, categorical
   measurements, independently recomputed validation R-loss, common populations,
   confounder/lock preservation, zero-budget behavior, exact resume without
   refitting, corruption detection, and version 1 preflight rejection. They do
   not measure a live LLM's ranking quality.

Method references: [EconML causal forest](https://www.pywhy.org/EconML/_autosummary/econml.grf.CausalForest.html)
and [GRF variable importance](https://grf-labs.github.io/grf/reference/variable_importance.html).
Count-selection objective: [EconML RScorer](https://www.pywhy.org/EconML/_autosummary/econml.score.RScorer.html).
