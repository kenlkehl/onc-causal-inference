# Stage 2 selection from multiple models

Implemented September 21, 2026; overlap-restricted modifier screening and joint
modifier-count/estimator selection updated September 22, 2026. Enable with
`stage2.statistical_selection.selection_mode: "multi_model"`.

## 1. Purpose and scope

1. Build a reproducible confounder/modifier list from complementary empirical
   evidence and LLM interpretation. A candidate discarded by one model remains
   available to other models and final review.
2. Use existing consolidated, ontology-reviewed, extracted measurements. This
   mode does not repeat discovery/extraction. It can now select the final CATE
   estimator as well as its modifier inputs.
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
8. **Every modifier screen uses the same configured overlap restriction**:
   univariable `T:candidate` tests, penalized outcome interactions, orthogonal
   linear models, candidate R-learners, and causal forests. Set
   `stage2.min_propensity: 0.1` and `stage2.max_propensity: 0.9`, as in the example,
   to retain estimated propensities **including** the endpoints. The previously
   unrestricted univariable and penalized interaction screens now honor these
   bounds. Bounds remain optional for other configurations; these are eligibility
   filters, separate from numerical propensity clipping.
9. Eligibility uses cross-fitted training propensities and training-only models
   for validation propensities. No oracle propensity or outer-test outcome enters
   screening. Insufficient eligible rows or treatment arms make a screen
   unevaluable; it never falls back to all patients.
10. Univariable treatment/outcome associations, penalized main-effect models,
    predictive forests, and scoring nuisances retain all sampled patients.
    Main-effect evidence from the *joint interaction* model necessarily shares
    its overlap-restricted population. Audits record the relevant row IDs, and
    LLM prompts describe these population differences without exposing row IDs.

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

## 5. Joint modifier-count and final-estimator selection

1. Enabled by default **within `multi_model` mode**, through `modifier_count`.
   The `estimators` list defaults to `["causal_forest", "linear_interactions"]`.
   Each architecture is evaluated at every modifier budget; the budget is not
   chosen using only a forest before considering the linear alternative.
2. `minimum_r_loss` chooses the architecture/count pair with the smallest mean
   validation R-loss. Exact ties prefer fewer modifiers, then the linear model.
   Optional `one_standard_error` chooses the simplest pair, in that same order,
   whose paired excess loss over the best is within one paired standard error
   across folds. This is a simplification heuristic, not a significance test.
3. Preserve **every confounder retained by full-training role adjudication**.
   This does not force its penalized coefficient to be nonzero. Investigator
   locks remain exact. Locked modifiers are always included outside the additional
   candidate budget; locked confounder-only variables cannot acquire interactions.
4. Additional budgets default to `0, 4, 8, 12, 16, 24, 32`, plus the complete
   ranked shortlist, capped at 64 unlocked candidates. Clip/deduplicate counts
   when fewer candidates exist. With no locked modifiers, zero is a constant
   residual-effect forest; the interaction model retains treatment and covariate
   main effects but has no explicit interactions. A logistic model can still
   produce varying probability differences because baseline risk varies.
5. Preserve the original inner folds. **Inside each fold's training portion**,
   create subfolds with `internal_cv_folds`, rerun all seven evidence families,
   and obtain a fresh LLM modifier ranking. Every other patient's label is masked
   before the nested evidence worker runs. Neither validation outcomes nor a
   global outcome-informed ranking enter these training fits.
6. The LLM considers every unlocked candidate, even those without a provisional
   modifier role. Bounded requests sort batches and merge ordered windows using
   prompt-safe aggregate evidence. Every candidate is accounted for; rankings
   require candidate-specific effect citations. This remains a heuristic ordering,
   not an exhaustive subset search or another measurement-consolidation pass.
7. Reuse the parent numerical pass's **fixed all-candidate elastic-net nuisances**.
   Training residuals are cross-fitted; validation residuals use models fitted
   only on that fold's training patients. Scoring residuals and overlap-eligible
   validation patients are identical for every count and architecture. Broad
   scoring nuisances do not depend on the full-training selected confounder set.
8. Fit two architecture families on eligible training patients:
   - **Causal forest:** production feature encoding, configured `estimation_trees`,
     minimum leaf size 10, square-root feature sampling, 45% subsampling, honesty
     and inference enabled. It fits the fixed residuals. Average three forest
     seeds per fold/count by default.
   - **Penalized outcome interactions:** treatment main effect, all frozen
     candidate main effects, and treatment interactions only with the selected
     modifier prefix and locked modifiers. Use a logistic link for binary
     outcomes and identity for continuous outcomes. Group elastic net uses the
     configured `l1_ratio`; tune its penalty by outcome loss inside training,
     refitting the encoder in every penalty-CV partition. One deterministic fit
     per fold/count predicts both treatment states; CATE is `mu1 - mu0` on the
     outcome/probability scale. No validation labels enter penalty fitting.
9. Score both families with
   `mean(((Y - m_hat) - (T - e_hat) * tau_hat)**2)` on the same validation rows.
   Average forest seeds within folds, then average fold losses equally. Require
   at least two common usable folds and finite scores for every option; failures
   are visible rather than silently comparing different populations. Seeds are
   not independent patient replications. Even when all features are locked,
   architectures are compared on the fixed locked modifier set.
10. Choose the pair, form a final ranking from all outer-training evidence, and
    take the chosen prefix. Preserve the selected architecture, complete loss
    surface, fold rankings, row IDs, predictions, model audits, seeds, locks,
    role decisions, and source/input fingerprints. Final refitting uses retained
    confounders/modifiers: the DML forest refits nuisances; the interaction outcome
    model retunes its penalty with retained main effects and selected interactions.
11. Validation therefore uses broader adjustment than the final retained-role
    refit. It is an architecture/count comparison conditional on that adjustment
    and the **frozen upstream catalog, extractions, ontology, and consolidation**,
    not a fully nested rerun of discovery or full LLM confounder selection. R-loss
    is noisy and does not optimize oracle recall or Pearson ITE correlation.

## 6. Routing, checkpoints, and use

1. The chosen architecture is passed to the final estimator on fresh and resumed
   selections. Forests use modifiers in X and pure confounders in W. Interaction
   models use retained covariate main effects, a treatment main effect, and
   treatment interactions with selected modifiers. No extraction definition is
   changed. AIPW ATE estimation/calibration continues using separate nuisance
   models for either architecture. Penalized interaction models currently emit
   CATE point estimates only; individual-effect intervals remain missing, while
   the existing AIPW ATE interval remains available. `mu0`/`mu1` in the common
   predictions file are the AIPW nuisance predictions, not interaction-model
   counterfactual predictions.
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
   Each LLM request and each fold/architecture/count/seed fit can resume independently;
   completed result hashes are validated. Count-only policy changes leave the
   underlying numerical-evidence fingerprint unchanged. When count selection is
   enabled, `estimation_trees` is also a selection parameter: changing it requires
   guarded reselection and invalidates the prior count-selection result.
8. This update uses `stage2_multi_model_selection_v3` and
   `stage2_nested_modifier_count_v2`. Version 1/2 configuration blocks are accepted
   as input, but existing runs require guarded reselection before adopting these
   changes. Preflight blocks stale completion shortcuts. Set `estimators` to
   `["causal_forest"]` for forest-only count tuning, or set `modifier_count.enabled`
   to false to retain broad LLM modifiers and the historical forest final model.
   Other selector modes retain their selection policies.

Configuration block:

```json
{
  "stage2": {
    "min_propensity": 0.1,
    "max_propensity": 0.9,
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
          "forest_seeds": 3,
          "estimators": ["causal_forest", "linear_interactions"]
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
`5 × 8 budgets × 3 seeds = 120` forest-validation fits (zero-budget constant
models do not need a forest), plus `5 × 8 = 40` interaction models with nested
penalty tuning. The nested numerical work is substantially more
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
4. Selection tests run real binary/continuous numerical models, forests, and
   penalized interaction outcome models with controlled LLM replies. They check nested row/label isolation, categorical
   measurements, independently recomputed validation R-loss, common populations,
   confounder/lock preservation, zero-budget behavior, exact resume without
   refitting, corruption detection, and version 1/2 preflight rejection. They do
   not measure a live LLM's ranking quality. Additional checks change excluded
   patients' outcomes while holding nuisances fixed, verify all effect evidence
   is unchanged, check probability-scale counterfactuals, and verify production
   routing plus prediction invariance to changed outer-test outcomes.

Method references: [EconML causal forest](https://www.pywhy.org/EconML/_autosummary/econml.grf.CausalForest.html)
and [GRF variable importance](https://grf-labs.github.io/grf/reference/variable_importance.html).
Count-selection objective: [EconML RScorer](https://www.pywhy.org/EconML/_autosummary/econml.score.RScorer.html).
