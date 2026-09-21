# Stage 2 selection from multiple models

Implemented September 21, 2026. Enable with
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
5. A second pass assigns confounder, modifier, both, or neither to every candidate.
   Nonlocked retained roles require citations: modifiers cite heterogeneity
   evidence; confounders address both treatment and outcome evidence. No positive
   p-value or minimum-frequency gate is imposed. Investigator locks are exact.
6. Decisions describe supporting/contrary evidence, fold consistency, disagreement
   across methods, rationale, and stability (`consistent`, `mixed`, or
   `insufficient`). Those labels are interpretations, not calibrated probabilities.
7. Explicit prompt allowlists exclude cell logs, row values/IDs, predictions,
   paths, exception messages, and oracle metadata. Oversized prompts split or
   fail with a budget error; no candidate is silently omitted.

## 5. Routing, checkpoints, and use

1. Final LLM roles feed the existing estimator adapter: confounders enter nuisance
   adjustment, modifiers enter forest X, and dual-role variables receive both
   uses. No extraction definitions are changed by theme review.
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
        "max_prompt_chars": 100000
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

With 352 candidates, five inner folds, and default repetitions, each forest
family fits `5 × (1 + 2 × ceil(352/32)) = 115` subset models. Treatment/outcome
forests are separate: 230 predictive forests plus 115 causal forests, alongside
linear models and univariable tests. Installing the mode does not launch a run.

## 6. Validation and limits

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

Method references: [EconML causal forest](https://www.pywhy.org/EconML/_autosummary/econml.grf.CausalForest.html)
and [GRF variable importance](https://grf-labs.github.io/grf/reference/variable_importance.html).
