# Stage 2 selection from multiple models

Implemented September 21, 2026; overlap-restricted modifier screening and joint
modifier-count/estimator selection updated September 22, 2026; nuisance-adjusted
interaction tests and balanced selection updated October 14, 2026. Enable with
`stage2.statistical_selection.selection_mode: "multi_model"`.

On `main`, Bernoulli DINA is used for binary orthogonal/candidate effect fits,
provides evidence alongside the forest, and participates in the final
architecture search. All five root synthetic shell launchers enable this mode
for fresh runs. See [DINA integration](gao_hastie_dina.md) for the equations and
the corresponding Stage 1 changes.

## 1. Purpose and scope

1. Build reproducible nuisance-input and modifier lists from predictive models
   and complementary empirical evidence. Nuisance retention uses task-specific
   grouped elastic-net supports; modifier ranking/count selection remains separate.
2. Use existing consolidated, ontology-reviewed, extracted measurements.
   An upstream filter drops candidates more than 95% missing across
   outer-training patients, before any numerical or LLM selection. Exactly
   95% missing is retained; held-out missingness cannot alter the catalog. This
   mode does not repeat discovery/extraction. It can now select the final CATE
   estimator as well as its modifier inputs.
   An optional upstream [estimand-informed ontology search](stage2_estimand_ontology.md)
   can compare alternative definitions and add supported measurements before
   this selector runs.
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
| Univariable models | Candidate association with treatment; outcome association adjusted for treatment | `Y ~ T + Z + f(e_oof) + g(m_oof) + T:Z`; logistic for binary outcomes, linear otherwise |
| Penalized main effects | Separate grouped elastic nets for treatment and marginal outcome | None; predictive support alone is not a modifier label |
| Penalized outcome interactions | Main-effect support in the joint outcome model | Grouped elastic net for `Y ~ T + all Z + all T:Z` |
| Orthogonal linear model | Cross-fitted grouped nuisances | Joint group elastic-net DINA for binary outcomes; grouped `Y_res ~ T_res * (constant + Z)` for continuous outcomes |
| Candidate effect learners (`univariable_rlearner`) | Same cross-fitted nuisances | One candidate group at a time; held-out DINA likelihood gain over a constant log odds ratio for binary outcomes, R-loss gain for continuous outcomes |
| Predictive forests | Held-out group-permutation importance for treatment and outcome | None; these are evidence models, not causal-forest nuisances |
| Causal forests | All-candidate elastic-net nuisance adjustment | Honest forests on residuals; held-out R-loss permutation importance and grouped split importance |
| DINA (binary only) | Cross-fitted propensity and arm-specific outcome probabilities | Additional held-out Bernoulli likelihood group-permutation evidence |
| Matched-batch contrasts (optional) | Shared cross-fitted treatment/outcome predictions define matching bins | Candidate-wise ridge on batch-average measurements; held-out deviation prediction gain over bin intercepts |

1. Categorical columns remain one candidate group. Interaction tests are omnibus;
   permutations move all encoded columns of a candidate together, preserving its
   category/missingness coding.
2. Conventional single-candidate modifier tests adjust for cross-fitted propensity
   `e_oof = P(T=1|X)` and marginal outcome prediction `m_oof = E(Y|X)` from all
   candidate covariates. Both reduced and full models include intercept, treatment,
   candidate main effects, and the **same fixed score bases on the same patients**.
   Only the full model adds `T:candidate`. Adjustment coefficients are refitted
   in each model; they are not constrained to be equal. Each nonconstant score
   gets a two-degree-of-freedom natural cubic spline with 10/50/90% quantile knots;
   tied knots and redundant columns reduce its dimension. Probability scores are
   clipped to `[1e-6, 1-1e-6]` and transformed to logits first; continuous outcome
   predictions retain their original scale before standardization. Binary tests
   use a likelihood-ratio test and continuous tests a nested F test. Scores are
   cross-fitted inside the evidence training partition; no validation labels enter.
   These are ordinary adjusted tests, not orthogonal score tests. Their nominal
   p-values condition on learned scores and do not guarantee post-selection inference.
3. Logistic interactions and binary DINA fits concern log odds ratios. Causal
   forests concern outcome/probability differences. A constant log odds ratio
   may imply varying risk differences, so the methods can support different
   modifiers. Final DINA CATEs are converted to probability differences.
4. Nominal p < 0.05 and BH q < 0.10 create separate support flags within each
   resample and endpoint. Neither is a mandatory per-variable inclusion gate.
   Q-value strength contributes to ranking and the balanced budget preference
   below. These q-values do not correct upstream adaptive discovery or guarantee
   calibrated false-discovery control with learned adjustment scores.
5. Penalized models choose regularization inside training, with encoders refitted
   inside every tuning partition. Only penalties converging in every tuning fold
   qualify. Nonconverged final fits are unevaluable. Clone audits record selected
   penalties, boundary selections, iteration counts, and convergence.
   The explicit main treatment coefficient is unpenalized in interaction outcome
   models and continuous-outcome R-learners, during both tuning and refitting.
   Covariates and treatment interactions retain their group-lasso/ridge penalties.
   DINA likewise leaves its constant log odds ratio unpenalized. Fit audits list
   `unpenalized_groups`; an empty modifier set therefore does not force zero
   treatment effects through regularization. A data-supported zero estimate is
   still possible. This change invalidates prior numerical evidence and downstream
   model caches; existing runs need guarded reselection to adopt it.
6. Whole-model held-out gains versus constant predictions/effects accompany
   coefficient and forest importance evidence. A nonzero coefficient or positive
   split importance does not establish that the model generalized.

## 3. Resampling and honest evaluation

1. Preserve original inner-fold memberships. Fit on inner-training patients and
   score predictions on untouched inner-validation patients. Splits must partition
   exactly the supplied outer-training rows. The original partition is retained,
   with two additional independently seeded partitions by default; each patient
   validates once per partition. No outer-test rows enter this resampling.
2. Fit all-candidate treatment and marginal-outcome elastic nets. Cross-fit them
   again inside inner training to obtain training residuals; separate models
   trained on that population predict validation residuals. Encoder and penalty
   fitting are confined to training. Binary DINA also uses separately
   cross-fitted treatment-arm outcome probabilities from these training rows.
3. Keep nuisances fixed across that fold's perturbations. This assesses evidence
   stability conditional on one honest nuisance construction, not a bootstrap of
   the entire discovery process.
4. Individual and diagnostic linear models retain three perturbations, with
   configured row subsampling and L1 ratios. Joint models use all their available
   training rows and the configured selection L1 ratio; DINA still tunes alpha
   internally. This separates model context from changes in rows and penalties.
5. Each partition fits one full candidate model and four balanced random feature
   partitions, normally 32 features per subset. Every candidate appears once in
   each feature partition. Equivalent measurement groups stay together. Full and
   subset evidence are scored separately, retaining broad nuisance adjustment.
6. Default forests: 800 trees, minimum leaf size 10. Causal forests retain honesty,
   45% tree subsampling and inference disabled for evidence fits.
7. Ten paired group permutations score validation patients. All encoded columns
   and missingness indicators move together. Conservative equivalent-measurement
   groups also receive joint permutations. Nuisance predictions stay fixed.
8. **Every modifier screen uses the same configured overlap restriction**:
   univariable `T:candidate` tests, penalized outcome interactions, orthogonal
   linear models, candidate effect learners, causal forests, DINA, and matched batches. Set
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
    Reports describe these population differences explicitly.

### Matched-batch contrast evidence (September 28, 2026)

Enable `stage2.statistical_selection.multi_model.matched_batch.enabled`. The full
multi-model example enables it; omitted settings preserve existing saved runs.
This adds one modifier-evidence family and is compatible with the upstream
estimand-informed refinement. No ontology/representation search, new extracted
variables, correlation consolidation, or fixed modifier budget is imported from
the experimental batch-contrast worktree.

For each inner fold and its full/80%/80% training repetitions:

1. Reuse the existing honest treatment and marginal-outcome nuisance predictions.
   Apply the configured propensity bounds, then fit two quantile bins per nuisance
   on that repetition's training patients (up to four joint bins). A bin needs
   eight patients per treatment arm to provide a reference contrast.
2. Shuffle each arm within bins, making batches of four treated and four control
   patients. Reject arm differences in either nuisance exceeding 0.25 standard
   deviations of that nuisance in training. Patients appear at most once per pass.
3. The target is the batch's observed treated-minus-control mean outcome minus
   the same contrast in its **training bin**, not the overall cohort contrast or
   a known causal effect. Use 100 fresh training passes and 50 validation passes.
4. Fit a ridge regression separately for each candidate's batch-average encoded
   values, with bin intercepts. Keep the supplied measurement encoding; exclude
   standalone missingness indicators. Default penalty 0.01 multiplies mean loss,
   so additional shuffles do not weaken regularization.
5. Primary training emphasizes absolute deviations at least both 0.10 outcome
   units and one estimated standard error (`min_delta` and `z_threshold`). For a
   binary outcome, 0.10 means ten percentage points. Continuous-outcome users
   should set `min_delta` in their outcome units. Also fit on all matched training
   batches as a sensitivity check. Validation uses all matched batches in bins
   supported by both fits, without selecting extreme validation outcomes.
6. Score `(bin-only validation MSE - candidate validation MSE) / bin-only MSE`.
   Both predictions use training-fitted bin intercepts, so an intercept improvement
   cannot count as candidate evidence. Positive gain supplies a support flag;
   there is no automatic inclusion threshold. Lack of batches, within-bin feature
   variation, or nonzero baseline loss produces unavailable evidence. A missing
   filtered fit can still report an unfiltered diagnostic without a primary vote.
7. Aggregate by fit/fold, including filtered/unfiltered agreement, unique patient
   counts, batch counts and matching rejections. Repeated batches are dependent,
   not extra independent patients. Raw sampling details remain in numerical cell
   checkpoints; only allowlisted summaries reach the role and concept prompts.

The same family runs inside the training-only nested modifier rankings. Final
modifier-count and architecture selection still uses the existing individual-row
R-loss comparison. The family runs on CPU and requests no additional LLM calls
of its own. Config, source code and inputs participate in checkpoint identity.
Turning it on for a saved run requires the normal scientific-policy resume guard;
adding the code alone does not enable it in an already-running experiment.

This measures within-bin observed contrast variation. Residual confounding and
prognostic imbalance can remain, and heterogeneity primarily between bins may be
missed. Validation is conditional on the existing upstream catalog/refinement.

## 4. Stability summaries and predictive nuisance selection

1. For each candidate, family, and task, report exposure/evaluable/support counts,
   support fraction, fold counts, score mean/range, p/q summaries when applicable,
   and model-quality diagnostics. Missing or nonconverged fits are not negative
   votes; support fractions divide by evaluated fits.
2. Keep families separate. Folds/repetitions share patients, so support fractions
   are not independent replications, causal probabilities, or formal
   error-controlled stability selection. There is no pooled popularity score.
3. The already-fitted, training-only nuisance models select inputs. A nonzero
   candidate-group norm above `coefficient_tolerance` in any training fold retains
   that variable for its task. Penalties use training-only predictive CV loss.
   Treatment and outcome supports remain separate. Outcome
   support is the union of marginal, control-arm, and treated-arm model supports.
4. `nuisance_selection` records supports, coefficient norms, model-fit audits,
   training/validation row IDs, and fold frequencies. Frequencies across repeated
   partitions are descriptive. No univariable p-value or LLM decision gates retention.
5. Investigator locks remain exact. The legacy `confounder` role means a variable
   is used by at least one nuisance task; `nuisance_model_roles` records which.
   These are predictive routing decisions, not identified causal roles. A modifier
   does not automatically enter propensity prediction, but remains an outcome input.
6. No LLM theme or role-retention requests run in `multi_model`, including when an
   older configuration has `role_adjudication.enabled=true`. LLM discovery,
   measurement refinement, extraction, and optional modifier-concept review remain.
7. The pooled support routes final outer-held-out nuisance prediction.
   Scoring nuisances and final external training-patient cross-fitting retain the complete
   frozen candidate catalog and regularize inside each training fold. The pooled
   shortlist cannot screen its own OOF prediction inputs. Final DINA uses these
   external cross-fitted training nuisances and task-specific held-out nuisances.
   The causal forest still fits its internal DML nuisances on retained X/W inputs;
   its adjustment set and modifier set are learned on outer training. Inner
   architecture/count validation remains a separate, nested procedure.

## 5. Joint modifier-count and final-estimator selection

1. Enabled by default **within `multi_model` mode**, through `modifier_count`.
   The `estimators` list defaults to
   `["causal_forest", "linear_interactions", "dina"]`; DINA is omitted for
   continuous outcomes. Each architecture is evaluated at every modifier budget.
2. The default `balanced_evidence` keeps architecture/count pairs whose mean
   excess validation R-loss over the best is no larger than one paired fold
   standard error plus `relative_r_loss_tolerance * abs(best_mean_r_loss)`.
   That tolerance defaults to `1e-4` (0.01% of the best loss), with an absolute
   floor of `1e-12`. Among eligible pairs, choose the budget with the highest
   mean **nested-training interaction utility**, breaking ties toward fewer
   modifiers. At that budget choose the lowest-loss eligible architecture;
   architectures within the numerical tolerance prefer `linear_interactions`,
   then `dina`, then `causal_forest`. This lets adjusted interaction evidence
   affect retention without overriding a clear predictive disadvantage.

   For each nested training ranking, a candidate's interaction utility is its
   coverage-weighted strength minus the same coverage-weighted strength at
   `q_threshold` (default 0.1). Sum utilities over the candidate prefix and divide
   by that ranking's total length. Average these prefix utilities across the
   count-validation folds. Repeated strong support favors inclusion; weak support
   can reduce a prefix's utility; missing tests contribute zero. Validation
   outcomes never determine these utilities. This is a selection heuristic, not
   a test, an error-rate guarantee, or a mandatory q-value cutoff.

   Explicit `minimum_r_loss` still chooses the smallest mean validation R-loss,
   with exact ties preferring fewer modifiers and the architecture order above.
   Optional `one_standard_error` chooses the simplest pair whose paired excess
   loss is within one paired standard error. Existing explicitly configured
   rules are honored; omitted rules use `balanced_evidence`.
3. Preserve **each statistically selected nuisance task's feature support**.
   This does not force its penalized coefficient to be nonzero. Investigator
   locks remain exact. Locked modifiers are always included outside the additional
   candidate budget; locked confounder-only variables cannot acquire interactions.
4. Additional budgets default to `0, 4, 8, 12, 16, 24, 32`, plus the complete
   ranked shortlist, capped at 64 unlocked candidates. Clip/deduplicate counts
   when fewer candidates exist. With no locked modifiers, zero is a constant
   residual-effect forest; the interaction model retains treatment and covariate
   main effects but has no explicit interactions. A logistic model can still
   produce varying probability differences because baseline risk varies. DINA
   fits a constant log odds ratio at zero modifiers and can likewise produce
   varying probability differences through its nuisance offset.
5. Preserve the original inner folds. **Inside each fold's training portion**,
   create subfolds with `internal_cv_folds`, rerun every enabled evidence family,
   and obtain a fresh deterministic statistical modifier ranking. Every other patient's label is masked
   before the nested evidence worker runs. Neither validation outcomes nor a
   global outcome-informed ranking enter these training fits.
6. Statistical ranking considers every unlocked candidate, even those without a
   provisional modifier role. It combines individual and joint-model evidence
   using the [criterion below](#statistical-modifier-ranking). Every candidate
   receives a recorded score and evidence provenance. It makes no LLM requests.
   This is a heuristic ordering, not an exhaustive subset search or a
   measurement-consolidation pass.
7. Reuse the parent numerical pass's **fixed all-candidate elastic-net nuisances**.
   Training residuals are cross-fitted; validation residuals use models fitted
   only on that fold's training patients. Scoring residuals and overlap-eligible
   validation patients are identical for every count and architecture. Broad
   scoring nuisances do not depend on the full-training selected confounder set.
8. Fit the configured architecture families on eligible training patients:
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
   - **Binary DINA:** fit a conditional log odds ratio over the selected modifier
     groups with a cross-fitted offset and treatment centering. Tune group
     elastic-net penalties inside training, retaining/dropping categorical
     levels and missingness together. Predict both treatment states through
     the logistic link and return their probability difference as CATE.
9. Score every family with
   `mean(((Y - m_hat) - (T - e_hat) * tau_hat)**2)` on the same validation rows.
   Average forest seeds within folds, then average fold losses equally. Require
   at least two scoring folds; a combination is eligible only if its scores are
   finite on every scoring fold. Unsupported combinations are marked not
   estimable while supported architectures continue. Seeds are
   not independent patient replications. Even when all features are locked,
   architectures are compared on the fixed locked modifier set.
10. Choose the pair, form a final ranking from all outer-training evidence, and
    take the chosen prefix. Preserve the selected architecture, complete loss
    surface, fold rankings, row IDs, predictions, model audits, seeds, locks,
    role decisions, and source/input fingerprints. Final refitting uses retained
    confounders/modifiers: the DML forest refits nuisances; the interaction outcome
    model retunes its penalty with retained main effects and selected interactions;
    DINA reuses the external nuisance predictions and fits the grouped log-odds effect model.
11. Validation therefore uses broader adjustment than the final retained-role
    refit. It is an architecture/count comparison conditional on that adjustment
    and the **frozen upstream catalog, extractions, ontology, and consolidation**,
    not a fully nested rerun of discovery. R-loss
    is noisy and does not optimize oracle recall or Pearson ITE correlation.

## 6. Routing, checkpoints, and use

`stage2.modeling_workers` controls one CPU process pool shared by all outer
folds. The default is `1`; fresh RTX PRO 6000 × 8 wrapper runs use `128` for
both the one-confounder/one-modifier and five-confounder/five-modifier cohorts.
The value `1` retains the original serial execution within each outer fold;
values above `1` enable the shared pool and specify its total process limit.
`STAGE2_MODELING_WORKERS` overrides this value for fresh or saved synthetic
launches. It is operational configuration, separate from LLM request workers,
GPU count and statistical policy.

Missing nuisance fits, independent family/repetition/subset evidence jobs and
architecture/count/seed evaluations use this pool. Coordinators remain in the
parent process, outstanding process jobs are bounded globally by the worker count,
and each fit uses one native thread. Selection splits still advance in order;
within-fit penalty warm starts stay sequential. Results are collected in the
original deterministic order with the original seeds and row boundaries.
Completed checkpoints are read locally without submitting a fit.

Final estimation uses the same budget. Its cross-fitting splits dispatch the
treatment, control-outcome, and treated-outcome nuisance fits independently;
the three full-training nuisance fits also run independently. With five inner
splits, this provides 15 cross-fitting jobs per outer fold. Each job saves its
validated predictions in `estimation/fit_checkpoints/<input-and-source-identity>/`.
The cache checks input identity, result checksum, row count, and finite values.
An interrupted run reuses completed jobs even when resumed with a different
worker count. Changed inputs or implementation use a separate cache directory.

For final interaction models, independent regularization-CV partitions run in
the pool and are checkpointed individually. Penalty warm starts within each
partition stay sequential, and the full-training refit uses the same selected
penalty. This optional dispatch does not change selection-stage fitting.

Final forests reserve joblib thread capacity from the same global budget. Their
per-fit share is `max(1, modeling_workers // concurrent_outer_folds)`; a fit waits
for that capacity when other modeling jobs are active. FIFO reservations prevent
a stream of single-fit jobs from starving a forest reservation. It covers fitting,
internal nuisance CV, and prediction, and is released on success or failure.
It cannot oversubscribe the process pool; individual native matrix operations
still use one thread. One-worker and direct-call execution remain serial within
each outer fold. The worker budget is independent of dataset and GPU count.

The scheduling-only transition includes exact old/new source-hash aliases in
`stage2_modeling_source_aliases.json`, preserving existing numerical evidence,
rankings and selection fits. Only the reviewed source bytes match; subsequent
source edits invalidate the alias automatically. Changes to the worker count
alone do not change scientific identities.

1. The chosen architecture is passed to the final estimator on fresh and resumed
   selections. Forests use modifiers in X and pure confounders in W. Interaction
   models use retained covariate main effects, a treatment main effect, and
   treatment interactions with selected modifiers. No extraction definition is
   changed. AIPW ATE estimation/calibration continues using separate nuisance
   models for all architectures. DINA and penalized interaction models emit
   CATE point estimates only; individual-effect intervals remain missing, while
   the existing AIPW ATE interval remains available. `mu0`/`mu1` in the common
   predictions file are the AIPW nuisance predictions, not interaction-model
   counterfactual predictions. DINA separately exports `dina_log_odds_ratio`,
   `dina_mu0`, and `dina_mu1`.
2. Omitted selectors in the core configuration preserve `llm_roles`; fresh root
   synthetic launchers explicitly select `multi_model`. `independent_tasks`
   keeps binding joint-model selection and advisory LLM annotations. New defaults
   are omitted from both legacy policy fingerprints.
   Explicit saved-run launches preserve the saved selector and estimator list;
   add `"dina"` to an older explicit list to include it in final-model search.
3. `role_adjudication.enabled` is not used for retention in `multi_model`.
   The bundled multi-model example and fresh synthetic launchers set it to false.
4. Each nuisance fit and family/repetition/subset has an atomic, integrity-checked
   checkpoint under `selection/multi_model/<input-fingerprint>/fold_*/`. Identity
   includes measurements, labels, definitions, splits, policy, seed, and numerical
   source hashes. Completed compatible leaves are reused on restart.
5. `selection/nuisance_candidates.json` saves the complete training candidate
   definitions and latent states for honest final cross-fitting and resume.
   Prior LLM retention checkpoints cannot supply the new selection.
6. Standard selection reports, definitions, measurement dependencies, and final
   estimation artifacts remain. Changing the mode or a completed multi-model
   policy requires guarded `--stage2-reselect`.
7. Count selection stores checkpoints under `selection/modifier_count/` and
   nested numerical evidence under `selection/multi_model/modifier_count_training/`.
   `selection/modifier_count/selection.json` points to the active count result.
   Each fold/architecture/count/seed fit can resume independently;
   completed result hashes are validated. Count-only policy changes leave the
   underlying numerical-evidence fingerprint unchanged. When count selection is
   enabled, `estimation_trees` is also a selection parameter: changing it requires
   guarded reselection and invalidates the prior count-selection result.
8. This update uses `stage2_multi_model_selection_v7_adjusted_interactions` and
   `stage2_nested_modifier_count_v7_balanced_interactions`. Version 1–6 configuration blocks are accepted
   as input, but existing runs require guarded reselection before adopting these
   changes. Preflight blocks stale completion shortcuts. Set `estimators` to
   `["causal_forest"]` for forest-only count tuning, or set `modifier_count.enabled`
   to false to use any-fold joint orthogonal-model group support for modifiers
   and the forest final model, without LLM retention.
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
        "forest_trees": 800,
        "forest_min_samples_leaf": 10,
        "feature_subset_size": 32,
        "permutation_repeats": 10,
        "stability": {"partition_repeats": 3, "subset_partitions": 4,
                      "group_equivalent_measurements": true},
        "nominal_p_threshold": 0.05,
        "q_threshold": 0.1,
        "interaction_ranking_weight": 0.5,
        "max_prompt_chars": 100000,
        "modifier_count": {
          "enabled": true,
          "candidate_counts": [0, 4, 8, 12, 16, 24, 32],
          "max_ranked_modifiers": 64,
          "selection_rule": "balanced_evidence",
          "relative_r_loss_tolerance": 0.0001,
          "forest_seeds": 3,
          "estimators": ["causal_forest", "linear_interactions", "dina"]
        }
      }
    },
    "role_adjudication": {"enabled": false}
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
their statistical rankings, one final full-training ranking, and up to
`5 × 8 budgets × 3 seeds = 120` forest-validation fits (zero-budget constant
models do not need a forest), plus `5 × 8 = 40` interaction models with nested
penalty tuning. The nested numerical work is substantially more
expensive than reusing a global ranking.

## 7. Validation and limits

1. Tests exercise real binary/continuous models, categorical encoding, nested
   nuisances, all evidence families, subset coverage, train/validation separation,
   and exact checkpoint reuse after changing outer-test labels and oracle columns.
2. Predictive nuisance tests check separate task supports, arm-outcome unions,
   exact investigator locks, grouped categorical selection, validation isolation,
   and zero LLM retention calls even with older enabled flags. Historical LLM
   adjudication tests remain for the retained legacy implementation.
3. A production comparison must still measure extraction quality, frozen-selection
   confounder/modifier recovery, held-out effect estimates, and outer-fold
   variability. More evidence families do not establish better sensitivity,
   specificity, or causal identification by themselves.
4. Selection tests run real binary/continuous numerical models, forests, and
   penalized interaction outcome models without LLM retention. They check nested row/label isolation, categorical
   measurements, independently recomputed validation R-loss, common populations,
   confounder/lock preservation, zero-budget behavior, exact resume without
   refitting, corruption detection, and obsolete-schema preflight rejection.
   Additional checks change excluded
   patients' outcomes while holding nuisances fixed, verify all effect evidence
   is unchanged, check probability-scale counterfactuals, and verify production
   routing plus prediction invariance to changed outer-test outcomes.

Method references: [EconML causal forest](https://www.pywhy.org/EconML/_autosummary/econml.grf.CausalForest.html)
and [GRF variable importance](https://grf-labs.github.io/grf/reference/variable_importance.html).
Count-selection objective: [EconML RScorer](https://www.pywhy.org/EconML/_autosummary/econml.score.RScorer.html).


## Statistical modifier ranking

The default ranking uses the saved numerical cells, not aggregate prose or an
LLM comparator. It replaces both initial LLM batch sorting and pairwise merges.
Two evidence groups receive equal default weight: **adjusted conventional
interaction tests** and **combined predictive evidence**. Within the predictive
group, binary outcomes use candidate DINA validation likelihood gain, grouped
joint DINA permutation likelihood-loss increase, and causal-forest permutation
R-loss increase. Continuous outcomes use candidate R-loss gain and forest
permutation R-loss increase. Thus adding predictive sources does not multiply
their collective weight against the conventional test. Globally disabled sources
are omitted; configured but failed sources retain their place in the denominator.
All encoded levels and missingness indicators stay with their feature group.

Within each source and model context, retain paired validation loss differences.
Average all appearances of each patient before calculating mean importance and a
patient-level standard-error diagnostic. Also record mean gains for each complete
patient partition and their standard deviation. The larger uncertainty measure
shrinks importance toward zero; repeated partitions are not independent experiments
and these uncertainty diagnostics carry no formal coverage guarantee.

Joint-model reliability is computed from its pooled paired gain over a constant
effect, separately for full and subset contexts. The weight is
`max(mean_gain, 0) / (abs(mean_gain) + noise_scale)` (zero for a zero denominator).
It applies to the entire context, equally for positive and negative feature
importance. Individual negative model fits no longer erase positive feature
importance while preserving negative importance. Raw signs remain in the audit.

Within a context, positive mean importance receives its within-source percentile,
multiplied by analogous uncertainty shrinkage and evaluated/planned coverage.
Joint contexts are combined with reliability weights, then multiplied by their
maximum reliability. Failed contexts supply no positive support. Individual
feature evidence remains a separate channel.

The predictive score is `(max(source scores) + 0.25 * mean(source scores)) / 1.25`,
preserving single-source support while giving a bounded agreement bonus and
putting the score on `[0, 1]`. Equivalent-measurement group evidence provides
additional channels only for members of accepted groups.

For the interaction group, transform each BH-adjusted q-value using
`s(q) = -log10(max(q, 1e-300)) / (1 - log10(max(q, 1e-300)))`.
Average strength across repeats within each fold, take the median across folds,
and multiply by evaluated/planned coverage. This bounded score rewards repeated
support without treating overlapping fits as independent experiments. Missing
or unevaluable tests supply no positive support. Legacy unadjusted tests cannot
supply this component.

The final score is `(1 - w) * predictive_score + w * interaction_score`, with
`interaction_ranking_weight = 0.5` by default. Setting it to zero restores
predictive-only primary ranking. If the conventional test family is globally
disabled, the predictive group has full weight; if only the conventional family
is enabled, it has full weight. A configured but failed group does not transfer
its weight to the other group. These are ranking heuristics, not probabilities
that variables are modifiers or calibrated significance probabilities.

Exact ties prefer average within-fold
regularized interaction/group selection stability, with equal weight per
available selection family. Next use median adjusted univariable interaction
q-value only when every candidate in that tie group has one; otherwise use the
stable feature ID. Q-values never gate entry. Main-effect prediction, raw
coefficient magnitudes, and forest split importance remain diagnostic evidence.

`ranking.json` includes all candidate component scores, raw fold summaries,
evaluation coverage, provenance, missing results, and generated explanations.
Only its top requested prefix is passed to count/architecture selection. Locks
remain outside that additional budget. Measurement grouping requires compatible
value types/units, identical terminal measurement names after removing cosmetic
words such as `value`/`measurement`, contextual name-token similarity of at least
0.8, at least 30 paired
training observations, at least 80% overlap, and Spearman correlation (or
categorical agreement) of at least 0.95. Groups use complete linkage; a chain of
weakly related measurements cannot merge a broad clinical theme. Negative
correlation and ambiguous text values do not qualify. Every scoring fold learns
its groups solely from its training measurements.

After group contribution is evaluated, one representative is shortlisted using
individual evidence, observed coverage, then stable ID. Original measurements and
group decisions remain in the audit; explicit investigator locks are preserved.
Count selection records the actual count if fewer representatives are available.
This conservative rule leaves some synonyms ungrouped, and a static ranking does not
estimate incremental value conditional on the chosen prefix. Final count and
architecture selection uses the same held-out probability/outcome-scale R-loss
as a predictive guardrail, with training-only interaction utility choosing among
eligible budgets under the default `balanced_evidence` rule. It includes the
zero-additional-modifier option.

Statistical ranking checkpoints are versioned by the numerical evidence,
definitions, criterion source, and requested maximum. LLM ranking checkpoints
are incompatible. The parent selection identity also includes the ranking
source hash, so a completed old selection cannot bypass a changed criterion.
Changing only the ranking weight leaves numerical caches reusable while
invalidating ranking and count results. Changing adjustment, stability settings,
or their implementation invalidates numerical evidence and downstream
selection; Stage 1 and frozen extracted measurements remain reusable.
When optional clinical concept review is enabled, its per-fold shortlists and
representative ordering use this statistical ranker; the concept review itself
remains an LLM task.
