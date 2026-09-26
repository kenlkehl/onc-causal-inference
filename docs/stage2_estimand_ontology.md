# Estimand-informed ontology refinement

`stage2.estimand_ontology.enabled=true` adds a bounded search for useful
alternative definitions of confounder and modifier candidates. It runs after
extraction-failure repair, value harmonization, and aggregate ontology review,
and before empirical alias consolidation and final role/model selection.

The search uses treatment and outcome information from outer-training patients.
The existing aggregate ontology supervisor remains outcome-blind. The two
components have separate prompts, configuration, and checkpoints.

## Procedure

1. **Establish a reference in each inner fold.** Ridge logistic models predict
   treatment; ridge logistic or linear models predict the outcome. Predictors
   include all original extracted candidates. Encoding, imputation, scaling,
   and rare-category handling use only the fitting rows. Three further
   training-only splits produce cross-fitted training residuals. Reference
   regularization is fixed in advance (`ridge_penalty=10`).
2. **Shortlist opportunities.** Confounder priority is the product of grouped
   coefficient norms for treatment prediction and outcome prediction conditional
   on treatment. Modifier priority is the grouped coefficient norm from a ridge
   R-learner. Average these training-split priorities over folds and take up to
   eight variables per role, deduplicating the union. Variables outside this
   shortlist remain available to downstream selection. Investigator-fixed
   definitions, constants, and variables with less than 20% observed values are
   excluded from proposal generation.
3. **Propose alternatives once.** The LLM receives one variable's definition,
   aggregate extracted-value summary, and clinical study question. It can
   propose up to two clinically justified definitions: a precise pretreatment
   time window, a named scale, a numerical measurement, or supported categories.
   It can also propose none. Patient labels, statistical scores, oracle columns,
   and outer-test patients are absent. Python supplies identifiers and clinical
   display labels. Investigator-fixed definitions remain unchanged.
4. **Extract the alternatives.** Proposals are requested concurrently under
   existing request limits. All proposed definitions are batched together through
   ordinary patient extraction. Original training columns are reused. Ordinary
   value harmonization handles mixed numerical/text values. These definitions
   are frozen before comparison.
5. **Compare the original with an added alternative.**
   - For **confounding adjustment**, refit treatment and conditional outcome
     nuisance models with all original candidates plus the alternative. Evaluate
     log loss for binary targets or squared error for continuous outcomes on
     inner-validation patients. Require improved outcome prediction, no more
     than 1% relative worsening of treatment log loss, and no more than a five
     percentage point decline in validation overlap coverage. Treatment
     prediction gains alone cannot qualify an alternative.
   - For **effect modification**, compare a ridge R-learner using the original
     measurement against one using that measurement plus the alternative.
     Evaluate `(Y - m(X) - (T - e(X)) * tau(X))²` on inner-validation patients.
     Reference nuisances and eligible patients stay fixed across definitions.
     Eligibility uses Stage 2 propensity bounds, defaulting to `[0.1, 0.9]`
     for this search. Each fold needs at least ten eligible training and
     validation patients. This is a small effect-model probe; the downstream
     architecture search still chooses the final model.
6. **Retain stable improvements.** A qualifying alternative must improve the
   mean loss by more than 0.5% and more than one paired standard error, and
   improve at least 60% of folds. These are tuning heuristics, not hypothesis
   tests. Choose at most one winning alternative per role per variable. A
   shared winner is added once. Original measurements remain unchanged.
7. **Continue selection and estimation.** The expanded catalog enters empirical
   alias consolidation, multi-model evidence, role adjudication, modifier-count
   selection, and final architecture search. A search win records a supported
   use; it does not lock a causal role. Only definitions required by final
   selection are extracted in outer-test patients using frozen rules.

## Configuration

The extension is disabled by default to preserve existing runs. To opt in:

```json
{
  "stage2": {
    "estimand_ontology": {
      "enabled": true,
      "max_features_per_role": 8,
      "max_alternatives_per_feature": 2,
      "minimum_nonmissing_fraction": 0.2,
      "minimum_relative_gain": 0.005,
      "paired_se_multiplier": 1.0,
      "minimum_winning_fold_fraction": 0.6,
      "maximum_propensity_loss_increase": 0.01,
      "maximum_overlap_fraction_drop": 0.05,
      "ridge_penalty": 10.0,
      "nuisance_crossfit_folds": 3,
      "minimum_training_rows": 20,
      "minimum_overlap_rows": 10,
      "max_prompt_chars": 40000
    }
  }
}
```

Defaults permit at most 16 original variables, 32 extracted alternatives, and
32 accepted alternatives across both roles. There is one proposal/comparison
round. Overlapping shortlists or a shared winner reduce these counts.

## Checkpoints and continuation

Each fold writes `estimand_ontology/report.json` and a content-addressed
`estimand_ontology/runs/<fingerprint>/` tree. It records definitions, training
measurement/label/text fingerprints, model identities, splits, nuisance cross-fit
membership, proposals, alternative extractions, fold losses, and winners.
Completed work is reused after interruption. Transport failures propagate;
invalid proposals and non-estimable comparisons retain originals with an audit.
Corrupt checkpoints fail visibly.

Expanded matrices live in `extraction/estimand_candidates_fit/<fingerprint>/`.
Selection records the exact matrix path. Guarded reselection preserves it and
the original matrix; adding alternatives never overwrites a frozen input.

Enabling the search preserves discovery and outcome-blind ontology-review
checkpoints. It changes the supervised selection fingerprint. For completed
runs, use the existing guarded `--stage2-reselect` workflow to archive prior
selection/estimation and reuse the frozen preselection matrix. Extraction must
remain enabled because new definitions require new measurements.

A Python process that already imported the Stage 2 analysis module continues
with its loaded implementation. Editing files or configuration does not activate
the extension inside that process. A subsequent invocation from saved
checkpoints adopts it. Do not run a second writer against an active output.

## Interpretation

Nuisance prediction and R-loss provide evidence of usefulness for estimation.
They cannot establish confounder status, prove absence of confounding, or
establish a uniquely correct ontology. Improved treatment prediction may also
emphasize an instrument, hence the requirement for outcome improvement.

Shortlisting, proposals, and harmonization use outer-training data. Inner CV
losses are tuning scores conditional on those upstream choices, not independent
performance estimates or a fully nested evaluation of discovery/extraction.
The outer-test fold evaluates the resulting procedure. The later modifier-count
search is likewise conditional on the chosen measurement catalog. This search
cannot recover concepts discarded in discovery or split an arbitrary composite
into an unconstrained set of new clinical variables.
