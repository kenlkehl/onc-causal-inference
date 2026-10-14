# How Stage 2 works

Stage 2 turns the Stage 1 evidence into **measurable clinical variables, a patient-by-variable dataset, and ultimately held-out treatment-effect estimates**.

This describes our current **`multi_model` pipeline with full-record LLM extraction, predictive nuisance selection, and programmatic modifier ranking**, as of October 14, 2026. The 1,000-patient run supplies the concrete example. The treatment-penalty fix, statistical nuisance selection, and balanced adjusted-interaction selection apply to future fits; the completed run predates them.

1. **Set aside the outer held-out patients.**

   The pipeline runs separately within each of five outer folds. For a given fold, roughly 800 patients are available for discovery, extraction refinement, feature selection, and model fitting. The other 200 patients supply the final held-out evaluation.

   There are additional splits *inside* those 800 patients:

   | Partition | Approximate size | Purpose |
   |---|---:|---|
   | Outer training | 800 | Develop and fit this fold’s pipeline |
   | Outer held-out | 200 | Obtain final held-out predictions |
   | One inner training partition | 640 | Construct a modifier ranking and fit candidate models |
   | Its inner validation partition | 160 | Compare modifier counts and architectures |
   | Further subdivisions of the 640 | Three folds | Build nested evidence and tune regularization |

   Each patient eventually belongs to the outer held-out set exactly once. Different outer folds can discover different variables and choose different models.

2. **Compile the Stage 1 evidence into readable cards.**

   Stage 1 supplies evidence from its different modeling approaches: lexical features, embedding contrasts, topics, neural queries, and so on. Stage 2’s evidence compiler groups related evidence and produces cards containing representative clinical excerpts, with source provenance retained.

   Our configuration allows **up to 400 cards per outer fold**, with up to four representative excerpts per card.

   A card is neither a patient nor a variable. One card can mention several clinical attributes, and the same attribute can appear on many cards. Consequently, 400 cards can produce substantially more than 400 candidate variables.

   Stage 1’s contribution is therefore partly determining **which pieces of clinical text become evidence for variable discovery**. [Evidence compiler](oci/inference/plain_handoff_stage2_evidence.py)

3. **Have Gemma identify clinical variables supported by each card.**

   Gemma receives one evidence card per discovery request and names the distinct, measurable attributes supported by its excerpts.

   For example, an illustrative excerpt mentioning:

   > ECOG 2, elevated neutrophil-to-lymphocyte ratio, and liver metastases

   could produce three candidates. The discovery prompt encourages finding all supported attributes, including secondary findings. It does not limit discovery to whatever feature originally caused Stage 1 to retrieve the excerpt.

   At this point, the question is **“What could we measure?”** We have not yet established that any candidate is a confounder or effect modifier.

   The synthetic dataset’s name—five confounders and five modifiers—does not tell this procedure which variables to find or how many to retain.

4. **Merge duplicate candidates and define exactly how each variable will be measured.**

   Candidates from different cards often overlap. The consolidation process compares bounded batches, using semantic similarity and name-based organization to bring likely duplicates together.

   This is a **merge-only operation**: equivalent names for the same scalar measurement can be consolidated, with their evidence combined. Distinct measurements should remain separate. For example, neutrophil count and neutrophil-to-lymphocyte ratio should not become one generic “inflammation” variable.

   Gemma then gives each retained variable an extraction contract specifying its meaning, value type, units or allowed categories, missing-value rules, and how to resolve repeated or conflicting observations.

   That last part matters. A variable might use the latest observation, maximum value, most frequently documented value, or an “any positive mention” rule. These are different measurements.

   **All supplied text is assumed to be pretreatment.** Dates can help resolve repeated observations; the pipeline is not supposed to re-establish pretreatment eligibility for each piece of text. [Discovery and consolidation](oci/inference/plain_handoff_stage2.py)

5. **Extract the candidate measurements from the outer-training patients.**

   The pipeline now applies those definitions to the approximately 800 training patients.

   Our current extraction configuration is:

   - **Gemma-based generative extraction**, with up to **10 variables per request**.
   - One patient’s record per extraction task.
   - Full-record processing; **Plumb and ColBERT extraction are disabled**.
   - Thinking initially disabled. Validation failures receive error-directed repairs, with the configured threshold of five repairs before enabling thinking on the same model.

   Long records are processed in ordered, lossless chunks. For ordinary scalar measurements, extraction carries forward the current values and relevant state as it moves through the record.

   Variables defined by the **mode**, meaning the most frequently documented value, use a separate observation-collection path. That path needs observations that can be counted across the record, rather than just a current best value.

   The result is a matrix: one row per patient and one column per candidate measurement. Extraction uses the text and measurement definitions; structured treatment/outcome labels and oracle features are not supplied as extraction targets.

6. **Repair measurement definitions, harmonize values, and remove extremely sparse candidates.**

   Several checks happen before supervised feature selection.

   Python validates the response structure and values. Repeated problems across patients can trigger an ontology revision—for example, clarifying a definition that repeatedly produces incompatible categories. Only measurements affected by changed definitions need re-extraction.

   Harmonization then standardizes observed representations. For example, different textual expressions of the same category can map to one canonical value.

   An aggregate supervisor reviews extraction summaries and failure patterns. These reviews concern **measurement quality**, without treatment/outcome-based model performance driving them. The run has bounded refinement rounds, and any last-round definition changes must be applied before modeling begins.

   After that, candidates observed in **fewer than 5% of outer-training patients** are removed. With 800 patients, fewer than 40 observed values means removal; exactly 40 survives. The full extraction remains available for audit.

   This produces a frozen preselection dataset and measurement catalog. Our optional outcome-informed ontology search and additional latent-feature consolidation are disabled in this configuration. [Extraction, review, and filtering](oci/inference/plain_handoff_stage2_analysis.py)

7. **Build statistical evidence about every surviving candidate.**

   This is the first major CPU modeling phase.

   The pipeline asks three different questions:

   - Does the variable help explain **treatment assignment**?
   - Does it help explain **outcome**?
   - Does it help explain **variation in the treatment effect**?

   It uses several complementary model families:

   | Evidence family | What it contributes |
   |---|---|
   | Univariable models | Treatment association, treatment-adjusted outcome association, and treatment–variable interaction evidence |
   | Grouped main-effect elastic nets | Joint treatment and outcome predictive support |
   | Grouped interaction outcome models | Support for covariates and treatment interactions in a joint outcome model |
   | Orthogonal effect models | Joint DINA evidence for binary outcomes; residualized R-learner evidence for continuous outcomes |
   | Single-candidate effect models | Whether one candidate improves held-out effect modeling over a constant-effect model |
   | Predictive forests | Nonlinear treatment/outcome prediction importance |
   | Causal forests | Nonlinear treatment-effect heterogeneity evidence |
   | DINA permutation evidence | How held-out binary-outcome likelihood changes when candidate measurements are disrupted |

   The models also need **nuisance predictions**: estimated treatment probabilities and expected outcomes given the clinical measurements. These let the effect models distinguish ordinary prognosis or treatment-selection patterns from treatment-effect variation.

   Nuisance predictions for training patients are cross-fitted: a patient’s prediction comes from a model fitted without that patient. Encoders and regularization tuning are likewise fitted within the applicable training partition.

   The evidence is repeated across patient partitions and feature subsets. In our configuration, joint models use the full candidate set plus four randomized feature partitions, with subsets generally containing 32 features. This helps assess a variable in multiple covariate contexts.

   Categorical levels and missingness columns stay grouped with their parent variable. Permutation checks move those columns together.

   We also retain whether the **whole effect model generalizes better than a constant-effect model**. A variable’s apparent importance is less persuasive when the model generating that importance performs poorly. [Statistical evidence procedure](docs/stage2_multi_model.md)

8. **Select nuisance-model inputs statistically, without Gemma retention decisions.**

   The numerical pass fits grouped elastic nets to predict treatment and outcome, tuning regularization within each training partition. It also fits separate control-arm and treated-arm outcome models.

   For each nuisance task, retain a candidate if its encoded group has a nonzero coefficient norm in any training fold. Outcome inputs use the union of support from the marginal and arm-specific outcome models. Treatment-only and outcome-only inputs remain distinct; categorical levels and missingness indicators stay grouped with their parent measurement.

   These are predictive routing decisions, not declarations that a variable is causally a confounder. The legacy `confounder` role records membership in either nuisance task. Investigator-specified roles remain locked.

   The later modifier search preserves each nuisance task's support. It can add modifier roles without forcing those variables into the propensity model; retained modifiers remain available to outcome models.

   Gemma makes no role-retention requests in this mode, even if an older configuration enables role adjudication. The modifier ranker still considers all unlocked candidates. Scoring nuisances and final external training-patient cross-fitting use all candidates with fold-local regularization, not the pooled shortlist that has already seen those patients' labels. The final forest's internal DML nuisances still use its retained X/W inputs.

9. **Rebuild evidence within each inner training partition and rank modifiers programmatically.**

   This is the expensive “nested evidence” work we discussed previously.

   Consider one 640/160 inner split. Before comparing models on the 160 validation patients, the pipeline rebuilds statistical evidence using only the 640 training patients, with further cross-validation inside them. It constructs a fresh modifier ranking from that evidence.

   **Why rebuild it?** A ranking constructed from all 800 patients would have already used the outcomes of those 160 validation patients. Evaluating models built from that ranking on those same patients would favor a ranking that had already seen their labels.

   The initial all-800 evidence remains useful for final nuisance-input selection and the eventual final ranking. The nested evidence constructs each ranking without access to its scoring partition's labels.

   The current ranker gives **equal default weight to two evidence groups**.
   The first is the conventional candidate-by-treatment interaction test, adjusted
   for cross-fitted propensity and marginal outcome prediction. Its reduced and
   full models contain the same treatment, candidate main effects, and flexible
   score terms. Only the full model adds the candidate-by-treatment interaction;
   the coefficients are fitted separately in the two models. Binary outcomes use
   a logistic likelihood-ratio test, and continuous outcomes a nested linear F test.

   BH-adjusted q-values are converted to bounded strengths, averaged within folds,
   and summarized by their median across folds with a penalty for missing tests.
   Overlapping repeats do not count as independent experiments. These are useful
   selection signals, not guaranteed post-selection significance statements.

   The second group combines predictive evidence. For binary outcomes this uses:

   - Single-candidate DINA validation gains.
   - Joint DINA permutation importance.
   - Causal-forest permutation importance measured through R-loss.

   It reduces support for noisy or incompletely evaluated signals and downweights joint evidence from unreliable models. Repeated appearances of the same patient are averaged before calculating uncertainty diagnostics.

   After those adjustments, the predictive source scores combine as:

   $$
   \text{predictive score}
   = \frac{\max(\text{source scores})
   + 0.25\,\operatorname{mean}(\text{source scores})}{1.25}.
   $$

   The final rank score is **0.5 × predictive score + 0.5 × adjusted-interaction score**. The weight is configurable. Conservative equivalent-measurement grouping helps avoid filling the shortlist with near-duplicate variables.

   **Gemma performs no pairwise ranking or merge-sort comparisons here anymore.** The ranking is deterministic given the saved evidence. [Ranking criterion](docs/stage2_multi_model.md#statistical-modifier-ranking)

10. **Choose the modifier count and final model architecture together.**

    Within each inner training partition, the pipeline fits models using different prefixes of that partition’s ranking.

    The configured additional-modifier budgets are:

    $$
    0,\ 4,\ 8,\ 12,\ 16,\ 24,\ 32,
    $$

    plus the complete ranked shortlist, capped at 64 candidates. Counts are reduced when fewer eligible representatives exist. Explicitly locked modifiers, if configured, are included separately.

    At each count, it compares three architectures:

    **Causal forest.** During this comparison, it models treatment-effect variation using fixed, cross-fitted treatment and outcome residuals. Three forest seeds are evaluated per fold/count.

    **Penalized interaction outcome model.** For binary outcomes, its form is:

    $$
    \operatorname{logit}P(Y=1\mid T,Z)
    =\alpha+\beta_TT+\beta^\top Z
      +T\,\gamma^\top Z_{\text{selected modifiers}}.
    $$

    During count selection, all frozen candidate main effects remain available, while only the selected modifier prefix gets treatment interactions. It predicts each patient under both treatment states and takes the probability difference.

    **DINA.** It fits a conditional log odds ratio, $\delta(X)$, using cross-fitted nuisance offsets and treatment centering:

    $$
    \operatorname{logit}P(Y=1\mid T,X)
    =\nu(X)+[T-a(X)]\delta(X).
    $$

    Here $a$ and $\nu$ are constructed from estimated propensity and arm-specific outcome probabilities. Its probability-scale effect is:

    $$
    \hat\tau(X)=
    \operatorname{expit}\{\nu+(1-a)\delta\}
    -
    \operatorname{expit}\{\nu-a\delta\}.
    $$

    **All three architectures are compared using the same validation R-loss**, on the same validation patients:

    $$
    \frac1n\sum_i
    \left[
      Y_i-\hat m(X_i)
      -\{T_i-\hat e(X_i)\}\hat\tau(X_i)
    \right]^2.
    $$

    Thus DINA is fitted using its Bernoulli objective, but its converted probability-difference predictions compete with the other architectures using a common criterion.

    The new default, **`balanced_evidence`**, first keeps pairs within one paired
    standard error of the best mean R-loss, plus a small tolerance of 0.01% of
    that loss. Among these pairs, it chooses the budget with the strongest
    repeated adjusted-interaction support from the nested training rankings.
    Support above the strength corresponding to q = 0.1 contributes positively;
    weaker support contributes negatively; missing tests contribute zero. These
    contributions are summed over each prefix, divided by the full ranking's
    length, and averaged across folds. A utility tie favors fewer modifiers.

    At that budget, predictive fit chooses the architecture, with numerical ties
    preferring linear interactions, then DINA, then forest. Thus a tiny R-loss
    difference cannot automatically erase strong interaction evidence, while
    clearly worse predictive performance still excludes a model. This remains
    a heuristic, not an error-rate guarantee. The completed 1,000-patient run
    used the older `minimum_r_loss` rule, which remains configurable. Neither
    rule uses oracle-feature recall or oracle ITE correlation for selection.

    Zero modifiers remains a valid candidate. With our latest fix, the interaction model’s main treatment coefficient is unpenalized; DINA already protects its constant log odds ratio. Zero modifiers therefore does not force zero treatment effects.

    One qualification: this nested comparison holds the upstream catalog, extraction, and ontology fixed. It does not redo the entire discovery pipeline inside each 640-patient partition. The outer 200-patient evaluation remains the held-out assessment of the complete pipeline. [Count and architecture selection](docs/stage2_multi_model.md#5-joint-modifier-count-and-final-estimator-selection)

11. **Freeze the selected features and extract them from the outer held-out patients.**

    Once the winning architecture and modifier count are chosen, the pipeline constructs the final ranking from all 800 training patients and takes its selected prefix, preserving separate treatment/outcome nuisance supports.

    It then measures the required variables in the 200 outer held-out patients, applying the frozen definitions and training-derived harmonization rules. Compatible previously extracted measurements can be reused.

    Held-out text is used to measure those patients’ covariates. Their treatment/outcome labels do not revise the feature set, ranking, or chosen architecture.

12. **Refit the chosen model on the outer-training cohort and predict held-out effects.**

    The selected architecture is fitted again using the retained variables and the available outer-training patients.

    The final refit differs somewhat from the broad-adjustment model comparisons:

    - The causal forest uses modifiers as its heterogeneity inputs and pure confounders as adjustment inputs, fitting fresh DML nuisances.
    - The interaction model uses retained covariate main effects and selected treatment interactions, retuning its penalty.
    - DINA uses the external cross-fitted training nuisances and task-specific held-out nuisances, then fits its grouped log-odds-ratio effect model.

    This produces one held-out `estimated_cate` per patient. That is the model’s estimated conditional treatment effect given the extracted measurements—the quantity we have been calling its estimated ITE.

    The current run has no propensity-based eligibility trimming configured. Numerical propensity clipping is used separately where needed for weighted calculations.

    Across five outer folds, the result is 1,000 held-out effect predictions, potentially produced by five different selected models. [Final estimation](oci/inference/plain_handoff_stage2_analysis.py)

13. **Compute ATE estimates, diagnostics, and the post-hoc oracle comparison.**

    Alongside the effect model, the pipeline fits separate propensity and arm-specific outcome nuisance models. These produce an augmented inverse-probability-weighted score:

    $$
    \psi_i=
    \hat\mu_1(X_i)-\hat\mu_0(X_i)
    +\frac{T_i\{Y_i-\hat\mu_1(X_i)\}}{\hat e(X_i)}
    -\frac{(1-T_i)\{Y_i-\hat\mu_0(X_i)\}}{1-\hat e(X_i)}.
    $$

    Averaging these scores gives the AIPW ATE estimate, with uncertainty summaries. **The AIPW ATE and the average predicted CATE are separate outputs.** Likewise, the common `mu0` and `mu1` columns are nuisance predictions, not necessarily the counterfactual predictions underlying `estimated_cate`.

    That distinction explains how the completed fold 1 could have zero predicted CATEs while still reporting a nonzero AIPW ATE.

    Finally, after the outer predictions are frozen, the synthetic-data evaluator joins them to the oracle probability-scale treatment effects. It calculates by-fold and pooled Pearson/Spearman correlations, prediction errors, and related diagnostics. Oracle values do not participate in selection or fitting. [Post-hoc oracle evaluation](oci/inference/plain_handoff_stage2.py)

Throughout these steps, compatible extraction, evidence, ranking, and fit results are checkpointed for reuse. LLM work runs through the managed Gemma servers; numerical work shares the configured **128-worker CPU budget across the run**, rather than receiving 128 workers per fold. Independent tasks can run concurrently, while dependent steps—such as ranking before evaluating its prefixes—retain their required order.
