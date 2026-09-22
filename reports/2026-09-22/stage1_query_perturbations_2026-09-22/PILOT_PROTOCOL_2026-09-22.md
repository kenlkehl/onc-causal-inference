# Prespecified query-perturbation pilot — September 22, 2026

1. **Scope**
   1. Use outer fold 1 only. Its 200 outer-test patients are excluded from all
      fitting, validation, prompts, candidate interpretation, and reporting.
   2. Use the five saved exact inner contexts (640 train / 160 validation).
      No full-800 query bank or other outer fold's fitted model is reused.
   3. Fit edits and probes on the 640 patients; select excerpts and construct
      LLM proposals from those same patients. Freeze all predictions and
      proposals before opening any inner-validation outcomes for scoring.
2. **Inputs and controls**
   1. Reproduce all saved query activations on training and validation inputs.
      Reuse frozen embeddings and five effect queries in each inner context.
   2. Reuse exact-context TF-IDF nuisance predictions only after verifying every
      prediction's training-row registry excludes the predicted patient, the
      current inner-validation patients, and every outer-test patient.
   3. These nuisances are honest but differ from the original neural-query
      optimizer's unpersisted nuisance fits. Recomputed baseline moments are
      pilot diagnostics, not claimed reproductions of the original fit scores.
      All patients are retained; nuisance predictions are fixed across edits.
   4. Target the effect moment; report treatment/outcome changes secondarily.
      Compare radii 0.02, 0.05, 0.10, 0.20 with 3 training-only restarts and
      50 Adam steps. Use a fixed baseline row-score scale, distance penalty,
      exact unit-vector distance constraint, and activation SD between 0.5 and
      2 times baseline. Save five seeded random controls at each edit's actual
      distance. No radius is selected from validation performance.
3. **Predictive probes**
   1. Fit linear and additive spline probes, with fixed regularization, both
      for a query alone and for the full 15-query bank with that query replaced.
      Include constant and remaining-bank-without-query reference models.
   2. Evaluate the original fitted probe applied to changed activations, and
      separately a probe refitted to changed training activations. Use log loss
      for treatment/outcome and fixed-nuisance R-loss for the effect model.
   3. These probe families cannot establish removal of all nonlinear or
      interaction information. Seeds/queries are not independent cohorts.
4. **LLM interpretation**
   1. Use `gemma4-31b` at `http://sn4622130540:8000/v1`, with reasoning enabled.
      Use the existing validated JSON request/retry machinery and save prompts,
      repairs, responses, model identity, and request-level completion hashes.
   2. Fix the interpretation radius at 0.20 before validation. For every query,
      independently request up to two candidate definitions in three conditions:
      original retrieval, targeted-edit contrast, and random-edit contrast
      (prespecified random control 0). There are 75 requests across five folds.
   3. Each condition receives at most six training patients and four complete
      excerpts per patient. Report actual excerpt counts and prompt lengths;
      duplicate excerpts are combined, so exact token counts need not match.
   4. Original retrieval selects the highest-activation patients. Contrasts
      select patients with the largest absolute change in their contribution
      to the training moment, then show original/edited peak and largest
      similarity increase/decrease. Patient treatment/outcome labels are omitted.
   5. Require literal evidence citations, temporal eligibility, measurement
      definitions, and uncertainty. No oracle labels or existing full-800
      candidate catalog is provided to the LLM. No cross-fold synthesis feeds
      back into fitting or validation.
5. **Reporting and limits**
   1. Report the signal-distance curve, random-control comparisons, frozen vs
      refitted probe losses, concrete evidence changes, and candidate examples.
   2. This is an exploratory query-edit and candidate-proposal pilot. Proposed
      variables are not yet extracted and tested in a new Stage 2 estimator.
      There is no outer-test CATE result or claim of improved oracle recovery.
   3. Additional residual discovery, concept-span deletion/rescue, and a complete
      downstream feature-extraction comparison are follow-on experiments.
