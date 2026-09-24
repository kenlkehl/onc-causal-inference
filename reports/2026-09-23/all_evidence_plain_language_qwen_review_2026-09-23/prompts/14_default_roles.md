# 14_default_roles: Assess a clinical variable's confounder and modifier roles

Clinical observations and numerical results below are invented examples.

## System message

```text
Decide whether one clinical variable should be used to adjust for confounding, to predict differences in treatment benefit between patients, or both.

What you receive

The treatment comparison, outcome, clinical variable definition, and results from several statistical analyses. A confounder can influence treatment choice and outcome. An effect modifier helps identify patients with different effects of treatment A compared with B.

How to decide

For confounding, consider whether the variable could influence both treatment choice and outcome, using the study description and the treatment/outcome results. Retain a credible common cause when the evidence is incomplete but supports that concern.

For effect modification, emphasize validation evidence for differences in treatment benefit. Consider agreement and disagreement across analyses, the amount of usable evidence, and whether the result concerns probability differences or log odds. Main-effect prediction alone leaves the modifier question open. Use the clinical description to interpret the results and state material uncertainty.

Choose the confidence labels qualitatively, considering consistency, the size of validation gains, and available uncertainty estimates. Explain that judgment in the rationale. When a clinical relationship is unknown and the relevant results are unavailable, use uncertain.

Use supported when relevant evidence agrees and supports the role; plausible when the role is credible with partial or noisy support; uncertain when missing information or unresolved disagreement prevents a judgment; and not_supported when the available evidence does not justify retaining that role in this analysis.

Judge stability separately for each role. consistent means the relevant results broadly agree; mixed means meaningful disagreement; insufficient means too little usable evidence to judge consistency.

How to read these results

Penalized main-effect selection means the variable received a nonzero coefficient when predicting treatment or outcome. Treatment interactions test whether the treatment association changes with the variable; logistic interactions are measured on the log-odds scale.

The R-learner measures differences in treatment effect after accounting for predictions of treatment and outcome. Its univariable version uses one candidate variable to model those differences. Validation R-loss measures the remaining squared prediction error on patients held aside for that analysis. A positive R-loss gain means improvement over a constant treatment effect. Without a baseline loss or uncertainty estimate, describe its direction and consistency and leave the practical size of the improvement unresolved.

The usable-analysis counts show how much evidence is available. Missing results leave a question open. Results from overlapping patient groups describe consistency within this dataset.

What to return

Return one JSON object with exactly confounder and effect_modifier. Each contains assign (boolean), assessment (supported, plausible, uncertain, or not_supported), stability (consistent, mixed, or insufficient), rationale (text), and evidence_comments (an array of short statements naming the relevant methods and findings). assign can be true for supported or plausible. Use an empty evidence_comments array when evidence is unavailable.
```

## User message

```text
Study
- Observational study of patients who received treatment A or treatment B.
- Compare treatment A with treatment B at treatment initiation.
- Outcome: a favorable event within one year, recorded as yes or no.
- Treatment effect: the difference in the probability of that event under A versus B.
- The variables describe the patient's health before treatment.

The analyses use three different, overlapping groups from the same training dataset. Effect analyses include patients with estimated probability of receiving A between 0.1 and 0.9.

Clinical background: In this illustrative study, renal function can influence treatment choice and outcome.

Clinical variable: Serum creatinine
Definition: Latest pretreatment serum creatinine in mg/dL.

Model results
- penalized main effects — treatment: 2 of 3 analyses met the criterion “nonzero coefficient group”.
- penalized main effects — outcome: 3 of 3 analyses met the criterion “nonzero coefficient group”.
- penalized treatment interactions — effect heterogeneity: 1 of 3 analyses met the criterion “nonzero interaction group”.
- univariable R-learner — effect heterogeneity: 0 of 3 analyses met the criterion “validation R-loss gain > 0”.
  Validation R-loss gains: -0.002, 0.0, -0.001.
```

## Developer integration notes — excluded from the prompt

Python supplies genuine study context and coherent evidence availability, attaches identity/provenance, preserves investigator-locked roles, and validates assignments. Qualitative assessments require evaluation before adoption.
