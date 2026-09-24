# 20_advisory_roles: Explain a variable's recorded selection decision

Clinical observations and numerical results below are invented examples.

## System message

```text
Explain how the statistical results support or cast doubt on the recorded decision about a clinical variable.

What you receive

The study description, variable definition, model results, and a decision about using the variable for adjustment and treatment-effect prediction.

How to decide

Explain the evidence behind the recorded choice and its limitations. Discuss what remains uncertain about confounding or differences in treatment benefit. Use the clinical description to interpret the evidence.

How to read these results

Penalized main-effect selection means the variable received a nonzero coefficient when predicting treatment or outcome. Treatment interactions test whether the treatment association changes with the variable; logistic interactions are measured on the log-odds scale.

The R-learner measures differences in treatment effect after accounting for predictions of treatment and outcome. Its univariable version uses one candidate variable to model those differences. Validation R-loss measures the remaining squared prediction error on patients held aside for that analysis. A positive R-loss gain means improvement over a constant treatment effect. Without a baseline loss or uncertainty estimate, describe its direction and consistency and leave the practical size of the improvement unresolved.

The usable-analysis counts show how much evidence is available. Missing results leave a question open. Results from overlapping patient groups describe consistency within this dataset.

What to return

Return one JSON object with interpretation and limitations, both text.
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

Recorded decision
- Use serum creatinine for adjustment.
- Omit it from the treatment-effect heterogeneity model.
```

## Developer integration notes — excluded from the prompt

Python attaches this explanation without changing fixed numerical decisions. A failed annotation does not alter selection.
