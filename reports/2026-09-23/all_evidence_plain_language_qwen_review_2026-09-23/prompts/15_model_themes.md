# 15_model_themes: Group variables into clinical themes

Clinical observations and numerical results below are invented examples.

## System message

```text
Group the supplied clinical variables into themes that help explain the pattern of statistical results.

What you receive

Clinical variable descriptions and their results from several statistical analyses. A theme brings related measurements together, such as several measures of kidney function.

How to decide

Choose clinically coherent groups and explain how the results agree or differ. Preserve distinct measurements within a group. Give an uncertain or unrelated variable its own theme. Include each variable in one theme, including those with weak or unavailable results. Use as many themes as the clinical meanings require.

How to read these results

Penalized main-effect selection means the variable received a nonzero coefficient when predicting treatment or outcome. Treatment interactions test whether the treatment association changes with the variable; logistic interactions are measured on the log-odds scale.

The R-learner measures differences in treatment effect after accounting for predictions of treatment and outcome. Its univariable version uses one candidate variable to model those differences. Validation R-loss measures the remaining squared prediction error on patients held aside for that analysis. A positive R-loss gain means improvement over a constant treatment effect. Without a baseline loss or uncertainty estimate, describe its direction and consistency and leave the practical size of the improvement unresolved.

The usable-analysis counts show how much evidence is available. Missing results leave a question open. Results from overlapping patient groups describe consistency within this dataset.

A univariable interaction screen tests one treatment-by-variable term in an outcome model; its p-value describes that association. When adjusted q-values are unavailable, nominal p-values offer limited evidence about a pattern found among many variables. The causal-forest result compares the same fitted forest's validation R-loss before and after shuffling this variable. A positive increase shows how much shuffling worsened that forest's predictions. Correlated measurements can share that predictive information.

What to return

Return one JSON object with themes, an array. Each theme has name (a clinical label), members (existing variable names), interpretation (text), and disagreements (text, or an empty string).
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

Clinical variable: Serum creatinine
Definition: Latest pretreatment serum creatinine in mg/dL.

Model results
- penalized main effects — treatment: 2 of 3 analyses met the criterion “nonzero coefficient group”.
- penalized main effects — outcome: 3 of 3 analyses met the criterion “nonzero coefficient group”.
- penalized treatment interactions — effect heterogeneity: 1 of 3 analyses met the criterion “nonzero interaction group”.
- univariable R-learner — effect heterogeneity: 0 of 3 analyses met the criterion “validation R-loss gain > 0”.
  Validation R-loss gains: -0.002, 0.0, -0.001.
- univariable interaction: 2 of 3 analyses met the criterion “nominal p < 0.05”.
  Interaction p-values: 0.02, 0.04, 0.2.
  Multiplicity-adjusted q-values are unavailable.
- causal forest: 1 of 3 analyses met the criterion “permutation R-loss increase > 0”.
  R-loss increases after permutation: -0.001, 0.001, 0.0.

Clinical variable: Emphysema on imaging
Definition: Explicit pretreatment imaging evidence of emphysema, Present or Absent.

Model results
- penalized main effects — treatment: results unavailable.
- penalized main effects — outcome: 1 of 3 analyses met the criterion “nonzero coefficient group”.
- penalized treatment interactions — effect heterogeneity: 3 of 3 analyses met the criterion “nonzero interaction group”.
- univariable R-learner — effect heterogeneity: 3 of 3 analyses met the criterion “validation R-loss gain > 0”.
  Validation R-loss gains: 0.004, 0.006, 0.003.
- univariable interaction: 2 of 3 analyses met the criterion “nominal p < 0.05”.
  Interaction p-values: 0.03, 0.06, 0.04.
  Multiplicity-adjusted q-values are unavailable.
- causal forest: 3 of 3 analyses met the criterion “permutation R-loss increase > 0”.
  R-loss increases after permutation: 0.003, 0.004, 0.002.
```

## Developer integration notes — excluded from the prompt

Python checks coverage, assigns theme IDs, and attaches member evidence. Theme grouping leaves measurement columns intact.
