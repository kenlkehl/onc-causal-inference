# Modifier-selection sample-size calculator

The calculator estimates, by simulation, how many patients a binary-outcome
study needs to **discover its effect modifiers**. You describe the study: how
many confounders, effect modifiers, dual-role variables, and noise variables
there are, how strong each is, the untreated outcome risk, and the treated
proportion. You also define success, for example finding at least 80% of the
true modifiers while selecting no more than 5% of the other variables. The
calculator then simulates repeated studies at each sample size you list, runs
a modifier selector on each, and reports how often the study succeeds. The
selector is a lasso for the effect scale or Stage 2's own orthogonal modifier
tests.

It runs on your machine as a local web dashboard or from the command line. Its
code is in `sample_size_calculator/`: a standard-library Python front end
(`app.py`, `population.py`, `index.html`, `app.js`), an R simulation engine
(`engine.R`, `run.R`, and the lasso selectors in `binary_selection_methods.R`),
and `orthogonal_tests.py`, which runs Stage 2's tests. It needs no GPU, LLM, or
clinical data, and does not change the OCI workflow. Only the orthogonal-tests
selector imports `oci`.

## Contents

- [How it relates to OCI](#how-it-relates-to-oci)
- [Requirements](#requirements)
- [Running it](#running-it)
- [Inputs](#inputs)
- [The population draw](#the-population-draw)
- [Data-generating mechanism](#data-generating-mechanism)
- [Selectors and nuisances](#selectors-and-nuisances)
- [Success and the reported probabilities](#success-and-the-reported-probabilities)
- [From pilot to a sample size](#from-pilot-to-a-sample-size)
- [Pause, resume, and cancel](#pause-resume-and-cancel)
- [Saved runs](#saved-runs)
- [Configuration file reference](#configuration-file-reference)
- [Legacy configurations](#legacy-configurations)
- [Run directory](#run-directory)
- [Checks](#checks)
- [Limitations](#limitations)

## How it relates to OCI

The calculator answers a design question that comes before an OCI analysis:
with this many patients, could the modifiers be found at all if they were
measured without error? Its answer is a planning guide, not a prediction of
OCI's results:

- **Selectors.** *Stage 2 orthogonal tests* runs the code of Stage 2's default
  modifier selection (`orthogonal_tests`: per-candidate R-learner projection,
  AIPW projection, and outcome-interaction tests, combined by a Cauchy
  combination, with Benjamini–Hochberg control; see
  [Stage 2 selection](stage2_selection.md)) with Stage 2's default settings.
  The two lasso selectors, a penalized [DINA](glossary.md#dina) fit (log odds
  ratio scale) and Zhao's residual-on-residual lasso (risk-difference scale),
  are alternatives that can be more or less powerful than Stage 2's tests in a
  given design. All three are described [below](#selectors-and-nuisances).
- **Error criterion.** The calculator's false positive rate is the fraction of
  true non-modifiers selected. Benjamini–Hochberg controls the false discovery
  rate, the fraction of selected variables that are not modifiers.
- **Effect scale of the truth.** A true modifier changes the effect on the
  simulated scale. Stage 2's tests detect modification on either scale: when
  the treatment has an effect, a variable that shifts the untreated risk
  changes the risk difference under a log odds ratio model, and the odds ratio
  under a risk-difference model. The calculator counts such selections as
  false positives and reports them separately as prognostic false selections
  ([Success](#success-and-the-reported-probabilities)).
- **Measured, not extracted, variables.** Every simulated covariate is
  observed exactly. In OCI, candidates come from Stage 1 text evidence and are
  measured by an extraction LLM, so some modifiers are never proposed and the
  rest carry measurement error. Both raise the sample size needed.
- **Simpler nuisances.** Every selector uses the calculator's propensity and
  outcome models, cross-fitted logistic lassos (or the true probabilities);
  Stage 2 tests with its final-estimation nuisances, the super learner by
  default.

The variable counts correspond roughly to the candidate variables Stage 2
tests after its informativeness filter. To measure OCI's own end-to-end
behavior at fixed sample sizes on synthetic text cohorts, use the
[benchmarks](benchmarks.md) (the baseline scenario S0 runs at 1,000, 5,000, and
10,000 patients) and `oci-evaluate-attribution --oracle`, which reports which
true variables the selected variables recover
([Stage 1 architectures](stage1_architectures.md#attribution-and-ablation)).

## Requirements

- Python 3.9 or later. The lasso selectors use only the standard library, so
  they need no OCI environment (`uv sync`). The orthogonal-tests selector runs
  Stage 2's code, so it needs the OCI environment (Python 3.12 or 3.13, `uv
  sync --frozen`; see [Installation](../README.md#installation)) and the
  calculator started with that interpreter:

  ```bash
  PYTHON=.venv/bin/python ./run_sample_size_calculator.sh
  ```

  Without it, a run with that selector is refused with this instruction.
- R with `Rscript` on `PATH`, and the R packages `glmnet` and `jsonlite`:

  ```bash
  Rscript -e 'install.packages(c("glmnet", "jsonlite"), repos="https://cloud.r-project.org")'
  ```

  On Linux, CRAN builds `glmnet` and its dependency `RcppEigen` from source,
  which needs a Fortran compiler and the LAPACK development files (on
  Debian/Ubuntu, `r-base-dev`, or `gfortran` and `liblapack-dev`). Without
  them, install prebuilt binaries from Posit Package Manager instead
  (replace `noble` with your Ubuntu release codename):

  ```bash
  Rscript -e 'options(HTTPUserAgent=sprintf("R/%s R (%s)", getRversion(), paste(getRversion(), R.version["platform"], R.version["arch"], R.version["os"]))); install.packages(c("glmnet", "jsonlite"), repos="https://packagemanager.posit.co/cran/__linux__/noble/latest")'
  ```

The launcher checks for `Rscript` and both packages before it starts and
prints the install command if any is missing.

## Running it

From the repository root:

```bash
./run_sample_size_calculator.sh                  # dashboard at http://127.0.0.1:8765
./run_sample_size_calculator.sh --port 8766      # if 8765 is in use
./run_sample_size_calculator.sh --runs-dir DIR   # save dashboard runs under DIR
```

Open the printed address in a browser. The server listens on the loopback
interface only. On a remote machine, forward the port and open
`http://127.0.0.1:8765` locally:

```bash
ssh -L 8765:127.0.0.1:8765 user@remote-host
```

Dashboard runs are saved under `artifacts/sample_size_calculator/` (ignored
by Git), one timestamped directory per run, or under `--runs-dir`. The
dashboard runs one simulation at a time. Stop the server with Ctrl-C after
any active run finishes. A browser refresh does not reload the Python server;
restart it after updating the code.

One run from a configuration file, into a new directory:

```bash
./run_sample_size_calculator.sh \
  --config sample_size_calculator/configs/pilot-v2.json \
  --output /tmp/calculator-pilot
```

The command prints the final progress record and the run directory, and exits
with status 0 only if the run completed. The dashboard's **Save inputs**
button downloads the current inputs, including the drawn population, as a
configuration file for `--config`. `--output` must not exist. Separate
command-line runs can execute in parallel in different output directories.
`./run_sample_size_calculator.sh` runs `python3 -m sample_size_calculator`
with the repository root on `PYTHONPATH`; set `PYTHON` to use another
interpreter, such as the OCI environment's for the orthogonal-tests selector.

**Run time.** Each simulated study fits the nuisance models and one selector.
With the dashboard defaults (110 variables, estimated nuisances), a pilot of 8
studies at each of 400 and 800 patients takes about a minute on one CPU core;
time grows with the sample size, the number of encoded columns, and the
number of repetitions. A planning grid of 200 repetitions at several sizes
can take hours. Oracle nuisances are much faster. The orthogonal-tests
selector costs about as much as the lasso
([below](#selectors-and-nuisances)).

## Inputs

**Effect scale.** *Log odds ratio* or *Risk difference*: the scale on which
the simulated modifiers act.

**Selection method.** *Lasso for the scale* selects with the penalized DINA
fit (Gao) on the log odds ratio scale and with Zhao's lasso on the
risk-difference scale. *Stage 2 orthogonal tests* selects with Stage 2's tests
on either scale; its Benjamini–Hochberg level is *BH q for orthogonal tests*
under the advanced inputs.

**Data generating variables.** A table of counts by role (confounder only,
modifier only, both roles, noise) and type (continuous, binary, categorical).
Every variable is a candidate for modifier selection, and every encoded column
enters the nuisance models. A categorical variable counts once although it is
encoded as several indicator columns.

**Modifier strength.** On the log odds ratio scale, an odds ratio: the ratio
of treatment odds ratios per unit contrast of the covariate (not the treatment
odds ratio itself); the field below shows the log odds ratio. On the
risk-difference scale, a risk difference per unit contrast.

**Confounding strength.** Separate odds ratios for each confounder's
association with treatment and with the untreated outcome, with their log
odds ratios.

**Untreated outcome risk.** The population-average probability of the outcome
without treatment.

**Sensitivity for modifier detection.** The minimum fraction of true modifiers
a study must select.

**False positive rate.** The maximum fraction of true non-modifiers a study
may select; specificity is one minus this rate. This is not the false
discovery rate.

**Desired probability of meeting both.** The target proportion of repeated
studies that meet both detection requirements.

**Advanced inputs.** The treated proportion; the *intercept of the effect*
(the effect at zero or reference covariate values, on the selected scale); the
binary-prevalence and category-probability distributions; the orthogonal
tests' BH q; estimated or oracle nuisances; the sample-size grid; the run stage (pilot or planning grid); the
repetitions per size; and the simulation seed.

## The population draw

Under **Per-variable strengths and prevalences**, three Normal distributions,
each with a mean and a standard deviation, generate the per-variable
coefficients: the modifier effects, the confounders' association with
treatment, and their association with the untreated outcome. The main
strength inputs and the linked mean fields set the means; the standard
deviations are set here. For log odds ratios, the odds ratio shown in the main
inputs is `exp(mean log OR)`, which is the median (geometric mean) odds ratio
when the standard deviation is positive, not the arithmetic mean. For
risk-difference modifier effects, the mean and the standard deviation are
risk differences. A standard deviation of zero gives every variable the same
coefficient; signs are not forced to alternate, and an odds ratio below 1 is a
negative coefficient.

Binary prevalences are drawn per variable from Beta(α, β) (mean α/(α+β)) or
fixed at one shared value. Category probabilities are drawn per variable from
a Dirichlet distribution whose parameters are the mean probability vector
times the concentration (higher concentration, less variation), or fixed at
the mean vector.

**Draw / reset population** draws every variable's coefficients and
prevalences with the population draw seed. Select a variable under *Variable
to inspect or edit* to change its coefficients, prevalence, or category
probabilities; a modifier whose coefficient is set to zero is no longer a true
modifier. The realized rows, edits included, are saved in the run's
`config.json` and held fixed across sample sizes and repetitions, so the
reported probability is conditional on this population. It is not averaged
over random populations. Changing a distribution or a variable count marks the
draw stale, and the dashboard refuses to run until you draw again. Drawing
again with the same settings and seed reproduces the draw and discards manual
edits; a new seed gives a new population.

Probabilities must lie strictly between 0 and 1 and are not clipped to a
floor, so extreme draws can leave a category empty or make a fit unstable;
those studies are recorded as they occur. A coefficient that is not finite or
lies outside [−10, 10] is rejected with an error rather than clipped. This is
an input bound, not a truncated-Normal sampler.

## Data-generating mechanism

All covariates are independent. A continuous covariate is
Uniform(−√3, √3), with standard deviation 1. For variable `g`, `h_g` is its
continuous value, its binary indicator, or, for a categorical variable with
`K` levels, the ordered score 0, 1/(K−1), …, 1. Fitting uses the `K − 1`
indicator columns, with the first level as reference. With the realized
coefficients:

```text
e(X)   = expit(alpha0 + sum_g alpha_g h_g(X))
mu0(X) = expit(gamma0 + sum_g gamma_g h_g(X))
tau(X) = intercept_of_effect + sum_g beta_g h_g(X)
log OR scale: mu1(X) = expit(logit(mu0(X)) + tau(X))
RD scale:     mu1(X) = mu0(X) + tau(X)
A ~ Bernoulli(e(X)),  Y ~ Bernoulli(mu_A(X))
```

Confounder-only variables have `beta = 0`; modifier-only variables have
`alpha = gamma = 0`; noise variables have all three at zero; dual-role
variables enter all three equations. A true modifier is a variable whose
realized `beta` is not zero. The intercepts `alpha0` and `gamma0` are solved
once, on 30,000 covariate draws with the calibration seed, so that the
average propensity equals the treated proportion and the average untreated
risk equals the requested risk. At fixed coefficients, adding confounders
increases their combined contribution.

On the risk-difference scale, a conservative check over the bounded
covariate support rejects any configuration whose treated-arm probabilities
could leave (0, 1); probabilities are never clipped.

Study `r` at the `i`-th smallest sample size uses seed
`seed + 1000 (2500 (i − 1) + r − 1)`: studies are 1,000 seeds apart (a study
uses its seed and the next 205 for covariates, outcomes, nuisance folds, and
multipliers) in blocks of 2,500, the replication limit, per sample size. No
two studies of a run share a seed, adding repetitions leaves the earlier
studies unchanged, and the calibration seed must lie outside
`[seed, seed + 2,500,000 × number of sizes)`.

## Selectors and nuisances

`engine.R` simulates the data and calls the lasso selectors `select_gao` and
`select_zhao` in `binary_selection_methods.R`, or sends the study to the
orthogonal-tests worker.

**Nuisances.** With *Estimated* nuisances, the propensity `e` and the arm
outcome probabilities `mu0` and `mu1` are two-fold cross-fitted logistic
lassos (`cv.glmnet`, five-fold deviance, `lambda.min`) on all encoded
columns, clipped to [0.01, 0.99]. On the risk-difference scale the true arm
probabilities need not be logistic-linear, so these models can be
misspecified. *Oracle* nuisances use the true probabilities, a diagnostic
benchmark that isolates the selector. From them, the DINA quantities are
`V_a = mu_a (1 − mu_a)`, `a = e V1 / (e V1 + (1 − e) V0)`, and
`nu = a logit(mu1) + (1 − a) logit(mu0)`, and the risk-difference quantities
are `m = (1 − e) mu0 + e mu1` and the residuals `A − e` and `Y − m`.

**Gao (log odds ratio).** A binomial lasso of `Y` on `(A − a)` and the
centered, scaled covariates multiplied by `(A − a)`, with offset `nu`, the
[DINA](glossary.md#dina) working model of Gao and Hastie. The `(A − a)` term,
the effect intercept, is unpenalized.

**Zhao (risk difference).** A lasso of the outcome residual `Y − m` on the
treatment residual `A − e` and the centered, scaled covariates multiplied by
it, with the `A − e` term unpenalized: the residual-on-residual loss of Zhao,
Small, and Ertefaie (*Selective inference for effect modification via the
lasso*, JRSS-B 2022). The residual scale is estimated from the full
unpenalized least-squares fit, which needs the smallest sample size to
exceed the number of encoded columns by at least 31.

Both selectors set the penalty from 500 Gaussian multiplier draws of the
score: `lambda` is 1.1 times the mean over the draws of the maximum absolute
score. They fit by coordinate descent, check the KKT conditions, and select
the coefficients with `|beta| > 1e−7`. A variable is selected if any of its
encoded columns is. Constant columns are dropped from selection, but their
variables stay in the sensitivity denominator.

**Stage 2 orthogonal tests (either scale).** `orthogonal_tests.py` calls
`candidate_tests` and `combine_and_select` in
`oci/inference/stage2_modifier_tests.py`, the functions behind Stage 2's
`orthogonal_tests` rule, with Stage 2's default settings:

- Each variable is one candidate: a continuous variable's value, or a binary
  or categorical variable's levels (the first level is the reference), tested
  jointly.
- The R-learner projection and AIPW projection tests (risk-difference scale;
  the propensity is clipped to [0.02, 0.98] in the AIPW pseudo-outcome) and
  the outcome-interaction likelihood-ratio test (odds ratio scale), combined
  by a Cauchy combination with weights 0.25, 0.25, and 0.5.
- Benjamini–Hochberg across all variables at `modifier_q` (default 0.1, Stage
  2's `modifier_selection.q`), with at most 64 selections (`max_modifiers`).
  Every patient is eligible, and an indicator column with fewer than 5
  patients on either side is dropped.

The selected variables are those Benjamini–Hochberg selects. The tests use the
same nuisances as the lasso selectors. `app.py` starts one worker per run with
the calculator's Python interpreter; the worker loads OCI once and R sends it
each study over a loopback socket. Starting the worker takes a few seconds,
and with the default 110 variables the tests take about a second per study. Each study's `modifier_tests.json` lists the selected
variables and, for every variable, the tested columns, the component and
combined p- and q-values, and the labels (`absolute`, `multiplicative`, or
`combined_only`) that Stage 2 reports.

## Success and the reported probabilities

A simulated study **succeeds** when its sensitivity (selected true modifiers
over true modifiers) reaches the target and its false positive rate (selected
non-modifiers over non-modifiers) does not exceed the allowance. If a
denominator is zero, that metric is undefined and its requirement holds
vacuously. A study whose fit fails counts as unsuccessful; the mean selected
counts, sensitivity, and specificity exclude failed studies.

For each sample size, `result.json` and the dashboard report the probability
of success with a 95% Wilson interval; the number of fit failures; the mean
numbers of true and false variables selected; the mean sensitivity,
specificity, and false positive rate; the probabilities of selecting exactly
the true set, of recovering every true modifier, and of selecting anything;
each true modifier's selection frequency; and the realized event rate,
treated rate, overlap fraction (`0.1 ≤ e ≤ 0.9`), propensity Brier score,
and outcome-probability mean squared error.

They also report the mean number of **prognostic false selections**
(`mean_false_selected_prognostic`): selected non-modifiers with a nonzero
untreated-outcome coefficient, that is, confounders. Whenever the treatment
has an effect, these variables modify it on the scale that was not simulated,
so a selector that tests both scales, such as Stage 2's, can select them by
its own standard. They still count as false positives in the success
criterion; a large count means the success probability understates detection
of modification on either scale.

The decision bound is a
simultaneous one-sided 95% exact (Clopper–Pearson) lower bound, with the 5%
split equally across the sample sizes.

## From pilot to a sample size

1. **Pilot** (*Small pilot*, at most 30 repetitions per size). Checks the
   mechanism, run time, and fitting; it never proposes a sample size.
2. **Planning grid** (*Planning grid*, at least 200 repetitions per size).
   Fix the assumptions, the selector, and the success definition first. The
   provisional size (`candidate_n`) is the smallest evaluated `n` such that
   it and every larger evaluated size meet the target with their lower bound
   and have no fit failures. There is no interpolation or extrapolation
   between or beyond the grid.
3. **Confirmation.** Before calling a candidate a recommendation, run a
   separate, frozen confirmation at that size with fresh simulation seeds;
   its one-sided 95% lower bound must meet the target. The calculator does
   not automate this step, and `recommended_n` is always null.

No grid starts automatically. At 200 repetitions, the bound may still be too
imprecise near the target.

## Pause, resume, and cancel

While the dashboard monitors a live run, it shows **Pause**, **Resume**, and
**Cancel run**. The controls take effect at the next study boundary, so a
long nuisance fit can delay them. A paused run keeps its process; keep the
server running, and **Resume** continues with the same seeds and records.
Pausing does not survive a server restart. **Cancel run** keeps the completed
studies and the source and configuration files but writes no summary or
sample-size result, and a cancelled run cannot be resumed. Loading saved runs
is disabled while a run is monitored, so the controls always address that
run.

## Saved runs

**Saved runs** lists earlier dashboard runs; **Load result & inputs** shows a
run's result and restores its inputs. **Manage saved runs** removes and
restores runs, one at a time or in bulk. Removal moves a run directory to
`.removed/` inside the runs directory without changing any file, so it frees
no disk space, and links to a removed run work again after it is restored.
Only finished runs (complete, failed, or cancelled) can be moved. There is no
permanent delete: delete directories under `.removed/` yourself.

## Configuration file reference

A configuration file is a JSON object; unknown keys are errors. Set
`"schema_version": 2`: a file without it is read as a
[legacy configuration](#legacy-configurations). Missing keys take the
defaults below. The dashboard writes coefficients as log odds ratios (or risk
differences) even where it displays odds ratios.

| Key | Default | Allowed values and meaning |
|---|---|---|
| `schema_version` | `2` (dashboard); `1` if omitted | `2`, or `1` for legacy behavior |
| `variables` | 5 continuous confounders; 3 continuous, 1 binary, 1 categorical modifier; 100 continuous noise | Counts (0–200) by role `confounder`, `modifier`, `both`, `noise` and type `continuous`, `binary`, `categorical`; 2–220 variables and at most 300 encoded columns in total |
| `scale` | `"logOR"` | `"logOR"` or `"RD"`: the scale of the simulated effects |
| `selector` | `"lasso"` | `"lasso"` (Gao with `"logOR"`, Zhao with `"RD"`) or `"orthogonal_tests"` (Stage 2's tests; needs the OCI environment) |
| `modifier_q` | `0.1` | Benjamini–Hochberg level of the orthogonal tests, 0.001 to 1 |
| `modifier_strength` | `0.6` | Mean modifier coefficient, −5 to 5 (log odds ratio, or risk difference with `"RD"`) |
| `modifier_sd` | `0` | Standard deviation of the modifier coefficients, 0 to 2 |
| `treatment_coefficient`, `treatment_sd` | `0.5`, `0` | Mean (−5 to 5) and standard deviation (0 to 2) of the confounders' log odds ratios with treatment |
| `outcome_coefficient`, `outcome_sd` | `0.5`, `0` | The same for the untreated outcome |
| `constant_effect` | `0.2` | Intercept of the effect, −2 to 2, on the selected scale |
| `baseline_risk` | `0.3` | Untreated outcome risk, 0.05 to 0.95 |
| `treated_fraction` | `0.5` | Treated proportion, 0.1 to 0.9 |
| `binary_mode` | `"beta"` | `"beta"` (one prevalence per variable from Beta(`binary_alpha`, `binary_beta`), each 0.1 to 1000, default 5) or `"fixed"` (`binary_prevalence`, 0.05 to 0.95, default 0.5) |
| `category_mode` | `"dirichlet"` | `"dirichlet"` (concentration `category_concentration`, 0.5 to 1000, default 20) or `"fixed"` |
| `category_probs` | `[0.5, 0.3, 0.2]` | Mean category probabilities, reference first: 2–5 values, each at least 0.05, summing to 1 |
| `population_seed` | `920000` | Seed of the population draw |
| `population` | `null` | The realized rows (`group`, `role`, `kind`, `alpha`, `gamma`, `beta`, `prevalence`, `probabilities`), or `null` to draw them from the distributions when the run is created |
| `sensitivity` | `0.8` | Required sensitivity, 0 to 1 |
| `false_positive_rate` | `0.05` | Maximum false positive rate, 0 to 1 |
| `target_probability` | `0.8` | Desired probability of success, 0.5 to 0.99 |
| `nuisance` | `"estimated"` | `"estimated"` or `"oracle"` |
| `sample_sizes` | `[400, 800]` | 1–8 distinct integers from 200 to 20,000; the smallest must exceed the number of encoded columns by at least 31 |
| `mode` | `"pilot"` | `"pilot"` (`replications` at most 30) or `"grid"` (`replications` at least 200) |
| `replications` | `8` | Studies per sample size, 1 to 2,500 |
| `seed` | `910100` | Simulation seed, 1 to 10,000,000; the studies use seeds from `seed` up to `seed + 2,500,000 × number of sizes` ([mechanism](#data-generating-mechanism)) |
| `calibration_seed` | `910000` | Seed of the 30,000-draw intercept calibration, outside the studies' seeds |
| `correlation` | `0` | Must be 0 with schema version 2 |
| `goal`, `recovery_fraction`, `false_allowance` | `"all"`, `0.8`, `1` | Legacy success definition; ignored with schema version 2 |

Example configurations are in `sample_size_calculator/configs/`:
`pilot-v2.json` (schema version 2, with an explicit population, oracle
nuisances, two studies at 400 patients) and the legacy `pilot.json` (the
dashboard defaults with estimated nuisances), `pilot-null.json` (no
modifiers), and `pilot-rd.json` (risk-difference scale).

## Legacy configurations

A configuration without `schema_version`, or with version 1, keeps the
earlier mechanism: every confounder has the coefficients
`treatment_coefficient` and `outcome_coefficient`, the modifier coefficients
alternate in sign with magnitude `modifier_strength`, binary prevalence and
category probabilities are shared, latent covariates can be correlated
(`correlation`, 0 to 0.8), and success means recovering all true modifiers
(`goal: "all"`) or at least `recovery_fraction` of them, with at most
`false_allowance` false selections. Version 1 configurations cannot include
`population` rows. When the dashboard loads a legacy run, it converts its
coefficients, alternating signs included, and its thresholds to editable
version 2 inputs, and warns that new runs use independent covariates; saved
legacy results stay labeled as legacy and are never rewritten.

## Run directory

```text
RUN/
├── config.json           # validated configuration, with the realized population rows
├── manifest.json         # start and finish times, configuration, Python version, source and output hashes, state
├── progress.json         # state (running, paused, complete, failed, cancelled) and counts
├── control.json          # requested action (running, paused, cancelled)
├── sources/              # copies of the code that ran (app.py, population.py, orthogonal_tests.py,
│                         #   engine.R, run.R, binary_selection_methods.R, index.html, app.js);
│                         #   R and the orthogonal-tests worker run these copies
├── execution.log         # R output, one line per study
├── groups.csv            # variables with their roles and types
├── calibration.json      # solved intercepts and the coefficients used
├── R_runtime.json        # R, glmnet, and jsonlite versions
├── replications.json     # one record per study: selections, metrics, errors, warnings
├── result.json           # per-size summary and candidate_n (complete runs only)
├── orthogonal_tests_worker.json  # orthogonal tests: settings, Python version, OCI source hashes
├── orthogonal_tests.log  # orthogonal tests: the worker's messages
└── n<n>_r<rep>/          # per study: nuisance/ (estimated nuisances: predictions, screening,
                          #   coefficients, lineage); coefficients.csv and fit.json (lasso) or
                          #   modifier_tests.json (orthogonal tests); the first study at each n
                          #   also keeps data.csv, derived_nuisances.csv, and groups.csv
```

The orthogonal tests run the OCI code installed in the interpreter, not a
copy; `orthogonal_tests_worker.json` records the paths and SHA-256 hashes of
the Stage 2 modules it loaded.

The first study at each sample size keeps the complete generated data and
nuisances, so it can be refitted independently.

## Checks

```bash
pytest -q tests/test_sample_size_calculator.py
# or, without pytest
python3 -m unittest tests.test_sample_size_calculator
```

The tests cover input validation, the population draws and their moments, run
creation and hashes, run management and controls, and the dashboard's
endpoints. With R and both packages installed, they also run the R engine
checks (`tests/test_sample_size_calculator_engine.R`: calibration, the
log odds ratio and risk-difference identities, the role structure, the
grid decision, the success definition, and unique study seeds) and one
end-to-end pilot through `run_sample_size_calculator.sh`; otherwise those two
tests are skipped. In the OCI environment they also check that the
orthogonal-tests settings equal Stage 2's defaults, the worker's encoding and
selection on a constructed study, and an end-to-end pilot with that selector;
otherwise those tests are skipped.

## Limitations

- The data-generating process is generic and tabular: independent covariates,
  linear effects on the chosen scale, and exact measurement. It does not
  reproduce OCI's synthetic text generator or any saved cohort.
- The probabilities are conditional on one population draw, edits included.
- A grid result is a provisional candidate, and confirmation is manual.
- Only binary outcomes are supported.
