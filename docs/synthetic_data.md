# Synthetic data

OCI includes a generator for synthetic oncology cohorts with a known causal
structure, and two bundled 1,000-patient cohorts made with it. Because the true
treatment effect of every synthetic patient is known, these cohorts let you
check whether the workflow recovers the confounders and effect modifiers that
generated the data, and how close its estimated effects are to the truth. Terms
such as [confounder](glossary.md#confounder),
[effect modifier](glossary.md#effect-modifier), and
[oracle columns](glossary.md#oracle-columns) are defined in the
[glossary](glossary.md).

The generated notes depend only on patient characteristics and never mention
the treatment or the outcome, so one corpus can serve many treatment and
outcome equations. [Re-equation](#re-equation) recomputes treatment and outcome
over an existing cohort in Python, without an LLM. Factors that change the text
itself ([generation-time factors](#generation-time-factors)) are chosen when a
corpus is generated.

## How a cohort is generated

The generator (`python -m synthetic_data.cli`) asks an LLM, served through any
OpenAI-compatible chat-completions endpoint, to design a data-generating process
(DGP) for a clinical question and then to write each patient's record. All
probabilities and outcomes are computed in Python from that DGP, so the truth
is exact.

1. **Variables.** The LLM proposes patient characteristics for the clinical
   question. Each is continuous or categorical and carries one or both roles:
   `confounder` or `effect_modifier`. When you request role counts, the
   generator keeps the first proposed variables that satisfy them. With
   `--num-instruments` or `--num-prognostic`, the confounder and total counts
   requested from the LLM grow by that many, and Python then gives the
   `instrument` or `prognostic` role to that many of the confounder-only
   variables, chosen with a seeded random draw (see [Causal roles](#causal-roles)).
2. **Equations.** The LLM writes a treatment equation and an outcome equation
   over the confounders; it never sees instrument or prognostic variables or
   proxies. Continuous terms are standardized (`(x − mean) / sd`), and
   categorical terms are 0/1 indicators for every level except the first listed
   level, which is the reference. The LLM's main coefficients are scaled by
   `main_coefficient_scale` (0.3) and its interactions by
   `interaction_coefficient_scale` (0.1). Python then adds:
   - the treatment main effect, `--treatment-effect`;
   - a treatment × term interaction for every effect-modifier term, with a
     coefficient drawn uniformly from (−1, 1), scaled by 0.1 and by
     `--modifier-strength`;
   - all pairwise confounder interactions in the treatment equation, drawn
     uniformly from (−1, 1) and scaled by 0.1;
   - a main term in the treatment equation for each instrument term, and in the
     outcome equation for each prognostic term and, with
     `--modifier-main-effects`, each effect-modifier-only term, each drawn
     uniformly from (−1, 1) and scaled by `main_coefficient_scale`.

   `--nonlinearity` reshapes the standardized value `z` of every continuous
   variable in the outcome main terms and the treatment interactions:
   `threshold` uses `1[z > 0]` and `quadratic` uses `z²`. The treatment equation
   and the pairwise interactions stay linear.
3. **Distributions.** The LLM supplies category proportions or a mean and
   standard deviation for each variable.
4. **Scaling.** Coefficients are rescaled so that the treatment linear predictor
   has standard deviation `--treatment-logit-std` and the outcome linear
   predictor `--outcome-logit-std` (both default to `--target-logit-std`; the
   outcome uses the larger of its standard deviations under the two
   treatments), estimated from 10,000 Monte Carlo draws. The treatment main
   effect is not rescaled.
5. **Calibration.** Intercepts are solved numerically so that the mean
   treatment probability equals `--target-treatment-rate` and the mean outcome
   probability under the comparator equals `--target-control-outcome-rate`.

For each patient, Python samples the characteristics (categorical levels from
the proportions; continuous values from a moment-matched log-normal when the
mean is positive, otherwise a normal distribution), computes the propensity
`e = σ(treatment predictor)`, and draws the treatment. A binary outcome is drawn
as Bernoulli(σ(outcome predictor)); a continuous outcome is the outcome
predictor plus normal noise with standard deviation `--outcome-noise-std`. The
true individual effect is the difference between the two potential outcomes:
`σ(lp₁) − σ(lp₀)` for binary outcomes (a risk difference) and `lp₁ − lp₀` for
continuous outcomes. If no variable has the effect-modifier role, the true
effect is the same for every patient.

**Text.** The LLM writes a timeline of 15 to 30 tagged clinical events that ends
before the treatment decision, given the patient's true characteristics. The
timeline never mentions the treatment or the outcome. Narrative events
(clinical notes, imaging, pathology, and NGS reports) are each expanded into a
full document. With `--structured-data`, encounter, hospitalization, laboratory,
and patient-reported-outcome events are rendered as fixed-format text blocks.
All documents are joined in date order with the note separator to form
`clinical_text`. A configurable fraction of drug mentions is replaced by brand
names or abbreviations (`--drug-perturbation-prob`). Timeline events state the
patient's age rather than a calendar date; only
[post-treatment dated notes](#post-treatment-dated-notes) assign dates.

## Causal roles

A variable's `roles` in `metadata.json` place it in the equations:

| Role | Treatment equation | Outcome equation |
|---|---|---|
| `confounder` | Main term | Main term |
| `effect_modifier` | None | Treatment interaction, plus a main term when modifier main effects are on |
| `confounder` and `effect_modifier` (dual role) | Main term | Main term and treatment interaction |
| `instrument` | Main term | None |
| `prognostic` | None | Main term |
| none | None | None |

A variable without a role (an unassigned variable in a re-equation, or a
[proxy](#near-duplicate-proxies)) enters no equation but still appears in the
patient's text. The design LLM tags only `confounder` and `effect_modifier`;
instrument and prognostic roles are always assigned in Python. With
overlapping role counts, the generator path already expresses dual roles:
`--num-confounders 4 --num-effect-modifiers 3 --num-features 6` gives exactly one
variable with both roles.

## Running the generator

A binary-outcome cohort with one confounder and one effect modifier, including
structured-data text:

```bash
python -m synthetic_data.cli \
  --api-url http://127.0.0.1:8000/v1 \
  --model MODEL \
  --clinical-question "Compare vinorelbine to gemcitabine for advanced/metastatic NSCLC." \
  --dataset-size 1000 \
  --num-confounders 1 \
  --num-effect-modifiers 1 \
  --structured-data \
  --output-dir ./synthetic_output/one_conf_one_mod
```

A continuous-outcome cohort with five confounders and three effect modifiers:

```bash
python -m synthetic_data.cli \
  --api-url http://127.0.0.1:8000/v1 \
  --model MODEL \
  --outcome-type continuous --outcome-noise-std 1.0 \
  --num-confounders 5 --num-effect-modifiers 3 \
  --dataset-size 5000 \
  --output-dir ./synthetic_output/continuous_example
```

A cohort whose notes leave values undocumented more often for lower values, and
that also has post-treatment dated notes:

```bash
python -m synthetic_data.cli \
  --api-url http://127.0.0.1:8000/v1 \
  --model MODEL \
  --num-confounders 5 --num-effect-modifiers 5 --structured-data \
  --documentation-rate 0.6 --documentation-mechanism mnar \
  --post-treatment-rate 0.8 --post-treatment-events toxicity,early_response \
  --dataset-size 1000 \
  --output-dir ./synthetic_output/mnar_post_treatment
```

`MODEL` is the model ID served at the endpoint. Output directories named
`synthetic_output*` are ignored by Git.

### Command-line options

**Scenario and causal structure**

| Option | Default | Meaning |
|---|---|---|
| `--config PATH` | none | JSON generation configuration, such as a saved `generation_config.json` (see [Configuration files](#configuration-files)) |
| `--clinical-question TEXT` | a metastatic ER+ breast cancer comparison | Treatment comparison the cohort represents. Treatment 1 is the first treatment it names |
| `--dataset-size N` | 500 | Number of patients |
| `--num-features N` | LLM chooses 8–12 | Total variables, not counting instruments, prognostic variables, or proxies. When omitted but a role count is given, it is the sum of the role counts |
| `--num-confounders N` | none | Variables with the confounder role |
| `--num-effect-modifiers N` | none | Variables with the effect-modifier role. A variable with both roles counts toward both |
| `--treatment-effect X` | 0.20 | Treatment main-effect coefficient: log odds for binary outcomes, outcome units for continuous outcomes. It is not the ATE, because effect-modifier interactions also contribute |
| `--target-treatment-rate P` | 0.5 | Mean treatment probability |
| `--target-control-outcome-rate P` | 0.5 | Mean outcome probability under the comparator |
| `--target-logit-std S` | 2.0 | Standard deviation of each linear predictor; smaller values push propensities toward 0.5 |
| `--enforce-positivity` | off | Clip every patient's propensity to `[--min-treatment-rate, --max-treatment-rate]` |
| `--min-treatment-rate P`, `--max-treatment-rate P` | 0.1, 0.9 | Clipping bounds used with `--enforce-positivity` |
| `--outcome-type` | `binary` | `binary` or `continuous` |
| `--outcome-noise-std S` | 1.0 | Noise standard deviation for continuous outcomes |
| `--seed N` | 42 | Random seed |

**Roles and equation shape**

| Option | Default | Meaning |
|---|---|---|
| `--num-instruments N` | 0 | Variables in the treatment equation only |
| `--num-prognostic N` | 0 | Variables in the outcome equation only |
| `--modifier-main-effects` | off | Give effect-modifier-only variables an outcome main term |
| `--nonlinearity` | `none` | `none`, `threshold`, or `quadratic` shape of continuous outcome main terms and treatment interactions |
| `--treatment-logit-std S`, `--outcome-logit-std S` | `--target-logit-std` | Standard deviation of each linear predictor separately |
| `--modifier-strength X` | 1.0 | Multiplier for the treatment × effect-modifier coefficients |

**Generation-time factors** (all off by default; see
[Generation-time factors](#generation-time-factors))

| Option | Default | Meaning |
|---|---|---|
| `--covariate-correlation R` | 0 | Exchangeable correlation of the copula's latent normals, in [0, 1) |
| `--documentation-rate P` | 1.0 | Probability that each value is given to the note writer; 1.0 gives every value |
| `--documentation-mechanism` | `mcar` | `mcar` (constant probability) or `mnar` (logistic in the standardized value) |
| `--documentation-mnar-slope X` | 1.0 | Slope of the MNAR logistic |
| `--documented-variables LIST` | all variables | Comma-separated variables under documentation control; the others are always documented |
| `--post-treatment-rate P` | off | Write post-treatment dated notes for this fraction of patients |
| `--post-treatment-events LIST` | `toxicity,early_response` | Comma-separated post-treatment events |
| `--post-treatment-regimen-mention-prob P` | 0.5 | Probability that a post-treatment note names the regimen |
| `--num-proxy-variables N` | 0 | Confounders that get a near-duplicate proxy variable |
| `--proxy-correlation R` | 0.9 | Latent correlation of each proxy with its confounder |

**Text**

| Option | Default | Meaning |
|---|---|---|
| `--min-events N`, `--max-events N` | 15, 30 | Events per patient timeline (minimum at least 5) |
| `--note-separator TEXT` | `"\n\n<new_note>\n\n"` | String placed between documents in `clinical_text` |
| `--drug-perturbation-prob P` | 0.3 | Probability of replacing a drug name with a brand name or abbreviation |
| `--structured-data` | off | Add encounter, hospitalization, laboratory, and patient-reported-outcome blocks |
| `--no-encounters`, `--no-labs`, `--no-hospitalizations`, `--no-pros` | off | Omit one structured-data block type |

**LLM endpoint and execution**

| Option | Default | Meaning |
|---|---|---|
| `--api-url URL` | `http://localhost:8000/v1` | OpenAI-compatible base URL |
| `--api-key KEY` | empty | API key, if the server requires one |
| `--model ID` | `openai/gpt-oss-120b` | Served model ID |
| `--max-tokens N` | 50000 | Output ceiling per request |
| `--num-workers N` | 40 | Concurrent patient threads |
| `--output-dir DIR` | `./synthetic_output` | Output directory |
| `--verbose`, `-v` | off | Debug logging |

**Saved DGP**

| Option | Default | Meaning |
|---|---|---|
| `--dgp-metadata PATH` | none | `metadata.json` of an existing cohort; see [Expanding a cohort from a saved DGP](#expanding-a-cohort-from-a-saved-dgp) |

**Optional extraction check.** After generation, the generator can ask the same
endpoint to extract each true variable back from `clinical_text`, as a check of
how recoverable the variables are from the text.

| Option | Default | Meaning |
|---|---|---|
| `--extract-features` (alias `--extract-confounders`), `--no-extract-features` | off | Run the extraction check |
| `--extraction-max-text-tokens N` | 100000 | Keep only the last N tokens of each record |
| `--extraction-max-completion-tokens N` | 5000 | Output ceiling per extraction request |
| `--extraction-batch-size N` | 10 | Progress-reporting batch size; requests are sent one at a time |
| `--reasoning-marker TEXT` | none | Marker that separates reasoning from the answer in responses |

The extraction check loads the tokenizer for `--model` from Hugging Face, so
`--model` must be a Hugging Face ID or a local path. It uses the command-line
endpoint, model, and key even when `--config` is given.

### Configuration files

`--config` loads a JSON file with the same fields as `generation_config.json`.
A command-line option overrides the file only when its value differs from the
option's default; the role, equation-shape, and generation-time factor options
override the file whenever they are given. `--output-dir` always comes from the
command line. A few settings exist only in the file:

| Field | Default | Meaning |
|---|---|---|
| `main_coefficient_scale` | 0.3 | Scale for the LLM's main-effect coefficients and the Python-drawn main terms |
| `interaction_coefficient_scale` | 0.1 | Scale for all interaction coefficients |
| `continuous_covariate_distribution` | `positive_lognormal` | `positive_lognormal` (log-normal when the mean is positive, otherwise normal) or `normal`. A configuration file without this key loads as `normal` |
| `note_types_to_expand` | clinical note, imaging, pathology, NGS reports | Event types expanded into full documents |
| `structured_data.pro_instruments` | `EORTC_QLQ_C30`, `PRO_CTCAE` | Patient-reported-outcome instruments |
| `llm.timeout` | 300 | Request timeout in seconds |
| `llm.api_base_urls` | none | List of equivalent endpoints used in rotation |

The command-line factor options correspond to the fields `num_instruments`,
`num_prognostic`, `modifier_main_effects`, `nonlinearity`,
`treatment_logit_std`, `outcome_logit_std`, `modifier_strength`,
`covariate_correlation`, `documentation_rate`, `documentation_mechanism`,
`documentation_mnar_slope`, `documented_variables_subset`,
`post_treatment_notes`, `num_proxy_variables`, and `proxy_correlation`. A file
without one of these fields uses its default. `post_treatment_notes` is `null`
(off) or an object:

```json
"post_treatment_notes": {
  "rate": 0.8,
  "events": ["toxicity", {"name": "early_response", "treatment_log_odds": 1.5}],
  "regimen_mention_prob": 0.5
}
```

Each event is a built-in name or an object with `name` and any of
`description`, `control_rate`, `treatment_log_odds`, `covariate`,
`covariate_log_odds`, `present`, and `absent` (the sentences the note writer
receives when the event did or did not occur). A name that is not built in
needs `description`, `control_rate`, `treatment_log_odds`, and
`covariate_log_odds`. Unknown keys are errors.

## Generation-time factors

These factors change what the LLM writes, so a corpus has them from the start;
re-equation keeps them as they are. A factor that is off changes no random draw
and no prompt, so a corpus generated with every factor off is the one the
generator writes without these options. Documentation flags and post-treatment
events use per-patient random streams keyed by `--seed`, so they leave the
characteristic, treatment, and outcome draws unchanged.

### Covariate correlation

With `--covariate-correlation R` (or any proxy), characteristics are drawn
from a Gaussian copula. Every variable has a standard-normal latent value, and
the latent values of any two variables have correlation `R` (one shared factor).
Continuous values keep their marginal distributions (normal, or the
moment-matched log-normal), so their correlation is `R` for normal variables and
somewhat lower for log-normal ones. A categorical variable takes the first level,
in its listed order, whose cumulative proportion exceeds the normal CDF of the
latent value, so its proportions are kept and later levels go with higher
latent values. The Monte Carlo draws for scaling and calibration use the same
copula.

### Documentation control

With `--documentation-rate P` below 1, each controlled variable's value is
given to the note writer with probability `P`, independently for each patient
and variable. `mcar` uses `P` for everyone. `mnar` uses
`σ(a + slope × s)`, where `s` is the standardized value (for a categorical
variable, its level index in listed order, standardized under the
proportions) and `a` is solved on 10,000 Monte Carlo draws so that the mean
probability is `P`; with a positive slope, higher values are documented more
often. `--documented-variables` limits the control to the named variables
(names come from the design, so this mainly suits a known variable such as
`age`); the others are always documented.

The timeline prompt lists only the documented values, followed by an
instruction not to state, estimate, or imply a value for any characteristic
that was not given, and its age rule allows histories that do not state the
age. The note-expansion prompt is told the same about characteristics the event
list does not give, and to report only biomarker results the event list gives.
The note writer can still write routinely recorded measurements, such as
laboratory panels, that it was not given; such values carry no information
about the true value. `patient_prompt` holds only the documented values, so the
optional extraction check parses no value for the others.

`true_<variable>__documented` (1 if the value was given to the note writer,
otherwise 0) is written for every variable, and `metadata.json` gets a
`documentation` block: the rate, mechanism, slope, controlled variables, MNAR
intercepts, oracle column names, and achieved rates.

### Post-treatment dated notes

`--post-treatment-rate P` (or a `post_treatment_notes` block) adds treatment-
caused events and notes written after treatment starts, to test what happens
when post-treatment text reaches the analysis.

- **Events.** For each event Python draws `Bernoulli(σ(logit(control_rate) +
  treatment_log_odds × T + covariate_log_odds × s))`, where `s` is the
  standardized value of one covariate: the configured `covariate`, or a
  seeded draw among the confounder-role variables. The built-in events are
  `toxicity` (a grade 3 or higher treatment-related toxicity; `control_rate`
  0.15, `treatment_log_odds` 1.0, `covariate_log_odds` 0.5) and
  `early_response` (an early objective tumor response; 0.25, 0.8, 0.5). The
  draw is `true_post_treatment_<event>`.
- **Dates.** The timeline carries no dates, so Python assigns them: the first
  event falls on a seeded day in 2012 to 2018, and each later event follows the
  previous one by the difference of the ages the events state, in years, less up
  to 90 days, or by 3 to 45 days when the stated age does not increase. Every
  pretreatment document takes its event's date. The timeline ends before the
  treatment decision and defines no decision date, so the
  [index date](glossary.md#index-date) is the day after the last timeline event
  (`index_date`).
- **Notes.** With probability `P` a patient gets one or two (equally likely)
  progress notes dated 7 to 90 days after the index date. The LLM receives the
  pretreatment timeline, the arm (the first or the second treatment named in the
  clinical question), each event's status, and, with probability
  `regimen_mention_prob` per note, an instruction to name the regimen;
  otherwise it is told to call it only the current regimen.

Outputs:

- `notes.parquet`, in the default format of
  [dated-notes input](configuration.md#notes) (`patient_id`, `note_date` as
  `YYYY-MM-DD`, `note_text`, `note_type`): one row per pretreatment document
  (its event type as `note_type`) and per post-index note (`clinical_note`);
- `index_date` in `dataset.parquet`;
- `clinical_text`, the pretreatment documents only;
- `clinical_text_with_post_treatment`, the pretreatment text with the post-index
  notes appended in date order;
- `true_post_treatment_<event>`, and a `post_treatment` block in
  `metadata.json` (rate, events with their covariates, achieved event rates,
  arm labels, note counts, file and column names).

To analyze the leaked text, point `columns.text` at
`clinical_text_with_post_treatment`. To analyze dated notes instead, use a
`notes` block with `path` set to `notes.parquet` and `index_date_column` set to
`index_date`, and set `columns.text` to a name the dataset does not have (the
dataset already has `clinical_text`); OCI then keeps only the notes dated before
the index date. Re-equation refuses these cohorts, because their post-index
notes depend on the original treatment.

### Near-duplicate proxies

With `--num-proxy-variables N`, Python picks `N` confounder-role variables (a
seeded draw), and the design LLM names and describes, for each, a distinct but
clinically related measurement that rises with it. The proxy's latent value is
`r × z + sqrt(1 − r²) × e`, where `z` is the confounder's latent value and
`r = --proxy-correlation`; its marginal distribution comes from the LLM's
summary statistics. A proxy has no role and enters no equation; it appears in
the text like any other characteristic, has its own `true_<proxy>` column, and
carries `proxy_of` in `features` (also listed under `proxies` in
`metadata.json`).

## Re-equation

Re-equation writes a cohort with the same text and true characteristics as a
source cohort, but with treatment and outcome recomputed from equations drawn
in Python for a scenario. It makes no LLM request.

```bash
python -m synthetic_data.cli reequation \
  --source ./synthetic_data/example_synthetic_datasets/five_confounders_five_effect_modifiers_nsclc_with_structured \
  --scenario scenario.json --size 1000 --seed 1 \
  --output-dir ./synthetic_output/reequated/s1_seed1
```

The same operation is `synthetic_data.reequation.reequation_cohort(source_dir,
scenario, *, size, seed, output_dir)`, which returns the output directory.

| Option | Meaning |
|---|---|
| `--source DIR` | Source cohort with `dataset.parquet`, `metadata.json`, and `generation_config.json` |
| `--scenario FILE` | Scenario JSON (below) |
| `--size N` | Patients to keep, drawn without replacement; omitted keeps every patient. A size larger than the source is an error |
| `--seed S` | Random seed (required) |
| `--output-dir DIR` | Output directory |

### Scenario JSON

Every key except `name` is optional; an omitted key takes the generator's
default, and the outcome type (binary or continuous) comes from the source.

```json
{
  "name": "instruments_and_prognostic",
  "roles": {"confounder": 3, "effect_modifier": 2, "dual_role": 1, "instrument": 2, "prognostic": 2},
  "role_assignment": "random",
  "modifier_main_effects": true,
  "nonlinearity": "none",
  "treatment_logit_std": 1.5,
  "outcome_logit_std": 2.0,
  "treatment_effect": 0.2,
  "modifier_strength": 1.0,
  "unmeasured_confounders": 0,
  "unmeasured_strength": 1.0,
  "target_treatment_rate": 0.5,
  "target_control_outcome_rate": 0.5,
  "outcome_noise_std": 1.0
}
```

| Key | Default | Meaning |
|---|---|---|
| `name` | required | Scenario name |
| `roles` | the source's roles | Disjoint groups `confounder`, `effect_modifier`, `dual_role` (both roles), `instrument`, and `prognostic`, each a count or a list of variable names |
| `role_assignment` | `random` | `random`: lists are fixed first, then counts are drawn with a seeded RNG from the remaining variables in the group order above. `named`: every group must be a list |
| `modifier_main_effects` | `false` | Give effect-modifier-only variables an outcome main term |
| `nonlinearity` | `none` | `none`, `threshold`, or `quadratic` (see [How a cohort is generated](#how-a-cohort-is-generated)) |
| `treatment_logit_std`, `outcome_logit_std` | 2.0 | Standard deviations of the linear predictors |
| `treatment_effect` | 0.2 | Treatment main-effect coefficient |
| `modifier_strength` | 1.0 | Multiplier for the treatment interaction coefficients |
| `unmeasured_confounders` | 0 | Number of unmeasured confounders U |
| `unmeasured_strength` | 1.0 | Size of U's coefficients relative to the measured confounders |
| `target_treatment_rate`, `target_control_outcome_rate` | 0.5, 0.5 | Calibration targets |
| `outcome_noise_std` | 1.0 | Noise standard deviation for continuous outcomes |

Unknown keys, unknown variable names, a variable in two groups, a proxy given a
role, and counts larger than the unassigned variables are errors. Variables left
unassigned get no role and stay in the text.

### Equations, U, and calibration

All coefficients are drawn in Python, the way the generator draws its own terms:
each main term `uniform(−1, 1) × main_coefficient_scale` and each interaction
`uniform(−1, 1) × interaction_coefficient_scale` (scales from the source's
`generation_config.json`). The treatment equation has confounder and instrument
main terms and the pairwise interactions of confounder terms from different
variables; the outcome equation has confounder and prognostic main terms,
modifier main terms when requested, the treatment main effect, and a treatment
interaction for every effect-modifier term, multiplied by `modifier_strength`.
`nonlinearity` applies as in generation.

Each unmeasured confounder `u_1`, `u_2`, … is a standard-normal variable drawn
in Python. It never appears in any text, which is copied unchanged from the
source. It enters both equations with coefficient ± `unmeasured_strength` times
the root mean square of that equation's measured confounder coefficients (a
random sign per equation); its values are in `true_u_<k>`.

The equations are then rescaled to the two standard deviations and the
intercepts calibrated to the targets with the generator's Monte Carlo procedure
(10,000 draws each), except that the draws are rows of the source cohort drawn
with replacement, with standard-normal U, rather than draws from the summary
statistics, so the targets hold for the covariates actually present. Treatment
and outcome are then drawn for every source patient. The equations depend only
on the source, the scenario, and the seed: subsamples drawn with one seed are
nested (every patient of a smaller size is in the larger one, with the same U,
treatment, and outcome), and repeating a call reproduces the cohort exactly.

### Outputs and refusals

The output directory holds:

| File | Content |
|---|---|
| `dataset.parquet` | Every source column, rows in source order; `treatment_indicator`, `outcome_indicator`, and the true probability and effect columns recomputed (the full set for the outcome type, see [Output files and columns](#output-files-and-columns)); plus `true_u_<k>` |
| `metadata.json` | Features with the scenario's roles, the role lists, the equations, `unmeasured_confounders` (`name`, `coefficients` for `treatment` and `outcome`, `oracle_column`), the resolved `scenario`, the source's `documentation` block, `post_treatment: null`, a `reequation` block (source, seed, size, scales, achieved Monte Carlo rates), and the dataset statistics |
| `reequation_manifest.json` | The source path and the SHA-256 of its three files, the scenario as given and resolved, the size, the seed, the code version and module hashes, and the SHA-256 of the outputs |
| `generation_config.json` | A byte copy of the source's |

Re-equation fails closed, before writing anything, on a source with
post-treatment notes (a `post_treatment` block,
`clinical_text_with_post_treatment`, a `true_post_treatment_*` column, or a
`notes.parquet`), a source without a `true_<variable>` column for every feature
in its metadata (or with values outside a variable's categories), a source that
is itself a re-equation output, and an invalid scenario, size, or seed. An
output directory that already holds the same re-equation (same source hashes,
scenario, size, seed, and code) is returned as is; one that holds anything else
is an error.

**Sources.** The bundled
`five_confounders_five_effect_modifiers_nsclc_with_structured` cohort (1,000
patients, ten `true_<variable>` columns) is a ready source for sizes up to
1,000. It was generated before category labels were matched across Unicode
hyphen spellings, so `prior_platinum_therapy` never takes the level
"First‑line" (written with U+2011): that level's indicator is always 0, and the
cohort's 1,000 patients split among None, Adjuvant, and Multiple. For larger
sizes, first expand the source with
[a saved DGP](#expanding-a-cohort-from-a-saved-dgp) (which samples every level),
then re-equate the expansion.

## Output files and columns

A generation run writes these files to the output directory:

| File | Content |
|---|---|
| `generation_config.json` | Resolved generation settings, written before generation starts |
| `dgp_design.json` | The DGP designed for this run and the settings it depends on; a resumed run reuses it instead of designing another |
| `checkpoint.json` | Completed patients, used to resume an interrupted run |
| `dataset.parquet` | One row per patient |
| `metadata.json` | The DGP: variables, roles, equations, distributions, and summary statistics of the generated cohort |
| `notes.parquet` | With post-treatment notes: every document with its date (see [Post-treatment dated notes](#post-treatment-dated-notes)) |
| `dataset_with_extraction.parquet`, `dataset_with_extraction.metrics.json` | Present when the extraction check runs |

`generation_config.json` and `metadata.json` never store the API key.

`metadata.json` holds `features` (each with `roles`, and `proxy_of` for a
proxy), the role lists `confounders`, `effect_modifiers`, `instruments`, and
`prognostic_variables` (a dual-role variable is in the first two), `proxies`,
`treatment_equation`, `outcome_equation` (with `nonlinearity` and
`nonlinear_terms` when the shape is not linear), `summary_statistics`,
`unmeasured_confounders` (empty for a generated cohort), `scenario` (`null` for
a generated cohort), `documentation` and `post_treatment` (`null` when off), and
`dataset_statistics`.

Columns of `dataset.parquet`:

| Column | Content |
|---|---|
| `patient_id` | Integer identifier, 0 to N − 1 |
| `clinical_text` | The patient's documents, joined in date order |
| `treatment_indicator` | 0 or 1 |
| `outcome_indicator` | 0 or 1 for binary outcomes; a real number for continuous outcomes |
| `num_notes` | Number of documents in `clinical_text` |
| `patient_prompt`, `event_timeline` | The characteristics given to the note writer as text, and the generated timeline |
| `true_treatment_prob` | True propensity `e(X)` |
| `true_y0`, `true_y1` | True expected outcome under comparator and under treatment (probabilities for binary outcomes) |
| `true_ite` | `true_y1 − true_y0`, on the outcome scale |
| `true_outcome_prob`, `true_y0_prob`, `true_y1_prob`, `true_ite_prob` | Binary outcomes only: the outcome probability at the observed treatment, and probability-scale copies of the three columns above |
| `true_<variable>` | True value of each generated variable, proxies included |
| `true_<variable>__documented` | With documentation control: 1 if the value was given to the note writer |
| `true_post_treatment_<event>` | With post-treatment notes: whether the event occurred |
| `index_date`, `clinical_text_with_post_treatment` | With post-treatment notes: the index date and the text with post-index notes appended |
| `true_u_<k>` | Re-equation only: each unmeasured confounder |

There is no `split` column. The workflow assigns folds itself.

`patient_prompt` and `event_timeline` contain the true characteristics in text
form but are not oracle-prefixed. Do not point `--text-column` at either of
them.

Rerunning the same command resumes an interrupted generation: the DGP is read
from `dgp_design.json` and completed patients from `checkpoint.json`. The run
stops with an error if the generation settings changed (anything other than
`output_dir`, `dataset_size`, or LLM connection settings) or if a checkpointed
patient does not match its re-sampled characteristics; use an empty output
directory for different settings. `--seed` fixes each patient's sampled
characteristics, documentation, and post-treatment events for a given DGP
regardless of `--num-workers`; the LLM-designed DGP and the generated text are
not seeded.

## Expanding a cohort from a saved DGP

To generate more patients from an existing cohort's causal process, pass its
configuration and metadata:

```bash
python -m synthetic_data.cli \
  --config ./synthetic_data/example_synthetic_datasets/five_confounders_five_effect_modifiers_nsclc_with_structured/generation_config.json \
  --dgp-metadata ./synthetic_data/example_synthetic_datasets/five_confounders_five_effect_modifiers_nsclc_with_structured/metadata.json \
  --api-url http://127.0.0.1:8000/v1 \
  --model MODEL \
  --dataset-size 10000 \
  --num-workers 32 \
  --output-dir ./synthetic_output/five_conf_five_mod_10000
```

The variables, distributions, equations, and calibrated intercepts are loaded
from `metadata.json` and kept exactly; nothing is recalibrated. The run samples
a separate cohort (`patient_id` 0 to N − 1) and writes text for it; it
does not append to the original rows. Every setting in the source
configuration except `dataset_size`, `output_dir`, `llm`, and `seed` must match
the configuration you pass, which is why `--config` should point to the source
cohort's `generation_config.json`; a factor setting the source configuration
lacks must keep its default. The model and endpoint can differ from the ones
that generated the source. Expansion refuses documentation control and
post-treatment notes, which change how the text is written, and refuses a
re-equation output as its source: expand the source corpus, then re-equate the
expansion.

The bundled cohorts were generated with normal draws for continuous variables
(some laboratory values are negative), and their `generation_config.json` files
set `"continuous_covariate_distribution": "normal"`, so expansions reproduce
those distributions. A saved configuration without the key is treated as
`normal`.

Saved-DGP expansion is designed for long runs:

- Each timeline, document, and completed patient is checkpointed under
  `patients/` and `requests/`; rerunning the same command resumes. The worker
  count and endpoint may change between attempts; any other change to the
  settings, the source metadata, or the generator code makes the run refuse to
  resume in that directory.
- Each request is tried three times. Truncated responses are rejected rather
  than stored as partial records, and the run stops after ten patients fail.
- An exclusive lock (`.generation.lock`) prevents two processes from writing
  the same directory.
- In addition to `dataset.parquet` and `metadata.json` (which records the
  source DGP and a `generation_status`), the directory holds
  `generation_manifest.json`, `patient_scaffold.parquet`,
  `scaffold_sha256.json`, and `progress.json`. The API key is not stored.

Cohorts with 10,000 or more patients placed under
`synthetic_data/example_synthetic_datasets/` should use a directory name
containing `_10000`, which Git ignores.

## Bundled cohorts

Both bundled cohorts are in `synthetic_data/example_synthetic_datasets/`. They
compare vinorelbine with gemcitabine for advanced or metastatic NSCLC, have
binary outcomes, include structured-data text, and were generated with seed 42.

| | One confounder, one modifier | Five confounders, five modifiers |
|---|---|---|
| Directory | `one_confounder_one_effect_modifier_nsclc_with_structured/` | `five_confounders_five_effect_modifiers_nsclc_with_structured/` |
| `run_synthetic.sh` name | `one_conf_one_mod` | `five_conf_five_mod` |
| Patients | 1,000 | 1,000 |
| Confounders | age | age, sex, ECOG performance status, creatinine clearance, prior platinum therapy |
| Effect modifiers | PD-L1 expression (<1%, 1–49%, ≥50%) | histology, EGFR mutation status, baseline neutrophil-to-lymphocyte ratio, brain metastases, baseline hemoglobin |
| Treated fraction | 0.487 | 0.521 |
| Outcome rate | 0.544 | 0.449 |
| Mean (SD) of `true_ite_prob` | 0.063 (0.163) | −0.074 (0.181) |
| Empty `clinical_text` | 1 patient | 5 patients |

Records average about 52,000 characters (maximum about 130,000) and 19
documents per patient.

Each directory contains `dataset.parquet`, `generation_config.json`,
`metadata.json`, and the extraction-check files
`dataset_with_extraction.parquet` and `dataset_with_extraction.metrics.json`.
The bundled `dataset.parquet` files carry the probability-scale oracle columns
(`true_treatment_prob`, `true_outcome_prob`, `true_y0_prob`, `true_y1_prob`,
`true_ite_prob`) and `true_<variable>` columns, but not `true_y0`, `true_y1`, or
`true_ite`. The evaluators therefore use `true_ite_prob` for these cohorts. In
`dataset_with_extraction.parquet`, the `true_<variable>` columns are parsed from
`patient_prompt`, so continuous values are rounded.

In the five-confounder cohort, `prior_platinum_therapy` never takes
"First‑line" (written with U+2011 in the variable definition): the cohort was
generated before category labels were matched across hyphen spellings, so that
level was never sampled. Its patients split among None (465), Adjuvant (393),
and Multiple (142). The cohort's re-equations inherit this; see
[Outputs and refusals](#outputs-and-refusals).

To run the full workflow on a bundled cohort, see the
[README](../README.md) and the `run_synthetic.sh` section of
[Configuration](configuration.md#the-synthetic-launcher-run_syntheticsh).

## Oracle-safe evaluation

Oracle columns are any columns whose names start with `true_`, `oracle_`, or
`ground_truth`. The workflow drops them before Stage 1, and Stage 2 refuses
prediction files that contain them. They are joined to results only after the
predictions are written and hashed. The documentation, post-treatment, and
unmeasured-confounder columns (`true_<variable>__documented`,
`true_post_treatment_<event>`, `true_u_<k>`) follow the same rule.

**Stage 2.** When the dataset contains an oracle effect column, Stage 2
evaluates its frozen per-patient predictions (`estimated_cate` in
`stage2/cross_fitted_predictions.csv`) after all modeling is complete. It uses
`true_ite` when present, otherwise `true_ite_prob` for binary outcomes; a
continuous-outcome dataset with only `true_ite_prob` is reported as not
evaluable. It writes:

- `stage2/posthoc_oracle_ite_metrics.json`: overall and per-fold Pearson and
  Spearman correlations, mean absolute error, root mean squared error, ATE bias,
  and standard deviations of true and estimated effects;
- `stage2/posthoc_predictions_with_oracle_ite.csv`: the predictions joined to
  the oracle column;
- the two correlations, copied into `stage2/causal_estimate.json`.

For datasets without oracle columns the metrics file records that evaluation is
unavailable.

**Stage 1.** `oci-evaluate-stage1` compares each frozen Stage 1 architecture
with the oracle columns. It hashes and loads all Stage 1 evidence, score,
split, and timing files before it reads the dataset, and it never fits or
selects a model.

```bash
oci-evaluate-stage1 --run-dir OUTPUT_DIR --architectures all
```

| Option | Default | Meaning |
|---|---|---|
| `--run-dir DIR` | required | Workflow output directory (must contain `run_config.json`) |
| `--dataset PATH` | `dataset` from `run_config.json` | Dataset with oracle columns |
| `--metadata PATH` | `metadata.json` beside the dataset, if present | DGP metadata, used for variable names |
| `--architectures LIST` | every architecture in the run | Comma-separated names, or `all` |
| `--output-dir DIR` | `OUTPUT_DIR/evaluation/stage1` | Where results are written |
| `--null-draws N` | 200 | Random-chunk null draws per architecture and context for lexical recall; 0 disables the null |

It reads `stage1_architectures/` when present; otherwise it builds that view
from the handoff and writes it into the run directory. On outer held-out rows
it reports, for each architecture, how strongly its scores associate with each
`true_<variable>` (absolute Spearman correlation or one-versus-rest AUC), how
strongly its effect scores correlate with `true_ite` (or `true_ite_prob` for
binary outcomes), nuisance discrimination, matching coverage where relevant,
and a normalized R-loss gain. The embedding architectures are scored on the
directions handed to Stage 2. It also reports lexical recall of each true
variable's name in the evidence, compared in each discovery context with
random text windows of the same number and length drawn from that context's
training rows, inner-fold stability, and the Stage 1 wall time of each
architecture. Association metrics take the best value over an architecture's
score columns and ignore sign. The outputs are `metrics.jsonl`,
`architectures/<architecture>/metrics.jsonl`, `comparison.csv`,
`lexical_recovery_null.csv`, `timing.csv`, `summary.json`, and
`evaluation_manifest.json` (input hashes and the oracle columns used); see
[Stage 1 architectures](stage1_architectures.md#outputs) for their contents.
Rows are aligned by position, so evaluate with the same dataset file the run
used. See
[Stage 1 architectures](stage1_architectures.md) for what each architecture
produces.

**Attribution and ablation.** `oci-evaluate-attribution --oracle` hashes the
frozen Stage 2 artifacts it reads before it loads the oracle columns, then
measures which `true_<variable>` columns the selected Stage 2 variables
recover by held-out R² and credits each recovery to the Stage 1 architectures
behind the variable. `scripts/ablate_stage1_architectures.py compare --oracle`
reports the same recovery for each leave-one-architecture-out rerun. See
[Stage 1 architectures](stage1_architectures.md#attribution-and-ablation).

**Standalone `oci run`.** The explicit-feature workflows write
`applied_inference/posthoc_oracle_metrics.json` under the same rules; see
[Optional extensions](optional_extensions.md#outputs).
