# Standalone HTR

`scripts/run_standalone_htr.py` (`oci/inference/standalone_htr.py`) estimates
treatment effects from text with HTR models alone, per outer fold, outside the
two-stage workflow: there is no Stage 1 handoff, LLM, variable extraction, or
Stage 2. It answers one question directly: how well do the HTR models'
own `tau(x)` predictions recover the treatment effect?

Two methods are available (`--method`, default `two_stage`).

## `two_stage`: cross-fitted HTR nuisance and effect ensembles

On each outer training fold, the Stage 1 `htr_neural` lane's training runs
through `MultiModelForestStage1HTRProvider`, with the Stage 1 template's
`architecture.agentic_attention_variable_forest` settings:

1. **Nuisances.** `nuisance_folds` HTR nuisance networks are fitted on Stage
   1's inner nuisance folds. Each network's out-of-fold predictions are
   calibrated out of fold (`nuisance_calibration`).
2. **Effects.** `effect_folds` HTR effect networks are fitted on inner folds
   with the Stage 1 effect objective: `dina` for a binary outcome (required),
   `pseudo_outcome_mse` by default for a continuous one (`squared_r_loss` also
   available). A continuous-outcome effect network fits the pseudo-outcome
   `(Y - m) / (T - e)` (or the R-loss) from the nuisance networks'
   out-of-fold predictions.
3. **Held-out predictions.** For the outer held-out rows, `tau` is the mean of
   the effect networks' predictions, and `e` and `m` are the means of the
   calibrated nuisance networks' predictions.

### DINA nuisances

A DINA effect network learns the treatment's log-odds-ratio modifier
`delta(x)`. Its loss and its `tau = mu1 - mu0` (risk difference) need the
nuisances `a(x)` and `nu(x)`, which are built from `e`, `mu0`, and `mu1`.
`run.dina_nuisances` (`--dina-nuisances`) chooses where they come from:

- **`htr` (default).** The nuisance networks have a propensity head and one
  outcome head per arm; each row trains only the outcome head of the arm it
  received. The propensity is calibrated out of fold on every row, `mu0` on
  control rows, and `mu1` on treated rows, and `m = e mu1 + (1 - e) mu0`. The
  effect networks train on the outer training rows' out-of-fold `e`, `mu0`,
  and `mu1`, and the held-out `tau` uses the nuisance ensemble's held-out
  means. These probabilities are clipped to
  `[dina_nuisance_clip, 1 - dina_nuisance_clip]` (default 0.01) before `a` and
  `nu` are formed, because `nu` is a logit and isotonic calibration returns
  exact 0 and 1 in pure tail blocks. The nuisances are cross-fitted once, on
  the nuisance folds: an outer training row's nuisances never come from a
  network that trained on it, but they may come from one that trained on rows
  of its own effect fold. Outer held-out rows enter no network.
- **`bow`.** Stage 1 unchanged: sparse bag-of-words logistic nuisances
  cross-fitted inside each effect fold (`stage1_dina.prepare_neural_nuisance`)
  set the loss and `tau`. The nuisance networks are Stage 1's (propensity and
  marginal outcome heads) and only set which rows train the effect
  (`r_stage_min_propensity`, `r_stage_max_propensity`; every row by default)
  and the held-out `e` and `m`.

Attention evidence, which only Stage 2 reads, is not computed. Every inner
network is a Stage 1 checkpoint unit under
`folds/fold_KK/stage1_checkpoints/` (`htr/arm_nuisance/fold_NNN` or
`htr/nuisance/fold_NNN`, `assembly/nuisance_ensemble`, and
`htr/effect/<objective>/fold_NNN`), so an interrupted fold resumes at its next
network. With K inner folds, each outer fold trains 2K networks.

## `joint`: one network with three heads

One network per outer fold, on the HTR extractor, with three heads:

| Head | Target | Loss |
|---|---|---|
| Propensity `e(x)` | `P(T = 1 \| x)` | Binary cross-entropy, label-smoothed |
| Outcome `m(x)` | `E[Y \| x]` (not conditional on `T`) | Binary cross-entropy (label-smoothed) or squared error |
| Effect `tau(x)` | Treatment effect on the outcome scale | `pseudo_outcome_mse`: `(tau(x) - (Y - m(x)) / (T - e(x)))^2` |

The loss per batch is
`outcome loss + alpha_propensity * propensity loss + effect_loss_weight * pseudo-outcome loss`.
The pseudo-outcome uses the network's own current `e` and `m` on the batch,
detached, with `e` clipped to `[e_clip, 1 - e_clip]`; the effect loss is off for
the first `effect_warmup_epochs` epochs.

These nuisances are in-sample: they are fitted on the rows they residualize,
so the pseudo-outcomes lack the protection cross-fitting gives `two_stage`. As
the network fits its training rows, `|T - e|` shrinks and the pseudo-outcomes
grow; each epoch's log line and `history.json` record the mean `|T - e|`, the
share of rows at the propensity clip, and the root mean square of the
pseudo-outcome.

`validation_fraction` of each outer training fold (stratified on treatment and,
for a binary outcome, outcome) is held out of training. With
`selection: "validation_loss"`, the model keeps the epoch after the warmup with
the lowest validation
`outcome loss + alpha_propensity * propensity loss + effect_loss_weight * R-loss`
(the R-loss `(Y - m - tau (T - e))^2` is bounded by the squared outcome
residual); `last_epoch` keeps the final epoch. The held-out `e` (and `m` for a
binary outcome) are calibrated on the validation rows (temperature scaling by
default).

## Folds and held-out quantities

The outer folds are the workflow's own (treatment × outcome stratified, from
`--seed`, through the same function as Stage 1, so `--seed 101` gives a
benchmark seed-101 cell's folds), or the folds of a run's
`components/tfidf/split_provenance.jsonl` (`--splits`). Every row is predicted
once, by models that did not train on it. From the held-out `e`, `m`, and `tau`:
`mu0 = m - e tau`, `mu1 = m + (1 - e) tau`, and the AIPW score
`tau + (T - e) / (e (1 - e)) * (Y - m - (T - e) tau)` with `e` clipped to
`run.aipw_clip`.

## Running it

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# Benchmark cell S0/n5000/seed101 with cross-fitted HTR ensembles (5 inner folds,
# DINA on HTR nuisances): five outer folds at once, three workers on cuda:0 and
# two on cuda:1
python scripts/run_standalone_htr.py run \
  --dataset BENCH/S0/n5000/seed101/data/dataset.parquet \
  --out OUT --seed 101 --inner-folds 5 \
  --devices cuda:0,cuda:1 --jobs-per-device 3 --evaluate

# Stage 1's DINA nuisances (bag-of-words) instead
python scripts/run_standalone_htr.py run --dataset DATA.parquet --out OUT --seed 101 \
  --dina-nuisances bow

# The joint network on the same folds as a finished workflow run
python scripts/run_standalone_htr.py run --method joint --dataset DATA.parquet --out OUT \
  --splits RUN/components/tfidf/split_provenance.jsonl

# Oracle evaluation of a finished run
python scripts/run_standalone_htr.py evaluate --out OUT
```

`--devices` lists devices and `--jobs-per-device` the worker processes on
each; outer folds are assigned to the workers in turn. Rerunning `run` with the
same settings reuses every finished fold (and, for `two_stage`, every finished
inner network). A different method, configuration, dataset, or split in an
existing output directory fails; use a new directory. Fitting reads only the
text, treatment, and outcome columns. `evaluate` (and `--evaluate`) verifies
the SHA-256 of the frozen predictions and of the dataset before it reads any
oracle column, and writes under `OUT/evaluation/`. Use GPUs for full cohorts;
CPU training is practical only for small smoke tests.

## Settings

All settings start from the Stage 1 template
(`example_configs/all_evidence_stage1_model.json`, or `--template`): its
`architecture.htr_*` keys, its `architecture.agentic_attention_variable_forest`
section, and its `training` section. `--htr-model` sets `htr_sentence_model`
(default `prajjwal1/bert-tiny`). Override any setting with `--set KEY=VALUE`
(JSON values; unknown keys are errors), where `KEY` is one of:

| Prefix | Settings |
|---|---|
| `run.` | `method` (`--method`), `outer_folds` (`--folds`, default 5), `seed` (`--seed`, default 42), `aipw_clip` (default 0.02), `dina_nuisances` (`--dina-nuisances`: `htr`, the default, or `bow`), `dina_nuisance_clip` (default 0.01) |
| `architecture.` | The `htr_*` fields and `feature_extractor_type` |
| `training.` | Learning rate, schedule, batch sizes, weight decay, gradient clipping, data-loader workers, `alpha_propensity` |
| `stage1_htr.` | `two_stage`: the `agentic_attention_variable_forest` fields, including `nuisance_folds` and `effect_folds` (`--inner-folds`, template 5), `nuisance_epochs` and `effect_epochs` (`--epochs`, template 10), `nuisance_calibration`, `nuisance_label_smoothing`, `nuisance_weight_decay`, `e_clip`, `effect_objective`, and `r_stage_*_propensity` |
| `joint.` | `joint`: the settings below; `--epochs` sets `joint.epochs` |

| `joint.` key | Default | Meaning |
|---|---|---|
| `epochs` | `30` | Training epochs per fold (`training.epochs` is not used) |
| `effect_loss_weight` | `1.0` | Weight of the pseudo-outcome loss |
| `effect_warmup_epochs` | `5` | Epochs before the effect loss starts (less than `epochs`) |
| `e_clip` | `0.01` | Propensity clip in the pseudo-outcome denominator |
| `label_smoothing` | `0.02` | Label smoothing of the binary nuisance targets |
| `validation_fraction` | `0.1` | Share of each outer training fold held out; `0` requires `selection: "last_epoch"` and `nuisance_calibration: "none"` |
| `selection` | `validation_loss` | `validation_loss` or `last_epoch` |
| `nuisance_calibration` | `temperature` | `none`, `temperature`, `isotonic`, or `temperature_isotonic` |

## Outputs

| Path | Contents |
|---|---|
| `run.json` | Dataset path and SHA-256, method, resolved settings, and the split fingerprints |
| `folds/fold_KK/predictions.parquet` | The fold's held-out predictions |
| `folds/fold_KK/fold.json` | Fold identity and prediction hash, row counts; `two_stage`: effect objective, DINA nuisance source, inner fold counts, reused checkpoint units; `joint`: selected epoch and calibration |
| `folds/fold_KK/inner_oof.parquet` | `two_stage`: the outer training rows' inner out-of-fold `e_hat`, `m_hat` (and, with HTR DINA nuisances, `mu0_hat`, `mu1_hat`, and the raw predictions), residuals, and `tau_hat_r_stage` (with `tau_logit_modifier` for DINA) |
| `folds/fold_KK/stage1_checkpoints/` | `two_stage`: one Stage 1 checkpoint unit per inner network, and `checkpoint_progress.json` |
| `folds/fold_KK/history.json` | `joint`: per-epoch training means and validation metrics |
| `predictions.parquet` | Every row's held-out `propensity`, `m_hat`, `estimated_cate`, `mu0`, `mu1`, `y_residual`, `t_residual`, and `aipw_score` (`joint` adds `propensity_raw` and `m_hat_raw`) |
| `predictions_manifest.json` | SHA-256 of `predictions.parquet` |
| `results.json` | Oracle-free summary: the AIPW ATE and the mean of `tau`; the pooled BLP heterogeneity test (`beta2` > 0, as in Stage 2); held-out R-loss of `tau` against a constant ATE and against zero; the distribution of `tau`; per-fold nuisance losses and AUROCs |
| `evaluation/oracle_metrics.json` | Against the oracle: ATE bias and coverage; CATE RMSE, Pearson and Spearman correlations, and improvement over a constant ATE (the benchmark's `cate_metrics`); per fold; propensity, `m`, `mu0`, and `mu1` agreement with their true values |
| `logs/` | `run.log`, and one log per worker process |
