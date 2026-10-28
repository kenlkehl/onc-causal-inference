# Standalone joint HTR

`scripts/run_standalone_joint_htr.py` (`oci/inference/standalone_joint_htr.py`)
trains one hierarchical transformer per outer fold that predicts the
propensity, the outcome, and the treatment effect at the same time. It runs
outside the two-stage workflow: there is no Stage 1 handoff, LLM, variable
extraction, or Stage 2. It answers one question directly: how well does a
single text model estimate `tau(x)` from the notes?

## Model

The network is the Stage 1 HTR extractor (the `htr_neural` lane's chunk
encoder and chunk transformer) with three heads:

| Head | Target | Loss |
|---|---|---|
| Propensity `e(x)` | `P(T = 1 \| x)` | Binary cross-entropy, label-smoothed |
| Outcome `m(x)` | `E[Y \| x]` (not conditional on `T`) | Binary cross-entropy (label-smoothed) or squared error |
| Effect `tau(x)` | Treatment effect on the outcome scale (the risk difference for a binary outcome) | `pseudo_outcome_mse`: `(tau(x) - (Y - m(x)) / (T - e(x)))^2` |

The propensity and outcome heads share one hidden layer; the effect head has
its own. All three train the shared extractor. The loss per batch is

```text
outcome loss + alpha_propensity * propensity loss + effect_loss_weight * pseudo-outcome loss
```

The pseudo-outcome uses the network's own current `e` and `m` on the batch,
detached, with `e` clipped to `[e_clip, 1 - e_clip]`. The effect loss never
moves the nuisance heads directly. It is off for the first
`effect_warmup_epochs` epochs, so the effect head starts from nuisances that
have already learned something.

Because the nuisances are fit on the same rows they residualize, they are
in-sample: as the network fits its training rows, `|T - e|` shrinks and the
pseudo-outcomes grow. Each epoch's log line and `history.json` record the mean
`|T - e|`, the share of rows at the propensity clip, and the root mean square
of the pseudo-outcome, so you can see when this happens. The Stage 1
`htr_neural` lane instead fits separate cross-fitted nuisance and effect
networks.

## Folds, epoch selection, and calibration

- **Outer folds.** By default, the workflow's own outer folds: treatment ×
  outcome stratified, from `joint.seed` (the same function as Stage 1, so
  `--seed 101` gives a benchmark seed-101 cell's folds). `--splits` reads the
  folds from a run's `components/tfidf/split_provenance.jsonl` instead. Every
  row is predicted once, by the model that did not train on it.
- **Validation rows.** `validation_fraction` of each outer training fold
  (stratified on treatment and, for a binary outcome, outcome) is held out of
  training.
- **Epoch selection.** With `selection: "validation_loss"`, the model keeps
  the epoch after the warmup with the lowest validation
  `outcome loss + alpha_propensity * propensity loss + effect_loss_weight * R-loss`,
  where the R-loss `(Y - m - tau (T - e))^2` uses the model's own
  validation-row predictions. The R-loss stands in for the pseudo-outcome MSE in
  this criterion because it is bounded by the squared outcome residual.
  `last_epoch` keeps the final epoch.
- **Calibration.** `e` (and `m` for a binary outcome) on the held-out fold are
  calibrated on the validation rows (`nuisance_calibration`, temperature
  scaling by default). `tau` is not recalibrated.

Held-out quantities follow from the three heads:
`mu0 = m - e tau`, `mu1 = m + (1 - e) tau`, and the AIPW score
`tau + (T - e) / (e (1 - e)) * (Y - m - (T - e) tau)` with `e` clipped to
`aipw_clip`.

## Running it

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# Benchmark cell S0/n5000/seed101 on two GPUs (one fold process per device),
# then compare the frozen predictions with the oracle columns
python scripts/run_standalone_joint_htr.py run \
  --dataset BENCH/S0/n5000/seed101/data/dataset.parquet \
  --out OUT --seed 101 --devices cuda:0,cuda:1 --evaluate

# The same outer folds as a finished workflow run
python scripts/run_standalone_joint_htr.py run --dataset DATA.parquet --out OUT \
  --splits RUN/components/tfidf/split_provenance.jsonl

# Oracle evaluation of a finished run
python scripts/run_standalone_joint_htr.py evaluate --out OUT
```

Rerunning `run` with the same settings reuses every finished fold. A different
configuration, dataset, or split in an existing output directory fails; use a
new directory. Fitting reads only the text, treatment, and outcome columns.
`evaluate` (and `--evaluate`) verifies the SHA-256 of the frozen predictions and
of the dataset before it reads any oracle column, and writes under
`OUT/evaluation/`.

Use GPUs for full cohorts; CPU training is practical only for small smoke tests.

## Settings

HTR architecture and optimizer settings come from the Stage 1 template
(`example_configs/all_evidence_stage1_model.json`, or `--template`): its
`architecture.htr_*` keys and its `training` section. `--htr-model` sets
`htr_sentence_model` (default `prajjwal1/bert-tiny`). Override any setting with
`--set architecture.<field>=VALUE`, `--set training.<field>=VALUE`, or
`--set joint.<field>=VALUE` (JSON values); unknown keys are errors.
`training.epochs` is not used; `joint.epochs` is.

| `joint.` key | Default | Meaning |
|---|---|---|
| `outer_folds` (`--folds`) | `5` | Outer cross-fitting folds |
| `seed` (`--seed`) | `42` | Outer folds, validation rows, and initialization |
| `epochs` (`--epochs`) | `30` | Training epochs per fold |
| `effect_loss_weight` | `1.0` | Weight of the pseudo-outcome loss |
| `effect_warmup_epochs` | `5` | Epochs before the effect loss starts (less than `epochs`) |
| `e_clip` | `0.01` | Propensity clip in the pseudo-outcome denominator |
| `label_smoothing` | `0.02` | Label smoothing of the binary nuisance targets |
| `validation_fraction` | `0.1` | Share of each outer training fold held out; `0` requires `selection: "last_epoch"` and `nuisance_calibration: "none"` |
| `selection` | `validation_loss` | `validation_loss` or `last_epoch` |
| `nuisance_calibration` | `temperature` | `none`, `temperature`, `isotonic`, or `temperature_isotonic` |
| `aipw_clip` | `0.02` | Propensity clip in the AIPW scores |

`training.alpha_propensity` weights the propensity loss;
`training.learning_rate`, `lr_schedule`, `batch_size`, `weight_decay`,
`gradient_clip_norm`, `dataloader_workers`, and `dataloader_prefetch_factor`
apply as in Stage 1.

## Outputs

| Path | Contents |
|---|---|
| `run.json` | Dataset path and SHA-256, resolved settings, and the split fingerprints |
| `folds/fold_KK/predictions.parquet` | The fold's held-out predictions |
| `folds/fold_KK/history.json` | Per-epoch training means and validation metrics |
| `folds/fold_KK/fold.json` | Fold identity, selected epoch, row counts, calibration, prediction hash |
| `predictions.parquet` | Every row's held-out `propensity_raw`, `propensity`, `m_hat_raw`, `m_hat`, `estimated_cate`, `mu0`, `mu1`, `y_residual`, `t_residual`, and `aipw_score` |
| `predictions_manifest.json` | SHA-256 of `predictions.parquet` |
| `results.json` | Oracle-free summary: the AIPW ATE and the mean of `tau`; the pooled BLP heterogeneity test (`beta2` > 0, as in Stage 2); held-out R-loss of `tau` against a constant ATE and against zero; the distribution of `tau`; per-fold nuisance losses, AUROCs, and selected epochs |
| `evaluation/oracle_metrics.json` | Against the oracle: ATE bias and coverage; CATE RMSE, Pearson and Spearman correlations, and improvement over a constant ATE (the benchmark's `cate_metrics`); per fold; propensity, `m`, `mu0`, and `mu1` agreement with their true values |
| `logs/` | `run.log`, and one log per device process |
