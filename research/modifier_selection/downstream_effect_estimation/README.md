# Downstream treatment-effect estimation

This experiment evaluates the third item in the follow-up to Ken: whether a
Gao-Hastie counterfactual estimator can recover patient-level binary treatment
effects more accurately than OCI's causal forest, and how much modifier
selection changes either estimator.

Read `PROTOCOL.md` for the frozen design. It compares both saved NSCLC
mechanisms, three sample sizes, true versus Gao-selected modifiers, and the two
downstream estimators over ten fixed replications.

Read `RESULTS.md` for the completed comparison. The checked-in frozen outputs
are under `results/full/`.

## Run

Create a Python environment, install the pinned requirements, and ensure R
packages `glmnet` and `jsonlite` are available. From the repository root:

```bash
python3 -m venv .venv-downstream
.venv-downstream/bin/python -m pip install \
  -r research/modifier_selection/downstream_effect_estimation/requirements.txt
```

Then run the full experiment with one command:

```bash
PYTHON=/path/to/python bash research/modifier_selection/downstream_effect_estimation/run.sh /tmp/oci-downstream
```

For a one-cell smoke run:

```bash
python research/modifier_selection/downstream_effect_estimation/run.py \
  --output /tmp/oci-downstream-smoke --sample-sizes 800 --replications 1
```

The output directory must not exist. The main files are:

- `summary.csv`: mean metrics and Monte Carlo standard errors;
- `replications.csv`: every fitted cell;
- `heldout_predictions.parquet`: exact auditable patient-level truth and predictions;
- `selection_manifest.jsonl`: selected support and penalty information;
- `selections/`: one readable selection record per mechanism/size/replication;
- `audit.json`: independent reconstruction result; and
- `runtime.json`: versions and executed settings.

The primary columns are `rd_pearson_mean` and `rd_rmse_mean`. Correlation is
reported directly; it is neither squared correlation nor prediction R-squared.
