# Modifier-selection reproduction for Note 030

This folder is a standalone research experiment for understanding Stage 2
effect-modifier selection. It reproduces the table in Xiang Meng's Note 030,
which compares:

- the balanced five-modifier mechanism from Note 025 and OCI's saved NSCLC
  five-confounder/five-modifier mechanism;
- selection sample sizes 800, 3,200, and 12,800;
- oracle and estimated nuisance functions; and
- Gao's binary/logOR sparse selector, Zhao's squared R-loss selector, and OCI's
  candidate-wise ridge score with its top-ten-per-fold union.

The experiment uses structured variables only. It does not run clinical-note
generation, feature discovery, extraction, an LLM, or the downstream causal
forest. Its purpose is to isolate modifier-selection behavior.

The follow-on downstream experiment is in
`downstream_effect_estimation/`. It crosses the true versus Gao-selected
modifier support with OCI's causal forest versus a Gao-Hastie counterfactual
estimator on both saved NSCLC mechanisms.

## Run it

Requirements:

- Python 3.11 or newer with the packages in `requirements.txt`;
- R with `glmnet` and `jsonlite`; and
- approximately 2 GB of temporary disk space.

From the OCI repository root, run:

```bash
bash research/modifier_selection/run.sh /tmp/note030-reproduction
```

The output directory must not already exist. The full run uses ten fixed
replications. It regenerates all inputs from fixed seeds and reads only the
tracked NSCLC `dataset.parquet` and `metadata.json`.

Run the fast deterministic tests separately with:

```bash
python -m unittest discover -s research/modifier_selection/tests
```

## Read the result

The primary output is:

```text
/tmp/note030-reproduction/full_table.csv
```

Other useful files are:

- `replications.csv`: all 360 mechanism/size/nuisance/method/replication rows;
- `summary.csv`: means and Monte Carlo standard errors;
- `verification.json`: comparison with the frozen Note 030 table;
- `sparse_coefficients.csv`: Gao and Zhao coordinate coefficients;
- `oci_candidate_scores.csv.gz`: OCI fold-specific candidate rankings;
- `fit_warnings.json`: retained fitting warnings; and
- `runtime.json`: Python, R, and package versions.

A successful run ends with:

```text
PASS: reproduced Note 030 at ...
```

The frozen expected table is `expected_note030.csv`. The runner fails if any
reported modifier/noise mean differs from that table.

The validated full run used Python 3.11.13, NumPy 1.26.4, pandas 2.3.1,
SciPy 1.16.0, scikit-learn 1.7.1, R 4.5.1, and glmnet 4.1-10. It reproduced
the table with maximum error zero. The run retained 183 internal glmnet
path-convergence warnings because it refits every cell rather than reusing
earlier saved fits; they are preserved in `fit_warnings.json`.

## What each file does

- `run.py` generates all fixed-seed inputs, calls R, computes OCI scores,
  summarizes results, and verifies the final table.
- `fit_methods.R` generates the larger R-based conditions, fits nuisances, and
  runs Gao and Zhao in every cell.
- `methods.R` contains the readable Gao and Zhao objectives and nuisance fits.
- `oci_ridge.py` contains the current OCI nested ridge calculation used here.
- `expected_note030.csv` is the frozen target table.
- `tests/test_core.py` checks rank handling and recovery of a known interaction.

## Future end-to-end integration

No end-to-end OCI code is changed on this branch. A later integration should:

1. define a common selector input containing the frozen patient-by-feature
   matrix, candidate group IDs, outcomes, treatment, fold IDs, and cross-fitted
   nuisance predictions;
2. expose Gao, Zhao, and current OCI scoring behind a Stage 2 configuration
   option after discovery, role adjudication, and extraction are complete;
3. preserve categorical feature groups so a later group-lasso implementation
   can select a clinical variable jointly rather than dummy by dummy;
4. keep validation outcomes isolated from nuisance fitting, penalty calibration,
   adaptive candidate generation, and final-estimator fitting;
5. pass only the selected modifier IDs into the existing downstream estimator,
   leaving that estimator and its evaluation split fixed; and
6. save candidate scores, selected support, nuisance diagnostics, configuration,
   and source provenance for every end-to-end run.

The first integration test should use OCI's frozen extracted features and
existing train/validation assignments. It should compare selectors while
holding discovery, extraction, nuisances, and the downstream estimator fixed.
