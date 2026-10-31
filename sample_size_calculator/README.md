# Modifier-selection sample-size calculator

A local dashboard that estimates, by simulation, how many patients a
binary-outcome study needs for a selector (Stage 2's orthogonal modifier tests,
or a lasso for the effect scale) to discover its effect modifiers. It needs
Python 3.9+, R, and the R packages `glmnet` and `jsonlite`; the
orthogonal-tests selector also needs the OCI environment. From the repository
root:

```bash
Rscript -e 'install.packages(c("glmnet", "jsonlite"), repos="https://cloud.r-project.org")'
./run_sample_size_calculator.sh                           # open http://127.0.0.1:8765
PYTHON=.venv/bin/python ./run_sample_size_calculator.sh   # with the orthogonal-tests selector
```

| File | Contents |
|---|---|
| `app.py` | Input validation, run creation, run management, the HTTP server, and the command line |
| `population.py` | Seeded per-variable coefficient and prevalence draws, and their validation |
| `orthogonal_tests.py` | Worker that runs Stage 2's orthogonal modifier tests on each simulated study (imports `oci`) |
| `index.html`, `app.js` | The dashboard page |
| `engine.R` | Data-generating process, intercept calibration, study seeds, one simulated study, and the summary |
| `run.R` | R entry point for one run, with cooperative pause and cancel |
| `binary_selection_methods.R` | Nuisance cross-fitting and the Gao (DINA) and Zhao (residual) lasso selectors |
| `configs/` | Example configurations for `--config` |

Documentation: [docs/sample_size_calculator.md](../docs/sample_size_calculator.md).
Tests: `tests/test_sample_size_calculator.py` and
`tests/test_sample_size_calculator_engine.R`.
