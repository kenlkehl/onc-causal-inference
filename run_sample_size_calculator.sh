#!/bin/sh
# Modifier-selection sample-size calculator (docs/sample_size_calculator.md).
#   ./run_sample_size_calculator.sh [--port 8765] [--runs-dir DIR]   # dashboard
#   ./run_sample_size_calculator.sh --config CONFIG.json --output NEW_DIR   # one run
# Needs python3 (3.9+), Rscript, and the R packages glmnet and jsonlite; not the
# OCI Python environment. Caller-relative --config and --output paths are kept.
set -eu
repo_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}" exec "${PYTHON:-python3}" -m sample_size_calculator "$@"
