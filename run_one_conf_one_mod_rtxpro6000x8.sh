#!/usr/bin/env bash

# Run or resume the one-confounder/one-modifier example on eight RTX PRO 6000
# Blackwell GPUs (96 GB each), with managed NVFP4 Gemma servers.
# Usage: ./run_one_conf_one_mod_rtxpro6000x8.sh [OUTPUT_DIR]

# Extraction defaults to full-record LLM requests with ten variables at a time.
# Set STAGE2_DECISION_EXTRACTION=1 to opt into Plumb/ColBERT extraction.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${repo_root}/scripts/run_synthetic_rtxpro6000x8.sh" one_conf_one_mod "$@"
