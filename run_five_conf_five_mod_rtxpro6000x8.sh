#!/usr/bin/env bash

# Run or resume the five-confounder/five-modifier example on eight RTX PRO 6000
# Blackwell GPUs (96 GB each), with managed NVFP4 Gemma servers.
# Usage: ./run_five_conf_five_mod_rtxpro6000x8.sh [OUTPUT_DIR]

# Extraction defaults to full-record LLM requests with ten variables at a time.
# Set STAGE2_DECISION_EXTRACTION=1 to opt into Plumb/ColBERT extraction.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${repo_root}/scripts/run_synthetic_rtxpro6000x8.sh" five_conf_five_mod "$@"
