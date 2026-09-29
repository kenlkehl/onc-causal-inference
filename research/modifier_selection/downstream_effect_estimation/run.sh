#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: bash research/modifier_selection/downstream_effect_estimation/run.sh OUTPUT_DIRECTORY" >&2
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON:-python3}"
"${python_bin}" "${script_dir}/run.py" --output "$1"
"${python_bin}" "${script_dir}/audit.py" "$1"
