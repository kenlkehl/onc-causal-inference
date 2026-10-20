#!/usr/bin/env bash
#
# Run or resume the synthetic benchmark end to end on one machine: LLM corpora
# where the selected cells need them, one cohort per scenario, size, and seed,
# one workflow run per cell (one at a time, smallest cells first), and the
# aggregate report against oracle truth.
#
# Usage:
#   [VARIABLE=VALUE ...] ./run_benchmark.sh MACHINE BENCH_DIR [OPTION ...]
#   ./run_benchmark.sh --help
#
# MACHINE
#   A preset in scripts/presets/: external, h100x8, rtxpro6000x8, or
#   dgx_station_gb300 (see run_synthetic.sh).  Each run is launched as
#   PRESET=MACHINE ./run_synthetic.sh BENCH_DIR/.../config.json.
# BENCH_DIR
#   The benchmark directory.  Rerun the same command to resume every step.
# OPTION
#   For example --scenarios S0 --sizes 1000 --seeds 101, --plan (print the
#   plan and stop), --through materialize, or --llm-url URL[,URL...] to use
#   running vLLM servers for every LLM role instead of the preset's managed
#   ones (the preset then supplies only its GPU selection, CPU workers, and
#   timeouts).  See ./run_benchmark.sh --help.
#
# Environment variables
#   The runs honor every variable of run_synthetic.sh (STAGE2_MODEL, ...).
#   Corpus generation, needed for S5 and S6 and for re-equation cells larger
#   than the source, uses:
#   GENERATOR_MODEL              model that writes the notes (openai/gpt-oss-120b,
#                                which wrote the bundled cohorts)
#   GENERATOR_ENDPOINT           comma-separated OpenAI-compatible URLs of
#                                running servers; without it a managed preset
#                                starts a vLLM pool for the generator and stops
#                                it before the first run (the external preset
#                                uses http://127.0.0.1:8000/v1)
#   GENERATOR_API_KEY            key for GENERATOR_ENDPOINT
#   GENERATOR_WORKERS            concurrent patients (32 per server; with
#                                --llm-url, --llm-concurrency per server, at most 1000)
#   GENERATOR_VLLM_GPUS          logical GPUs for the pool (the detected GPUs)
#   GENERATOR_VLLM_GPUS_PER_SERVER (1), GENERATOR_VLLM_BASE_PORT (8210),
#   GENERATOR_VLLM_INTERNAL_PORT_BASE (40000),
#   GENERATOR_VLLM_EXTRA_ARGS_JSON (["--gpu-memory-utilization","0.90","--max-model-len","131072"])
#   OCI_PYTHON                   use this interpreter instead of uv-syncing .venv
#                                (the sync adds the local-llm extra, vLLM, only
#                                when OCI starts vLLM servers: not for the
#                                external machine or with --llm-url)

set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
presets="$(cd "${repo_root}/scripts/presets" && ls -- *.env | sed 's/\.env$//' | tr '\n' ' ')"
case "${1:-}" in
    -h|--help) ;;
    "") echo "Usage: $0 MACHINE BENCH_DIR [OPTION ...]   (MACHINE: ${presets})" >&2; exit 2 ;;
    *)
        if [[ ! -f "${repo_root}/scripts/presets/${1}.env" ]]; then
            echo "Unknown MACHINE=${1}; choose one of: ${presets}" >&2
            exit 2
        fi ;;
esac

# Each concurrent LLM request holds a socket; raise the soft open-file limit to
# the hard limit so thousands of requests do not exhaust a default of 1024.
hard_files="$(ulimit -Hn)"
if [[ "$hard_files" == unlimited ]]; then
    hard_files=1048576
fi
ulimit -Sn "$hard_files" 2>/dev/null || true
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
cd "$repo_root"

if [[ -n "${OCI_PYTHON:-}" ]]; then
    python_bin="$OCI_PYTHON"
    echo "Using OCI_PYTHON=${python_bin}; dependency synchronization is skipped."
else
    if ! command -v uv >/dev/null 2>&1; then
        echo "uv is required but was not found on PATH: https://docs.astral.sh/uv/" >&2
        exit 1
    fi
    # vLLM is needed only when OCI starts its own servers: a managed machine
    # preset without --llm-url.
    managed_servers=1
    if [[ "${1:-}" == external ]]; then
        managed_servers=0
    fi
    for argument in "$@"; do
        case "$argument" in
            --llm-url|--llm-url=*) managed_servers=0 ;;
        esac
    done
    echo "Synchronizing ${repo_root}/.venv from the lockfile..."
    if (( ! managed_servers )); then
        uv sync --frozen
    else
        uv sync --frozen --extra local-llm
        "${repo_root}/.venv/bin/python" "${repo_root}/scripts/configure_local_cuda.py"
    fi
    python_bin="${repo_root}/.venv/bin/python"
fi

exec "$python_bin" -m oci.evaluation.benchmark_driver "$@"
