#!/usr/bin/env bash
#
# Run or resume the all-evidence workflow (Stage 1 -> handoff -> Stage 2) on a
# bundled synthetic NSCLC cohort.
#
# Usage:
#   [PRESET=NAME] [VARIABLE=VALUE ...] ./run_synthetic.sh COHORT [OUTPUT_DIR] [WORKFLOW_OPTION ...]
#
# COHORT
#   one_conf_one_mod     one confounder and one effect modifier
#   five_conf_five_mod   five confounders and five effect modifiers
# OUTPUT_DIR
#   Defaults to artifacts/research_all_evidence/COHORT_nsclc_full for the
#   external preset and COHORT_nsclc_PRESET_full for the others, with _no_htr
#   appended when DISABLE_HTR=1.  Rerun the same command to resume a run.
# WORKFLOW_OPTION
#   Passed unchanged to the workflow after the launcher's own options, so a
#   later --set wins, for example --set stage2.consolidation_max_rounds=12.
#   See: python scripts/run_all_evidence.py --help
#
# PRESET selects serving and hardware defaults from scripts/presets/NAME.env:
#   external      (default) prestarted OpenAI-compatible servers: the primary
#                 model at http://127.0.0.1:8010/v1 and the extraction model at
#                 http://127.0.0.1:8020/v1
#   h100x8        eight 80 GB H100 GPUs; OCI starts and stops both vLLM pools
#   rtxpro6000x8  eight 96 GB RTX PRO 6000 GPUs; OCI serves one NVFP4 Gemma
#                 model for both Stage 2 roles on all eight GPUs
#
# Cohort defaults, applied after the preset for fresh runs:
#   one_conf_one_mod    STAGE2_WORKERS=4, STAGE2_EXTRACTION_WORKERS=4,
#                       STAGE2_REQUEST_ATTEMPT_TIMEOUT=1800,
#                       STAGE2_REQUEST_TIMEOUT=6000,
#                       STAGE2_EXTRACTION_CONTEXT_WINDOW_TOKENS=128000,
#                       STAGE2_EXTRACTION_CONTEXT_MARGIN_TOKENS=4096
#                       (sized for one external reasoning server with a
#                       128K-token extraction window)
#   five_conf_five_mod  none: 32 concurrent requests per role and the Stage 2
#                       timeout and context-window defaults
# Both cohorts use the same Stage 2 ontology settings (see stage2_defaults below).
#
# Environment variables.  Explicit values override preset and cohort defaults.
#   GPU_COUNT, PHYSICAL_GPUS     Stage 1 GPUs: how many eligible GPUs to select,
#                                or explicit host indices (CUDA_VISIBLE_DEVICES)
#   MIN_FREE_GPU_GB              free memory a GPU needs to be eligible (0)
#   STAGE1_WORKERS               Stage 1 CPU workers (auto)
#   STAGE1_ARCHITECTURES         comma-separated Stage 1 architectures, or all
#                                (default: the template's eight default lanes)
#   DISABLE_HTR=1                skip all HTR modeling
#   STAGE2_ENDPOINT              primary-model URL; set it empty to run Stage 1 only
#   STAGE2_MODEL, STAGE2_EXTRACTION_MODEL
#   STAGE2_EXTRACTION_ENDPOINT   extraction-model URL
#   STAGE2_EXTRACTION_ENDPOINTS  JSON array of {"endpoint": URL, "max_concurrency": N}
#   STAGE2_VLLM_SERVERS, STAGE2_VLLM_GPUS_PER_SERVER
#                                either one makes OCI manage the primary-model pool
#   STAGE2_VLLM_GPUS, STAGE2_VLLM_BASE_PORT, STAGE2_VLLM_INTERNAL_PORT_BASE,
#   STAGE2_VLLM_DOWNLOAD_DIR, STAGE2_VLLM_EXTRA_ARGS_JSON (JSON array of tokens),
#   STAGE2_VLLM_RAPID_SWITCH_SECONDS
#   STAGE2_EXTRACTION_VLLM_*     the same pool settings for the extraction model
#                                (there is no extraction RAPID_SWITCH_SECONDS)
#   STAGE2_WORKERS, STAGE2_EXTRACTION_WORKERS, STAGE2_MODELING_WORKERS
#   STAGE2_REQUEST_TIMEOUT, STAGE2_REQUEST_ATTEMPT_TIMEOUT   (seconds)
#   STAGE2_EXTRACTION_CONTEXT_WINDOW_TOKENS, STAGE2_EXTRACTION_CONTEXT_MARGIN_TOKENS
#   OCI_PYTHON                   use this interpreter instead of uv-syncing .venv
# Any other Stage 2 setting is passed as a WORKFLOW_OPTION: --set stage2.KEY=VALUE.
#
# Stage 2-only relaunch of a saved run.  The saved run_config.json supplies all
# settings; presets and cohort defaults are not applied.
#   STAGE2_ONLY=1 ./run_synthetic.sh COHORT [OUTPUT_DIR]
#   STAGE2_ONLY=1 OCI_RUN_CONFIG=/path/to/run_config.json ./run_synthetic.sh COHORT
# A read-only preflight always runs first.  OCI_PREFLIGHT_ONLY=1 stops after it,
# and STAGE2_RESELECT=1 requests guarded reselection.  Only the serving and
# concurrency variables above apply (not the extraction context budgets).

set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
usage="Usage: $0 {one_conf_one_mod|five_conf_five_mod} [OUTPUT_DIR] [WORKFLOW_OPTION ...]"
datasets="${repo_root}/synthetic_data/example_synthetic_datasets"
case "${1:-}" in
    one_conf_one_mod)
        dataset="${datasets}/one_confounder_one_effect_modifier_nsclc_with_structured/dataset.parquet" ;;
    five_conf_five_mod)
        dataset="${datasets}/five_confounders_five_effect_modifiers_nsclc_with_structured/dataset.parquet" ;;
    *) echo "$usage" >&2; exit 2 ;;
esac
cohort="$1"
shift
output_dir=""
if (( $# )) && [[ "$1" != --* ]]; then
    output_dir="$1"
    shift
fi
workflow_options=("$@")
preset="${PRESET:-external}"
output_name="${cohort}_nsclc_full"
if [[ "$preset" != external ]]; then
    output_name="${cohort}_nsclc_${preset}_full"
fi
if [[ "${DISABLE_HTR:-0}" == 1 ]]; then
    output_name+="_no_htr"
fi
default_output_dir="${repo_root}/artifacts/research_all_evidence/${output_name}"
stage2_only="${STAGE2_ONLY:-0}"

# Serving and concurrency settings, honored for fresh and saved runs alike.
runtime_variables=(
    STAGE2_ENDPOINT STAGE2_MODEL STAGE2_WORKERS STAGE2_MODELING_WORKERS
    STAGE2_REQUEST_TIMEOUT STAGE2_REQUEST_ATTEMPT_TIMEOUT
    STAGE2_EXTRACTION_ENDPOINT STAGE2_EXTRACTION_ENDPOINTS
    STAGE2_EXTRACTION_MODEL STAGE2_EXTRACTION_WORKERS
    STAGE2_VLLM_RAPID_SWITCH_SECONDS
)
for prefix in STAGE2_VLLM_ STAGE2_EXTRACTION_VLLM_; do
    for suffix in SERVERS GPUS GPUS_PER_SERVER BASE_PORT INTERNAL_PORT_BASE DOWNLOAD_DIR EXTRA_ARGS_JSON; do
        runtime_variables+=("${prefix}${suffix}")
    done
done
allowed=" STAGE2_ONLY ${runtime_variables[*]} "
if [[ "$stage2_only" == 1 ]]; then
    allowed+="STAGE2_RESELECT "
else
    allowed+="STAGE2_EXTRACTION_CONTEXT_WINDOW_TOKENS STAGE2_EXTRACTION_CONTEXT_MARGIN_TOKENS "
    for control in STAGE2_RESELECT OCI_PREFLIGHT_ONLY OCI_RUN_CONFIG; do
        if [[ -n "${!control:-}" && "${!control}" != 0 ]]; then
            echo "$control requires STAGE2_ONLY=1." >&2
            exit 2
        fi
    done
fi
for name in $(compgen -v STAGE2_ || true); do
    if [[ "$allowed" != *" $name "* ]]; then
        echo "$name is not used by this launcher in this mode; see the header of $0, or pass --set stage2.KEY=VALUE explicitly." >&2
        exit 2
    fi
done
for name in STAGE2_VLLM_SERVERS STAGE2_EXTRACTION_VLLM_SERVERS; do
    if [[ "${!name:-}" == 0 ]]; then
        unset "$name"
    fi
done

export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
cd "$repo_root"

args=()
set_from() {  # set_from VARIABLE CONFIG_KEY: pass a non-empty variable through --set
    if [[ -n "${!1:-}" ]]; then
        args+=(--set "$2=${!1}")
    fi
}
add_stage2_runtime_args() {
    set_from STAGE2_ENDPOINT stage2.endpoint
    set_from STAGE2_MODEL stage2.model
    set_from STAGE2_WORKERS stage2.workers
    set_from STAGE2_MODELING_WORKERS stage2.modeling_workers
    set_from STAGE2_REQUEST_TIMEOUT stage2.request_timeout
    set_from STAGE2_REQUEST_ATTEMPT_TIMEOUT stage2.request_attempt_timeout
    set_from STAGE2_EXTRACTION_CONTEXT_WINDOW_TOKENS stage2.extraction_context_window_tokens
    set_from STAGE2_EXTRACTION_CONTEXT_MARGIN_TOKENS stage2.extraction_context_margin_tokens
    set_from STAGE2_VLLM_RAPID_SWITCH_SECONDS stage2.vllm_rapid_switch_seconds
    set_from STAGE2_EXTRACTION_MODEL stage2.extraction_llm.model
    set_from STAGE2_EXTRACTION_WORKERS stage2.extraction_llm.workers
    # These flags also drop the other external route inherited from a saved config.
    if [[ -n "${STAGE2_EXTRACTION_ENDPOINTS:-}" ]]; then
        args+=(--stage2-extraction-endpoints "$STAGE2_EXTRACTION_ENDPOINTS")
    fi
    if [[ -n "${STAGE2_EXTRACTION_ENDPOINT:-}" ]]; then
        args+=(--stage2-extraction-endpoint "$STAGE2_EXTRACTION_ENDPOINT")
    fi
    local prefix key
    for prefix in STAGE2_VLLM_ STAGE2_EXTRACTION_VLLM_; do
        key=stage2.vllm
        if [[ "$prefix" == STAGE2_EXTRACTION_VLLM_ ]]; then
            key=stage2.extraction_llm.vllm
        fi
        set_from "${prefix}SERVERS" "${key}.server_count"
        set_from "${prefix}GPUS" "${key}.gpus"
        set_from "${prefix}GPUS_PER_SERVER" "${key}.gpus_per_server"
        set_from "${prefix}BASE_PORT" "${key}.base_port"
        set_from "${prefix}INTERNAL_PORT_BASE" "${key}.internal_port_base"
        set_from "${prefix}DOWNLOAD_DIR" "${key}.download_dir"
        set_from "${prefix}EXTRA_ARGS_JSON" "${key}.extra_args"
    done
}

if [[ "$stage2_only" == 1 ]]; then
    python_bin="${OCI_PYTHON:-${repo_root}/.venv/bin/python}"
    if [[ -n "${OCI_RUN_CONFIG:-}" ]]; then
        if [[ -n "$output_dir" ]]; then
            echo "Pass either OCI_RUN_CONFIG or OUTPUT_DIR for a saved run, not both." >&2
            exit 2
        fi
        run_config="$OCI_RUN_CONFIG"
    else
        run_config="${output_dir:-${default_output_dir}}/run_config.json"
    fi
    args=(--config "$run_config" --stage2-only)
    if [[ "${STAGE2_RESELECT:-0}" == 1 ]]; then
        args+=(--stage2-reselect)
    fi
    add_stage2_runtime_args
    args+=("${workflow_options[@]}")
    echo "Saved run:      ${run_config}"
    "$python_bin" -m oci.inference.research_all_evidence_workflow "${args[@]}" --preflight-only
    if [[ "${OCI_PREFLIGHT_ONLY:-0}" == 1 ]]; then
        exit 0
    fi
    exec "$python_bin" -m oci.inference.research_all_evidence_workflow "${args[@]}"
fi

preset_file="${repo_root}/scripts/presets/${preset}.env"
if [[ ! -f "$preset_file" ]]; then
    echo "Unknown PRESET=${preset}; choose one of: $(cd "${repo_root}/scripts/presets" && ls -- *.env | sed 's/\.env$//' | tr '\n' ' ')" >&2
    exit 2
fi
# shellcheck source=/dev/null
source "$preset_file"
if [[ "$cohort" == one_conf_one_mod ]]; then
    : "${STAGE2_WORKERS:=4}"
    : "${STAGE2_EXTRACTION_WORKERS:=4}"
    : "${STAGE2_REQUEST_ATTEMPT_TIMEOUT:=1800}"
    : "${STAGE2_REQUEST_TIMEOUT:=6000}"
    : "${STAGE2_EXTRACTION_CONTEXT_WINDOW_TOKENS:=128000}"
    : "${STAGE2_EXTRACTION_CONTEXT_MARGIN_TOKENS:=4096}"
fi

managed_primary=0
if [[ -n "${STAGE2_VLLM_SERVERS:-}${STAGE2_VLLM_GPUS_PER_SERVER:-}" ]]; then
    managed_primary=1
else
    : "${STAGE2_ENDPOINT=http://127.0.0.1:8010/v1}"
fi
managed_extraction=0
if [[ -n "${STAGE2_EXTRACTION_VLLM_SERVERS:-}${STAGE2_EXTRACTION_VLLM_GPUS_PER_SERVER:-}" ]]; then
    managed_extraction=1
elif [[ -z "${STAGE2_EXTRACTION_ENDPOINTS:-}" ]]; then
    : "${STAGE2_EXTRACTION_ENDPOINT=http://127.0.0.1:8020/v1}"
fi
stage2_enabled=0
if (( managed_primary )) || [[ -n "${STAGE2_ENDPOINT:-}" ]]; then
    stage2_enabled=1
fi
managed_any=$(( stage2_enabled && (managed_primary || managed_extraction) ))

if [[ -n "${PHYSICAL_GPUS:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="$PHYSICAL_GPUS"
fi
if [[ -n "${OCI_PYTHON:-}" ]]; then
    python_bin="$OCI_PYTHON"
    echo "Using OCI_PYTHON=${python_bin}; dependency synchronization is skipped."
else
    if ! command -v uv >/dev/null 2>&1; then
        echo "uv is required but was not found on PATH: https://docs.astral.sh/uv/" >&2
        exit 1
    fi
    echo "Synchronizing ${repo_root}/.venv from the lockfile..."
    if (( managed_any )); then
        uv sync --frozen --extra local-llm
        "${repo_root}/.venv/bin/python" "${repo_root}/scripts/configure_local_cuda.py"
    else
        uv sync --frozen
    fi
    python_bin="${repo_root}/.venv/bin/python"
fi
"$python_bin" -c 'from sentence_transformers import SentenceTransformer'
if (( managed_any )); then
    if ! "$python_bin" -c 'import importlib.util, sys; sys.exit(importlib.util.find_spec("vllm") is None)'; then
        echo "Managed Stage 2 requires vLLM in the Python environment (install .[local-llm])." >&2
        exit 1
    fi
fi

detect_args=(
    --outer-folds 5 --inner-folds 5
    --gpu-count "${GPU_COUNT:-auto}"
    --workers "${STAGE1_WORKERS:-auto}"
    --min-free-vram-gib "${MIN_FREE_GPU_GB:-0}"
)
if [[ -n "${STAGE2_WORKERS:-}" ]]; then
    detect_args+=(--stage2-workers "$STAGE2_WORKERS")
fi
hardware_line="$("$python_bin" "${repo_root}/scripts/detect_all_evidence_hardware.py" "${detect_args[@]}")" || exit 1
IFS=$'\t' read -r _gpu_count devices worker_count STAGE2_WORKERS cpu_count gpu_summary <<< "$hardware_line"
if [[ -z "$devices" || -z "$worker_count" || -z "$STAGE2_WORKERS" ]]; then
    echo "Hardware detection returned an incomplete result: ${hardware_line}" >&2
    exit 1
fi
if [[ -z "${STAGE2_EXTRACTION_ENDPOINTS:-}" ]]; then
    : "${STAGE2_EXTRACTION_WORKERS:=$STAGE2_WORKERS}"
fi
if (( managed_primary )); then
    : "${STAGE2_VLLM_GPUS:=$devices}"
fi

output_dir="${output_dir:-${default_output_dir}}"
args=(
    --dataset "$dataset"
    --output-dir "$output_dir"
    --unit-id-column patient_id
    --text-column clinical_text
    --treatment-column treatment_indicator
    --outcome-column outcome_indicator
    --outcome-type binary
    --clinical-question "For advanced or metastatic NSCLC, identify pretreatment text features that confound assignment to vinorelbine versus gemcitabine or modify their treatment effect."
    --outer-folds 5
    --inner-folds 5
    --seed 42
    --devices "$devices"
    --workers "$worker_count"
    --htr-model prajjwal1/bert-tiny
    --embedding-model Qwen/Qwen3-Embedding-8B
    --set science.stage1.architecture.multi_model_forest.embedding_contrast.max_chunks=512
    --set science.stage1.architecture.htr_max_chunks=512
)
if [[ -n "${STAGE1_ARCHITECTURES:-}" ]]; then
    args+=(--architectures "$STAGE1_ARCHITECTURES")
fi
if [[ "${DISABLE_HTR:-0}" == 1 ]]; then
    args+=(--disable-htr)
fi
if (( stage2_enabled )); then
    # Stage 2 ontology settings shared by both bundled cohorts.
    stage2_defaults=(
        stage2.consolidation_batch_size=20
        stage2.consolidation_max_rounds=55
        stage2.operationalization_max_prompt_chars=640000
        stage2.ontology_refinement_min_failure_patients=3
        stage2.max_ontology_refinement_rounds=2
    )
    for setting in "${stage2_defaults[@]}"; do
        args+=(--set "$setting")
    done
    add_stage2_runtime_args
    if (( managed_primary )); then
        stage2_description="managed vLLM ${STAGE2_MODEL:-} on ${STAGE2_VLLM_GPUS}"
    else
        stage2_description="${STAGE2_ENDPOINT} ${STAGE2_MODEL:-}"
    fi
    stage2_description+=" (${STAGE2_WORKERS} concurrent requests); extraction: "
    if (( managed_extraction )); then
        stage2_description+="managed vLLM ${STAGE2_EXTRACTION_MODEL:-} on ${STAGE2_EXTRACTION_VLLM_GPUS:-}"
        if (( managed_primary )); then
            if [[ "${STAGE2_MODEL:-}" == "${STAGE2_EXTRACTION_MODEL:-}" ]]; then
                stage2_description+=", one shared pool over both GPU sets"
            else
                stage2_description+=", models alternate across the GPU union"
            fi
        fi
    elif [[ -n "${STAGE2_EXTRACTION_ENDPOINTS:-}" ]]; then
        stage2_description+="external server pool ${STAGE2_EXTRACTION_MODEL:-}"
    else
        stage2_description+="${STAGE2_EXTRACTION_ENDPOINT:-} ${STAGE2_EXTRACTION_MODEL:-}"
    fi
    stage2_description+=" (${STAGE2_EXTRACTION_WORKERS:-sum of server caps} concurrent requests)"
else
    args+=(--stage1-only)
    stage2_description="disabled (STAGE2_ENDPOINT is empty)"
fi
args+=("${workflow_options[@]}")

echo "Dataset:        ${dataset}"
echo "Output:         ${output_dir}"
echo "Progress:       ${output_dir}/progress.json"
echo "Log:            ${output_dir}/logs/workflow.log"
echo "Preset:         ${preset}"
echo "CUDA devices:   ${devices}"
echo "GPU memory:     ${gpu_summary}"
echo "CPU budget:     ${worker_count} workers (${cpu_count} available)"
echo "Stage 2:        ${stage2_description}"
echo "Architectures:  ${STAGE1_ARCHITECTURES:-default lanes from the Stage 1 template}"
echo "HTR modeling:   $([[ "${DISABLE_HTR:-0}" == 1 ]] && echo disabled || echo enabled)"

exec "$python_bin" -m oci.inference.research_all_evidence_workflow "${args[@]}"
