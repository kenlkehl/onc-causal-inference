#!/usr/bin/env bash

# Shared implementation for the researcher-facing synthetic Stage 1 -> 2 examples.
# A completed handoff automatically selects endpoint-backed Stage 2-only mode,
# which requires no local GPU inspection or allocation.

set -euo pipefail

if (( $# < 2 || $# > 3 )); then
    echo "Usage: $0 DATASET_RELATIVE_PATH OUTPUT_NAME [OUTPUT_DIR]" >&2
    exit 2
fi

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
dataset="${repo_root}/$1"
output_name="$2"
requested_output_dir="${3:-}"
cd "${repo_root}"

# Validate before dependency synchronization, GPU inspection, or checkpoint writes.
for control in STAGE2_ONLY STAGE2_RESELECT OCI_PREFLIGHT_ONLY; do
    value="${!control-0}"
    if [[ "$value" != "0" && "$value" != "1" ]]; then
        echo "$control must be 0 or 1." >&2
        exit 2
    fi
done
case "${STAGE2_SELECTION_MODE:-}" in
    ""|llm_roles|independent_tasks) ;;
    *) echo "STAGE2_SELECTION_MODE must be llm_roles or independent_tasks." >&2; exit 2 ;;
esac
if [[ "${STAGE2_RESELECT:-0}" == "1" && "${STAGE2_ONLY:-0}" != "1" ]]; then
    echo "STAGE2_RESELECT=1 requires STAGE2_ONLY=1." >&2
    exit 2
fi
if [[ -n "${OCI_RUN_CONFIG:-}" || -n "${OCI_STAGE2_SOURCE:-}" || "${STAGE2_ONLY:-0}" == "1" || "${OCI_PREFLIGHT_ONLY:-0}" == "1" ]]; then
    if [[ "${STAGE2_ONLY:-0}" != "1" ]]; then
        echo "OCI_RUN_CONFIG and OCI_PREFLIGHT_ONLY require STAGE2_ONLY=1." >&2
        exit 2
    fi
    if [[ "${STAGE2_ENDPOINT-unset}" == "" ]]; then
        echo "Stage 2-only reuse cannot be combined with STAGE2_ENDPOINT= (Stage 2 disabled)." >&2
        exit 2
    fi
    python_bin="${OCI_PYTHON:-${repo_root}/.venv/bin/python}"
    if [[ ! -x "$python_bin" ]]; then
        echo "Set OCI_PYTHON to an existing executable; saved-run launch never syncs dependencies." >&2
        exit 2
    fi
    if [[ -z "$requested_output_dir" && -z "${OCI_RUN_CONFIG:-}" ]]; then
        requested_output_dir="${repo_root}/artifacts/research_all_evidence/${output_name}"
    fi
    export PYTHONDONTWRITEBYTECODE=1
    exec "$python_bin" "${repo_root}/scripts/launch_saved_stage2.py" "$dataset" "$requested_output_dir"
fi

disable_htr="${DISABLE_HTR:-0}"
gpu_limit="${GPU_COUNT:-auto}"
physical_gpus="${PHYSICAL_GPUS:-}"
stage1_workers="${STAGE1_WORKERS:-auto}"
stage2_workers="${STAGE2_WORKERS:-32}"
stage2_request_timeout="${STAGE2_REQUEST_TIMEOUT:-}"
stage2_request_attempt_timeout="${STAGE2_REQUEST_ATTEMPT_TIMEOUT:-}"
stage2_model="${STAGE2_MODEL:-}"
stage2_extraction_model="${STAGE2_EXTRACTION_MODEL:-}"
stage2_extraction_workers="${STAGE2_EXTRACTION_WORKERS:-}"
stage2_max_tokens="${STAGE2_MAX_TOKENS:-}"
stage2_extraction_max_tokens="${STAGE2_EXTRACTION_MAX_TOKENS:-}"
stage2_extraction_chunk_size_tokens="${STAGE2_EXTRACTION_CHUNK_SIZE_TOKENS:-}"
stage2_extraction_context_window_tokens="${STAGE2_EXTRACTION_CONTEXT_WINDOW_TOKENS:-}"
stage2_extraction_context_margin_tokens="${STAGE2_EXTRACTION_CONTEXT_MARGIN_TOKENS:-}"
stage2_vllm_servers="${STAGE2_VLLM_SERVERS:-0}"
stage2_vllm_gpus="${STAGE2_VLLM_GPUS:-}"
stage2_vllm_gpus_per_server="${STAGE2_VLLM_GPUS_PER_SERVER:-}"
stage2_vllm_rapid_switch_seconds="${STAGE2_VLLM_RAPID_SWITCH_SECONDS:-}"
stage2_vllm_base_port="${STAGE2_VLLM_BASE_PORT:-}"
stage2_vllm_internal_port_base="${STAGE2_VLLM_INTERNAL_PORT_BASE:-}"
stage2_vllm_download_dir="${STAGE2_VLLM_DOWNLOAD_DIR:-}"
stage2_vllm_extra_args_json="${STAGE2_VLLM_EXTRA_ARGS_JSON:-}"
stage2_extraction_vllm_servers="${STAGE2_EXTRACTION_VLLM_SERVERS:-0}"
stage2_extraction_vllm_gpus="${STAGE2_EXTRACTION_VLLM_GPUS:-}"
stage2_extraction_vllm_gpus_per_server="${STAGE2_EXTRACTION_VLLM_GPUS_PER_SERVER:-}"
stage2_extraction_vllm_base_port="${STAGE2_EXTRACTION_VLLM_BASE_PORT:-}"
stage2_extraction_vllm_internal_port_base="${STAGE2_EXTRACTION_VLLM_INTERNAL_PORT_BASE:-}"
stage2_extraction_vllm_download_dir="${STAGE2_EXTRACTION_VLLM_DOWNLOAD_DIR:-}"
stage2_extraction_vllm_extra_args_json="${STAGE2_EXTRACTION_VLLM_EXTRA_ARGS_JSON:-}"
stage2_operationalization_max_prompt_chars="${STAGE2_OPERATIONALIZATION_MAX_PROMPT_CHARS:-}"
stage2_consolidation_batch_size="${STAGE2_CONSOLIDATION_BATCH_SIZE:-}"
stage2_consolidation_alphabetical_rounds="${STAGE2_CONSOLIDATION_ALPHABETICAL_ROUNDS:-}"
stage2_consolidation_max_rounds="${STAGE2_CONSOLIDATION_MAX_ROUNDS:-}"
stage2_extraction_feature_batch_size="${STAGE2_EXTRACTION_FEATURE_BATCH_SIZE:-}"
stage2_ontology_refinement_min_failure_patients="${STAGE2_ONTOLOGY_REFINEMENT_MIN_FAILURE_PATIENTS:-}"
stage2_max_ontology_refinement_rounds="${STAGE2_MAX_ONTOLOGY_REFINEMENT_ROUNDS:-}"
stage2_cluster_similarity_threshold="${STAGE2_CLUSTER_SIMILARITY_THRESHOLD:-}"
stage2_cluster_consensus_fraction="${STAGE2_CLUSTER_CONSENSUS_FRACTION:-}"
stage2_cluster_max_size="${STAGE2_CLUSTER_MAX_SIZE:-}"
stage2_max_latents_per_cluster="${STAGE2_MAX_LATENTS_PER_CLUSTER:-}"
stage2_cluster_tool_call_limit="${STAGE2_CLUSTER_TOOL_CALL_LIMIT:-}"
stage2_adjudicator_tool_call_limit="${STAGE2_ADJUDICATOR_TOOL_CALL_LIMIT:-}"
stage1_architectures="${STAGE1_ARCHITECTURES:-}"
min_free_gpu_gib="${MIN_FREE_GPU_GB:-20}"
python_bin="${OCI_PYTHON:-}"
outer_folds=5
inner_folds=5

if [[ "${disable_htr}" != "0" && "${disable_htr}" != "1" ]]; then
    echo "DISABLE_HTR must be 0 or 1." >&2
    exit 1
fi
if [[ ! "${stage2_vllm_servers}" =~ ^[0-9]+$ ]]; then
    echo "STAGE2_VLLM_SERVERS must be a nonnegative integer." >&2
    exit 1
fi
if [[ ! "${stage2_extraction_vllm_servers}" =~ ^[0-9]+$ ]]; then
    echo "STAGE2_EXTRACTION_VLLM_SERVERS must be a nonnegative integer." >&2
    exit 1
fi
if [[ -n "${stage2_vllm_gpus_per_server}" && ! "${stage2_vllm_gpus_per_server}" =~ ^[1-9][0-9]*$ ]]; then
    echo "STAGE2_VLLM_GPUS_PER_SERVER must be a positive integer." >&2
    exit 1
fi
if [[ -n "${stage2_extraction_vllm_gpus_per_server}" && ! "${stage2_extraction_vllm_gpus_per_server}" =~ ^[1-9][0-9]*$ ]]; then
    echo "STAGE2_EXTRACTION_VLLM_GPUS_PER_SERVER must be a positive integer." >&2
    exit 1
fi
stage2_vllm_servers=$((10#${stage2_vllm_servers}))
stage2_extraction_vllm_servers=$((10#${stage2_extraction_vllm_servers}))
stage2_managed_orchestrator=0
if (( stage2_vllm_servers > 0 )) || [[ -n "${stage2_vllm_gpus_per_server}" ]]; then
    stage2_managed_orchestrator=1
fi
stage2_managed_extractor=0
if (( stage2_extraction_vllm_servers > 0 )) || [[ -n "${stage2_extraction_vllm_gpus_per_server}" ]]; then
    stage2_managed_extractor=1
fi
stage2_managed_any=$((stage2_managed_orchestrator || stage2_managed_extractor))
if (( stage2_managed_orchestrator )); then
    stage2_endpoint="${STAGE2_ENDPOINT:-}"
else
    stage2_endpoint="${STAGE2_ENDPOINT-http://127.0.0.1:8010/v1}"
fi
if (( stage2_managed_extractor )); then
    stage2_extraction_endpoint="${STAGE2_EXTRACTION_ENDPOINT:-}"
else
    stage2_extraction_endpoint="${STAGE2_EXTRACTION_ENDPOINT-http://127.0.0.1:8020/v1}"
fi
if (( stage2_managed_orchestrator )) && [[ -n "${stage2_endpoint}" ]]; then
    echo "Set either STAGE2_ENDPOINT or managed orchestrator vLLM settings, not both." >&2
    exit 1
fi
if (( stage2_managed_orchestrator )) && [[ -z "${stage2_model}" ]]; then
    echo "STAGE2_MODEL is required for managed orchestrator vLLM." >&2
    exit 1
fi
if (( stage2_managed_extractor )) && [[ -n "${stage2_extraction_endpoint}" ]]; then
    echo "Set either STAGE2_EXTRACTION_ENDPOINT or managed extraction vLLM settings, not both." >&2
    exit 1
fi
if (( stage2_managed_extractor )) && [[ -z "${stage2_extraction_model}" ]]; then
    echo "STAGE2_EXTRACTION_MODEL is required for managed extraction vLLM." >&2
    exit 1
fi
if (( stage2_managed_extractor )) && [[ -z "${stage2_extraction_vllm_gpus}" ]]; then
    echo "STAGE2_EXTRACTION_VLLM_GPUS is required for managed extraction vLLM." >&2
    exit 1
fi
if (( stage2_managed_orchestrator && stage2_managed_extractor )) && [[ -z "${stage2_vllm_gpus}" ]]; then
    echo "STAGE2_VLLM_GPUS is required when both managed model pools are enabled." >&2
    exit 1
fi
stage2_enabled=0
if (( stage2_managed_orchestrator )) || [[ -n "${stage2_endpoint}" ]]; then
    stage2_enabled=1
fi
if (( ! stage2_enabled )) && [[ -n "${STAGE2_SELECTION_MODE:-}" ]]; then
    echo "STAGE2_SELECTION_MODE requires Stage 2 to be enabled." >&2
    exit 2
fi
if (( stage2_managed_extractor && ! stage2_enabled )); then
    echo "Managed extraction vLLM also requires an external or managed orchestrator." >&2
    exit 1
fi
if (( stage2_enabled && ! stage2_managed_extractor )) && [[ -z "${stage2_extraction_endpoint}" ]]; then
    echo "STAGE2_EXTRACTION_ENDPOINT or managed extraction vLLM settings are required whenever Stage 2 is enabled." >&2
    exit 1
fi
if [[ "${disable_htr}" == "1" ]]; then
    default_output_dir="${repo_root}/artifacts/research_all_evidence/${output_name}_no_htr"
    htr_args=(--disable-htr)
else
    default_output_dir="${repo_root}/artifacts/research_all_evidence/${output_name}"
    htr_args=()
fi
output_dir="${requested_output_dir:-${default_output_dir}}"
stage2_only=0
if [[
    -n "${stage2_endpoint}"
    && "${stage2_managed_any}" == "0"
    && -f "${output_dir}/handoff/evidence.jsonl"
    && -f "${output_dir}/handoff/complete.json"
]]; then
    stage2_only=1
fi
if (( ! stage2_only )) && [[ -n "${physical_gpus}" && "${gpu_limit}" != "auto" ]]; then
    echo "Set either PHYSICAL_GPUS or GPU_COUNT, not both." >&2
    exit 1
fi

if [[ ! -f "${dataset}" ]]; then
    echo "Dataset not found: ${dataset}" >&2
    exit 1
fi
if (( ! stage2_only )) && [[ -n "${physical_gpus}" ]]; then
    IFS=',' read -r -a physical_gpu_ids <<< "${physical_gpus}"
    if (( ${#physical_gpu_ids[@]} < 1 )); then
        echo "PHYSICAL_GPUS must contain at least one GPU index." >&2
        exit 1
    fi
    declare -A seen_gpu_ids=()
    for gpu_id in "${physical_gpu_ids[@]}"; do
        if [[ ! "${gpu_id}" =~ ^[0-9]+$ ]]; then
            echo "PHYSICAL_GPUS must contain comma-separated nonnegative integers." >&2
            exit 1
        fi
        if [[ -n "${seen_gpu_ids[${gpu_id}]:-}" ]]; then
            echo "PHYSICAL_GPUS contains duplicate GPU index ${gpu_id}." >&2
            exit 1
        fi
        seen_gpu_ids["${gpu_id}"]=1
    done
    export CUDA_VISIBLE_DEVICES="${physical_gpus}"
fi

if [[ -z "${python_bin}" ]]; then
    if ! command -v uv >/dev/null 2>&1; then
        echo "uv is required but was not found on PATH: https://docs.astral.sh/uv/" >&2
        exit 1
    fi
    echo "Synchronizing ${repo_root}/.venv from the lockfile..."
    if (( stage2_managed_any )); then
        uv sync --frozen --extra local-llm
    else
        uv sync --frozen
    fi
    python_bin="${repo_root}/.venv/bin/python"
else
    echo "Using OCI_PYTHON=${python_bin}; dependency synchronization is skipped."
fi
if [[ ! -x "${python_bin}" ]]; then
    echo "Python interpreter is not executable: ${python_bin}" >&2
    exit 1
fi
if (( ! stage2_only )); then
    "${python_bin}" -c 'from sentence_transformers import SentenceTransformer'
fi
if (( stage2_managed_any )); then
    if ! "${python_bin}" -c 'import importlib.util, sys; sys.exit(importlib.util.find_spec("vllm") is None)'; then
        echo "Managed Stage 2 requires vLLM in OCI_PYTHON (install .[local-llm])." >&2
        exit 1
    fi
fi

hardware_args=(
    --workers "${stage1_workers}"
    --stage2-workers "${stage2_workers}"
    --outer-folds "${outer_folds}"
    --inner-folds "${inner_folds}"
)
if (( stage2_only )); then
    hardware_args+=(--stage2-only)
else
    hardware_args+=(
        --gpu-count "${gpu_limit}"
        --min-free-vram-gib "${min_free_gpu_gib}"
    )
fi
hardware_line="$(
    "${python_bin}" "${repo_root}/scripts/detect_all_evidence_hardware.py" \
        "${hardware_args[@]}"
)" || exit 1
IFS=$'\t' read -r gpu_count devices worker_count resolved_stage2_workers cpu_count gpu_summary <<< "${hardware_line}"
if [[ -z "${gpu_count}" || -z "${devices}" || -z "${worker_count}" ]]; then
    echo "Hardware detection returned an incomplete result: ${hardware_line}" >&2
    exit 1
fi

stage2_policy_args=()
if [[ -n "${STAGE2_SELECTION_MODE:-}" ]]; then
    stage2_policy_args+=(--set "stage2.statistical_selection.selection_mode=${STAGE2_SELECTION_MODE}")
fi
if (( stage2_managed_extractor )); then
    stage2_policy_args+=(
        --stage2-extraction-model "${stage2_extraction_model}"
        --stage2-extraction-workers "${stage2_extraction_workers:-${resolved_stage2_workers}}"
        --stage2-extraction-vllm-gpus "${stage2_extraction_vllm_gpus}"
    )
    if (( stage2_extraction_vllm_servers > 0 )); then
        stage2_policy_args+=(
            --stage2-extraction-vllm-servers "${stage2_extraction_vllm_servers}"
        )
    fi
    if [[ -n "${stage2_extraction_vllm_gpus_per_server}" ]]; then
        stage2_policy_args+=(
            --stage2-extraction-vllm-gpus-per-server "${stage2_extraction_vllm_gpus_per_server}"
        )
    fi
    if [[ -n "${stage2_extraction_vllm_base_port}" ]]; then
        stage2_policy_args+=(
            --stage2-extraction-vllm-base-port "${stage2_extraction_vllm_base_port}"
        )
    fi
    if [[ -n "${stage2_extraction_vllm_internal_port_base}" ]]; then
        stage2_policy_args+=(
            --stage2-extraction-vllm-internal-port-base "${stage2_extraction_vllm_internal_port_base}"
        )
    fi
    if [[ -n "${stage2_extraction_vllm_download_dir}" ]]; then
        stage2_policy_args+=(
            --stage2-extraction-vllm-download-dir "${stage2_extraction_vllm_download_dir}"
        )
    fi
    if [[ -n "${stage2_extraction_vllm_extra_args_json}" ]]; then
        stage2_policy_args+=(
            --set "stage2.extraction_llm.vllm.extra_args=${stage2_extraction_vllm_extra_args_json}"
        )
    fi
elif [[ -n "${stage2_extraction_endpoint}" ]]; then
    stage2_policy_args+=(
        --stage2-extraction-endpoint "${stage2_extraction_endpoint}"
        --stage2-extraction-workers "${stage2_extraction_workers:-${resolved_stage2_workers}}"
    )
    if [[ -n "${stage2_extraction_model}" ]]; then
        stage2_policy_args+=(--stage2-extraction-model "${stage2_extraction_model}")
    fi
fi
if [[ -n "${stage2_request_timeout}" ]]; then
    stage2_policy_args+=(--set "stage2.request_timeout=${stage2_request_timeout}")
fi
if [[ -n "${stage2_request_attempt_timeout}" ]]; then
    stage2_policy_args+=(--set "stage2.request_attempt_timeout=${stage2_request_attempt_timeout}")
fi
if [[ -n "${stage2_max_tokens}" ]]; then
    stage2_policy_args+=(--stage2-max-tokens "${stage2_max_tokens}")
fi
if [[ -n "${stage2_extraction_max_tokens}" ]]; then
    stage2_policy_args+=(
        --stage2-extraction-max-tokens "${stage2_extraction_max_tokens}"
    )
fi
if [[ -n "${stage2_extraction_chunk_size_tokens}" ]]; then
    stage2_policy_args+=(
        --stage2-extraction-chunk-size-tokens "${stage2_extraction_chunk_size_tokens}"
    )
fi
if [[ -n "${stage2_extraction_context_window_tokens}" ]]; then
    stage2_policy_args+=(
        --stage2-extraction-context-window-tokens "${stage2_extraction_context_window_tokens}"
    )
fi
if [[ -n "${stage2_extraction_context_margin_tokens}" ]]; then
    stage2_policy_args+=(
        --stage2-extraction-context-margin-tokens "${stage2_extraction_context_margin_tokens}"
    )
fi
if [[ -n "${stage2_operationalization_max_prompt_chars}" ]]; then
    stage2_policy_args+=(
        --set "stage2.operationalization_max_prompt_chars=${stage2_operationalization_max_prompt_chars}"
    )
fi
if [[ -n "${stage2_consolidation_batch_size}" ]]; then
    stage2_policy_args+=(
        --set "stage2.consolidation_batch_size=${stage2_consolidation_batch_size}"
    )
fi
if [[ -n "${stage2_consolidation_alphabetical_rounds}" ]]; then
    stage2_policy_args+=(
        --set "stage2.consolidation_alphabetical_rounds=${stage2_consolidation_alphabetical_rounds}"
    )
fi
if [[ -n "${stage2_consolidation_max_rounds}" ]]; then
    stage2_policy_args+=(
        --set "stage2.consolidation_max_rounds=${stage2_consolidation_max_rounds}"
    )
fi
if [[ -n "${stage2_extraction_feature_batch_size}" ]]; then
    stage2_policy_args+=(
        --stage2-extraction-feature-batch-size "${stage2_extraction_feature_batch_size}"
    )
fi
if [[ -n "${stage2_ontology_refinement_min_failure_patients}" ]]; then
    stage2_policy_args+=(
        --set "stage2.ontology_refinement_min_failure_patients=${stage2_ontology_refinement_min_failure_patients}"
    )
fi
if [[ -n "${stage2_max_ontology_refinement_rounds}" ]]; then
    stage2_policy_args+=(
        --set "stage2.max_ontology_refinement_rounds=${stage2_max_ontology_refinement_rounds}"
    )
fi
if [[ -n "${stage2_cluster_similarity_threshold}" ]]; then
    stage2_policy_args+=(--stage2-cluster-similarity-threshold "${stage2_cluster_similarity_threshold}")
fi
if [[ -n "${stage2_cluster_consensus_fraction}" ]]; then
    stage2_policy_args+=(--stage2-cluster-consensus-fraction "${stage2_cluster_consensus_fraction}")
fi
if [[ -n "${stage2_cluster_max_size}" ]]; then
    stage2_policy_args+=(--stage2-cluster-max-size "${stage2_cluster_max_size}")
fi
if [[ -n "${stage2_max_latents_per_cluster}" ]]; then
    stage2_policy_args+=(--stage2-max-latents-per-cluster "${stage2_max_latents_per_cluster}")
fi
if [[ -n "${stage2_cluster_tool_call_limit}" ]]; then
    stage2_policy_args+=(--stage2-cluster-tool-call-limit "${stage2_cluster_tool_call_limit}")
fi
if [[ -n "${stage2_adjudicator_tool_call_limit}" ]]; then
    stage2_policy_args+=(--stage2-adjudicator-tool-call-limit "${stage2_adjudicator_tool_call_limit}")
fi

if (( stage2_managed_orchestrator )); then
    resolved_stage2_vllm_gpus="${stage2_vllm_gpus:-${devices}}"
    stage_mode_args=(
        --stage2-model "${stage2_model}"
        --stage2-vllm-gpus "${resolved_stage2_vllm_gpus}"
        --set "stage2.workers=${resolved_stage2_workers}"
        "${stage2_policy_args[@]}"
    )
    if (( stage2_vllm_servers > 0 )); then
        stage_mode_args+=(--stage2-vllm-servers "${stage2_vllm_servers}")
    fi
    if [[ -n "${stage2_vllm_gpus_per_server}" ]]; then
        stage_mode_args+=(
            --stage2-vllm-gpus-per-server "${stage2_vllm_gpus_per_server}"
        )
    fi
    if [[ -n "${stage2_vllm_rapid_switch_seconds}" ]]; then
        stage_mode_args+=(
            --stage2-vllm-rapid-switch-seconds "${stage2_vllm_rapid_switch_seconds}"
        )
    fi
    if [[ -n "${stage2_vllm_base_port}" ]]; then
        stage_mode_args+=(--stage2-vllm-base-port "${stage2_vllm_base_port}")
    fi
    if [[ -n "${stage2_vllm_internal_port_base}" ]]; then
        stage_mode_args+=(
            --stage2-vllm-internal-port-base "${stage2_vllm_internal_port_base}"
        )
    fi
    if [[ -n "${stage2_vllm_download_dir}" ]]; then
        stage_mode_args+=(--stage2-vllm-download-dir "${stage2_vllm_download_dir}")
    fi
    if [[ -n "${stage2_vllm_extra_args_json}" ]]; then
        stage_mode_args+=(--set "stage2.vllm.extra_args=${stage2_vllm_extra_args_json}")
    fi
    orchestrator_server_description="${stage2_vllm_servers} servers"
    if (( stage2_vllm_servers == 0 )); then
        orchestrator_server_description="auto replicas"
    fi
    stage2_description="managed orchestrator vLLM: ${orchestrator_server_description} on ${resolved_stage2_vllm_gpus} (${resolved_stage2_workers} concurrent requests)"
elif [[ -z "${stage2_endpoint}" ]]; then
    stage_mode_args=(--stage1-only)
    stage2_description="disabled (STAGE2_ENDPOINT is empty)"
else
    stage_mode_args=()
    if (( stage2_only )); then
        stage_mode_args+=(--stage2-only)
    fi
    stage_mode_args+=(
        --stage2-endpoint "${stage2_endpoint}"
        --set "stage2.workers=${resolved_stage2_workers}"
        "${stage2_policy_args[@]}"
    )
    if [[ -n "${stage2_model}" ]]; then
        stage_mode_args+=(--stage2-model "${stage2_model}")
    fi
    if (( stage2_only )); then
        stage2_description="${stage2_endpoint} (${resolved_stage2_workers} concurrent requests; Stage 2-only resume)"
    else
        stage2_description="${stage2_endpoint} (${resolved_stage2_workers} concurrent requests)"
    fi
fi
if (( stage2_managed_extractor )); then
    extractor_server_description="${stage2_extraction_vllm_servers} servers"
    if (( stage2_extraction_vllm_servers == 0 )); then
        extractor_server_description="auto replicas"
    fi
    stage2_description+="; managed extractor vLLM: ${extractor_server_description} on ${stage2_extraction_vllm_gpus} (${stage2_extraction_workers:-${resolved_stage2_workers}} concurrent requests)"
    if (( stage2_managed_orchestrator )); then
        stage2_description+="; models alternate across the GPU union with adaptive configured-split fallback"
    fi
elif [[ -n "${stage2_extraction_endpoint}" && "${stage2_enabled}" == "1" ]]; then
    stage2_description+="; extractor ${stage2_extraction_endpoint} (${stage2_extraction_workers:-${resolved_stage2_workers}} concurrent requests)"
fi

if [[ -n "${stage1_architectures}" ]]; then
    architecture_args=(--architectures "${stage1_architectures}")
    architecture_description="${stage1_architectures}"
else
    architecture_args=()
    architecture_description="legacy enabled set"
fi

echo "Dataset:        ${dataset}"
echo "Output:         ${output_dir}"
echo "Progress:       ${output_dir}/progress.json"
echo "Log:            ${output_dir}/logs/workflow.log"
if (( stage2_only )); then
    echo "CUDA devices:   not required for endpoint-backed Stage 2"
    echo "GPU memory:     ${gpu_summary}"
else
    echo "CUDA devices:   ${devices}"
    echo "GPU memory:     ${gpu_summary}"
fi
echo "CPU budget:     ${worker_count} workers (${cpu_count} available)"
echo "Stage 2:        ${stage2_description}"
echo "Architectures:  ${architecture_description}"
if (( stage2_only )); then
    echo "HTR modeling:   not run during Stage 2-only resume"
else
    echo "HTR modeling:   $([[ "${disable_htr}" == "1" ]] && echo disabled || echo enabled)"
fi

export PYTHONUNBUFFERED=1

exec "${python_bin}" -m oci.inference.research_all_evidence_workflow \
    --dataset "${dataset}" \
    --output-dir "${output_dir}" \
    --unit-id-column patient_id \
    --text-column clinical_text \
    --treatment-column treatment_indicator \
    --outcome-column outcome_indicator \
    --outcome-type binary \
    --clinical-question "For advanced or metastatic NSCLC, identify pretreatment text features that confound assignment to vinorelbine versus gemcitabine or modify their treatment effect." \
    --outer-folds "${outer_folds}" \
    --inner-folds "${inner_folds}" \
    --seed 42 \
    --devices "${devices}" \
    --workers "${worker_count}" \
    --htr-model prajjwal1/bert-tiny \
    --embedding-model Qwen/Qwen3-Embedding-8B \
    --set science.stage1.architecture.multi_model_forest.embedding_contrast.max_chunks=512 \
    --set science.stage1.architecture.htr_max_chunks=512 \
    "${architecture_args[@]}" \
    "${htr_args[@]}" \
    "${stage_mode_args[@]}"
