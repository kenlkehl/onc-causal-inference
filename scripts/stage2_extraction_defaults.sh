#!/usr/bin/env bash

# Source inside a fresh-run guard, with the launcher's legacy model as $1.
# Saved runs retain their serialized extraction backend and serving settings.
export STAGE2_DECISION_EXTRACTION="${STAGE2_DECISION_EXTRACTION:-1}"
case "$STAGE2_DECISION_EXTRACTION" in
    1)
        export STAGE2_EXTRACTION_MODEL="${STAGE2_EXTRACTION_MODEL:-crh225/plumb-4b}"
        # Stage 2 adds the next-token readout conversion arguments itself.
        plumb_vllm_args='["--revision","24f7bf77e7ee258a2d158c61ea2dce2b60321010","--gpu-memory-utilization","0.28","--max-model-len","3072","--max-num-seqs","8","--max-num-batched-tokens","4096","--enforce-eager","--dtype","bfloat16"]'
        export STAGE2_EXTRACTION_VLLM_EXTRA_ARGS_JSON="${STAGE2_EXTRACTION_VLLM_EXTRA_ARGS_JSON:-${plumb_vllm_args}}"
        unset plumb_vllm_args
        ;;
    0)
        export STAGE2_EXTRACTION_MODEL="${STAGE2_EXTRACTION_MODEL:-$1}"
        ;;
    *) echo "STAGE2_DECISION_EXTRACTION must be 0 or 1." >&2; exit 2 ;;
esac
