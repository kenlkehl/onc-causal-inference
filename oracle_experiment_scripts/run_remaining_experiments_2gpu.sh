#!/bin/bash
# Run all 26 remaining experiments split across 2 GPUs
#
# Usage:
#   ./oracle_experiment_scripts/run_remaining_experiments_2gpu.sh
#
# Or with custom dataset/output:
#   DATASET=path/to/data.parquet OUTPUT_DIR=path/to/output ./run_remaining_experiments_2gpu.sh

set -e

# Default paths (can be overridden via environment variables)
DATASET="${DATASET:-../pcori_experiments/explicit_confounder_experiments_1-19-26/dataset_with_extraction.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-../pcori_experiments/explicit_confounder_experiments_1-19-26/remaining_experiments}"
EPOCHS="${EPOCHS:-20}"

echo "=============================================="
echo "Running 26 remaining experiments on 2 GPUs"
echo "=============================================="
echo "Dataset: $DATASET"
echo "Output:  $OUTPUT_DIR"
echo "Epochs:  $EPOCHS"
echo "=============================================="

# GPU 0: First 13 experiments (CNN, BERT, GRU, start of Confounder)
echo "[GPU 0] Starting 13 experiments: CNN, BERT, GRU, Confounder (dragonnet)..."
python oracle_experiment_scripts/run_remaining_experiments.py \
    --dataset "$DATASET" \
    --output-dir "$OUTPUT_DIR" \
    --device cuda:0 \
    --epochs "$EPOCHS" \
    --experiments \
        cnn_causal_forest cnn_uplift cnn_traditional_logreg \
        bert_dragonnet bert_causal_forest bert_uplift bert_traditional_logreg \
        gru_dragonnet gru_rlearner gru_causal_forest gru_uplift gru_traditional_logreg \
        confounder_dragonnet \
    > "${OUTPUT_DIR}/gpu0.log" 2>&1 &

GPU0_PID=$!
echo "[GPU 0] Started with PID $GPU0_PID"

# GPU 1: Last 13 experiments (rest of Confounder, Hier Transformer, Gated MIL, GRU-MIL, GRU-Pool)
echo "[GPU 1] Starting 13 experiments: Confounder (rest), Hier Transformer, Gated MIL, GRU-MIL, GRU-Pool..."
python oracle_experiment_scripts/run_remaining_experiments.py \
    --dataset "$DATASET" \
    --output-dir "$OUTPUT_DIR" \
    --device cuda:1 \
    --epochs "$EPOCHS" \
    --experiments \
        confounder_causal_forest confounder_uplift confounder_traditional_logreg \
        hier_transformer_causal_forest hier_transformer_uplift hier_transformer_traditional_logreg \
        gated_mil_uplift gated_mil_traditional_logreg \
        gru_mil_causal_forest gru_mil_uplift gru_mil_traditional_logreg \
        gru_pool_uplift gru_pool_traditional_logreg \
    > "${OUTPUT_DIR}/gpu1.log" 2>&1 &

GPU1_PID=$!
echo "[GPU 1] Started with PID $GPU1_PID"

echo ""
echo "Both GPUs running in background."
echo "Logs: ${OUTPUT_DIR}/gpu0.log and ${OUTPUT_DIR}/gpu1.log"
echo ""
echo "To monitor progress:"
echo "  tail -f ${OUTPUT_DIR}/gpu0.log"
echo "  tail -f ${OUTPUT_DIR}/gpu1.log"
echo ""
echo "Waiting for both to complete..."

# Wait for both processes
wait $GPU0_PID
GPU0_STATUS=$?
echo "[GPU 0] Finished with status $GPU0_STATUS"

wait $GPU1_PID
GPU1_STATUS=$?
echo "[GPU 1] Finished with status $GPU1_STATUS"

echo ""
echo "=============================================="
echo "All experiments complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=============================================="

# Exit with error if either failed
if [ $GPU0_STATUS -ne 0 ] || [ $GPU1_STATUS -ne 0 ]; then
    echo "WARNING: One or more experiments may have failed. Check logs."
    exit 1
fi
