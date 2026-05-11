#!/bin/bash
set -e

export HF_TOKEN=$(cat hugginface_token.txt)
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUTPUT_DIR="results/qampari_snowflake"
DATASET="qampari"
RETRIEVER="Snowflake/snowflake-arctic-embed-l-v2.0"
K=5
BATCH_SIZE=32
NUM_QUERIES=100

GENERATORS=(
    "meta-llama/Llama-3.1-8B-Instruct"
    "google/gemma-3-12b-it"
    "Qwen/Qwen2.5-7B-Instruct"
)

echo "=========================================================="
echo "Starting QAMPARI Snowflake Arctic Embed Pipeline"
echo "Retriever: $RETRIEVER | Queries: $NUM_QUERIES | K: $K"
echo "=========================================================="

for GENERATOR in "${GENERATORS[@]}"; do
    echo ""
    echo "----------------------------------------------------------"
    echo "Generator: $GENERATOR"
    echo "----------------------------------------------------------"

    python scripts/run_intertwined_analysis.py \
        --dataset "$DATASET" \
        --retriever_id "$RETRIEVER" \
        --generator_id "$GENERATOR" \
        --output_path "$OUTPUT_DIR" \
        --k "$K" \
        --batch_size "$BATCH_SIZE" \
        --num_queries "$NUM_QUERIES" \
        --system_prompt "qampari"

    echo ">>> Finished $GENERATOR"
done

echo ""
echo "=========================================================="
echo "All Snowflake experiments completed. Results in $OUTPUT_DIR"
echo "=========================================================="
