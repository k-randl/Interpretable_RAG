#!/bin/bash
# run_all_musique.sh
set -e

# GPU Allocation
export CUDA_VISIBLE_DEVICES=4,5,6,7
export HF_TOKEN=$(cat hugginface_token.txt)

RETRIEVERS=(
    "Snowflake/snowflake-arctic-embed-l-v2.0"
    "dragon"
)

GENERATORS=(
    "meta-llama/Llama-3.1-8B-Instruct"
    "google/gemma-7b-it"
    "Qwen/Qwen2.5-7B-Instruct"
)

OUTPUT_DIR="results/musique_experiments"

for RETRIEVER in "${RETRIEVERS[@]}"; do
    for GENERATOR in "${GENERATORS[@]}"; do
        echo "=========================================================="
        echo "Starting Experiment: $RETRIEVER + $GENERATOR"
        echo "=========================================================="
        
        python scripts/run_musique_experiment.py \
            --retriever_id "$RETRIEVER" \
            --generator_id "$GENERATOR" \
            --output_path "$OUTPUT_DIR"
            
        # Get simplified names for directories
        RET_NAME=$(basename "$RETRIEVER")
        GEN_NAME=$(basename "$GENERATOR")
        
        # Run Correlation Analysis on the output directory
        echo "Running Correlation Analysis..."
        python analysis/correlation_analysis.py \
            --results_dir "$OUTPUT_DIR/${RET_NAME}_${GEN_NAME}" \
            --ground_truth_path "$OUTPUT_DIR/${RET_NAME}_${GEN_NAME}/ground_truths.csv" \
            --output_path "$OUTPUT_DIR/${RET_NAME}_${GEN_NAME}/correlation_results.csv"
            
    done
done

echo "All Experiments Completed Successfully."
