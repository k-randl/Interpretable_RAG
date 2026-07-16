#!/bin/bash
set -euo pipefail

# Usage: ./run_pipeline.sh [--devices DEVICE_IDS]
# Example: ./run_pipeline.sh --devices 0,1,2

DEVICES=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --devices) DEVICES="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

DEVICE_ARGS=()
[[ -n "$DEVICES" ]] && DEVICE_ARGS=(--devices "$DEVICES")

cd "$(dirname "${BASH_SOURCE[0]}")"

python3 compute_warg_popqa.py --retriever dragon    --generator llama8B    "${DEVICE_ARGS[@]}"
python3 compute_warg_popqa.py --retriever snowflake --generator llama8B    "${DEVICE_ARGS[@]}"
python3 compute_warg_popqa.py --retriever dragon    --generator qwen7B     "${DEVICE_ARGS[@]}"
python3 compute_warg_popqa.py --retriever dragon    --generator gemma3_12B "${DEVICE_ARGS[@]}"
python3 compute_warg_popqa.py --retriever snowflake --generator qwen7B     "${DEVICE_ARGS[@]}"
python3 compute_warg_popqa.py --retriever snowflake --generator gemma3_12B "${DEVICE_ARGS[@]}"

python3 explain_popqa.py --retriever dragon    --generator llama8B    "${DEVICE_ARGS[@]}"
python3 explain_popqa.py --retriever snowflake --generator llama8B    "${DEVICE_ARGS[@]}"
python3 explain_popqa.py --retriever dragon    --generator qwen7B     "${DEVICE_ARGS[@]}"
python3 explain_popqa.py --retriever dragon    --generator gemma3_12B "${DEVICE_ARGS[@]}"
python3 explain_popqa.py --retriever snowflake --generator qwen7B     "${DEVICE_ARGS[@]}"
python3 explain_popqa.py --retriever snowflake --generator gemma3_12B "${DEVICE_ARGS[@]}"

python3 perturb_popqa.py --retriever dragon    --generator llama8B    "${DEVICE_ARGS[@]}"
python3 perturb_popqa.py --retriever snowflake --generator llama8B    "${DEVICE_ARGS[@]}"
python3 perturb_popqa.py --retriever dragon    --generator qwen7B     "${DEVICE_ARGS[@]}"
python3 perturb_popqa.py --retriever dragon    --generator gemma3_12B "${DEVICE_ARGS[@]}"
python3 perturb_popqa.py --retriever snowflake --generator qwen7B     "${DEVICE_ARGS[@]}"
python3 perturb_popqa.py --retriever snowflake --generator gemma3_12B "${DEVICE_ARGS[@]}"

python3 explain_msmarco.py --retriever dragon    --generator llama8B    "${DEVICE_ARGS[@]}"
python3 explain_msmarco.py --retriever snowflake --generator llama8B    "${DEVICE_ARGS[@]}"
python3 explain_msmarco.py --retriever dragon    --generator qwen7B     "${DEVICE_ARGS[@]}"
python3 explain_msmarco.py --retriever snowflake --generator qwen7B     "${DEVICE_ARGS[@]}"
python3 explain_msmarco.py --retriever dragon    --generator gemma3_12B "${DEVICE_ARGS[@]}"
python3 explain_msmarco.py --retriever snowflake --generator gemma3_12B "${DEVICE_ARGS[@]}"