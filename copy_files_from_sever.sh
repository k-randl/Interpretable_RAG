#!/bin/bash

# This script copies specific result files from the remote server to your local machine,
# preserving the directory structure.

# --- Configuration ---
SERVER="francomaria.nardini@sobigdatadgx.unipi.it"
REMOTE_BASE_DIR="/home/francomaria.nardini/raid/guidorocchietti/code/Interpretable_RAG"
LOCAL_BASE_DIR="." # Assumes you run this script from the project's root directory

# --- Files to Copy ---
# An array containing the relative paths of all files to be copied.
FILES_TO_COPY=(
    "results/generation/efra_chunks_17_11_2025/original/Llama-3.1-8B-Instruct_qid_1.pkl"
    "results/generation/trec19_19_11_2025/original/Llama-3.1-8B-Instruct_qid_31_1.pkl"
    "results/generation/trec19_19_11_2025/original/Llama-3.1-8B-Instruct_qid_31_2.pkl"
    "results/retrieval/trec19/query_31_1.pkl"
    "results/retrieval/trec19/query_31_2.pkl"
    "results/retrieval/efra_06_11/query_0.pkl"
    "results/retrieval/efra_06_11/query_1.pkl"
)

# --- Copy Logic ---
# Loop through each file in the FILES_TO_COPY array.
for file in "${FILES_TO_COPY[@]}"; do
    REMOTE_PATH="$REMOTE_BASE_DIR/$file"
    LOCAL_PATH="$LOCAL_BASE_DIR/$file"
    LOCAL_DIR=$(dirname "$LOCAL_PATH")

    echo "Preparing to copy $file..."

    # Create the local directory structure if it doesn't already exist.
    # The '-p' flag ensures that parent directories are created as needed.
    echo "Ensuring local directory exists: $LOCAL_DIR"
    mkdir -p "$LOCAL_DIR"

    # Copy the file from the remote server to the local path using scp.
    echo "Copying from ${SERVER}:${REMOTE_PATH} to ${LOCAL_PATH}"
    scp "${SERVER}:${REMOTE_PATH}" "${LOCAL_PATH}"

    # Check the exit status of the scp command to see if it was successful.
    if [ $? -eq 0 ]; then
        echo "Successfully copied $file"
    else
        echo "Error copying $file. Please check the connection and paths."
    fi
    echo # Add a blank line for better readability
done

echo "---"
echo "All copy operations complete."
