#!/bin/bash

# Default values
datasets=("statpearls" "textbooks")
index_num_chunks=100  # Default value for number of chunks
storage_dir=""
download_workers=1
batch_size=64
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-}"

if [[ -z "$PYTHON_BIN" ]]; then
    if command -v python >/dev/null 2>&1; then
        PYTHON_BIN="python"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="python3"
    else
        echo "Error: Python interpreter not found. Install python3 or set PYTHON_BIN."
        exit 1
    fi
fi

# This script will download all the corpus we need for the FedRAG workflow
# and also prepare the indices using the FAISS library for document retrieval

set -e

usage() {
    echo "Usage: ./data/prepare.sh [--datasets DATASET ...] [--index_num_chunks N] [--storage_dir PATH] [--download_workers N] [--batch_size N]"
    echo ""
    echo "Examples:"
    echo "  ./data/prepare.sh"
    echo "  ./data/prepare.sh --datasets wikipedia --index_num_chunks 0"
    echo "  ./data/prepare.sh --datasets pubmed statpearls textbooks wikipedia --index_num_chunks 0"
    echo "  ./data/prepare.sh --datasets pubmed wikipedia --index_num_chunks 0 --download_workers 4"
    echo "  ./data/prepare.sh --datasets pubmed wikipedia --index_num_chunks 0 --storage_dir '/mnt/gdrive/My Drive/fedrag/corpus'"
        echo "  ./data/prepare.sh --datasets pubmed wikipedia --index_num_chunks 0 --batch_size 128"
    exit 1
}

cd "$REPO_ROOT"

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --datasets)
            shift
            datasets=()  # Clear default datasets when --datasets is provided
            while [[ "$#" -gt 0 && ! "$1" =~ ^-- ]]; do
                datasets+=("$1")
                shift
            done
            ;;
        --index_num_chunks)
            index_num_chunks="$2"
            shift 2
            ;;
        --storage_dir)
            storage_dir="$2"
            shift 2
            ;;
        --download_workers)
            download_workers="$2"
            shift 2
            ;;
        --batch_size)
            batch_size="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done


# Construct the command to pass arguments to Python script
cmd=("$PYTHON_BIN" "-m" "data.prepare")

# Add datasets to the command
cmd+=("--datasets")
for dataset in "${datasets[@]}"; do
    cmd+=("$dataset")
done

# Add number of chunks to consider for the index of each corpus
cmd+=("--index_num_chunks")
cmd+=("$index_num_chunks")

# Optional custom storage directory (supports spaces, e.g. "My Drive")
if [[ -n "$storage_dir" ]]; then
    cmd+=("--storage_dir")
    cmd+=("$storage_dir")
fi

cmd+=("--download_workers")
cmd+=("$download_workers")

cmd+=("--batch_size")
cmd+=("$batch_size")

# Run the Python script
echo "Executing:" "${cmd[@]}"
"${cmd[@]}"
