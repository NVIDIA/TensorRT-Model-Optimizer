#!/bin/bash
set -e
set -x
set -o pipefail

while [ $# -gt 0 ]; do
  case "$1" in
    --model*)
      if [[ "$1" != *=* ]]; then shift; fi
      MODEL="${1#*=}"
      ;;
    --data*)
      if [[ "$1" != *=* ]]; then shift; fi
      DATA="${1#*=}"
      ;;
    --output_path*)
      if [[ "$1" != *=* ]]; then shift; fi
      OUTPUT_PATH="${1#*=}"
      ;;
    --server*)
      if [[ "$1" != *=* ]]; then shift; fi
      SERVER="${1#*=}"
      ;;
    *)
      >&2 printf "Error: Invalid argument\n"
      exit 1
      ;;
  esac
  shift
done


MODEL=${MODEL:-"TinyLlama/TinyLlama-1.1B-Chat-v1.0"}
SERVER=${SERVER:-False}

OUTPUT_DIR="$(dirname "${OUTPUT_PATH}")"
mkdir -p $OUTPUT_DIR

echo "Installing VLLM and openai"
pip install vllm>=0.4.2 openai==1.29.0

if [ "$SERVER" = True ] ; then
  CMD="python -m vllm.entrypoints.openai.api_server --model $MODEL --api-key token-abc123 --port 8000  --tensor-parallel-size 1"
else
  CMD="python vllm_generate.py --data_path $DATA --output_path $OUTPUT_PATH --num_threads 8 --max_tokens 1000 --temperature 0.5 --chat"
fi

sh -c "$CMD"
