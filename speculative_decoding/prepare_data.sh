#!/bin/bash
set -e
set -x
set -o pipefail

while [ $# -gt 0 ]; do
  case "$1" in
    --data*)
      if [[ "$1" != *=* ]]; then shift; fi
      DATA="${1#*=}"
      ;;
    --output_path*)
      if [[ "$1" != *=* ]]; then shift; fi
      OUTPUT_PATH="${1#*=}"
      ;;
    --max_token*)
      if [[ "$1" != *=* ]]; then shift; fi
      MAX_TOKEN="${1#*=}"
      ;;
    *)
      >&2 printf "Error: Invalid argument\n"
      exit 1
      ;;
  esac
  shift
done

OUTPUT_DIR="$(dirname "${OUTPUT_PATH}")"
mkdir -p $OUTPUT_DIR

CMD="python vllm_generate.py --data_path $DATA --output_path $OUTPUT_PATH --num_threads 8 --max_tokens $MAX_TOKEN --temperature 0.5 --chat"

sh -c "$CMD"
