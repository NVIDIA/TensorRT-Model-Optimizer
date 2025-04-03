#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
      >&2 printf "Error: Invalid argument ${1#*=}\n"
      exit 1
      ;;
  esac
  shift
done

OUTPUT_DIR="$(dirname "${OUTPUT_PATH}")"
mkdir -p $OUTPUT_DIR

CMD="python vllm_generate.py --data_path $DATA --output_path $OUTPUT_PATH --num_threads 8 --max_tokens $MAX_TOKEN --temperature 0.0 --chat"

sh -c "$CMD"
