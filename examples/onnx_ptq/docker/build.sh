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

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Default values
IMAGE_NAME="modelopt_onnx_examples:latest"
DOCKERFILE_PATH="examples/onnx_ptq/docker/Dockerfile"

# Function to show usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
  -t, --tag IMAGE_NAME   Docker image name (default: $IMAGE_NAME)
  -h, --help             Show this help message

This script automatically detects whether you're running from:
  • modelopt/ root directory
  • modelopt/examples/onnx_ptq/ directory

and builds the Docker image accordingly.
EOF
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--tag)
            [[ -n "${2:-}" ]] || { echo "Error: --tag requires a value"; exit 1; }
            IMAGE_NAME="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Error: Unknown option '$1'"
            usage
            ;;
    esac
done

# Function to find modelopt root directory
find_modelopt_root() {
    local current_dir="$PWD"

    # Check current directory first
    if [[ -f "setup.py" && -f "pyproject.toml" && -d "modelopt" ]]; then
        echo "$current_dir"
        return 0
    fi

    # Check parent directories (up to 3 levels)
    for i in {1..3}; do
        local parent_dir
        parent_dir=$(dirname "$current_dir")
        [[ "$parent_dir" == "$current_dir" ]] && break  # Reached filesystem root

        if [[ -f "$parent_dir/setup.py" && -f "$parent_dir/pyproject.toml" && -d "$parent_dir/modelopt" ]]; then
            echo "$parent_dir"
            return 0
        fi
        current_dir="$parent_dir"
    done

    return 1
}

# Find modelopt root directory
echo "🔍 Locating modelopt root directory..."
if ROOT_DIR=$(find_modelopt_root); then
    echo "✅ Found modelopt root: $ROOT_DIR"
    cd "$ROOT_DIR"
else
    cat << EOF
❌ Error: Cannot locate modelopt root directory.

Expected structure:
  modelopt/
  ├── setup.py
  ├── pyproject.toml
  ├── modelopt/
  └── examples/onnx_ptq/docker/

Please run this script from within the modelopt repository.
EOF
    exit 1
fi

# Validate that Dockerfile exists
if [[ ! -f "$DOCKERFILE_PATH" ]]; then
    echo "❌ Error: Dockerfile not found at $DOCKERFILE_PATH"
    exit 1
fi

# Build Docker image
echo "🐳 Building Docker image..."
echo "  • Image name: $IMAGE_NAME"
echo "  • Build context: $(pwd)"
echo "  • Dockerfile: $DOCKERFILE_PATH"
echo

docker build \
    --file "$DOCKERFILE_PATH" \
    --tag "$IMAGE_NAME" \
    . \
    "$@"

echo
echo "✅ Docker image built successfully: $IMAGE_NAME"
echo
echo "🚀 To run the container:"
echo "  docker run --user 0:0 -it --gpus all --shm-size=2g \\"
echo "    -v /path/to/ImageNet/dataset:/workspace/imagenet \\"
echo "    $IMAGE_NAME"
