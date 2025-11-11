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

# Download RULER data files for attention sparsity calibration.
# Downloads Paul Graham Essays URL list and essay content from official sources.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
ESSAYS_DIR="${DATA_DIR}/essays"
RULER_URLS_FILE="${DATA_DIR}/PaulGrahamEssays_URLs.txt"
RULER_URLS_URL="https://raw.githubusercontent.com/NVIDIA/RULER/main/scripts/data/synthetic/json/PaulGrahamEssays_URLs.txt"

echo "Downloading RULER data files for attention sparsity calibration..."

# Create directories
mkdir -p "${DATA_DIR}"
mkdir -p "${ESSAYS_DIR}"

# Step 1: Download URL list
if [ -f "${RULER_URLS_FILE}" ]; then
    echo "URL list already exists: ${RULER_URLS_FILE}"
else
    echo "Downloading URL list..."
    curl -fsSL "${RULER_URLS_URL}" -o "${RULER_URLS_FILE}"
    echo "Downloaded: ${RULER_URLS_FILE}"
fi

# Step 2: Download essay files (only GitHub .txt files)
echo "Downloading essay files..."
DOWNLOAD_COUNT=0
SKIP_COUNT=0

while IFS= read -r url; do
    # Only process GitHub .txt URLs
    if [[ "${url}" == https://github.com*.txt ]]; then
        # Extract filename from URL
        filename=$(basename "${url}")
        filepath="${ESSAYS_DIR}/${filename}"
        
        if [ -f "${filepath}" ]; then
            ((SKIP_COUNT++))
        else
            # Convert GitHub URL to raw URL
            raw_url="${url/github.com/raw.githubusercontent.com}"
            raw_url="${raw_url/\/raw\//\/}"
            
            if curl -fsSL "${raw_url}" -o "${filepath}" 2>/dev/null; then
                ((DOWNLOAD_COUNT++))
            else
                echo "Warning: Failed to download ${filename}"
            fi
        fi
    fi
done < "${RULER_URLS_FILE}"

echo "Downloaded ${DOWNLOAD_COUNT} new essays, ${SKIP_COUNT} already existed."
echo "Done! RULER data files are ready in ${DATA_DIR}"
