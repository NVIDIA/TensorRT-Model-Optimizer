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

# Check if enough arguments are provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <HF_CKPT_DIR>"
    exit 1
fi

# Set directories based on user input
HF_CKPT_DIR="$1"
echo "Using Hugging Face checkpoint directory: $HF_CKPT_DIR"

# Deploy with vLLM and run an example
echo "Deploying the quantized checkpoint with vLLM..."

# Attempt to grant write permission, if necessary
TARGET_DIR="."
if [ ! -w "$TARGET_DIR" ]; then
    echo "Trying to grant write permission to $TARGET_DIR..."
    chmod u+w "$TARGET_DIR" || {
        echo "Error: Could not grant write permission to $TARGET_DIR. Please check permissions or run the script in a writable directory."
        exit 1
    }
fi

cat <<EOF > "$TARGET_DIR/vllm_deploy.py"
from vllm import LLM, SamplingParams

model_id = "$HF_CKPT_DIR"
sampling_params = SamplingParams(temperature=0.8, top_p=0.9)

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

llm = LLM(model=model_id, quantization="modelopt")
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
EOF

python "$TARGET_DIR/vllm_deploy.py" || {
    echo "Error during vLLM deployment."
    exit 1
}

echo "Script completed successfully."
