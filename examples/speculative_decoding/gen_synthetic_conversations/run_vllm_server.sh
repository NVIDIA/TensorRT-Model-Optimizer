# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Example launch configuration for a vLLM server
# On 8xB200, Llama 3.3 70B runs comfortably with TP=2 at high batch sizes.

# Achieve data parallelism by running multiple vLLM servers on different GPUs.
# CUDA_VISIBLE_DEVICES=0,1 vllm serve meta-llama/Llama-3.3-70B-Instruct --tensor-parallel-size 2 --max-num-batched-tokens 32768 --max-seq-len 8192 --disable-log-requests --max-num-seqs 1024 --port 8000 &
# CUDA_VISIBLE_DEVICES=2,3 vllm serve meta-llama/Llama-3.3-70B-Instruct --tensor-parallel-size 2 --max-num-batched-tokens 32768 --max-seq-len 8192 --disable-log-requests --max-num-seqs 1024 --port 8001 &
# CUDA_VISIBLE_DEVICES=4,5 vllm serve meta-llama/Llama-3.3-70B-Instruct --tensor-parallel-size 2 --max-num-batched-tokens 32768 --max-seq-len 8192 --disable-log-requests --max-num-seqs 1024 --port 8002 &
# CUDA_VISIBLE_DEVICES=6,7 vllm serve meta-llama/Llama-3.3-70B-Instruct --tensor-parallel-size 2 --max-num-batched-tokens 32768 --max-seq-len 8192 --disable-log-requests --max-num-seqs 1024 --port 8003 &

# Alternatively, use vLLM's built-in data parallelism.
# vllm serve meta-llama/Llama-3.3-70B-Instruct --tensor-parallel-size 2 --data-parallel-size 4 --max-num-batched-tokens 32768 --max-seq-len 8192 --disable-log-requests --max-num-seqs 1024 --port 8000

# Default to a small model for testing.
# vllm serve meta-llama/Llama-3.2-1B-Instruct --tensor-parallel-size 1 --data-parallel-size 8 --max-num-batched-tokens 32768 --max-seq-len 8192 --disable-log-requests --max-num-seqs 1024 --port 8000

CUDA_VISIBLE_DEVICES=0 vllm serve meta-llama/Llama-3.2-1B-Instruct --tensor-parallel-size 1 --max-num-batched-tokens 32768 --max-seq-len 8192 --max-num-seqs 1024 --port 8000
# CUDA_VISIBLE_DEVICES=1 vllm serve meta-llama/Llama-3.2-1B-Instruct --tensor-parallel-size 1 --max-num-batched-tokens 32768 --max-seq-len 8192 --disable-log-requests --max-num-seqs 1024 --port 8001 &
# CUDA_VISIBLE_DEVICES=2 vllm serve meta-llama/Llama-3.2-1B-Instruct --tensor-parallel-size 1 --max-num-batched-tokens 32768 --max-seq-len 8192 --disable-log-requests --max-num-seqs 1024 --port 8002 &
# CUDA_VISIBLE_DEVICES=3 vllm serve meta-llama/Llama-3.2-1B-Instruct --tensor-parallel-size 1 --max-num-batched-tokens 32768 --max-seq-len 8192 --disable-log-requests --max-num-seqs 1024 --port 8003 &
# CUDA_VISIBLE_DEVICES=4 vllm serve meta-llama/Llama-3.2-1B-Instruct --tensor-parallel-size 1 --max-num-batched-tokens 32768 --max-seq-len 8192 --disable-log-requests --max-num-seqs 1024 --port 8004 &
# CUDA_VISIBLE_DEVICES=5 vllm serve meta-llama/Llama-3.2-1B-Instruct --tensor-parallel-size 1 --max-num-batched-tokens 32768 --max-seq-len 8192 --disable-log-requests --max-num-seqs 1024 --port 8005 &
# CUDA_VISIBLE_DEVICES=6 vllm serve meta-llama/Llama-3.2-1B-Instruct --tensor-parallel-size 1 --max-num-batched-tokens 32768 --max-seq-len 8192 --disable-log-requests --max-num-seqs 1024 --port 8006 &
# CUDA_VISIBLE_DEVICES=7 vllm serve meta-llama/Llama-3.2-1B-Instruct --tensor-parallel-size 1 --max-num-batched-tokens 32768 --max-seq-len 8192 --disable-log-requests --max-num-seqs 1024 --port 8007 &
