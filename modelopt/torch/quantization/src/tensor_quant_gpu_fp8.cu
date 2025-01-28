/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp8.h>
#include <torch/extension.h>

#define BLOCK_SIZE 128

#define AT_DISPATCH_CASE_FLOATING_TYPES(...)                                                       \
  AT_DISPATCH_CASE(at::ScalarType::Double, __VA_ARGS__)                                            \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)                                             \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)                                              \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define AT_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...)                                                \
  AT_DISPATCH_SWITCH(TYPE, NAME, AT_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))

template <typename T> __global__ void fake_e4m3fy_kernel(const T *inputs, size_t n, T *outputs) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (int idx = 4 * tid; idx < 4 * (tid + 1) && idx < n; ++idx) {
    outputs[idx] = static_cast<T>(
        static_cast<float>(static_cast<__nv_fp8_e4m3>(static_cast<float>(inputs[idx]))));
  }
}

template <typename T>
__global__ void fused_fake_e4m3fy_kernel(const T *inputs, size_t n, float *amax,
                                         bool per_block_scaling_factor, size_t blocksize,
                                         float zero_threshold, T *outputs) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (int idx = 4 * tid; idx < 4 * (tid + 1) && idx < n; ++idx) {
    float x = static_cast<float>(inputs[idx]);

    // generate mask for zeroing tiny values
    float x_abs = fabsf(x);
    bool zero_mask = x_abs < zero_threshold;

    // grab the global scaling factor
    size_t amax_idx = (per_block_scaling_factor) ? (idx / blocksize) : 0;

    // compute scale and inverse-scales
    float scale = 448.f / (amax[amax_idx]);
    float inv_scale = 1.f / scale;

    // compute the output
    float output = static_cast<float>(static_cast<__nv_fp8_e4m3>(scale * x)) * inv_scale;

    // zero out small values
    if (zero_mask) {
      output = 0.f;
    }

    outputs[idx] = output;
  }
}

at::Tensor fused_fake_e4m3fy_cuda(at::Tensor inputs, at::Tensor amax, const float zero_threshold) {
  size_t numel = inputs.numel();
  auto outputs = torch::empty_like(inputs);

  bool per_block_scaling_factor = false;
  size_t blocksize = numel;

  int amax_ndim = amax.dim();
  int input_ndim = inputs.dim();

  // 3 options:
  // 1.
  //    inputs[numel], amax[1] -> per-tensor scaling
  // 2.
  //    inputs[numel], amax[numel/num_cols] -> per-row / per-channel scaling
  // 3.
  //    inputs[numel/bs, bs], amax[numel/bs, 1] -> blockwise scaling
  if (amax.numel() == 1) {
    // case 1.
    per_block_scaling_factor = false;
  } else if (amax.numel() > 1 && (amax_ndim > 1 && (amax.size(-1) == amax.numel()))) {
    // case 2.
    per_block_scaling_factor = true;
    blocksize = numel / amax.numel();
  } else {
    throw std::runtime_error("invalid combination of inputs and amax shapes/sizes");
  }

  auto stream = c10::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(inputs.type().scalarType(), "fused_fake_e4m3fy_cuda", [&] {
    fused_fake_e4m3fy_kernel<<<numel / (BLOCK_SIZE * 4) + 1, BLOCK_SIZE, 0, stream>>>(
        inputs.data_ptr<scalar_t>(), numel, amax.data_ptr<float>(), per_block_scaling_factor,
        blocksize, zero_threshold, outputs.data_ptr<scalar_t>());
  });
  return outputs;
}

at::Tensor fake_e4m3fy_cuda(at::Tensor inputs) {
  size_t numel = inputs.numel();
  auto outputs = torch::empty_like(inputs);
  auto stream = c10::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(inputs.type().scalarType(), "fake_e4m3fy_cuda", [&] {
    fake_e4m3fy_kernel<<<numel / (BLOCK_SIZE * 4) + 1, BLOCK_SIZE, 0, stream>>>(
        inputs.data_ptr<scalar_t>(), numel, outputs.data_ptr<scalar_t>());
  });
  return outputs;
}
