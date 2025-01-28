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
#include <cuda_fp8.h>
#include <torch/extension.h>

at::Tensor fake_e4m3fy_cuda(at::Tensor inputs);
at::Tensor fused_fake_e4m3fy_cuda(at::Tensor inputs, at::Tensor amax, const float zero_threshold);

at::Tensor fake_e4m3fy(at::Tensor inputs) {
  if (inputs.is_cuda()) {
    return fake_e4m3fy_cuda(inputs.contiguous());
  } else {
    TORCH_CHECK(inputs.dtype() == at::ScalarType::Float);
    TORCH_CHECK(inputs.is_contiguous());
    auto out = at::zeros_like(inputs);
    for (int i = 0; i < inputs.numel(); ++i) {
      out.data_ptr<float>()[i] =
          static_cast<float>(static_cast<__nv_fp8_e4m3>(inputs.data_ptr<float>()[i]));
    }
    return out;
  }
}

at::Tensor fused_fake_e4m3fy(at::Tensor inputs, at::Tensor amax, const float zero_threshold) {
  return fused_fake_e4m3fy_cuda(inputs.contiguous(), amax, zero_threshold);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fake_e4m3fy", &fake_e4m3fy, "Reduce precision to E4M3", py::arg("inputs"));
  m.def("fused_fake_e4m3fy", &fused_fake_e4m3fy, "Reduce precision to E4M3 (fused)",
        py::arg("inputs"), py::arg("amax"), py::arg("zero_threshold"));
}
