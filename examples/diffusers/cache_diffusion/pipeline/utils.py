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

from collections import OrderedDict

import numpy as np
import tensorrt as trt
import torch
from cuda import cudart
from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.trt import engine_from_bytes

numpy_to_torch_dtype_dict = {
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}


class Engine:
    def __init__(
        self,
    ):
        self.engine = None
        self.context = None
        self.buffers = OrderedDict()
        self.tensors = OrderedDict()
        self.cuda_graph_instance = None  # cuda graph
        self.has_cross_attention = False

    def __del__(self):
        del self.engine
        del self.context
        del self.buffers
        del self.tensors

    def load(self, engine_path):
        self.engine = engine_from_bytes(bytes_from_path(engine_path))

    def activate(self, reuse_device_memory=None):
        if reuse_device_memory:
            self.context = self.engine.create_execution_context_without_device_memory()  # type: ignore[union-attr]
            self.context.device_memory = reuse_device_memory
        else:
            self.context = self.engine.create_execution_context()  # type: ignore[union-attr]

    def allocate_buffers(self, shape_dict=None, device="cuda", batch_size=1):
        for binding in range(self.engine.num_io_tensors):  # type: ignore[union-attr]
            name = self.engine.get_tensor_name(binding)  # type: ignore[union-attr]
            if shape_dict and name in shape_dict:
                shape = shape_dict[name]
            else:
                shape = self.engine.get_tensor_shape(name)  # type: ignore[union-attr]
                shape = (batch_size * 2, *shape[1:])
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))  # type: ignore[union-attr]
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:  # type: ignore[union-attr]
                self.context.set_input_shape(name, shape)  # type: ignore[union-attr]
            tensor = torch.empty(tuple(shape), dtype=numpy_to_torch_dtype_dict[dtype]).to(
                device=device
            )
            self.tensors[name] = tensor

    def __call__(self, feed_dict, stream, use_cuda_graph=False):
        for name, buf in feed_dict.items():
            self.tensors[name].copy_(buf)

        for name, tensor in self.tensors.items():
            self.context.set_tensor_address(name, tensor.data_ptr())  # type: ignore[union-attr]

        if use_cuda_graph:
            if self.cuda_graph_instance is not None:
                cuassert(cudart.cudaGraphLaunch(self.cuda_graph_instance, stream))
                cuassert(cudart.cudaStreamSynchronize(stream))
            else:
                # do inference before CUDA graph capture
                noerror = self.context.execute_async_v3(stream)  # type: ignore[union-attr]
                if not noerror:
                    raise ValueError("ERROR: inference failed.")
                # capture cuda graph
                cuassert(
                    cudart.cudaStreamBeginCapture(
                        stream, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal
                    )
                )
                self.context.execute_async_v3(stream)  # type: ignore[union-attr]
                self.graph = cuassert(cudart.cudaStreamEndCapture(stream))
                self.cuda_graph_instance = cuassert(cudart.cudaGraphInstantiate(self.graph, 0))
        else:
            noerror = self.context.execute_async_v3(stream)  # type: ignore[union-attr]
            if not noerror:
                raise ValueError("ERROR: inference failed.")

        return self.tensors


def cuassert(cuda_ret):
    err = cuda_ret[0]
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(
            f"CUDA ERROR: {err}, error code reference: "
            "https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__ERROR.html"
        )
    if len(cuda_ret) > 1:
        return cuda_ret[1]
    return None
