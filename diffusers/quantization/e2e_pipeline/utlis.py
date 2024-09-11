# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


from collections import OrderedDict

import numpy as np
import tensorrt as trt
import torch
from cuda import cudart
from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.trt import (
    engine_from_bytes,
)
from tensorrt.tensorrt import DataType

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
    torch.bfloat16: torch.bfloat16,
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
            self.context = self.engine.create_execution_context_without_device_memory()  # type: ignore
            self.context.device_memory = reuse_device_memory
        else:
            self.context = self.engine.create_execution_context()  # type: ignore

    def allocate_buffers(self, shape_dict=None, device="cuda", infer_batch_size: int = 2):
        for binding in range(self.engine.num_io_tensors):  # type: ignore
            name = self.engine.get_tensor_name(binding)  # type: ignore
            if shape_dict and name in shape_dict:
                shape = shape_dict[name]
            else:
                shape = self.engine.get_tensor_shape(name)  # type: ignore
                shape = (infer_batch_size,) + shape[1:]
            _tensor_dtype = self.engine.get_tensor_dtype(name)  # type: ignore
            if _tensor_dtype == DataType.BF16:
                dtype = torch.bfloat16
            else:
                dtype = trt.nptype(_tensor_dtype)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:  # type: ignore
                self.context.set_input_shape(name, shape)  # type: ignore
            tensor = torch.empty(tuple(shape), dtype=numpy_to_torch_dtype_dict[dtype]).to(
                device=device
            )
            self.tensors[name] = tensor

    def __call__(self, feed_dict, stream, use_cuda_graph=False):

        for name, buf in feed_dict.items():
            self.tensors[name].copy_(buf)

        for name, tensor in self.tensors.items():
            self.context.set_tensor_address(name, tensor.data_ptr())  # type: ignore

        if use_cuda_graph:
            if self.cuda_graph_instance is not None:
                cuassert(cudart.cudaGraphLaunch(self.cuda_graph_instance, stream))
                cuassert(cudart.cudaStreamSynchronize(stream))
            else:
                # do inference before CUDA graph capture
                noerror = self.context.execute_async_v3(stream)  # type: ignore
                if not noerror:
                    raise ValueError("ERROR: inference failed.")
                # capture cuda graph
                cuassert(
                    cudart.cudaStreamBeginCapture(
                        stream, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal
                    )
                )
                self.context.execute_async_v3(stream)  # type: ignore
                self.graph = cuassert(cudart.cudaStreamEndCapture(stream))
                self.cuda_graph_instance = cuassert(cudart.cudaGraphInstantiate(self.graph, 0))
        else:
            noerror = self.context.execute_async_v3(stream)  # type: ignore
            if not noerror:
                raise ValueError("ERROR: inference failed.")

        return self.tensors


def cuassert(cuda_ret):
    err = cuda_ret[0]
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(
            f"CUDA ERROR: {err}, error code reference: https://nvidia.github.io/cuda-python/module/cudart.html#cuda.cudart.cudaError_t"
        )
    if len(cuda_ret) > 1:
        return cuda_ret[1]
    return None
