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

from typing import List, Tuple

import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import tensorrt as trt
import torch

TRT_DYNAMIC_DIM = -1


def get_engine_bytes(engine: trt.tensorrt.ICudaEngine) -> bytes:
    """Return serialized TensorRT engine bytes."""
    return bytearray(engine.serialize())


class HostDeviceMem(object):
    """Simple helper data class to store Host and Device memory."""

    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(
    engine: trt.ICudaEngine, batch_size: int
) -> Tuple[List[HostDeviceMem], List[HostDeviceMem], List[pycuda._driver.DeviceAllocation]]:
    """Function to allocate buffers and bindings for TensorRT inference.

    Args:
        engine: De-serialized TensorRT engine.
        batch_size: Batch size to be used during inference.

    Returns:
        input_buffers: List of host and device buffer of inputs.
        output_buffers: List of host and device buffer of outputs.
        dbindings: List of addresses of memory allocated in device.
    """
    input_buffers = []
    output_buffers = []
    dbindings = []

    for tensor_name in engine:
        binding_shape = engine.get_tensor_shape(tensor_name)
        if binding_shape[0] == TRT_DYNAMIC_DIM:  # dynamic shape
            size = batch_size * abs(trt.volume(binding_shape))
        else:
            size = abs(trt.volume(binding_shape))
        dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))

        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        # Append the device buffer to device bindings
        dbindings.append(int(device_mem))

        # Append to the appropriate list (input/output)
        if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
            input_buffers.append(HostDeviceMem(host_mem, device_mem))
        else:
            output_buffers.append(HostDeviceMem(host_mem, device_mem))

    return input_buffers, output_buffers, dbindings


def inference(
    context: pycuda._driver.Context,
    dbindings: List[pycuda._driver.DeviceAllocation],
    input_buffers: List[HostDeviceMem],
    output_buffers: List[HostDeviceMem],
    batch_size: int,
    inputs_batch: torch.Tensor,
) -> torch.Tensor:
    try:
        # Load images in Host (flatten and copy to page-locked buffer in Host)
        data = inputs_batch.numpy().astype(np.float32).ravel()
        pagelocked_buffer = input_buffers[0].host
        np.copyto(pagelocked_buffer, data)
    except RuntimeError:
        raise RuntimeError("Failed to load images in host.")

    # Transfer input data from Host to Device (GPU)
    inp = input_buffers[0]
    cuda.memcpy_htod(inp.device, inp.host)

    # Run inference
    context.execute_v2(dbindings)

    # Transfer predictions back to Host from GPU
    out = output_buffers[0]
    cuda.memcpy_dtoh(out.host, out.device)

    # Split 1-D output of length N*labels into 2-D array of (N, labels)
    return torch.from_numpy(np.array(np.split(np.array(out.host), batch_size)))


def create_context(
    engine_path: str, batch_size: int
) -> Tuple[trt.ICudaEngine, pycuda._driver.Context]:
    def override_shape(shape: tuple) -> tuple:
        """Overrides the batch dimension if dynamic engine is provided."""
        if TRT_DYNAMIC_DIM in shape:
            shape = tuple([batch_size if dim == TRT_DYNAMIC_DIM else dim for dim in shape])
        return shape

    # Open engine and initialize runtime
    with open(engine_path, "rb") as f, trt.Runtime(trt.Logger(trt.Logger.ERROR)) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

        # Contexts are used to perform inference
        with engine.create_execution_context() as context:
            # Resolves dynamic shapes in the context
            for tensor_name in engine:
                binding_shape = engine.get_tensor_shape(tensor_name)
                if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                    binding_shape = override_shape(binding_shape)
                    context.set_input_shape(tensor_name, binding_shape)

            return engine, context
