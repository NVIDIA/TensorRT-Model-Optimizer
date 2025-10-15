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

import platform

import pytest
import torch

try:
    import pynvml

    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


# GPU Architecture mapping table for better readability
# Format: SM version range -> Architecture name
GPU_ARCH_TABLE = [
    # SM Version, Architecture Name
    (100, float("inf"), "Blackwell"),  # SM 10.0+
    (90, 99, "Hopper"),  # SM 9.0-9.9
    (89, 89, "Ada"),  # SM 8.9
    (80, 88, "Ampere"),  # SM 8.0-8.8
    (70, 79, "Volta"),  # SM 7.0-7.9
]

# Define GPU dtypes with clear names
GPU_DTYPES = [
    "int8",      # INT8 quantization (for onnx_ptq)
    "int4",      # INT4 quantization (for onnx_ptq)
    "int8_sq",   # INT8 sparse quantization
    "int4_awq",  # INT4 AWQ quantization
    "w4a8_awq",  # Weight-4bit Activation-8bit AWQ
    "fp8",       # FP8 format
    "nvfp4",     # NVIDIA FP4 format
    "nvfp4_awq", # NVIDIA FP4 AWQ quantization
    "bf16",      # BFloat16 format
    "fp16",      # Float16 format
]

# 1 = supported, 0 = not supported (common/mainstream software stack and hardware path)
GPU_DTYPE_MATRIX = {
    # Arch:     [int8, int4, int8_sq, int4_awq, w4a8_awq, fp8, nvfp4, nvfp4_awq, bf16, fp16]
    "Ampere":    [1,    1,    1,      1,       0,        0,   0,     0,        1,    1],
    "Ada":       [1,    1,    1,      1,       1,        1,   0,     0,        1,    1],
    "Hopper":    [1,    1,    1,      1,       1,        1,   0,     0,        1,    1],
    "Blackwell": [1,    1,    1,      1,       1,        1,   1,     1,        1,    1],
    "Volta":     [0,    0,    0,      0,       0,        0,   0,     0,        0,    1],
}


# Create DTYPE support dictionary for compatibility
GPU_DTYPE_SUPPORT = {}
for arch, supports in GPU_DTYPE_MATRIX.items():
    GPU_DTYPE_SUPPORT[arch] = {}
    for idx, dtype in enumerate(GPU_DTYPES):
        GPU_DTYPE_SUPPORT[arch][dtype] = bool(supports[idx])


def get_gpu_sm_version(gpu_index=0):
    """
    Get the SM version of the GPU.

    Args:
        gpu_index (int): GPU device index (default: 0)

    Returns:
        int: SM version as an integer (e.g., 80 for SM 8.0) or None if not available

    Raises:
        ValueError: if gpu_index is invalid or SM version calculation results in invalid value
    """
    # validate gpu_index
    if not isinstance(gpu_index, int) or gpu_index < 0:
        raise ValueError(f"gpu_index must be a non-negative integer, got {gpu_index}")

    # try using torch first (most reliable when available)
    if torch.cuda.is_available():
        try:
            if gpu_index >= torch.cuda.device_count():
                return None
            major, minor = torch.cuda.get_device_capability(gpu_index)

            # validate SM version calculation to avoid confusion
            if minor >= 10:
                raise ValueError(f"Invalid SM version: {major}.{minor}, minor version >= 10")

            sm_version = major * 10 + minor
            return sm_version
        except (RuntimeError, ValueError):
            # only catch specific exceptions
            pass

    # fall back to pynvml if torch doesn't work
    if PYNVML_AVAILABLE:
        nvml_initialized = False
        try:
            pynvml.nvmlInit()
            nvml_initialized = True

            device_count = pynvml.nvmlDeviceGetCount()
            if gpu_index >= device_count:
                return None

            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            compute_capability = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            major, minor = compute_capability[0], compute_capability[1]

            # validate SM version calculation to avoid confusion
            if minor >= 10:
                raise ValueError(f"Invalid SM version: {major}.{minor}, minor version >= 10")

            sm_version = major * 10 + minor
            return sm_version
        except (pynvml.NVMLError, RuntimeError, ValueError):
            # only catch specific exceptions
            return None
        finally:
            # ensure pynvml is properly shut down even if error occurs
            if nvml_initialized:
                pynvml.nvmlShutdown()

    return None


def get_gpu_name(gpu_index=0):
    """
    Get the GPU name based on the GPU index.
    """
    try:
        if gpu_index >= torch.cuda.device_count():
            return "Unknown device"
        return torch.cuda.get_device_name(gpu_index)
    except Exception:
        return "Unknown device"


def get_gpu_arch(gpu_index=0):
    """
    Get the GPU architecture based on SM version using the lookup table.

    Args:
        gpu_index (int): GPU device index (default: 0)

    Returns:
        str: GPU architecture name or "Unknown" if not available or not recognized
    """
    sm_version = get_gpu_sm_version(gpu_index)
    if sm_version is None:
        return "Unknown"

    # look up architecture in the table
    for min_sm, max_sm, arch_name in GPU_ARCH_TABLE:
        if min_sm <= sm_version <= max_sm:
            return arch_name

    # SM version not in any known range
    return "Unknown"


def get_cpu_arch():
    """
    Get the CPU architecture (x86 or aarch64).

    Returns:
        str: CPU architecture ('x86', 'aarch64', or 'unknown')
    """
    arch = platform.machine().lower()
    if "x86" in arch or "amd64" in arch:
        return "x86"
    elif "aarch64" in arch or "arm64" in arch:
        return "aarch64"
    else:
        return "unknown"


def skip_if_dtype_unsupported_by_arch(need_dtype, need_cpu_arch=None, gpu_index=0):
    """
    Function to skip tests if the specified data type is not supported on the current GPU architecture.
    This function is meant to be called inside test functions to conditionally skip tests.

    Args:
        need_dtype (str): Data type to check for support (e.g., "fp8", "int4_awq", "nvfp4")
        need_cpu_arch (str, optional): Platform architecture to check against. If specified,
                                      will check if current CPU architecture matches.
        gpu_index (int): GPU device index (default: 0)

    Returns:
        None: If the test should continue
        pytest.skip: Skips the test if the required DTYPEs are not supported
    """
    # 1. Log debug information about the check being performed
    print(f"[SKIP DEBUG]: Checking support for dtype '{need_dtype}' on GPU {gpu_index}")

    # 2. Validate input: check if the requested dtype is in our supported DTYPEs list
    if need_dtype not in GPU_DTYPES:
        raise ValueError(f"Unknown dtype '{need_dtype}'. Supported dtypes: {GPU_DTYPES}")

    # 3. Check CPU architecture compatibility if a specific CPU arch is required
    if need_cpu_arch:
        current_cpu_arch = get_cpu_arch()
        if current_cpu_arch != need_cpu_arch:
            # 3a. If CPU architecture doesn't match, skip the test
            gpu_name = get_gpu_name(gpu_index)
            skip_msg = (
                f"SKIP -- [CPU ARCH MISMATCH] "
                f"Required: {need_cpu_arch}, Current: {current_cpu_arch} ({gpu_name})"
            )
            pytest.skip(skip_msg)

    # 4. Get the current GPU architecture
    current_arch = get_gpu_arch(gpu_index)

    # 5. Determine if the DTYPE is supported on current architecture
    is_supported = False
    if current_arch in GPU_DTYPE_SUPPORT:
        # 5a. Look up the DTYPE support in our compatibility matrix
        is_supported = GPU_DTYPE_SUPPORT[current_arch].get(need_dtype, False)
        print(f"[SKIP DEBUG]: DTYPE '{need_dtype}' supported on {current_arch}: {is_supported}")
    else:
        # 5b. Log if the architecture isn't in our support matrix
        print(f"[SKIP DEBUG]: Architecture {current_arch} not found in GPU_DTYPE_SUPPORT")

    # 6. If the DTYPE is not supported, skip the test
    if not is_supported:
        pytest.skip(
            f"SKIP -- [UNSUPPORTED DTYPE] {need_dtype} not available on {current_arch} GPU ({get_gpu_name(gpu_index)})"
        )
