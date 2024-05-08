/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef CUDA_DRIVER_WRAPPER_H
#define CUDA_DRIVER_WRAPPER_H

#include <cstdint>
#include <cstdio>
#include <cuda.h>

#define cuErrCheck(stat, wrap)                                                                     \
  { nvinfer1::cuErrCheck_((stat), wrap, __FILE__, __LINE__); }

namespace nvinfer1 {
class CUDADriverWrapper {
public:
  CUDADriverWrapper();

  ~CUDADriverWrapper();

  // Delete default copy constructor and copy assignment constructor
  CUDADriverWrapper(const CUDADriverWrapper &) = delete;
  CUDADriverWrapper &operator=(const CUDADriverWrapper &) = delete;

  CUresult cuGetErrorName(CUresult error, const char **pStr) const;

  CUresult cuFuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib, int value) const;

  CUresult cuLinkComplete(CUlinkState state, void **cubinOut, size_t *sizeOut) const;

  CUresult cuModuleUnload(CUmodule hmod) const;

  CUresult cuLinkDestroy(CUlinkState state) const;

  CUresult cuModuleLoadData(CUmodule *module, const void *image) const;

  CUresult cuLinkCreate(uint32_t numOptions, CUjit_option *options, void **optionValues,
                        CUlinkState *stateOut) const;

  CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name) const;

  CUresult cuLinkAddFile(CUlinkState state, CUjitInputType type, const char *path,
                         uint32_t numOptions, CUjit_option *options, void **optionValues) const;

  CUresult cuLinkAddData(CUlinkState state, CUjitInputType type, void *data, size_t size,
                         const char *name, uint32_t numOptions, CUjit_option *options,
                         void **optionValues) const;

  CUresult cuLaunchCooperativeKernel(CUfunction f, uint32_t gridDimX, uint32_t gridDimY,
                                     uint32_t gridDimZ, uint32_t blockDimX, uint32_t blockDimY,
                                     uint32_t blockDimZ, uint32_t sharedMemBytes, CUstream hStream,
                                     void **kernelParams) const;

  CUresult cuLaunchKernel(CUfunction f, uint32_t gridDimX, uint32_t gridDimY, uint32_t gridDimZ,
                          uint32_t blockDimX, uint32_t blockDimY, uint32_t blockDimZ,
                          uint32_t sharedMemBytes, CUstream hStream, void **kernelParams,
                          void **extra) const;

private:
  void *handle;
  CUresult (*_cuGetErrorName)(CUresult, const char **);
  CUresult (*_cuFuncSetAttribute)(CUfunction, CUfunction_attribute, int);
  CUresult (*_cuLinkComplete)(CUlinkState, void **, size_t *);
  CUresult (*_cuModuleUnload)(CUmodule);
  CUresult (*_cuLinkDestroy)(CUlinkState);
  CUresult (*_cuLinkCreate)(unsigned int, CUjit_option *, void **, CUlinkState *);
  CUresult (*_cuModuleLoadData)(CUmodule *, const void *);
  CUresult (*_cuModuleGetFunction)(CUfunction *, CUmodule, const char *);
  CUresult (*_cuLinkAddFile)(CUlinkState, CUjitInputType, const char *, unsigned int,
                             CUjit_option *, void **);
  CUresult (*_cuLinkAddData)(CUlinkState, CUjitInputType, void *, size_t, const char *,
                             unsigned int, CUjit_option *, void **);
  CUresult (*_cuLaunchCooperativeKernel)(CUfunction, unsigned int, unsigned int, unsigned int,
                                         unsigned int, unsigned int, unsigned int, unsigned int,
                                         CUstream, void **);
  CUresult (*_cuLaunchKernel)(CUfunction f, uint32_t gridDimX, uint32_t gridDimY, uint32_t gridDimZ,
                              uint32_t blockDimX, uint32_t blockDimY, uint32_t blockDimZ,
                              uint32_t sharedMemBytes, CUstream hStream, void **kernelParams,
                              void **extra);
};

inline void cuErrCheck_(CUresult stat, const CUDADriverWrapper &wrap, const char *file, int line) {
  if (stat != CUDA_SUCCESS) {
    const char *msg = nullptr;
    wrap.cuGetErrorName(stat, &msg);
    fprintf(stderr, "CUDA Error: %s %s %d\n", msg, file, line);
  }
}

} // namespace nvinfer1

#endif // CUDA_DRIVER_WRAPPER_H
