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

#ifndef TRT_PLUGIN_H
#define TRT_PLUGIN_H
#include "NvInferPlugin.h"
#include "common/checkMacrosPlugin.h"
#include <cstring>
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <string>

typedef enum {
  STATUS_SUCCESS = 0,
  STATUS_FAILURE = 1,
  STATUS_BAD_PARAM = 2,
  STATUS_NOT_SUPPORTED = 3,
  STATUS_NOT_INITIALIZED = 4
} pluginStatus_t;

namespace nvinfer1 {

namespace pluginInternal {

class BasePlugin : public IPluginV2 {
protected:
  void setPluginNamespace(const char *libNamespace) noexcept override { mNamespace = libNamespace; }

  const char *getPluginNamespace() const noexcept override { return mNamespace.c_str(); }

  std::string mNamespace;
};

class BaseCreator : public IPluginCreator {
public:
  void setPluginNamespace(const char *libNamespace) noexcept override { mNamespace = libNamespace; }

  const char *getPluginNamespace() const noexcept override { return mNamespace.c_str(); }

protected:
  std::string mNamespace;
};

} // namespace pluginInternal

namespace plugin {

// Write values into buffer
template <typename T> void write(char *&buffer, const T &val) {
  std::memcpy(buffer, &val, sizeof(T));
  buffer += sizeof(T);
}

// Read values from buffer
template <typename T> T read(const char *&buffer) {
  T val{};
  std::memcpy(&val, buffer, sizeof(T));
  buffer += sizeof(T);
  return val;
}

inline int32_t getTrtSMVersionDec(int32_t smVersion) {
  // Treat SM89 as SM86 temporarily.
  return (smVersion == 89) ? 86 : smVersion;
}

inline int32_t getTrtSMVersionDec(int32_t majorVersion, int32_t minorVersion) {
  return getTrtSMVersionDec(majorVersion * 10 + minorVersion);
}

// Check that all required field names are present in the PluginFieldCollection.
// If not, throw a PluginError with a message stating which fields are missing.
void validateRequiredAttributesExist(std::set<std::string> requiredFieldNames,
                                     PluginFieldCollection const *fc);

template <typename Dtype> struct CudaBind {
  size_t mSize;
  void *mPtr;

  CudaBind(size_t size) {
    mSize = size;
    PLUGIN_CUASSERT(cudaMalloc(&mPtr, sizeof(Dtype) * mSize));
  }

  ~CudaBind() {
    if (mPtr != nullptr) {
      PLUGIN_CUASSERT(cudaFree(mPtr));
      mPtr = nullptr;
    }
  }
};

} // namespace plugin
} // namespace nvinfer1

#ifndef DEBUG

#define PLUGIN_CHECK(status)                                                                       \
  do {                                                                                             \
    if (status != 0)                                                                               \
      abort();                                                                                     \
  } while (0)

#define ASSERT_PARAM(exp)                                                                          \
  do {                                                                                             \
    if (!(exp))                                                                                    \
      return STATUS_BAD_PARAM;                                                                     \
  } while (0)

#define ASSERT_FAILURE(exp)                                                                        \
  do {                                                                                             \
    if (!(exp))                                                                                    \
      return STATUS_FAILURE;                                                                       \
  } while (0)

#define CSC(call, err)                                                                             \
  do {                                                                                             \
    cudaError_t cudaStatus = call;                                                                 \
    if (cudaStatus != cudaSuccess) {                                                               \
      return err;                                                                                  \
    }                                                                                              \
  } while (0)

#define DEBUG_PRINTF(...)                                                                          \
  do {                                                                                             \
  } while (0)

#else

#define ASSERT_PARAM(exp)                                                                          \
  do {                                                                                             \
    if (!(exp)) {                                                                                  \
      fprintf(stderr, "Bad param - " #exp ", %s:%d\n", __FILE__, __LINE__);                        \
      return STATUS_BAD_PARAM;                                                                     \
    }                                                                                              \
  } while (0)

#define ASSERT_FAILURE(exp)                                                                        \
  do {                                                                                             \
    if (!(exp)) {                                                                                  \
      fprintf(stderr, "Failure - " #exp ", %s:%d\n", __FILE__, __LINE__);                          \
      return STATUS_FAILURE;                                                                       \
    }                                                                                              \
  } while (0)

#define CSC(call, err)                                                                             \
  do {                                                                                             \
    cudaError_t cudaStatus = call;                                                                 \
    if (cudaStatus != cudaSuccess) {                                                               \
      printf("%s %d CUDA FAIL %s\n", __FILE__, __LINE__, cudaGetErrorString(cudaStatus));          \
      return err;                                                                                  \
    }                                                                                              \
  } while (0)

#define PLUGIN_CHECK(status)                                                                       \
  {                                                                                                \
    if (status != 0) {                                                                             \
      DEBUG_PRINTF("%s %d CUDA FAIL %s\n", __FILE__, __LINE__, cudaGetErrorString(status));        \
      abort();                                                                                     \
    }                                                                                              \
  }

#define DEBUG_PRINTF(...)                                                                          \
  do {                                                                                             \
    printf(__VA_ARGS__);                                                                           \
  } while (0)

#endif // DEBUG

#endif // TRT_PLUGIN_H
