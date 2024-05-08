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

#ifndef TRT_FP8Conv2D_PLUGIN_H
#define TRT_FP8Conv2D_PLUGIN_H

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "common/bertCommon.h"
#include "common/plugin.h"
#include "cuda_fp16.h"

#include <cstdint>
#include <cstdlib>
#include <vector>

namespace nvinfer1 {
namespace plugin {
class FP8Conv2DPlugin : public IPluginV2DynamicExt {
public:
  FP8Conv2DPlugin() = delete;
  FP8Conv2DPlugin(std::string const &name, const std::vector<long int> &stride,
                  const std::vector<long int> &dilation, const std::vector<long int> &padding,
                  nvinfer1::Weights const &bias, half x_scale, half w_scale, std::int32_t k);

  FP8Conv2DPlugin(std::string const &name, const std::vector<long int> &stride,
                  const std::vector<long int> &dilation, const std::vector<long int> &padding,
                  float *bias_gpu, half x_scale, half w_scale, std::int32_t k);

  FP8Conv2DPlugin(std::string const &name, void const *buffer, size_t length);
  ~FP8Conv2DPlugin() = default;

  FP8Conv2DPlugin(const FP8Conv2DPlugin & /*other*/) = default;
  FP8Conv2DPlugin &operator=(const FP8Conv2DPlugin & /*other*/) = delete;
  FP8Conv2DPlugin(FP8Conv2DPlugin && /*other*/) noexcept = delete;
  FP8Conv2DPlugin &operator=(FP8Conv2DPlugin && /*other*/) noexcept = delete;

  void init(std::string const &name, const std::vector<long int> &stride,
            const std::vector<long int> &dilation, const std::vector<long int> &padding,
            half x_scale, half w_scale, std::int32_t k);

  // Methods inherited from IPluginV2
  char const *getPluginType() const noexcept override;
  char const *getPluginVersion() const noexcept override;
  int32_t getNbOutputs() const noexcept override;
  int32_t initialize() noexcept override;
  void terminate() noexcept override;
  size_t getSerializationSize() const noexcept override;
  void serialize(void *buffer) const noexcept override;
  void destroy() noexcept override;
  void setPluginNamespace(char const *pluginNamespace) noexcept override;
  char const *getPluginNamespace() const noexcept override;

  // Method inherited from IPluginV2Ext
  DataType getOutputDataType(int32_t index, DataType const *inputTypes,
                             int32_t nbInputs) const noexcept override;

  // // Methods inherited from IPluginV2DynamicExt
  IPluginV2DynamicExt *clone() const noexcept override;
  DimsExprs getOutputDimensions(int32_t outputIndex, DimsExprs const *inputs, int32_t nbInputs,
                                IExprBuilder &exprBuilder) noexcept override;
  bool supportsFormatCombination(int32_t pos, PluginTensorDesc const *inOut, int32_t nbInputs,
                                 int32_t nbOutputs) noexcept override;
  void configurePlugin(DynamicPluginTensorDesc const *in, int32_t nbInputs,
                       DynamicPluginTensorDesc const *out, int32_t nbOutputs) noexcept override;
  size_t getWorkspaceSize(PluginTensorDesc const *inputs, int32_t nbInputs,
                          PluginTensorDesc const *outputs,
                          int32_t nbOutputs) const noexcept override;
  int32_t enqueue(PluginTensorDesc const *inputDesc, PluginTensorDesc const *outputDesc,
                  void const *const *inputs, void *const *outputs, void *workspace,
                  cudaStream_t stream) noexcept override;

private:
  float *bias_dev = nullptr;
  __half *W_scale_dev = nullptr;
  __half *X_scale_dev = nullptr;
  float *conv_scale_dev = nullptr;

  half W_scale_host = 0.0;
  half X_scale_host = 0.0;
  float conv_scale_host = 0.0;

  size_t workspace_sz;
  void *workspace_h = nullptr;
  void *workspace_d = nullptr;

  int32_t conv_k = 640;
  int32_t sm = 0;
  int32_t sm_count = 0;

  std::vector<long int> conv_stride = {1, 1};
  std::vector<long int> conv_dilation = {1, 1};
  std::vector<long int> conv_padding = {1, 1};

  std::string mNameSpace;
  std::string mName;
};

class FP8Conv2DPluginCreator : public nvinfer1::pluginInternal::BaseCreator {
public:
  FP8Conv2DPluginCreator();
  ~FP8Conv2DPluginCreator();

  FP8Conv2DPluginCreator(const FP8Conv2DPluginCreator & /*other*/) = delete;
  FP8Conv2DPluginCreator &operator=(const FP8Conv2DPluginCreator & /*other*/) = delete;
  FP8Conv2DPluginCreator(FP8Conv2DPluginCreator && /*other*/) noexcept = delete;
  FP8Conv2DPluginCreator &operator=(FP8Conv2DPluginCreator && /*other*/) noexcept = delete;

  char const *getPluginName() const noexcept override;
  char const *getPluginVersion() const noexcept override;
  PluginFieldCollection const *getFieldNames() noexcept override;
  IPluginV2 *createPlugin(char const *name, PluginFieldCollection const *fc) noexcept override;
  IPluginV2 *deserializePlugin(char const *name, void const *serialData,
                               size_t serialLength) noexcept override;

private:
  static PluginFieldCollection mFC;
  static std::vector<PluginField> mPluginAttributes;
  std::string mNameSpace;
};

} // namespace plugin
} // namespace nvinfer1

#endif
