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

#include "FP8Conv2DPlugin.h"
#include "common/serialize.hpp"
#include <cmath>

using namespace nvinfer1;
using namespace plugin;

using nvinfer1::plugin::FP8Conv2DPlugin;
using nvinfer1::plugin::FP8Conv2DPluginCreator;

namespace {
static std::string const kFP8_CONV_2D_PLUGIN_NAME{"FP8Conv2D"};
static std::string const kFP8_CONV_2D_PLUGIN_VERSION{"1"};
} // namespace

extern "C" int fp8_conv_kernel_launch(void *act_dev, void *flt_dev, void *out_dev, float *bias_dev,
                                      float conv_scale_host, size_t workspace_sz, void *workspace_h,
                                      void *workspace_d, int32_t conv_n, int32_t conv_c,
                                      int32_t conv_h, int32_t conv_w, int32_t conv_k,
                                      int32_t conv_r, int32_t conv_s, long int *conv_stride,
                                      long int *conv_dilation, long int *conv_padding, int32_t sm,
                                      int32_t sm_count, cudaStream_t stream);

FP8Conv2DPlugin::FP8Conv2DPlugin(std::string const &name, const std::vector<long int> &stride,
                                 const std::vector<long int> &dilation,
                                 const std::vector<long int> &padding,
                                 nvinfer1::Weights const &bias, half x_scale, half w_scale,
                                 std::int32_t k) {

  W_scale_host = w_scale;
  PLUGIN_CUASSERT(cudaMalloc((void **)&(W_scale_dev), (size_t)sizeof(W_scale_host)));
  PLUGIN_CUASSERT(
      cudaMemcpy(W_scale_dev, &W_scale_host, sizeof(W_scale_host), cudaMemcpyHostToDevice));

  PLUGIN_CUASSERT(cudaMalloc((void **)&(bias_dev), (size_t)(sizeof(float) * bias.count)));
  PLUGIN_CUASSERT(
      cudaMemcpy(bias_dev, bias.values, sizeof(float) * bias.count, cudaMemcpyHostToDevice));

  init(name, stride, dilation, padding, x_scale, w_scale, k);
}

FP8Conv2DPlugin::FP8Conv2DPlugin(std::string const &name, const std::vector<long int> &stride,
                                 const std::vector<long int> &dilation,
                                 const std::vector<long int> &padding, float *bias_gpu,
                                 half x_scale, half w_scale, std::int32_t k) {
  W_scale_host = w_scale;
  PLUGIN_CUASSERT(cudaMalloc((void **)&(bias_dev), (size_t)(sizeof(float) * k)));
  PLUGIN_CUASSERT(
      cudaMemcpy(bias_dev, bias_gpu, (size_t)(sizeof(float) * k), cudaMemcpyDeviceToDevice));

  init(name, stride, dilation, padding, x_scale, w_scale, k);
}

void FP8Conv2DPlugin::init(std::string const &name, const std::vector<long int> &stride,
                           const std::vector<long int> &dilation,
                           const std::vector<long int> &padding, half x_scale, half w_scale,
                           std::int32_t k) {

  X_scale_host = x_scale;
  PLUGIN_CUASSERT(cudaMalloc((void **)&(X_scale_dev), (size_t)sizeof(X_scale_host)));
  PLUGIN_CUASSERT(
      cudaMemcpy(X_scale_dev, &X_scale_host, sizeof(X_scale_host), cudaMemcpyHostToDevice));

  conv_scale_host = 1.0 / (float(X_scale_host) * float(W_scale_host));
  std::vector<float> conv_scale_host_vector(k, conv_scale_host);

  PLUGIN_CUASSERT(cudaMalloc((void **)&(conv_scale_dev), (size_t)(sizeof(conv_scale_host) * k)));
  PLUGIN_CUASSERT(cudaMemcpy(conv_scale_dev, conv_scale_host_vector.data(),
                             sizeof(conv_scale_host) * k, cudaMemcpyHostToDevice));
  conv_k = k;

  conv_stride.assign(stride.begin(), stride.end());
  conv_dilation.assign(dilation.begin(), dilation.end());
  conv_padding.assign(padding.begin(), padding.end());

  cudaDeviceProp props;
  PLUGIN_CUASSERT(cudaGetDeviceProperties(&props, 0));
  sm = props.major * 10 + props.minor;
  sm_count = props.multiProcessorCount;
}

FP8Conv2DPlugin::FP8Conv2DPlugin(std::string const &name, void const *buffer, size_t length) {

  deserialize_value(&buffer, &length, &conv_k);

  // TODO, a bug here stride can't get correct number
  deserialize_value(&buffer, &length, &conv_stride[0]);
  deserialize_value(&buffer, &length, &conv_stride[1]);
  deserialize_value(&buffer, &length, &conv_dilation[0]);
  deserialize_value(&buffer, &length, &conv_dilation[1]);
  deserialize_value(&buffer, &length, &conv_padding[0]);
  deserialize_value(&buffer, &length, &conv_padding[1]);
  deserialize_value(&buffer, &length, &X_scale_host);
  deserialize_value(&buffer, &length, &W_scale_host);

  const char *d = static_cast<const char *>(buffer);
  int32_t bias_memory_size = sizeof(float) * conv_k;

  PLUGIN_CUASSERT(cudaMalloc((void **)&(bias_dev), (size_t)(bias_memory_size)));
  PLUGIN_CUASSERT(cudaMemcpy(bias_dev, d, bias_memory_size, cudaMemcpyHostToDevice));

  init(name, conv_stride, conv_dilation, conv_padding, X_scale_host, W_scale_host, conv_k);
}

char const *FP8Conv2DPlugin::getPluginType() const noexcept {
  return kFP8_CONV_2D_PLUGIN_NAME.c_str();
}

char const *FP8Conv2DPlugin::getPluginVersion() const noexcept {
  return kFP8_CONV_2D_PLUGIN_VERSION.c_str();
}

int32_t FP8Conv2DPlugin::getNbOutputs() const noexcept { return 1; }

int32_t FP8Conv2DPlugin::initialize() noexcept { return 0; }

void FP8Conv2DPlugin::setPluginNamespace(char const *pluginNamespace) noexcept {
  mNameSpace = pluginNamespace;
}

char const *FP8Conv2DPlugin::getPluginNamespace() const noexcept { return mNameSpace.c_str(); }

IPluginV2DynamicExt *FP8Conv2DPlugin::clone() const noexcept {
  try {
    auto p = new FP8Conv2DPlugin(mName, conv_stride, conv_dilation, conv_padding, bias_dev,
                                 X_scale_host, W_scale_host, conv_k);

    p->setPluginNamespace(mNameSpace.c_str());
    return p;
  } catch (std::exception const &e) {
    caughtError(e);
  }
  return nullptr;
}

void FP8Conv2DPlugin::configurePlugin(DynamicPluginTensorDesc const *in, int32_t nbInputs,
                                      DynamicPluginTensorDesc const *out,
                                      int32_t nbOutputs) noexcept {}

bool FP8Conv2DPlugin::supportsFormatCombination(int32_t pos, PluginTensorDesc const *inOut,
                                                int32_t nbInputs, int32_t nbOutputs) noexcept {
  try {
    if (pos == 0 || pos == 1) {
      return inOut[pos].type == DataType::kFP8 && inOut[pos].format == TensorFormat::kLINEAR;
    } else if (pos == 2) {
      return inOut[pos].type == DataType::kFLOAT && inOut[pos].format == TensorFormat::kLINEAR;
      // return inOut[pos].type == DataType::kHALF && inOut[pos].format ==
      // TensorFormat::kLINEAR;
    } else {
      return false;
    }
  } catch (std::exception const &e) {
    caughtError(e);
  }
  return false;
}

size_t FP8Conv2DPlugin::getWorkspaceSize(PluginTensorDesc const *inputs, int32_t nbInputs,
                                         PluginTensorDesc const *outputs,
                                         int32_t nbOutputs) const noexcept {
  return 0;
}

DimsExprs FP8Conv2DPlugin::getOutputDimensions(int32_t outputIndex, DimsExprs const *inputs,
                                               int32_t nbInputs,
                                               IExprBuilder &exprBuilder) noexcept {
  DimsExprs ret{};
  try {
    if (outputIndex == 0) {
      ret.nbDims = 4;
      ret.d[0] = inputs[0].d[0];

      auto const1 = exprBuilder.constant(1);
      auto const_padding1 = exprBuilder.constant(2 * conv_padding[0]);
      auto const_stride1 = exprBuilder.constant(conv_stride[0]);
      auto h_out = exprBuilder.operation(
          DimensionOperation::kSUM,
          *(exprBuilder.operation(
              DimensionOperation::kFLOOR_DIV,
              *(exprBuilder.operation(DimensionOperation::kSUB,
                                      *exprBuilder.operation(DimensionOperation::kSUM,
                                                             *inputs[0].d[1], *const_padding1),
                                      *inputs[1].d[1])),
              *const_stride1)),
          *const1);
      ret.d[1] = h_out;

      auto const_padding2 = exprBuilder.constant(2 * conv_padding[1]);
      auto const_stride2 = exprBuilder.constant(conv_stride[1]);
      auto w_out = exprBuilder.operation(
          DimensionOperation::kSUM,
          *(exprBuilder.operation(
              DimensionOperation::kFLOOR_DIV,
              *(exprBuilder.operation(DimensionOperation::kSUB,
                                      *exprBuilder.operation(DimensionOperation::kSUM,
                                                             *inputs[0].d[2], *const_padding2),
                                      *inputs[1].d[2])),
              *const_stride2)),
          *const1);
      ret.d[2] = w_out;

      ret.d[3] = exprBuilder.constant(conv_k);
      return ret;
    }
  } catch (std::exception const &e) {
    caughtError(e);
  }
  return ret;
}

DataType FP8Conv2DPlugin::getOutputDataType(int32_t index, DataType const *inputTypes,
                                            int32_t nbInputs) const noexcept {
  DataType ret{};
  try {
    PLUGIN_VALIDATE(inputTypes != nullptr);
    PLUGIN_VALIDATE(nbInputs > 0);
    // ret = inputTypes[0];
    ret = DataType::kFLOAT;
    // ret = DataType::kHALF;
  } catch (std::exception const &e) {
    caughtError(e);
  }
  return ret;
}

int32_t FP8Conv2DPlugin::enqueue(PluginTensorDesc const *inputDesc,
                                 PluginTensorDesc const *outputDesc, void const *const *inputs,
                                 void *const *outputs, void *workspace,
                                 cudaStream_t stream) noexcept {
  try {
    void *act_dev = const_cast<void *>(inputs[0]);
    void *flt_dev = const_cast<void *>(inputs[1]);
    void *out_dev = outputs[0];
    int32_t conv_n = inputDesc[0].dims.d[0];
    int32_t conv_h = inputDesc[0].dims.d[1];
    int32_t conv_w = inputDesc[0].dims.d[2];
    int32_t conv_c = inputDesc[0].dims.d[3];
    int32_t conv_r = inputDesc[1].dims.d[1];
    int32_t conv_s = inputDesc[1].dims.d[2];

    return fp8_conv_kernel_launch(act_dev, flt_dev, out_dev, bias_dev, conv_scale_host,
                                  workspace_sz, workspace_h, workspace_d, conv_n, conv_c, conv_h,
                                  conv_w, conv_k, conv_r, conv_s, conv_stride.data(),
                                  conv_dilation.data(), conv_padding.data(), sm, sm_count, stream);
  } catch (std::exception const &e) {
    caughtError(e);
  }
  return -1;
}

void FP8Conv2DPlugin::terminate() noexcept {}

void FP8Conv2DPlugin::destroy() noexcept {
  cudaFree(bias_dev);
  cudaFree(W_scale_dev);
  cudaFree(X_scale_dev);
  cudaFree(conv_scale_dev);
}

size_t FP8Conv2DPlugin::getSerializationSize() const noexcept {
  size_t res = 0;
  res += sizeof(conv_k);
  res += sizeof(conv_dilation[0]) * 6;
  res += conv_k * sizeof(float);
  return res;
}
void FP8Conv2DPlugin::serialize(void *buffer) const noexcept {
  serialize_value(&buffer, conv_k);
  serialize_value(&buffer, conv_stride[0]);
  serialize_value(&buffer, conv_stride[1]);
  serialize_value(&buffer, conv_dilation[0]);
  serialize_value(&buffer, conv_dilation[1]);
  serialize_value(&buffer, conv_padding[0]);
  serialize_value(&buffer, conv_padding[1]);
  serialize_value(&buffer, X_scale_host);
  serialize_value(&buffer, W_scale_host);

  int32_t bias_memory_size = conv_k * sizeof(float);
  char *d = static_cast<char *>(buffer);
  PLUGIN_CUASSERT(cudaMemcpy(d, bias_dev, bias_memory_size, cudaMemcpyDeviceToHost));
}

PluginFieldCollection FP8Conv2DPluginCreator::mFC{};
std::vector<PluginField> FP8Conv2DPluginCreator::mPluginAttributes;

FP8Conv2DPluginCreator::FP8Conv2DPluginCreator() {
  mPluginAttributes.clear();
  mPluginAttributes.emplace_back(PluginField("k", nullptr, PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(PluginField("stride", nullptr, PluginFieldType::kINT32, 2));
  mPluginAttributes.emplace_back(PluginField("dilation", nullptr, PluginFieldType::kINT32, 2));
  mPluginAttributes.emplace_back(PluginField("padding", nullptr, PluginFieldType::kINT32, 2));
  mPluginAttributes.emplace_back(PluginField("x_scale", nullptr, PluginFieldType::kFLOAT32, 1));
  mPluginAttributes.emplace_back(PluginField("w_scale", nullptr, PluginFieldType::kFLOAT32, 1));
  mPluginAttributes.emplace_back(PluginField("bias", nullptr, PluginFieldType::kFLOAT32));

  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

FP8Conv2DPluginCreator::~FP8Conv2DPluginCreator() {}

IPluginV2 *FP8Conv2DPluginCreator::createPlugin(char const *name,
                                                PluginFieldCollection const *fc) noexcept {
  try {
    std::vector<long int> stride = {1, 1};
    std::vector<long int> dilation = {1, 1};
    std::vector<long int> padding = {1, 1};
    Weights bias{DataType::kFLOAT, nullptr, 0};
    half w_scale = 0.0;
    half x_scale = 0.0;
    int32_t k = 640;

    for (int32_t i = 0; i < fc->nbFields; ++i) {
      if (fc->fields[i].name == std::string("k")) {
        k = *static_cast<int32_t const *>(fc->fields[i].data);
        continue;
      }
      if (fc->fields[i].name == std::string("x_scale")) {
        x_scale = (half)(*static_cast<float const *>(fc->fields[i].data));
        continue;
      }
      if (fc->fields[i].name == std::string("w_scale")) {
        w_scale = (half)(*static_cast<float const *>(fc->fields[i].data));
        continue;
      }

      if (fc->fields[i].name == std::string("stride")) {
        const int32_t *stride_field = static_cast<int32_t const *>(fc->fields[i].data);
        for (int i = 0; i < stride.size(); i++) {
          stride[i] = (long int)stride_field[i];
        }
        continue;
      }
      if (fc->fields[i].name == std::string("dilation")) {
        const int32_t *dilation_field = static_cast<int32_t const *>(fc->fields[i].data);
        for (int i = 0; i < dilation.size(); i++) {
          dilation[i] = (long int)dilation_field[i];
        }
        continue;
      }
      if (fc->fields[i].name == std::string("padding")) {
        const int32_t *padding_field = static_cast<int32_t const *>(fc->fields[i].data);
        for (int i = 0; i < padding.size(); i++) {
          padding[i] = (long int)padding_field[i];
        }
        continue;
      }
      if (fc->fields[i].name == std::string("bias")) {
        bias.values = fc->fields[i].data;
        bias.count = fc->fields[i].length;
        bias.type = nvinfer1::DataType::kFLOAT;
        continue;
      }
    }
    return new FP8Conv2DPlugin(name, stride, dilation, padding, bias, x_scale, w_scale, k);
  } catch (std::exception const &e) {
    caughtError(e);
  }
  return nullptr;
}

IPluginV2 *FP8Conv2DPluginCreator::deserializePlugin(char const *name, void const *serialData,
                                                     size_t serialLength) noexcept {
  try {
    return new FP8Conv2DPlugin(name, serialData, serialLength);
  } catch (std::exception const &e) {
    caughtError(e);
  }
  return nullptr;
}

char const *FP8Conv2DPluginCreator::getPluginName() const noexcept {
  return kFP8_CONV_2D_PLUGIN_NAME.c_str();
}

char const *FP8Conv2DPluginCreator::getPluginVersion() const noexcept {
  return kFP8_CONV_2D_PLUGIN_VERSION.c_str();
}

PluginFieldCollection const *FP8Conv2DPluginCreator::getFieldNames() noexcept { return &mFC; }

REGISTER_TENSORRT_PLUGIN(FP8Conv2DPluginCreator);
