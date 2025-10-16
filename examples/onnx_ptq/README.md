# ONNX Post-training quantization (PTQ)

This ONNX PTQ Toolkit provides a comprehensive suite of tools designed to optimize ONNX (Open Neural Network Exchange) models through quantization. Our toolkit is aimed at developers looking to enhance performance, reduce model size, and accelerate inference times without compromising the accuracy of their neural networks when deployed with TensorRT.

Quantization is an effective model optimization technique that compresses your models. Quantization with Model Optimizer can compress model size by 2x-4x, speeding up inference while preserving model quality.

Model Optimizer enables highly performant quantization formats including NVFP4, FP8, INT8, INT4 and supports advanced algorithms such as AWQ and Double Quantization with easy-to-use Python APIs.

<div align="center">

| **Section** | **Description** | **Link** | **Docs** |
| :------------: | :------------: | :------------: | :------------: |
| Pre-Requisites | Required & optional packages to use this technique | [Link](#pre-requisites) | |
| Getting Started | Learn how to optimize your models using PTQ to reduce precision and improve inference efficiency | [Link](#getting-started) | [docs](https://nvidia.github.io/TensorRT-Model-Optimizer/guides/_onnx_quantization.html) |
| Support Matrix | View the ONNX export supported LLM models | [Link](#onnx-export-supported-llm-models) | |
| PyTorch to ONNX | Example scripts demonstrating how to quantize with PyTorch and then convert to ONNX | [Link](#torch-quantization-to-onnx-example-for-mxfp8-int4-or-nvfp4-precision) | |
| Advanced Features | Examples demonstrating use advanced ONNX quantization features | [Link](#advanced-features) | |
| Pre-Quantized Checkpoints | Ready to deploy Hugging Face pre-quantized checkpoints | [Link](#pre-quantized-checkpoints) | |
| Resources | Extra links to relevant resources | [Link](#resources) | |

</div>

## Pre-Requisites

### Docker

Please use the TensorRT docker image (e.g., `nvcr.io/nvidia/tensorrt:25.08-py3`) or visit our [installation docs](https://nvidia.github.io/TensorRT-Model-Optimizer/getting_started/2_installation.html) for more information.

Set the following environment variables inside the TensorRT docker.

```bash
export CUDNN_LIB_DIR=/usr/lib/x86_64-linux-gnu/
export LD_LIBRARY_PATH="${CUDNN_LIB_DIR}:${LD_LIBRARY_PATH}"
```

Also follow the installation steps below to upgrade to the latest version of Model Optimizer and install example-specific dependencies.

### Local Installation

Install Model Optimizer with `onnx` dependencies using `pip` from [PyPI](https://pypi.org/project/nvidia-modelopt/) and install the requirements for the example:

```bash
pip install -U nvidia-modelopt[onnx]
pip install -r requirements.txt
```

For TensorRT Compiler framework workloads:

Install the latest [TensorRT](https://developer.nvidia.com/tensorrt) from [here](https://developer.nvidia.com/tensorrt/download).

## Getting Started

### Prepare the example model

Most of the examples in this doc use `vit_base_patch16_224.onnx` as the input model. The model can be downloaded with the following script:

```bash
python download_example_onnx.py \
    --vit \
    --onnx_save_path=vit_base_patch16_224.onnx \
    --fp16 # <Optional, if the desired output ONNX precision is FP16>
```

### Prepare calibration data

Calibration data is a representative subset of your training or validation dataset used during quantization to determine the optimal scale factors for converting floating-point values to lower precision formats (INT8, FP8, INT4). This data helps maintain model accuracy after quantization by analyzing the distribution of activations throughout the network.

First, prepare some calibration data. TensorRT recommends calibration data size to be at least 500 for CNN and ViT models. The following command picks up 500 images from the [tiny-imagenet](https://huggingface.co/datasets/zh-plus/tiny-imagenet) dataset and converts them to a numpy-format calibration array. Reduce the calibration data size for resource constrained environments.

```bash
python image_prep.py \
    --calibration_data_size=500 \
    --output_path=calib.npy \
    --fp16 # <Optional, if the input ONNX is in FP16 precision>
```

> *For Int4 quantization, it is recommended to set `--calibration_data_size=64`.*

### Quantize ONNX Model to FP8, INT8 or INT4

The model can be quantized as an FP8, INT8 or INT4 model using either the CLI or Python API. For FP8 and INT8 quantization, you have a choice between `max` and `entropy` calibration algorithms. For INT4 quantization, [awq_clip](https://arxiv.org/abs/2306.00978) or [rtn_dq](https://ar5iv.labs.arxiv.org/html/2301.12017) algorithms can be chosen.

> *For NVFP4 and MXFP8 ONNX, see the [PyTorch to ONNX section](#torch-quantization-to-onnx-example-for-mxfp8-int4-or-nvfp4-precision).*

> *Minimum opset requirements: int8 (13+), fp8 (21+), int4 (21+). ModelOpt will automatically upgrade lower opset versions to meet these requirements.*

#### Option 1: Command-line interface

```bash
python -m modelopt.onnx.quantization \
    --onnx_path=vit_base_patch16_224.onnx \
    --quantize_mode=<fp8|int8|int4> \
    --calibration_data=calib.npy \
    --calibration_method=<max|entropy|awq_clip|rtn_dq> \
    --output_path=vit_base_patch16_224.quant.onnx
```

#### Option 2: Python API

```python
from modelopt.onnx.quantization import quantize

quantize(
    onnx_path="vit_base_patch16_224.onnx",
    quantize_mode="int8",       # fp8, int8, int4 etc.
    calibration_data="calib.npy",
    calibration_method="max",   # max, entropy, awq_clip, rtn_dq etc.
    output_path="vit_base_patch16_224.quant.onnx",
)
```

### Evaluate the quantized ONNX model

The following evaluation requires the `val` directory of the [ImageNet dataset](https://www.kaggle.com/c/imagenet-object-localization-challenge/data). Alternatively, you can prepare it from [this](https://huggingface.co/datasets/mrm8488/ImageNet1K-val) Hugging Face dataset. Once you have it, the quantized ONNX ViT model can be evaluated on the ImageNet dataset as follows:

```bash
python evaluate.py \
    --onnx_path=<path to classification model> \
    --imagenet_path=<path to the ImageNet dataset> \
    --engine_precision=stronglyTyped \
    --model_name=vit_base_patch16_224
```

This script converts the quantized ONNX model to a TensorRT engine and does the evaluation with that engine. Finally, the evaluation result will be reported as follows:

```bash
The top1 accuracy of the model is <accuracy score between 0-100%>
The top5 accuracy of the model is <accuracy score between 0-100%>
Inference latency of the model is <X> ms
```

## Torch quantization to ONNX example for MXFP8, INT4 or NVFP4 precision

This example demonstrates how to quantize a [timm](https://github.com/huggingface/pytorch-image-models) vision model using MXFP8, INT4 or NVFP4 precision formats, and then export it to ONNX. The script leverages the ModelOpt toolkit for both quantization and ONNX export.

> *Opset 20 is used to export the torch models to ONNX.*

### What it does

- Loads a pretrained timm torch model (default: ViT-Base).
- Quantizes the torch model to MXFP8, INT4 or NVFP4 using ModelOpt.
- Exports the quantized model to ONNX.
- Postprocesses the ONNX model to be compatible with TensorRT.
- Saves the final ONNX model.

### Usage

```bash
python torch_quant_to_onnx.py \
    --timm_model_name=vit_base_patch16_224 \
    --quantize_mode=<mxfp8|nvfp4|int4_awq> \
    --onnx_save_path=<path to save the exported ONNX model>
```

### Evaluation

If the input model is of type image classification, use the following script to evaluate it.

> *Note: TensorRT 10.11 or later is required to evaluate the MXFP8 or NVFP4 ONNX models.*

```bash
python evaluate.py \
    --onnx_path=<path to the exported ONNX model> \
    --imagenet_path=<path to the ImageNet dataset> \
    --engine_precision=stronglyTyped \
    --model_name=vit_base_patch16_224
```

### ONNX Export Supported LLM Models

| Model | FP16 | INT4 | FP8 | NVFP4 |
| :---: | :---: | :---: | :---: | :---: |
| [Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) | ✅ | ✅ | ✅ | ✅ |
| [Llama3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) | ✅ | ✅ | ✅ | ✅ |
| [Llama3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B) | ✅ | ✅ | ✅ | ✅ |
| [Qwen2-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct) | ✅ | ✅ | ✅ | ✅ |
| [Qwen2-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct) | ✅ | ✅ | ✅ | ✅ |
| [Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct) | ✅ | ✅ | ✅ | ✅ |
| [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | ✅ | ✅ | ✅ | ✅ |
| [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | ✅ | ✅ | ✅ | ✅ |
| [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) | ✅ | ✅ | ✅ | ✅ |
| [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | ✅ | ✅ | ✅ | ✅ |

## Advanced Features

### Per node calibration of ONNX models

Per node calibration is a memory optimization feature designed to reduce memory consumption during quantization of large ONNX models. Instead of running inference over the entire network at once, this feature processes the model node-by-node, which can significantly reduce peak memory usage and prevent out-of-memory (OOM) errors.

#### How it works

When per node calibration is enabled, the quantization process:

1. **Decomposes the model**: Splits the original ONNX model into multiple single-node sub-models
1. **Manages dependencies**: Tracks input/output dependencies between nodes to ensure correct execution order
1. **Processes sequentially**: Runs calibration on each node individually using a topological processing order
1. **Manages memory**: Automatically cleans up intermediate results and manages reference counting to minimize memory usage
1. **Aggregates results**: Combines calibration data from all nodes to produce the final quantized model

#### When to use per node calibration

Per node calibration is particularly beneficial for:

- **Large models** that cause OOM errors during standard calibration
- **Memory-constrained environments** where GPU memory is limited
- **Models with complex architectures** that have high intermediate memory requirements

#### Usage

To enable per node calibration, add the `--calibrate_per_node` flag to your quantization command:

```bash
python -m modelopt.onnx.quantization \
    --onnx_path=vit_base_patch16_224.onnx \
    --quantize_mode=<int8/fp8> \
    --calibration_data=calib.npy \
    --calibrate_per_node \
    --output_path=vit_base_patch16_224.quant.onnx
```

> **Note**: Per node calibration is not available for INT4 quantization methods (`awq_clip`, `rtn_dq`)

### Quantize an ONNX model with custom op

This feature requires `TensorRT 10+` and `ORT>=1.20`. For proper usage, please make sure that the paths to `libcudnn*.so` and TensorRT `lib/` are in the `LD_LIBRARY_PATH` env variable and that the `tensorrt` python package is installed.

Please see the sample example below.

**Step 1**: Obtain the sample ONNX model and TensorRT plugin from [TensorRT-Custom-Plugin-Example](https://github.com/leimao/TensorRT-Custom-Plugin-Example).

&#160; **1.1.** Change directory to `TensorRT-Custom-Plugin-Example`:

```bash
cd /path/to/TensorRT-Custom-Plugin-Example
```

&#160; **1.2.** Compile the TensorRT plugin:

```bash
cmake -B build \
    -DNVINFER_LIB=$TRT_LIBPATH/libnvinfer.so.10 \
    -DNVINFER_PLUGIN_LIB=$TRT_LIBPATH/libnvinfer_plugin.so.10 \
    -DNVONNXPARSER_LIB=$TRT_LIBPATH/libnvonnxparser.so.10 \
    -DCMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES=/usr/include/x86_64-linux-gnu
```

```bash
cmake --build build --config Release --parallel
```

This generates a plugin in `TensorRT-Custom-Plugin-Example/build/src/plugins/IdentityConvIPluginV2IOExt/libidentity_conv_iplugin_v2_io_ext.so`

&#160; **1.3.** Create the ONNX file.

```bash
python scripts/create_identity_neural_network.py
```

This generates the identity_neural_network.onnx model in `TensorRT-Custom-Plugin-Example/data/identity_neural_network.onnx`

**Step 2**: Quantize the ONNX model. We will be using the `libidentity_conv_iplugin_v2_io_ext.so` plugin for this example.

```bash
python -m modelopt.onnx.quantization \
    --onnx_path=/path/to/identity_neural_network.onnx \
    --trt_plugins=/path/to/libidentity_conv_iplugin_v2_io_ext.so
```

**Step 3**: Deploy the quantized model with TensorRT.

```bash
trtexec --onnx=/path/to/identity_neural_network.quant.onnx \
    --staticPlugins=/path/to/libidentity_conv_iplugin_v2_io_ext.so
```

## Pre-Quantized Checkpoints

- Ready-to-deploy checkpoints that can be exported to ONNX format (if supported as per the [Support Matrix](#onnx-export-supported-llm-models)) \[[🤗 Hugging Face - Nvidia TensorRT Model Optimizer Collection](https://huggingface.co/collections/nvidia/model-optimizer-66aa84f7966b3150262481a4)\]

## Resources

- 📅 [Roadmap](https://github.com/NVIDIA/TensorRT-Model-Optimizer/issues/146)
- 📖 [Documentation](https://nvidia.github.io/TensorRT-Model-Optimizer)
- 🎯 [Benchmarks](../benchmark.md)
- 💡 [Release Notes](https://nvidia.github.io/TensorRT-Model-Optimizer/reference/0_changelog.html)
- 🐛 [File a bug](https://github.com/NVIDIA/TensorRT-Model-Optimizer/issues/new?template=1_bug_report.md)
- ✨ [File a Feature Request](https://github.com/NVIDIA/TensorRT-Model-Optimizer/issues/new?template=2_feature_request.md)

### Technical Resources

There are many quantization schemes supported in the example scripts:

1. The [FP8 format](https://developer.nvidia.com/blog/nvidia-arm-and-intel-publish-fp8-specification-for-standardization-as-an-interchange-format-for-ai/) is available on the Hopper and Ada GPUs with [CUDA compute capability](https://developer.nvidia.com/cuda-gpus) greater than or equal to 8.9.

1. The [INT4 AWQ](https://arxiv.org/abs/2306.00978) is an INT4 weight only quantization and calibration method. INT4 AWQ is particularly effective for low batch inference where inference latency is dominated by weight loading time rather than the computation time itself. For low batch inference, INT4 AWQ could give lower latency than FP8/INT8 and lower accuracy degradation than INT8.

1. The [NVFP4](https://blogs.nvidia.com/blog/generative-ai-studio-ces-geforce-rtx-50-series/) is one of the new FP4 formats supported by NVIDIA Blackwell GPU and demonstrates good accuracy compared with other 4-bit alternatives. NVFP4 can be applied to both model weights as well as activations, providing the potential for both a significant increase in math throughput and reductions in memory footprint and memory bandwidth usage compared to the FP8 data format on Blackwell.
