# ONNX PTQ overview

This ONNX PTQ Toolkit provides a comprehensive suite of tools designed to optimize ONNX (Open Neural Network Exchange) models through quantization. Our toolkit is aimed at developers looking to enhance performance, reduce model size, and accelerate inference times without compromising the accuracy of their neural networks when deployed with TensorRT.

Quantization is a technique that converts a model from using floating-point numbers to lower-precision formats like integers, which are computationally less expensive. This process can significantly speed up execution on compatible hardware and reduce memory and bandwidth requirements.
To learn more about the quantization feature, please refer to the [documentation](https://nvidia.github.io/TensorRT-Model-Optimizer/guides/1_quantization.html).

Note that this example is for ONNX model quantization. For end to end quantization examples with Large Language models, please refer to: [llm_ptq](https://github.com/NVIDIA/TensorRT-Model-Optimizer/tree/main/examples/llm_ptq)

## Environment setup

### Linux

Please follow the main [README](../README.md#docker) to build the docker image with TensorRT 10.x pre-installed.

The container can be run with the following command:

```bash
docker run --user 0:0 -it --gpus all --shm-size=2g -v /path/to/ImageNet/dataset:/workspace/imagenet docker.io/library/modelopt_examples:latest
```

### Prepare the example model

Most of the examples in this doc use `vit_base_patch16_224.onnx` as the input model. The model can be downloaded with the following script:

```bash
python download_example_onnx.py \
    --vit \
    --onnx_save_path=vit_base_patch16_224.onnx \
    --fp16 `# <Optional, if the desired output ONNX precision is FP16>`
```

## Quantize an ONNX model

First, prepare some calibration data. TensorRT recommends calibration data size to be at least 500 for CNN and ViT models. The following command picks up 500 images from the [tiny-imagenet](https://huggingface.co/datasets/zh-plus/tiny-imagenet) dataset and converts them to a numpy-format calibration array. Reduce the calibration data size for resource constrained environments.

```bash
python image_prep.py \
    --calibration_data_size=500 \
    --output_path=calib.npy \
    --fp16 `# <Optional, if the input ONNX is in FP16 precision>`
```

> *For Int4 quantization, it is recommended to set `--calibration_data_size=64`.*

The model can be quantized as an FP8, INT8 or INT4 model. For FP8 quantization `max` calibration is used. For INT8 quantization, you have choice between `max` and `entropy` calibration algorithms and for INT4, [awq_clip](https://arxiv.org/abs/2306.00978) or [rtn_dq](https://ar5iv.labs.arxiv.org/html/2301.12017) can be chosen.

> *Note that, INT4 TensorRT engines are not performant yet compared to FP16 engines. Stay tuned for the next update.*

> *Minimum opset requirements: int8 (13+), fp8 (21+), int4 (21+). ModelOpt will automatically upgrade lower opset versions to meet these requirements.*

```bash
python -m modelopt.onnx.quantization \
    --onnx_path=vit_base_patch16_224.onnx \
    --quantize_mode=<fp8|int8|int4> \
    --calibration_data=calib.npy \
    --calibration_method=<max|entropy|awq_clip|rtn_dq> \
    --output_path=vit_base_patch16_224.quant.onnx
```

### Evaluate the quantized ONNX model

The following evaluation requires the `val` directory of the [ImageNet dataset](https://www.kaggle.com/c/imagenet-object-localization-challenge/data). Alternatively, you can prepare it from [this](https://huggingface.co/datasets/mrm8488/ImageNet1K-val) Hugging Face dataset. Once you have it, the quantized ONNX ViT model can be evaluated on the ImageNet dataset as follows:

```bash
python evaluate.py \
    --onnx_path=<path to classification model> \
    --imagenet_path=<path to the ImageNet dataset> \
    --quantize_mode=<fp8|int8|int4> \
    --model_name=vit_base_patch16_224
```

This script converts the quantized ONNX model to a TensorRT engine and does the evaluation with that engine. Finally, the evaluation result will be reported as follows:

```bash
The top1 accuracy of the model is <accuracy score between 0-100%>
The top5 accuracy of the model is <accuracy score between 0-100%>
Inference latency of the model is <X> ms
```

## Quantize an ONNX model with custom op

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

## Torch quantization to ONNX example for MXFP8 and NVFP4 precision

This example demonstrates how to quantize a [timm](https://github.com/huggingface/pytorch-image-models) vision model using either MXFP8 or NVFP4 precision formats, and then export it to ONNX. The script leverages the ModelOpt toolkit for both quantization and ONNX export.

> *Note: Users may experience performance issues with MXFP8 and NVFP4, which will be fixed in the upcoming release.*
> *Opset 20 is used to export the NVFP4 and MXFP8 ONNX models.*

### What it does

- Loads a pretrained timm torch model (default: ViT-Base).
- Quantizes the torch model to MXFP8 or NVFP4 using ModelOpt.
- Exports the quantized model to ONNX.
- Postprocesses the ONNX model to be compatible with TensorRT.
- Saves the final ONNX model.

### Usage

```bash
python torch_quant_to_onnx.py \
    --timm_model_name=<timm model name> \
    --quantize_mode=<mxfp8|nvfp4> \
    --onnx_save_path=<path to save the exported ONNX model>
```

### Evaluation

If the input model is of type image classification, use the following script to evaluate it.

> *Note: TensorRT 10.11 or later is required to evaluate the MXFP8 or NVFP4 ONNX models.*

```bash
python evaluate.py \
    --onnx_path=<path to the exported ONNX model> \
    --imagenet_path=<path to the ImageNet dataset> \
    --quantize_mode=stronglyTyped \
    --model_name=vit_base_patch16_224
```

## Per node calibration of ONNX models

Per node calibration is a memory optimization feature designed to reduce memory consumption during quantization of large ONNX models. Instead of running inference over the entire network at once, this feature processes the model node-by-node, which can significantly reduce peak memory usage and prevent out-of-memory (OOM) errors.

### How it works

When per node calibration is enabled, the quantization process:

1. **Decomposes the model**: Splits the original ONNX model into multiple single-node sub-models
1. **Manages dependencies**: Tracks input/output dependencies between nodes to ensure correct execution order
1. **Processes sequentially**: Runs calibration on each node individually using a topological processing order
1. **Manages memory**: Automatically cleans up intermediate results and manages reference counting to minimize memory usage
1. **Aggregates results**: Combines calibration data from all nodes to produce the final quantized model

### When to use per node calibration

Per node calibration is particularly beneficial for:

- **Large models** that cause OOM errors during standard calibration
- **Memory-constrained environments** where GPU memory is limited
- **Models with complex architectures** that have high intermediate memory requirements

### Usage

To enable per node calibration, add the `--calibrate_per_node` flag to your quantization command:

```bash
python -m modelopt.onnx.quantization \
    --onnx_path=vit_base_patch16_224 \
    --quantize_mode=<int8/fp8> \
    --calibration_data=calib.npy \
    --calibrate_per_node \
    --output_path=vit_base_patch16_224.quant.onnx
```

> **Note**: Per node calibration is not available for INT4 quantization methods (`awq_clip`, `rtn_dq`)
