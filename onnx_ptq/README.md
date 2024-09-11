# ONNX PTQ overview

This ONNX PTQ Toolkit provides a comprehensive suite of tools designed to optimize ONNX (Open Neural Network Exchange) models through quantization. Our toolkit is aimed at developers looking to enhance performance, reduce model size, and accelerate inference times without compromising the accuracy of their neural networks when deployed with TensorRT.

Quantization is a technique that converts a model from using floating-point numbers to lower-precision formats like integers, which are computationally less expensive. This process can significantly speed up execution on compatible hardware and reduce memory and bandwidth requirements.

Note that this example is for ONNX model quantization. For end to end quantization examples with Large Language models, please refer to: [llm_ptq](https://github.com/NVIDIA/TensorRT-Model-Optimizer/tree/main/llm_ptq)

# Environment setup

## Linux

Please follow the main [README](../README.md#docker) to build the docker image with TensorRT 10.x pre-installed.

The container can be run with the following command:

```bash
docker run --user 0:0 -it --gpus all --shm-size=2g -v /path/to/ImageNet/dataset:/workspace/imagenet docker.io/library/modelopt_examples:latest
```

## Prepare the example model

Most of the examples in this doc use `vit_base_patch16_224.onnx` as the input model. The model can be downloaded with the following script:

```bash
python download_example_onnx.py --vit --output_path=vit_base_patch16_224.onnx
```

# Quantize an ONNX model

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

```bash
python -m modelopt.onnx.quantization \
    --onnx_path=vit_base_patch16_224.onnx \
    --quantize_mode=<fp8|int8|int4> \
    --calibration_data=calib.npy \
    --calibration_method=<max|entropy|awq_clip|rtn_dq> \
    --output_path=vit_base_patch16_224.quant.onnx
```

## Evaluate the quantized ONNX model

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

# Quantize an ONNX model with custom op

This feature is supported with TensorRT 10+ and `ORT>=1.18`. For proper usage, please make sure that the TensorRT libraries are in the `PATH` or `LD_LIBRARY_PATH` variables, that your GPU is supported.

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

**Step 2**: Modify the `identity_neural_network.onnx` model so that the custom ops are supported by ORT.

```bash
python trt_plugin_support_ort.py \
    --onnx=/path/to/identity_neural_network.onnx \
    --custom_ops=IdentityConv
```

This will generate another ONNX file (`*_ort_support.onnx`) with the `trt.plugins` updated domain in the custom ops and in the graph itself, which
is necessary for ORT to recognize custom ops as a TensorRT plugin.

**Step 3**: Quantize the post-processed ONNX model. We will be using the `libidentity_conv_iplugin_v2.so` plugin for this example

```bash
python -m modelopt.onnx.quantization \
    --onnx_path=/path/to/identity_neural_network_ort_support.onnx \
    --trt_plugins=/path/to/libidentity_conv_iplugin_v2.so
```

**Step 4**: Deploy the quantized model with TensorRT.

```bash
trtexec --onnx=/path/to/identity_neural_network_ort_support.onnx \
    --staticPlugins=/path/to/libidentity_conv_iplugin_v2.so
```

## End-to-end example with Python Plugin

The python_plugin_example.py file contains an end-to-end example that quantizes an ONNX model with a custom operation, implemented via a Python plugin, and runs inference in the ONNX model via ONNX Runtime (ORT). The inference is run for random inputs.
In this example, we have the `dino_fan_small`, an image segmentation ONNX model which uses the `multi_scale_deformable_attn_pytorch()` plugin.
The model can be downloaded from the [NGC catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/pretrained_dino_coco/files?version=dino_fan_small_deployable_v1.0).
It can be quantized and run with the following script:

```bash
python python_plugin_example.py --onnx_path=<path to the dino_fan_small ONNX model>
```
