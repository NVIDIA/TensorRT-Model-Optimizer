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

The model can be quantized as an INT8 or INT4 model. For INT8 quantization, you have choice between `minmax` and `entropy` calibration algorithms and for INT4, [awq_clip](https://arxiv.org/abs/2306.00978) or [rtn_dq](https://ar5iv.labs.arxiv.org/html/2301.12017) can be chosen.

> *Note that, INT4 TensorRT engines are not performant yet compared to FP16 engines. Stay tuned for the next update.*

```bash
python -m modelopt.onnx.quantization \
    --onnx_path=vit_base_patch16_224.onnx \
    --quantize_mode=<int8|int4> \
    --calibration_data=calib.npy \
    --calibration_method=<minxmax|entropy|awq_clip|rtn_dq> \
    --output_path=vit_base_patch16_224.quant.onnx
```

# Evaluate the quantized ONNX model

The following evaluation requires the `val` directory of the [ImageNet dataset](https://www.kaggle.com/c/imagenet-object-localization-challenge/data). Alternatively, you can prepare it from [this](https://huggingface.co/datasets/mrm8488/ImageNet1K-val) Hugging Face dataset. Once you have it, the quantized ONNX ViT model can be evaluated on the ImageNet dataset as follows:

```bash
python evaluate_vit.py \
    --onnx_path=<path to the ONNX checkpoint> \
    --imagenet_path=<path to the ImageNet dataset> \
    --quantize_mode=<int8|int4>
```

This script converts the quantized ONNX model to a TensorRT engine and does the evaluation with that engine. Finally, the evaluation result will be reported as follows:

```bash
The top1 accuracy of the model is <accuracy score between 0-100%>
```

# INT4 LLM E2E example

This example requires high CPU and GPU memory, if you encounter OOM, please reduce the calibration data size or use a system with large memory. We will use `Llama-2-7b` from [Hugging Face](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf). First, you will need to authenticate your HuggingFace credentials:

```bash
huggingface-cli login
```

The LLAMA model can be downloaded with the following script:

```bash
python download_example_onnx.py --llama --output_path=Llama-2-7b-chat-hf-onnx/model.onnx
```

The model can be quantized to INT4 using the script below which uses `modelopt.onnx.quantization` python APIs.

```bash
python quantize_llama.py \
    --model_name=meta-llama/Llama-2-7b-chat-hf \
    --onnx_path=Llama-2-7b-chat-hf-onnx/model.onnx \
    --output_path=Llama-2-7b-chat-hf-onnx-quant/model.quant.onnx
```

After the INT4 model is generated at the specified output path, you can run the inference with llama deployed on TensorRT as follows:

```bash
python infer_llama.py \
    --model_name=meta-llama/Llama-2-7b-chat-hf \
    --onnx_path=Llama-2-7b-chat-hf-onnx-quant/model.quant.onnx \
    --prompt="I want to book a vacation to Hawaii. First, I need to "
```

This script will generate text by inferencing the INT4 engine for the given text prompt as follows:

```bash
TRT-llama response: ['I want to book a vacation to Hawaii. First, I need to 1.\nI want to book a vacation to Hawaii. First, I need to']
```
