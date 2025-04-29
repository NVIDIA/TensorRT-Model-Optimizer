## Overview

The example script showcases how to utilize the **ModelOpt-Windows** toolkit for optimizing ONNX (Open Neural Network Exchange) models through quantization. This toolkit is designed for developers looking to enhance model performance, reduce size, and accelerate inference times, while preserving the accuracy of neural networks deployed with backends like DirectML on local RTX GPUs running Windows.

Quantization is a technique that converts models from floating-point to lower-precision formats, such as integers, which are more computationally efficient. This process can significantly speed up execution on supported hardware, while also reducing memory and bandwidth requirements.

This example takes an ONNX model as input, along with the necessary quantization settings, and generates a quantized ONNX model as output. This script can be used for quantizing popular, [ONNX Runtime GenAI](https://onnxruntime.ai/docs/genai) built Large Language Models (LLMs) in the ONNX format.

### Setup

1. Install ModelOpt-Windows. Refer [installation instructions](../README.md).

1. Install required dependencies

   ```bash
   pip install -r requirements.txt
   ```

### Prepare ORT-GenAI Compatible Base Model

You may generate the base model using the model builder that comes with onnxruntime-genai. The ORT-GenAI's [model-builder](https://github.com/microsoft/onnxruntime-genai/tree/main/src/python/py/models) downloads the original Pytorch model from Hugging Face, and produces an ONNX GenAI compatible base model in ONNX format. See example command-line below:

```bash
python -m onnxruntime_genai.models.builder -m meta-llama/Meta-Llama-3-8B -p fp16 -e dml -o E:\llama3-8b-fp16-dml-genai
```

### Quantization

To begin quantization, run the script like below:

```bash
python quantize.py --model_name=meta-llama/Meta-Llama-3-8B \
                          --onnx_path="E:\model_store\genai\llama3-8b-fp16-dml-genai\opset_21\model.onnx" \
                          --output_path="E:\model_store\genai\llama3-8b-fp16-dml-genai\opset_21\cnn_32_lite_0.1_16\model.onnx" \
                          --calib_size=32 --algo=awq_lite --dataset=cnn
```

#### Command Line Arguments

The table below lists key command-line arguments of the ONNX PTQ example script.

| **Argument** | **Supported Values** | **Description** |
|---------------------------|------------------------------------------------------|-------------------------------------------------------------|
| `--calib_size` | 32 (default), 64, 128 | Specifies the calibration size. |
| `--dataset` | cnn (default), pilevel | Choose calibration dataset: cnn_dailymail or pile-val. |
| `--algo` | awq_lite (default), awq_clip | Select the quantization algorithm. |
| `--onnx_path` | input .onnx file path | Path to the input ONNX model. |
| `--output_path` | output .onnx file path | Path to save the quantized ONNX model. |
| `--use_zero_point` | True, False (default) | Enable zero-point based quantization. |
| `--block-size` | 32, 64, 128 (default) | Block size for AWQ. |
| `--awqlite_alpha_step` | 0.1 (default) | Step-size for AWQ scale search, user-defined |
| `--awqlite_run_per_subgraph` | True, False (default) | If True, runs AWQ scale search at the subgraph level |
| `--awqlite_fuse_nodes` | True (default), False | If True, fuses input scales in parent nodes. |
| `--awqclip_alpha_step` | 0.05 (default) | Step-size for AWQ weight clipping, user-defined |
| `--awqclip_alpha_min` | 0.5 (default) | Minimum AWQ weight-clipping threshold, user-defined |
| `--awqclip_bsz_col` | 1024 (default) | Chunk size in columns during weight clipping, user-defined |
| `--calibration_eps` | dml, cuda, cpu (default: [dml,cpu]) | List of calibration endpoints. |

Run the following command to view all available parameters in the script:

```bash
python quantize.py --help
```

Please refer to `quantize.py` for further details on command-line parameters.

### Evaluate the Quantized Model

To evaluate the quantized model, please refer to the [accuracy benchmarking](../accuracy_benchmark/README.md) and [onnxruntime-genai performance benchmarking](https://github.com/microsoft/onnxruntime-genai/tree/main/benchmark/python).

### Deployment

Once an ONNX FP16 model is quantized using ModelOpt-Windows, the resulting quantized ONNX model can be deployed on the DirectML backend using [ORT-GenAI](https://onnxruntime.ai/) or [ORT](https://onnxruntime.ai/).

Refer to the following example scripts and tutorials for deployment:

1. [ORT GenAI examples](https://github.com/microsoft/onnxruntime-genai/tree/main/examples/python)
1. [ONNX Runtime documentation](https://onnxruntime.ai/docs/api/python/)

### Model Support Matrix

Please refer to [support matrix](https://nvidia.github.io/TensorRT-Model-Optimizer/guides/0_support_matrix.html) for a full list of supported features and models.

### Troubleshoot

1. **Configure Directories**

   - Update the `cache_dir` variable in the `main()` function to specify the path where you want to store Hugging Face files (optional).
   - If you're low on space on the C: drive, change the TMP and TEMP environment variable to a different drive (e.g., `D:\temp`).

1. **Authentication for Restricted Models**

   If the model you wish to use is hosted on Hugging Face and requires authentication, log in using the [huggingface-cli](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli#huggingface-cli-login) before running the quantization script.

   ```bash
   huggingface-cli login --token <HF_TOKEN>
   ```

1. **Check Read/Write Permissions**

   Ensure that both the input and output model paths have the necessary read and write permissions to avoid any permission-related errors.

1. **Check Output Path**

   Ensure that output .onnx file doesn't exist already. For example, if the output path is `C:\dir1\dir2\quant\model_quant.onnx` then the path `C:\dir1\dir2\quant` should be valid and the directory `quant` should not already contain `model_quant.onnx` file before quantization. If the output .onnx file already exists, then that can get appended during saving of the quantized model resulting in corrupted or invalid output model.

1. **Check Input Model**

   During INT4 AWQ execution, the input onnx model (one mentioned in `--onnx_path` argument) will be run with onnxruntime (ORT) for calibration (using ORT EP mentioned in `--calibration_eps` argument). So, make sure that input onnx model is running fine with the specified ORT EP.
