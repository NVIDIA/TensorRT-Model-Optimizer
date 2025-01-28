<div align="center">

# NVIDIA TensorRT Model Optimizer

#### A Library to Quantize and Compress Deep Learning Models for Optimized Inference on GPUs

[![Documentation](https://img.shields.io/badge/Documentation-latest-brightgreen.svg?style=flat)](https://nvidia.github.io/TensorRT-Model-Optimizer)
[![version](https://img.shields.io/pypi/v/nvidia-modelopt?label=Release)](https://pypi.org/project/nvidia-modelopt/)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue)](./LICENSE)

[Examples](#examples) |
[Documentation](https://nvidia.github.io/TensorRT-Model-Optimizer) |
[Benchmark Results](#benchmark) |
[Roadmap](#roadmap) |
[ModelOpt-Windows](./examples/windows/README.md)

</div>

## Latest News

- \[2024/10/23\] Quantized FP8 Llama-3.1 Instruct models available on Hugging Face for download: [8B](https://huggingface.co/nvidia/Llama-3.1-8B-Instruct-FP8), [70B](https://huggingface.co/nvidia/Llama-3.1-70B-Instruct-FP8), [405B](https://huggingface.co/nvidia/Llama-3.1-405B-Instruct-FP8)
- \[2024/9/10\] [Post-Training Quantization of LLMs with NVIDIA NeMo and TensorRT Model Optimizer](https://developer.nvidia.com/blog/post-training-quantization-of-llms-with-nvidia-nemo-and-nvidia-tensorrt-model-optimizer/)
- \[2024/8/28\] [Boosting Llama 3.1 405B Performance up to 44% with TensorRT Model Optimizer on NVIDIA H200 GPUs](https://developer.nvidia.com/blog/boosting-llama-3-1-405b-performance-by-up-to-44-with-nvidia-tensorrt-model-optimizer-on-nvidia-h200-gpus/)
- \[2024/8/28\] [Up to 1.9X Higher Llama 3.1 Performance with Medusa](https://developer.nvidia.com/blog/low-latency-inference-chapter-1-up-to-1-9x-higher-llama-3-1-performance-with-medusa-on-nvidia-hgx-h200-with-nvlink-switch/)
- \[2024/08/15\] New features in recent releases: [Cache Diffusion](https://github.com/NVIDIA/TensorRT-Model-Optimizer/tree/main/examples/diffusers/cache_diffusion), [QLoRA workflow with NVIDIA NeMo](https://docs.nvidia.com/nemo-framework/user-guide/latest/sft_peft/qlora.html), and more. Check out [our blog](https://developer.nvidia.com/blog/nvidia-tensorrt-model-optimizer-v0-15-boosts-inference-performance-and-expands-model-support/) for details.
- \[2024/06/03\] Model Optimizer now has an experimental feature to deploy to vLLM as part of our effort to support popular deployment frameworks. Check out the workflow [here](./examples/llm_ptq/README.md#deploy-fp8-quantized-model-using-vllm)
- \[2024/05/08\] [Announcement: Model Optimizer Now Formally Available to Further Accelerate GenAI Inference Performance](https://developer.nvidia.com/blog/accelerate-generative-ai-inference-performance-with-nvidia-tensorrt-model-optimizer-now-publicly-available/)
- \[2024/03/27\] [Model Optimizer supercharges TensorRT-LLM to set MLPerf LLM inference records](https://developer.nvidia.com/blog/nvidia-h200-tensor-core-gpus-and-nvidia-tensorrt-llm-set-mlperf-llm-inference-records/)
- \[2024/03/18\] [GTC Session: Optimize Generative AI Inference with Quantization in TensorRT-LLM and TensorRT](https://www.nvidia.com/en-us/on-demand/session/gtc24-s63213/)
- \[2024/03/07\] [Model Optimizer's 8-bit Post-Training Quantization enables TensorRT to accelerate Stable Diffusion to nearly 2x faster](https://developer.nvidia.com/blog/tensorrt-accelerates-stable-diffusion-nearly-2x-faster-with-8-bit-post-training-quantization/)
- \[2024/02/01\] [Speed up inference with Model Optimizer quantization techniques in TRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/quantization-in-TRT-LLM.md)

## Table of Contents

- [Model Optimizer Overview](#model-optimizer-overview)
- [Installation](#installation--docker)
- [Techniques](#techniques)
  - [Quantization](#quantization)
  - [Distillation](#distillation)
  - [Pruning](#pruning)
  - [Sparsity](#sparsity)
- [Examples](#examples)
- [Support Matrix](#model-support-matrix)
- [Benchmark](#benchmark)
- [Quantized Checkpoints](#quantized-checkpoints)
- [Roadmap](#roadmap)
- [Release Notes](#release-notes)
- [Contributing](#contributing)

## Model Optimizer Overview

Minimizing inference costs presents a significant challenge as generative AI models continue to grow in complexity and size.
The **NVIDIA TensorRT Model Optimizer** (referred to as **Model Optimizer**, or **ModelOpt**) is a library comprising state-of-the-art model optimization techniques including [quantization](#quantization), [distillation](#distillation), [pruning](#pruning), and [sparsity](#sparsity) to compress models.
It accepts a torch or [ONNX](https://github.com/onnx/onnx) model as inputs and provides Python APIs for users to easily stack different model optimization techniques to produce an optimized quantized checkpoint.
Seamlessly integrated within the NVIDIA AI software ecosystem, the quantized checkpoint generated from Model Optimizer is ready for deployment in downstream inference frameworks like [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/quantization) or [TensorRT](https://github.com/NVIDIA/TensorRT).
ModelOpt is integrated with [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) and [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) for training-in-the-loop optimization techniques.
For enterprise users, the 8-bit quantization with Stable Diffusion is also available on [NVIDIA NIM](https://developer.nvidia.com/blog/nvidia-nim-offers-optimized-inference-microservices-for-deploying-ai-models-at-scale/).

## Installation / Docker

To use Model Optimizer with full dependencies (e.g. TensorRT-LLM deployment), we recommend using the provided docker image.

After installing the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
please run the following commands to build the Model Optimizer docker container which has all the necessary
dependencies pre-installed for running the examples.

```bash
# Clone the ModelOpt repository
git clone https://github.com/NVIDIA/TensorRT-Model-Optimizer.git
cd TensorRT-Model-Optimizer

# Build the docker (will be tagged `docker.io/library/modelopt_examples:latest`)
# You may customize `docker/Dockerfile` to include or exclude certain dependencies you may or may not need.
./docker/build.sh

# Run the docker image
docker run --gpus all -it --shm-size 20g --rm docker.io/library/modelopt_examples:latest bash

# Check installation (inside the docker container)
python -c "import modelopt; print(modelopt.__version__)"
```

Alternatively, you can install it from [NVIDIA PyPI](https://pypi.org/project/nvidia-modelopt/) without TRT-LLM etc.

```bash
pip install "nvidia-modelopt[all]" -U --extra-index-url https://pypi.nvidia.com
```

To install from source for local development, you can install it as follows:

```bash
pip install -e ".[all]" --extra-index-url https://pypi.nvidia.com
```

When installing from source, please make sure to re-run the install command everytime you pull new changes in the repository so dependencies are also updated.

See the [installation guide](https://nvidia.github.io/TensorRT-Model-Optimizer/getting_started/2_installation.html) for more details on alternate pre-built docker images or installation in a local environment.

NOTE: Unless specified otherwise, all example READMEs assume they are using the above ModelOpt docker image for running the examples. The example specific dependencies are required to be install separately from their respective `requirements.txt` files if not using the ModelOpt's docker image.

## Techniques

### Quantization

Quantization is an effective model optimization technique for large models. Quantization with Model Optimizer can compress model size by 2x-4x, speeding up inference while preserving model quality. Model Optimizer enables highly performant quantization formats including FP8, INT8, INT4, etc and supports advanced algorithms such as SmoothQuant, AWQ, and Double Quantization with easy-to-use Python APIs. Both Post-training quantization (PTQ) and Quantization-aware training (QAT) are supported.

### Distillation

Knowledge Distillation allows for increasing the accuracy and/or convergence speed of a desired model architecture
by using a more powerful model's learned features to guide a student model's objective function into imitating it.

### Pruning

Pruning is a technique to reduce the model size and accelerate the inference by removing unnecessary weights. Model Optimizer provides Python APIs to prune Linear and Conv layers, and Transformer attention heads, MLP, embedding hidden size and number of layers (depth).

### Sparsity

Sparsity is a technique to further reduce the memory footprint of deep learning models and accelerate the inference. Model Optimizer Python APIs to apply weight sparsity to a given model. It also supports [NVIDIA 2:4 sparsity pattern](https://arxiv.org/pdf/2104.08378) and various sparsification methods, such as [NVIDIA ASP](https://github.com/NVIDIA/apex/tree/master/apex/contrib/sparsity) and [SparseGPT](https://arxiv.org/abs/2301.00774).

## Examples

Please see examples [here](./examples/README.md).

## Model Support Matrix

- For LLM quantization, please refer to this [support matrix](./examples/llm_ptq/README.md#model-support-list).
- For VLM quantization, please refer to this [support matrix](./examples/vlm_ptq/README.md#model-support-list).
- For Diffusion, Model Optimizer supports [FLUX](https://huggingface.co/black-forest-labs/FLUX.1-dev), [Stable Diffusion 3](https://huggingface.co/stabilityai/stable-diffusion-3-medium), [Stable Diffusion XL](https://huggingface.co/papers/2307.01952), [SDXL-Turbo](https://huggingface.co/stabilityai/sdxl-turbo), and [Stable Diffusion 2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1).
- For speculative decoding, please refer to this [support matrix](./examples/speculative_decoding/README.md#model-support-list).

## Benchmark

Please find the benchmarks at [here](./examples/benchmark.md).

## Quantized Checkpoints

[Quantized checkpoints](https://huggingface.co/collections/nvidia/model-optimizer-66aa84f7966b3150262481a4) in Hugging Face model hub are ready for [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) and [vLLM](https://github.com/vllm-project/vllm) deployments. More models coming soon.

## Roadmap

Please see our [product roadmap](https://github.com/NVIDIA/TensorRT-Model-Optimizer/issues/108).

## Release Notes

Please see Model Optimizer Changelog [here](https://nvidia.github.io/TensorRT-Model-Optimizer/reference/0_changelog.html).

## Contributing

At the moment, we are not accepting external contributions. However, this will soon change with a focus on extensibility. We welcome any feedback and feature requests. Please open an issue if you have any suggestions or questions.
