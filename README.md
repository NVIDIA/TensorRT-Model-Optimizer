<div align="center">

# NVIDIA TensorRT Model Optimizer

#### A Library to Quantize and Compress Deep Learning Models for Optimized Inference on GPUs

[![Documentation](https://img.shields.io/badge/Documentation-latest-brightgreen.svg?style=flat)](https://nvidia.github.io/TensorRT-Model-Optimizer)
[![version](https://img.shields.io/pypi/v/nvidia-modelopt?label=Release)](https://pypi.org/project/nvidia-modelopt/)
[![license](https://img.shields.io/badge/License-MIT-blue)](./LICENSE)

[Examples](#examples) |
[Benchmark Results](./benchmark.md) |
[Documentation](https://nvidia.github.io/TensorRT-Model-Optimizer)

</div>

## Latest News

- \[2024/8/28\] [Boosting Llama 3.1 405B Performance up to 44% with TensorRT Model Optimizer on NVIDIA H200 GPUs](https://developer.nvidia.com/blog/boosting-llama-3-1-405b-performance-by-up-to-44-with-nvidia-tensorrt-model-optimizer-on-nvidia-h200-gpus/)
- \[2024/8/28\] [Up to 1.9X Higher Llama 3.1 Performance with Medusa](https://developer.nvidia.com/blog/low-latency-inference-chapter-1-up-to-1-9x-higher-llama-3-1-performance-with-medusa-on-nvidia-hgx-h200-with-nvlink-switch/)
- \[2024/08/15\] New features in recent releases: [Cache Diffusion](https://github.com/NVIDIA/TensorRT-Model-Optimizer/tree/main/diffusers/cache_diffusion), [QLoRA workflow with NVIDIA NeMo](https://docs.nvidia.com/nemo-framework/user-guide/latest/sft_peft/qlora.html), and more. Check out [our blog](https://developer.nvidia.com/blog/nvidia-tensorrt-model-optimizer-v0-15-boosts-inference-performance-and-expands-model-support/) for details.
- \[2024/06/03\] Model Optimizer now has an experimental feature to deploy to vLLM as part of our effort to support popular deployment frameworks. Check out the workflow [here](./llm_ptq/README.md#deploy-fp8-quantized-model-using-vllm)
- \[2024/05/08\] [Announcement: Model Optimizer Now Formally Available to Further Accelerate GenAI Inference Performance](https://developer.nvidia.com/blog/accelerate-generative-ai-inference-performance-with-nvidia-tensorrt-model-optimizer-now-publicly-available/)
- \[2024/03/27\] [Model Optimizer supercharges TensorRT-LLM to set MLPerf LLM inference records](https://developer.nvidia.com/blog/nvidia-h200-tensor-core-gpus-and-nvidia-tensorrt-llm-set-mlperf-llm-inference-records/)
- \[2024/03/18\] [GTC Session: Optimize Generative AI Inference with Quantization in TensorRT-LLM and TensorRT](https://www.nvidia.com/en-us/on-demand/session/gtc24-s63213/)
- \[2024/03/07\] [Model Optimizer's 8-bit Post-Training Quantization enables TensorRT to accelerate Stable Diffusion to nearly 2x faster](https://developer.nvidia.com/blog/tensorrt-accelerates-stable-diffusion-nearly-2x-faster-with-8-bit-post-training-quantization/)
- \[2024/02/01\] [Speed up inference with Model Optimizer quantization techniques in TRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/quantization-in-TRT-LLM.md)

## Table of Contents

- [Model Optimizer Overview](#model-optimizer-overview)
- [Installation](#installation)
- [Techniques](#techniques)
  - [Quantization](#quantization)
  - [Sparsity](#sparsity)
  - [Distillation](#distillation)
  - [Pruning](#pruning)
- [Examples](#examples)
- [Support Matrix](#support-matrix)
- [Benchmark](#benchmark)
- [Release Notes](#release-notes)

## Model Optimizer Overview

Minimizing inference costs presents a significant challenge as generative AI models continue to grow in complexity and size.
The **NVIDIA TensorRT Model Optimizer** (referred to as **Model Optimizer**, or **ModelOpt**) is a library comprising state-of-the-art model optimization techniques including [quantization](#quantization), [sparsity](#sparsity), [distillation](#distillation), and [pruning](#pruning) to compress models.
It accepts a torch or [ONNX](https://github.com/onnx/onnx) model as inputs and provides Python APIs for users to easily stack different model optimization techniques to produce an optimized quantized checkpoint.
Seamlessly integrated within the NVIDIA AI software ecosystem, the quantized checkpoint generated from Model Optimizer is ready for deployment in downstream inference frameworks like [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/quantization) or [TensorRT](https://github.com/NVIDIA/TensorRT).
Further integrations are planned for [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) and [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) for training-in-the-loop optimization techniques.
For enterprise users, the 8-bit quantization with Stable Diffusion is also available on [NVIDIA NIM](https://developer.nvidia.com/blog/nvidia-nim-offers-optimized-inference-microservices-for-deploying-ai-models-at-scale/).

Model Optimizer is available for free for all developers on [NVIDIA PyPI](https://pypi.org/project/nvidia-modelopt/). This repository is for sharing examples and GPU-optimized recipes as well as collecting feedback from the community.

## Installation

### [PIP](https://pypi.org/project/nvidia-modelopt/)

```bash
pip install "nvidia-modelopt[all]~=0.17.0" --extra-index-url https://pypi.nvidia.com
```

See the [installation guide](https://nvidia.github.io/TensorRT-Model-Optimizer/getting_started/2_installation.html) for more fine-grained control over the installation.

Make sure to also install example-specific dependencies from their respective `requirements.txt` files if any.

### Docker

After installing the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit),
please run the following commands to build the Model Optimizer example docker container which has all the necessary
dependencies pre-installed for running the examples.

```bash
# Build the docker
docker/build.sh

# Obtain and start the basic docker image environment.
# The default built docker image is docker.io/library/modelopt_examples:latest
docker run --gpus all -it --shm-size 20g --rm docker.io/library/modelopt_examples:latest bash

# Check installation
python -c "import modelopt"
```

NOTE: Unless specified otherwise, all example READMEs assume they are using the ModelOpt docker image for running the examples.

Alternatively for PyTorch, you can also use [NVIDIA NGC PyTorch container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags) with Model Optimizer pre-installed starting from 24.06 container. Make sure to update the Model Optimizer version to the latest one if not already.

## Techniques

### Quantization

Quantization is an effective model optimization technique for large models. Quantization with Model Optimizer can compress model size by 2x-4x, speeding up inference while preserving model quality. Model Optimizer enables highly performant quantization formats including FP8, INT8, INT4, etc and supports advanced algorithms such as SmoothQuant, AWQ, and Double Quantization with easy-to-use Python APIs. Both Post-training quantization (PTQ) and Quantization-aware training (QAT) are supported.

### Sparsity

Sparsity is a technique to further reduce the memory footprint of deep learning models and accelerate the inference. Model Optimizer Python APIs to apply weight sparsity to a given model. It also supports [NVIDIA 2:4 sparsity pattern](https://arxiv.org/pdf/2104.08378) and various sparsification methods, such as [NVIDIA ASP](https://github.com/NVIDIA/apex/tree/master/apex/contrib/sparsity) and [SparseGPT](https://arxiv.org/abs/2301.00774).

### Pruning

Pruning is a technique to reduce the model size and accelerate the inference by removing unnecessary weights. Model Optimizer provides Python APIs to prune Linear and Conv layers, and Transformer attention heads, MLP, and depth.

### Distillation

Knowledge Distillation allows for increasing the accuracy and/or convergence speed of a desired model architecture
by using a more powerful model's learned features to guide a student model's objective function into imitating it.

## Examples

- [PTQ for LLMs](./llm_ptq/README.md) covers how to use Post-training quantization (PTQ) and export to [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) for deployment for popular pre-trained models from frameworks like
  - [Hugging Face](https://huggingface.co/docs/hub/en/models-the-hub)
  - [NVIDIA NeMo](https://github.com/NVIDIA/NeMo)
  - [NVIDIA Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
  - [Medusa](https://github.com/FasterDecoding/Medusa)
- [PTQ for Diffusers](./diffusers/quantization/README.md) walks through how to quantize a diffusion model with FP8 or INT8, export to ONNX, and deploy with [TensorRT](https://github.com/NVIDIA/TensorRT/tree/release/10.0/demo/Diffusion). The Diffusers example in this repo is complementary to the [demoDiffusion example in TensorRT repo](https://github.com/NVIDIA/TensorRT/tree/release/10.0/demo/Diffusion#introduction) and includes FP8 plugins as well as the latest updates on INT8 quantization.
- [QAT for LLMs](./llm_qat/README.md) demonstrates the recipe and workflow for Quantization-aware Training (QAT), which can further preserve model accuracy at low precisions (e.g., INT4, or 4-bit in [NVIDIA Blackwell platform](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)).
- [Sparsity for LLMs](./llm_sparsity/README.md) shows how to perform Post-training Sparsification and Sparsity-aware fine-tuning on a pre-trained Hugging Face model.
- [Pruning](./pruning/README.md) demonstrates how to optimally prune Linear and Conv layers, and Transformer attention heads, MLP, and depth using the Model Optimizer for following frameworks:
  - [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) / [NVIDIA Megatron-LM](https://github.com/NVIDIA/Megatron-LM) GPT-style models (e.g. Llama 3, Mistral NeMo, etc.)
  - Hugging Face language models like BERT and GPT-J
  - Computer Vision models like [NVIDIA Tao](https://developer.nvidia.com/tao-toolkit) framework detection models.
- [ONNX PTQ](./onnx_ptq/README.md) shows how to quantize the ONNX models in INT4 or INT8 quantization mode. The examples also include the deployment of quantized ONNX models using TensorRT.
- [Distillation for LLMs](./llm_distill/README.md) demonstrates how to use Knowledge Distillation, which can increasing the accuracy and/or convergence speed for finetuning / QAT.
- [Chained Optimizations](./chained_optimizations/README.md) shows how to chain multiple optimizations together (e.g. Pruning + Distillation + Quantization).

## Support Matrix

- For LLM quantization, please refer to this [support matrix](./llm_ptq/README.md#model-support-list).
- For Diffusion, the Model Optimizer supports [Stable Diffusion 1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5), [Stable Diffusion XL](https://huggingface.co/papers/2307.01952), and [SDXL-Turbo](https://huggingface.co/stabilityai/sdxl-turbo).

## Benchmark

Please find the benchmarks [here](./benchmark.md).

## Release Notes

Please see Model Optimizer Changelog [here](https://nvidia.github.io/TensorRT-Model-Optimizer/reference/0_changelog.html).
