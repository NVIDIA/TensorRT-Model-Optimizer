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

- [2025/04/05] [NVIDIA Accelerates Inference on Meta Llama 4 Scout and Maverick](https://developer.nvidia.com/blog/nvidia-accelerates-inference-on-meta-llama-4-scout-and-maverick/). Check out how to quantize Llama4 for deployment acceleration [here](./examples/llm_ptq/README.md#llama-4)
- [2025/03/18] [World's Fastest DeepSeek-R1 Inference with Blackwell FP4 & Increasing Image Generation Efficiency on Blackwell](https://developer.nvidia.com/blog/nvidia-blackwell-delivers-world-record-deepseek-r1-inference-performance/)
- [2025/02/25] Model Optimizer quantized NVFP4 models available on Hugging Face for download: [DeepSeek-R1-FP4](https://huggingface.co/nvidia/DeepSeek-R1-FP4), [Llama-3.3-70B-Instruct-FP4](https://huggingface.co/nvidia/Llama-3.3-70B-Instruct-FP4), [Llama-3.1-405B-Instruct-FP4](https://huggingface.co/nvidia/Llama-3.1-405B-Instruct-FP4)
- [2025/01/28] Model Optimizer has added support for NVFP4. Check out an example of NVFP4 PTQ [here](./examples/llm_ptq/README.md#model-quantization-and-trt-llm-conversion).
- [2025/01/28] Model Optimizer is now open source!
- [2024/10/23] Model Optimizer quantized FP8 Llama-3.1 Instruct models available on Hugging Face for download: [8B](https://huggingface.co/nvidia/Llama-3.1-8B-Instruct-FP8), [70B](https://huggingface.co/nvidia/Llama-3.1-70B-Instruct-FP8), [405B](https://huggingface.co/nvidia/Llama-3.1-405B-Instruct-FP8).
- [2024/09/10] [Post-Training Quantization of LLMs with NVIDIA NeMo and TensorRT Model Optimizer](https://developer.nvidia.com/blog/post-training-quantization-of-llms-with-nvidia-nemo-and-nvidia-tensorrt-model-optimizer/).
- [2024/08/28] [Boosting Llama 3.1 405B Performance up to 44% with TensorRT Model Optimizer on NVIDIA H200 GPUs](https://developer.nvidia.com/blog/boosting-llama-3-1-405b-performance-by-up-to-44-with-nvidia-tensorrt-model-optimizer-on-nvidia-h200-gpus/)
- [2024/08/28] [Up to 1.9X Higher Llama 3.1 Performance with Medusa](https://developer.nvidia.com/blog/low-latency-inference-chapter-1-up-to-1-9x-higher-llama-3-1-performance-with-medusa-on-nvidia-hgx-h200-with-nvlink-switch/)
- [2024/08/15] New features in recent releases: [Cache Diffusion](https://github.com/NVIDIA/TensorRT-Model-Optimizer/tree/main/examples/diffusers/cache_diffusion), [QLoRA workflow with NVIDIA NeMo](https://docs.nvidia.com/nemo-framework/user-guide/24.09/sft_peft/qlora.html), and more. Check out [our blog](https://developer.nvidia.com/blog/nvidia-tensorrt-model-optimizer-v0-15-boosts-inference-performance-and-expands-model-support/) for details.
- [2024/06/03] Model Optimizer now has an experimental feature to deploy to vLLM as part of our effort to support popular deployment frameworks. Check out the workflow [here](./examples/llm_ptq/README.md#deploy-fp8-quantized-model-using-vllm)

<details close>
<summary>Previous News</summary>

- [2024/05/08] [Announcement: Model Optimizer Now Formally Available to Further Accelerate GenAI Inference Performance](https://developer.nvidia.com/blog/accelerate-generative-ai-inference-performance-with-nvidia-tensorrt-model-optimizer-now-publicly-available/)
- [2024/03/27] [Model Optimizer supercharges TensorRT-LLM to set MLPerf LLM inference records](https://developer.nvidia.com/blog/nvidia-h200-tensor-core-gpus-and-nvidia-tensorrt-llm-set-mlperf-llm-inference-records/)
- [2024/03/18] [GTC Session: Optimize Generative AI Inference with Quantization in TensorRT-LLM and TensorRT](https://www.nvidia.com/en-us/on-demand/session/gtc24-s63213/)
- [2024/03/07] [Model Optimizer's 8-bit Post-Training Quantization enables TensorRT to accelerate Stable Diffusion to nearly 2x faster](https://developer.nvidia.com/blog/tensorrt-accelerates-stable-diffusion-nearly-2x-faster-with-8-bit-post-training-quantization/)
- [2024/02/01] [Speed up inference with Model Optimizer quantization techniques in TRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/quantization-in-TRT-LLM.md)

</details>

## Table of Contents

- [Model Optimizer Overview](#model-optimizer-overview)
- [Installation](#installation--docker)
- [Techniques](#techniques)
  - [Quantization](#quantization-examples-docs)
    - [Quantized Checkpoints](#quantized-checkpoints)
  - [Pruning](#pruning-examples-docs)
  - [Distillation](#distillation-examples-docs)
  - [Speculative Decoding](#speculative-decoding-examples-docs)
  - [Sparsity](#sparsity-examples-docs)
- [Examples](#examples)
- [Support Matrix](#model-support-matrix)
- [Benchmark](#benchmark)
- [Roadmap](#roadmap)
- [Release Notes](#release-notes)
- [Contributing](#contributing)

## Model Optimizer Overview

Minimizing inference costs presents a significant challenge as generative AI models continue to grow in complexity and size.
The **NVIDIA TensorRT Model Optimizer** (referred to as **Model Optimizer**, or **ModelOpt**) is a library comprising state-of-the-art model optimization techniques including [quantization](#quantization), [distillation](#distillation), [pruning](#pruning), [speculative decoding](#speculative-decoding) and [sparsity](#sparsity) to accelerate models.
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

**NOTE:** Unless specified otherwise, all example READMEs assume they are using the above ModelOpt docker image for running the examples. The example-specific dependencies are required to be installed separately from their respective `requirements.txt` files if not using the ModelOpt docker image.

## Techniques

Below is a short description of the techniques supported by Model Optimizer.

### Quantization \[[examples](./examples/README.md#quantization)\] \[[docs](https://nvidia.github.io/TensorRT-Model-Optimizer/guides/1_quantization.html)\]

Quantization is an effective model optimization technique for large models. Quantization with Model Optimizer can compress model size by 2x-4x, speeding up inference while preserving model quality. Model Optimizer enables highly performant quantization formats including NVFP4, FP8, INT8, INT4, etc and supports advanced algorithms such as SmoothQuant, AWQ, SVDQuant, and Double Quantization with easy-to-use Python APIs. Both Post-training quantization (PTQ) and Quantization-aware training (QAT) are supported.

#### Quantized Checkpoints

[Quantized checkpoints](https://huggingface.co/collections/nvidia/model-optimizer-66aa84f7966b3150262481a4) in Hugging Face model hub are ready for [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) and [vLLM](https://github.com/vllm-project/vllm) deployments. More models coming soon.

### Pruning \[[examples](./examples/README.md#pruning)\] \[[docs](https://nvidia.github.io/TensorRT-Model-Optimizer/guides/2_pruning.html)\]

Pruning is a technique to reduce the model size and accelerate the inference by removing unnecessary weights. Model Optimizer provides Python APIs to prune Linear and Conv layers, and Transformer attention heads, MLP, embedding hidden size and number of layers (depth).

### Distillation \[[examples](./examples/README.md#distillation)\] \[[docs](https://nvidia.github.io/TensorRT-Model-Optimizer/guides/4_distillation.html)\]

Knowledge Distillation allows for increasing the accuracy and/or convergence speed of a desired model architecture
by using a more powerful model's learned features to guide a student model's objective function into imitating it.

### Speculative Decoding \[[examples](./examples/README.md#speculative-decoding)\] \[[docs](https://nvidia.github.io/TensorRT-Model-Optimizer/guides/7_speculative_decoding.html)\]

Speculative Decoding enables your model to generate multiple tokens in each generate step by using a draft model to predict tokens that are then validated by the original model in a single forward pass.
This can be useful for reducing the latency of your model and speeding up inference.
Currently, Model Optimizer supports Medusa and EAGLE speculative decoding algorithms.

### Sparsity \[[examples](./examples/README.md#sparsity)\] \[[docs](https://nvidia.github.io/TensorRT-Model-Optimizer/guides/5_sparsity.html)\]

Sparsity is a technique to further reduce the memory footprint of deep learning models and accelerate the inference. Model Optimizer provides Python APIs to apply weight sparsity to a given model. It also supports [NVIDIA 2:4 sparsity pattern](https://arxiv.org/pdf/2104.08378) and various sparsification methods, such as [NVIDIA ASP](https://github.com/NVIDIA/apex/tree/master/apex/contrib/sparsity) and [SparseGPT](https://arxiv.org/abs/2301.00774).

## Examples

Please see the [complete list of examples](./examples/README.md).

## Model Support Matrix

| Model Type | Support Matrix |
|------------|----------------|
| LLM Quantization | [View Support Matrix](./examples/llm_ptq/README.md#model-support-list) |
| VLM Quantization | [View Support Matrix](./examples/vlm_ptq/README.md#model-support-list) |
| Diffusion Models | [View Support Matrix](./examples/diffusers/README.md#model-support-list) |
| Speculative Decoding | [View Support Matrix](./examples/speculative_decoding/README.md#model-support-list) |

## Benchmark

Please find the [detailed performance benchmarks](./examples/benchmark.md).

## Roadmap

Please see our [product roadmap](https://github.com/NVIDIA/TensorRT-Model-Optimizer/issues/146).

## Release Notes

Please see Model Optimizer Changelog [here](https://nvidia.github.io/TensorRT-Model-Optimizer/reference/0_changelog.html).

## Contributing

Model Optimizer is now open source! We welcome any feedback, feature requests and PRs. We will soon be adding more examples/tutorials/explanations for faster integration!
Please review our [Contributing](./CONTRIBUTING.md) guidelines before submitting a PR.
