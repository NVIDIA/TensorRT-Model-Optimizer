# Post-training quantization (PTQ) for Vision Language Models

To learn more about the quantization feature, please refer to the [documentation](https://nvidia.github.io/TensorRT-Model-Optimizer/guides/1_quantization.html).

Quantization is an effective model optimization technique that compresses your models. Quantization with Model Optimizer can compress model size by 2x-4x, speeding up inference while preserving model quality. \
Model Optimizer enables highly performant quantization formats including NVFP4, FP8, INT8, INT4 and supports advanced algorithms such as SmoothQuant, AWQ, SVDQuant, and Double Quantization with easy-to-use Python APIs.

This section focuses on Post-training quantization for VLM (Vision Language Models), a technique that reduces model precision after training to improve inference efficiency without requiring retraining.

<div align="center">

| **Section** | **Description** | **Link** | **Docs** |
| :------------: | :------------: | :------------: | :------------: |
| Pre-Requisites | Required & optional packages to use this technique | \[[Link](#pre-requisites)\] | |
| Getting Started | Learn how to optimize your models using PTQ to reduce precision and improve inference efficiency | \[[Link](#getting-started)\] | \[[docs](https://nvidia.github.io/TensorRT-Model-Optimizer/guides/1_quantization.html)\] |
| Support Matrix | View the support matrix to see quantization compatibility and feature availability across different models | \[[Link](#support-matrix)\] | |
| Framework Scripts | Example scripts demonstrating quantization techniques for optimizing Hugging Face / NeMo / Megatron-LM models | \[[Link](#framework-scripts)\] | |
| Pre-Quantized Checkpoints | Ready to deploy Hugging Face pre-quantized checkpoints | \[[Link](#pre-quantized-checkpoints)\] | |
| Resources | Extra links to relevant resources | \[[Link](#resources)\] | |

</div>

## Pre-Requisites

Please refer to the [llm_ptq/README.md](../llm_ptq/README.md#pre-requisites) for the pre-requisites.

## Getting Started

Please refer to the [llm_ptq/README.md](../llm_ptq/README.md#getting-started) for the getting-started.

## Support Matrix

### Current out of the box configs

Please refer to the [llm_ptq/README.md](../llm_ptq/README.md#current-out-of-the-box-configs) for details on the current out of the box configs.

### Supported Models

| Model | type | fp8 | int8_sq | int4_awq | w4a8_awq<sup>1</sup> | nvfp4<sup>2</sup> |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Llava | llava | ✅ | ✅ | ✅ | ✅ | ❌ |
| VILA | vila | ✅ | ✅ | ✅ | ✅ | ❌ |
| Phi-3-vision | phi | ✅ | ✅ | ✅ | ✅ | ❌ |

> *<sup>1.</sup>The w4a8_awq is an experimental quantization scheme that may result in a higher accuracy penalty.* \
> *<sup>2.</sup>A selective set of the popular models are internally tested. The actual model support list may be longer. NVFP4 inference requires Blackwell GPUs and TensorRT-LLM v0.17 or later.*

> *The accuracy loss after PTQ may vary depending on the actual model and the quantization method. Different models may have different accuracy loss and usually the accuracy loss is more significant when the base model is small. If the accuracy after PTQ is not meeting the requirement, please try either modifying [hf_ptq.py](../llm_ptq/hf_ptq.py) and disabling the KV cache quantization or using the [QAT](./../llm_qat/README.md) instead.*

## Framework Scripts

Please refer to the [llm_ptq/README.md](../llm_ptq/README.md) about the details of model quantization.

The following scripts provide an all-in-one and step-by-step model quantization example for Llava, VILA and Phi-3-vision models. The quantization format and the number of GPUs will be supplied as inputs to these scripts. By default, we build the engine for the fp8 format and 1 GPU.

### Hugging Face Example [Script](./scripts/huggingface_example.sh)

For [Llava](https://huggingface.co/llava-hf/llava-1.5-7b-hf):

```bash
git clone https://huggingface.co/llava-hf/llava-1.5-7b-hf
scripts/huggingface_example.sh --type llava --model llava-1.5-7b-hf --quant [fp8|int8_sq|int4_awq|w4a8_awq] --tp [1|2|4|8]
```

For VILA models like [VILA1.5-3b](https://huggingface.co/Efficient-Large-Model/VILA1.5-3b):

```bash
git clone https://huggingface.co/Efficient-Large-Model/VILA1.5-3b vila1.5-3b
scripts/huggingface_example.sh --type vila --model vila1.5-3b --quant [fp8|int8_sq|int4_awq|w4a8_awq] --tp [1|2|4|8]
```

For [Phi-3-vision](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct):

```bash
git clone https://huggingface.co/microsoft/Phi-3-vision-128k-instruct
scripts/huggingface_example.sh --type phi --model Phi-3-vision-128k-instruct --quant [fp8|int8_sq|int4_awq|w4a8_awq]
```

The example scripts above also have an additional flag `--tasks gqa`, which will trigger evaluation of the built TensorRT engine using GQA benchmark. Details of the evaluation is explained in this [tutorial](../vlm_eval/README.md).

If you encounter Out of Memory (OOM) issues during inference or evaluation, you can try lowering the `--kv_cache_free_gpu_memory_fraction` argument (default is 0.8) to reduce GPU memory usage for kv_cache:

```bash
scripts/huggingface_example.sh --type phi --model Phi-3-vision-128k-instruct --quant fp8 --kv_cache_free_gpu_memory_fraction 0.5
```

## Pre-Quantized Checkpoints

- Ready-to-deploy checkpoints \[[🤗 Hugging Face - Nvidia TensorRT Model Optimizer Collection](https://huggingface.co/collections/nvidia/model-optimizer-66aa84f7966b3150262481a4)\]
- Deployable on [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM), [vLLM](https://github.com/vllm-project/vllm) and [SGLang](https://github.com/sgl-project/sglang)
- More models coming soon!

## Resources

- 📅 [Roadmap](https://github.com/NVIDIA/TensorRT-Model-Optimizer/issues/146)
- 📖 [Documentation](https://nvidia.github.io/TensorRT-Model-Optimizer)
- 🎯 [Benchmarks](../benchmark.md)
- 💡 [Release Notes](https://nvidia.github.io/TensorRT-Model-Optimizer/reference/0_changelog.html)
- 🐛 [File a bug](https://github.com/NVIDIA/TensorRT-Model-Optimizer/issues/new?template=1_bug_report.md)
- ✨ [File a Feature Request](https://github.com/NVIDIA/TensorRT-Model-Optimizer/issues/new?template=2_feature_request.md)
