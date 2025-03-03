# NVIDIA TensorRT Model Optimizer Examples

### Quantization

- [PTQ for LLMs](./llm_ptq/README.md) covers how to use Post-training quantization (PTQ) and export to [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) for deployment for popular pre-trained models from frameworks like
  - [Hugging Face](https://huggingface.co/docs/hub/en/models-the-hub)
  - [NVIDIA NeMo](https://github.com/NVIDIA/NeMo)
  - [NVIDIA Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
  - [Medusa](https://github.com/FasterDecoding/Medusa)
- [PTQ for DeepSeek](./deepseek/README.md) shows how to quantize the DeepSeek model with FP4 and export to TensorRT-LLM.
- [PTQ for Diffusers](./diffusers/quantization/README.md) walks through how to quantize a diffusion model with FP8 or INT8, export to ONNX, and deploy with [TensorRT](https://github.com/NVIDIA/TensorRT/tree/release/10.0/demo/Diffusion). The Diffusers example in this repo is complementary to the [demoDiffusion example in TensorRT repo](https://github.com/NVIDIA/TensorRT/tree/release/10.0/demo/Diffusion#introduction) and includes FP8 plugins as well as the latest updates on INT8 quantization.
- [PTQ for VLMs](./vlm_ptq/README.md) covers how to use Post-training quantization (PTQ) and export to [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) for deployment for popular Vision Language Models (VLMs).
- [PTQ for ONNX Models](./onnx_ptq/README.md) shows how to quantize the ONNX models in INT4 or INT8 quantization mode. The examples also include the deployment of quantized ONNX models using TensorRT.
- [QAT for LLMs](./llm_qat/README.md) demonstrates the recipe and workflow for Quantization-aware Training (QAT), which can further preserve model accuracy at low precisions (e.g., INT4, or FP4 in [NVIDIA Blackwell platform](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)).
- [AutoDeploy for AutoQuant LLM models](./llm_autodeploy/README.md) demonstrates how to deploy mixed-precision models using ModelOpt's AutoQuant and TRT-LLM's AutoDeploy.

### Pruning

- [Pruning](./pruning/README.md) demonstrates how to optimally prune Linear and Conv layers, and Transformer attention heads, MLP, and depth using the Model Optimizer for following frameworks:
  - [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) / [NVIDIA Megatron-LM](https://github.com/NVIDIA/Megatron-LM) GPT-style models (e.g. Llama 3, Mistral NeMo, etc.)
  - Hugging Face language models BERT and GPT-J
  - Computer Vision models like [NVIDIA Tao](https://developer.nvidia.com/tao-toolkit) or [MMDetection](https://github.com/open-mmlab/mmdetection) framework models.

### Distillation

- [Distillation for LLMs](./llm_distill/README.md) demonstrates how to use Knowledge Distillation, which can increasing the accuracy and/or convergence speed for finetuning / QAT.

### Speculative Decoding

- [Speculative Decoding](./speculative_decoding/README.md) demonstrates how to use speculative decoding to accelerate the text generation of large language models.

### Sparsity

- [Sparsity for LLMs](./llm_sparsity/README.md) shows how to perform Post-training Sparsification and Sparsity-aware fine-tuning on a pre-trained Hugging Face model.

### Evaluation

- [Evaluation for LLMs](./llm_eval/README.md) shows how to evaluate the performance of LLMs on popular benchmarks for quantized models or TensorRT-LLM engines.
- [Evaluation for VLMs](./vlm_eval/README.md) shows how to evaluate the performance of VLMs on popular benchmarks for quantized models or TensorRT-LLM engines.

### Chaining

- [Chained Optimizations](./chained_optimizations/README.md) shows how to chain multiple optimizations together (e.g. Pruning + Distillation + Quantization).

### Model Hub

- [Model Hub](./model_hub/) provides an example to deploy and run quantized Llama 3.1 8B instruct model from Nvidia's Hugging Face model hub on both TensorRT-LLM and vLLM.

### Windows

- [Windows](./windows/README.md) contains examples for Model Optimizer on Windows.
