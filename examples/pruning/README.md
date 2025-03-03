# Pruning

The Model Optimizer's `modelopt.torch.prune` module provides advanced state-of-the-art pruning algorithms that enable you to search for the best subnet architecture from your provided base model.

To learn more about the pruning feature, please refer to the [documentation](https://nvidia.github.io/TensorRT-Model-Optimizer/guides/2_pruning.html).

Model Optimizer can be used in one of the following complementary pruning modes to create a search space for optimizing the model:

1. [Minitron](https://arxiv.org/pdf/2408.11796): A pruning method developed by NVIDIA Research for pruning GPT-style models in NVIDIA NeMo or Megatron-LM framework that are using Pipeline or Tensor Parallelisms. It uses the activation magnitudes to prune the mlp, attention heads, GQA query groups, embedding hidden size and number of layers of the model.
1. GradNAS: A light-weight pruning method recommended for language models like Hugging Face BERT, GPT-J. It uses the gradient information to prune the model's linear layers and attention heads to meet the given constraints.
1. FastNAS: A pruning method recommended for Computer Vision models. Given a pretrained model, FastNAS finds the subnet which maximizes the score function while meeting the given constraints.

## Documentation

Checkout the [Quick Start: Pruning](https://nvidia.github.io/TensorRT-Model-Optimizer/getting_started/5_pruning.html) and the detailed [Optimization Guide](https://nvidia.github.io/TensorRT-Model-Optimizer/guides/2_pruning.html) in the Model Optimizer documentation for more information on how to use the above pruning algorithms in Model Optimizer.

## Algorithms

### Minitron Pruning for NVIDIA NeMo / Megatron-LM LLMs (e.g. Llama 3)

Checkout the Minitron pruning example in the [NVIDIA NeMo repository](https://docs.nvidia.com/nemo-framework/user-guide/latest/model-optimization/pruning/pruning.html) which showcases the usage of the powerful Minitron pruning algorithm developed by NVIDIA Research for pruning LLMs like Llama 3.1 8B or Mistral NeMo 12B.

You can also look at the tutorial notebooks [here](https://github.com/NVIDIA/NeMo/tree/main/tutorials/llm/llama/pruning-distillation) which showcase the usage of Minitron pruning followed by distillation for Llama 3.1 8B step-by-step in NeMo framework.

NOTE: If you wish to use this algorithm for pruning Hugging Face LLMs, you can first use the HF to NeMo converters, then use Minitron pruning, optionally followed by distillation in NeMo framework and then convert back to Hugging Face format.
You can use the converter scripts in the [NeMo repository](https://github.com/NVIDIA/NeMo/tree/main/scripts/checkpoint_converters/).

### GradNAS Pruning for HuggingFace Language Models (e.g. BERT)

Checkout the BERT pruning example in [chained_optimizations](../chained_optimizations/README.md) directory
which showcases the usage of GradNAS for pruning BERT model for Question Answering followed by fine-tuning
with distillation and quantization. The example also demonstrates how to save and restore pruned models.

### FastNAS Pruning for PyTorch Computer Vision Models

Checkout the FastNAS pruning interactive notebook [cifar_resnet](./cifar_resnet.ipynb) in this directory
which showcases the usage of FastNAS for pruning a ResNet 20 model for the CIFAR-10 dataset. The notebook
also how to profiling the model to understand the search space of possible pruning options and demonstrates
the usage saving and restoring pruned models.
