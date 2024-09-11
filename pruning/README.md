# Pruning

The Model Optimizer's `modelopt.torch.prune` module provides advanced state-of-the-art pruning algorithms that enable you to search for the best subnet architecture from your provided base model.

Model Optimizer can be used in one of the following complementary pruning modes to create a search space for optimizing the model:

1. [Minitron](https://arxiv.org/pdf/2408.11796): A pruning method developed by NVIDIA Research for pruning GPT-style models in NVIDIA NeMo or Megatron-LM framework that are using Pipeline or Tensor Parallellisms. It uses the activation magnitudes to prune the mlp, attention heads, and GQA query groups the model.
1. GradNAS: A light-weight pruning method recommended for language models like Hugging Face BERT, GPT-J. It uses the gradient information to prune the model's linear layers and attention heads to meet the given constraints.
1. FastNAS: A pruning method recommended for Computer Vision models. Given a pretrained model, FastNAS finds the subnet which maximizes the score function while meeting the given constraints.

## Documentation

Checkout the [Quick Start: Pruning](https://nvidia.github.io/TensorRT-Model-Optimizer/getting_started/4_pruning.html) and the detailed [Optimization Guide](https://nvidia.github.io/TensorRT-Model-Optimizer/guides/2_pruning.html) in the Model Optimizer documentation for more information on how to use the above pruning algorithms in Model Optimizer.

## Algorithms

### Pruning HuggingFace Language Models (e.g. BERT) using GradNAS

Checkout the BERT pruning example in [chained_optimizations](../chained_optimizations/README.md) directory
which showcases the usage of GradNAS for pruning BERT model for Question Answering followed by fine-tuning
with distillation and quantization. The example also demonstrates how to save and restore pruned models.

### Pruning PyTorch Computer Vision Models using FastNAS

Checkout the FastNAS pruning interactive notebook [cifar_resnet](./cifar_resnet.ipynb) in this directory
which showcases the usage of FastNAS for pruning a ResNet 20 model for the CIFAR-10 dataset. The notebook
also how to profiling the model to understand the search space of possible pruning options and demonstrates
the usage saving and restoring pruned models.
