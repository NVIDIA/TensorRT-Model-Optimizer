# Pruning

Model pruning is a technique that removes redundant or less important parameters/connections from a neural network to reduce complexity and improve efficiency while maintaining performance.

Pruning can involve removal (prune) of Linear and Conv layers, and Transformer attention heads, MLP, and depth.

This section focuses on applying Model Optimizer's state-of-the-art complementary pruning modes to enable you to search for the best subnet architecture from your provided base model:

1. [Minitron](https://arxiv.org/pdf/2408.11796): A pruning method developed by NVIDIA Research for pruning GPT, Mamba and Hybrid Transformer Mamba models in NVIDIA NeMo or Megatron-LM framework. It uses the activation magnitudes to prune the embedding hidden size, mlp ffn hidden size, transformer attention heads, GQA query groups, mamba heads and head dimension, and number of layers of the model.
1. FastNAS: A pruning method recommended for Computer Vision models. Given a pretrained model, FastNAS finds the subnet which maximizes the score function while meeting the given constraints.
1. GradNAS: A light-weight pruning method recommended for language models like Hugging Face BERT, GPT-J. It uses the gradient information to prune the model's linear layers and attention heads to meet the given constraints.

<div align="center">

| **Section** | **Description** | **Link** | **Docs** |
| :------------: | :------------: | :------------: | :------------: |
| Pre-Requisites | Required & optional packages to use this technique | \[[Link](#pre-requisites)\] | |
| Getting Started | Learn how to use the pruning API | \[[Link](#getting-started)\] | \[[docs](https://nvidia.github.io/TensorRT-Model-Optimizer/guides/3_pruning.html)\] |
| Support Matrix | View the support matrix to see available pruning algorithms and their compatibility with different models and frameworks | \[[Link](#support-matrix)\] | |
| Resources | Extra links to relevant resources | \[[Link](#resources)\] | |

</div>

## Pre-Requisites

For Minitron pruning for Megatron-LM / NeMo models, use the NeMo container (e.g., `nvcr.io/nvidia/nemo:25.07`) which has all the dependencies installed.

For FastNAS pruning for PyTorch Computer Vision models, no additional dependencies are required.

For GradNAS pruning for Hugging Face BERT / GPT-J, no additional dependencies are requisred.

## Getting Started

As part of the pruning process, you will need to set up the training and/or validation data loaders, and optionally define a validation score function (FastNAS) or loss function (GradNAS) and specify the desired pruning constraints (See [Support Matrix](#support-matrix) for available pruning constraints).

To prune your model, you can simply call the `mtp.prune` API and save the pruned model. If the model is pruned using FastNAS or GradNAS, you need to use `mto.save` and `mto.restore` to save and restore the pruned model; while for Minitron pruning, you can use your standard saving and loading functions since it is a homogeneous pruning.

Please see an example of Minitron pruning for Megatron-Core GPT model below (for other algorithms, please refer to the examples below).

```python
import modelopt.torch.prune as mtp
from megatron.core.models.gpt import GPTModel
from megatron.core.post_training.modelopt.gpt.model_specs import get_gpt_modelopt_spec
from megatron.core.transformer.transformer_config import TransformerConfig

# Load the Megatron-Core GPTModel with ModelOpt transformer layer spec
config = TransformerConfig(...)
model = GPTModel(
    config=config,
    transformer_layer_spec=get_gpt_modelopt_spec(config, remap_te_layernorm=True),
    ...
)

# Set up the forward loop to run on 512-1024 train samples
# For Megatron-LM framework, you can use the following utility function
from megatron.training.training import evaluate_and_print_results

def forward_loop(model):
    evaluate_and_print_results(model, ...)


# Specify the pruning constraints
export_config = {
    "hidden_size": 3072,
    "ffn_hidden_size": 9216,
}


# Run the pruning process
mtp.prune(
    model,
    mode="mcore_minitron",
    constraints={"export_config": export_config},
    dummy_input=None,  # Not used
    config={"forward_loop": forward_loop},
)
```

> [!Note]
> Fine-tuning / distillation is required after pruning to recover the accuracy. Please refer to pruning [fine-tuning](https://nvidia.github.io/TensorRT-Model-Optimizer/guides/3_pruning.html#pruning-fine-tuning) for more details.

## Support Matrix

| **Algorithm** | **Model** | **Pruning Constraints** |
| :---: | :---: | :---: |
| Minitron | Megatron-core / NeMo based GPT / Mamba / Hybrid Models<sup>1</sup> | Export config with width (`hidden_size`, `ffn_hidden_size`, `num_attention_heads`, `num_query_groups`, `mamba_num_heads`, `mamba_head_dim`) and/or depth (`num_layers`) values |
| FastNAS | Computer Vision models | flops, parameters |
| GradNAS | HuggingFace BERT, GPT-J | flops, parameters |

> *<sup>1.</sup>Only Pipeline Parallel models are supported. Hugging Face models can be converted to NeMo format and used subsequently.*

## Examples

### Minitron Pruning for Megatron-LM / NeMo Framework LLMs (e.g. Llama 3.1, Nemotron Nano)

Checkout the Minitron pruning example for the [Megatron-LM Framework](../megatron-lm/README.md#-pruning) and [NeMo Framework](https://docs.nvidia.com/nemo-framework/user-guide/latest/model-optimization/pruning/pruning.html) which showcases the usage of the powerful Minitron pruning algorithm developed by NVIDIA Research for pruning LLMs like Llama 3.1 8B, Qwen 3 8B, Nemotron Nano 12B v2, etc.

You can also look at the NeMo tutorial notebooks [here](https://github.com/NVIDIA-NeMo/NeMo/tree/main/tutorials/llm/llama/pruning-distillation) which showcase the usage of Minitron pruning followed by distillation for Llama 3.1 8B step-by-step in NeMo framework. Hugging Face models can also be converted to NeMo format and used subsequently as shown in the tutorial.

Some of the models pruned using Minitron method followed by distillation and post-training are:

- [Minitron Collection on Hugging Face](https://huggingface.co/collections/nvidia/minitron-669ac727dc9c86e6ab7f0f3e)
- [NVIDIA-Nemotron-Nano-9B-v2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2)

### FastNAS Pruning for PyTorch Computer Vision Models

Checkout the FastNAS pruning interactive notebook [cifar_resnet](./cifar_resnet.ipynb) in this directory
which showcases the usage of FastNAS for pruning a ResNet 20 model for the CIFAR-10 dataset. The notebook
also how to profiling the model to understand the search space of possible pruning options and demonstrates
the usage saving and restoring pruned models.

### GradNAS Pruning for HuggingFace Language Models (e.g. BERT)

Checkout the BERT pruning example in [chained_optimizations](../chained_optimizations/README.md) directory
which showcases the usage of GradNAS for pruning BERT model for Question Answering followed by fine-tuning
with distillation and quantization. The example also demonstrates how to save and restore pruned models.

## Resources

- 📅 [Roadmap](https://github.com/NVIDIA/TensorRT-Model-Optimizer/issues/146)
- 📖 [Documentation](https://nvidia.github.io/TensorRT-Model-Optimizer)
- 🎯 [Benchmarks](../benchmark.md)
- 💡 [Release Notes](https://nvidia.github.io/TensorRT-Model-Optimizer/reference/0_changelog.html)
- 🐛 [File a bug](https://github.com/NVIDIA/TensorRT-Model-Optimizer/issues/new?template=1_bug_report.md)
- ✨ [File a Feature Request](https://github.com/NVIDIA/TensorRT-Model-Optimizer/issues/new?template=2_feature_request.md)
