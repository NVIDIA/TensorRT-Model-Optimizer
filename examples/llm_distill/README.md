# Knowledge Distillation

Knowledge Distillation is a machine learning technique where a compact "student" model learns to replicate the behavior of a larger, more complex "teacher" model to achieve comparable performance with improved efficiency.

Model Optimizer's Distillation is a set of wrappers and utilities to easily perform Knowledge Distillation among teacher and student models. Given a pretrained teacher model, Distillation has the potential to train a smaller student model faster and/or with higher accuracy than the student model could achieve on its own.

This section focuses on demonstrating how to apply Model Optimizer to perform knowledge distillation with ease.

<div align="center">

| **Section** | **Description** | **Link** | **Docs** |
| :------------: | :------------: | :------------: | :------------: |
| Pre-Requisites | Required & optional packages to use this technique | \[[Link](#pre-requisites)\] | |
| Getting Started | Learn how to optimize your models using distillation to produce more intellegant smaller models | \[[Link](#getting-started)\] | \[[docs](https://nvidia.github.io/TensorRT-Model-Optimizer/guides/4_distillation.html)\] |
| Support Matrix | View the support matrix to see compatibility and feature availability across different models | \[[Link](#support-matrix)\] | |
| Distillation with NeMo | Learn how to distill your models with NeMo Framework | \[[Link](#knowledge-distillation-kd-for-nvidia-nemo-models)\] | \[[docs](https://nvidia.github.io/TensorRT-Model-Optimizer/guides/4_distillation.html)\] |
| Distillation with Huggingface | Learn how to distill your models with Hugging Face | \[[Link](#knowledge-distillation-kd-for-huggingface-models)\] | \[[docs](https://nvidia.github.io/TensorRT-Model-Optimizer/guides/4_distillation.html)\] |
| Resources | Extra links to relevant resources | \[[Link](#resources)\] | |

</div>

## Pre-Requisites

### Docker

For Hugging Face models, please use the PyTorch docker image (e.g., `nvcr.io/nvidia/pytorch:25.06-py3`).
For NeMo models, use the NeMo container (e.g., `nvcr.io/nvidia/nemo:25.07`) which has all the dependencies installed.
Visit our [installation docs](https://nvidia.github.io/TensorRT-Model-Optimizer/getting_started/2_installation.html) for more information.

Also follow the installation steps below to upgrade to the latest version of Model Optimizer and install example-specific dependencies.

### Local Installation

For Hugging Face models, install Model Optimizer with `hf` dependencies using `pip` from [PyPI](https://pypi.org/project/nvidia-modelopt/) and install the requirements for the example:

```bash
pip install -U nvidia-modelopt[hf]
pip install -r requirements.txt
```

## Getting Started

### Set up your base models

First obtain both a pretrained model to act as the teacher and a (usually smaller) model to serve as the student.

```python
from transformers import AutoModelForCausalLM

# Define student & teacher
student_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
teacher_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-70B-Instruct")
```

### Set up the meta model

As Knowledge Distillation involves (at least) two models, ModelOpt simplifies the integration process by wrapping both student and teacher into one meta model.

Please see an example Distillation setup below. This example assumes the outputs of `teacher_model` and `student_model` are logits.

```python
import modelopt.torch.distill as mtd

distillation_config = {
    "teacher_model": teacher_model,
    "criterion": mtd.LogitsDistillationLoss(),  # callable receiving student and teacher outputs, in order
    "loss_balancer": mtd.StaticLossBalancer(),  # combines multiple losses; omit if only one distillation loss used
}

distillation_model = mtd.convert(student_model, mode=[("kd_loss", distillation_config)])
```

The `teacher_model` can be either a `nn.Module`, a callable which returns an `nn.Module`, or a tuple of `(model_cls, args, kwargs)`. The `criterion` is the distillation loss used between student and teacher tensors. The `loss_balancer` determines how the original and distillation losses are combined (if needed).

See [Distillation](https://nvidia.github.io/TensorRT-Model-Optimizer/guides/4_distillation.html) for more info.

### Distill during training

To Distill from teacher to student, simply use the meta model in the usual training loop, while also using the meta model’s `.compute_kd_loss()` method to compute the distillation loss, in addition to the original user loss.

An example of Distillation training is given below:

```python
# Setup the data loaders. As example:
train_loader = get_train_loader()

# Define user loss function. As example:
loss_fn = get_user_loss_fn()

for input, labels in train_dataloader:
    distillation_model.zero_grad()
    # Forward through the wrapped models
    out = distillation_model(input)
    # Same loss as originally present
    loss = loss_fn(out, labels)
    # Combine distillation and user losses
    loss_total = distillation_model.compute_kd_loss(student_loss=loss)
    loss_total.backward()
```

> [!NOTE]
> DataParallel may break ModelOpt’s Distillation feature. Note that HuggingFace Trainer uses DataParallel by default.

### Export trained model

The model can easily be reverted to its original class for further use (i.e deployment) without any ModelOpt modifications attached.

```python
model = mtd.export(distillation_model)
```

## Support Matrix

### Current out of the box components

Loss criterion:

- `mtd.LogitsDistillationLoss()` - Standard KL-Divergence on output logits
- `mtd.MGDLoss()` - Masked Generative Distillation loss for 2D convolutional outputs
- `mtd.MFTLoss()` - KL-divergence loss with Minifinetuning threshold modification

Loss balancers:

- `mtd.StaticLossBalancer()` - Combines original student loss and KD loss into a single weighted sum (without changing over time)

### Supported Models

> [!NOTE]
> The following are models that were confirmed to run with ModelOpt distillation, but it is absolutely not limited to these

| Model | type | confirmed compatible |
| :---: | :---: | :---: |
| Nemotron | gpt | ✅ |
| Llama 3 | llama | ✅ |
| Llama 4 | llama | ✅ |
| Gemma 2 | gemma | ✅ |
| Gemma 3 | gemma | ✅ |
| Phi 3 | phi | ✅ |
| Qwen 2 | phi | ✅ |
| Qwen 3 | phi | ✅ |
| Mamba | mamba | ✅ |

## Knowledge Distillation (KD) for NVIDIA NeMo Models

Checkout the stand-alone distillation script in the [NVIDIA NeMo repository](https://docs.nvidia.com/nemo-framework/user-guide/latest/model-optimization/distillation/distillation.html).

You can also look at the tutorial notebooks [here](https://github.com/NVIDIA-NeMo/NeMo/tree/main/tutorials/llm/llama/pruning-distillation) which showcase the usage of Minitron pruning followed by distillation for Llama 3.1 8B step-by-step in NeMo framework.

## Knowledge Distillation (KD) for HuggingFace Models

In this e2e example we finetune Llama-2 models on the [OpenOrca](https://huggingface.co/datasets/Open-Orca/OpenOrca)
question-answer dataset as a minimal example to demonstrate a simple way of integrating Model Optimizer's KD feature.

First we do supervised finetuning (SFT) of a Llama-2-7b on OpenOrca dataset as the teacher, then distill it into
a 1B-parameter model.

Keep in mind the training loss of the distillation run is not directly comparable to the training loss of the teacher run.

> [!NOTE]
> We can fit the following in memory using [FSDP](https://huggingface.co/docs/accelerate/en/usage_guides/fsdp) enabled on 8x RTX 6000 (total ~400GB VRAM)

### Train teacher

```bash
accelerate launch --config-file ./accelerate_config/fsdp2.yaml \
    main.py \
    --single_model \
    --teacher_name_or_path 'meta-llama/Llama-2-7b-hf' \
    --output_dir ./llama2-7b-sft \
    --max_length 2048 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --max_steps 400 \
    --logging_steps 5
```

### Distill teacher into student

```bash
accelerate launch --config-file ./accelerate_config/fsdp2.yaml \
    --fsdp_cpu_ram_efficient_loading False \
    --fsdp_activation_checkpointing False \
    main.py \
    --teacher_name_or_path ./llama2-7b-sft \
    --student_name_or_path 'TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T' \
    --output_dir ./llama2-distill \
    --max_length 2048 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --max_steps 200 \
    --logging_steps 5
```

> [!NOTE]
> If you receive a `RuntimeError: unable to open file <...> in read-only mode: No such file or directory` simply re-run the command a second time.

## Resources

- 📅 [Roadmap](https://github.com/NVIDIA/TensorRT-Model-Optimizer/issues/146)
- 📖 [Documentation](https://nvidia.github.io/TensorRT-Model-Optimizer)
- 🎯 [Benchmarks](../benchmark.md)
- 💡 [Release Notes](https://nvidia.github.io/TensorRT-Model-Optimizer/reference/0_changelog.html)
- 🐛 [File a bug](https://github.com/NVIDIA/TensorRT-Model-Optimizer/issues/new?template=1_bug_report.md)
- ✨ [File a Feature Request](https://github.com/NVIDIA/TensorRT-Model-Optimizer/issues/new?template=2_feature_request.md)
