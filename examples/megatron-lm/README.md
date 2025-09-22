<div align="center">

# Megatron-LM Integrated Examples

[Local Examples](#getting-started-in-a-local-environment) |
[Configuration](#learn-more-about-configuration) |
[Advanced Topics](ADVANCED.md) |
[Megatron-LM Integration](https://github.com/NVIDIA/Megatron-LM/tree/main/examples/post_training/modelopt)

</div>

## Major Features

- Start from Hugging Face pretrained model checkpoint with on the fly conversion.
- Support all kinds of model parallelism (TP, EP, ETP, PP).
- Export to TensorRT-LLM, vLLM, and SGLang ready unified checkpoint.

## Support Matrix: {Model}x{Features}

| Model | Quantization | EAGLE3 | Q-LoRA | Pruning (PP only) | Distillation |
| :---: | :---: | :---: | :---: | :---: | :---: |
| `moonshotai/Kimi-K2-Instruct` | ✅ | **Online** | | | |
| `Qwen/Qwen3-{30B-A3B, 235B-A22B}` | **WAR** | **Online** | | | |
| `Qwen/Qwen3-{0.6B, 8B}` | ✅ | **Online** | | ✅ | ✅ |
| `deepseek-ai/DeepSeek-R1` | ✅ | **Online** | | | |
| `meta-llama/Llama-{3.1-8B, 3.1-405B, 3.2-1B}-Instruct` | ✅ | **Online** | | ✅ | ✅ |

## Getting Started in a Local Environment

Given that only `megatron.core` can be pip-install, the examples are containerized with
[Megatron-LM](https://github.com/NVIDIA/Megatron-LM) and
[TensorRT Model Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer)
pre-installed. **Use the following command to build the container.**

```sh
docker build --no-cache --network=host --rm -t nvidia-modelopt-megatron:latest .
```

> **📙 NOTE:** If you plan to use `slurm` for multi-node execution, push the image to a container registry.

For local execution, a **READ/WRITE scratch space** needs to be mounted. Mount additional volumes for
checkpoints, datasets, and other artifact.

```sh
USER_FSW=<path_to_scratch_space> bash interactive.sh
```

> **📙 NOTE:** The current dir will be mounted to `$(pwd):/workspace/nmm-sandbox` and the scratch
> space will be mounted to `/workspace/scratch`.

<br>

### ⭐ FP8 Post-Training Quantization (PTQ)

Provide the pretrained checkpoint path through variable `${HF_MODEL_CKPT}`:

```sh
\
    TP=1 \
    HF_MODEL_CKPT=<pretrained_model_name_or_path> \
    MLM_MODEL_SAVE=/tmp/Llama-3.2-1B-Instruct-FP8 \
    bash megatron-lm/examples/post_training/modelopt/quantize.sh meta-llama/Llama-3.2-1B-Instruct fp8

\
    PP=1 \
    HF_MODEL_CKPT=<pretrained_model_name_or_path> \
    MLM_MODEL_LOAD=/tmp/Llama-3.2-1B-Instruct-FP8 \
    EXPORT_DIR=/tmp/Llama-3.2-1B-Instruct-Export \
    bash megatron-lm/examples/post_training/modelopt/export.sh meta-llama/Llama-3.2-1B-Instruct

```

You can find a resumable Megatron-LM checkpoint for quantization-aware training or simulated evaluation
(`/tmp/Llama-3.2-1B-Instruct-FP8`) and a Hugging Face-Like exported checkpoint for
deployment (`/tmp/Llama-3.2-1B-Instruct-Export`).

<br>

### ⭐ Online BF16 EAGLE3 Training

Online EAGLE3 training has both the target (frozen) and draft models in the memory where the `hidden_states`
required for training is generated on the fly.

```sh
\
    TP=1 \
    HF_MODEL_CKPT=<pretrained_model_name_or_path> \
    MLM_MODEL_SAVE=/tmp/Llama-3.2-1B-Eagle3 \
    bash megatron-lm/examples/post_training/modelopt/eagle3.sh meta-llama/Llama-3.2-1B-Instruct

\
    PP=1 \
    HF_MODEL_CKPT=<pretrained_model_name_or_path> \
    MLM_MODEL_LOAD=/tmp/Llama-3.2-1B-Eagle3 \
    EXPORT_DIR=/tmp/Llama-3.2-1B-Eagle3-Export \
    bash megatron-lm/examples/post_training/modelopt/export.sh meta-llama/Llama-3.2-1B-Instruct
```

Periodically, **acceptance length (AL)** is evaluated on MT-Bench prompts. You can find resumable
Megatron-LM checkpoint (`/tmp/Llama-3.2-1B-Eagle3`) and a Hugging Face-Like exported checkpoiint
for deployment (`/tmp/Llama-3.2-1B-Eagle3-Export`).

See [ADVANCED.md](ADVANCED.md) for a multi-gpu multi-node training example for `moonshotai/Kimi-K2-Instruct`.

<br>

### ⭐ Offline BF16 EAGLE3 Training

Coming soon ...

### ⭐ Pruning

Pruning is supported for GPT and Mamba models in Pipeline Parallel mode. Available pruning options are:

- `TARGET_FFN_HIDDEN_SIZE`
- `TARGET_HIDDEN_SIZE`
- `TARGET_NUM_ATTENTION_HEADS`
- `TARGET_NUM_QUERY_GROUPS`
- `TARGET_MAMBA_NUM_HEADS`
- `TARGET_MAMBA_HEAD_DIM`
- `TARGET_NUM_LAYERS`
- `LAYERS_TO_DROP` (comma separated, 1-indexed list of layer numbers to directly drop)

```sh
PP=1 \
TARGET_NUM_LAYERS=24 \
HF_MODEL_CKPT=<pretrained_model_name_or_path> \
MLM_MODEL_SAVE=/tmp/Qwen3-8B-DPruned \
bash megatron-lm/examples/post_training/modelopt/prune.sh qwen/Qwen3-8B
```

## Learn More About Configuration

For simplicity, we use `shell` scripts and variables as arguments. Each script has at least 1 positional
argument `[pretrained_model_card]`. Some scripts may require more such as `[qformat]` is needed for
quantization.

```sh
\
    HF_MODEL_CKPT=<pretrained_model_name_or_path> \
    bash megatron-lm/examples/post_training/modelopt/quantize.sh [pretrained_model_card] [qformat]
```

> **❗ IMPORTANT:** `pretrained_model_card` **CANNOT** be a path to a local pretrained checkpoint.
> It is used to get the corresponding Megatron-LM `${MODEL_ARGS}`. For example,
> `meta-llama/Llama-3.1-8B-Instruct` or `deepseek-ai/DeepSeek-R1` are both supported.
> \
> Provide the pretrained checkpoint through variable `${HF_MODEL_CKPT}` in commandline or
> in `env_setup_template.sh`. More variables (e.g. `${TP}`, `${EP}`, ...) can be provided though
> commandline but we recommend passing all variable in a another `shell` script.

When `${HF_MODEL_CKPT}` is not set through the commandline, `./env_setup_template.sh` can be used
to pass all variables instead. If you have your own script, use `${SANDBOX_ENV_SETUP}`.

```sh
\
    SANDBOX_ENV_SETUP=<path_to_your_script> \
    bash megatron-lm/examples/post_training/modelopt/quantize.sh [pretrained_model_card] [qformat]
```

If you use our `slurm` script, then you **MUST USE** `${SANDBOX_ENV_SETUP}` (default: `./env_setup_template.sh`).
Other variables are not passing through `sbatch` and `srun` automatically.

See [ADVANCED.md](ADVANCED.md) to learn all the configurable variables.
