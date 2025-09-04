# NeMo QAT/QAD Simplified Flow Example

## Overview

This directory also contains an end-to-end NeMo QAT Simplified Flow example, which supports both QAT with cross-entropy loss and QAD (quantization-aware distillation) with knowledge-distillation loss between the full-precision teacher and quantized student models.

## Usage

### Prerequisites

To run the example, launch a [NeMo container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo) with version 25.04.01 or higher using Docker/Slurm. Mount your cloned `modelopt` repository to the container by adding this mount flag to your Docker/Slurm command: `-v <modelopt-path>:/workspace/modelopt -v <modelopt-path>/modelopt:/usr/local/lib/python3.12/dist-packages/modelopt`.

To run SFT properly you may also need to clone NeMo and Megatron-LM at the respective commits, and mount to `/opt/NeMo` and `/opt/megatron-lm`:

- `git clone https://github.com/NVIDIA-NeMo/NeMo && cd NeMo && git checkout d7b87b1`
- `git clone https://github.com/NVIDIA/Megatron-LM.git && cd Megatron-LM && git checkout 8c15450`

### Running the Flow

#### QAT

From the `nemo_run` folder, launch the example with `python qat/nemo_qat_flow.py --model-name <hf-model-name> --finetune-recipe <recipe-name>`. Available NeMo recipe names are listed [here](https://github.com/NVIDIA-NeMo/NeMo/tree/main/nemo/collections/llm/recipes). To provide your own custom dataset, use the `--data-path` flag, otherwise the default [LIMA](https://huggingface.co/datasets/GAIR/lima) dataset will be used.

To perform QAT, run:

```bash
python qat/nemo_qat_flow.py \
    --model-name meta-llama/Meta-Llama-3.1-8B-Instruct \
    --finetune-recipe llama31_8b \
    --algorithm fp8 \
    --chat-template llama_chat_template.txt \
    --experiment llama3_qat_nemo
```

> **_NOTE:_** To enable KV cache quantization, add `--enable-kv-cache` and specify qformat using `--kv-cache-qformat <fp8, nvfp4>`.

#### QAD

In order to train using QAD, launch the example with `python qat/nemo_qat_flow.py --model-name <hf-model-name> --distill`. It will utilize `distillation_recipe` with quantized student model and full precision teacher model to train the quantized model.

To perform QAD training, run:

```bash
python qat/nemo_qat_flow.py \
    --model-name meta-llama/Meta-Llama-3.1-8B-Instruct \
    --distill \
    --algorithm fp8 \
    --chat-template llama_chat_template.txt \
    --experiment llama3_qad_nemo
```

### Custom Chat Template

By default the script will use the model/tokenizer's chat template, which may not contain the `{% generation %}` and `{% endgeneration %}` tags around the assistant tokens which are needed to generate the assistant loss mask (see [this PR](https://github.com/huggingface/transformers/pull/30650)). To provide path to a custom chat template, use the `--chat-template <my_template.txt>` flag.

## Flow Stages

Currently the Simplified Flow runs the following steps in order:

1. Process LIMA data (if `--data-path` is not specified)
1. Import NeMo model checkpoint
1. PTQ the model
1. SFT (finetune) the model
1. Export model to Unified checkpoint (HuggingFace) format

## Supported models

Currently supports models that can be trained on 1 node with 8 x 80GB GPUs. The default configuration uses:

- **Model**: Meta-Llama-3.1-8B-Instruct
- **Recipe**: llama31_8b
