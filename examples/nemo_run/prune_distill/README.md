# Pruning and Knowledge Distillation Nemo Run example

## Overview

This directory contains the NeMo 2.0 Pruning + Knowledge Distillation flow implementation. The main script `nemo_prune_kd_flow.py` enables model compression through structured pruning followed by knowledge distillation to recover performance.

## Usage

### Prerequisites

#### Install NeMo 2.0 and related dependencies

To run the example, launch a [NeMo container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo) with version 25.04.01 or higher using Docker/Slurm. Mount your cloned `modelopt` repository to the container by adding this mount flag to your Docker/Slurm command: `-v <modelopt-path>:/workspace/modelopt -v <modelopt-path>/modelopt:/usr/local/lib/python3.12/dist-packages/modelopt`.

To run SFT properly you may also need to clone NeMo and Megatron-LM at the respective commits, and mount to `/opt/NeMo` and `/opt/megatron-lm`:

- `git clone https://github.com/NVIDIA-NeMo/NeMo && cd NeMo && git checkout d7b87b1`
- `git clone https://github.com/NVIDIA/Megatron-LM.git && cd Megatron-LM && git checkout 8c15450`

### Data Preparation

The script supports chat datasets in ShareGPT or HuggingFace/OpenAI chat format. You can prepare your dataset in JSONL format with the required chat structure. To provide your own custom dataset, use the `--data-path` flag, otherwise the default [LIMA](https://huggingface.co/datasets/GAIR/lima) dataset will be used.

### Running the Flow

#### Standard Usage

From the `nemo_run` folder, run:

```bash
python prune_distill/nemo_prune_kd_flow.py --data_path your_dataset.jsonl
```

#### Mock Run (for testing)

To test the flow without actual data, run the following command from the `nemo_run` folder:

```bash
python prune_distill/nemo_prune_kd_flow.py --mock_run
```

### Flow Stages

The script executes the following stages in sequence:

1. Process LIMA data (if `--data-path` is not specified)
1. **Import Model**: Imports the HuggingFace model to NeMo format
1. **Fine-tuning**: Fine-tunes the model on the provided dataset
1. **Pruning**: Prunes the fine-tuned model to create a smaller student model
1. **Knowledge Distillation**: Distills knowledge from the teacher to the pruned student model
1. **Export**: Exports the final compressed model

### Configuration Parameters

The script includes several configurable parameters:

- **GPUS**: Number of GPUs (default: 8)
- **SEQUENCE_LENGTH**: Maximum sequence length (default: 8192)
- **MBS**: Micro batch size (default: 2)
- **GBS**: Global batch size (default: 2048 for real runs, 8 for mock runs)
- **FINETUNE_STEPS**: Number of fine-tuning steps (default: 2500 for real runs, 20 for mock runs)
- **DISTILL_STEPS**: Number of distillation steps (default: 7500 for real runs, 20 for mock runs)
- **VAL_INTERVAL**: Validation interval (default: 500 for real runs, 10 for mock runs)
- **PRUNE_SAMPLES**: Number of samples for pruning calibration (default: 1024 for real runs, 3 for mock runs)

### Pruning Configuration

- **Target Hidden Size**: Default is 3072 (configurable via `--prune_target_hidden_size`)
- **Target FFN Hidden Size**: Automatically set to 3 × target_hidden_size
- **Pruning Method**: Structured pruning to reduce model dimensions

### Output

The script generates the following outputs in the specified log directory:

- `{model_name}_initial/`: Initial NeMo checkpoint
- `finetune_log_dir/`: Fine-tuning logs and checkpoints (teacher model)
- `{model_name}_pruned/`: Pruned student model
- `distill_log_dir/`: Knowledge distillation logs and checkpoints
- `{model_name}_final/`: Final compressed model after distillation

### Supported Models

Currently supports models that can be trained on 1 node with 8 x 80GB GPUs. The default configuration uses:

- **Model**: Meta-Llama-3.1-8B
- **Recipe**: llama31_8b
- **Pruning Strategy**: Structured pruning with knowledge distillation recovery

### Troubleshooting

1. **GPU Memory Issues**: Reduce batch sizes (MBS, GBS) if encountering OOM errors
1. **Data Format**: Ensure your dataset follows the expected chat format
1. **NeMo Installation**: If encountering NeMo-related errors, use the recommended docker container
1. **Model Size**: Ensure your model fits within the 8-GPU configuration
