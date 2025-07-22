# Speculative Decoding and SFT Nemo Run example

## Overview

This directory contains the NeMo 2.0 Speculative Decoding + SFT (Supervised Fine-Tuning) flow implementation. The main script `nemo_eagle_sft_flow.py` enables fine-tuning of language models using speculative decoding techniques.

## Usage

### Prerequisites

To run the example, launch a [NeMo container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo) with version 25.04.01 or higher using Docker/Slurm. Mount your cloned `modelopt` repository to the container by adding this mount flag to your Docker/Slurm command: `-v <modelopt-path>:/workspace/modelopt -v <modelopt-path>/modelopt:/usr/local/lib/python3.12/dist-packages/modelopt`.

To run SFT properly you may also need to clone NeMo and Megatron-LM at the respective commits, and mount to `/opt/NeMo` and `/opt/megatron-lm`:

- `git clone https://github.com/NVIDIA/NeMo && cd NeMo && git checkout d7b87b1`
- `git clone https://github.com/NVIDIA/Megatron-LM.git && cd Megatron-LM && git checkout 8c15450`

### Data Preparation

The script supports chat datasets in ShareGPT or HuggingFace/OpenAI chat format. You can use the provided `download_magpie_dataset.py` script to download and prepare the MagPie dataset:

```bash
python download_magpie_dataset.py --output magpie_dataset.jsonl
```

### Running the Flow

#### Standard Usage

```bash
python nemo_eagle_sft_flow.py --data_path magpie_dataset.jsonl
```

#### Mock Run (for testing)

To test the flow without actual data:

```bash
python nemo_eagle_sft_flow.py --mock_run
```

### Flow Stages

The script executes the following stages in sequence:

1. **Import Model**: Imports the HuggingFace model to NeMo format
1. **Convert to Speculative Decoding**: Converts the model to use Eagle3 speculative decoding
1. **Supervised Fine-Tuning**: Performs SFT on the speculative decoding model
1. **Export**: Exports the fine-tuned speculative decoding model

### Configuration Parameters

The script includes several configurable parameters:

- **GPUS**: Number of GPUs (default: 8)
- **SEQUENCE_LENGTH**: Maximum sequence length (default: 8192)
- **MBS**: Micro batch size (default: 2)
- **GBS**: Global batch size (default: 128 for real runs, 8 for mock runs)
- **FINETUNE_STEPS**: Number of fine-tuning steps (default: 5000 for real runs, 20 for mock runs)
- **VAL_INTERVAL**: Validation interval (default: 500 for real runs, 10 for mock runs)

### Output

The script generates the following outputs in the specified log directory:

- `{model_name}_initial/`: Initial NeMo checkpoint
- `{model_name}_specdec/`: Speculative decoding model
- `finetune_log_dir/`: Fine-tuning logs and checkpoints
- `{model_name}_specdec-only/`: Final exported speculative decoding model

### Supported Models

Currently supports models that can be trained on 1 node with 8 x 80GB GPUs. The default configuration uses:

- **Model**: Meta-Llama-3.1-8B-Instruct
- **Recipe**: llama31_8b
- **Speculative Decoding Algorithm**: Eagle 3

### Troubleshooting

1. **GPU Memory Issues**: Reduce batch sizes (MBS, GBS) if encountering OOM errors
1. **Data Format**: Ensure your dataset follows the expected chat format
1. **NeMo Installation**: If encountering NeMo-related errors, use the recommended docker container
1. **Model Size**: Ensure your model fits within the 8-GPU configuration
