# Compress Algorithm Tutorial

This tutorial demonstrates how to compress large language models using the compress algorithm based on the [Puzzle paper](https://arxiv.org/abs/2411.19146).

In this example, we compress [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) model by searching for the optimal `ffn_intermediate_size` across MLP layers, resulting in a heterogeneous architecture while reducing GPU memory usage by 20%.

## Compress the Model

1. Specify the `puzzle_dir`, `input_hf_model_path`, `dataset_path`, `intermediate_size_list`, and `target_memory` arguments in the [llama-3_1-8B_pruneffn_memory.yaml](./configs/llama-3_1-8B_pruneffn_memory/llama-3_1-8B_pruneffn_memory.yaml) configuration file.

2. Download and prepare the [Nemotron-Post-Training-Dataset-v2](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2).

   dataset split: "code", "math", "stem", "chat", excluding reasoning samples (2.62GB)

   ```bash
   python -m modelopt.torch._compress.dataset.prepare_dataset --dataset_name nvidia/Nemotron-Post-Training-Dataset-v2 --output_dir path/to/Nemotron-Post-Training-Dataset-v2
   ```

3. Run the compression script.

   ```bash
   torchrun examples/compress/main.py --config path/to/llama-3_1-8B_pruneffn_memory.yaml
   ```

## Evaluate model accuracy

```bash
# TODO
```

## Re-run MIP Search with different memory constraints

If you want to try different memory constraints without re-running the expensive pruning and scoring steps, use the `--mip-only` flag:

```bash
torchrun examples/compress/main.py \
  --config path/to/llama-3_1_8B_pruneffn_memory.yaml \
  --mip-only
```

This assumes pruning, replacement library building, NAS scoring, and subblock stats calculation have already been completed.

## Deploy to TensorRT-LLM

```bash
# TODO
```

## Export to NeMo for Knowledge Distillation

```bash
# TODO
```

## Advanced usage

Modify `path/to/Llama-3_1-8B yaml` file for advanced compression scenarios.
