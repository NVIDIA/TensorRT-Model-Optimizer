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
   torchrun examples/compress/main.py --config path/to/llama-3_1-8B_pruneffn_memory.yaml 2>&1 | tee ./log.txt |grep "Compress Progress"
   ```

   screen output:

   ```bash
   [2025-11-01 19:26:38] Compress Progress 1/8: starting compression pipeline
   [2025-11-01 19:26:38] Compress Progress 2/8: converting model from HF to DeciLM
   [2025-11-01 19:26:39] Compress Progress 3/8: scoring pruning activations
   [2025-11-01 19:26:46] Compress Progress 4/8: pruning the model and saving pruned checkpoints
   [2025-11-01 19:26:46] Compress Progress 5/8: building replacement library and calculating subblock statistics
   [2025-11-01 19:26:46] Compress Progress 6/8: calculating one block scores
   [2025-11-01 19:26:52] Compress Progress 7/8: running MIP and realizing models
   [2025-11-01 19:26:59] Compress Progress 8/8: compression pipeline completed
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
