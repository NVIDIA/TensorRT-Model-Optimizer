# Compress Algorithm Tutorial

This tutorial demonstrates how to compress large language models using the compress algorithm based on the [Puzzle paper](https://arxiv.org/abs/2411.19146).

In this example, we compress Llama 3.2 1B by searching for the optimal `ffn_intermediate_size` across MLP layers, resulting in a heterogeneous architecture while reducing GPU memory usage by 20%.

## Compress the Model

```bash
torchrun examples/compress/main.py \
  --config path/to/llama_3.2_1B_pruneffn_memory.yaml
```

## Evaluate Model Accuracy

```bash
# TODO
```

## Re-run MIP Search with Different Memory Constraints

If you want to try different memory constraints without re-running the expensive pruning and scoring steps, use the `--mip-only` flag:

```bash
torchrun examples/compress/main.py \
  --config path/to/llama_3.2_1B_pruneffn_memory.yaml \
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

Modify `path/to/Llama-3_2-1B yaml` file for advanced compression scenarios.
