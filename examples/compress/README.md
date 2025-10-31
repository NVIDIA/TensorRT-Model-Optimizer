# Compress Algorithm Tutorial

This tutorial demonstrates how to compress large language models using the compress algorithm based on the [Puzzle paper](https://arxiv.org/abs/2411.19146).

In this example, we compress Llama 3.2 1B by searching for the optimal `ffn_intermediate_size` across MLP layers, resulting in a heterogeneous architecture while reducing GPU memory usage by 20%.

## Compress the Model

```bash
# TODO
torchrun examples/compress/main.py
```

## Evaluate Model Accuracy

```bash
# TODO
```

## Re-run MIP Search with Different Memory Constraints

```bash
# TODO
```

## Deploy to TensorRT-LLM

```bash
# TODO
```

## Export to NeMo for Knowledge Distillation

```bash
# TODO
```
