# Chaining Multiple Optimizations Techniques

This example demonstrates how to chain multiple optimization techniques like Pruning, Distillation, and Quantization together to
achieve the best performance on a given model.

## HuggingFace BERT Pruning + Distillation + Quantization

This example shows how to compress a [Hugging Face Bert large model for Question Answering](https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad)
using the combination of `modelopt.torch.prune`, `modelopt.torch.distill` and `modelopt.torch.quantize`. More specifically, we will:

1. Prune the Bert large model to 50% FLOPs with GradNAS algorithm and fine-tune with distillation
1. Quantize the fine-tuned model to INT8 precision with Post-Training Quantization (PTQ) and Quantize Aware Training (QAT) with distillation
1. Export the quantized model to ONNX format for deployment with TensorRT

The main python file is [bert_prune_distill_quantize.py](./bert_prune_distill_quantize.py) and scripts for running it
for all 3 steps are available in the [scripts](./scripts/) directory.
More details on this example (including highlighted code snippets) can be found in the Model Optimizer documentation
[here](https://nvidia.github.io/TensorRT-Model-Optimizer/examples/2_bert_prune_distill_quantize.html)
