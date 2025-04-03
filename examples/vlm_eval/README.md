# Evaluation scripts for VLM tasks

This folder includes popular 3rd-party VLM benchmarks for VLM accuracy evaluation.

The following instructions show how to evaluate the VLM (including Model Optimizer quantized LLM) with the benchmarks, including the TensorRT-LLM deployment.

## GQA

[GQA: a dataset for real-world visual reasoning and compositional question answering](https://arxiv.org/abs/1902.09506). Upon completing the benchmark, the model's accuracy (in percentage format) will be displayed, providing a clear metric for performance evaluation.

First log in to Hugging Face account with your token.

```bash
huggingface-cli login
```

### Baseline

```bash
bash gqa.sh --hf_model <dir_of_hf_model>
```

### Quantized (simulated)

```bash
# MODELOPT_QUANT_CFG: Choose from [INT8_SMOOTHQUANT_CFG|FP8_DEFAULT_CFG|INT4_AWQ_CFG|W4A8_AWQ_BETA_CFG]
bash gqa.sh --hf_model <dir_hf_model> --quant_cfg MODELOPT_QUANT_CFG
```

### Evaluate the TensorRT-LLM engine

TensorRT engine could be built following this [guide](../vlm_ptq/README.md)

```bash
bash gqa.sh --hf_model <dir_hf_model> --visual_engine <dir_visual_engine> --llm_engine <dir_llm_engine>
```

If you encounter Out of Memory (OOM) issues during evaluation, you can try lowering the `--kv_cache_free_gpu_memory_fraction` argument (default is 0.8) to reduce GPU memory usage for kv_cache:

```bash
bash gqa.sh --hf_model <dir_hf_model> --visual_engine <dir_visual_engine> --llm_engine <dir_llm_engine> --kv_cache_free_gpu_memory_fraction 0.5
```
