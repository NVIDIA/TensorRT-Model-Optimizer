# Evaluation scripts for LLM tasks

This folder includes popular 3rd-party LLM benchmarks for LLM accuracy evaluation.

The following instructions show how to evaluate the Model Optimizer quantized LLM with the benchmarks, including the TensorRT-LLM deployment.

## MMLU

[Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300). A score (0-1, higher is better) will be printed at the end of the benchmark.

### Setup

Download data

```bash
mkdir -p data
wget https://people.eecs.berkeley.edu/~hendrycks/data.tar -O data/mmlu.tar
tar -xf data/mmlu.tar -C data && mv data/data data/mmlu
cd ..
```

### Baseline

```bash
python mmlu.py --model_name causal --model_path <HF model folder or model card>
```

### Quantized (simulated)

```bash
# MODELOPT_QUANT_CFG: Choose from [INT8_SMOOTHQUANT_CFG|FP8_DEFAULT_CFG|INT4_AWQ_CFG|W4A8_AWQ_BETA_CFG]
python mmlu.py --model_name causal --model_path <HF model folder or model card> --quant_cfg MODELOPT_QUANT_CFG
```

### Evaluate the TensorRT-LLM engine

```bash
python mmlu.py --model_name causal --model_path <HF model folder or model card> --engine_dir <built TensorRT-LLM folder>
```

## Human-eval

[HumanEval](https://arxiv.org/abs/2107.03374). A score (0-1, higher is better) will be printed at the end of the benchmark.

> *Due to various prompt and generation postprocessing methods, the final score might be different compared with the published numbers from the model developer.*

### Setup

Clone [Instruct-eval](https://github.com/declare-lab/instruct-eval/tree/main) and add a softlink to folder [human_eval](https://github.com/declare-lab/instruct-eval/tree/main/human_eval) from `instruct_eval/`

### Baseline

```sh
python humaneval.py --model_name causal --model_path <HF model folder or model card> --n_sample 1
```

### Quantized (simulated)

```sh
# MODELOPT_QUANT_CFG: Choose from [INT8_SMOOTHQUANT_CFG|FP8_DEFAULT_CFG|INT4_AWQ_CFG|W4A8_AWQ_BETA_CFG]
python humaneval.py --model_name causal --model_path <HF model folder or model card> --n_sample 1 --quant_cfg MODELOPT_QUANT_CFG
```

### Evaluate the TRT-LLM engine

```sh
python humaneval.py --model_name causal --model_path <HF model folder or model card> --engine_dir <built TensorRT-LLM folder> --n_sample 1
```
