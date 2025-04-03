# Evaluation scripts for LLM tasks

This folder includes popular 3rd-party LLM benchmarks for LLM accuracy evaluation.

The following instructions show how to evaluate the Model Optimizer quantized LLM with the benchmarks, including the TensorRT-LLM deployment.

## LM-Eval-Harness

[LM-Eval-Harness](https://github.com/EleutherAI/lm-evaluation-harness) provides a unified framework to test generative language models on a large number of different evaluation tasks.

The supported eval tasks are [here](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks).

### Baseline

- For models which fit on a single GPU:

```sh
python lm_eval_hf.py --model hf --model_args pretrained=<HF model folder or model card> --tasks <comma separated tasks> --batch_size 4
```

- With model-sharding (for models which require multiple GPUs):

```sh
python lm_eval_hf.py --model hf --model_args pretrained=<HF model folder or model card>,parallelize=True --tasks <comma separated tasks> --batch_size 4
```

- For data-parallel evaluation with model-sharding:

With the following command, the model will be sharded across `total_num_of_available_gpus/num_copies_of_your_model` with a data-parallelism of `num_copies_of_your_model`

```sh
accelerate launch --multi_gpu --num_processes <num_copies_of_your_model> \
    lm_eval_hf.py --model hf \
    --tasks <comma separated tasks> \
    --model_args pretrained=<HF model folder or model card>,parallelize=True \
    --batch_size 4
```

### Quantized (simulated)

- For simulated quantization with any of the default quantization formats:

Multi-GPU evaluation without data-parallelism:

```sh
# MODELOPT_QUANT_CFG: Choose from [INT8_SMOOTHQUANT_CFG|FP8_DEFAULT_CFG|NVFP4_DEFAULT_CFG|INT4_AWQ_CFG|W4A8_AWQ_BETA_CFG|MXFP8_DEFAULT_CFG]
python lm_eval_hf.py --model hf \
    --tasks <comma separated tasks> \
    --model_args pretrained=<HF model folder or model card>,parallelize=True \
    --quant_cfg <MODELOPT_QUANT_CFG> \
    --batch_size 4
```

> **_NOTE:_** `MXFP8_DEFAULT_CFG` is one the [OCP Microscaling Formats (MX Formats)](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) family which defines a set of block-wise dynamic quantization formats. The specifications can be found in the [offical documentation](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf). Currently we support all MX formats for simulated quantization, including `MXFP8 (E5M2, E4M3), MXFP6 (E3M2, E2M3), MXFP4, MXINT8`. However, only `MXFP8 (E4M3)` is in our example configurations, users can create their own configurations for other MX formats by simply modifying the `num_bits` field in the `MXFP8_DEFAULT_CFG`.

> **_NOTE:_** ModelOpt's triton kernels give faster NVFP4 simulated quantization. For details, please see the [installation guide](https://nvidia.github.io/TensorRT-Model-Optimizer/getting_started/_installation_for_Linux.html#accelerated-quantization-with-triton-kernels).

For data-parallel evaluation, launch with `accelerate launch --multi_gpu --num_processes <num_copies_of_your_model>` (as shown earlier).

- For simulated optimal per-layer quantization with `AutoQuantize`:

Multi-GPU evaluation without data-parallelism:

```sh
# MODELOPT_QUANT_CFG_TO_SEARCH: Choose the formats to search separated by commas from [W4A8_AWQ_BETA_CFG,FP8_DEFAULT_CFG|NVFP4_DEFAULT_CFG,NONE]
# EFFECTIVE_BITS: Effective bits constraint for AutoQuantize

# Examples settings for optimally quantized model with W4A8 & FP8 with effective bits to 4.8:
# MODELOPT_QUANT_CFG_TO_SEARCH=W4A8_AWQ_BETA_CFG,FP8_DEFAULT_CFG|NVFP4_DEFAULT_CFG,NONE
# EFFECTIVE_BITS=4.8

python lm_eval_hf.py --model hf \
    --tasks <comma separated tasks> \
    --model_args pretrained=<HF model folder or model card>,parallelize=True \
    --quant_cfg <AUTOQUANTIZE_SEARCH_FORMATS> \
    --auto_quantize_bits <EFFECTIVE_BITS> \
    --batch_size 4
```

For data-parallel evaluation, launch with `accelerate launch --multi_gpu --num_processes <num_copies_of_your_model>` (as shown earlier).

- If evaluating T5 models:

  - use `--model hf-seq2seq` instead.

```sh
# MODELOPT_QUANT_CFG: Choose from [INT8_SMOOTHQUANT_CFG|FP8_DEFAULT_CFG|NVFP4_DEFAULT_CFG|INT4_AWQ_CFG|W4A8_AWQ_BETA_CFG|MXFP8_DEFAULT_CFG]
python lm_eval_hf.py --model hf-seq2seq --model_args pretrained=t5-small --quant_cfg=<MODELOPT_QUANT_CFG> --tasks <comma separated tasks> --batch_size 4
```

If `trust_remote_code` needs to be true, please append the command with the `--trust_remote_code` flag.

### TensorRT-LLM

```sh
python lm_eval_tensorrt_llm.py --model trt-llm --model_args tokenizer=<HF model folder>,engine_dir=<TRT LLM engine dir> --tasks <comma separated tasks> --batch_size <engine batch size>
```

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
# MODELOPT_QUANT_CFG: Choose from [INT8_SMOOTHQUANT_CFG|FP8_DEFAULT_CFG|NVFP4_DEFAULT_CFG|INT4_AWQ_CFG|W4A8_AWQ_BETA_CFG|MXFP8_DEFAULT_CFG]
python mmlu.py --model_name causal --model_path <HF model folder or model card> --quant_cfg MODELOPT_QUANT_CFG
```

### AutoQuantize (simulated)

```bash
# MODELOPT_QUANT_CFG_TO_SEARCH: Choose the formats to search separated by commas from [W4A8_AWQ_BETA_CFG,FP8_DEFAULT_CFG|NVFP4_DEFAULT_CFG,NONE]
# EFFECTIVE_BITS: Effective bits constraint for AutoQuantize

# Examples settings for optimally quantized model with W4A8 & FP8 with effective bits to 4.8:
# MODELOPT_QUANT_CFG_TO_SEARCH=W4A8_AWQ_BETA_CFG,FP8_DEFAULT_CFG|NVFP4_DEFAULT_CFG,NONE
# EFFECTIVE_BITS=4.8

python mmlu.py --model_name causal --model_path <HF model folder or model card> --quant_cfg $MODELOPT_QUANT_CFG_TO_SEARCH --auto_quantize_bits $EFFECTIVE_BITS --batch_size 4
```

### Evaluate the TensorRT-LLM engine

```bash
python mmlu.py --model_name causal --model_path <HF model folder or model card> --engine_dir <built TensorRT-LLM folder>
```

## MT-Bench

[MT-Bench](https://arxiv.org/abs/2306.05685). These responses are generated using [FastChat](https://github.com/lm-sys/FastChat).

### Baseline

```bash
bash run_fastchat.sh -h <HF model folder or model card>
```

### Quantized (simulated)

```bash
# MODELOPT_QUANT_CFG: Choose from [INT8_SMOOTHQUANT_CFG|FP8_DEFAULT_CFG|INT4_AWQ_CFG|W4A8_AWQ_BETA_CFG]
bash run_fastchat.sh -h <HF model folder or model card> --quant_cfg MODELOPT_QUANT_CFG
```

### Evaluate the TensorRT-LLM engine

```bash
bash run_fastchat.sh -h <HF model folder or model card> <built TensorRT-LLM folder>
```

### Judging the responses

The responses to questions from MT Bench will be stored under `data/mt_bench/model_answer`.
The quality of the responses can be judged using [llm_judge](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge) from the FastChat repository. Please refer to the [llm_judge](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge) to compute the final MT-Bench score.

## LiveCodeBench

[LiveCodeBench](https://livecodebench.github.io/) is a holistic and contamination-free evaluation benchmark of LLMs for code that continuously collects new problems over time.

We support running LiveCodeBench against a local running OpenAI API compatible server. For example, quantized TensorRT-LLM checkpoint or engine can be loaded using [trtllm-serve](https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve.html) command. Once the local server is up, the following command can be used to run the LiveCodeBench:

```bash
bash run_livecodebench.sh <custom defined model name> <prompt batch size in parallel> <max output tokens> <local model server port>
```

## Simple Evals

[Simple Evals](https://github.com/openai/simple-evals) is a lightweight library for evaluating language models published from OpenAI. This eval includes "simpleqa", "mmlu", "math", "gpqa", "mgsm", "drop" and "humaneval" benchmarks.

Similarly, we support running simple evals against a local running OpenAI API compatible server. Once the local server is up, the following command can be used to run the Simple Evals:

```bash
bash run_simple_eval.sh <custom defined model name> <comma separated eval names> <max output tokens> <local model server port>
```
