# Speculative Decoding (SpecDec) Bench

## Purpose

Collect relevant metrics on acceptance rate, timing, and outputs for Speculative Decoding methods.
Acceptance rate refers to the number of tokens generated on every iteration.  For a standard Autoregressive LLM, this number
is just 1.  

## Getting Started

A basic example run script is provided which benchmarks MTBench (a standard 160 prompts spanning 8 categories).
MTBench is available [here](https://huggingface.co/datasets/HuggingFaceH4/mt_bench_prompts)

### Running MTBench on GPT OSS + Eagle3

Download `nvidia/gpt-oss-120b-Eagle3` to a local directory `/path/to/eagle`.

```bash
python3 run.py --model_dir openai/gpt-oss-120b --tokenizer openai/gpt-oss-120b --draft_model_dir /path/to/eagle --mtbench question.jsonl --tp_size 1 --ep_size 1 --draft_length 3 --output_length 4096 --num_requests 80 --engine TRTLLM --concurrency 1

```

### Running Random ids on GPT OSS + Eagle3

Download `nvidia/gpt-oss-120b-Eagle3` to a local directory `/path/to/eagle`.

```bash
python3 run.py --model_dir openai/gpt-oss-120b --tokenizer openai/gpt-oss-120b --draft_model_dir /path/to/eagle --random_isl 1024 --tp_size 1 --ep_size 1 --draft_length 3 --output_length 4096 --num_requests 40 --engine TRTLLM --concurrency 1

```
