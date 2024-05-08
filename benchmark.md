# TensorRT Model Optimizer Benchmark Reference

This document summarizes performance and accuracy measurements of [TensorRT Model Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer) for a few popular models.
The benchmark in the following tables is provided as reference points and **should not be considered as the peak
performance** that can be delivered by Model Optimizer. All performance numbers are tested with [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) or [TensorRT](https://developer.nvidia.com/tensorrt-getting-started).

### 1. Post-training quantization (PTQ) for LLMs

#### 1.1 Performanace

Config: H100, nvidia-modelopt v0.11.0, TensorRT-LLM v0.9, latency measured with full batch inference (no inflight batching).
Memory saving and inference speedup are compared to the FP16 baseline. Speedup is normalized to the GPU count.

|            |            |            |     FP8    |         |   |            |  INT4 AWQ  |         |
|:----------:|:----------:|:----------:|:----------:|:-------:|:-:|:----------:|:----------:|:-------:|
|    Model   | Batch Size | Mem Saving | Tokens/sec | Speedup |   | Mem Saving | Tokens/sec | Speedup |
|  Llama3-8B |      2     |    1.66x   |   337.67   |  1.39x  |   |    2.37x   |   392.99   |  1.61x  |
|            |     32     |    1.56x   |   2368.69  |  1.66x  |   |    1.86x   |   2037.54  |  1.43x  |
|            |     64     |    1.54x   |   2404.86  |  1.43x  |   |    1.76x   |   2308.57  |  1.37x  |
| Llama3-70B |      2     |    1.98x   |    64.35   |  2.11x  |   |    3.49x   |    77.36   |  2.54x  |
|            |     32     |    1.95x   |   391.73   |  3.03x  |   |    2.94x   |   479.11   |  3.71x  |
|            |     64     |    1.91x   |   383.42   |  2.41x  |   |    2.46x   |   348.65   |  2.19x  |

### 1.2 Accuracy

The table below shows the MMLU loss in percentage compared to FP16 baseline.
Config: H100, nvidia-modelopt v0.11.0, TenorR-LLM v0.9.
Note that typically FP8 or INT4 AWQ is the go-to choices for H100.
More benchmark with earlier version of Model Optimizer can be found in this [TensorRT-LLM README](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/quantization-in-TRT-LLM.md#benchmark).
|     Model \\ MMLU loss       | FP8  | INT4 AWQ |
|:----------:|:-------------:|:--------:|
|  Llama3-8B |     0.46%     |    4.60% |
| Llama3-70b |     0.51%     |    1.29% |

## 2. PTQ for Stable Diffusion

The following table shows inference speedup for INT8 and FP8 on a Stable Diffusion XL 1.0 base model compared to the FP16 baseline.
Config: Image resolution=1024×1024, 30 steps. TensorRT v9.3. num-warmup-runs=1. Batch size=1.
|       GPU      | INT8 Latency (ms) | FP8 Latency (ms) | Speedup (INT8 v.s. FP16) | Speedup (FP8 v.s. FP16) |
|:--------------:|:-----------------:|:----------------:|:------------------------:|:-----------------------:|
|  RTX 6000 Ada  |      2,479.19     |     2,441.16     |           1.43x          |          1.45x          |
|    RTX 4090    |      2,058.11     |     2,161.38     |           1.20x          |          1.14x          |
|      L40S      |      2,338.88     |     2,167.82     |           1.25x          |          1.35x          |
| H100 80GB HBM3 |      1,209.04     |     1,216.18     |           1.08x          |          1.07x          |

## 3. Quantization-aware training

The below table demonstrates the validation loss of Quantization-aware training (QAT) compared to PTQ of a Llama 2 7B model using nvidia-modelopt v0.11.0.
The baseline is fine-tuned on the target dataset. Note that we use INT4 to showcase that QAT can better preserve model accuracy at low precision. This implies that QAT can be applied with a low training cost, enabling generative AI applications that are sensitive to accuracy drop to preserve accuracy even at ultra-low precisions where both weight and activations are 4-bit for [NVIDIA Blackwell platform](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/).

|            Method            |        Dataset       | Val loss - BF16 Baseline | Val loss - PTQ | Val loss - QAT (lower is better) |
|:----------------------------:|:--------------------:|:------------------------:|:--------------:|:--------------:|
| INT4 Weight, FP16 Activation |        samsum        |           1.036          |      1.059     |      **1.044**     |
| INT4 Weight, INT8 Activation |        samsum        |           1.036          |      3.321     |      **1.294**     |
| INT4 Weight, FP16 Activation | databricks-dolly-15k |           1.151          |      1.305     |      **1.172**     |
| INT4 Weight, INT8 Activation | databricks-dolly-15k |           1.151          |      2.313     |      **1.640**     |

## 4. Sparsity

### 4.1 Performance

The table shows the inference speedup of a sparsified Llama 2 70B model compared to the baseline dense model in different batch sizes.
The benchmark with batch_size=896 is part of [MLPerf Inference v4.0](https://developer.nvidia.com/blog/nvidia-h200-tensor-core-gpus-and-nvidia-tensorrt-llm-set-mlperf-llm-inference-records/).
Config: NVIDIA H100 80GB GPU. FP8, TP=1, PP=1 for all sparsified models. The dense model needs TP=2 due to larger weight sizes.

| Batch Size | Inference speedup (compared to the FP8 dense model) |
|:----------:|:---------------------------------------------------:|
|     32     |                        1.62x                        |
|     64     |                        1.52x                        |
|     128    |                        1.35x                        |
|     896    |                        1.30x                        |

### 4.2 Accuracy

We recommend using sparsity with fine-tuning to avoid accuracy degradation.
The following table shows the comparison of validation loss of a Llama 2 70B using sparsity with and without fine-tuning. Finetuning and validation are done on the Open-Orca dataset.

|                 Method           | Validation loss (lower is better) |
|:--------------------------------:|:---------------------------------:|
|          FP8 (baseline)          |                             0.721 |
| FP8 + SparseGPT, no fine-tuning  |                             2.724 |
| FP8 + Sparsity, with fine-tuning |                          **1.01** |
