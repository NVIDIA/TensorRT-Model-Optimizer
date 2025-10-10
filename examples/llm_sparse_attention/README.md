# Sparse Attention for Large Language Models

This example demonstrates how to apply sparse attention optimization to Large Language Models (LLMs) using TensorRT-Model-Optimizer's attention sparsity module.

<div align="center">

| **Section** | **Description** | **Link** | **Docs** |
| :------------: | :------------: | :------------: | :------------: |
| Pre-Requisites | Required & optional packages to use this technique | \[[Link](#pre-requisites)\] | |
| Getting Started | Learn how to apply sparse attention to optimize inference efficiency | \[[Link](#getting-started)\] | \[[docs](https://nvidia.github.io/TensorRT-Model-Optimizer/)\] |
| Support Matrix | View the support matrix to see sparse attention compatibility across different models | \[[Link](#support-matrix)\] | |
| Framework Scripts | Example scripts demonstrating sparse attention techniques for optimizing models | \[[Link](#framework-scripts)\] | |
| Evaluate Accuracy | Evaluate your model's accuracy with sparse attention! | \[[Link](#evaluate-accuracy)\] | |
| Exporting Checkpoints | Export to Hugging Face Unified Checkpoint and deploy on TRT-LLM/vLLM/SGLang | \[[Link](#exporting-checkpoints)\] | \[[docs](https://nvidia.github.io/TensorRT-Model-Optimizer/deployment/3_unified_hf.html)\] |
| Pre-Sparsified Checkpoints | Ready to deploy Hugging Face pre-sparsified checkpoints | \[[Link](#pre-sparsified-checkpoints)\] | |
| Resources | Extra links to relevant resources | \[[Link](#resources)\] | |

</div>

## Overview

Sparse attention reduces the computational complexity of attention mechanisms by selectively computing only the most important attention scores. This can significantly speed up inference and reduce memory usage, especially for long sequences.

## Features

- **Sparse Attention Method**:
  - Softmax Skip: Threshold-based masking for efficient attention computation
  - Extensible architecture: Easy to add new sparse attention methods in the future
- **Calibration Support**: Automatically find optimal sparsity parameters
- **HuggingFace Integration**: Works with any HuggingFace transformer model
- **Composable**: Can be combined with quantization and other optimizations

## Pre-Requisites

### Docker

For Hugging Face models, please use the TensorRT-LLM docker image (e.g., `nvcr.io/nvidia/tensorrt-llm/release:1.1.0rc2.post2`).
For NeMo models, use the NeMo container (e.g., `nvcr.io/nvidia/nemo:25.07`).
Visit our [installation docs](https://nvidia.github.io/TensorRT-Model-Optimizer/getting_started/2_installation.html) for more information.

Also follow the installation steps below to upgrade to the latest version of Model Optimizer and install example-specific dependencies.

### Local Installation

For Hugging Face models, install Model Optimizer with `hf` dependencies using `pip` from [PyPI](https://pypi.org/project/nvidia-modelopt/) and install the requirements for the example:

```bash
pip install -U nvidia-modelopt[hf]
pip install -r requirements.txt
```

For TensorRT-LLM deployment, please use the TensorRT-LLM docker image or follow their [installation docs](https://nvidia.github.io/TensorRT-LLM/installation/index.html).
Similarly, for vLLM or SGLang deployment, please use their installation docs.

> *When loading models from HuggingFace, `trust_remote_code=False` is used by default for security. If your model requires custom code, you may need to modify the script to set `trust_remote_code=True` in the `AutoModelForCausalLM.from_pretrained()` call.*

> *If model loading fails on a multi-GPU system due to mismatched tensor placement, try setting CUDA_VISIBLE_DEVICES to limit the number of visible GPUs.*

> *For large models with limited GPU memory, adjust `--seq_len` or `--num_samples` parameters. You can also modify the script to use HuggingFace's `device_map="auto"` feature in model loading to automatically distribute across GPUs.*

## Getting Started

```python
import modelopt.torch.sparsity as mts  # Similar to mtq for quantization
from transformers import AutoModelForCausalLM

# Load model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B")

# Define sparse attention config
sparse_config = {
    "method": "softmax_skip",
    "sparse_cfg": {
        "*attn*": {"threshold": 1e-4, "enable": True},
        "default": {"enable": False}
    }
}

# Apply sparse attention
sparse_model = mts.attention_sparsity.sparsify(model, config=sparse_config)

# Use the model as usual
output = sparse_model.generate(input_ids, max_new_tokens=100)
```

### Command Line Usage

The `hf_spar_attn.py` script applies sparse attention to HuggingFace models:

```bash
# Basic usage: Apply sparse attention and test generation
python hf_spar_attn.py --pyt_ckpt_path Qwen/Qwen3-8B

# With output verification: Compare baseline vs sparse attention outputs
python hf_spar_attn.py --pyt_ckpt_path Qwen/Qwen3-8B --verify_output

# Export to unified HuggingFace checkpoint  
python hf_spar_attn.py --pyt_ckpt_path Qwen/Qwen3-8B --export_dir ./sparse_model
```

## Examples

### Basic Usage

Apply sparse attention to a model and test generation quality:

```bash
python hf_spar_attn.py --pyt_ckpt_path Qwen/Qwen3-8B \
    --sparse_attn skip_softmax \
    --verify_output
```

Available Options:

- `--pyt_ckpt_path`: Model checkpoint path or HuggingFace model card (required)
- `--sparse_attn`: Sparse attention method (default: skip_softmax)
- `--verify_output`: Compare baseline vs sparse attention outputs
- `--export_dir`: Export model to specified directory
- `--backend`: Backend for computation - pytorch or triton (default: pytorch)
- `--seq_len`: Maximum sequence length (default: 2048)
- `--num_samples`: Number of test samples from NarrativeQA (default: 3)
- `--max_new_tokens`: Maximum new tokens to generate (default: 50)
- `--do_sample`: Use sampling for generation
- `--temperature`: Temperature for sampling (default: 0.7)

Note: Sparsity statistics are automatically displayed after applying sparse attention.

## Exporting Checkpoints

### Export Model

Export the sparse attention model to unified HuggingFace checkpoint format for deployment:

```bash
python hf_spar_attn.py --pyt_ckpt_path Qwen/Qwen3-8B \
    --sparse_attn skip_softmax \
    --export_dir ./sparse_model
```

The exported model will contain:

- Model weights with sparse attention applied
- `config.json` with `sparse_attention_config` section
- Tokenizer files

### Exported Config Format

The `config.json` includes a `sparse_attention_config` section using the `config_groups` pattern (similar to `quantization_config`):

**For calibrated models:**

```json
{
  "sparse_attention_config": {
    "config_groups": {
      "group_0": {
        "sparse_algo": "softmax_skip",
        "targets": ["LlamaAttention"]
      }
    },
    "threshold_scale_factor": 437.7,
    "target_sparsity": 0.3,
    "producer": {
      "name": "modelopt",
      "version": "0.37.0"
    }
  }
}
```

**For non-calibrated models:**

```json
{
  "sparse_attention_config": {
    "config_groups": {
      "group_0": {
        "sparse_algo": "softmax_skip",
        "threshold": 0.0001,
        "targets": ["LlamaAttention"]
      }
    },
    "producer": {
      "name": "modelopt",
      "version": "0.37.0"
    }
  }
}
```

This format enables inference engines to reconstruct the sparse attention configuration from the checkpoint.

### Deployment

Deployment examples for TensorRT-LLM, vLLM, and SGLang will be added soon.

### Unified HF Checkpoint Deployment Model Support Matrix

Support matrix showing which models and sparse attention methods work with each deployment framework will be added.

## Evaluate Accuracy

### Accuracy Validation

Evaluating the impact of sparse attention on model accuracy is crucial. The `hf_spar_attn.py` script provides built-in support for validation through the `--verify_output` flag, which compares outputs between baseline and sparse attention models.

Sparsity statistics are automatically displayed to help monitor sparsity levels across different attention layers:

```bash
python hf_spar_attn.py --pyt_ckpt_path Qwen/Qwen3-8B \
    --sparse_attn skip_softmax \
    --verify_output
```

For comprehensive accuracy evaluation, additional benchmarks are available in the [llm_eval](../llm_eval/README.md) directory, including:

- MMLU (Massive Multitask Language Understanding)
- lm_evaluation_harness for various language modeling tasks

Please refer to the [llm_eval README](../llm_eval/README.md) for detailed instructions on running these evaluation benchmarks.

> *Sparsity statistics are automatically displayed for each attention layer. Monitor these carefully to ensure the threshold settings are achieving the desired balance between performance and accuracy.*

> *Different models may have varying sensitivity to sparse attention. It's recommended to evaluate on task-specific benchmarks relevant to your use case.*

## Support Matrix

### Hugging Face Supported Models

Support matrix will be added as testing is completed for various models and sparse attention methods.

> *This section is under active development. The sparse attention feature is currently being validated across different model architectures.*

## Framework Scripts

### Hugging Face Example

For LLM models like [Llama](https://huggingface.co/meta-llama) or [Qwen](https://huggingface.co/Qwen):

```bash
# Apply sparse attention and test generation
python hf_spar_attn.py --pyt_ckpt_path Qwen/Qwen3-8B \
    --sparse_attn skip_softmax \
    --verify_output

# Export to unified HuggingFace checkpoint
python hf_spar_attn.py --pyt_ckpt_path Qwen/Qwen3-8B \
    --sparse_attn skip_softmax \
    --export_dir ./sparse_model
```

**Key Command-Line Flags:**

- `--pyt_ckpt_path`: Model checkpoint path or HuggingFace model card (required)
- `--sparse_attn`: Sparse attention method - currently supports `skip_softmax` (default) and `skip_softmax_calib`
- `--verify_output`: Compare baseline vs sparse attention outputs for validation
- `--export_dir`: Directory to save the exported sparse model
- `--backend`: Backend for computation - `pytorch` or `triton` (default: pytorch)
- `--seq_len`: Maximum sequence length for input prompts (default: 2048)
- `--num_samples`: Number of test samples from NarrativeQA dataset (default: 3)
- `--max_new_tokens`: Maximum new tokens to generate (default: 50)

Note: Sparsity statistics are automatically displayed after applying sparse attention - no flag needed.

> *When loading models from HuggingFace, `trust_remote_code=False` is used by default for security. If the model requires custom code, you'll need to manually set `trust_remote_code=True` in the model loading code.*

> *If GPU out-of-memory error is reported, try reducing `--seq_len` or `--num_samples`. For very large models, consider using HuggingFace's `device_map="auto"` feature in the model loading code to distribute across GPUs.*

> *Sparse attention works best with models using eager attention implementation. Models with fused attention kernels may require modifications.*

### NeMo Example Script

NeMo framework sparse attention examples will be added in future releases.

### Megatron-LM Example Script

Megatron-LM framework sparse attention examples will be added in future releases.

## Configuration Options

### Pre-defined Configuration

ModelOpt provides a unified configuration that supports both simple and phase-aware thresholds:

```python
import modelopt.torch.sparsity as mts

# The default config supports phase-aware thresholds
SOFTMAX_SKIP_CFG = {
    "method": "softmax_skip",
    "sparse_cfg": {
        "*attn*": {
            "threshold": {
                "prefill": 1e-3,  # More aggressive during prefill
                "decode": 1e-5,   # Conservative during decode  
            },
            "enable": True,
        },
        "default": {"enable": False},
    },
}

# Use the config
model = mts.attention_sparsity.sparsify(model, config=SOFTMAX_SKIP_CFG)
```

### Custom Configuration

You can create custom configurations with simple or phase-aware thresholds:

```python
# Simple threshold (same for all phases)
simple_config = {
    "method": "softmax_skip",
    "sparse_cfg": {
        "*attn*": {
            "threshold": 1e-4,  # Single threshold for all phases
            "enable": True,
        },
        "default": {"enable": False},
    }
}

# Phase-aware threshold
phase_aware_config = {
    "method": "softmax_skip",
    "sparse_cfg": {
        "*attn*": {
            "threshold": {
                "prefill": 1e-3,    # Prefill phase
                "decode": 1e-5,     # Decode phase
            },
            "enable": True,
        },
        "default": {"enable": False},
    }
}
```

### Adding Custom Methods

The architecture is designed to easily support new sparse attention methods. Refer to [`FlashSoftmaxSkipMethod`](../../modelopt/torch/sparsity/attention_sparsity/methods/flash_softmax_skip.py) source code for implementing custom methods.

### Pattern-Based Configuration

Apply different configurations to different layers:

```python
config = {
    "sparse_cfg": {
        "*layers.[0-12].*attention*": {"enable": True, "threshold": 1e-3},  # More aggressive for early layers
        "*layers.[13-24].*attention*": {"enable": True, "threshold": 1e-4},  # Conservative for later layers
    }
}
```

## Performance Considerations

1. **Threshold Tuning**:
   - Lower thresholds (e.g., 1e-5) preserve more accuracy but less sparsity
   - Higher thresholds (e.g., 1e-3) provide more sparsity but may impact accuracy
   - Use calibration to find optimal values

2. **Memory Usage**:
   - Sparse attention reduces peak memory usage during inference
   - Especially beneficial for long sequences (>1024 tokens)

3. **Model Compatibility**:
   - Works best with models using eager attention implementation
   - Compatible with all HuggingFace transformer models

## Resources

- 📅 [Roadmap](https://github.com/NVIDIA/TensorRT-Model-Optimizer/issues/146)
- 📖 [Documentation](https://nvidia.github.io/TensorRT-Model-Optimizer)
- 🎯 [Benchmarks](../benchmark.md)
- 💡 [Release Notes](https://nvidia.github.io/TensorRT-Model-Optimizer/reference/0_changelog.html)
- 🐛 [File a bug](https://github.com/NVIDIA/TensorRT-Model-Optimizer/issues/new?template=1_bug_report.md)
- ✨ [File a Feature Request](https://github.com/NVIDIA/TensorRT-Model-Optimizer/issues/new?template=2_feature_request.md)

### Technical Resources

Sparse attention reduces the computational cost of attention mechanisms by selectively computing attention scores. The primary method currently supported is:

1. **Softmax Skip**: A threshold-based approach that skips computation of attention scores below a certain threshold. This method is particularly effective for long sequences where many attention scores are near zero. The implementation is available in [`FlashSoftmaxSkipMethod`](../../modelopt/torch/sparsity/attention_sparsity/methods/flash_softmax_skip.py).

**Further Reading:**

- [Sparse Attention Papers Collection](https://github.com/topics/sparse-attention)
- [TensorRT-Model-Optimizer Sparse Attention Documentation](https://nvidia.github.io/TensorRT-Model-Optimizer/)

## Pre-Sparsified Checkpoints

Pre-sparsified model checkpoints will be made available on Hugging Face in the future.

- Ready-to-deploy checkpoints will be published to the [🤗 Hugging Face - Nvidia TensorRT Model Optimizer Collection](https://huggingface.co/collections/nvidia/model-optimizer-66aa84f7966b3150262481a4)
- Deployable on [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM), [vLLM](https://github.com/vllm-project/vllm) and [SGLang](https://github.com/sgl-project/sglang)
- More models coming soon!
