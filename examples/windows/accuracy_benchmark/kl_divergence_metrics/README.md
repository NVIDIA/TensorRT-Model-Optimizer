# KL Divergence Model Validation Toolkit

This toolkit provides comprehensive model validation capabilities using KL divergence metrics to compare different model implementations and execution providers. It's designed to evaluate the similarity between model outputs across different optimization techniques and hardware backends.

## Overview

The toolkit includes several Python scripts for:

1. **Extracting logits** from both Hugging Face and ONNX Runtime GenAI models
2. **Computing KL divergence** between model pairs or multiple models
3. **Comparing execution providers** (CUDA, DirectML, CPU) against baseline models
4. **Validating model optimization** quality by measuring output similarity

## Key Components

### Core Scripts

| Script | Purpose | Usage |
|--------|---------|--------|
| `extract_logits_hf.py` | Extract logits from Hugging Face models using transformers | Baseline model logit extraction |
| `extract_logits.py` | Extract logits from ONNX Runtime GenAI models | Optimized model logit extraction |
| `KL_divergence_metrics_same_ep.py` | Compare two ONNX Runtime GenAI models directly | Same execution provider comparison |
| `compute_kl_divergence.py` | Unified comparison framework  | All-in-one comparison tool |

### Datasets Used

- **Wikitext-2** test split for consistent evaluation across all models
- Automatic dataset loading and preprocessing via HuggingFace datasets

## Installation

1. **Install base requirements:**

   ```bash
   pip install -r requirements.txt
   ```

   Note: Install torch with cuda for faster inference "pip install torch torchvision torchaudio --index-url <https://download.pytorch.org/whl/cu129>"

2. **Install execution provider-specific packages** (as needed):

   ```bash
   # For CUDA support
   pip install onnxruntime-genai-cuda
   
   # For DirectML support  
   pip install onnxruntime-genai-directml
   
   # For CPU support (default)
   pip install onnxruntime-genai
   ```

## Usage Examples

### 1. Unified Comparison Tool

The `compute_kl_divergence.py` script is the unified tool for all comparison scenarios:

#### Compare HF baseline vs ONNX models

```bash
python compute_kl_divergence.py \
    --hf_model "path/to/hf/model" \
    --ep cuda --path "path/to/cuda/model" \
    --output "hf_vs_cuda_results.json"
```

#### Compare HF vs multiple execution providers

```bash
python compute_kl_divergence.py \
    --hf_model "path/to/hf/model" \
    --ep cuda --path "path/to/cuda/model" \
    --ep directml --path "path/to/directml/model" \
    --output "multi_provider_comparison.json"
```

#### Compare ONNX models WITHOUT HF baseline (NEW!)

```bash
# Two models with same EP (automatically uses optimized same_ep script)
python compute_kl_divergence.py \
    --ep cuda --path "path/to/cuda_fp16/model" \
    --ep cuda --path "path/to/cuda_int4/model" \
    --output "cuda_fp16_vs_int4.json"

# Two models with different EPs
python compute_kl_divergence.py \
    --ep cuda --path "path/to/cuda/model" \
    --ep directml --path "path/to/directml/model" \
    --output "cuda_vs_directml.json"

# Multiple ONNX models with mixed EPs
python compute_kl_divergence.py \
    --ep cuda --path "path/to/cuda_fp16/model" \
    --ep cuda --path "path/to/cuda_int4/model" \
    --ep directml --path "path/to/directml/model" \
    --output "multi_onnx_comparison.json"
```

#### Enable debug output for detailed logging

```bash
python compute_kl_divergence.py \
    --ep cuda --path "path/to/model1" \
    --ep directml --path "path/to/model2" \
    --output "results.json" \
    --debug
```

### 2. Direct Same-EP Comparison (Alternative)

For comparing two ONNX models with the same execution provider, you can also use:

```bash
python KL_divergence_metrics_same_ep.py \
    --reference_model "path/to/reference/model" \
    --target_model "path/to/target/model"
```

### 3. Extract Logits Separately (Advanced)

If you need to extract logits separately for reuse:

#### From Hugging Face Model

```bash
python extract_logits_hf.py \
    --model_path "path/to/huggingface/model" \
    --output_file "hf_logits.pkl" \
    --device cuda \
    --debug
```

#### From ONNX Runtime GenAI Model

```bash
python extract_logits.py \
    --model_path "path/to/onnx/model" \
    --output_file "onnx_logits.pkl" \
    --provider cuda \
    --debug
```

## Configuration Parameters

### compute_kl_divergence.py Parameters

- `--hf_model`: Path to Hugging Face baseline model (optional - can compare ONNX models directly)
- `--ep`: Execution provider (can be specified multiple times for multiple models)
  - Supported: `cuda`, `directml`, `cpu`
- `--path`: Model path (must match order of --ep arguments)
- `--output`: Output JSON file for results (required)
- `--device`: Device for HF model inference (default: `cuda`, choices: `cuda`, `cpu`)
- `--keep_logits`: Keep extracted logits files after comparison
- `--debug`: Enable verbose debug output with detailed logging

### Other Script Parameters

- `--model_path`: Path to model (for extract_logits scripts)
- `--output_file`: Output file for extracted logits (`.pkl` format)
- `--provider`: Execution provider for ONNX models (`cuda`, `directml`, `cpu`)
- `--reference_model`: Reference model path (for same_ep script)
- `--target_model`: Target model path (for same_ep script)

### Model Processing Parameters

- **Max context length**: 1024 tokens (configurable in code)
- **Chunk processing**: Automatic chunking for memory management
- **Deterministic generation**: No sampling for consistent results

## Output Files and Interpretation

### Logits Files (`.pkl`)

Pickled files containing:

- **logits**: List of numpy arrays with model logits per chunk
- **chunk_info**: Metadata about each processed chunk
- **model_path**: Path to the source model
- **provider**: Execution provider used
- **total_chunks**: Number of chunks processed

### Results Files (`.json`)

JSON files containing:

- **models**: Paths to all compared models
- **kl_divergences**: Pairwise KL divergence values
  - `total`: Sum of KL divergences across all chunks
  - `average`: Mean KL divergence per chunk
- **chunk_results**: Detailed per-chunk analysis
- **summary**: Interpretation and key metrics

### Example Results Structure

```json
{
  "models": {
    "huggingface": "path/to/hf/model",
    "cuda": "path/to/cuda/model",
    "directml": "path/to/directml/model"
  },
  "kl_divergences": {
    "huggingface_vs_cuda": {
      "total": ,
      "average": 
    },
    "huggingface_vs_directml": {
      "total": ,
      "average": 
    },
    "cuda_vs_directml": {
      "total": ,
      "average": 
    }
  },
  "summary": {
    "interpretation": "Lower KL divergence indicates more similar model outputs",
    "baseline_reference": "huggingface",
    "pairwise_averages": {
      "huggingface_vs_cuda": ,
      "huggingface_vs_directml": ,
      "cuda_vs_directml": 
    }
  }
}
```

## Interpreting KL Divergence Values

| KL Divergence Range | Interpretation |
|-------------------|----------------|
| **0 - 1** | Nearly identical outputs |
| **1 - 10** | Very similar outputs |
| **10 - 50** | Moderately similar outputs |
| **50+** | Significantly different outputs |

### Key Insights from Results

- **Lower values** indicate better optimization quality (closer to baseline)
- **Baseline comparison** shows how much optimization affects output quality
- **Provider comparison** reveals differences between execution backends
- **Consistency check** ensures model optimization maintains output quality

## Key Features

### Flexible Comparison Modes

1. **HF vs ONNX models**: Compare Hugging Face baseline against one or more ONNX models
2. **ONNX-only comparison**: Compare ONNX models directly without HF baseline
3. **Mixed execution providers**: Compare models across different hardware backends
4. **Multiple same-EP models**: Compare multiple variants of the same execution provider

### Output Verbosity Control

- **Without `--debug`**: Clean, minimal progress output showing key steps
  - Model extraction progress
  - Environment switching notifications
  - Computation progress
  - Final results summary
- **With `--debug`**: Comprehensive logging including:
  - Detailed model paths and configurations
  - Package installation details
  - Chunk-by-chunk processing
  - Validation warnings
  - Temporary file management
  - Full error tracebacks

### Automatic Package Management

- The script automatically installs/uninstalls the correct ONNX Runtime packages based on execution provider
- Minimizes package switching by reusing environments when possible
- Handles CUDA, DirectML, and CPU providers seamlessly

## Notes

- The comparison uses the Wikitext-2 dataset for evaluation
- Processing is done in chunks (1024 tokens) to handle memory constraints
- The script automatically handles package installation/uninstallation for different providers
- Results are deterministic (no sampling) for consistent comparisons
- All pairwise comparisons are computed for multi-model scenarios
- HF model is  optional - you can compare ONNX models directly
