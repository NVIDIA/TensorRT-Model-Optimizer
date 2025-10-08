# Perplexity Evaluation Tool

## Overview

This tool evaluates the perplexity of ONNX Runtime GenAI models using the [WikiText-2](https://huggingface.co/datasets/wikitext) dataset. Perplexity is a standard metric for language models: lower values indicate better predictive performance.

## Attribution

This script is originally based on [perplexity_metrics.py](https://github.com/microsoft/onnxruntime-genai/blob/main/tools/python/model_validation/perplexity_metrics.py) from the Microsoft ONNX Runtime GenAI repository. It has been modified to handle:

- Multiple context lengths
- Configurable chunk sizes
- Enhanced prefill chunking handling

## Scripts

- **`perplexity_metrics.py`**: Core evaluation logic for computing perplexity.
- **`run_perplexity.py`**: Command-line utility for evaluating one or more models and saving results to CSV.

## Requirements

- Python 3.8+
- Install dependencies:

  ```bash
  pip install -r requirements.txt
  ```

- [HuggingFace CLI](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli) login is required to access the WikiText-2 dataset:

  ```bash
  huggingface-cli login
  ```

## Supported Models

- Any ONNX Runtime GenAI model exported with a compatible `genai_config.json` and tokenizer.
- Supported architectures include: Gemma, Llama, Mistral, Phi (language + vision), Qwen.
- Supported execution providers: CPU, DirectML, CUDA, NvTensorRtRtx.

## How to Run

### Evaluate a Single Model

```bash
python run_perplexity.py --models /path/to/model
```

### Multiple models

```bash
python run_perplexity.py --models /path/to/model1 /path/to/model2
```

### Custom input sequence length(s)

You can specify the input sequence length(s) to evaluate using the `--i` argument.  
For example, to evaluate with input lengths:
Note: higher isl is only supported when model has Kv chunking enabled in genai config

```bash
python run_perplexity.py --models /path/to/model --i 1024,2048,4096,8192,12288
```

### Custom prefill chunk size

You can specify the prefill chunk size to evaluate using the `--chunk_size` argument.  
For example:
Note: higher isl is only supported when model has Kv chunking enabled in genai config

```bash
python run_perplexity.py --models /path/to/model --i 1024,2048,4096,8192,12288 --chunk_size=1024
```

### Custom output file

```bash
python run_perplexity.py --models /path/to/model --output results.csv
```

## Expected output

The expected score is between 2 to 1000 , lower score means better model performance

### Perplexity configuration setting

- If **kv_chunking** is enabled in the model configuration (i.e., `"chunk_size"` is present in the `"search"` section of `genai_config.json`), then:
  - `max_input_seq_length` is set to **8192**
  - `stride` is set to the value of `chunk_size`
- If **kv_chunking** is not enabled (default):
  - `max_input_seq_length` is **1024**
  - `stride` is **512**

### Console output

```text
============================================================
Evaluating perplexity for: /path/to/model
============================================================
[INFO] Loading Wikitext-2 'test' split ...
[TOKENIZER] Tokenizing ...

[RESULT] Perplexity of /path/to/model: 45.28

Perplexity evaluation completed successfully
```

### CSV output

Generated file contains:

- Model Path
- Perplexity score
- Status (Success/Failed)
- Error details (if any)

## Debug mode

Set `DEBUG = True` in `perplexity_metrics.py` for detailed logs.

## Typical perplexity ranges

- Excellent : 2-20
- Good: 20-40
- Ok: 40-80  
- Poor: 100+
