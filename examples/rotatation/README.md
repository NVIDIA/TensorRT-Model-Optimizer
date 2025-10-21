# Model Rotation for Quantization

This module implements the fusible rotation technique from QuaRot/SpinQuant papers, specifically the R1 and R2 rotation matrices ([SpinQuant paper](https://arxiv.org/pdf/2405.16406)). Rotation helps improve quantization accuracy by redistributing activation outliers.

## Available Configurations

We provide several pre-built configurations:
- `nemotron_h` - Optimized for Nemotron models
- `transformer_universal` - General configuration for transformer-based models (e.g., LLaMA, Qwen)

## Configuration Format

Below is an example configuration for universal transformer models:

```yaml
# Universal transformer rotation configuration

rotation_matrices:
  r1:
    dim: "hidden_size"  # Can be an integer or a key from model.config (e.g., "hidden_size")
    mode: hadamard      # Initialization mode: 'hadamard' (random Hadamard), 'base_hadamard' (basic Hadamard by Sylvester's construction), or 'orthogonal' (random orthogonal)
    per_layer: false    # false = model-wise shared; true = layer-wise shared; or a string pattern to specify layer grouping
  r2:
    dim: "head_dim"
    mode: hadamard
    per_layer: true

# Module rotation specification
rotation_config:
  # Embeddings and language model head
  "*embed_tokens": [r1, null]
  "*lm_head": [r1, null]

  # Attention projections
  "*q_proj|*k_proj": [r1, null]     # Q/K: R1 on input
  "*v_proj": [r1, r2]                # V: R1 on input, per-head R2 on output
  "*o_proj": [r2, r1]                # O: per-head R2 on input, R1 on output

  # MLP projections
  "*up_proj|*gate_proj|*w1|*w3": [r1, null]  # Gate/up: R1 on input
  "*down_proj|*w2": [null, r1]               # Down: R1 on output

  # Mamba2/hybrid architecture support
  "*in_proj": [r1, null]
  "*out_proj": [null, r1]

# LayerNorm fusion configuration
# Note: LayerNorm must be fused to adjacent linear layers before rotation to preserve rotational invariance
norm_fuse_config:
  decoder_layer_fuse:
    - [input_layernorm, [self_attn.q_proj, self_attn.k_proj, self_attn.v_proj]]
    - [post_attention_layernorm, [mlp.up_proj, mlp.gate_proj]]
  lm_head_fuse:
    - [model.norm, lm_head]

# Precision settings
use_float64: true  # Use float64 for higher numerical accuracy during rotation
```

## Usage

### Python API

```python
import modelopt.torch.quantization as mtq
from transformers import AutoModelForCausalLM

# Load your model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")

# Apply rotation with a config file or dictionary
mtq.rotate(model, config_path_or_dict="transformer_universal")

# Save the rotated model
model.save_pretrained("output/Llama-3-8B-rotated")
```

### Command-Line Script

We provide a convenience script for transformer models:

```bash
python main.py \
    --model "meta-llama/Meta-Llama-3-8B" \
    --config transformer_universal \
    --output Llama-3-8B-rotated
```

## Next Steps

After rotation, the model can be:
- Loaded as a standard HuggingFace model
- Quantized using your preferred framework
- Deployed with improved quantization accuracy



