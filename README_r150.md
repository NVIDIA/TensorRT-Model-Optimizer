# Setup Guide

## Requirements

Hopper GPUs. PSX-Formats does not work with Ampere GPUs.

## Install ModelOpt from modelopt-r-150 fork

```bash
git clone ssh://git@gitlab-master.nvidia.com:12051/omniml/modelopt-r-150.git
git checkout YOUR_BRANCH

# Enable ModelOpt installation overwrite - Needed for Nvidia containers with pre-installed ModelOpt
export PIP_CONSTRAINT="" 
pip install -e .[dev]

# [Optional] Pre-compile ModeOpt cuda extensions;
# ModelOpt cuda extensions are compiled and cached on the first runtime use
python -c "import modelopt.torch.quantization.extensions as ext; ext.precompile()"

```

For more details on ModelOpt installation, see [ModelOpt docs](https://nvidia.github.io/TensorRT-Model-Optimizer/getting_started/2_installation.html)

For ModelOpt API docs see [here](https://nvidia.github.io/TensorRT-Model-Optimizer/reference/generated/modelopt.torch.quantization.html#module-modelopt.torch.quantization) (Based on Github Open sourced main branch - could have mismatches; but generally should work).

## Install PSX-Formats

```bash
git clone https://gitlab-master.nvidia.com/compute/psx/psx-formats/psx-formats.git
cd psx-formats
git submodule update --init --recursive
python setup.py build_ext --inplace

# Add to PYTHONPATH (adjust /PATH_TO/ to your path)
export PYTHONPATH=/PATH_TO/psx-formats:$PYTHONPATH
```

## Usage Example

See `tests/gpu/torch/quantization/plugins/test_psx_formats.py`.
