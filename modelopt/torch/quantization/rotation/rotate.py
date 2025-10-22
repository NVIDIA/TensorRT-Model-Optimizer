"""Rotation preprocessing for quantization."""

from pathlib import Path

import torch
import yaml

from .rotate_utils import (
    RotationMatrixStore,
    _iter_modules_by_pattern,
    fuse_layernorms,
    rotate_bias,
    rotate_weight,
)


def build_rotation_config_from_yaml(yaml_path: str | Path) -> dict:
    """Load rotation config from YAML and build with matrices generated."""
    # Resolve path relative to this module's directory if it's a relative path
    yaml_path = Path(yaml_path)
    if not yaml_path.is_absolute() and not yaml_path.is_file():
        module_dir = Path(__file__).parent
        yaml_path = module_dir / "configs" / yaml_path
        if not yaml_path.exists() and not yaml_path.suffix:
            yaml_path = yaml_path.with_suffix(".yaml")
    yaml_path = yaml_path.resolve()

    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    return config


def apply_rotation(model: torch.nn.Module, config: str) -> None:
    """Apply R1/R2 rotations to model weights and register online transforms.

    Args:
        model (torch.nn.Module): The model to apply rotations to.
        config (str): The path to the YAML configuration file.
    """
    config = build_rotation_config_from_yaml(config)
    if config.get("norm_fuse_config"):
        print("Fusing layer norms")
        fuse_layernorms(model, config["norm_fuse_config"])
    use_float64 = config.get("use_float64", True)

    rotation_cfg = config["rotation_config"]
    rot_mat_store = RotationMatrixStore(config["rotation_matrices"], model)
    for pattern, (rin_name, rout_name) in rotation_cfg.items():
        for module_name, module in _iter_modules_by_pattern(model, pattern):
            if not hasattr(module, "weight"):
                continue
            print(f"Applying rotation to {module_name}, input: {rin_name}, output: {rout_name}")
            rin = rot_mat_store.get(rin_name, module_name) if rin_name else None
            rout = rot_mat_store.get(rout_name, module_name) if rout_name else None
            module.weight.data.copy_(
                rotate_weight(module.weight, rin, rout, use_float64=use_float64).data
            )
            if hasattr(module, "bias") and module.bias is not None and rout is not None:
                module.bias.data.copy_(rotate_bias(module.bias, rout, use_float64=use_float64).data)

    rot_mat_store.clear()
