"""Rotation preprocessing for quantization."""

import torch
import yaml

from .online_rotations import OnlineHadamardTransform, create_online_transform_hook
from .rotate_utils import (
    RotationMatrixStore,
    _iter_modules_by_pattern,
    fuse_layernorms,
    rotate_bias,
    rotate_weight,
)

# def _apply_weight_rotation(
#     weight: torch.Tensor, rin: torch.Tensor | None, rout: torch.Tensor | None
# ) -> torch.Tensor:
#     """Apply rotation to weight tensor."""
#     w = weight.to(torch.float32)
#     if rin is not None:
#         w = torch.matmul(w, rin.to(w))
#     if rout is not None:
#         w = torch.matmul(rout.to(w).t(), w)
#     return w.to(weight.dtype)


# def _apply_bias_rotation(bias: torch.Tensor, rout: torch.Tensor) -> torch.Tensor:
#     """Apply rotation to bias tensor (only for rout/left multiply case)."""
#     b = bias.to(torch.float32)
#     b_rotated = torch.matmul(rout.to(b).t(), b)
#     return b_rotated.to(bias.dtype)


# def build_rotation_config(config: dict, model: torch.nn.Module) -> dict:
#     """Build rotation config with all matrices generated and dimensions filled."""
#     cfg = model.config
#     hidden_size = cfg.hidden_size
#     num_heads = getattr(cfg, "num_attention_heads", None)
#     num_layers = getattr(cfg, "num_hidden_layers", len(model.model.layers))
#     head_dim = hidden_size // num_heads if num_heads else hidden_size

#     # Generate rotation matrices
#     device = next(model.parameters()).device
#     rot_mats = {}

#     # Generate R1 (full hidden_size)
#     rot_mats["r1"] = get_orthogonal_matrix(hidden_size, "hadamard", device)

#     # Generate per-head R2 matrices per layer (for V projection)
#     per_layer_r2 = {}
#     if num_heads is not None:
#         for i in range(num_layers):
#             per_layer_r2[f"r2_{i}"] = get_orthogonal_matrix(head_dim, "hadamard", device)

#     # Fill per-head config dimensions
#     per_head_cfg = config.get("per_head_config", {})
#     for spec in per_head_cfg.values():
#         spec.setdefault("num_heads", num_heads)
#         spec.setdefault("head_dim", head_dim)

#     return {
#         "rotation_matrices": rot_mats,
#         "per_layer_r2": per_layer_r2,
#         "rotation_config": config.get("rotation_config", {}),
#         "per_head_config": per_head_cfg,
#         "norm_fuse_config": config.get("norm_fuse_config"),
#         "head_dim": head_dim,
#         "device": device,
#     }


def build_rotation_config_from_yaml(yaml_path: str, model: torch.nn.Module) -> dict:
    """Load rotation config from YAML and build with matrices generated."""
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    # return build_rotation_config(config, model)
    print("config:", config)
    return config


def register_online_transforms(model: torch.nn.Module, config: dict) -> None:
    """Register forward hooks for online Hadamard transforms on O and down projections."""
    per_head_cfg = config.get("per_head_config", {})
    per_layer_r2 = config.get("per_layer_r2", {})
    head_dim = config.get("head_dim")

    if not head_dim:
        return

    for pattern, spec in per_head_cfg.items():
        if not spec.get("enabled"):
            continue

        transpose_first = spec.get("transpose_first", False)
        if transpose_first:
            continue  # V projection doesn't need online transform

        # O and down projections need online transforms
        use_per_head = spec.get("use_per_head", True)

        for module_name, module in _iter_modules_by_pattern(model, pattern):
            if not isinstance(module, torch.nn.Linear):
                continue

            if use_per_head:
                # O projection: per-head Hadamard with R2
                layer_idx = int(module_name.split(".")[2]) if "layers." in module_name else 0
                r2 = per_layer_r2.get(f"r2_{layer_idx}")
                transform = OnlineHadamardTransform(had_dim=head_dim, r2_matrix=r2)
            else:
                # Down projection: full Hadamard
                transform = OnlineHadamardTransform(had_dim=-1, r2_matrix=None)

            # Register forward pre-hook
            hook = create_online_transform_hook(transform)
            handle = module.register_forward_pre_hook(hook)
            module._rotation_hook_handle = handle


def apply_rotation(model: torch.nn.Module, config: dict) -> None:
    """Apply R1/R2 rotations to model weights and register online transforms."""
    if config.get("norm_fuse_config"):
        fuse_layernorms(model, config["norm_fuse_config"])

    # rot_mats =
    rotation_cfg = config["rotation_config"]
    rot_mat_store = RotationMatrixStore(config["rotation_matrices"], model)

    for pattern, (rin_name, rout_name) in rotation_cfg.items():
        for module_name, module in _iter_modules_by_pattern(model, pattern):
            if not hasattr(module, "weight"):
                continue
            rin = rot_mat_store.get(rin_name, module_name) if rin_name else None
            rout = rot_mat_store.get(rout_name, module_name) if rout_name else None
            module.weight.data.copy_(rotate_weight(module.weight, rin, rout))
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.copy_(rotate_bias(module.bias, rout))

    # Register online activation transforms for O and down projections
    # register_online_transforms(model, config)
