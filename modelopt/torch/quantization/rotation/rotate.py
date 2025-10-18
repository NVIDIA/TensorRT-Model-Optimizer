"""Rotation preprocessing for quantization."""

import fnmatch

import torch
import yaml

from .online_rotations import OnlineHadamardTransform, create_online_transform_hook
from .rotate_utils import (
    apply_full_rotation,
    apply_per_head_rotation,
    fuse_layernorms,
    get_orthogonal_matrix,
)


def _is_pow2(n: int) -> bool:
    """Check if n is a power of 2."""
    return (n & (n - 1) == 0) and (n > 0)


def _iter_modules_by_pattern(
    model: torch.nn.Module, pattern: str
) -> list[tuple[str, torch.nn.Module]]:
    """Find all modules matching the given glob pattern.

    Supports OR logic with pipe separator: '*q_proj|*k_proj' matches either pattern.
    """
    results: list[tuple[str, torch.nn.Module]] = []
    patterns = pattern.split("|")

    for name, module in model.named_modules():
        for p in patterns:
            if fnmatch.fnmatch(name, p):
                results.append((name, module))
                break
    return results


def _apply_weight_rotation(
    weight: torch.Tensor, rin: torch.Tensor | None, rout: torch.Tensor | None
) -> torch.Tensor:
    """Apply rotation to weight tensor."""
    w = weight.to(torch.float32)
    if rin is not None:
        w = torch.matmul(w, rin.to(w))
    if rout is not None:
        w = torch.matmul(rout.to(w).t(), w)
    return w.to(weight.dtype)


def _apply_bias_rotation(bias: torch.Tensor, rout: torch.Tensor) -> torch.Tensor:
    """Apply rotation to bias tensor (only for rout/left multiply case)."""
    b = bias.to(torch.float32)
    b_rotated = torch.matmul(rout.to(b).t(), b)
    return b_rotated.to(bias.dtype)


def build_rotation_config(config: dict, model: torch.nn.Module) -> dict:
    """Build rotation config with all matrices generated and dimensions filled."""
    cfg = model.config
    hidden_size = cfg.hidden_size
    num_heads = getattr(cfg, "num_attention_heads", None)
    num_layers = getattr(cfg, "num_hidden_layers", len(model.model.layers))
    head_dim = hidden_size // num_heads if num_heads else hidden_size

    # Generate rotation matrices
    device = next(model.parameters()).device
    rot_mats = {}

    # Generate R1 (full hidden_size)
    rot_mats["r1"] = get_orthogonal_matrix(hidden_size, "hadamard", device)

    # Generate per-head R2 matrices per layer (for V projection)
    per_layer_r2 = {}
    if num_heads is not None:
        for i in range(num_layers):
            per_layer_r2[f"r2_{i}"] = get_orthogonal_matrix(head_dim, "hadamard", device)

    # Fill per-head config dimensions
    per_head_cfg = config.get("per_head_config", {})
    for spec in per_head_cfg.values():
        spec.setdefault("num_heads", num_heads)
        spec.setdefault("head_dim", head_dim)

    return {
        "rotation_matrices": rot_mats,
        "per_layer_r2": per_layer_r2,
        "rotation_config": config.get("rotation_config", {}),
        "per_head_config": per_head_cfg,
        "norm_fuse_config": config.get("norm_fuse_config"),
        "head_dim": head_dim,
        "device": device,
    }


def build_rotation_config_from_yaml(yaml_path: str, model: torch.nn.Module) -> dict:
    """Load rotation config from YAML and build with matrices generated."""
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    return build_rotation_config(config, model)


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

    rot_mats = config["rotation_matrices"]
    per_layer_r2 = config["per_layer_r2"]
    rotation_cfg = config["rotation_config"]
    per_head_cfg = config["per_head_config"]
    head_dim = config["head_dim"]
    device = config["device"]

    for pattern, (rin_name, rout_name) in rotation_cfg.items():
        for module_name, module in _iter_modules_by_pattern(model, pattern):
            if not hasattr(module, "weight"):
                continue

            per_head_spec = per_head_cfg.get(pattern)
            if per_head_spec and per_head_spec.get("enabled"):
                # Start with original weight
                new_w = module.weight.data

                # Apply R1 rotation FIRST (matches SpinQuant order)
                if rin_name in rot_mats:
                    new_w = _apply_weight_rotation(new_w, rot_mats[rin_name], None)
                elif rout_name in rot_mats:
                    new_w = _apply_weight_rotation(new_w, None, rot_mats[rout_name])
                    # Also rotate bias if it exists (matching SpinQuant)
                    if hasattr(module, "bias") and module.bias is not None:
                        module.bias.data.copy_(
                            _apply_bias_rotation(module.bias.data, rot_mats[rout_name])
                        )

                # Then apply R2 rotation
                transpose_first = per_head_spec.get("transpose_first", False)
                use_per_head = per_head_spec.get("use_per_head", True)

                if use_per_head:
                    # V/O projection: per-head R2 (matching SpinQuant)
                    layer_idx = int(module_name.split(".")[2]) if "layers." in module_name else 0
                    r2 = per_layer_r2.get(f"r2_{layer_idx}")
                    if r2 is not None:
                        if transpose_first:
                            # V projection: per-head R2 on output dimension
                            weight_out_dim = new_w.shape[0]
                            actual_num_heads = weight_out_dim // head_dim
                        else:
                            # O projection: per-head R2 on input dimension
                            weight_in_dim = new_w.shape[1]
                            actual_num_heads = weight_in_dim // head_dim

                        new_w = apply_per_head_rotation(
                            new_w, r2, actual_num_heads, transpose_first
                        )
                else:
                    # Down projection: full R2 on input dimension
                    input_dim = new_w.shape[1]
                    mode = "hadamard" if _is_pow2(input_dim) else "random"
                    r2_full = get_orthogonal_matrix(input_dim, mode, device)
                    new_w = apply_full_rotation(new_w, r2_full)

                module.weight.data.copy_(new_w)
            else:
                rin = rot_mats.get(rin_name) if rin_name else None
                rout = rot_mats.get(rout_name) if rout_name else None
                module.weight.data.copy_(_apply_weight_rotation(module.weight.data, rin, rout))

    # Register online activation transforms for O and down projections
    register_online_transforms(model, config)
