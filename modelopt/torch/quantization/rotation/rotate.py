import fnmatch
from collections.abc import Iterable
from dataclasses import dataclass, field

import torch

from .rotate_utils import fuse_layernorms, get_orthogonal_matrix


@dataclass
class RotationMatrixSpec:
    dim: int
    mode: str  # "random" | "hadamard" | variants like "random hadamard"


@dataclass
class NormFuseConfig:
    decoder_fuse: list[tuple[str, list[str]]] = field(default_factory=list)
    lm_head_fuse: list[tuple[str, str]] = field(default_factory=list)

    def decoder_layer_fuse(self) -> Iterable[tuple[str, list[str]]]:
        return self.decoder_fuse


@dataclass
class RotationFlowConfig:
    rotation_matrices: dict[str, RotationMatrixSpec]
    rotation_config: dict[str, tuple[str | None, str | None]]
    norm_fuse_config: NormFuseConfig | None = None


def _coerce_rotation_matrices(
    spec: dict[str, dict[str, int | str]], device: torch.device
) -> dict[str, torch.Tensor]:
    mats: dict[str, torch.Tensor] = {}
    for name, info in spec.items():
        dim = int(info["dim"])  # required
        mode = str(info["mode"])  # required
        mats[name] = get_orthogonal_matrix(dim, mode, device)
    return mats


def _iter_modules_by_pattern(
    model: torch.nn.Module, pattern: str
) -> list[tuple[str, torch.nn.Module]]:
    results: list[tuple[str, torch.nn.Module]] = []
    for name, module in model.named_modules():
        if fnmatch.fnmatch(name, pattern):
            results.append((name, module))
    return results


def _apply_weight_rotation(
    weight: torch.Tensor, rin: torch.Tensor | None, rout: torch.Tensor | None
) -> torch.Tensor:
    # weight: [out_features, in_features]
    w = weight.to(torch.float32)
    if rin is not None:
        w = torch.matmul(w, rin.to(w))
    if rout is not None:
        w = torch.matmul(rout.to(w).t(), w)
    return w.to(weight.dtype)


@torch.inference_mode()
def apply_rotation(model: torch.nn.Module, config: RotationFlowConfig) -> None:
    if config.norm_fuse_config is not None:
        fuse_layernorms(model, config.norm_fuse_config)

    # Allow both dataclass and plain dict usage for rotation_matrices
    if isinstance(next(iter(config.rotation_matrices.values())), RotationMatrixSpec):
        spec_dict = {k: {"dim": v.dim, "mode": v.mode} for k, v in config.rotation_matrices.items()}
    else:
        spec_dict = config.rotation_matrices  # type: ignore[assignment]

    rot_mats = _coerce_rotation_matrices(spec_dict, model.device)

    for pattern, (rin_name, rout_name) in config.rotation_config.items():
        modules = _iter_modules_by_pattern(model, pattern)
        rin = rot_mats.get(rin_name) if rin_name is not None else None
        rout = rot_mats.get(rout_name) if rout_name is not None else None
        for _, module in modules:
            if not hasattr(module, "weight"):
                continue
            new_w = _apply_weight_rotation(module.weight.data, rin, rout)
            module.weight.data.copy_(new_w)


def build_rotation_flow_config(obj: dict) -> RotationFlowConfig:
    # rotation_matrices: { name: {dim, mode} }
    rm_specs = {
        name: RotationMatrixSpec(dim=spec["dim"], mode=spec["mode"])
        for name, spec in obj.get("rotation_matrices", {}).items()
    }

    # rotation_config: { pattern: [rin, rout] | (rin, rout) }
    rot_cfg: dict[str, tuple[str | None, str | None]] = {}
    for pattern, pair in obj.get("rotation_config", {}).items():
        if isinstance(pair, (list, tuple)):
            rin, rout = pair
        else:
            raise ValueError("rotation_config entries must be a 2-item list/tuple")
        rin = None if rin in [None, "", "None"] else str(rin)
        rout = None if rout in [None, "", "None"] else str(rout)
        rot_cfg[str(pattern)] = (rin, rout)

    # norm_fuse_config optional
    nf = obj.get("norm_fuse_config")
    norm_cfg: NormFuseConfig | None = None
    if nf is not None:
        decoder_fuse = []
        for item in nf.get("decoder_layer_fuse", []):
            ln = item[0]
            linears = item[1]
            decoder_fuse.append((str(ln), [str(x) for x in linears]))
        lm_head_fuse = []
        for item in nf.get("lm_head_fuse", []):
            lm_head_fuse.append((str(item[0]), str(item[1])))
        norm_cfg = NormFuseConfig(decoder_fuse=decoder_fuse, lm_head_fuse=lm_head_fuse)

    return RotationFlowConfig(
        rotation_matrices=rm_specs, rotation_config=rot_cfg, norm_fuse_config=norm_cfg
    )
