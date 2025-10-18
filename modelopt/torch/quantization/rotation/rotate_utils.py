"""Rotation utility functions for preprocessing."""

import fnmatch
import gc
import re
import typing

import torch

from .hadamard_utils import matmul_hadU_cuda, random_hadamard_matrix


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


def fuse_ln_linear(
    layernorm: torch.nn.Module, linear_layers: typing.Iterable[torch.nn.Linear]
) -> None:
    """Fuse the linear operations in Layernorm into the adjacent linear blocks."""
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype

        # Calculating new weight and bias
        w_ = linear.weight.data.double()
        linear.weight.data = (w_ * layernorm.weight.double()).to(linear_dtype)

        if hasattr(layernorm, "bias"):
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(
                    torch.zeros(linear.out_features, dtype=torch.float64)
                )
            linear.bias.data = linear.bias.data.double() + torch.matmul(w_, layernorm.bias.double())
            linear.bias.data = linear.bias.data.to(linear_dtype)
            torch.nn.init.zeros_(linear.bias)
    torch.nn.init.ones_(layernorm.weight)


@torch.no_grad()
def fuse_layernorms(model: torch.nn.Module, norm_fuse_config: dict) -> None:
    """Fuse LayerNorm scale/bias into neighboring Linear layers as configured.

    Args:
        model: Model to apply fusion to
        norm_fuse_config: Dict with keys:
            - decoder_layer_fuse: List[tuple[str, list[str]]] - decoder layer norm fusions
            - lm_head_fuse: List[tuple[str, str]] - lm head norm fusions
    """
    # Embedding fusion
    emb_module = model.model.embed_tokens
    emb_weight = emb_module.weight.data.double()
    emb_module.weight.data = (emb_weight - emb_weight.mean(dim=-1, keepdim=True)).to(
        emb_module.weight.dtype
    )

    for name, module in model.named_modules():
        if re.search("layers\\.[0-9]+$", name) is not None:
            for layer_norm, linear_layers in norm_fuse_config.get("decoder_layer_fuse", []):
                fuse_ln_linear(
                    module.get_submodule(layer_norm),
                    [module.get_submodule(linear) for linear in linear_layers],
                )
    for layer_norm, linear in norm_fuse_config.get("lm_head_fuse", []):
        fuse_ln_linear(model.get_submodule(layer_norm), [model.get_submodule(linear)])


def random_orthogonal_matrix(size: int, device: torch.device) -> torch.Tensor:
    """Generate a random orthogonal matrix of the specified size on device."""
    return torch.nn.init.orthogonal_(torch.empty(size, size, device=device))


def _normalize_mode(mode: str) -> str:
    m = mode.lower().strip().replace("_", " ")
    if "hadamard" in m:
        return "hadamard"
    return "random"


def get_orthogonal_matrix(size: int, mode: str, device: torch.device) -> torch.Tensor:
    """Get an orthogonal matrix of the specified size and mode on device."""
    mode = _normalize_mode(mode)
    if mode == "random":
        return random_orthogonal_matrix(size, device)
    if mode == "hadamard":
        return random_hadamard_matrix(size, device)
    raise ValueError(f"Unknown rotation matrix mode: {mode}")


def extract_layer_index(module_name: str) -> int | None:
    """Extract layer index from module name.

    Args:
        module_name: Module name like 'model.layers.12.self_attn.o_proj'

    Returns:
        Layer index as integer, or None if not found
    """
    match = re.search(r"layers\.(\d+)", module_name)
    return int(match.group(1)) if match else None


def diag_matmul(
    m1: torch.Tensor, m2: torch.Tensor, transpose=True, block_size=None
) -> torch.Tensor:
    """m1 @ m2. m1 can be an element matrix of a diagonal matrix. e.g. diag(m1, m1, ..., m1).

    In which case, there should be m2.shape[0] % m1.shape[1] == 0.
    If m1 is not a diagnoal, peform a normal matrix multiplication.
    """
    if transpose:
        m1 = m1.t()
    if m1.shape[-1] == m2.shape[-2]:
        return torch.matmul(m1, m2)
    m2_shape = m2.shape
    size = m1.shape[1]
    m2 = m2.view(*m2.shape[:-2], m2.shape[-2] // size, size, m2.shape[-1])
    # print(m1.shape, m2.shape)
    output = m1 @ m2  # torch.matmul(m1, m2)
    return output.contiguous().view(*m2_shape)


def matmul_diag(
    m1: torch.Tensor, m2: torch.Tensor, transpose=False, block_size=None
) -> torch.Tensor:
    """m1 @ m2. m2 can be an element matrix of a diagonal matrix. e.g. diag(m2, m2, ..., m2).

    In which case, there should be m1.shape[-1] % m2.shape[0] == 0.
    If m2 is not a diagnoal, peform a normal matrix multiplication.
    """
    if transpose:
        # print("transposed")
        m2 = m2.t()
    if m1.shape[-1] == m2.shape[-2]:
        return torch.matmul(m1, m2)
    m1_shape = m1.shape
    size = m2.shape[0]
    # assert block_size == size
    m1 = m1.view(*m1.shape[:-1], m1.shape[-1] // size, size)
    output = m1 @ m2
    return output.contiguous().view(*m1_shape)


def matmul_diag_fast_hadamard(m1, had_k, transpose=False, block_size=None):
    """Fast hadamard transform, m1 @ hadamard matrix, if m1.shape[-1] is not a power of 2, a had_k must be provided.

    The hadamard matrix is by Sylvester’s construction, which is also a symmetric matrix. So transpose is not needed.
    """
    if block_size is not None:
        shape = m1.shape
        m1 = m1.view(*shape[:-1], shape[-1] // block_size, block_size)

        return matmul_hadU_cuda(m1, None).view(*shape)
    return matmul_hadU_cuda(m1, had_k)


def rotate_weight(weight, input_spin, output_spin, fast_hadamard=False, block_size=None):
    """Rotating a matrix."""
    dtype = weight.dtype
    weight = weight.to(torch.float32)
    if input_spin is not None:
        input_spin = input_spin.to(torch.float32)
    if output_spin is not None:
        output_spin = output_spin.to(torch.float32)
    matmul_spin = matmul_diag_fast_hadamard if fast_hadamard else matmul_diag
    if input_spin is not None or (input_spin is None and fast_hadamard and block_size is not None):
        weight = matmul_spin(
            weight, input_spin, block_size=block_size
        )  # torch.matmul(weight, input_spin)
    if output_spin is not None:
        weight = diag_matmul(output_spin, weight)
    return weight.to(dtype)


def rotate_bias(bias, output_spin, fast_hadamard=False, block_size=None):
    """Rotate a bias."""
    dtype = bias.dtype
    bias = bias.to(torch.float32)
    if output_spin is not None:
        output_spin = output_spin.to(torch.float32)
    matmul_spin = matmul_diag_fast_hadamard if fast_hadamard else matmul_diag
    bias = bias.unsqueeze(0)
    bias = matmul_spin(bias, output_spin, block_size=block_size, transpose=True)
    bias = bias.squeeze(0)
    return bias.to(dtype)


class RotationMatrixStore:
    """Store the rotation matrices."""

    def __init__(self, config, model):
        """Initialize the rotation matrix store."""
        self.config = config
        # list of (module, name), module.name is the rotation matrix
        self.matrices = []
        self.model = model
        self._num_decoder_layers = getattr(
            model.config, "num_hidden_layers", len(model.model.layers)
        )
        self.init_matrices()

    # def format_matrix_name(self, name):
    #     return f"rotation_{name}"

    def init_matrices(self):
        """Initialize the rotation matrices."""
        for name in self.config:
            per_layer = self.config.get(name).get("per_layer", False)
            if isinstance(per_layer, str):
                for moudle_name, module in _iter_modules_by_pattern(self.model, per_layer):
                    setattr(
                        module,
                        name,
                        get_orthogonal_matrix(
                            self.config.get(name).get("dim"),
                            self.config.get(name).get("mode", "hadamard"),
                            module.device,
                        ),
                    )
                    self.matrices.append((module, name))
            elif per_layer:
                for i in range(self._num_decoder_layers):
                    module = self.model.get_submodule(f"*.layers.{i}")
                    setattr(
                        module,
                        name,
                        get_orthogonal_matrix(
                            self.config.get(name).get("dim"),
                            self.config.get(name).get("mode", "hadamard"),
                            module.device,
                        ),
                    )
                    self.matrices.append((module, name))
            else:
                setattr(
                    self.model,
                    name,
                    get_orthogonal_matrix(
                        self.config.get(name).get("dim"),
                        self.config.get(name).get("mode", "hadamard"),
                        self.model.device,
                    ),
                )
                self.matrices.append((self.model, name))

    def get(self, name, module_name=None):
        """Get the rotation matrix by name.

        Args:
            name: Name of the rotation matrix
            module_name: Name of the module that the rotation matrix will be applied to.

        Returns:
            The rotation matrix
        """
        if module_name is not None:
            module = self.model.get_submodule(module_name)
            while not hasattr(module, name):
                module_name = module_name.rsplit(".", 1)[0]
                module = self.model.get_submodule(module_name)
            return getattr(module, name)

        return getattr(self.model, name)

    def clear(self):
        """Clear the rotation matrices."""
        for module, name in self.matrices:
            delattr(module, name)
        gc.collect()
        torch.cuda.empty_cache()


# def apply_full_rotation(weight: torch.Tensor, r2_full: torch.Tensor) -> torch.Tensor:
#     """Apply full-dimension R2 rotation to weight input dimension.

#     Used for O projection and down projection in QuaRot.
#     Applies W @ R2_full (right multiply on input dimension).

#     Args:
#         weight: Weight tensor [out_features, in_features]
#         r2_full: Full rotation matrix [in_features, in_features]

#     Returns:
#         Rotated weight tensor W @ R2_full
#     """
#     dtype = weight.dtype
#     device = weight.device
#     w = weight.to(torch.float64)
#     r2 = r2_full.to(dtype=torch.float64, device=device)

#     w_rotated = torch.matmul(w, r2)

#     return w_rotated.to(device=device, dtype=dtype)


# def apply_per_head_rotation(
#     weight: torch.Tensor, r2: torch.Tensor, num_heads: int, transpose_first: bool = False
# ) -> torch.Tensor:
#     """Apply R2 rotation per attention head (matching SpinQuant).

#     Args:
#         weight: Weight tensor [out_features, in_features]
#         r2: Rotation matrix [head_dim, head_dim]
#         num_heads: Number of attention heads
#         transpose_first: If True, apply on output dim (V). If False, apply on input dim (O)

#     Returns:
#         Rotated weight tensor
#     """
#     head_dim = r2.shape[0]
#     dtype = weight.dtype
#     device = weight.device
#     w = weight.to(torch.float64)

#     if transpose_first:
#         # V projection: per-head R2 on output dimension (output=True in SpinQuant)
#         w = w.T  # Transpose to make output dim last
#         transposed_shape = w.shape
#         w_reshaped = w.reshape(-1, transposed_shape[-1] // head_dim, head_dim)
#         w_rotated = w_reshaped @ r2.to(dtype=torch.float64, device=device)
#         w = w_rotated.reshape(transposed_shape).T
#     else:
#         # O projection: per-head R2 on input dimension (output=False in SpinQuant)
#         init_shape = w.shape
#         w_reshaped = w.reshape(-1, init_shape[-1] // head_dim, head_dim)
#         w_rotated = w_reshaped @ r2.to(dtype=torch.float64, device=device)
#         w = w_rotated.reshape(init_shape)

#     return w.to(device=device, dtype=dtype)


# The rest of the reference implementation (online transforms, model-specific helpers)
# is intentionally omitted for QuaRot fusable-rotation flow.


# @torch.inference_mode
# def online_rotate(module, inp):
#     x = torch.nn.functional.linear(inp[0], module.Q)
#     return (x,) + inp[1:]


# def register_online_rotation(module, Q: torch.Tensor):
#     assert not hasattr(module, "Q")
#     module.register_buffer("Q", Q.T.to(module.weight.data))  # Note F.linear(x, A) performs x@A.T

#     # We use forward_pre_hook because we capture the input using forward_hook, which could then capture the rotated input.
#     # If we implement in the forward() the un-rotated original input will be captured.
#     module.rotate_handle = module.register_forward_pre_hook(online_rotate)


# class QKRotationWrapper(torch.nn.Module):
#     def __init__(self, func, config, *args, **kwargs):
#         super().__init__()
#         self.config = config
#         num_heads = config.num_attention_heads
#         model_dim = config.hidden_size
#         head_dim = model_dim // num_heads
#         assert is_pow2(head_dim), "Only power of 2 head_dim is supported for K-cache Quantization!"
#         self.func = func
#         self.k_quantizer = quant_utils.ActQuantizer()
#         self.k_bits = 16
#         if kwargs is not None:
#             assert kwargs["k_groupsize"] in [-1, head_dim], (
#                 f"Only token-wise/{head_dim}g quantization is supported for K-cache"
#             )
#             self.k_bits = kwargs["k_bits"]
#             self.k_groupsize = kwargs["k_groupsize"]
#             self.k_sym = kwargs["k_sym"]
#             self.k_clip_ratio = kwargs["k_clip_ratio"]
#             self.k_quantizer.configure(
#                 bits=self.k_bits,
#                 groupsize=-1,  # we put -1 to be toke-wise quantization and handle head-wise quantization by ourself
#                 sym=self.k_sym,
#                 clip_ratio=self.k_clip_ratio,
#             )

#     def forward(self, *args, **kwargs):
#         q, k = self.func(*args, **kwargs)
#         dtype = q.dtype
#         q = hadamard_transform(q.float(), scale=1 / math.sqrt(q.shape[-1])).to(dtype)
#         k = hadamard_transform(k.float(), scale=1 / math.sqrt(k.shape[-1])).to(dtype)
#         (bsz, num_heads, seq_len, head_dim) = k.shape

#         if self.k_groupsize == -1:  # token-wise quantization
#             token_wise_k = k.transpose(1, 2).reshape(-1, self.config.hidden_size)
#             self.k_quantizer.find_params(token_wise_k)
#             k = (
#                 self.k_quantizer(token_wise_k)
#                 .reshape((bsz, seq_len, num_heads, head_dim))
#                 .transpose(1, 2)
#                 .to(q)
#             )
#         else:  # head-wise quantization
#             per_head_k = k.view(-1, head_dim)
#             self.k_quantizer.find_params(per_head_k)
#             k = self.k_quantizer(per_head_k).reshape((bsz, num_heads, seq_len, head_dim)).to(q)

#         self.k_quantizer.free()

#         return q, k


# def add_qk_rotation_wrapper_after_function_call_in_forward(module, function_name, *args, **kwargs):
#     """This function adds a rotation wrapper after the output of a function call in forward.
#     Only calls directly in the forward function are affected. calls by other functions called in forward are not affected.
#     """
#     import functools

#     import monkeypatch

#     attr_name = f"{function_name}_qk_rotation_wrapper"
#     assert not hasattr(module, attr_name)
#     wrapper = monkeypatch.add_wrapper_after_function_call_in_method(
#         module, "forward", function_name, functools.partial(QKRotationWrapper, *args, **kwargs)
#     )
#     setattr(module, attr_name, wrapper)
