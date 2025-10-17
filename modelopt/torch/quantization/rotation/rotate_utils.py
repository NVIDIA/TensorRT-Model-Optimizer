import re
import typing

import torch

from .hadamard_utils import random_hadamard_matrix


def fuse_ln_linear(
    layernorm: torch.nn.Module, linear_layers: typing.Iterable[torch.nn.Linear]
) -> None:
    """Fuse the linear operations in Layernorm into the adjacent linear blocks."""
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype

        # Calculating new weight and bias
        W_ = linear.weight.data.double()
        linear.weight.data = (W_ * layernorm.weight.double()).to(linear_dtype)

        if hasattr(layernorm, "bias"):
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(
                    torch.zeros(linear.out_features, dtype=torch.float64)
                )
            linear.bias.data = linear.bias.data.double() + torch.matmul(W_, layernorm.bias.double())
            linear.bias.data = linear.bias.data.to(linear_dtype)
            torch.nn.init.zeros_(linear.bias)
    torch.nn.init.ones_(layernorm.weight)


@torch.no_grad()
def fuse_layernorms(model: torch.nn.Module, norm_fuse_config) -> None:
    """Fuse LayerNorm scale/bias into neighboring Linear layers as configured.

    norm_fuse_config must provide:
      - decoder_layer_fuse(): Iterable[tuple[str, list[str]]] where the strings are
        module names relative to each decoder block module (e.g., 'input_layernorm',
        ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj']).
      - lm_head_fuse: Iterable[tuple[str, str]] where the strings are absolute
        module paths on the model (e.g., 'model.norm', 'lm_head').
    """
    # Embedding fusion
    emb_module = model.model.embed_tokens
    emb_weight = emb_module.weight.data.double()
    emb_module.weight.data = (emb_weight - emb_weight.mean(dim=-1, keepdim=True)).to(
        emb_module.weight.dtype
    )

    for name, module in model.named_modules():
        if re.search("layers\\.[0-9]+$", name) is not None:
            for layer_norm, linear_layers in norm_fuse_config.decoder_layer_fuse():
                fuse_ln_linear(
                    module.get_submodule(layer_norm),
                    [module.get_submodule(linear) for linear in linear_layers],
                )
    for layer_norm, linear in getattr(norm_fuse_config, "lm_head_fuse", []):
        fuse_ln_linear(model.get_submodule(layer_norm), [model.get_submodule(linear)])


# def bake_mean_into_linear(linear: torch.nn.Linear) -> None:
#     """This function takes a linear layer and subtracts the means from the
#     weights and biases. This will result in the linear layer performing
#     the mean substitution which is usually done inside layernorm.
#     """
#     linear_dtype = linear.weight.dtype
#     W_ = linear.weight.data.double()
#     linear.weight.data = W_ - W_.mean(dim=-2, keepdim=True)
#     linear.weight.data = linear.weight.data.to(linear_dtype)
#     if linear.bias is not None:
#         b_ = linear.bias.data.double()
#         linear.bias.data = b_ - b_.mean()
#         linear.bias.data = linear.bias.data.to(linear_dtype)


# def fuse_layer_norms(model):
#     model_type = model_utils.get_model_type(model)

#     kwargs = {"model": model, "model_type": model_type}

#     # Embedding fusion
#     for W in model_utils.get_embeddings(**kwargs):
#         W_ = W.weight.data.double()
#         W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(W.weight.data.dtype)

#     layers = model_utils.get_transformer_layers(**kwargs)

#     # Fuse the linear operations in Layernorm into the adjacent linear blocks.
#     for layer in layers:
#         # fuse the input layernorms into the linear layers
#         if model_type == model_utils.LLAMA_MODEL:
#             fuse_ln_linear(layer.post_attention_layernorm, [layer.mlp.up_proj, layer.mlp.gate_proj])
#             fuse_ln_linear(
#                 layer.input_layernorm,
#                 [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj],
#             )
#         elif model_type == model_utils.OPT_MODEL:
#             fuse_ln_linear(
#                 layer.self_attn_layer_norm,
#                 [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj],
#             )
#             fuse_ln_linear(layer.final_layer_norm, [layer.fc1])
#         else:
#             raise ValueError(f"Unknown model type {model_type}")

#         if model_type == model_utils.OPT_MODEL:
#             bake_mean_into_linear(layer.self_attn.out_proj)
#             bake_mean_into_linear(layer.fc2)

#     fuse_ln_linear(
#         model_utils.get_pre_head_layernorm(**kwargs), [model_utils.get_lm_head(**kwargs)]
#     )

#     model_utils.replace_modules(
#         model,
#         transformers.models.llama.modeling_llama.LlamaRMSNorm
#         if model_type == model_utils.LLAMA_MODEL
#         else torch.nn.LayerNorm,
#         lambda _: model_utils.RMSN(model.config.hidden_size),
#         replace_layers=False,
#     )


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
