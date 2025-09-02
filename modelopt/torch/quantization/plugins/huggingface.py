# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Support quantization for huggingface layers."""

import inspect
import warnings
from contextlib import contextmanager
from functools import partial
from typing import TYPE_CHECKING

import torch

try:
    from torch.distributed.tensor import Shard
except ImportError:
    Shard = None

import torch.nn as nn
import transformers
from transformers.models.t5.modeling_t5 import T5Attention

from modelopt.torch.opt.dynamic import DynamicModule
from modelopt.torch.utils.distributed import ParallelState

from ..algorithms import AutoQuantizeSearcher
from ..conversion import register
from ..nn import QuantInputBase, QuantModule, QuantModuleRegistry, TensorQuantizer
from ..nn.modules.quant_linear import _QuantLinear
from ..utils import replace_function
from .attention import register_attention_for_kv_quant
from .custom import CUSTOM_MODEL_PLUGINS, _ParallelLinear, _QuantFunctionalMixin

if TYPE_CHECKING:
    from types import ModuleType

__all__ = ["register_hf_attentions_on_the_fly"]


class _QuantAttention(QuantModule):
    """Attention class for KV Cache quantization compatible with new_attention_interface in transformers >= 4.48.0."""

    def _setup(self):
        self.q_bmm_quantizer = TensorQuantizer()
        self.k_bmm_quantizer = TensorQuantizer()
        self.v_bmm_quantizer = TensorQuantizer()

    @staticmethod
    def _quantized_attention(
        original_attention_interface, self, query_states, key_states, value_states, *args, **kwargs
    ):
        query_states = self.q_bmm_quantizer(query_states)
        key_states = self.k_bmm_quantizer(key_states)
        value_states = self.v_bmm_quantizer(value_states)
        return original_attention_interface(
            self, query_states, key_states, value_states, *args, **kwargs
        )

    def forward(self, *args, **kwargs):
        """Forward method for KV cache quantization compatible with new_attention_interface in transformers >= 4.48.0.

        The forward method is used to patch the attention interface with _quantized_attention.
        Once output tensors are generated, it restores the original attention interface.
        """

        def _is_eager_attention():
            if self.config._attn_implementation == "eager":
                return True
            return bool(
                self.config._attn_implementation == "sdpa"
                and kwargs.get("output_attentions", False)
            )

        # Get the original transformers module before wrapped in any ModelOpt DynamicModule
        module: ModuleType = inspect.getmodule(self.get_attn_type(self))

        # Preprocessing logic to patch attention interface
        original_attention_interface = (
            module.eager_attention_forward
            if _is_eager_attention()
            else module.ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        )
        patch_fn = partial(self._quantized_attention, original_attention_interface)

        if _is_eager_attention():
            if not hasattr(module, "eager_attention_forward"):
                raise AssertionError(
                    f"Module {module} does not have `eager_attention_forward` to enable KV Cache quantization. "
                    "Please use a different attention implementation such as `sdpa` by setting "
                    "`model.config._attn_implementation = 'sdpa'` before quantization."
                )
            module.eager_attention_forward = patch_fn  # type: ignore[attr-defined]
        else:
            module.ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation] = patch_fn

        try:
            outputs = super().forward(*args, **kwargs)
        finally:
            # Cleanup logic to restore the original attention interface
            if _is_eager_attention():
                module.eager_attention_forward = original_attention_interface  # type: ignore[attr-defined]
            else:
                module.ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation] = (
                    original_attention_interface
                )

        return outputs

    @staticmethod
    def is_compatible_attention(attn):
        # The new_attention_interface is only available in transformers >= 4.48.0
        # In addition, the new attention interface is not available for some models such as T5
        # Hence lets do a crude check here to see if the attention module is using the new_attention_interface
        # This is not foolproof but should work for most cases
        if transformers.__version__ < "4.48.0":
            return False
        module = inspect.getmodule(attn)
        return getattr(module, "ALL_ATTENTION_FUNCTIONS", None) is not None

    @staticmethod
    def get_attn_type(attn_module) -> type:
        # If this is a DynamicModule, it means that the module class has been wrapped by ModelOpt
        # Hence, we need to get the original class by level=0
        return (
            attn_module.get_original_cls_by_level(level=0)
            if isinstance(attn_module, DynamicModule)
            else type(attn_module)
        )


class _T5QuantAttention(QuantModule):
    """Attention class for KV Cache quantization compatible with T5 Model."""

    def _quantized_matmul(self, batch1, batch2):
        # T5Attention has two matmul operations, one for the query and key and one for the attention and value.
        # The first matmul is quantized with the q_bmm_quantizer and k_bmm_quantizer. The second matmul is
        # quantized with the v_bmm_quantizer.
        if self.qk_quant_matmul:
            self.qk_quant_matmul = False
            q, k = batch1, batch2
            return torch._matmul(
                self.q_bmm_quantizer(q), self.k_bmm_quantizer(k.transpose(3, 2)).transpose(3, 2)
            )
        else:
            self.qk_quant_matmul = True
            attn, v = batch1, batch2
            return torch._matmul(attn, self.v_bmm_quantizer(v))

    def _setup(self):
        self.q_bmm_quantizer = TensorQuantizer(QuantInputBase.default_quant_desc_input)
        self.k_bmm_quantizer = TensorQuantizer(QuantInputBase.default_quant_desc_input)
        self.v_bmm_quantizer = TensorQuantizer(QuantInputBase.default_quant_desc_input)

    @staticmethod
    def is_compatible_attention(attn):
        return issubclass(attn, T5Attention)

    def forward(self, *args, **kwargs):
        # self.qk_quant_matmul is used to alternate between the two matmul operations for T5Attention
        self.qk_quant_matmul = True
        with replace_function(torch, "matmul", self._quantized_matmul):
            return super().forward(*args, **kwargs)


def register_hf_attentions_on_the_fly(model):
    """Find HF Attention modules in the model and register them for KV Cache quantization.

    This function attempts to find child modules ending with "Attention" in the name.
    If such child modules are not found, or the corresponding class does not contain
    identifiable attention patterns, the function will not register any new modules.
    """
    if not _is_supported_hf_model(model):
        return

    attention_cls = set()
    registered_attn_module = False
    for name, module in model.named_modules():
        # Only register attention classes that are from Huggingface transformers
        if type(module).__name__.endswith("Attention"):
            attention_type = _QuantAttention.get_attn_type(module)
            # Add modules to be registered only if they arent already registered
            if (
                QuantModuleRegistry.get(attention_type) is None
                and attention_type not in attention_cls
            ):
                if _QuantAttention.is_compatible_attention(attention_type):
                    # Lets register the attention class for KV Cache quantization
                    register(attention_type, _QuantAttention)
                    registered_attn_module = True
                    print(
                        f"Registered {attention_type} to {_QuantAttention.__name__} for KV Cache quantization"
                    )
                elif _T5QuantAttention.is_compatible_attention(attention_type):
                    register(attention_type, _T5QuantAttention)
                    registered_attn_module = True
                    print(
                        f"Registered {attention_type} to {_T5QuantAttention.__name__} for KV Cache quantization"
                    )
                else:
                    attention_cls.add(attention_type)
                    print(
                        f"Registered {attention_type} to AST based quantized class for KV Cache quantization"
                    )

    # Check if the attention class has been registered
    # For T5Attention, we want to avoid registering T5LayerCrossAttention and T5LayerSelfAttention.
    # Hence we check if the attention class has been registered.
    if registered_attn_module or not attention_cls:
        return

    # this is the case for models that do not use the new_attention_interface or transformers version < 4.48.0
    # Register the attention class for KV Cache quantization
    success = any(register_attention_for_kv_quant(cls) for cls in attention_cls)
    if not success:
        warnings.warn(
            f"Could not create a quantized attention class for  {attention_cls} from this model. "
            "To enable KV Cache quantization, please create a custom quantized attention class for this model and "
            "register it to ModelOpt using `mtq.register` "
            "(see https://nvidia.github.io/TensorRT-Model-Optimizer/guides/_pytorch_quantization.html#custom-quantized-module-and-quantizer-placement)"
        )


class HFParallelLinear(torch.nn.Linear, DynamicModule):
    supported_hf_tp_plans = []
    shard = None

    def _setup(self):
        assert self.weight.placements == self.shard, (
            f"Received unexpected shard {self.weight.placements} for {self}"
        )
        tp_group = self.weight.device_mesh.get_group()
        self._parallel_state = ParallelState(data_parallel_group=-1, tensor_parallel_group=tp_group)

    @classmethod
    def is_compatible(cls, linear) -> bool:
        if not isinstance(linear, torch.nn.Linear):
            return False
        if not hasattr(linear, "_hf_tp_plan"):
            return False
        return linear._hf_tp_plan in cls.supported_hf_tp_plans

    # This is hack for now, otherwise DMRegistry treats this class same as nn.Linear
    def forward(self, x):
        return super().forward(x)


class HFColumnParallelLinear(HFParallelLinear):
    supported_hf_tp_plans = ["colwise", "colwise_rep"]
    shard = (Shard(0),) if Shard is not None else None


class HFRowParallelLinear(HFParallelLinear):
    supported_hf_tp_plans = ["rowwise", "rowwise_rep"]
    shard = (Shard(1),) if Shard is not None else None


class _QuantHFParallelLinear(_ParallelLinear):
    _functionals_to_replace = [(torch.nn.functional, "linear")]

    def fold_weight(self):
        with self.enable_weight_access_and_writeback():
            super().fold_weight()

    @contextmanager
    def enable_weight_access_and_writeback(self):
        assert self.weight.placements == self.shard, (
            f"Received unexpected shard {self.weight.placements} for {self}"
        )
        weight = self.weight
        # TODO: To support TP + FSDP, we need to redistribute the tensor with replicate instead of shard
        self.weight = nn.Parameter(weight.to_local())
        yield
        self.weight = weight


@QuantModuleRegistry.register({HFColumnParallelLinear: "HFColumnParallelLinear"})
class QuantHFColumnParallelLinear(_QuantHFParallelLinear):
    _is_column_parallel = True


@QuantModuleRegistry.register({HFRowParallelLinear: "HFRowParallelLinear"})
class QuantHFRowParallelLinear(_QuantHFParallelLinear):
    _is_row_parallel = True


def convert_hf_parallel_linears_on_the_fly(model):
    """Convert nn.Linear layers that have been TP sharded by HF.

    Huggingface shards regular nn.Linear layers to rowwise or columnwise tensor-parallel layers dynamically.
    This method converts them to `HFColumnParallelLinear` and `HFRowParallelLinear` so that they
    can be treated as TP sharded layers and not like regular nn.Linear layers.
    """
    for name, module in model.named_modules():
        if HFColumnParallelLinear.is_compatible(module):
            HFColumnParallelLinear.convert(module)
        elif HFRowParallelLinear.is_compatible(module):
            HFRowParallelLinear.convert(module)


if transformers.pytorch_utils.Conv1D not in QuantModuleRegistry:
    # transformers.pytorch_utils.Conv1D used in HF-GPT2 is not a real Conv1D
    # It is actually a Linear layer where weight is transposed and torch.addmm is used
    @QuantModuleRegistry.register({transformers.pytorch_utils.Conv1D: "Conv1D"})
    class _QuantConv1D(_QuantLinear):
        @classmethod
        @torch.no_grad()
        def convert(cls, module: nn.Module) -> "_QuantConv1D":
            module.weight = nn.Parameter(module.weight.T.contiguous())
            module.out_features, module.in_features = module.weight.shape
            # We want the forward method of nn.Linear to be called instead of the forward method of Conv1D
            dyn_cls: QuantModule = QuantModuleRegistry.get(nn.Linear)
            return dyn_cls.convert(module)


class _TransposedQuantization(torch.autograd.Function):
    """Applies transposed quantization.

    This is useful for weight quantization of some MoEs such as gpt-oss or Llama4 which has expert weights
    of shape (num_experts, in_dim, out_dim). Per-channel/Per-block quantization from ModelOpt
    assumes that `in_dim` is -1 dim. Hence for quantizing such MoE weights, lets use transposed quantization.
    """

    # Note: TransposedQuantization uses STE with no clipping
    @staticmethod
    def forward(ctx, inputs, quantizer):
        return quantizer(inputs.transpose(-1, -2).contiguous()).transpose(-1, -2)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


_transposed_quantize = _TransposedQuantization.apply


class _QuantMoeSparseMoe(QuantModule):
    def _setup(self):
        pass

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if any(getattr(m, "_if_calib", False) for m in self.experts.modules()):
            # If any of the experts are in calibration mode, we will forward all tokens to all experts
            # This is used only for calibration, we need to re-calculate the actual outputs again using
            # the original top_k
            original_top_k = self.top_k
            self.top_k = self.num_experts
            super().forward(hidden_states)
            self.top_k = original_top_k
        return super().forward(hidden_states)


class _QuantLlama4TextExperts(QuantModule):
    def _setup(self):
        self.gate_up_proj_input_quantizer = TensorQuantizer()
        self.gate_up_proj_weight_quantizer = TensorQuantizer()
        self.down_proj_input_quantizer = TensorQuantizer()
        self.down_proj_weight_quantizer = TensorQuantizer()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.view(self.num_experts, -1, self.hidden_size)
        gate_up = torch.bmm(
            self.gate_up_proj_input_quantizer(hidden_states),
            _transposed_quantize(self.gate_up_proj, self.gate_up_proj_weight_quantizer),
        )
        gate, up = gate_up.chunk(2, dim=-1)  # not supported for DTensors
        next_states = torch.bmm(
            self.down_proj_input_quantizer(up * self.act_fn(gate)),
            _transposed_quantize(self.down_proj, self.down_proj_weight_quantizer),
        )
        next_states = next_states.view(-1, self.hidden_size)
        return next_states


# For more information on DbrxExpert, see https://github.com/huggingface/transformers/blob/dcdda532/src/transformers/models/dbrx/modeling_dbrx.py#L756
class _QuantDbrxExperts(QuantModule):
    def _setup(self):
        """Modify the DbrxExpert."""
        # No setup is needed for DbrxExpert, we only need to update DbrxExpertGLU

    # forward method copied from the original dbrx repo - https://github.com/databricks/dbrx/blob/a3200393/model/modeling_dbrx.py#L795
    def forward(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        top_weights: torch.Tensor,
        top_experts: torch.LongTensor,
    ) -> torch.Tensor:
        bsz, q_len, hidden_size = x.shape
        x = x.view(-1, hidden_size)
        out = torch.zeros_like(x)

        expert_mask = nn.functional.one_hot(top_experts, num_classes=self.moe_num_experts).permute(
            2, 1, 0
        )
        for expert_idx in range(self.moe_num_experts):
            topk_idx, token_idx = torch.where(expert_mask[expert_idx])
            if token_idx.shape[0] == 0:
                continue

            token_list = token_idx.tolist()
            topk_list = topk_idx.tolist()

            expert_tokens = x[None, token_list].reshape(-1, hidden_size)
            expert_out = (
                self.mlp(expert_tokens, expert_idx) * top_weights[token_list, topk_list, None]
            )

            out.index_add_(0, token_idx, expert_out)

        out = out.reshape(bsz, q_len, hidden_size)
        return out


class _QuantDbrxExpertGLU(QuantModule):
    def _setup(self):
        """Modify the DbrxExpertGLU by using nn.Linear layers."""
        dtype, device = self.w1.dtype, self.w1.device

        def _copy_weights(modules, weights):
            modules.to(dtype=dtype, device=device)
            for expert_idx, module in enumerate(modules):
                with torch.no_grad():
                    module.weight.copy_(weights[expert_idx].detach())

        self.w1_linear = nn.ModuleList(
            [
                nn.Linear(self.hidden_size, self.ffn_hidden_size, bias=False)
                for _ in range(self.moe_num_experts)
            ]
        )
        _copy_weights(
            self.w1_linear,
            self.w1.view(self.moe_num_experts, self.ffn_hidden_size, self.hidden_size),
        )
        delattr(self, "w1")

        self.v1_linear = nn.ModuleList(
            [
                nn.Linear(self.hidden_size, self.ffn_hidden_size, bias=False)
                for _ in range(self.moe_num_experts)
            ]
        )
        _copy_weights(
            self.v1_linear,
            self.v1.view(self.moe_num_experts, self.ffn_hidden_size, self.hidden_size),
        )
        delattr(self, "v1")

        self.w2_linear = nn.ModuleList(
            [
                nn.Linear(self.ffn_hidden_size, self.hidden_size, bias=False)
                for _ in range(self.moe_num_experts)
            ]
        )
        _copy_weights(
            self.w2_linear,
            self.w2.view(self.moe_num_experts, self.ffn_hidden_size, self.hidden_size).transpose(
                1, 2
            ),
        )
        delattr(self, "w2")

    def forward(self, x: torch.Tensor, expert_idx: int) -> torch.Tensor:
        x1 = self.w1_linear[expert_idx](x)
        x2 = self.v1_linear[expert_idx](x)
        x1 = self.activation_fn(x1)
        x1 = x1 * x2
        return self.w2_linear[expert_idx](x1)


class _QuantDbrxFFN(_QuantMoeSparseMoe):
    @property
    def num_experts(self):
        return self.router.moe_num_experts

    @property
    def top_k(self):
        return self.router.moe_top_k

    @top_k.setter
    def top_k(self, value):
        self.router.moe_top_k = value


try:
    from transformers.models.llama4.modeling_llama4 import Llama4TextExperts, Llama4TextMoe

    if Llama4TextMoe not in QuantModuleRegistry:
        QuantModuleRegistry.register({Llama4TextMoe: "hf.Llama4TextMoe"})(_QuantMoeSparseMoe)

    if Llama4TextExperts not in QuantModuleRegistry:
        QuantModuleRegistry.register({Llama4TextExperts: "hf.Llama4TextExperts"})(
            _QuantLlama4TextExperts
        )
except ImportError:
    pass

try:
    from transformers.models.dbrx.modeling_dbrx import DbrxExpertGLU, DbrxExperts, DbrxFFN

    if DbrxExperts not in QuantModuleRegistry:
        QuantModuleRegistry.register({DbrxExperts: "hf.DbrxExperts"})(_QuantDbrxExperts)

    if DbrxExpertGLU not in QuantModuleRegistry:
        QuantModuleRegistry.register({DbrxExpertGLU: "hf.DbrxExpertGLU"})(_QuantDbrxExpertGLU)

    if DbrxFFN not in QuantModuleRegistry:
        QuantModuleRegistry.register({DbrxFFN: "hf.DbrxFFN"})(_QuantDbrxFFN)
except ImportError:
    pass

try:
    from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

    if MixtralSparseMoeBlock not in QuantModuleRegistry:
        QuantModuleRegistry.register({MixtralSparseMoeBlock: "hf.MixtralSparseMoeBlock"})(
            _QuantMoeSparseMoe
        )
except ImportError:
    pass

try:
    from transformers.models.falcon.modeling_falcon import FalconLinear

    if FalconLinear not in QuantModuleRegistry:
        QuantModuleRegistry.register({FalconLinear: "hf.FalconLinear"})(_QuantLinear)
except ImportError:
    pass

try:
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

    if Qwen3MoeSparseMoeBlock not in QuantModuleRegistry:
        QuantModuleRegistry.register({Qwen3MoeSparseMoeBlock: "hf.Qwen3MoeSparseMoeBlock"})(
            _QuantMoeSparseMoe
        )
except ImportError:
    pass

try:
    from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock

    if Qwen2MoeSparseMoeBlock not in QuantModuleRegistry:
        QuantModuleRegistry.register({Qwen2MoeSparseMoeBlock: "hf.Qwen2MoeSparseMoeBlock"})(
            _QuantMoeSparseMoe
        )
except ImportError:
    pass


class _QuantGptOssExperts(_QuantFunctionalMixin):
    """Quantized wrapper for `transformers.GptOssExperts`.

    Quantizes `gate_up_proj` and `down_proj` weights via dynamic attributes inside `quantize_weight()`.
    Activations into `gate_up_proj` are quantized by `gate_up_proj_input_quantizer`. For `down_proj`
    activation quantization, we intercept `torch.Tensor.__matmul__`/`torch.bmm` and quantize inputs
    on every second call (since the first call computes `gate_up_proj` outputs and second call
    computes `down_proj` outputs).
    """

    @staticmethod
    def _get_quantized_weight(quantizer, module, weight):
        # MoE weight is accessed for each expert in one forward pass. so lets cache it
        if module._enable_weight_quantization:
            if hasattr(quantizer, "_cached_quant_val"):
                return getattr(quantizer, "_cached_quant_val")
            quantizer._cached_quant_val = _transposed_quantize(weight, quantizer)
            return quantizer._cached_quant_val
        return weight

    def _setup_for_weight_quantization(self):
        self._register_dynamic_attribute(
            "gate_up_proj", partial(self._get_quantized_weight, self.gate_up_proj_weight_quantizer)
        )
        self._register_dynamic_attribute(
            "down_proj", partial(self._get_quantized_weight, self.down_proj_weight_quantizer)
        )

    def _setup(self):
        assert not hasattr(self, "kernel_layer_name"), (
            "ModelOpt quantization does not support patched forward for kernel_hub"
        )
        self.gate_up_proj_input_quantizer = TensorQuantizer()
        self.gate_up_proj_weight_quantizer = TensorQuantizer()
        self.down_proj_input_quantizer = TensorQuantizer()
        self.down_proj_weight_quantizer = TensorQuantizer()

        self._register_temp_attribute("_enable_weight_quantization", False)
        self._register_temp_attribute("_down_proj_mul", False)
        self._setup_for_weight_quantization()

    @property
    def functionals_to_replace(self):
        def _quantized_bmm(batch1, batch2):
            batch1 = self.down_proj_input_quantizer(batch1) if self._down_proj_mul else batch1
            self._down_proj_mul = not self._down_proj_mul  # toggle the flag
            return torch._bmm(batch1, batch2)

        def _tensor_matmul(self_t, other):
            self_t = self.down_proj_input_quantizer(self_t) if self._down_proj_mul else self_t
            self._down_proj_mul = not self._down_proj_mul
            return torch.matmul(self_t, other)

        return [
            (torch, "bmm", _quantized_bmm),
            (torch.Tensor, "__matmul__", _tensor_matmul),
        ]

    @contextmanager
    def quantize_weight(self):
        """Context in which MoE weight is quantized."""
        self._enable_weight_quantization = True
        try:
            yield
        finally:
            for module in self.modules():
                if isinstance(module, TensorQuantizer) and hasattr(module, "_cached_quant_val"):
                    delattr(module, "_cached_quant_val")
        self._enable_weight_quantization = False

    def forward(
        self, hidden_states: torch.Tensor, router_indices=None, routing_weights=None
    ) -> torch.Tensor:
        """Forward method to add quantization."""
        hidden_states = self.gate_up_proj_input_quantizer(hidden_states)
        with self.quantize_weight():
            return super().forward(hidden_states, router_indices, routing_weights)


try:
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts

    if GptOssExperts not in QuantModuleRegistry:
        QuantModuleRegistry.register({GptOssExperts: "hf.GptOssExperts"})(_QuantGptOssExperts)
except ImportError:
    pass


def register_dbrx_moe_on_the_fly(model):
    """Register DBRX MoE modules as QUANT_MODULE.

    The MoE class in DBRX is `transformers_modules.modeling_dbrx.DbrxExpertGLU`, which loads dynamically.
    """
    if type(model).__name__ in ["DbrxForCausalLM"]:
        moe_type = type(model.transformer.blocks[0].ffn.experts.mlp)
        # Create a QuantDbrxExpertGLU class on the fly
        if QuantModuleRegistry.get(moe_type) is None:
            QuantModuleRegistry.register({moe_type: moe_type.__name__})(_QuantDbrxExpertGLU)


def register_falcon_linears_on_the_fly(model):
    """Register Falcon linear modules as a QUANT_MODULE.

    Certain falcon models (for example, falcon 40b) use remote code, which are loaded dynamically, to build their model.
    Therefore, we need to register the linear on the fly before quantization.
    """
    if type(model).__name__ in ["RWForCausalLM", "FalconForCausalLM"]:
        linear_type = type(model.transformer.h[0].self_attention.dense)
        # Create a QuantFalconLinear class on the fly
        if QuantModuleRegistry.get(linear_type) is None:
            QuantModuleRegistry.register({linear_type: linear_type.__name__})(_QuantLinear)


def _is_supported_hf_model(model):
    """Check if the model a valid model for transformers quantization specific support."""
    supported_models = [transformers.PreTrainedModel]
    try:
        from peft import PeftModel

        supported_models.append(PeftModel)
    except ImportError:
        pass
    return isinstance(model, tuple(supported_models))


@contextmanager
def setup_model_for_gradient_checkpointing(model: nn.Module):
    use_cache = None
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        # Disable use_cache explicitly before forward is called
        use_cache = model.config.use_cache
        model.config.use_cache = False

    if not hasattr(model, "gradient_checkpointing_enable") or not (
        hasattr(model, "supports_gradient_checkpointing") and model.supports_gradient_checkpointing
    ):
        warnings.warn(
            "AutoQuantize: Huggingface model without gradient checkpointing support detected. "
            "AutoQuantize will consume more memory."
        )
    else:
        try:
            warnings.warn(
                "AutoQuantize: Huggingface model detected - Enabling gradient checkpointing. "
                "Disable gradient checkpointing after AutoQuantize if this is not desired!"
            )
            model.gradient_checkpointing_enable({"use_reentrant": True})
            model.train()  # Model needs to be in training mode to enable gradient checkpointing
            # Set all dropout layers to eval mode for deterministic auto-quantize scores
            for name, module in model.named_modules():
                if isinstance(model, torch.nn.Dropout):
                    module.eval()
        except Exception as e:
            warnings.warn(
                f"AutoQuantize: Error enabling gradient checkpointing for huggingface model due to: {e}, "
                "AutoQuantize will consume more memory."
            )
    yield
    if use_cache is not None:
        model.config.use_cache = use_cache


AutoQuantizeSearcher.register_gradient_checkpointing_enable_context(
    _is_supported_hf_model, setup_model_for_gradient_checkpointing
)

CUSTOM_MODEL_PLUGINS.update(
    [
        register_falcon_linears_on_the_fly,
        register_dbrx_moe_on_the_fly,
        register_hf_attentions_on_the_fly,
        convert_hf_parallel_linears_on_the_fly,
    ]
)
