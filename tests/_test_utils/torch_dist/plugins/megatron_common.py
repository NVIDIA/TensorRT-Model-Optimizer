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
import copy
import re
from collections import defaultdict
from warnings import warn

import torch
import torch.nn as nn
import torch.nn.functional as F
from _test_utils.import_helper import skip_if_no_megatron

skip_if_no_megatron()

from megatron.core import dist_checkpointing
from megatron.core.inference.communication_utils import broadcast_from_last_pipeline_stage
from megatron.core.inference.contexts import StaticInferenceContext
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import (
    InferenceWrapperConfig,
)
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.models.mamba import MambaModel
from megatron.core.parallel_state import (
    get_expert_model_parallel_group,
    get_expert_tensor_parallel_group,
    get_expert_tensor_parallel_rank,
    initialize_model_parallel,
    is_pipeline_first_stage,
    is_pipeline_last_stage,
)
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig

import modelopt.torch.quantization as mtq
from modelopt.torch.export.unified_export_megatron import import_mcore_gpt_from_hf
from modelopt.torch.opt.plugins.mcore_dist_checkpointing import (
    restore_sharded_modelopt_state,
    save_sharded_modelopt_state,
)
from modelopt.torch.utils import to_empty_if_meta_device

try:
    from megatron.core.extensions.transformer_engine import TENorm
    from megatron.core.post_training.modelopt.gpt.model_specs import get_gpt_modelopt_spec

    HAS_TE = True
except ImportError as e:
    warn(f"Transformer Engine not installed: {e}")
    HAS_TE = False

try:
    from megatron.core.post_training.modelopt.mamba.model_specs import get_mamba_stack_modelopt_spec
    from megatron.core.ssm.mamba_layer import MambaLayer

    HAS_MAMBA = True
except ImportError as e:
    warn(f"Mamba not installed: {e}")
    HAS_MAMBA = False

try:
    import apex  # noqa: F401

    HAS_APEX = True
except ImportError as e:
    warn(f"Apex not installed: {e}")
    HAS_APEX = False


class MegatronModel(MegatronModule):
    def __init__(
        self, tp_size: int = 1, cp_size: int = 1, use_te_norm: bool = False, tp_group=None
    ):
        config = TransformerConfig(
            tensor_model_parallel_size=tp_size,
            context_parallel_size=cp_size,
            pipeline_model_parallel_size=1,
            normalization="LayerNorm",
            # Unused parameters below are set to avoid ZeroDivisionError in __post_init__
            num_layers=1,
            hidden_size=tp_size,
            num_attention_heads=tp_size,
        )
        super().__init__(config)
        self.fc1 = ColumnParallelLinear(
            32,
            64,
            config=config,
            init_method=torch.nn.init.xavier_uniform_,
            bias=True,
            gather_output=False,
            skip_bias_add=True,
            is_expert=False,
            tp_group=tp_group,
        )
        self.activation = nn.ReLU()
        if use_te_norm:
            assert HAS_TE
            self.norm = TENorm(config, 64)
        self.fc2 = RowParallelLinear(
            64,
            32,
            config=config,
            init_method=torch.nn.init.xavier_uniform_,
            bias=True,
            skip_bias_add=True,
            input_is_parallel=True,
            is_expert=False,
            tp_group=tp_group,
        )

    def forward(self, x):
        for block in self.children():
            x = block(x)
            if isinstance(x, tuple):
                x = x[0]
        return x

    def get_dummy_input(self, seed: int | None = None) -> torch.Tensor:
        if seed is not None:
            gen = torch.Generator()
            gen.manual_seed(seed)
            return torch.randn(1, 4, 32, generator=gen)
        return torch.randn(1, 4, 32)


def get_mcore_gpt_model(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
    expert_tensor_parallel_size: int | None = None,
    initialize_megatron: bool = False,
    *,
    num_layers: int = 2,
    num_layers_in_first_pipeline_stage: int | None = None,
    num_layers_in_last_pipeline_stage: int | None = None,
    hidden_size: int = 64,
    num_attention_heads: int = 8,
    num_query_groups: int | None = None,
    ffn_hidden_size: int | None = 128,
    max_sequence_length: int = 16,
    vocab_size: int = 64,
    activation_func: str = "swiglu",
    normalization: str = "LayerNorm",
    transformer_impl: str = "modelopt" if HAS_TE else "local",
    use_cpu_initialization: bool = False,
    num_moe_experts: int | None = None,
    moe_grouped_gemm: bool = False,
    bf16: bool = True,
    use_te: bool = False,
) -> GPTModel:
    assert activation_func in ["swiglu", "squared_relu"]
    assert normalization in ["LayerNorm", "RMSNorm"]
    assert transformer_impl in ["local", "transformer_engine", "modelopt"]
    print(f"Using `{transformer_impl=}` model spec for building GPT Model.")

    if initialize_megatron:
        initialize_for_megatron(
            tensor_model_parallel_size,
            pipeline_model_parallel_size,
            expert_model_parallel_size=expert_model_parallel_size,
            expert_tensor_parallel_size=expert_tensor_parallel_size,
        )

    def squared_relu(x):
        return torch.pow(F.relu(x), 2)

    config = TransformerConfig(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        expert_model_parallel_size=expert_model_parallel_size,
        expert_tensor_parallel_size=expert_tensor_parallel_size,
        sequence_parallel=False,
        moe_grouped_gemm=moe_grouped_gemm,
        num_layers=num_layers,
        num_layers_in_first_pipeline_stage=num_layers_in_first_pipeline_stage,
        num_layers_in_last_pipeline_stage=num_layers_in_last_pipeline_stage,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_query_groups=num_query_groups,
        ffn_hidden_size=ffn_hidden_size,
        num_moe_experts=num_moe_experts,
        activation_func=squared_relu if activation_func == "squared_relu" else F.silu,
        normalization=normalization,
        gated_linear_unit=(activation_func == "swiglu"),
        add_bias_linear=False,
        use_cpu_initialization=use_cpu_initialization,
        pipeline_dtype=torch.bfloat16 if bf16 else torch.float32,
        bf16=bf16,
    )

    if transformer_impl == "local":
        assert HAS_APEX, "Apex not installed"
        transformer_layer_spec = get_gpt_layer_local_spec(
            num_experts=num_moe_experts,
            normalization=normalization,
            moe_grouped_gemm=moe_grouped_gemm,
            # TODO: uncomment this when TEGroupedMLP is enabled in Megatron-LM
            # use_te=use_te,
        )
    else:
        assert HAS_TE, "Transformer Engine not installed"
        transformer_layer_spec = (
            get_gpt_modelopt_spec(
                config,
                remap_te_layernorm=True,
                # TODO: uncomment this when TEGroupedMLP is enabled in Megatron-LM
                # moe_grouped_gemm=moe_grouped_gemm
            )
            if transformer_impl == "modelopt"
            else get_gpt_layer_with_transformer_engine_spec()
        )

    model = GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=vocab_size,
        max_sequence_length=max_sequence_length,
        pre_process=is_pipeline_first_stage(),
        post_process=is_pipeline_last_stage(),
        share_embeddings_and_output_weights=False,
        position_embedding_type="rope",
    )

    if bf16:
        model = model.to(torch.bfloat16)

    return model


def get_mcore_qwen3_600m(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
) -> GPTModel:
    config = TransformerConfig(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        sequence_parallel=False,
        num_layers=28,
        hidden_size=1024,
        num_attention_heads=16,
        num_query_groups=8,
        kv_channels=128,
        ffn_hidden_size=3072,
        activation_func=F.silu,
        normalization="RMSNorm",
        qk_layernorm=True,
        gated_linear_unit=True,
        add_bias_linear=False,
        use_cpu_initialization=False,
        pipeline_dtype=torch.bfloat16,
        bf16=True,
    )

    transformer_layer_spec = get_gpt_modelopt_spec(config, remap_te_layernorm=True)

    model = GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=151936,
        max_sequence_length=2048,
        pre_process=is_pipeline_first_stage(),
        post_process=is_pipeline_last_stage(),
        share_embeddings_and_output_weights=True,
        position_embedding_type="rope",
    )

    model = model.to(torch.bfloat16)

    import_mcore_gpt_from_hf(model, pretrained_model_path="Qwen/Qwen3-0.6B")

    return model


def get_mcore_mamba_model(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    initialize_megatron: bool = False,
    *,
    num_layers: int = 3,
    num_layers_in_first_pipeline_stage: int | None = None,
    num_layers_in_last_pipeline_stage: int | None = None,
    hybrid_override_pattern: str | None = None,
    hidden_size: int = 64,
    num_attention_heads: int = 8,
    num_query_groups: int | None = None,
    ffn_hidden_size: int | None = 128,
    max_sequence_length: int = 4,
    vocab_size: int = 64,
    bf16: bool = True,
    # Mamba-specific parameters
    mamba_state_dim: int = 32,
    mamba_head_dim: int = 16,
    mamba_num_groups: int = 2,
) -> MambaModel:
    assert HAS_MAMBA, "Mamba not installed"

    if initialize_megatron:
        initialize_for_megatron(tensor_model_parallel_size, pipeline_model_parallel_size)

    config = TransformerConfig(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        sequence_parallel=False,
        num_layers=num_layers,
        num_layers_in_first_pipeline_stage=num_layers_in_first_pipeline_stage,
        num_layers_in_last_pipeline_stage=num_layers_in_last_pipeline_stage,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_query_groups=num_query_groups,
        ffn_hidden_size=ffn_hidden_size,
        mamba_state_dim=mamba_state_dim,
        mamba_head_dim=mamba_head_dim,
        mamba_num_groups=mamba_num_groups,
        pipeline_dtype=torch.bfloat16 if bf16 else torch.float32,
        bf16=bf16,
    )

    if hybrid_override_pattern is None:
        # Generate pattern by repeating "M*-" and trimming to match num_layers
        # For num_layers=3, return "M*-" (Mamba -> Attention -> MLP)
        # For num_layers=5, return "M*-M*" (Mamba -> Attention -> MLP -> Mamba -> Attention)
        hybrid_override_pattern = ("M*-" * num_layers)[:num_layers]
    else:
        assert len(hybrid_override_pattern) == num_layers
    print(f"Using `{hybrid_override_pattern=}` for building Mamba Model.")

    model = MambaModel(
        config=config,
        mamba_stack_spec=get_mamba_stack_modelopt_spec(remap_te_layernorm=True),
        vocab_size=vocab_size,
        max_sequence_length=max_sequence_length,
        hybrid_override_pattern=hybrid_override_pattern,
        pre_process=is_pipeline_first_stage(),
        post_process=is_pipeline_last_stage(),
        share_embeddings_and_output_weights=False,
        position_embedding_type="none",
    )
    if bf16:
        model = model.to(torch.bfloat16)
    return model


@torch.no_grad()
def run_mcore_inference(
    model: GPTModel | MambaModel,
    prompt_tokens: torch.Tensor,
    active_hidden_size: int | None = None,
) -> torch.Tensor:
    """Run inference on a wrapped Megatron GPT or Mamba model.

    Args:
        model: Megatron GPT or Mamba model.
        prompt_tokens: Input tokens for inference.
        active_hidden_size: Hidden size to use for inference. If not provided, infer the hidden_size
            NOTE: `model.config.hidden_size` may not be the same as the active hidden size
                for the model since for a NAS search space-converted model, the hidden size
                may be different until the model is exported.
            NOTE: If depth pruned model and some PP have 0 layers, this would not work.
    """
    batch_size = prompt_tokens.shape[0]
    if active_hidden_size is None:
        if HAS_MAMBA and isinstance(model.decoder.layers[0], MambaLayer):
            active_hidden_size = model.decoder.layers[0].mixer.d_model
        elif isinstance(model.decoder.layers[0].self_attention, SelfAttention):
            active_hidden_size = model.decoder.layers[0].self_attention.linear_qkv.input_size
        elif isinstance(model.decoder.layers[0].mlp, MLP):
            active_hidden_size = model.decoder.layers[0].mlp.linear_fc1.input_size
        else:
            raise ValueError(f"Cannot infer hidden size from {type(model.decoder.layers[0])=}")

    inference_wrapper_config = InferenceWrapperConfig(
        hidden_size=active_hidden_size,
        inference_batch_times_seqlen_threshold=batch_size * model.max_sequence_length,
        fp32_residual_connection=False,
        params_dtype=torch.bfloat16 if model.config.bf16 else torch.float32,
        padded_vocab_size=model.vocab_size,
    )
    # Get full sequence output instead of only last token logits
    inference_context = StaticInferenceContext.from_config(inference_wrapper_config)
    inference_context.materialize_only_last_token_logits = False

    wrapped_model = GPTInferenceWrapper(model, inference_wrapper_config, inference_context)
    wrapped_model.prep_model_for_inference()

    inference_input = wrapped_model.prep_inference_input(prompt_tokens)
    inference_input = wrapped_model.get_batch_for_context_window(
        inference_input, 0, model.max_sequence_length
    )

    # Note: This is returned in all TP ranks or last PP stage in PP models
    logits = wrapped_model.run_one_forward_step(inference_input)
    logits = broadcast_from_last_pipeline_stage(
        [batch_size, model.max_sequence_length, model.vocab_size],
        dtype=torch.bfloat16 if model.config.bf16 else torch.float32,
        tensor=logits,
    )
    return logits  # shape: (batch_size, max_sequence_length, vocab_size)


def run_mcore_inference_with_dummy_input(
    model: GPTModel | MambaModel, batch_size: int = 2, hidden_size: int | None = None
) -> torch.Tensor:
    """Run inference on a Megatron GPT or Mamba model with random dummy input."""
    prompt_tokens = torch.randint(
        0, model.vocab_size, (batch_size, model.max_sequence_length)
    ).cuda()
    return run_mcore_inference(model, prompt_tokens, hidden_size)


def initialize_for_megatron(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    seed=1234,
    context_parallel_size=1,
    expert_model_parallel_size=1,
    expert_tensor_parallel_size=None,
):
    """Initialize Megatron model parallelism.

    NOTE: If used in a non-spawned process, make sure to call `megatron.core.parallel_state.destroy_model_parallel()`.
    """
    initialize_model_parallel(
        tensor_model_parallel_size,
        pipeline_model_parallel_size,
        context_parallel_size=context_parallel_size,
        expert_tensor_parallel_size=expert_tensor_parallel_size,
        expert_model_parallel_size=expert_model_parallel_size,
    )
    model_parallel_cuda_manual_seed(seed)


def save_distributed_checkpoint(checkpoint_path, gpt_model):
    sharded_state_dict = gpt_model.sharded_state_dict(prefix="")
    dist_checkpointing.save(sharded_state_dict=sharded_state_dict, checkpoint_dir=checkpoint_path)


def load_distributed_checkpoint(checkpoint_path, gpt_model):
    sharded_state_dict = gpt_model.sharded_state_dict(prefix="")
    checkpoint = dist_checkpointing.load(
        sharded_state_dict=sharded_state_dict, checkpoint_dir=checkpoint_path
    )
    gpt_model.load_state_dict(checkpoint)
    return gpt_model


def sharded_state_dict_test_helper(
    tmp_path, model_ref, model_test, forward_fn, meta_device=False, version=None
):
    logits_ref = forward_fn(model_ref)
    state_dict = copy.deepcopy(model_ref.state_dict())

    # Save Megatron-Core checkpoint and modelopt_state with `torch-dist` format.
    save_distributed_checkpoint(tmp_path, model_ref)
    save_sharded_modelopt_state([model_ref], tmp_path)

    # Restore model_test from `torch-dist`.
    restore_sharded_modelopt_state([model_test], tmp_path)
    if meta_device:
        to_empty_if_meta_device(model_test, device="cuda")
    model_test = load_distributed_checkpoint(tmp_path, model_test)

    state_dict_test = model_test.state_dict()
    assert state_dict.keys() == state_dict_test.keys(), (
        f"{set(state_dict.keys()) - set(state_dict_test.keys())}"
    )

    def convert_maybe_fp8(v):
        if v.dtype == torch.float8_e4m3fn:
            return v.to(torch.float16)
        return v

    for k, v in state_dict.items():
        # sharded_state_dict will omit output_layer since we are lacking support on vocab padding
        # extra_state can be a byte Tensor where the value can change due to different serialized
        # order (serialized from a dict). As a result, we must skip checking extra_state.
        if (
            "_extra_state" in k
            or "output_layer" in k
            or k.endswith("._amax_for_smoothing")
            or not isinstance(v, torch.Tensor)
        ):
            continue
        assert v.dtype == state_dict_test[k].dtype, f"{k} v:{v}, s[k]: {state_dict_test[k]}"
        assert torch.allclose(convert_maybe_fp8(v), convert_maybe_fp8(state_dict_test[k])), (
            f"{k} v:{v}, s[k]: {state_dict_test[k]}"
        )

    logits_test = forward_fn(model_test)

    logits_diff = (logits_test - logits_ref) / logits_ref
    assert torch.allclose(logits_ref, logits_test), (
        f"diff: {logits_diff.max()} ref: {logits_ref}, test: {logits_test}"
    )


def copy_weights_from_grouped_to_non_grouped(te_grouped_moe_model, sequential_moe_model):
    """Copy weights from TEGrouped MoE model to sequential MoE model."""
    te_grouped_state = te_grouped_moe_model.state_dict()
    sequential_state = sequential_moe_model.state_dict()

    # Map grouped weights to sequential weights
    weight_mapping = {}
    sequential_key_template = "decoder.layers.{}.mlp.experts.local_experts.{}.linear_fc{}"
    for key, value in te_grouped_state.items():
        if "experts.linear_fc" in key and any(param in key for param in ("weight", "bias")):
            # Extract expert index from grouped weight name
            # Format: decoder.layers.X.mlp.experts.linear_fcY.weightZ
            parts = key.split(".")
            layer_idx = parts[2]  # X
            fc_idx = parts[5]  # Y (linear_fc1 or linear_fc2)
            param_idx = parts[6]  # weight0 / bias0 / etc.
            match = re.search(r"\d+", param_idx)
            expert_idx = match.group(0) if match else "0"  # Z for expert index
            # Map to sequential format: decoder.layers.X.mlp.experts.local_experts.Y.linear_fcZ
            sequential_key = sequential_key_template.format(layer_idx, expert_idx, fc_idx[-1])
            param_name = "weight" if "weight" in param_idx else "bias"
            weight_mapping[f"{sequential_key}.{param_name}"] = value
        elif isinstance(value, torch.Tensor):
            weight_mapping[key] = value

    # Copy weights to sequential model
    for sequential_key in sequential_state:
        if sequential_key in weight_mapping:
            sequential_state[sequential_key] = weight_mapping[sequential_key].clone()

    sequential_moe_model.load_state_dict(sequential_state)


def compare_amax_sync_across_expert_parallel(model, compare_across_experts=True):
    """
    Test if amax values are synchronized across expert parallel groups.

    Returns True if synchronized, False otherwise.
    """

    ep_group = get_expert_model_parallel_group(check_initialized=False)
    etp_group = get_expert_tensor_parallel_group(check_initialized=False)

    # Check if we have either expert model parallel or expert tensor parallel
    has_expert_parallel = (ep_group is not None and ep_group.size() > 1) or (
        etp_group is not None and etp_group.size() > 1
    )

    assert has_expert_parallel, "No expert parallelism detected"
    # Collect amax values from expert quantizers only
    expert_amax_values = {}
    for name, module in model.named_modules():
        if isinstance(module, mtq.nn.TensorQuantizer) and hasattr(module, "_amax"):
            # Check for both TEGrouped and sequential MoE patterns
            if "local_experts" in name or ("experts" in name and "linear_fc" in name):
                # Convert to scalar only if tensor has a single element
                expert_amax_values[name] = module.amax.detach().clone().cpu()

    # Early return if no expert quantizers found
    assert expert_amax_values, "No expert quantizers found"

    # Gather amax values from all ranks
    world_size = torch.distributed.get_world_size()
    all_amax_values = [None] * world_size
    torch.distributed.all_gather_object(all_amax_values, expert_amax_values)

    # Group quantizers by type (ignoring specific expert indices) and check sync
    expert_quantizers = defaultdict(dict)
    for rank_idx, rank_amax in enumerate(all_amax_values):
        for name, amax_val in rank_amax.items():
            # Create quantizer type key by normalizing the name
            quantizer_type = (
                re.sub(r"local_experts\.\d+", "local_experts.*", name)
                if "local_experts" in name
                else name
            )

            if (
                quantizer_type in expert_quantizers
                and rank_idx in expert_quantizers[quantizer_type]
            ):
                if compare_across_experts:
                    # compare expert value across expert for sequential MoE
                    prev_val = expert_quantizers[quantizer_type][rank_idx]
                    # Handle both scalar and tensor comparisons
                    if isinstance(amax_val, torch.Tensor) and isinstance(prev_val, torch.Tensor):
                        are_equal = torch.allclose(prev_val, amax_val, rtol=1e-6, atol=1e-6)
                    else:
                        are_equal = prev_val == amax_val
                    assert are_equal, (
                        f"{rank_idx}, {quantizer_type}, expert_quantizers[quantizer_type][rank_idx]: "
                        f"{expert_quantizers[quantizer_type][rank_idx]}, amax_val: {amax_val}"
                    )
            expert_quantizers[quantizer_type][rank_idx] = amax_val

    rank_info = {
        "global_rank": torch.distributed.get_rank(),
        "etp_rank": get_expert_tensor_parallel_rank(),
    }

    all_rank_info = [None] * world_size
    torch.distributed.all_gather_object(all_rank_info, rank_info)

    # Group ranks by ETP rank for fc1 (ColumnParallel: same output channels should match)
    etp_groups = defaultdict(list)
    for info in all_rank_info:
        etp_groups[info["etp_rank"] if info["etp_rank"] else 0].append(info["global_rank"])

    for quantizer_type, rank_values in expert_quantizers.items():
        # Determine which ranks should have same amax
        # Find which rank should have same amax
        #
        # fc1: ColumnParallel: X @ [A_1, A_2] (weights split along Cout)
        # so amax should be the same across same ETP rank
        # if EP is 2, ETP is 2, we have 4 ranks, EP1, ETP1: 0, EP1, ETP2: 1, EP2, ETP1: 2, EP2, ETP2: 3
        # so we need to compare amax across same ETP rank [0, 2] [1, 3] for per-channel quantization
        #
        # fc2: RowParallel:    [X_1, X_2] @  [A_1
        #                                     A_2] (weights split along Cin)
        # amax should be the same across all ranks
        rank_groups = (
            list(etp_groups.values())
            if "linear_fc1" in quantizer_type and (next(iter(rank_values.values()))).ndim > 0
            else [list(range(world_size))]
        )

        # Check each group independently
        for group in rank_groups:
            group_values = [rank_values[r] for r in group if r in rank_values]
            if len(group_values) > 1:
                # All values in this group should be identical
                first_val = group_values[0]
                for val in group_values[1:]:
                    if isinstance(first_val, torch.Tensor):
                        if not torch.allclose(first_val, val, rtol=1e-6, atol=1e-6):
                            group_rank_values = {
                                r: rank_values[r] for r in group if r in rank_values
                            }
                            return False, f"{quantizer_type} (group {group})", group_rank_values
                    elif abs(first_val - val) > 1e-6:
                        group_rank_values = {r: rank_values[r] for r in group if r in rank_values}
                        return False, f"{quantizer_type} (group {group})", group_rank_values

    return True, None, None
