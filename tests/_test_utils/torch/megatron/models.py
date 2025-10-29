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
from warnings import warn

import torch
import torch.nn as nn
import torch.nn.functional as F
from _test_utils.import_helper import skip_if_no_megatron
from huggingface_hub import constants as hf_constants

skip_if_no_megatron()

from _test_utils.torch.megatron.utils import initialize_for_megatron
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.models.mamba import MambaModel
from megatron.core.parallel_state import is_pipeline_first_stage, is_pipeline_last_stage
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig

from modelopt.torch.export.unified_export_megatron import import_mcore_gpt_from_hf

try:
    from megatron.core.extensions.transformer_engine import TENorm
    from megatron.core.post_training.modelopt.gpt.model_specs import get_gpt_modelopt_spec

    HAS_TE = True
except ImportError as e:
    warn(f"Transformer Engine not installed: {e}")
    HAS_TE = False

try:
    from megatron.core.post_training.modelopt.mamba.model_specs import get_mamba_stack_modelopt_spec
    from megatron.core.ssm.mamba_layer import MambaLayer  # noqa: F401

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
    workspace_dir: str | None = None,
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

    # Use HF hub cache directory as default workspace_dir
    if workspace_dir is None:
        workspace_dir = hf_constants.HF_HUB_CACHE

    import_mcore_gpt_from_hf(
        model, pretrained_model_path="Qwen/Qwen3-0.6B", workspace_dir=workspace_dir
    )

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
