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
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from _test_utils.import_helper import skip_if_no_megatron
from packaging.version import Version

import modelopt.torch.opt as mto

skip_if_no_megatron()

from megatron.core import __version__ as mcore_version
from megatron.core import dist_checkpointing
from megatron.core.inference.communication_utils import broadcast_from_last_pipeline_stage
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
from megatron.core.parallel_state import (
    initialize_model_parallel,
    is_pipeline_first_stage,
    is_pipeline_last_stage,
)
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig

try:
    from megatron.core.extensions.transformer_engine import TENorm
    from megatron.core.inference.modelopt_support.gpt.model_specs import get_gpt_layer_modelopt_spec

    HAS_TE = True
except ImportError:
    HAS_TE = False

try:
    import apex  # noqa: F401

    HAS_APEX = True
except ImportError:
    HAS_APEX = False


class MegatronModel(MegatronModule):
    def __init__(self, tp_size: int = 1, use_te_norm: bool = False):
        config = TransformerConfig(
            tensor_model_parallel_size=tp_size,
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
        )

    def forward(self, x):
        for block in self.children():
            x = block(x)
            if isinstance(x, tuple):
                x = x[0]
        return x

    def get_dummy_input(self) -> torch.Tensor:
        return torch.randn(1, 4, 32)


def get_mcore_gpt_model(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    *,
    num_layers: int = 2,
    hidden_size: int = 64,
    num_attention_heads: int = 8,
    num_query_groups: Optional[int] = None,
    ffn_hidden_size: Optional[int] = 128,
    max_sequence_length: int = 32,
    vocab_size: int = 64,
    activation_func: str = "swiglu",
    normalization: str = "LayerNorm",
    transformer_impl: str = "modelopt" if HAS_TE else "local",
) -> GPTModel:
    assert activation_func in ["swiglu", "squared_relu"]
    assert normalization in ["LayerNorm", "RMSNorm"]
    assert transformer_impl in ["local", "transformer_engine", "modelopt"]
    print(f"Using `transformer_impl={transformer_impl}` model spec for building GPT Model.")

    def squared_relu(x):
        return torch.pow(F.relu(x), 2)

    config = TransformerConfig(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        sequence_parallel=False,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_query_groups=num_query_groups,
        ffn_hidden_size=ffn_hidden_size,
        activation_func=squared_relu if activation_func == "squared_relu" else F.silu,
        normalization=normalization,
        gated_linear_unit=(activation_func == "swiglu"),
        pipeline_dtype=torch.float32,
        add_bias_linear=False,
    )

    if transformer_impl == "local":
        assert HAS_APEX, "Apex not installed"
        transformer_layer_spec = get_gpt_layer_local_spec()
    else:
        assert HAS_TE, "Transformer Engine not installed"
        transformer_layer_spec = (
            get_gpt_layer_modelopt_spec()
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

    return model


@torch.no_grad()
def run_mcore_gpt_inference(
    model: GPTModel, prompt_tokens: torch.Tensor, active_hidden_size: Optional[int] = None
) -> torch.Tensor:
    """Run inference on a wrapped Megatron GPT model.

    Args:
        model: Megatron GPT model.
        prompt_tokens: Input tokens for inference.
        active_hidden_size: Hidden size to use for inference. If not provided, infer the hidden_size
            from `model.decoder.layers[0].self_attention.linear_qkv.input_size`.
            NOTE: `model.config.hidden_size` may not be the same as the active hidden size
                for the model since for a NAS search space-converted model, the hidden size
                may be different until the model is exported.
            NOTE: If depth pruned model and some PP have 0 layers, this would not work.
    """
    batch_size = prompt_tokens.shape[0]
    active_hidden_size = (
        active_hidden_size or model.decoder.layers[0].self_attention.linear_qkv.input_size
    )
    inference_wrapper_config = InferenceWrapperConfig(
        hidden_size=active_hidden_size,
        inference_batch_times_seqlen_threshold=batch_size * model.max_sequence_length,
        fp32_residual_connection=False,
        params_dtype=torch.float,
        padded_vocab_size=model.vocab_size,
    )
    wrapped_model = GPTInferenceWrapper(model, inference_wrapper_config)
    wrapped_model.prep_model_for_inference(prompt_tokens)
    if Version(mcore_version) >= Version("0.11"):
        inference_input = wrapped_model.prep_inference_input(prompt_tokens)
        inference_input = wrapped_model.get_batch_for_context_window(
            inference_input, 0, model.max_sequence_length
        )
    else:
        inference_input = wrapped_model.get_batch_for_context_window(0, model.max_sequence_length)

    # Note: This is returned in all TP ranks or last PP stage in PP models
    logits = wrapped_model.run_one_forward_step(inference_input)
    logits = broadcast_from_last_pipeline_stage(
        [batch_size, model.max_sequence_length, model.vocab_size],
        dtype=torch.float32,
        tensor=logits,
    )
    return logits  # shape: (batch_size, max_sequence_length, vocab_size)


def run_mcore_gpt_inference_with_dummy_input(
    model: GPTModel, batch_size: int = 2, hidden_size: Optional[int] = None
) -> torch.Tensor:
    """Run inference on a wrapped Megatron GPT model."""
    prompt_tokens = torch.randint(
        0, model.vocab_size, (batch_size, model.max_sequence_length)
    ).cuda()
    return run_mcore_gpt_inference(model, prompt_tokens, hidden_size)


def initialize_for_megatron(
    tensor_model_parallel_size=1, pipeline_model_parallel_size=1, seed=1234
):
    """Initialize Megatron model parallelism.

    NOTE: If used in a non-spawned process, make sure to call `megatron.core.parallel_state.destroy_model_parallel()`.
    """
    initialize_model_parallel(tensor_model_parallel_size, pipeline_model_parallel_size)
    model_parallel_cuda_manual_seed(seed)


def save_distributed_checkpoint(checkpoint_path, gpt_model):
    if Version(mcore_version) <= Version("0.8") and Version(torch.__version__) >= Version("2.4"):
        # Dont add pytest.skip here since this fails if run inside spawned processes
        raise RuntimeError("Megatron Core 0.9+ is required for dist checkpointing with torch < 2.4")
    sharded_state_dict = gpt_model.sharded_state_dict(prefix="")
    dist_checkpointing.save(sharded_state_dict=sharded_state_dict, checkpoint_dir=checkpoint_path)


def load_distributed_checkpoint(checkpoint_path, gpt_model):
    sharded_state_dict = gpt_model.sharded_state_dict(prefix="")
    checkpoint = dist_checkpointing.load(
        sharded_state_dict=sharded_state_dict, checkpoint_dir=checkpoint_path
    )
    gpt_model.load_state_dict(checkpoint)
    return gpt_model


def sharded_state_dict_test_helper(tmp_path, model_ref, model_test, forward_fn):
    logits_ref = forward_fn(model_ref)
    modelopt_state = mto.modelopt_state(model_ref)
    state_dict = copy.deepcopy(model_ref.state_dict())

    save_distributed_checkpoint(tmp_path, model_ref)
    # Test that model_ref has not been modified during distributed save
    modelopt_state_after_distributed_save = mto.modelopt_state(model_ref)
    assert modelopt_state == modelopt_state_after_distributed_save
    state_dict_after_distributed_save = model_ref.state_dict()
    assert state_dict.keys() == state_dict_after_distributed_save.keys()
    for k, v in state_dict.items():
        assert not isinstance(v, torch.Tensor) or torch.allclose(
            v.to(torch.float32), state_dict_after_distributed_save[k].to(torch.float32)
        )
    logits_ref_after_distributed_save = forward_fn(model_ref)
    assert torch.allclose(logits_ref, logits_ref_after_distributed_save)

    model_test = mto.restore_from_modelopt_state(model_test, modelopt_state)
    model_test = load_distributed_checkpoint(tmp_path, model_test)

    modelopt_state_test = mto.modelopt_state(model_test)
    assert modelopt_state == modelopt_state_test

    state_dict_test = model_test.state_dict()
    assert state_dict.keys() == state_dict_test.keys()
    for k, v in state_dict.items():
        # sharded_state_dict will omit output_layer since we are lacking support on vocab padding
        if "output_layer" in k:
            continue
        assert not isinstance(v, torch.Tensor) or torch.allclose(
            v.to(torch.float32), state_dict_test[k].to(torch.float32)
        ), "{} {} {}".format(k, v, state_dict_test[k])

    logits_test = forward_fn(model_test)
    assert torch.allclose(logits_ref, logits_test), "{} {}".format(logits_ref, logits_test)
