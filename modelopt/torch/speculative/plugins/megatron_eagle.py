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

"""Plugin to add EAGLE support for Megatron-Core GPT model."""

import copy
import warnings
from collections import deque

import megatron.core
import torch
from megatron.core import InferenceParams, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
from megatron.core.extensions.transformer_engine import TENorm
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.gpt import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import (
    get_data_parallel_rank,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from megatron.core.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
    gather_from_tensor_model_parallel_region,
    scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.transformer.utils import sharded_state_dict_default
from megatron.core.utils import make_tp_sharded_tensor_for_checkpoint
from packaging.version import Version

from ...opt.plugins.megatron_model_config import Llama31Config8B
from ..eagle.conversion import EagleDMRegistry
from ..eagle.eagle_model import EagleModel
from ..utils import (
    AcceptanceRateValidation,
    Tree,
    TreeNode,
    get_default_attention_mask_and_position_ids,
)

try:
    from megatron.core.post_training.modelopt.gpt.model_specs import get_gpt_modelopt_spec
    from megatron.core.post_training.modelopt.layers import Linear
except ImportError:
    warnings.warn("Fail to import megatron.core.post_training! EAGLE feature will be disable!")


def mcore_version_higher_than(target_version: str):
    """Check if megatron-core is least this version."""
    return Version(megatron.core.__version__) > Version(target_version)


def logits_kld_loss(logits, gt_logits, mapping=None):
    """KL Divergence loss using ground truth logits."""
    gathered_logits = gather_from_tensor_model_parallel_region(logits)
    gathered_gt_logits = gather_from_tensor_model_parallel_region(gt_logits)
    if mapping is not None:
        reverse_mapping = torch.arange(len(mapping)).to(mapping.device) + mapping
        gathered_gt_logits = gathered_gt_logits[:, :, reverse_mapping]
    loss = torch.nn.Softmax(dim=2)(gathered_gt_logits) * torch.nn.LogSoftmax(dim=2)(gathered_logits)
    loss = -torch.sum(loss, 2)
    return loss.transpose(0, 1)


def right_padding(input_ids: torch.Tensor, hidden_states: torch.Tensor = None):
    """Pad zeros to the right so that the padded_input_ids is a multiple of tp."""
    tp = get_tensor_model_parallel_world_size()
    seq_len = input_ids.shape[-1]
    right_padding_len = 0 if seq_len % tp == 0 else (tp - seq_len % tp)

    if right_padding_len > 0:
        right_token_pad = torch.zeros(
            (input_ids.shape[0], right_padding_len),
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        padded_input_ids = torch.cat((input_ids, right_token_pad), dim=-1)
        if hidden_states is not None:
            # If sequence_parallel is used, the input hidden_states is already gathered
            padding_zeros = torch.zeros(
                (right_padding_len, hidden_states.shape[1], hidden_states.shape[2]),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            padded_hidden_states = torch.cat((hidden_states, padding_zeros), dim=0)
    else:
        padded_input_ids = input_ids
        padded_hidden_states = hidden_states

    if hidden_states is not None:
        return padded_input_ids, seq_len, padded_hidden_states
    else:
        return padded_input_ids, seq_len


def set_multi_step_attention_mask(attn_mask, step):
    """Given an original attention_mask, construct a multi-step attention_mask.

    i0 i1 i2 i3 i4 i5 i6 i7  (base input_ids)
    =======================
    h0 h1 h2 h3 h4 h5 h6 h7  (base hidden_states)
    l0 l1 l2 l3 l4 l5 l6 l7  (base labels)


    (1st)         | i1 i2 i3 i4 i5 i6 i7 -- |
    (out)         | h0 h1 h2 h3 h4 h5 h6 h7 |
    =========================================
    f1 l1 | i1 h0 |  x                      |
    f2 l2 | i2 h1 |  x  x                   |
    f3 l3 | i3 h2 |  x  x  x                |
    f4 l4 | i4 h3 |  x  x  x  x             |
    f5 l5 | i5 h4 |  x  x  x  x  x          |
    f6 l6 | i6 h5 |  x  x  x  x  x  x       |
    f7 l7 | i7 h6 |  x  x  x  x  x  x  x    |
    -- -- | -- h7 |  o  o  o  o  o  o  o  o |
    =========================================


    (2nd)         | i1 i2 i3 i4 i5 i6 i7 -- | i1 i2 i3 i4 i5 i6 i7 -- |
    (out)         | h0 h1 h2 h3 h4 h5 h6 h7 | -- F1 F2 F3 F4 F5 F6 F7 |
    ===================================================================
    F1 l1 | i1 h0 |  x                      |                         |
    F2 l2 | i2 h1 |  x  x                   |                         |
    F3 l3 | i3 h2 |  x  x  x                |                         |
    F4 l4 | i4 h3 |  x  x  x  x             |                         |
    F5 l5 | i5 h4 |  x  x  x  x  x          |                         |
    F6 l6 | i6 h5 |  x  x  x  x  x  x       |                         |
    F7 l7 | i7 h6 |  x  x  x  x  x  x  x    |                         |
    -- -- | -- h7 |  o  o  o  o  o  o  o  o |                         |
    ===================================================================
    -- -- | i1 -- |                         |                         |
    G2 l2 | i2 F1 |  x  o                   |     x                   |
    G3 l3 | i3 F2 |  x  x  o                |        x                |
    G4 l4 | i4 F3 |  x  x  x  o             |           x             |
    G5 l5 | i5 F4 |  x  x  x  x  o          |              x          |
    G6 l6 | i6 F5 |  x  x  x  x  x  o       |                 x       |
    G7 l7 | i7 F6 |  x  x  x  x  x  x  o    |                    x    |
    -- -- | -- F7 |                         |                         |
    ===================================================================


    (3rd)         | i1 i2 i3 i4 i5 i6 i7 -- | i1 i2 i3 i4 i5 i6 i7 -- | i1 i2 i3 i4 i5 i6 i7 -- |
    (out)         | h0 h1 h2 h3 h4 h5 h6 h7 | -- F1 F2 F3 F4 F5 F6 F7 | -- -- G2 G3 G4 G5 G6 G7 |
    =============================================================================================
    F1 l1 | i1 h0 |  x                      |                         |                         |
    F2 l2 | i2 h1 |  x  x                   |                         |                         |
    F3 l3 | i3 h2 |  x  x  x                |                         |                         |
    F4 l4 | i4 h3 |  x  x  x  x             |                         |                         |
    F5 l5 | i5 h4 |  x  x  x  x  x          |                         |                         |
    F6 l6 | i6 h5 |  x  x  x  x  x  x       |                         |                         |
    F7 l7 | i7 h6 |  x  x  x  x  x  x  x    |                         |                         |
    -- -- | -- h7 |  o  o  o  o  o  o  o  o |                         |                         |
    =============================================================================================
    -- -- | i1 -- |                         |                         |                         |
    G2 l2 | i2 F1 |  x  o                   |     x                   |                         |
    G3 l3 | i3 F2 |  x  x  o                |        x                |                         |
    G4 l4 | i4 F3 |  x  x  x  o             |           x             |                         |
    G5 l5 | i5 F4 |  x  x  x  x  o          |              x          |                         |
    G6 l6 | i6 F5 |  x  x  x  x  x  o       |                 x       |                         |
    G7 l7 | i7 F6 |  x  x  x  x  x  x  o    |                    x    |                         |
    -- -- | -- F7 |                         |                         |                         |
    =============================================================================================
    -- -- | i1 -- |                         |                         |                         |
    -- -- | i2 -- |                         |                         |                         |
    H3 l3 | i3 G2 |  x  o  o                |     x  o                |        x                |
    H4 l4 | i4 G3 |  x  x  o  o             |        x  o             |           x             |
    H5 l5 | i5 G4 |  x  x  x  o  o          |           x  o          |              x          |
    H6 l6 | i6 G5 |  x  x  x  x  o  o       |              x  o       |                 x       |
    H7 l7 | i7 G6 |  x  x  x  x  x  o  o    |                 x  o    |                    x    |
    -- -- | -- G7 |                         |                         |                         |
    =============================================================================================


    (4th)         | i1 i2 i3 i4 i5 i6 i7 -- | i1 i2 i3 i4 i5 i6 i7 -- | i1 i2 i3 i4 i5 i6 i7 -- | i1 i2 i3 i4 i5 i6 i7 -- |
    (out)         | h0 h1 h2 h3 h4 h5 h6 h7 | -- F1 F2 F3 F4 F5 F6 F7 | -- -- G2 G3 G4 G5 G6 G7 | -- -- -- H3 H4 H5 H6 H7 |
    =======================================================================================================================
    F1 l1 | i1 h0 |  x                      |                         |                         |                         |
    F2 l2 | i2 h1 |  x  x                   |                         |                         |                         |
    F3 l3 | i3 h2 |  x  x  x                |                         |                         |                         |
    F4 l4 | i4 h3 |  x  x  x  x             |                         |                         |                         |
    F5 l5 | i5 h4 |  x  x  x  x  x          |                         |                         |                         |
    F6 l6 | i6 h5 |  x  x  x  x  x  x       |                         |                         |                         |
    F7 l7 | i7 h6 |  x  x  x  x  x  x  x    |                         |                         |                         |
    -- -- | -- h7 |  o  o  o  o  o  o  o  o |                         |                         |                         |
    =======================================================================================================================
    -- -- | i1 -- |                         |                         |                         |                         |
    G2 l2 | i2 F1 |  x  o                   |     x                   |                         |                         |
    G3 l3 | i3 F2 |  x  x  o                |        x                |                         |                         |
    G4 l4 | i4 F3 |  x  x  x  o             |           x             |                         |                         |
    G5 l5 | i5 F4 |  x  x  x  x  o          |              x          |                         |                         |
    G6 l6 | i6 F5 |  x  x  x  x  x  o       |                 x       |                         |                         |
    G7 l7 | i7 F6 |  x  x  x  x  x  x  o    |                    x    |                         |                         |
    -- -- | -- F7 |                         |                         |                         |                         |
    =======================================================================================================================
    -- -- | i1 -- |                         |                         |                         |                         |
    -- -- | i2 -- |                         |                         |                         |                         |
    H3 l3 | i3 G2 |  x  o  o                |     x  o                |        x                |                         |
    H4 l4 | i4 G3 |  x  x  o  o             |        x  o             |           x             |                         |
    H5 l5 | i5 G4 |  x  x  x  o  o          |           x  o          |              x          |                         |
    H6 l6 | i6 G5 |  x  x  x  x  o  o       |              x  o       |                 x       |                         |
    H7 l7 | i7 G6 |  x  x  x  x  x  o  o    |                 x  o    |                    x    |                         |
    -- -- | -- G7 |                         |                         |                         |                         |
    =======================================================================================================================
    -- -- | i1 -- |                         |                         |                         |                         |
    -- -- | i2 -- |                         |                         |                         |                         |
    -- -- | i3 -- |                         |                         |                         |                         |
    K4 l4 | i4 H3 |  x                      |     x                   |        x                |          x              |
    K5 l5 | i5 H4 |  x  x                   |        x                |           x             |             x           |
    K6 l6 | i6 H5 |  x  x  x                |           x             |              x          |                x        |
    K7 l7 | i7 H6 |  x  x  x  x             |              x          |                 x       |                   x     |
    -- -- | -- H7 |                         |                         |                         |                         |
    =======================================================================================================================
    """  # noqa: E501
    assert step > 1, "step should be larger than 1 in multi-step attention mask."
    assert step <= 4, "Currently only a step of 4 or smaller is supported!"

    s = attn_mask.shape[-1]
    zero_mask = torch.ones_like(attn_mask).bool()
    mask_2_1 = attn_mask.clone().detach()
    mask_2_1[:, :, :, :-1] = mask_2_1[:, :, :, 1:]
    mask_2_2 = torch.ones_like(attn_mask).bool()
    for i in range(1, s - 1):
        mask_2_2[:, :, i, i] = False

    if step == 2:
        attn_mask = torch.cat(
            (
                torch.cat((attn_mask, zero_mask), dim=-1),
                torch.cat((mask_2_1, mask_2_2), dim=-1),
            ),
            dim=-2,
        )
        return attn_mask

    mask_3_1 = mask_2_1.clone().detach()
    mask_3_1[:, :, :, :-1] = mask_3_1[:, :, :, 1:]
    mask_3_2 = mask_2_2.clone().detach()
    mask_3_2[:, :, :, :-1] = mask_3_2[:, :, :, 1:]
    mask_3_2[:, :, 1, 0] = True
    mask_3_3 = mask_2_2.clone().detach()
    mask_3_3[:, :, 1, 1] = True

    if step == 3:
        attn_mask = torch.cat(
            (
                torch.cat((attn_mask, zero_mask, zero_mask), dim=-1),
                torch.cat((mask_2_1, mask_2_2, zero_mask), dim=-1),
                torch.cat((mask_3_1, mask_3_2, mask_3_3), dim=-1),
            ),
            dim=-2,
        )
        return attn_mask

    mask_4_1 = mask_3_1.clone().detach()
    mask_4_1[:, :, :, :-1] = mask_4_1[:, :, :, 1:]
    mask_4_2 = mask_3_2.clone().detach()
    mask_4_2[:, :, :, :-1] = mask_4_2[:, :, :, 1:]
    mask_4_2[:, :, 2, 0] = True
    mask_4_3 = mask_3_3.clone().detach()
    mask_4_3[:, :, :, :-1] = mask_4_3[:, :, :, 1:]
    mask_4_3[:, :, 2, 1] = True
    mask_4_4 = mask_3_3.clone().detach()
    mask_4_4[:, :, 2, 2] = True

    attn_mask = torch.cat(
        (
            torch.cat((attn_mask, zero_mask, zero_mask, zero_mask), dim=-1),
            torch.cat((mask_2_1, mask_2_2, zero_mask, zero_mask), dim=-1),
            torch.cat((mask_3_1, mask_3_2, mask_3_3, zero_mask), dim=-1),
            torch.cat((mask_4_1, mask_4_2, mask_4_3, mask_4_4), dim=-1),
        ),
        dim=-2,
    )
    return attn_mask


class EagleLanguageModelEmbedding(LanguageModelEmbedding):
    """Allow last pp stage to also load the embedding."""

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple[tuple[int, int, int]] = (),
        metadata: dict | None = None,
    ) -> ShardedStateDict:
        """Different from the default, we change the state_dict to have 1 replica at pp."""
        state_dict = self.state_dict(prefix="", keep_vars=True)

        weight_prefix = f"{prefix}word_embeddings.weight"
        return {
            weight_prefix: make_tp_sharded_tensor_for_checkpoint(
                tensor=state_dict["word_embeddings.weight"],
                key=weight_prefix,
                allow_shape_mismatch=True,
                prepend_offsets=sharded_offsets,
                # (PP, TP, DP)
                replica_id=(1, 0, get_data_parallel_rank(with_context_parallel=True)),
            )
        }


class EagleTransformerBlock(TransformerBlock):
    """Only store the EAGLE decoder in the last pp stage."""

    def sharded_state_dict(
        self, prefix: str = "", sharded_offsets: tuple = (), metadata: dict | None = None
    ) -> ShardedStateDict:
        """Generate a sharded state dictionary for the transformer block.

        Args:
            prefix (str, optional): Prefix to be added to all keys in the state dict.
                Defaults to an empty string.
            sharded_offsets (tuple, optional): Tuple of sharding offsets.
            metadata (dict, optional): Additional metadata for sharding.
                Can specify if layers are non-homogeneous. Defaults to None.

        Returns:
            ShardedStateDict: A dictionary containing the sharded state of the model.
        """
        assert not sharded_offsets, "Unexpected sharded offsets"

        sharded_state_dict = {}

        layer_prefix = f"{prefix}layers."
        num_layers = self.config.num_layers
        for global_layer_offset, layer in enumerate(self.layers):
            # global_layer_offset = layer.layer_number - 1  # self.layer_number starts at 1
            state_dict_prefix = (
                f"{layer_prefix}{global_layer_offset}."  # module list index in TransformerBlock
            )
            sharded_prefix = layer_prefix
            sharded_pp_offset = [
                (0, global_layer_offset, num_layers)
            ]  # PP sharding offset for ShardedTensors

            layer_sharded_state_dict = layer.sharded_state_dict(
                state_dict_prefix, sharded_pp_offset, metadata
            )
            replace_prefix_for_sharding(layer_sharded_state_dict, state_dict_prefix, sharded_prefix)
            sharded_state_dict.update(layer_sharded_state_dict)

        # Add modules other than self.layers
        for name, module in self.named_children():
            if module is not self.layers:
                sharded_state_dict.update(
                    sharded_state_dict_default(
                        module, f"{prefix}{name}.", sharded_offsets, metadata
                    )
                )

        return sharded_state_dict


class EagleModule(MegatronModule):
    """EagleModule definition.

    EagleModule consists of an FC projection and additional decoder layers.
    """

    def __init__(
        self,
        config,
        rotary_pos_emb: torch.nn.Module,
        num_layers: int,
        use_last_layernorm: bool,
        use_input_layernorm_in_first_layer: bool = True,
        use_mtp_layernorm: bool = False,
        bias: bool = False,
        num_aux_hidden_states: int = 0,
    ):
        """Constructor.

        EagleModule is essentially a GPTModel except that it only exists in
        the last pp stage. As a result, pre_process must be True (otherwise
        the decoder expects the input is from the receive buffer).
        post_process must be True to perform the final_layernorm.

        Args:
            config: MCore transformer config
            num_layers: number of Eagle layers
            rotary_pos_emb: If None, use the default Llama-3.1 rope (GPT-NeoX).
        """
        # Override transformer_config before superclass initialization
        self._num_eagle_layers = num_layers
        self._use_input_layernorm_in_first_layer = use_input_layernorm_in_first_layer
        self._use_mtp_layernorm = use_mtp_layernorm
        self._num_aux_hidden_states = num_aux_hidden_states
        eagle_config = self._get_eagle_transformer_config(config)
        super().__init__(config=eagle_config)

        eagle_transformer_layer_spec = self._get_eagle_transformer_layer_spec(eagle_config)

        if self._num_aux_hidden_states > 0:
            self.enorm = TENorm(
                eagle_config, eagle_config.hidden_size, eagle_config.layernorm_epsilon
            )
            self._embeddings = None
        elif self._use_mtp_layernorm:
            self.enorm = TENorm(
                eagle_config, eagle_config.hidden_size, eagle_config.layernorm_epsilon
            )
            self.hnorm = TENorm(
                eagle_config, eagle_config.hidden_size, eagle_config.layernorm_epsilon
            )

        device = "cpu" if config.use_cpu_initialization else torch.cuda.current_device()

        # EAGLE-3 uses aux_hidden_states (usually >= 3); otherwise EAGLE-1
        fc_input_size_multiplier = (
            self._num_aux_hidden_states if self._num_aux_hidden_states > 0 else 2
        )

        # This linear was previously a ColumnParallelLinear. We changed it to a normal linear
        # since ColumnParallelLinear will have try to gather the input sequence when sequence
        # parallel is used and does not allow gathering the outputs.
        self.fc = Linear(
            eagle_config.hidden_size * fc_input_size_multiplier,
            eagle_config.hidden_size,
            config=eagle_config,
            init_method=(lambda w: None),  # not used
            bias=bias,
        ).to(device)

        self.rotary_pos_emb = rotary_pos_emb

        # Eagle does not use the final_layernorm in decoder.
        self.decoder = EagleTransformerBlock(
            config=eagle_config,
            spec=eagle_transformer_layer_spec,
            post_layer_norm=use_last_layernorm,
            pre_process=True,
            post_process=True,
        ).to(device)

        if self._num_aux_hidden_states > 0:
            layer = self.decoder.layers[0]
            layer.register_forward_hook(self._eagle3_layer_forward_hook)

            self_attention = layer.self_attention
            if not isinstance(self_attention, SelfAttention):
                raise ValueError("EAGLE-3 only support SelfAttention (MHA, GQA).")

            # EAGLE-3's first attention require [input_layernorm_output, aux_hidden_states]
            self_attention.register_forward_pre_hook(self._eagle3_attention_forward_pre_hook)

            # EAGLE-3's first layer reduces hidden_states from 2h to h.
            self_attention.linear_qkv = tensor_parallel.ColumnParallelLinear(
                self_attention.config.hidden_size * 2,
                self_attention.query_projection_size + 2 * self_attention.kv_projection_size,
                config=self_attention.config,
                init_method=self_attention.config.init_method,
                gather_output=False,
                bias=self_attention.config.add_bias_linear or self_attention.config.add_qkv_bias,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name="qkv",
            )

        # Sanity check
        if self.decoder.layers[0].self_attention.attention_type == AttnMaskType.arbitrary:
            raise ValueError("EAGLE-3 must use arbitrary attention mask.")

    def _get_eagle_transformer_config(self, base_model_config):
        eagle_config = copy.deepcopy(base_model_config)
        eagle_config.num_layers = self._num_eagle_layers
        # Unset the PP config.
        eagle_config.pipeline_model_parallel_size = 1
        eagle_config.virtual_pipeline_model_parallel_size = None
        eagle_config.num_layers_in_first_pipeline_stage = None
        eagle_config.num_layers_in_last_pipeline_stage = None
        return eagle_config

    def _get_eagle_transformer_layer_spec(self, eagle_config):
        """Get the TransformerLayer implementation spec.

        IMPORTANT: EagleModule must use arbitrary_attention_mask since we need to
                   manupulate the mask to compute the correct loss. The default
                   causal mask will result in leaking.
        """
        transformer_layer_spec = get_gpt_modelopt_spec(
            eagle_config,
            remap_te_layernorm=True,
            use_arbitrary_attention_mask=True,
        )
        # If heterogenous layers (e.g. DeepSeek), transformer_layer_spec is a
        # TransformerBlockSubmodules instead. We use the last layer_specs.
        if "TransformerBlockSubmodules" in str(type(transformer_layer_spec)):
            eagle_transformer_layer_spec = copy.deepcopy(transformer_layer_spec.layer_specs[-1])
        else:
            eagle_transformer_layer_spec = copy.deepcopy(transformer_layer_spec)

        # Force TransformerLayer in case RealQuantTransformerLayer was used.
        eagle_transformer_layer_spec.module = TransformerLayer

        if not self._use_input_layernorm_in_first_layer:
            eagle_transformer_layer_spec.submodules.input_layernorm = IdentityOp
        return eagle_transformer_layer_spec

    def _eagle3_layer_forward_hook(self, module, input, output) -> None:
        if not isinstance(module, TransformerLayer):
            raise ValueError(
                "_eagle3_layer_forward_hook can only be registered to TransformerLayer"
            )
        hidden_states = (
            output.clone().detach()
            if isinstance(output, torch.Tensor)
            else output[0].clone().detach()
        )
        self._next_hidden_states_input = hidden_states

    def _eagle3_attention_forward_pre_hook(self, module, input_layernorm_output):
        assert isinstance(input_layernorm_output[0], torch.Tensor)
        assert self._embeddings is not None
        embeddings = self._embeddings
        self._embeddings = None
        return (torch.cat((embeddings, input_layernorm_output[0]), dim=-1),)

    def forward(
        self,
        embeddings: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        rotary_pos_emb: torch.Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict | None = None,
    ) -> torch.Tensor:
        """Forward function."""
        # NOTE: Even if sequence_parallel is used, the rotary_seq_len must be in the original
        #       length. Since we get the seq_len from hidden_states.shape[0], we need to
        #       multiply the the tp back.
        rotary_seq_len = hidden_states.shape[0]
        if self.config.sequence_parallel:
            rotary_seq_len *= self.config.tensor_model_parallel_size

        if self._use_mtp_layernorm:
            embeddings = self.enorm(embeddings)
            hidden_states = self.hnorm(hidden_states)

        # EAGLE-1 uses [s, b, h] input but EAGLE-3 uses [s, b, 2h] input
        if self._num_aux_hidden_states == 0:
            # [s, b, 2h]
            decoder_input = torch.cat((embeddings, hidden_states), dim=-1)
            decoder_input = self.fc(decoder_input)[0]
        else:
            # EAGLE-3 forward
            # EAGLE-3 uses self.fc outside eagle_module forward to convert hidden_states from [s, b, 3h]
            self._embeddings = self.enorm(embeddings)
            decoder_input = hidden_states

        if rotary_pos_emb is None:
            rotary_pos_emb = (
                None if self.config.multi_latent_attention else self.rotary_pos_emb(rotary_seq_len)
            )

        self._next_hidden_states_input = None

        decoder_input_list = [decoder_input]
        self.decoder.set_input_tensor(decoder_input_list[0])
        hidden_states = self.decoder(
            hidden_states=decoder_input,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=packed_seq_params,
            **(extra_block_kwargs or {}),
        )

        if self._next_hidden_states_input is None:
            next_hidden_states_input = hidden_states
        else:
            next_hidden_states_input = self._next_hidden_states_input
            self._next_hidden_states_input = None

        return hidden_states, next_hidden_states_input


class EagleLlama3Module(EagleModule):
    """EagleLlama3Module definition.

    EagleLlama3Module is the default subclass which uses Llama3 architecture
    and rotary position embedding.
    """

    def __init__(
        self,
        config,
        num_layers: int,
        use_last_layernorm: bool,
        use_input_layernorm_in_first_layer: bool = True,
        use_mtp_layernorm: bool = False,
        bias: bool = False,
        num_aux_hidden_states: int = 0,
        ffn_hidden_size: int | None = 0,
    ):
        """Constructor."""
        eagle_config = Llama31Config8B(
            # Getting ModelParallelConfig from the base model
            tensor_model_parallel_size=config.tensor_model_parallel_size,
            sequence_parallel=config.sequence_parallel,
            expert_tensor_parallel_size=config.expert_tensor_parallel_size,
            use_cpu_initialization=config.use_cpu_initialization,
            fp16=config.fp16,
            bf16=config.bf16,
            params_dtype=config.params_dtype,
            # Override hidden_size and ffn_hidden_size from the base model
            hidden_size=config.hidden_size,
            ffn_hidden_size=config.ffn_hidden_size,
        )

        # If base model is using MHA/GQA, then use the same config to simply KV-cache impl.
        if config.kv_channels is not None:
            eagle_config.kv_channels = config.kv_channels
        if config.num_attention_heads > 0:
            eagle_config.num_attention_heads = config.num_attention_heads
        else:
            eagle_config.num_attention_heads = eagle_config.hidden_size // eagle_config.kv_channels
        if config.num_query_groups is not None:
            eagle_config.num_query_groups = config.num_query_groups

        # Override ffn_hidden_size if provided to widen the transformer.
        if ffn_hidden_size > 0:
            eagle_config.ffn_hidden_size = ffn_hidden_size

        rotary_pos_emb = RotaryEmbedding(
            kv_channels=eagle_config.kv_channels,
            rotary_percent=1.0,
            rotary_interleaved=False,
            seq_len_interpolation_factor=None,
            rotary_base=500000.0,
            rope_scaling=True,
            rope_scaling_factor=8.0,
            use_cpu_initialization=eagle_config.use_cpu_initialization,
        )

        super().__init__(
            eagle_config,
            rotary_pos_emb,
            num_layers,
            use_last_layernorm,
            use_input_layernorm_in_first_layer=use_input_layernorm_in_first_layer,
            use_mtp_layernorm=use_mtp_layernorm,
            bias=bias,
            num_aux_hidden_states=num_aux_hidden_states,
        )


@EagleDMRegistry.register({GPTModel: "megatron.core.models.gpt.GPTModel"})
class _DynamicEagleGPTModel(EagleModel):
    """A ``megatron.core.models.gpt.GPTModel`` model with dynamic hyperparams."""

    def _set_default_aux_hidden_state_layers(self):
        num_layers = self.config.num_layers
        self.eagle_aux_hidden_state_layer_ids = [1, num_layers // 2 - 1, num_layers - 4]

    def _transformer_layer_forward_hook(self, module, input, output) -> None:
        if not isinstance(module, TransformerLayer):
            raise ValueError(
                "_transformer_layer_forward_hook can only be registered to TransformerLayer"
            )
        if module.layer_number - 1 not in self.eagle_aux_hidden_state_layer_ids:
            return
        hidden_states = (
            output.clone().detach()
            if isinstance(output, torch.Tensor)
            else output[0].clone().detach()
        )
        self._aux_hidden_states.append(hidden_states)

    def _setup(self):
        super()._setup()
        self._register_temp_attribute("eagle_freeze_base_model", True)
        self._register_temp_attribute("calibration_mode", False)

    def modify(
        self,
        eagle_num_layers=0,
        use_input_layernorm_in_first_layer=True,
        use_last_layernorm=True,
        eagle_hidden_state_distillation=False,
        use_aux_hidden_state=False,
        eagle_aux_hidden_state_layer_ids=[],
        eagle_disable_moe=False,
        draft_vocab_size=0,
        use_mtp_layernorm=False,
        parallel_draft_step=1,
        eagle_self_logit_distillation=True,
        eagle_freeze_base_model=True,
        eagle_report_acc=True,
        ffn_hidden_size=0,
    ):
        if self.config.pipeline_model_parallel_size > 1:
            warnings.warn(
                "Pipeline parallelism detected! _DynamicEagleGPTModel only supports "
                "pipeline parallelism during TensorRT-LLM checkpoint export."
            )

        # Since there is a chance that EAGLE3 can have heterogenous layers (1st layer
        # qkv is 2x large than the rest), we enable MCore hetereogeneous checkpoint.
        if hasattr(self.config, "hetereogenous_dist_checkpoint"):
            self.config.hetereogenous_dist_checkpoint = True

        super().modify(
            eagle_num_layers=eagle_num_layers,
            use_input_layernorm_in_first_layer=use_input_layernorm_in_first_layer,
            use_last_layernorm=use_last_layernorm,
            eagle_hidden_state_distillation=eagle_hidden_state_distillation,
            use_aux_hidden_state=use_aux_hidden_state,
            eagle_aux_hidden_state_layer_ids=eagle_aux_hidden_state_layer_ids,
            eagle_disable_moe=eagle_disable_moe,
            draft_vocab_size=draft_vocab_size,
            use_mtp_layernorm=use_mtp_layernorm,
            parallel_draft_step=parallel_draft_step,
        )
        self.eagle_report_acc = eagle_report_acc
        self.eagle_self_logit_distillation = eagle_self_logit_distillation
        self.eagle_freeze_base_model = eagle_freeze_base_model

        # EAGLE-3 auxiluary hidden_states (only work for TP+EP, does not work for PP)
        self._aux_hidden_states = []

        if self.position_embedding_type not in ["rope", "yarn"]:
            raise ValueError("For EAGLE, only RoPE or YaRN embedding are supported")

        if not self.pre_process and self.post_process:
            self.embedding = EagleLanguageModelEmbedding(
                config=self.config,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_sequence_length,
                position_embedding_type=self.position_embedding_type,
            )

        # Register TransformerLayer forward hook to extract aux hidden_states.
        if len(self.eagle_aux_hidden_state_layer_ids) > 0:
            for layer in self.decoder.layers:
                layer.register_forward_hook(self._transformer_layer_forward_hook)

        # Freeze all parameters
        if self.eagle_freeze_base_model:
            for name, param in self.named_parameters():
                param.requires_grad = False

        # Only the last PP stage has the additional projection and decoder layer.
        # This is to simplify the export.
        if self.post_process:
            if self.eagle_disable_moe:
                self.eagle_module = EagleLlama3Module(
                    self.config,
                    self.eagle_num_layers,
                    self.use_last_layernorm,
                    use_input_layernorm_in_first_layer=use_input_layernorm_in_first_layer,
                    use_mtp_layernorm=self.use_mtp_layernorm,
                    num_aux_hidden_states=len(self.eagle_aux_hidden_state_layer_ids),
                    bias=False,
                    ffn_hidden_size=ffn_hidden_size,
                )
            else:
                self.eagle_module = EagleModule(
                    self.config,
                    self.rotary_pos_emb,
                    self.eagle_num_layers,
                    self.use_last_layernorm,
                    use_input_layernorm_in_first_layer=use_input_layernorm_in_first_layer,
                    use_mtp_layernorm=self.use_mtp_layernorm,
                    num_aux_hidden_states=len(self.eagle_aux_hidden_state_layer_ids),
                    bias=False,
                )

            # Eagle loss functions
            self.kld = logits_kld_loss

            if self.draft_vocab_size > 0:
                # Need an extra lm_head for eagle module since vocab size is reduced.
                assert self.draft_vocab_size <= self.vocab_size, (
                    "EAGLE module's vocab size should be <= base model vocab size!"
                )
                assert eagle_self_logit_distillation, (
                    "Only logit distillation is supported when draft_vocab_size > 0!"
                )

                self.eagle_module.register_buffer(
                    "d2t", torch.zeros(self.draft_vocab_size, dtype=torch.int64)
                )
                self.eagle_module.eagle_output_layer = tensor_parallel.ColumnParallelLinear(
                    self.config.hidden_size,
                    self.draft_vocab_size,
                    config=self.output_layer.config,
                    init_method=self.config.init_method,
                    bias=False,
                    skip_bias_add=False,
                    gather_output=False,
                    skip_weight_param_allocation=False,
                )

    def _get_eagle_input_hidden_states(self, hidden_states: torch.Tensor, apply_fc: bool = True):
        """When _aux_hidden_states is not empty, then this is EAGLE-3.

        Args:
            hidden_states: last hidden_states
            apply_fc: whether to apply EAGLE3 fc
        """
        if len(self._aux_hidden_states) == 0:
            return hidden_states

        # [s / TP, b, len(self._aux_hidden_states) * h]
        aux_hidden_states = torch.cat(self._aux_hidden_states, dim=-1)
        self._aux_hidden_states.clear()

        if apply_fc:
            # [s / TP, b, 3h] -> [s / TP, b, h]
            return self.eagle_module.fc(aux_hidden_states)[0]
        else:
            return aux_hidden_states

    def _get_eagle_module_inputs(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        features: torch.Tensor | None = None,
    ):
        """Getting EAGLE module inputs."""
        b = hidden_states.shape[1]
        h = hidden_states.shape[2]

        # [b, 1]
        id_padding = torch.zeros((b, 1), dtype=input_ids.dtype, device=input_ids.device)
        padded_input_ids = torch.cat((input_ids[:, 1:], id_padding), dim=-1)

        rotary_pos_emb = self.eagle_module.rotary_pos_emb(padded_input_ids.shape[-1])

        attn_mask = attention_mask.clone().detach()
        attn_mask[:, :, :-1, :-1] = attention_mask[:, :, 1:, 1:]
        attn_mask[:, :, -1, :] = True
        attn_mask[:, :, :, -1] = True

        eagle_inputs = {}

        if self.parallel_draft_step > 1:
            eagle_inputs["input_ids"] = padded_input_ids
            eagle_inputs["position_ids"] = position_ids
            if rotary_pos_emb is not None:
                eagle_inputs["rotary_pos_emb"] = rotary_pos_emb
            else:
                # [TODO] (yeyu): there will be problem here with MLA
                eagle_inputs["rotary_pos_emb"] = None

            if self.config.sequence_parallel:
                gathered_hidden_states = gather_from_sequence_parallel_region(hidden_states)
            else:
                gathered_hidden_states = hidden_states
            eagle_inputs["hidden_states"] = gathered_hidden_states

            for i in range(self.parallel_draft_step - 1):
                eagle_inputs["input_ids"] = torch.cat(
                    (
                        eagle_inputs["input_ids"],
                        torch.full(
                            padded_input_ids.shape,
                            getattr(self, f"mask_token_{i}"),
                            device=padded_input_ids.device,
                            dtype=padded_input_ids.dtype,
                        ),
                    ),
                    dim=-1,
                )

                eagle_inputs["hidden_states"] = torch.cat(
                    (
                        eagle_inputs["hidden_states"],
                        torch.zeros(
                            (1 + i, b, h), dtype=hidden_states.dtype, device=hidden_states.device
                        ),
                        gathered_hidden_states[: -(1 + i)],
                    ),
                    dim=0,
                )

                eagle_inputs["position_ids"] = torch.cat(
                    (eagle_inputs["position_ids"], position_ids), dim=-1
                )

                if rotary_pos_emb is not None:
                    eagle_inputs["rotary_pos_emb"] = torch.cat(
                        (eagle_inputs["rotary_pos_emb"], rotary_pos_emb), dim=0
                    )

            if self.config.sequence_parallel:
                eagle_inputs["hidden_states"] = scatter_to_sequence_parallel_region(
                    eagle_inputs["hidden_states"]
                )

            eagle_inputs["attention_mask"] = set_multi_step_attention_mask(
                attn_mask, self.parallel_draft_step
            )
        elif features is None:
            eagle_inputs["input_ids"] = padded_input_ids
            eagle_inputs["hidden_states"] = hidden_states
            eagle_inputs["attention_mask"] = attn_mask
            eagle_inputs["position_ids"] = position_ids
            eagle_inputs["rotary_pos_emb"] = rotary_pos_emb
        elif features.shape[0] == hidden_states.shape[0]:
            eagle_inputs["input_ids"] = torch.cat(
                (padded_input_ids, padded_input_ids),
                dim=-1,
            )

            if self.config.sequence_parallel:
                gathered_hidden_states = gather_from_sequence_parallel_region(hidden_states)
                gathered_features = gather_from_sequence_parallel_region(features)
            else:
                gathered_hidden_states = hidden_states
                gathered_features = features
            eagle_inputs["hidden_states"] = torch.cat(
                (
                    gathered_hidden_states,
                    torch.zeros((1, b, h), dtype=hidden_states.dtype, device=hidden_states.device),
                    gathered_features[:-1, :, :],
                ),
                dim=0,
            )
            if self.config.sequence_parallel:
                eagle_inputs["hidden_states"] = scatter_to_sequence_parallel_region(
                    eagle_inputs["hidden_states"]
                )

            eagle_inputs["attention_mask"] = set_multi_step_attention_mask(attn_mask, 2)
            eagle_inputs["position_ids"] = torch.cat((position_ids, position_ids), dim=-1)

            if rotary_pos_emb is not None:
                eagle_inputs["rotary_pos_emb"] = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=0)
            else:
                # [TODO] (yeyu): there will be problem here with MLA
                eagle_inputs["rotary_pos_emb"] = None
        elif features.shape[0] == hidden_states.shape[0] * 2:
            eagle_inputs["input_ids"] = torch.cat(
                (padded_input_ids, padded_input_ids, padded_input_ids),
                dim=-1,
            )

            if self.config.sequence_parallel:
                gathered_hidden_states = gather_from_sequence_parallel_region(hidden_states)
                gathered_features = gather_from_sequence_parallel_region(features)
            else:
                gathered_hidden_states = hidden_states
                gathered_features = features
            eagle_inputs["hidden_states"] = torch.cat(
                (
                    gathered_hidden_states,
                    torch.zeros((1, b, h), dtype=hidden_states.dtype, device=hidden_states.device),
                    gathered_features[:-1, :, :],
                ),
                dim=0,
            )
            if self.config.sequence_parallel:
                eagle_inputs["hidden_states"] = scatter_to_sequence_parallel_region(
                    eagle_inputs["hidden_states"]
                )

            eagle_inputs["attention_mask"] = set_multi_step_attention_mask(attn_mask, 3)
            eagle_inputs["position_ids"] = torch.cat(
                (position_ids, position_ids, position_ids), dim=-1
            )

            if rotary_pos_emb is not None:
                eagle_inputs["rotary_pos_emb"] = torch.cat(
                    (rotary_pos_emb, rotary_pos_emb, rotary_pos_emb),
                    dim=0,
                )
            else:
                # [TODO] (yeyu): there will be problem here with MLA
                eagle_inputs["rotary_pos_emb"] = None
        else:
            eagle_inputs["input_ids"] = torch.cat(
                (padded_input_ids, padded_input_ids, padded_input_ids, padded_input_ids),
                dim=-1,
            )

            if self.config.sequence_parallel:
                gathered_hidden_states = gather_from_sequence_parallel_region(hidden_states)
                gathered_features = gather_from_sequence_parallel_region(features)
            else:
                gathered_hidden_states = hidden_states
                gathered_features = features
            eagle_inputs["hidden_states"] = torch.cat(
                (
                    gathered_hidden_states,
                    torch.zeros((1, b, h), dtype=hidden_states.dtype, device=hidden_states.device),
                    gathered_features[:-1, :, :],
                ),
                dim=0,
            )
            if self.config.sequence_parallel:
                eagle_inputs["hidden_states"] = scatter_to_sequence_parallel_region(
                    eagle_inputs["hidden_states"]
                )

            eagle_inputs["attention_mask"] = set_multi_step_attention_mask(attn_mask, 4)
            eagle_inputs["position_ids"] = torch.cat(
                (position_ids, position_ids, position_ids, position_ids), dim=-1
            )

            if rotary_pos_emb is not None:
                eagle_inputs["rotary_pos_emb"] = torch.cat(
                    (rotary_pos_emb, rotary_pos_emb, rotary_pos_emb, rotary_pos_emb),
                    dim=0,
                )
            else:
                # [TODO] (yeyu): there will be problem here with MLA
                eagle_inputs["rotary_pos_emb"] = None

        eagle_inputs["embedding"] = self.embedding(
            input_ids=eagle_inputs["input_ids"],
            position_ids=eagle_inputs["position_ids"],
        )

        return eagle_inputs

    def _compute_eagle_loss(self, logits, labels, eagle_logits):
        """Compute the total loss for EAGLE.

        logits: [s, b, vocab // TP]
        labels: [b, s]
        eagle_logits: [s, b, vocab // TP]
        """
        # Compute lm loss (classification loss) or KLDivergence
        if self.eagle_self_logit_distillation:
            mapping = self.eagle_module.d2t if self.draft_vocab_size > 0 else None
            token_loss = self.kld(eagle_logits[:-1, :, :], logits[1:, :, :], mapping)
        else:
            token_loss = self.compute_language_model_loss(labels[:, 1:], eagle_logits[:-1, :, :])

        # [b, s - 1]
        return token_loss

    def _base_model_forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input: torch.Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict | None = None,
        return_eagle_inputs: bool = False,
    ):
        # Word and rotary positional embeddings
        if decoder_input is not None:
            pass
        elif self.pre_process:
            decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)
        else:
            # intermediate stage of pipeline
            # decoder will get hidden_states from decoder.input_tensor
            decoder_input = None

        extra_kwargs = {"packed_seq_params": None} if mcore_version_higher_than("0.9.0") else {}

        rotary_pos_emb = None
        yarn_mscale = 1.0
        if self.config.multi_latent_attention:
            # For MLA, rotary_pos_emb is computed per attention.
            rotary_pos_emb = None
        elif self.position_embedding_type == "rope":
            rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                inference_params,
                self.decoder,
                decoder_input,
                self.config,
                **extra_kwargs,
            )
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len)
        elif self.position_embedding_type == "yarn":
            rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                inference_params,
                None,
                decoder_input,
                self.config,
                **extra_kwargs,
            )
            rotary_pos_emb, yarn_mscale = self.rotary_pos_emb(rotary_seq_len)
        else:
            raise ValueError(
                f"Only RoPE or YaRN are supported but got {self.position_embedding_type}"
            )

        # [TODO]: yarn_mscale needs to be passed into TransformerBlock forward when supported.
        # Now the default value for yarn_mscale = 1.0
        hidden_states = self.decoder(
            hidden_states=decoder_input,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=packed_seq_params,
            **(extra_block_kwargs or {}),
        )

        if return_eagle_inputs:
            return hidden_states, decoder_input
        else:
            return hidden_states, None

    def _eagle_forward(
        self,
        eagle_inputs,
        output_weight,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict | None = None,
    ):
        eagle_hidden_states, eagle_hidden_states_pre_final_layernorm = self.eagle_module(
            eagle_inputs["embedding"],
            eagle_inputs["hidden_states"],
            eagle_inputs["attention_mask"],
            eagle_inputs["rotary_pos_emb"],
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            **(extra_block_kwargs or {}),
        )

        if self.draft_vocab_size > 0:
            eagle_logits, _ = self.eagle_module.eagle_output_layer(eagle_hidden_states)
        else:
            eagle_logits, _ = self.output_layer(eagle_hidden_states, weight=output_weight)

        return eagle_hidden_states, eagle_logits, eagle_hidden_states_pre_final_layernorm

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        decoder_input: torch.Tensor = None,
        labels: torch.Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict | None = None,
        return_eagle_inputs: bool = False,
        loss_decay_factor: float = 0.9,
        **kwargs,
    ) -> torch.Tensor:
        if input_ids is not None and (position_ids is None or attention_mask is None):
            attention_mask, position_ids = get_default_attention_mask_and_position_ids(input_ids)

        # When return_eagle_inputs is True, return decoder_input_for_eagle.
        # When LLM, decoder_input_for_eagle is just the text embeddings. However, when VLM
        # decoder_input_for_eagle will also contain projected image/video embeddings.
        hidden_states, decoder_input_for_eagle = self._base_model_forward(
            input_ids,
            position_ids,
            attention_mask,
            decoder_input,
            inference_params,
            packed_seq_params,
            extra_block_kwargs,
            return_eagle_inputs=return_eagle_inputs,
        )

        # Typically, this is only the case when PP > 1.
        if not self.post_process:
            return hidden_states

        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()
        logits_sbh, _ = self.output_layer(hidden_states, weight=output_weight)

        # If EAGLE-3, aux_hidden_states are gathered by the forward_hook
        if return_eagle_inputs:
            eagle_module_input_hidden_states = self._get_eagle_input_hidden_states(
                hidden_states, apply_fc=False
            )
            # In case of VLM, there will be other fields for pixels.
            return {
                "input_ids": input_ids,
                "decoder_input": decoder_input_for_eagle,
                "hidden_states": eagle_module_input_hidden_states,
                "logits": logits_sbh,
            }
        else:
            eagle_module_input_hidden_states = self._get_eagle_input_hidden_states(
                hidden_states, apply_fc=True
            )

        # Either inference or calibration mode, we want to make sure all weights have been exercised.
        # This makes sure all quantized weights have amax calibrated
        if inference_params is None or self.calibration_mode:
            eagle_inputs_0 = self._get_eagle_module_inputs(
                input_ids=input_ids,
                hidden_states=eagle_module_input_hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

            _, eagle_logits_0, eagle_hidden_states_0_pre_norm = self._eagle_forward(
                eagle_inputs_0,
                output_weight,
                inference_params=inference_params,
                packed_seq_params=packed_seq_params,
                **(extra_block_kwargs or {}),
            )

        # If labels are not provided, return the original logits. We only return after
        # all eagle weights have been exercised for quantization calibration purpose.
        if labels is None:
            return logits_sbh.transpose(0, 1).contiguous()

        # If eagle_freeze_base_model is set to True,
        # the base model is frozen .
        loss = self.compute_language_model_loss(labels, logits_sbh)
        loss = 0.0 * loss

        if self.parallel_draft_step > 1:
            for i in range(self.parallel_draft_step):
                eagle_logits = eagle_logits_0[i * labels.shape[1] : (i + 1) * labels.shape[1]]
                loss_ = self._compute_eagle_loss(logits_sbh, labels, eagle_logits)
                loss_ = loss_[:, i:]
                loss[:, i + 1 :] += 1.0 * loss_
            return loss

        loss_0 = self._compute_eagle_loss(logits_sbh, labels, eagle_logits_0)
        loss[:, 1:] += loss_decay_factor * loss_0

        if self.eagle_report_acc and not self.training:
            acc = []
            with torch.no_grad():
                gathered_logits = gather_from_tensor_model_parallel_region(
                    eagle_logits_0[:-1, :, :]
                )
                eagle_top1 = gathered_logits.transpose(0, 1).argmax(dim=-1)
                if self.draft_vocab_size > 0:
                    eagle_top1 += self.eagle_module.d2t[eagle_top1]
                top1_p = torch.eq(labels[:, 1:], eagle_top1).sum() / eagle_top1.numel()
                acc.append(top1_p)

            if get_tensor_model_parallel_rank() == 0:
                print(
                    f"{torch.distributed.get_rank():3}/{torch.distributed.get_world_size():3} EAGLE 1st Top-1: {acc}",
                    flush=True,
                )

        # Second round of EAGLE loss
        eagle_inputs_1 = self._get_eagle_module_inputs(
            input_ids=input_ids,
            hidden_states=eagle_module_input_hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            features=eagle_hidden_states_0_pre_norm,
        )

        _, eagle_logits_2x, eagle_hidden_states_2x_pre_norm = self._eagle_forward(
            eagle_inputs_1,
            output_weight,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            **(extra_block_kwargs or {}),
        )
        eagle_logits_1 = eagle_logits_2x[labels.shape[1] :, :, :]

        loss_1 = self._compute_eagle_loss(logits_sbh, labels, eagle_logits_1)
        # [b, s - 2]
        loss_1 = loss_1[:, 1:]
        loss[:, 2:] += loss_decay_factor**2 * loss_1

        if self.eagle_report_acc and not self.training:
            acc = []
            with torch.no_grad():
                gathered_logits = gather_from_tensor_model_parallel_region(
                    eagle_logits_1[1:-1, :, :]
                )
                eagle_top1 = gathered_logits.transpose(0, 1).argmax(dim=-1)
                if self.draft_vocab_size > 0:
                    eagle_top1 += self.eagle_module.d2t[eagle_top1]
                top1_p = torch.eq(labels[:, 2:], eagle_top1).sum() / eagle_top1.numel()
                acc.append(top1_p)

            if get_tensor_model_parallel_rank() == 0:
                print(
                    f"{torch.distributed.get_rank():3}/{torch.distributed.get_world_size():3} EAGLE 2nd Top-1: {acc}",
                    flush=True,
                )

        # Third EAGLE loss
        eagle_inputs_2 = self._get_eagle_module_inputs(
            input_ids=input_ids,
            hidden_states=eagle_module_input_hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            features=eagle_hidden_states_2x_pre_norm,
        )

        _, eagle_logits_3x, eagle_hidden_states_3x_pre_norm = self._eagle_forward(
            eagle_inputs_2,
            output_weight,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            **(extra_block_kwargs or {}),
        )

        eagle_logits_2 = eagle_logits_3x[-labels.shape[1] :, :, :]

        loss_2 = self._compute_eagle_loss(logits_sbh, labels, eagle_logits_2)
        # [b, s - 3]
        loss_2 = loss_2[:, 2:]
        loss[:, 3:] += loss_decay_factor**3 * loss_2

        if self.eagle_report_acc and not self.training:
            acc = []
            with torch.no_grad():
                gathered_logits = gather_from_tensor_model_parallel_region(
                    eagle_logits_2[2:-1, :, :]
                )
                eagle_top1 = gathered_logits.transpose(0, 1).argmax(dim=-1)
                if self.draft_vocab_size > 0:
                    eagle_top1 += self.eagle_module.d2t[eagle_top1]
                top1_p = torch.eq(labels[:, 3:], eagle_top1).sum() / eagle_top1.numel()
                acc.append(top1_p)

            if get_tensor_model_parallel_rank() == 0:
                print(
                    f"{torch.distributed.get_rank():3}/{torch.distributed.get_world_size():3} EAGLE 3rd Top-1: {acc}",
                    flush=True,
                )

        # Forth EAGLE loss
        eagle_inputs_3 = self._get_eagle_module_inputs(
            input_ids=input_ids,
            hidden_states=eagle_module_input_hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            features=eagle_hidden_states_3x_pre_norm,
        )

        _, eagle_logits_4x, eagle_hidden_states_4x_pre_norm = self._eagle_forward(
            eagle_inputs_3,
            output_weight,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            **(extra_block_kwargs or {}),
        )

        eagle_logits_3 = eagle_logits_4x[-labels.shape[1] :, :, :]

        loss_3 = self._compute_eagle_loss(logits_sbh, labels, eagle_logits_3)
        # [b, s - 4]
        loss_3 = loss_3[:, 3:]
        loss[:, 4:] += loss_decay_factor**4 * loss_3

        if self.eagle_report_acc and not self.training:
            acc = []
            with torch.no_grad():
                gathered_logits = gather_from_tensor_model_parallel_region(
                    eagle_logits_3[3:-1, :, :]
                )
                eagle_top1 = gathered_logits.transpose(0, 1).argmax(dim=-1)
                if self.draft_vocab_size > 0:
                    eagle_top1 += self.eagle_module.d2t[eagle_top1]
                top1_p = torch.eq(labels[:, 4:], eagle_top1).sum() / eagle_top1.numel()
                acc.append(top1_p)

            if get_tensor_model_parallel_rank() == 0:
                print(
                    f"{torch.distributed.get_rank():3}/{torch.distributed.get_world_size():3} EAGLE 4th Top-1: {acc}",
                    flush=True,
                )

        return loss

    def tree_decode(self, input_ids: torch.Tensor, tree: Tree):
        """Tree-based decoding for EAGLE model using a mask-based approach.

        This function implements a tree-based decoding strategy where each path of the tree
        represents potential token sequences. The function uses attention masks to control
        token dependencies and generate multiple candidate sequences in parallel.

        Args:
            input_ids (torch.Tensor): Input token IDs of shape [batch_size, seq_len]
            treepaths (list[list[int]]): List of treepaths to decode

        Returns:
            tuple: (base_token, base_draft_node, draft_tokens)
                - base_token: The next token predicted by the base model
                - base_draft_node: A TreeNode containing the base token prediction with a
                                   hierarchical structure of child nodes, where each child node
                                   represents a draft token generated by EAGLE
                - draft_tokens: all the draft tokens generated by EAGLE
        """
        # Initial setup and base model forward pass
        padded_input_ids, seq_len = right_padding(input_ids)
        attention_mask, position_ids = get_default_attention_mask_and_position_ids(padded_input_ids)

        # Get base model hidden states
        hidden_states, _ = self._base_model_forward(
            padded_input_ids,
            position_ids,
            attention_mask,
        )

        if not self.post_process:
            return hidden_states

        # Generate base token prediction
        output_weight = (
            self.shared_embedding_or_output_weight()
            if self.share_embeddings_and_output_weights
            else None
        )
        logits_sbh, _ = self.output_layer(hidden_states, weight=output_weight)
        logits_sbh = logits_sbh[:seq_len, :, :]

        base_token = (
            gather_from_tensor_model_parallel_region(logits_sbh)[-1:, :, :]
            .argmax(dim=-1)
            .transpose(0, 1)
        )

        # Early return if no steps needed
        if not tree.root.children:
            self._aux_hidden_states.clear()
            return base_token, None, None

        # Prepare for tree decoding
        eagle_ids = torch.cat((input_ids[:, 1:], base_token), dim=-1)
        # EAGLE-3
        # Only the first iteration input_hidden_states are from aux_hidden_state layers
        hidden_states = self._get_eagle_input_hidden_states(hidden_states)

        if self.config.sequence_parallel:
            hidden_states = gather_from_sequence_parallel_region(hidden_states)
        hidden_states = hidden_states[:seq_len, :, :]

        # relative id from [seq_len-1, seq_len] contains draft token position
        # [seq_len, seq_len + num_child_level_1] contains the number of children for level 1 and so on
        relative_ids = torch.tensor(
            [seq_len - 1, *list(tree.num_children.values())],
            device=input_ids.device,
        ).cumsum(dim=0)

        draft_position_ids = torch.arange(relative_ids[-1], device=input_ids.device)
        cur_pos = seq_len - 1
        for idx in range(len(relative_ids) - 1):
            draft_position_ids[relative_ids[idx] : relative_ids[idx + 1]] = cur_pos
            cur_pos += 1

        draft_attention_mask = torch.full(
            (1, 1, relative_ids[-1], relative_ids[-1]), True, device=input_ids.device
        ).triu_(1)
        draft_attention_mask[:, :, :, seq_len:] = True
        draft_attention_mask[:, :, seq_len - 1 :, seq_len - 1 :] = tree.attention_mask

        draft_rotary_pos_emb = self.eagle_module.rotary_pos_emb(seq_len + tree.max_depth)
        draft_rotary_pos_emb = torch.cat(
            [draft_rotary_pos_emb[index : index + 1] for index in draft_position_ids], dim=0
        )

        base_draft_node = TreeNode(base_token)
        queue = deque([(base_draft_node, tree.root)])
        draft_tokens = []
        # Tree decoding loop
        for step in range(tree.max_depth):
            # Prepare inputs for EAGLE forward pass
            padded_eagle_ids, seq_len, padded_hidden_states = right_padding(
                eagle_ids, hidden_states
            )

            if self.config.sequence_parallel:
                padded_hidden_states = scatter_to_sequence_parallel_region(padded_hidden_states)

            eagle_attention_mask, eagle_position_ids = get_default_attention_mask_and_position_ids(
                padded_eagle_ids
            )
            length = eagle_ids.shape[-1]
            eagle_attention_mask[:, :, :length, :length] = draft_attention_mask[
                :, :, :length, :length
            ]
            eagle_attention_mask[:, :, length:, length:] = True
            eagle_position_ids[:length] = draft_position_ids[:length]
            padded_rotary_pos_emb = self.eagle_module.rotary_pos_emb(padded_eagle_ids.shape[-1])
            padded_rotary_pos_emb[:length] = draft_rotary_pos_emb[:length]

            eagle_inputs = {
                "input_ids": padded_eagle_ids,
                "embedding": self.embedding(
                    input_ids=padded_eagle_ids,
                    position_ids=eagle_position_ids,
                ),
                "hidden_states": padded_hidden_states,
                "attention_mask": eagle_attention_mask,
                "rotary_pos_emb": padded_rotary_pos_emb,
            }

            # Forward pass through EAGLE
            _, eagle_logits, eagle_next_hidden_states_input = self._eagle_forward(
                eagle_inputs,
                output_weight,
            )
            # Process EAGLE outputs
            eagle_logits = eagle_logits[:seq_len, :, :]
            if self.config.sequence_parallel:
                eagle_next_hidden_states_input = gather_from_sequence_parallel_region(
                    eagle_next_hidden_states_input
                )
            eagle_next_hidden_states_input = eagle_next_hidden_states_input[:seq_len, :, :]
            # Generate and store top-k tokens for each tree node
            for rel_idx in range(relative_ids[step], relative_ids[step + 1]):
                draft_node, tree_node = queue.popleft()
                n_topk = max(tree_node.children.keys()) + 1 if tree_node.children else 0
                # Get top-k tokens for current position
                new_ids = (
                    gather_from_tensor_model_parallel_region(eagle_logits)[
                        rel_idx : rel_idx + 1, :, :
                    ]
                    .topk(n_topk, dim=-1)[1]
                    .squeeze(0)
                )

                for child_idx, child_node in tree_node.children.items():
                    eagle_ids = torch.cat(
                        (eagle_ids, new_ids[:, child_idx : child_idx + 1]), dim=-1
                    )
                    # value of the node is token id
                    new_draft_node = TreeNode(new_ids[:, child_idx])
                    draft_tokens.append(new_ids[:, child_idx])
                    draft_node.children[child_idx] = new_draft_node
                    queue.append((new_draft_node, child_node))

                # Update hidden states for each branch
                hidden_states = torch.cat(
                    (
                        hidden_states,
                        eagle_next_hidden_states_input[rel_idx : rel_idx + 1].repeat(
                            len(tree_node.children), 1, 1
                        ),
                    ),
                    dim=0,
                )
        draft_tokens = torch.cat(draft_tokens, dim=-1)
        return base_token, base_draft_node, draft_tokens

    def pseudo_speculative_generate(
        self,
        input_ids: torch.Tensor,
        steps: int = 1,
    ):
        """Pseudo generate of the EAGLE GPTModel.

        Returns:
            base_token (torch.Tensor): token from base model
            draft_tokens (torch.Tensor): draft tokens from eagle module
        """
        padded_input_ids, seq_len = right_padding(input_ids)

        attention_mask, position_ids = get_default_attention_mask_and_position_ids(padded_input_ids)

        hidden_states, _ = self._base_model_forward(
            padded_input_ids,
            position_ids,
            attention_mask,
        )

        if not self.post_process:
            return hidden_states

        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()
        logits_sbh, _ = self.output_layer(hidden_states, weight=output_weight)

        # Remove the padding
        logits_sbh = logits_sbh[:seq_len, :, :]

        base_token = (
            gather_from_tensor_model_parallel_region(logits_sbh)[-1:, :, :]
            .argmax(dim=-1)
            .transpose(0, 1)
        )

        # Early return
        if steps < 1:
            self._aux_hidden_states.clear()
            return base_token, None

        eagle_ids = torch.cat((input_ids[:, 1:], base_token), dim=-1)

        # EAGLE-3
        # Only the first iteration input_hidden_states are from aux_hidden_state layers
        hidden_states = self._get_eagle_input_hidden_states(hidden_states)
        # Remove the padding
        if self.config.sequence_parallel:
            hidden_states = gather_from_sequence_parallel_region(hidden_states)
        hidden_states = hidden_states[:seq_len, :, :]

        draft_tokens = []
        for _ in range(steps):
            if self.parallel_draft_step > 1:
                for i in range(self.parallel_draft_step - 1):
                    eagle_ids = torch.cat(
                        (eagle_ids, getattr(self, f"mask_token_{i}").view((1, 1))), dim=-1
                    )
                    hidden_states = torch.cat((hidden_states, hidden_states[-1:]), dim=0)
            padded_eagle_ids, seq_len, padded_hidden_states = right_padding(
                eagle_ids, hidden_states
            )
            if self.config.sequence_parallel:
                padded_hidden_states = scatter_to_sequence_parallel_region(padded_hidden_states)
            eagle_attention_mask, eagle_position_ids = get_default_attention_mask_and_position_ids(
                padded_eagle_ids
            )

            eagle_inputs = {}
            eagle_inputs["input_ids"] = padded_eagle_ids
            eagle_inputs["embedding"] = self.embedding(
                input_ids=padded_eagle_ids,
                position_ids=eagle_position_ids,
            )
            eagle_inputs["hidden_states"] = padded_hidden_states
            eagle_inputs["attention_mask"] = eagle_attention_mask

            # [TODO] (chenhany): let the module compute itself
            eagle_inputs["rotary_pos_emb"] = None

            _, eagle_logits, eagle_next_hidden_states_input = self._eagle_forward(
                eagle_inputs,
                output_weight,
            )

            eagle_logits = eagle_logits[:seq_len, :, :]
            if self.config.sequence_parallel:
                eagle_next_hidden_states_input = gather_from_sequence_parallel_region(
                    eagle_next_hidden_states_input
                )
            eagle_next_hidden_states_input = eagle_next_hidden_states_input[:seq_len, :, :]

            if self.parallel_draft_step > 1:
                draft_token = (
                    gather_from_tensor_model_parallel_region(eagle_logits)[
                        -self.parallel_draft_step :, :, :
                    ]
                    .argmax(dim=-1)
                    .transpose(0, 1)
                )
            else:
                draft_token = (
                    gather_from_tensor_model_parallel_region(eagle_logits)[-1:, :, :]
                    .argmax(dim=-1)
                    .transpose(0, 1)
                )
            if self.draft_vocab_size > 0:
                draft_token += self.eagle_module.d2t[draft_token]

            if self.parallel_draft_step > 1:
                return base_token, draft_token

            draft_tokens.append(draft_token)

            eagle_ids = torch.cat((eagle_ids, draft_token), dim=-1)
            hidden_states = torch.cat(
                (hidden_states, eagle_next_hidden_states_input[-1:, :, :]), dim=0
            )

        draft_tokens = torch.cat(draft_tokens, dim=-1)

        return base_token, draft_tokens


class MegatronARValidation(AcceptanceRateValidation):
    """This is the subclass for megatron model AR validation."""

    def get_ground_truth(self, input_ids, osl):
        """This function returns ground truth output tokens from the base model."""
        input_ids = copy.deepcopy(input_ids)
        for _ in range(osl):
            input_id, _ = self.model.pseudo_speculative_generate(input_ids, steps=0)
            input_ids = torch.cat((input_ids, input_id), dim=-1)
            if input_id[0, 0] == self.end_token:
                break
        return input_ids
